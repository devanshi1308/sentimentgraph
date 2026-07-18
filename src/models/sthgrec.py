"""ST-HGRec: a heterogeneous LightGCN over the user/item/aspect graph.

Implements STHGREC_DESIGN.md §2 (Model). ST-HGRec is LightGCN with three edge
types instead of one: it keeps LightGCN's discipline — **no feature transforms,
no nonlinearities, no self-loops-with-weights, no attention** — and adds
sentiment + trust purely as *structure and edge weights on the propagation*
(never as an MLP). This is the only place we touch PyG-adjacent sparse ops
directly (Option D); until here, RecBole config files do the work.

Key pieces:
  * §2.1 One embedding table E ∈ R^{(n_users+n_items+n_aspects) × d}, block
    offsets [users | items | aspects].
  * §2.2 L-layer propagation summing per-relation normalized-adjacency messages;
    layer-mean combine (LightGCN).
  * §2.3 signed sentiment weights; §3 trust-4A interaction reweighting — both
    baked into the adjacencies by ``graph_tensors``.
  * §2.4 dot-product scoring; §4 BPR + L2 reg; §7 explanation hooks.

RecBole-id contract (STHGREC_DESIGN.md §5, T4.3 subtlety):
  RecBole assigns its own internal contiguous ids to users/items (id 0 is the
  ``[PAD]`` token) via ``dataset.field2id_token``. These do NOT match our sorted
  ID map. The model consumes RecBole's user/item ids for loss/predict, so at
  init we build a translation layer: RecBole internal user/item id -> our global
  node index (``recbole_user2node`` / ``recbole_item2node``). A RecBole id whose
  raw token is absent from our train-built map (PAD, or an unseen valid/test id)
  maps to a dedicated zero-embedding slot so scoring never crashes on unseen ids
  (matches baseline cold-start behavior, STHGREC_DESIGN.md §8.3).
"""

from __future__ import annotations

import numpy as np
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from src.models.graph_tensors import GraphTensors


class STHGRec(GeneralRecommender):
    r"""LightGCN-heterogeneous backbone over the ST-HGRec graph.

    Constructed as ``STHGRec(config, dataset, graph_tensors=...)``. The
    ``graph_tensors`` bundle (built from the source CSVs by
    :func:`src.models.graph_tensors.build_graph_tensors`) carries the
    pre-normalized per-relation sparse adjacencies over the *single global node
    index space*, plus the explanation-facing aggregates.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, graph_tensors: GraphTensors):
        super().__init__(config, dataset)

        self.gt = graph_tensors
        id_maps = graph_tensors.id_maps

        # --- hyperparameters (STHGREC_DESIGN.md §4; == configs/lightgcn.yaml) ---
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.require_pow = config["require_pow"] if config["require_pow"] is not None else False

        # relation-strength scalars (STHGREC_DESIGN.md §2.2, default 1.0)
        self.sentiment_weight = float(config["sentiment_weight"]) if config["sentiment_weight"] is not None else 1.0
        self.aspect_weight = float(config["aspect_weight"]) if config["aspect_weight"] is not None else 1.0
        self.trust_weight = float(config["trust_weight"]) if config["trust_weight"] is not None else 1.0

        self.use_sentiment_edges = bool(graph_tensors.use_sentiment_edges)
        self.use_trust_edges = bool(graph_tensors.use_trust_edges)

        # --- node-index bookkeeping (STHGREC_DESIGN.md §2.1) ---
        self.n_graph_users = id_maps.n_users
        self.n_graph_items = id_maps.n_items
        self.n_aspects = id_maps.n_aspects
        self.user_offset = id_maps.user_offset
        self.item_offset = id_maps.item_offset
        self.aspect_offset = id_maps.aspect_offset
        # +1 extra row: shared zero slot for unseen/PAD ids (never trained).
        self.n_graph_nodes = id_maps.n_nodes
        self._pad_slot = self.n_graph_nodes  # index of the zero row

        # --- single embedding table (STHGREC_DESIGN.md §2.1) ---
        # Row [n_graph_nodes] is the PAD/unseen zero slot; kept out of propagation.
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.n_graph_nodes + 1, embedding_dim=self.latent_dim
        )

        # --- normalized adjacencies (built off-graph; registered as buffers) ---
        self.register_buffer("adj_interacts", graph_tensors.adj_interacts.coalesce())
        self._has_sentiment = self.use_sentiment_edges and graph_tensors.adj_sentiment is not None
        self._has_aspect = self.use_sentiment_edges and graph_tensors.adj_has_aspect is not None
        if self._has_sentiment:
            self.register_buffer("adj_sentiment", graph_tensors.adj_sentiment.coalesce())
        if self._has_aspect:
            self.register_buffer("adj_has_aspect", graph_tensors.adj_has_aspect.coalesce())

        # --- RecBole-id translation layer (STHGREC_DESIGN.md §5 id contract) ---
        self._build_recbole_translation(dataset, id_maps)

        # --- loss ---
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # full-sort eval caches
        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        # Zero the PAD/unseen slot AFTER init so unseen ids contribute nothing.
        with torch.no_grad():
            self.embedding.weight[self._pad_slot].zero_()
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    # ------------------------------------------------------------------
    # RecBole id <-> global node index translation
    # ------------------------------------------------------------------
    def _build_recbole_translation(self, dataset, id_maps) -> None:
        """Map RecBole internal user/item ids -> our global node indices.

        ``dataset.field2id_token[field]`` is a numpy array: internal id -> raw
        token (index 0 == ``[PAD]``). We invert through our sorted map. Unknown
        tokens (PAD, unseen valid/test ids not in the train-built map) point at
        the zero PAD slot.
        """
        user_tokens = dataset.field2id_token[self.USER_ID]  # len == recbole n_users
        item_tokens = dataset.field2id_token[self.ITEM_ID]

        u2node = np.full(len(user_tokens), self._pad_slot, dtype=np.int64)
        for rid, tok in enumerate(user_tokens):
            local = id_maps.user_id2idx.get(str(tok))
            if local is not None:
                u2node[rid] = local + self.user_offset

        i2node = np.full(len(item_tokens), self._pad_slot, dtype=np.int64)
        for rid, tok in enumerate(item_tokens):
            local = id_maps.item_id2idx.get(str(tok))
            if local is not None:
                i2node[rid] = local + self.item_offset

        self.register_buffer("recbole_user2node", torch.as_tensor(u2node))
        self.register_buffer("recbole_item2node", torch.as_tensor(i2node))

    # ------------------------------------------------------------------
    # Propagation (STHGREC_DESIGN.md §2.2)
    # ------------------------------------------------------------------
    def get_ego_embeddings(self) -> torch.Tensor:
        """The graph-node embeddings (rows [0, n_graph_nodes)), excluding PAD."""
        return self.embedding.weight[: self.n_graph_nodes]

    def forward(self) -> torch.Tensor:
        """L-layer heterogeneous propagation; returns final per-node embeddings.

        E^{(l)} = Â_int·E + λ_s·Â_sent·E + λ_a·Â_aspect·E   (relations gated by
        the ablation flags). E_final = mean over layers 0..L.

        Uses ``torch.sparse.mm`` directly (Option C sparse path, sanctioned by
        the harness decision fix #2): each normalized adjacency is a coalesced
        sparse COO tensor over the single global node index space, so one spmm
        per active relation per layer. See report for PyG-vs-sparse rationale.
        """
        all_emb = self.get_ego_embeddings()
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            agg = torch.sparse.mm(self.adj_interacts, all_emb)
            if self._has_sentiment:
                agg = agg + self.sentiment_weight * torch.sparse.mm(self.adj_sentiment, all_emb)
            if self._has_aspect:
                agg = agg + self.aspect_weight * torch.sparse.mm(self.adj_has_aspect, all_emb)
            all_emb = agg
            emb_list.append(all_emb)

        final = torch.stack(emb_list, dim=1).mean(dim=1)
        return final

    def _final_embeddings(self) -> torch.Tensor:
        """Full node embedding table with a zero PAD row appended for gather."""
        final = self.forward()
        pad = torch.zeros(1, self.latent_dim, device=final.device, dtype=final.dtype)
        return torch.cat([final, pad], dim=0)  # index self._pad_slot -> zeros

    # ------------------------------------------------------------------
    # Loss / predict (STHGREC_DESIGN.md §2.4, §4)
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        final = self._final_embeddings()
        u_nodes = self.recbole_user2node[user]
        pos_nodes = self.recbole_item2node[pos_item]
        neg_nodes = self.recbole_item2node[neg_item]

        u_e = final[u_nodes]
        pos_e = final[pos_nodes]
        neg_e = final[neg_nodes]

        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # L2 reg on the *ego* embeddings of the touched nodes (LightGCN parity).
        u_ego = self.embedding(u_nodes)
        pos_ego = self.embedding(pos_nodes)
        neg_ego = self.embedding(neg_nodes)
        reg = self.reg_loss(u_ego, pos_ego, neg_ego, require_pow=self.require_pow)

        return mf_loss + self.reg_weight * reg

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        final = self._final_embeddings()
        u_e = final[self.recbole_user2node[user]]
        i_e = final[self.recbole_item2node[item]]
        return torch.mul(u_e, i_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            final = self._final_embeddings()
            # Materialize per-RecBole-id user/item tables so scores align with
            # RecBole's item id space (index 0 = PAD -> zero row).
            self.restore_user_e = final[self.recbole_user2node]
            self.restore_item_e = final[self.recbole_item2node]
        u_e = self.restore_user_e[user]
        scores = torch.matmul(u_e, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

    # ------------------------------------------------------------------
    # Explanation hooks (STHGREC_DESIGN.md §7 — P5 dependency)
    # ------------------------------------------------------------------
    def _node_of_user(self, user_raw_id) -> int | None:
        local = self.gt.id_maps.user_id2idx.get(str(user_raw_id))
        return None if local is None else local + self.user_offset

    def _node_of_item(self, item_raw_id) -> int | None:
        local = self.gt.id_maps.item_id2idx.get(str(item_raw_id))
        return None if local is None else local + self.item_offset

    @torch.no_grad()
    def aspect_item_polarity(self, item_raw_id) -> dict:
        """Signed ``has_aspect`` weights for an item (STHGREC_DESIGN.md §7).

        Reads the *same* aggregation §2.3 uses (mean signed polarity per
        (item, aspect)), returning plain-python floats keyed by aspect name.
        """
        item_key = str(item_raw_id)
        out: dict = {}
        for (it, aspect), w in self.gt.has_aspect_weights.items():
            if str(it) == item_key:
                out[str(aspect)] = float(w)
        return out

    @torch.no_grad()
    def explain_pair(self, user_raw_id, item_raw_id) -> dict:
        """Per-recommendation facts for the honest generator (STHGREC_DESIGN.md §7).

        Returns only facts derivable from the graph + trained embeddings — no LLM,
        no new claims — in plain python types.
        """
        u_node = self._node_of_user(user_raw_id)
        i_node = self._node_of_item(item_raw_id)

        score = None
        if u_node is not None and i_node is not None:
            final = self._final_embeddings()
            score = float(torch.dot(final[u_node], final[i_node]).item())

        aspect_contributions = self.aspect_item_polarity(item_raw_id)

        supporting = []
        if len(self.gt.sentiment_rows):
            rows = self.gt.sentiment_rows
            sub = rows[rows["item_id"].astype(str) == str(item_raw_id)]
            supporting = [
                {"aspect": str(r["aspect"]), "sentiment": str(r["sentiment"])}
                for _, r in sub.iterrows()
            ]

        trust_score = self.gt.trust_scores.get(str(user_raw_id))
        used_trust = self.use_trust_edges
        user_trust = {
            "trust_score": (float(trust_score) if trust_score is not None else None),
            "role": (
                "interaction edges up-weighted by trust_score"
                if used_trust
                else "trust disabled (use_trust_edges=false)"
            ),
        }

        return {
            "score": score,
            "aspect_contributions": aspect_contributions,
            "supporting_sentiment_edges": supporting,
            "user_trust": user_trust,
        }

    @torch.no_grad()
    def user_neighborhood(self, user_raw_id, top_k: int = 10) -> list:
        """Items most responsible for the user's embedding (STHGREC_DESIGN.md §7).

        Ranks the user's directly-interacted items by dot-product of their final
        embeddings with the user's — a faithful 'because you liked …' signal.
        Returns [(item_raw_id, score), ...] in plain python.
        """
        u_node = self._node_of_user(user_raw_id)
        if u_node is None:
            return []
        final = self._final_embeddings()
        u_e = final[u_node]

        # Neighbors via the interacts adjacency (item nodes linked to this user).
        adj = self.adj_interacts.coalesce()
        idx = adj.indices()
        mask = idx[0] == u_node
        neigh = idx[1][mask]
        neigh = neigh[(neigh >= self.item_offset) & (neigh < self.aspect_offset)]
        if neigh.numel() == 0:
            return []
        neigh = torch.unique(neigh)

        scores = torch.mv(final[neigh], u_e)
        order = torch.argsort(scores, descending=True)[:top_k]

        idx2item = {v: k for k, v in self.gt.id_maps.item_id2idx.items()}
        out = []
        for pos in order.tolist():
            node = int(neigh[pos].item())
            raw = idx2item.get(node - self.item_offset)
            out.append((raw, float(scores[pos].item())))
        return out
