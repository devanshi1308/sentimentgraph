"""ID mapping + per-edge-type sparse adjacency construction for ST-HGRec.

Implements STHGREC_DESIGN.md §1 (Graph -> tensors). Builds the heterogeneous
adjacency ST-HGRec message-passes over **directly from the source CSVs**
(``video_games.train.inter``, ``absa_normalized.csv``, ``trust_signals.csv``) —
*not* from the ``graph.gpickle`` (a regenerable, untracked binary). This keeps
the exact same deterministic ID mapping and dedup/aggregation policy the model,
evaluator, and P5 explanation hooks all share.

Design anchors (STHGREC_DESIGN.md):
  * §1.1 Deterministic, seed-independent, sorted, contiguous ID mapping,
    persisted to ``sthgrec_id_maps.json``.
  * §1.3 Per-relation dedup/aggregation: interacts collapse-to-one (weight 1);
    sentiment -> one signed weight per (u,i) (mean of Positive=+1 / Negative=-1 /
    Neutral=0, clipped to [-1,+1]); has_aspect -> mean signed polarity per (i,a).
  * §2.3 Signed sentiment weights; degree normalization uses |w| so negatives do
    not cancel the normalizer while  keeps the sign.
  * §3 Trust Option 4A: user ``trust_score`` scales that user's interaction-edge
    weights before normalization. Missing trust -> weight 1.0 (documented).
  * §6 Ablation flags: ``use_sentiment_edges=False`` drops the sentiment AND
    has_aspect relations *entirely*; ``use_trust_edges=False`` reverts every
    interaction weight to 1.0 (exact equivalence with the untrusted graph).

Single global node index space (STHGREC_DESIGN.md §2.1), block-offset ordered:
    [0, n_users)                      -> users
    [n_users, n_users+n_items)        -> items
    [.., .. + n_aspects)              -> aspects (10 rows)

All relations are stored as one directed COO ``edge_index`` (+ weight) and
symmetrized (transpose added) at normalization time, exactly as LightGCN builds
its symmetric norm adjacency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Signed polarity map (STHGREC_DESIGN.md §2.3). PyABSA emits these three labels.
_POLARITY = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0}


# ---------------------------------------------------------------------------
# CSV loaders (mirror build_graph.py column handling)
# ---------------------------------------------------------------------------

def load_interactions(inter_path: Path) -> pd.DataFrame:
    """Read a RecBole atomic ``.inter`` (tab-sep, ``field:type`` headers)."""
    df = pd.read_csv(inter_path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]  # strip :token/:float
    return df


def load_sentiment(sentiment_path: Optional[Path]) -> pd.DataFrame:
    """Read ``absa_normalized.csv`` (user_id,item_id,aspect,sentiment).

    A missing path yields an empty frame — the sentiment-off ablation and the
    "no sentiment file at all" case both flow through this cleanly.
    """
    cols = ["user_id", "item_id", "aspect", "sentiment"]
    if sentiment_path is None or not Path(sentiment_path).exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(sentiment_path)


def load_trust(trust_path: Optional[Path]) -> pd.DataFrame:
    """Read ``trust_signals.csv`` (needs at least ``user_id`` + ``trust_score``)."""
    if trust_path is None or not Path(trust_path).exists():
        return pd.DataFrame(columns=["user_id", "trust_score"])
    return pd.read_csv(trust_path)


# ---------------------------------------------------------------------------
# ID maps (STHGREC_DESIGN.md §1.1)
# ---------------------------------------------------------------------------

@dataclass
class IdMaps:
    """Deterministic raw-id <-> tensor-index mapping for all three node types.

    Indices are *local* to each block; global indices add the block offset
    (STHGREC_DESIGN.md §2.1): user_offset=0, item_offset=n_users,
    aspect_offset=n_users+n_items.
    """

    user_id2idx: dict
    item_id2idx: dict
    aspect2idx: dict
    source: str = "video_games.train.inter"
    built_from: str = "csv"

    @property
    def n_users(self) -> int:
        return len(self.user_id2idx)

    @property
    def n_items(self) -> int:
        return len(self.item_id2idx)

    @property
    def n_aspects(self) -> int:
        return len(self.aspect2idx)

    @property
    def user_offset(self) -> int:
        return 0

    @property
    def item_offset(self) -> int:
        return self.n_users

    @property
    def aspect_offset(self) -> int:
        return self.n_users + self.n_items

    @property
    def n_nodes(self) -> int:
        return self.n_users + self.n_items + self.n_aspects

    # -- persistence --------------------------------------------------------
    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_aspects": self.n_aspects,
            # JSON keys are strings; keep raw ids as-is (they are strings anyway).
            "user_id2idx": {str(k): v for k, v in self.user_id2idx.items()},
            "item_id2idx": {str(k): v for k, v in self.item_id2idx.items()},
            "aspect2idx": {str(k): v for k, v in self.aspect2idx.items()},
            "source": self.source,
            "built_from": self.built_from,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: Path) -> "IdMaps":
        with open(path) as f:
            d = json.load(f)
        return cls(
            user_id2idx=d["user_id2idx"],
            item_id2idx=d["item_id2idx"],
            aspect2idx=d["aspect2idx"],
            source=d.get("source", ""),
            built_from=d.get("built_from", ""),
        )


def build_id_maps(
    interactions: pd.DataFrame,
    sentiment: pd.DataFrame,
    source: str = "video_games.train.inter",
) -> IdMaps:
    """Sorted, contiguous, seed-independent id maps (STHGREC_DESIGN.md §1.1).

    Users/items come from the TRAIN interactions (leakage guard). Aspects come
    from the sentiment file's aspect column, sorted lexicographically. Sorting
    the *raw* ids removes any dependence on CSV row order or graph iteration.
    """
    user_raw = sorted(str(u) for u in interactions["user_id"].unique())
    item_raw = sorted(str(i) for i in interactions["item_id"].unique())
    aspect_raw = (
        sorted(str(a) for a in sentiment["aspect"].unique())
        if len(sentiment)
        else []
    )
    return IdMaps(
        user_id2idx={raw: k for k, raw in enumerate(user_raw)},
        item_id2idx={raw: k for k, raw in enumerate(item_raw)},
        aspect2idx={raw: k for k, raw in enumerate(aspect_raw)},
        source=source,
    )


# ---------------------------------------------------------------------------
# Per-relation edge construction (STHGREC_DESIGN.md §1.3, §2.3, §3)
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """One typed relation as a *directed* COO edge list over global indices.

    Symmetrization (adding the transpose) and D^-1/2 A D^-1/2 normalization
    happen in :func:`normalized_adj`. ``weight`` may carry signs (sentiment).
    """

    edge_index: torch.Tensor  # (2, E) long, global indices
    weight: torch.Tensor      # (E,) float

    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def build_interacts_relation(
    interactions: pd.DataFrame,
    id_maps: IdMaps,
    trust: pd.DataFrame,
    use_trust_edges: bool,
) -> Relation:
    """user -> item ``interacts`` relation (STHGREC_DESIGN.md §1.3, §3 Trust 4A).

    - 5-core is de-duplicated per (u,i); a stray repeat collapses to one edge.
    - Base weight is 1.0 (LightGCN binary adjacency; ratings/timestamps unused).
    - Trust 4A: when ``use_trust_edges``, multiply each user's edge weights by
      that user's ``trust_score``; users with no trust row keep weight 1.0.
      When trust is off, every weight is exactly 1.0 (clean ablation).
    """
    u_idx = interactions["user_id"].astype(str).map(id_maps.user_id2idx)
    i_idx = interactions["item_id"].astype(str).map(id_maps.item_id2idx)
    mask = u_idx.notna() & i_idx.notna()
    u_idx = u_idx[mask].astype(np.int64).to_numpy()
    i_idx = i_idx[mask].astype(np.int64).to_numpy()

    # Collapse to one edge per (u,i) pair.
    pairs = np.unique(np.stack([u_idx, i_idx], axis=1), axis=0)
    u_local, i_local = pairs[:, 0], pairs[:, 1]

    weight = np.ones(len(u_local), dtype=np.float64)
    if use_trust_edges and len(trust) and "trust_score" in trust.columns:
        trust_map = dict(
            zip(trust["user_id"].astype(str), trust["trust_score"].astype(float))
        )
        # Reverse the user local-index -> raw-id lookup once.
        idx2user = {v: k for k, v in id_maps.user_id2idx.items()}
        for e in range(len(u_local)):
            raw = idx2user[int(u_local[e])]
            # Missing trust -> 1.0 (documented default, STHGREC_DESIGN.md §3).
            weight[e] = trust_map.get(raw, 1.0)

    src = torch.as_tensor(u_local + id_maps.user_offset, dtype=torch.long)
    dst = torch.as_tensor(i_local + id_maps.item_offset, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return Relation(edge_index=edge_index, weight=torch.as_tensor(weight, dtype=torch.float))


def _signed(series: pd.Series) -> pd.Series:
    return series.map(_POLARITY).astype(float)


def build_sentiment_relation(
    sentiment: pd.DataFrame, id_maps: IdMaps
) -> Relation:
    """user -> item ``sentiment`` relation (STHGREC_DESIGN.md §2.3).

    One signed weight per (u,i) = mean of signed polarities across that pair's
    aspects, clipped to [-1, +1]. Rows referencing unknown users/items are
    dropped (test-only pairs). An empty result yields a zero-edge relation.
    """
    if not len(sentiment):
        return Relation(torch.empty((2, 0), dtype=torch.long), torch.empty(0))

    df = sentiment.copy()
    df["u"] = df["user_id"].astype(str).map(id_maps.user_id2idx)
    df["i"] = df["item_id"].astype(str).map(id_maps.item_id2idx)
    df = df[df["u"].notna() & df["i"].notna()]
    if not len(df):
        return Relation(torch.empty((2, 0), dtype=torch.long), torch.empty(0))
    df["s"] = _signed(df["sentiment"])

    agg = df.groupby(["u", "i"], sort=True)["s"].mean().reset_index()
    agg["s"] = agg["s"].clip(-1.0, 1.0)

    src = torch.as_tensor(agg["u"].to_numpy(dtype=np.int64) + id_maps.user_offset)
    dst = torch.as_tensor(agg["i"].to_numpy(dtype=np.int64) + id_maps.item_offset)
    edge_index = torch.stack([src, dst], dim=0)
    weight = torch.tensor(agg["s"].to_numpy(dtype=np.float64), dtype=torch.float)
    return Relation(edge_index=edge_index, weight=weight)


def build_has_aspect_relation(
    sentiment: pd.DataFrame, id_maps: IdMaps
) -> Relation:
    """item -> aspect ``has_aspect`` relation (STHGREC_DESIGN.md §2.3).

    One signed weight per (i,aspect) = mean signed polarity of that aspect for
    that item. Rows with unknown item are dropped. Empty -> zero-edge relation.
    """
    if not len(sentiment):
        return Relation(torch.empty((2, 0), dtype=torch.long), torch.empty(0))

    df = sentiment.copy()
    df["i"] = df["item_id"].astype(str).map(id_maps.item_id2idx)
    df["a"] = df["aspect"].astype(str).map(id_maps.aspect2idx)
    df = df[df["i"].notna() & df["a"].notna()]
    if not len(df):
        return Relation(torch.empty((2, 0), dtype=torch.long), torch.empty(0))
    df["s"] = _signed(df["sentiment"])

    agg = df.groupby(["i", "a"], sort=True)["s"].mean().reset_index()

    src = torch.as_tensor(agg["i"].to_numpy(dtype=np.int64) + id_maps.item_offset)
    dst = torch.as_tensor(agg["a"].to_numpy(dtype=np.int64) + id_maps.aspect_offset)
    edge_index = torch.stack([src, dst], dim=0)
    weight = torch.tensor(agg["s"].to_numpy(dtype=np.float64), dtype=torch.float)
    return Relation(edge_index=edge_index, weight=weight)


# ---------------------------------------------------------------------------
# Symmetric normalization (STHGREC_DESIGN.md §1.3, §2.3)
# ---------------------------------------------------------------------------

def normalized_adj(relation: Relation, n_nodes: int) -> torch.Tensor:
    """Symmetric-normalized sparse adjacency  = D^-1/2 A D^-1/2.

    A is the *symmetrized* (undirected) weighted adjacency of the relation:
    the directed edges plus their transpose. Degree D is computed on the
    **absolute** edge weights (STHGREC_DESIGN.md §2.3) so that negative
    sentiment weights do not cancel the normalizer;  keeps the sign.

    Returns a coalesced ``torch.sparse_coo_tensor`` of shape (n_nodes, n_nodes).
    A zero-edge relation returns an empty sparse tensor (contributes nothing).
    """
    E = relation.num_edges()
    if E == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0),
            (n_nodes, n_nodes),
        ).coalesce()

    ei = relation.edge_index
    w = relation.weight

    # Symmetrize: add transpose (undirected propagation).
    row = torch.cat([ei[0], ei[1]])
    col = torch.cat([ei[1], ei[0]])
    val = torch.cat([w, w])

    # Degree on |w| (STHGREC_DESIGN.md §2.3).
    deg = torch.zeros(n_nodes, dtype=val.dtype)
    deg.scatter_add_(0, row, val.abs())
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]

    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, norm_val, (n_nodes, n_nodes))
    return adj.coalesce()


# ---------------------------------------------------------------------------
# Top-level graph tensor container
# ---------------------------------------------------------------------------

@dataclass
class GraphTensors:
    """Everything ST-HGRec needs to propagate + explain.

    Holds the id maps, the pre-normalized sparse adjacencies (only for active
    relations; ablated relations are ``None``), and the raw aggregated
    sentiment/has_aspect/trust facts for the P5 explanation hooks.
    """

    id_maps: IdMaps
    adj_interacts: torch.Tensor
    adj_sentiment: Optional[torch.Tensor]
    adj_has_aspect: Optional[torch.Tensor]
    use_sentiment_edges: bool
    use_trust_edges: bool
    # explanation-facing aggregates (plain data, keyed by raw ids)
    trust_scores: dict          # raw user_id -> trust_score (float)
    has_aspect_weights: dict     # (raw_item_id, aspect_name) -> signed weight
    sentiment_rows: pd.DataFrame  # normalized absa rows restricted to known ids

    @property
    def n_nodes(self) -> int:
        return self.id_maps.n_nodes


def build_graph_tensors(
    inter_csv: Path,
    sentiment_csv: Optional[Path],
    trust_csv: Optional[Path],
    use_sentiment_edges: bool = True,
    use_trust_edges: bool = True,
    id_maps: Optional[IdMaps] = None,
    id_map_json: Optional[Path] = None,
) -> GraphTensors:
    """Build the full ST-HGRec adjacency set from source CSVs.

    STHGREC_DESIGN.md §1-3. Ablation flags gate whole relations:
      * ``use_sentiment_edges=False`` -> sentiment + has_aspect relations are
        never built (identical to having no sentiment file at all).
      * ``use_trust_edges=False`` -> interaction weights all revert to 1.0.

    ``id_maps``/``id_map_json``: pass a prebuilt map (or a JSON path) to keep the
    raw<->index contract stable across ablations; otherwise one is built from the
    CSVs. When ``id_map_json`` points at an existing file it is asserted-equal
    (never silently regenerated); a missing path is written.
    """
    interactions = load_interactions(Path(inter_csv))

    # Sentiment is only read when the ablation keeps it. Reading nothing makes
    # sentiment-off byte-identical to "no sentiment file at all".
    if use_sentiment_edges:
        sentiment = load_sentiment(sentiment_csv)
    else:
        sentiment = load_sentiment(None)  # empty frame

    trust = load_trust(trust_csv)

    # ID maps: aspects only appear when sentiment is on. To keep the user/item
    # index space identical across ablations, aspects are built from the (always
    # available) sentiment file if a JSON contract is provided; otherwise from
    # whatever sentiment we loaded.
    if id_maps is None and id_map_json is not None and Path(id_map_json).exists():
        id_maps = IdMaps.from_json(Path(id_map_json))
    if id_maps is None:
        # Build aspect map from the on-disk sentiment file regardless of the
        # ablation, so user/item/aspect index spaces are stable run-to-run.
        aspect_src = load_sentiment(sentiment_csv)
        id_maps = build_id_maps(interactions, aspect_src, source=str(Path(inter_csv).name))
        if id_map_json is not None:
            id_maps.to_json(Path(id_map_json))

    n_nodes = id_maps.n_nodes

    # interacts (always present)
    rel_interacts = build_interacts_relation(
        interactions, id_maps, trust, use_trust_edges
    )
    adj_interacts = normalized_adj(rel_interacts, n_nodes)

    # sentiment + has_aspect (only if enabled)
    adj_sentiment = None
    adj_has_aspect = None
    if use_sentiment_edges:
        rel_sent = build_sentiment_relation(sentiment, id_maps)
        rel_aspect = build_has_aspect_relation(sentiment, id_maps)
        adj_sentiment = normalized_adj(rel_sent, n_nodes)
        adj_has_aspect = normalized_adj(rel_aspect, n_nodes)

    # ---- explanation-facing aggregates (plain python data) ----
    trust_scores: dict = {}
    if len(trust) and "trust_score" in trust.columns:
        trust_scores = dict(
            zip(trust["user_id"].astype(str), trust["trust_score"].astype(float))
        )

    has_aspect_weights: dict = {}
    sentiment_rows = load_sentiment(sentiment_csv) if use_sentiment_edges else load_sentiment(None)
    if len(sentiment_rows):
        sr = sentiment_rows.copy()
        sr = sr[
            sr["item_id"].astype(str).isin(id_maps.item_id2idx)
            & sr["aspect"].astype(str).isin(id_maps.aspect2idx)
        ]
        sentiment_rows = sr.reset_index(drop=True)
        if len(sr):
            sr = sr.copy()
            sr["s"] = _signed(sr["sentiment"])
            agg = sr.groupby([sr["item_id"].astype(str), sr["aspect"].astype(str)])["s"].mean()
            has_aspect_weights = {k: float(v) for k, v in agg.items()}

    return GraphTensors(
        id_maps=id_maps,
        adj_interacts=adj_interacts,
        adj_sentiment=adj_sentiment,
        adj_has_aspect=adj_has_aspect,
        use_sentiment_edges=use_sentiment_edges,
        use_trust_edges=use_trust_edges,
        trust_scores=trust_scores,
        has_aspect_weights=has_aspect_weights,
        sentiment_rows=sentiment_rows,
    )
