"""ST-HGRec model-core acceptance tests (STHGREC_DESIGN.md).

Run:  python -m pytest tests/ -x -q   (from repo root, in the p4 venv)

Covers the seven acceptance requirements:
  a. PARITY  — sentiment+trust off ≡ hand-computed vanilla LightGCN layers.
  b. SIGNED  — negative sentiment contributes with opposite sign; degree on |w|.
  c. TRUST   — 4A scaling; use_trust_edges=False restores untrusted equality.
  d. ABLATION— use_sentiment_edges=False ≡ no-sentiment-file graph.
  e. DETERM  — two builds+forwards at seed 2020 are identical.
  f. E2E     — tiny synthetic atomic files -> run_sthgrec for 2 epochs, finite
               metrics, correct summary schema (proves RecBole-id translation).
  g. EXPLAIN — per-aspect contribution API is sane; removing the top aspect edge
               changes that aspect's contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import graph_tensors as GT  # noqa: E402


# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------

def _write_inter(path: Path, pairs, users=None, items=None) -> None:
    """Write a RecBole-style tab-sep .inter (:token/:float headers)."""
    rows = []
    for u, i in pairs:
        rows.append((u, i, 5.0, 1.0))
    df = pd.DataFrame(rows, columns=["user_id:token", "item_id:token", "rating:float", "timestamp:float"])
    df.to_csv(path, sep="\t", index=False)


def _write_sentiment(path: Path, rows) -> None:
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "aspect", "sentiment"])
    df.to_csv(path, index=False)


def _write_trust(path: Path, rows) -> None:
    df = pd.DataFrame(rows, columns=["user_id", "trust_score"])
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# a. PARITY vs hand-computed vanilla LightGCN
# ---------------------------------------------------------------------------

def test_parity_lightgcn(tmp_path):
    """5-node bipartite graph, sentiment+trust off, matches closed-form LightGCN."""
    # 2 users (u0,u1), 3 items (i0,i1,i2). Edges: u0-i0, u0-i1, u1-i1, u1-i2.
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u0", "i1"), ("u1", "i1"), ("u1", "i2")])

    gt = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=None, trust_csv=None,
        use_sentiment_edges=False, use_trust_edges=False,
    )
    n = gt.id_maps.n_nodes
    assert n == 5  # 2 users + 3 items, no aspects

    # --- independent closed-form reference ---
    # sorted user ids: u0->0, u1->1 ; items: i0->0,i1->1,i2->2 (global 2,3,4)
    # Adjacency A (symmetric, undirected), global index [u0,u1,i0,i1,i2]=[0,1,2,3,4]
    A = np.zeros((5, 5))
    for a, b in [(0, 2), (0, 3), (1, 3), (1, 4)]:
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    Ahat = (dinv[:, None] * A) * dinv[None, :]

    torch.manual_seed(0)
    E0 = torch.randn(5, 4, dtype=torch.float64)

    L = 3
    ref = [E0.numpy()]
    cur = E0.numpy()
    for _ in range(L):
        cur = Ahat @ cur
        ref.append(cur)
    ref_final = np.mean(np.stack(ref, axis=0), axis=0)

    # --- our path: normalized_adj + same propagation ---
    adj = gt.adj_interacts.to(torch.float64)
    emb = E0.clone()
    ours = [emb]
    cur_t = emb
    for _ in range(L):
        cur_t = torch.sparse.mm(adj, cur_t)
        ours.append(cur_t)
    ours_final = torch.stack(ours, dim=1).mean(dim=1).numpy()

    assert np.allclose(ours_final, ref_final, atol=1e-10), \
        f"max diff {np.abs(ours_final - ref_final).max()}"


# ---------------------------------------------------------------------------
# b. SIGNED-weight semantics: negative opposite sign; degree on |w|
# ---------------------------------------------------------------------------

def test_signed_sentiment_opposite_sign(tmp_path):
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u0", "i1")])

    # Positive sentiment on (u0,i0); Negative on (u0,i1).
    sent_pos = tmp_path / "pos.csv"
    _write_sentiment(sent_pos, [("u0", "i0", "gameplay", "Positive"),
                                ("u0", "i1", "gameplay", "Negative")])

    gt = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=sent_pos, trust_csv=None,
        use_sentiment_edges=True, use_trust_edges=False,
    )
    adj = gt.adj_sentiment.to_dense().to(torch.float64)
    # global: u0=0, i0=1, i1=2 (2 users? no — 1 user, 2 items). n_users=1.
    im = gt.id_maps
    u0 = im.user_id2idx["u0"] + im.user_offset
    i0 = im.item_id2idx["i0"] + im.item_offset
    i1 = im.item_id2idx["i1"] + im.item_offset
    # Positive edge normalized entry > 0; Negative edge < 0 (opposite sign).
    assert adj[u0, i0] > 0
    assert adj[u0, i1] < 0
    # symmetric
    assert torch.isclose(adj[u0, i0], adj[i0, u0])


def test_degree_uses_abs_weight(tmp_path):
    """Degree normalization uses |w|; a wrong (signed-sum) degree differs."""
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u0", "i1")])
    # u0 has one +1 and one -1 sentiment edge. Signed-sum degree of u0 would be 0
    # (=> divide-by-zero / dropped), whereas |w|-degree = 2. Construct and assert
    # the |w| path yields finite, nonzero normalized entries.
    sent = tmp_path / "s.csv"
    _write_sentiment(sent, [("u0", "i0", "gameplay", "Positive"),
                            ("u0", "i1", "price", "Negative")])
    gt = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=sent, trust_csv=None,
        use_sentiment_edges=True, use_trust_edges=False,
    )
    adj = gt.adj_sentiment.to_dense().to(torch.float64)
    im = gt.id_maps
    u0 = im.user_offset + im.user_id2idx["u0"]
    i0 = im.item_offset + im.item_id2idx["i0"]
    i1 = im.item_offset + im.item_id2idx["i1"]
    # |w|-degree(u0)=2, deg(i0)=1, deg(i1)=1.
    # Â[u0,i0] = 1/(sqrt(2)*sqrt(1)) * (+1) = +1/sqrt(2)
    # Â[u0,i1] = 1/(sqrt(2)*sqrt(1)) * (-1) = -1/sqrt(2)
    assert torch.isclose(adj[u0, i0], torch.tensor(1.0 / np.sqrt(2), dtype=torch.float64), atol=1e-12)
    assert torch.isclose(adj[u0, i1], torch.tensor(-1.0 / np.sqrt(2), dtype=torch.float64), atol=1e-12)
    # A signed-sum degree (=0) would have produced inf/nan — assert all finite.
    assert torch.isfinite(adj).all()


# ---------------------------------------------------------------------------
# c. TRUST 4A
# ---------------------------------------------------------------------------

def test_trust_scales_interactions(tmp_path):
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u1", "i0")])
    trust = tmp_path / "t.csv"
    # u0 trust 0.5, u1 trust 1.0.
    _write_trust(trust, [("u0", 0.5), ("u1", 1.0)])

    gt = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=None, trust_csv=trust,
        use_sentiment_edges=False, use_trust_edges=True,
    )
    im = gt.id_maps
    u0 = im.user_offset + im.user_id2idx["u0"]
    u1 = im.user_offset + im.user_id2idx["u1"]
    i0 = im.item_offset + im.item_id2idx["i0"]
    adj = gt.adj_interacts.to_dense().to(torch.float64)

    # Weighted adjacency: A[u0,i0]=0.5, A[u1,i0]=1.0.
    # deg(u0)=0.5, deg(u1)=1.0, deg(i0)=0.5+1.0=1.5.
    # Â[u0,i0] = 0.5 / (sqrt(0.5)*sqrt(1.5))
    # Â[u1,i0] = 1.0 / (sqrt(1.0)*sqrt(1.5))
    exp_u0 = 0.5 / (np.sqrt(0.5) * np.sqrt(1.5))
    exp_u1 = 1.0 / (np.sqrt(1.0) * np.sqrt(1.5))
    assert torch.isclose(adj[u0, i0], torch.tensor(exp_u0, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(adj[u1, i0], torch.tensor(exp_u1, dtype=torch.float64), atol=1e-12)
    # u0 (trust 0.5) pushes less hard than u1 (trust 1.0) on the shared item.
    assert adj[u0, i0] < adj[u1, i0]


def test_trust_off_restores_untrusted(tmp_path):
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u1", "i0")])
    trust = tmp_path / "t.csv"
    _write_trust(trust, [("u0", 0.5), ("u1", 1.0)])

    gt_off = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=None, trust_csv=trust,
        use_sentiment_edges=False, use_trust_edges=False,
    )
    gt_notrust = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=None, trust_csv=None,
        use_sentiment_edges=False, use_trust_edges=False,
    )
    a = gt_off.adj_interacts.to_dense()
    b = gt_notrust.adj_interacts.to_dense()
    assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# d. ABLATION exactness
# ---------------------------------------------------------------------------

def test_sentiment_off_equals_no_file(tmp_path):
    """sentiment-off ≡ graph built with no sentiment file.

    Ablations share a fixed id-map contract (STHGREC_DESIGN.md §1.1/§6) so the
    node index space (and thus adjacency shape) is stable; the interacts
    adjacency must be byte-identical and the sentiment/has_aspect terms absent.
    """
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u1", "i1")])
    sent = tmp_path / "s.csv"
    _write_sentiment(sent, [("u0", "i0", "gameplay", "Positive"),
                            ("u1", "i1", "price", "Negative")])
    idmap = tmp_path / "idmap.json"

    # Establish the shared id-map contract from the full (sentiment-on) build.
    GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=sent, trust_csv=None,
        use_sentiment_edges=True, use_trust_edges=False, id_map_json=idmap,
    )
    gt_off = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=sent, trust_csv=None,
        use_sentiment_edges=False, use_trust_edges=False, id_map_json=idmap,
    )
    gt_nofile = GT.build_graph_tensors(
        inter_csv=inter, sentiment_csv=None, trust_csv=None,
        use_sentiment_edges=True, use_trust_edges=False, id_map_json=idmap,
    )
    # No sentiment/has_aspect terms when sentiment is off.
    assert gt_off.adj_sentiment is None
    assert gt_off.adj_has_aspect is None
    # interacts adjacency identical across sentiment-off and no-file.
    a = gt_off.adj_interacts.to_dense()
    b = gt_nofile.adj_interacts.to_dense()
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# e. DETERMINISM
# ---------------------------------------------------------------------------

def test_determinism(tmp_path):
    inter = tmp_path / "vg.train.inter"
    _write_inter(inter, [("u0", "i0"), ("u0", "i1"), ("u1", "i1"), ("u1", "i2")])
    sent = tmp_path / "s.csv"
    _write_sentiment(sent, [("u0", "i0", "gameplay", "Positive"),
                            ("u1", "i2", "price", "Negative")])
    trust = tmp_path / "t.csv"
    _write_trust(trust, [("u0", 0.7), ("u1", 0.3)])

    def build_and_forward():
        from recbole.utils import init_seed
        init_seed(2020, True)
        gt = GT.build_graph_tensors(
            inter_csv=inter, sentiment_csv=sent, trust_csv=trust,
            use_sentiment_edges=True, use_trust_edges=True,
        )
        torch.manual_seed(2020)
        E = torch.randn(gt.id_maps.n_nodes, 8, dtype=torch.float64)
        out = torch.sparse.mm(gt.adj_interacts.to(torch.float64), E)
        out = out + torch.sparse.mm(gt.adj_sentiment.to(torch.float64), E)
        out = out + torch.sparse.mm(gt.adj_has_aspect.to(torch.float64), E)
        return gt.adj_interacts.to_dense(), out

    a_adj, a_out = build_and_forward()
    b_adj, b_out = build_and_forward()
    assert torch.equal(a_adj, b_adj)
    assert torch.equal(a_out, b_out)


# ---------------------------------------------------------------------------
# g. EXPLANATION hook (model-level; uses tiny RecBole dataset)
# ---------------------------------------------------------------------------

def _build_toy_recbole(tmp_path, sentiment=True, trust=True,
                       drop_sentiment_rows=None, seed=2020):
    """Build a tiny RecBole dataset + STHGRec model for hook/e2e tests."""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed
    from src.models.sthgrec import STHGRec
    from src.eval.run_sthgrec import _CUSTOM_KEYS

    ds_dir = tmp_path / "dataset" / "toy"
    ds_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    users = [f"u{k}" for k in range(30)]
    items = [f"i{k}" for k in range(20)]
    aspects = ["gameplay", "graphics", "price"]

    # Dense-ish interactions so every user/item appears >=1 in each split.
    pairs = []
    ts = 1
    for u in users:
        chosen = rng.choice(len(items), size=6, replace=False)
        for c in chosen:
            pairs.append((u, items[c], ts)); ts += 1

    def split_rows(rows):
        df = pd.DataFrame(rows, columns=["u", "i", "t"])
        # per-user temporal-ish split: 4 train / 1 valid / 1 test
        tr, va, te = [], [], []
        for u, g in df.groupby("u"):
            g = g.sort_values("t")
            tr.append(g.iloc[:4]); va.append(g.iloc[4:5]); te.append(g.iloc[5:6])
        return pd.concat(tr), pd.concat(va), pd.concat(te)

    tr, va, te = split_rows(pairs)

    def write_split(df, name):
        out = pd.DataFrame({
            "user_id:token": df["u"], "item_id:token": df["i"],
            "rating:float": 5.0, "timestamp:float": df["t"].astype(float),
        })
        out.to_csv(ds_dir / f"toy.{name}.inter", sep="\t", index=False)

    write_split(tr, "train"); write_split(va, "valid"); write_split(te, "test")
    train_inter = ds_dir / "toy.train.inter"

    sent_path = None
    if sentiment:
        srows = []
        for u, i, _ in [(r.u, r.i, r.t) for r in tr.itertuples()]:
            a = aspects[hash((u, i)) % len(aspects)]
            pol = "Positive" if (hash((i, a)) % 2 == 0) else "Negative"
            srows.append((u, i, a, pol))
        if drop_sentiment_rows:
            srows = [r for r in srows if r not in drop_sentiment_rows]
        sent_path = tmp_path / "absa.csv"
        _write_sentiment(sent_path, srows)

    trust_path = None
    if trust:
        trust_path = tmp_path / "trust.csv"
        _write_trust(trust_path, [(u, float(rng.uniform(0.2, 1.0))) for u in users])

    cfg = dict(_CUSTOM_KEYS)
    cfg.update({
        "model": "STHGRec", "dataset": "toy", "data_path": str(tmp_path / "dataset"),
        "seed": seed, "reproducibility": True,
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {"mode": "full", "split": {"RS": [0.8, 0.1, 0.1]},
                      "order": "TO", "group_by": "user"},
        "embedding_size": 8, "n_layers": 2, "reg_weight": 1e-4,
        "train_batch_size": 64, "learning_rate": 0.01, "epochs": 2,
        "use_sentiment_edges": sentiment, "use_trust_edges": trust,
        "graph_inter_csv": str(train_inter),
        "graph_sentiment_csv": str(sent_path) if sent_path else None,
        "graph_trust_csv": str(trust_path) if trust_path else None,
        "graph_id_map_json": str(tmp_path / "idmap.json"),
        "metrics": ["Recall", "NDCG"], "topk": [10, 20],
        "valid_metric": "NDCG@10", "device": "cpu",
    })

    config = Config(model=STHGRec, config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    from src.eval.run_sthgrec import build_model
    model, gt = build_model(config, train_data.dataset)
    return config, dataset, train_data, valid_data, test_data, model, gt, (tr, va, te)


def test_explanation_hook(tmp_path):
    (config, dataset, train_data, valid_data, test_data,
     model, gt, splits) = _build_toy_recbole(tmp_path)

    tr = splits[0]
    # Pick a (user,item) pair present in train.
    u_raw, i_raw = tr.iloc[0].u, tr.iloc[0].i

    out = model.explain_pair(u_raw, i_raw)
    # plain-python structure
    assert isinstance(out, dict)
    assert set(out) == {"score", "aspect_contributions",
                        "supporting_sentiment_edges", "user_trust"}
    assert out["score"] is None or isinstance(out["score"], float)
    assert isinstance(out["aspect_contributions"], dict)
    for k, v in out["aspect_contributions"].items():
        assert isinstance(k, str) and isinstance(v, float) and -1.0 <= v <= 1.0
    assert isinstance(out["supporting_sentiment_edges"], list)
    assert isinstance(out["user_trust"]["trust_score"], (float, type(None)))

    pol = model.aspect_item_polarity(i_raw)
    assert isinstance(pol, dict) and len(pol) >= 1

    # user_neighborhood returns plain tuples
    nb = model.user_neighborhood(u_raw, top_k=5)
    assert isinstance(nb, list)
    for raw, sc in nb:
        assert isinstance(raw, str) and isinstance(sc, float)


def test_remove_reason_changes_contribution(tmp_path):
    """Removing the top-contributing aspect edge and rebuilding changes it
    (a mini remove-a-reason precheck for P5)."""
    (config, dataset, train_data, valid_data, test_data,
     model, gt, splits) = _build_toy_recbole(tmp_path)
    tr = splits[0]

    # Find an item with a nonzero aspect contribution.
    target_item, target_aspect = None, None
    for (it, aspect), w in gt.has_aspect_weights.items():
        if abs(w) > 0:
            target_item, target_aspect = it, aspect
            break
    assert target_item is not None

    before = model.aspect_item_polarity(target_item).get(target_aspect)
    assert before is not None and before != 0.0

    # Rebuild dropping every sentiment row for (target_item, target_aspect).
    sent_df = pd.read_csv(config["graph_sentiment_csv"])
    drop = [
        (r.user_id, r.item_id, r.aspect, r.sentiment)
        for r in sent_df.itertuples()
        if str(r.item_id) == str(target_item) and str(r.aspect) == str(target_aspect)
    ]
    (_, _, _, _, _, model2, gt2, _) = _build_toy_recbole(
        tmp_path / "v2", drop_sentiment_rows=drop
    )
    after = model2.aspect_item_polarity(target_item).get(target_aspect)
    # The aspect's contribution must change (removed -> absent/different).
    assert after != before


# ---------------------------------------------------------------------------
# f. END-TO-END SMOKE (run_sthgrec against a derived config, 2 epochs)
# ---------------------------------------------------------------------------

def test_end_to_end_smoke(tmp_path):
    (config, dataset, train_data, valid_data, test_data,
     model, gt, splits) = _build_toy_recbole(tmp_path)

    from recbole.trainer import Trainer
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, show_progress=False, verbose=False)
    result = trainer.evaluate(test_data, show_progress=False)

    # metrics finite
    assert len(result) > 0
    for k, v in result.items():
        assert np.isfinite(v), f"{k} = {v} not finite"

    # summary schema matches the existing baseline schema exactly.
    from src.eval.run_sthgrec import flatten_results
    row = flatten_results("STHGRec", config["seed"], dict(result))
    assert row["model"] == "STHGRec" and row["seed"] == 2020
    expected_metric_keys = {"recall@10", "recall@20", "ndcg@10", "ndcg@20"}
    assert expected_metric_keys.issubset({k.lower() for k in row})


# ---------------------------------------------------------------------------
# h. YAML-WINS: ablation flags in YAML actually reach the model (BUG-1 fix)
# ---------------------------------------------------------------------------

def test_yaml_ablation_flags_win(tmp_path):
    """YAML use_sentiment_edges: false must reach Config and the built model.

    This test exercises the REAL file path through load_config() — the same
    function main() now calls — and asserts three things:
      (a) config["use_sentiment_edges"] is False (YAML value won over the
          True default in _CUSTOM_KEYS; would fail against the pre-fix code
          because RecBole 1.2.1 config_dict overrides config_file_list).
      (b) The built GraphTensors has adj_sentiment = None (no sentiment adj).
      (c) The STHGRec model's _has_sentiment is False.

    Pre-fix failure mode: with config_dict=_CUSTOM_KEYS (all defaults), RecBole
    would override the YAML value, reporting use_sentiment_edges=True even when
    the YAML said false — exactly the BUG-1 scenario.
    """
    import tempfile
    import yaml as _yaml

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed
    from src.models.sthgrec import STHGRec
    from src.eval.run_sthgrec import load_config, build_model

    # ------------------------------------------------------------------
    # Build a minimal synthetic dataset so RecBole can create_dataset().
    # ------------------------------------------------------------------
    ds_dir = tmp_path / "dataset" / "ab_toy"
    ds_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    users = [f"u{k}" for k in range(20)]
    items = [f"i{k}" for k in range(15)]

    pairs, ts = [], 1
    for u in users:
        chosen = rng.choice(len(items), size=6, replace=False)
        for c in chosen:
            pairs.append((u, items[c], ts)); ts += 1

    df_all = pd.DataFrame(pairs, columns=["u", "i", "t"])
    def _write_split(rows_df, split_name):
        out = pd.DataFrame({
            "user_id:token": rows_df["u"],
            "item_id:token": rows_df["i"],
            "rating:float": 5.0,
            "timestamp:float": rows_df["t"].astype(float),
        })
        out.to_csv(ds_dir / f"ab_toy.{split_name}.inter", sep="\t", index=False)

    for u, g in df_all.groupby("u"):
        g = g.sort_values("t")
        _write_split(g.iloc[:4], "train")
    # write valid/test (one row each per user, reuse last two rows)
    va_rows, te_rows = [], []
    for u, g in df_all.groupby("u"):
        g = g.sort_values("t")
        va_rows.append(g.iloc[4:5]); te_rows.append(g.iloc[5:6])
    _write_split(pd.concat(va_rows).reset_index(drop=True), "valid")
    _write_split(pd.concat(te_rows).reset_index(drop=True), "test")

    train_inter = ds_dir / "ab_toy.train.inter"

    # Sentiment CSV (present, but will be disabled via YAML flag).
    sent_path = tmp_path / "absa.csv"
    _write_sentiment(sent_path, [
        ("u0", "i0", "gameplay", "Positive"),
        ("u1", "i1", "price", "Negative"),
    ])

    id_map_path = tmp_path / "idmap.json"

    # ------------------------------------------------------------------
    # Write a temp YAML with use_sentiment_edges: false pointing at the
    # synthetic files.
    # ------------------------------------------------------------------
    yaml_content = {
        "model": "STHGRec",
        "dataset": "ab_toy",
        "data_path": str(tmp_path / "dataset"),
        "seed": 2020,
        "reproducibility": True,
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {
            "mode": "full",
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO",
            "group_by": "user",
        },
        "embedding_size": 8,
        "n_layers": 2,
        "reg_weight": 1e-4,
        "train_batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 1,
        # THE FLAG UNDER TEST — must survive into the built model:
        "use_sentiment_edges": False,
        "use_trust_edges": False,
        "graph_inter_csv": str(train_inter),
        "graph_sentiment_csv": str(sent_path),
        "graph_trust_csv": None,
        "graph_id_map_json": str(id_map_path),
        "metrics": ["Recall", "NDCG"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        "device": "cpu",
    }
    yaml_path = tmp_path / "ablation_test.yaml"
    with open(yaml_path, "w") as f:
        _yaml.dump(yaml_content, f)

    # ------------------------------------------------------------------
    # Call the SAME load_config() that main() calls (identical code path).
    # ------------------------------------------------------------------
    config = load_config(yaml_path)

    # (a) YAML ablation flag reaches Config.
    assert config["use_sentiment_edges"] is False, (
        f"Expected False from YAML, got {config['use_sentiment_edges']!r}. "
        "This indicates BUG-1 is NOT fixed: config_dict default (True) is "
        "overriding the YAML value."
    )

    # (b) & (c) Build the model via the same build_model() path and verify the
    # sentiment adjacency and model flag are also False/None.
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    _, valid_data, _ = data_preparation(config, dataset)

    model, gt = build_model(config, dataset)

    # (b) GraphTensors has no sentiment adjacency.
    assert gt.adj_sentiment is None, (
        f"Expected adj_sentiment=None when use_sentiment_edges=False, "
        f"got {type(gt.adj_sentiment)}"
    )

    # (c) Model's internal flag is False.
    assert model._has_sentiment is False, (
        f"Expected model._has_sentiment=False, got {model._has_sentiment!r}"
    )


def test_resolve_device_mps_falls_back_to_cpu():
    """f. DEVICE — a requested 'mps' device is overridden to cpu (no sparse kernels).

    ST-HGRec's torch.sparse.mm propagation has no MPS kernel, so run_sthgrec must
    downgrade mps -> cpu (and write the choice back into the config so the
    Trainer and .to(...) calls agree). cuda-when-unavailable is also downgraded.
    """
    import torch
    from src.eval.run_sthgrec import resolve_device

    class _Cfg(dict):
        pass

    mps_cfg = _Cfg(device=torch.device("mps"))
    assert resolve_device(mps_cfg).type == "cpu"
    assert mps_cfg["device"].type == "cpu"  # written back

    # A plain string is accepted too.
    str_cfg = _Cfg(device="mps")
    assert resolve_device(str_cfg).type == "cpu"

    # cuda is kept iff actually available on this host.
    cuda_cfg = _Cfg(device=torch.device("cuda"))
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(cuda_cfg).type == expected

    # cpu is untouched.
    cpu_cfg = _Cfg(device=torch.device("cpu"))
    assert resolve_device(cpu_cfg).type == "cpu"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-x", "-q"]))
