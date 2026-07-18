"""Shared toy-graph builders for the ST-HGRec test suite.

Provides a self-contained synthetic RecBole dataset + STHGRec model builder that
mirrors the pattern in ``tests/test_sthgrec.py`` (which keeps its own private copy
unchanged). ``tests/test_explain.py`` uses this to build a small connected graph
with known aspect/sentiment/trust facts so the honest/ungrounded/faithfulness
paths can be exercised without touching real data.

This module is NEW and imported only by the explanation tests; the pre-existing
model-core tests continue to use their own in-file helpers, so their behavior is
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_sentiment(path: Path, rows) -> None:
    pd.DataFrame(rows, columns=["user_id", "item_id", "aspect", "sentiment"]).to_csv(
        path, index=False
    )


def write_trust(path: Path, rows) -> None:
    pd.DataFrame(rows, columns=["user_id", "trust_score"]).to_csv(path, index=False)


def build_toy_recbole(
    tmp_path,
    sentiment: bool = True,
    trust: bool = True,
    drop_sentiment_rows=None,
    seed: int = 2020,
):
    """Build a tiny RecBole dataset + STHGRec model (connected toy graph).

    Returns ``(config, dataset, train_data, valid_data, test_data, model, gt,
    (tr, va, te), gt_build_args)`` where ``gt_build_args`` is the exact kwargs dict
    passed to ``build_graph_tensors`` (so the faithfulness rebuild can reuse it,
    including the shared ``id_map_json``).
    """
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    from src.eval.run_sthgrec import _CUSTOM_KEYS, build_model
    from src.models.sthgrec import STHGRec

    ds_dir = tmp_path / "dataset" / "toy"
    ds_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    users = [f"u{k}" for k in range(30)]
    items = [f"i{k}" for k in range(20)]
    aspects = ["gameplay", "graphics", "price"]

    pairs = []
    ts = 1
    for u in users:
        chosen = rng.choice(len(items), size=6, replace=False)
        for c in chosen:
            pairs.append((u, items[c], ts))
            ts += 1

    def split_rows(rows):
        df = pd.DataFrame(rows, columns=["u", "i", "t"])
        tr, va, te = [], [], []
        for u, g in df.groupby("u"):
            g = g.sort_values("t")
            tr.append(g.iloc[:4])
            va.append(g.iloc[4:5])
            te.append(g.iloc[5:6])
        return pd.concat(tr), pd.concat(va), pd.concat(te)

    tr, va, te = split_rows(pairs)

    def write_split(df, name):
        out = pd.DataFrame(
            {
                "user_id:token": df["u"],
                "item_id:token": df["i"],
                "rating:float": 5.0,
                "timestamp:float": df["t"].astype(float),
            }
        )
        out.to_csv(ds_dir / f"toy.{name}.inter", sep="\t", index=False)

    write_split(tr, "train")
    write_split(va, "valid")
    write_split(te, "test")
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
        write_sentiment(sent_path, srows)

    trust_path = None
    if trust:
        trust_path = tmp_path / "trust.csv"
        write_trust(trust_path, [(u, float(rng.uniform(0.2, 1.0))) for u in users])

    id_map_json = tmp_path / "idmap.json"

    cfg = dict(_CUSTOM_KEYS)
    cfg.update(
        {
            "model": "STHGRec",
            "dataset": "toy",
            "data_path": str(tmp_path / "dataset"),
            "seed": seed,
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
            "epochs": 2,
            "use_sentiment_edges": sentiment,
            "use_trust_edges": trust,
            "graph_inter_csv": str(train_inter),
            "graph_sentiment_csv": str(sent_path) if sent_path else None,
            "graph_trust_csv": str(trust_path) if trust_path else None,
            "graph_id_map_json": str(id_map_json),
            "metrics": ["Recall", "NDCG"],
            "topk": [10, 20],
            "valid_metric": "NDCG@10",
            "device": "cpu",
        }
    )

    config = Config(model=STHGRec, config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model, gt = build_model(config, train_data.dataset)

    gt_build_args = {
        "inter_csv": str(train_inter),
        "sentiment_csv": str(sent_path) if sent_path else None,
        "trust_csv": str(trust_path) if trust_path else None,
        "use_sentiment_edges": sentiment,
        "use_trust_edges": trust,
        "id_map_json": str(id_map_json),
    }

    return (
        config,
        dataset,
        train_data,
        valid_data,
        test_data,
        model,
        gt,
        (tr, va, te),
        gt_build_args,
    )
