"""Tests for the ST-HGRec ablation driver (src/eval/ablations.py).

Tests are self-contained: all fixture code is inline (copied/adapted from
test_sthgrec.py's test_end_to_end_smoke pattern). This file MUST NOT import
from tests/test_sthgrec.py and MUST NOT edit it — another agent owns that
surface.

Run:
    /tmp/p4venv/bin/python -m pytest tests/test_ablations.py -q

Total runtime target: < 90 s.

Acceptance criteria (per task spec):
  a. Output CSV has exactly the requested variant rows with the exact schema.
  b. All metrics are finite.
  c. The -sentiment variant structurally dropped the sentiment adjacency
     (verified via load_config/build_model path).
  d. Running the driver twice appends without duplicating headers.
  e. Full-variant metrics are identical across two deterministic runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Inline fixture helpers (adapted from test_sthgrec.py's fixture pattern)
# ---------------------------------------------------------------------------

def _write_inter(path: Path, rows_df: "pd.DataFrame") -> None:
    """Write a RecBole-style tab-sep .inter (:token/:float headers)."""
    out = pd.DataFrame({
        "user_id:token": rows_df["u"],
        "item_id:token": rows_df["i"],
        "rating:float": 5.0,
        "timestamp:float": rows_df["t"].astype(float),
    })
    out.to_csv(path, sep="\t", index=False)


def _write_sentiment(path: Path, rows) -> None:
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "aspect", "sentiment"])
    df.to_csv(path, index=False)


def _write_trust(path: Path, rows) -> None:
    df = pd.DataFrame(rows, columns=["user_id", "trust_score"])
    df.to_csv(path, index=False)


def _build_synthetic_dataset(tmp_path: Path) -> dict:
    """Build a tiny synthetic dataset in a tmp dir and return all relevant paths.

    Returns a dict with:
      ds_dir, train_inter, sent_path, trust_path, id_map_path, users, items,
      tr (train DataFrame with columns u/i/t)

    The dataset has 30 users x 20 items, 6 interactions per user, split 4/1/1
    (train/valid/test), with sentiment rows on the train pairs and trust scores
    for all users. This mirrors the _build_toy_recbole fixture in test_sthgrec.py.
    """
    ds_dir = tmp_path / "dataset" / "abl"
    ds_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    users = [f"u{k}" for k in range(30)]
    items = [f"i{k}" for k in range(20)]
    aspects = ["gameplay", "graphics", "price"]

    # Dense-ish interactions: each user gets 6 items.
    pairs = []
    ts = 1
    for u in users:
        chosen = rng.choice(len(items), size=6, replace=False)
        for c in chosen:
            pairs.append((u, items[c], ts))
            ts += 1

    df_all = pd.DataFrame(pairs, columns=["u", "i", "t"])

    # Per-user split: 4 train / 1 valid / 1 test (temporal order).
    tr_parts, va_parts, te_parts = [], [], []
    for _, grp in df_all.groupby("u"):
        grp = grp.sort_values("t")
        tr_parts.append(grp.iloc[:4])
        va_parts.append(grp.iloc[4:5])
        te_parts.append(grp.iloc[5:6])
    tr = pd.concat(tr_parts).reset_index(drop=True)
    va = pd.concat(va_parts).reset_index(drop=True)
    te = pd.concat(te_parts).reset_index(drop=True)

    _write_inter(ds_dir / "abl.train.inter", tr)
    _write_inter(ds_dir / "abl.valid.inter", va)
    _write_inter(ds_dir / "abl.test.inter",  te)

    train_inter = ds_dir / "abl.train.inter"

    # Sentiment: one row per train (u,i) pair.
    srows = []
    for row in tr.itertuples():
        a = aspects[hash((row.u, row.i)) % len(aspects)]
        pol = "Positive" if (hash((row.i, a)) % 2 == 0) else "Negative"
        srows.append((row.u, row.i, a, pol))
    sent_path = tmp_path / "absa.csv"
    _write_sentiment(sent_path, srows)

    # Trust: one row per user.
    trust_path = tmp_path / "trust.csv"
    _write_trust(trust_path, [(u, float(rng.uniform(0.2, 1.0))) for u in users])

    id_map_path = tmp_path / "idmap.json"

    return {
        "ds_dir": ds_dir,
        "train_inter": train_inter,
        "sent_path": sent_path,
        "trust_path": trust_path,
        "id_map_path": id_map_path,
        "users": users,
        "items": items,
        "tr": tr,
        "dataset_name": "abl",
        "data_path": str(tmp_path / "dataset"),
    }


def _make_base_yaml(info: dict) -> dict:
    """Build the base YAML dict that ablations.run_variant will patch."""
    return {
        "model": "STHGRec",
        "dataset": info["dataset_name"],
        "data_path": info["data_path"],
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
        "epochs": 2,
        "use_sentiment_edges": True,
        "use_trust_edges": True,
        "sentiment_weight": 1.0,
        "aspect_weight": 1.0,
        "trust_weight": 1.0,
        "require_pow": False,
        "graph_inter_csv": str(info["train_inter"]),
        "graph_sentiment_csv": str(info["sent_path"]),
        "graph_trust_csv": str(info["trust_path"]),
        "graph_id_map_json": str(info["id_map_path"]),
        "metrics": ["Recall", "NDCG"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        "device": "cpu",
    }


def _write_base_yaml(tmp_path: Path, base_yaml: dict) -> Path:
    yaml_path = tmp_path / "base.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(base_yaml, f, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# a + b. CSV schema and finite metrics
# ---------------------------------------------------------------------------

def test_csv_schema_and_finite_metrics(tmp_path):
    """(a) Output CSV has exactly the requested variant rows + exact schema.
    (b) All metrics are finite.
    """
    import tempfile
    from src.eval.ablations import run_variant, _append_ablation_row, _SCHEMA_COLS

    info = _build_synthetic_dataset(tmp_path)
    base_yaml = _make_base_yaml(info)
    out_path = tmp_path / "out" / "summary.csv"

    variants_to_run = ["full", "-sentiment", "-both"]

    with tempfile.TemporaryDirectory(prefix="abl_test_") as tmp_dir:
        for variant in variants_to_run:
            row = run_variant(
                variant=variant,
                base_yaml=base_yaml,
                tmp_dir=tmp_dir,
                epochs_override=2,
            )
            _append_ablation_row(row, out_path)

    assert out_path.exists(), "summary.csv was not created"
    df = pd.read_csv(out_path)

    # (a) Exactly 3 rows.
    assert len(df) == len(variants_to_run), (
        f"Expected {len(variants_to_run)} rows, got {len(df)}"
    )

    # (a) Exact schema: all required columns present.
    for col in _SCHEMA_COLS:
        assert col in df.columns, f"Missing column: {col!r}"

    # (a) Variant column contains exactly the expected variants.
    assert set(df["variant"]) == set(variants_to_run), (
        f"Variant values mismatch: {set(df['variant'])} vs {set(variants_to_run)}"
    )

    # (b) All metrics are finite.
    metric_cols = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]
    for col in metric_cols:
        for val in df[col]:
            assert math.isfinite(val), f"Non-finite metric {col}={val}"


# ---------------------------------------------------------------------------
# c. Structural proof: -sentiment variant dropped the sentiment adjacency
# ---------------------------------------------------------------------------

def test_sentiment_off_structural_proof(tmp_path):
    """(c) The -sentiment variant's build really dropped the sentiment adjacency.

    We exercise the same load_config/build_model path that the driver uses:
    write the variant YAML (with use_sentiment_edges=False), call load_config,
    then build_model and assert adj_sentiment is None on the returned GraphTensors.
    """
    import tempfile
    from src.eval.ablations import _write_variant_yaml, _VARIANT_FLAGS, _load_base_yaml
    from src.eval.run_sthgrec import load_config, build_model
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    info = _build_synthetic_dataset(tmp_path)
    base_yaml = _make_base_yaml(info)

    # First run 'full' to write the shared id-map JSON.
    with tempfile.TemporaryDirectory(prefix="abl_c_") as tmp_dir:
        # Write and load the 'full' variant first so the id-map JSON is created.
        full_yaml_path = _write_variant_yaml(
            base_yaml, _VARIANT_FLAGS["full"], tmp_dir, "full", epochs_override=2
        )
        full_cfg = load_config(full_yaml_path)
        init_seed(full_cfg["seed"], full_cfg["reproducibility"])
        full_ds = create_dataset(full_cfg)
        full_train, _, _ = data_preparation(full_cfg, full_ds)
        build_model(full_cfg, full_train.dataset)  # writes id-map JSON as side effect

        # Now write and load the '-sentiment' variant YAML.
        sent_yaml_path = _write_variant_yaml(
            base_yaml, _VARIANT_FLAGS["-sentiment"], tmp_dir, "-sentiment",
            epochs_override=2
        )
        sent_cfg = load_config(sent_yaml_path)

        # (c-i) Config flag must be False (YAML-wins semantics exercised).
        assert sent_cfg["use_sentiment_edges"] is False, (
            f"Expected use_sentiment_edges=False in Config, got "
            f"{sent_cfg['use_sentiment_edges']!r}"
        )

        # (c-ii) Build model and verify adj_sentiment is None.
        init_seed(sent_cfg["seed"], sent_cfg["reproducibility"])
        ds = create_dataset(sent_cfg)
        train_data, _, _ = data_preparation(sent_cfg, ds)
        model, gt = build_model(sent_cfg, train_data.dataset)

        assert gt.adj_sentiment is None, (
            f"Expected adj_sentiment=None when use_sentiment_edges=False, "
            f"got {type(gt.adj_sentiment)}"
        )
        assert model._has_sentiment is False, (
            f"Expected model._has_sentiment=False, got {model._has_sentiment!r}"
        )


# ---------------------------------------------------------------------------
# d. Append semantics: no duplicate headers
# ---------------------------------------------------------------------------

def test_append_no_duplicate_headers(tmp_path):
    """(d) Running the driver twice appends without duplicating headers."""
    import tempfile
    from src.eval.ablations import run_variant, _append_ablation_row

    info = _build_synthetic_dataset(tmp_path)
    base_yaml = _make_base_yaml(info)
    out_path = tmp_path / "dedup" / "summary.csv"

    # Run once (full variant only) to create the file.
    with tempfile.TemporaryDirectory(prefix="abl_d1_") as tmp_dir:
        row1 = run_variant(
            variant="full",
            base_yaml=base_yaml,
            tmp_dir=tmp_dir,
            epochs_override=2,
        )
    _append_ablation_row(row1, out_path)

    # Run again (full variant again) to exercise the append path.
    with tempfile.TemporaryDirectory(prefix="abl_d2_") as tmp_dir:
        row2 = run_variant(
            variant="-both",
            base_yaml=base_yaml,
            tmp_dir=tmp_dir,
            epochs_override=2,
        )
    _append_ablation_row(row2, out_path)

    # Read raw lines to check there is exactly one header line.
    lines = out_path.read_text().strip().splitlines()
    header_lines = [l for l in lines if l.startswith("variant")]
    assert len(header_lines) == 1, (
        f"Expected exactly 1 header line, found {len(header_lines)}: {header_lines}"
    )

    # And the CSV should have exactly 2 data rows.
    df = pd.read_csv(out_path)
    assert len(df) == 2, f"Expected 2 rows after two appends, got {len(df)}"


# ---------------------------------------------------------------------------
# e. Determinism: full-variant metrics identical across two runs
# ---------------------------------------------------------------------------

def test_determinism_full_variant(tmp_path):
    """(e) Full-variant metrics are identical across two seed-2020 runs."""
    import tempfile
    from src.eval.ablations import run_variant

    info = _build_synthetic_dataset(tmp_path)
    base_yaml = _make_base_yaml(info)

    # Run 1.
    with tempfile.TemporaryDirectory(prefix="abl_e1_") as tmp_dir:
        row1 = run_variant(
            variant="full",
            base_yaml=base_yaml,
            tmp_dir=tmp_dir,
            epochs_override=2,
        )

    # Run 2.
    with tempfile.TemporaryDirectory(prefix="abl_e2_") as tmp_dir:
        row2 = run_variant(
            variant="full",
            base_yaml=base_yaml,
            tmp_dir=tmp_dir,
            epochs_override=2,
        )

    metric_cols = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]
    for col in metric_cols:
        v1 = row1.get(col, float("nan"))
        v2 = row2.get(col, float("nan"))
        assert v1 == v2, (
            f"Non-deterministic metric {col}: run1={v1}, run2={v2}"
        )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-x", "-q"]))
