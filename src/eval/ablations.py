"""Switch-off ablation driver for ST-HGRec (PRD step 10 / STHGREC_DESIGN.md §6).

Runs the ST-HGRec run matrix — four variants that switch off heterogeneous graph
components one at a time — by writing per-variant temp YAMLs that flow through
the SAME ``load_config`` path as the main runner (exercising the YAML-wins
semantics fixed by BUG-1).

Ablation matrix (STHGREC_DESIGN.md §6):

  ============  ====================  ================  ========================
  Variant       use_sentiment_edges   use_trust_edges   Purpose
  ============  ====================  ================  ========================
  full          true                  true              Headline ST-HGRec number
  -sentiment    false                 true              Isolates sentiment value
  -trust        true                  false             Isolates trust value
  -both         false                 false             ≈ LightGCN sanity anchor
  ============  ====================  ================  ========================

Mechanism: for each variant, load the BASE yaml, write a temp variant YAML that
overrides only ``use_sentiment_edges`` and ``use_trust_edges`` (all other keys
pass through unchanged), then build+train+evaluate via the run_sthgrec machinery.
All variants share the same ``graph_id_map_json`` path so the raw<->index
contract is stable across runs (stale-map assertion in ``build_graph_tensors``
would fire if each variant regenerated a mismatched map).

Output: ``results/ablations/summary.csv`` with schema
``variant,model,seed,recall@10,recall@20,ndcg@10,ndcg@20`` (append semantics:
header written once, no duplicate headers).

Usage:
    python -m src.eval.ablations [--config configs/sthgrec.yaml]
        [--variants full,-sentiment,-trust,-both]
        [--epochs N]
        [--out results/ablations/summary.csv]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from src.eval.run_sthgrec import load_config, build_model, _CUSTOM_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sthgrec.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "ablations" / "summary.csv"

# The four canonical variants (STHGREC_DESIGN.md §6).
ALL_VARIANTS = ["full", "-sentiment", "-trust", "-both"]

# Ablation flags per variant: (use_sentiment_edges, use_trust_edges)
_VARIANT_FLAGS: Dict[str, Dict[str, bool]] = {
    "full":       {"use_sentiment_edges": True,  "use_trust_edges": True},
    "-sentiment": {"use_sentiment_edges": False, "use_trust_edges": True},
    "-trust":     {"use_sentiment_edges": True,  "use_trust_edges": False},
    "-both":      {"use_sentiment_edges": False, "use_trust_edges": False},
}

# CSV schema (variant prepended to the baselines schema).
_SCHEMA_COLS = ["variant", "model", "seed", "recall@10", "recall@20", "ndcg@10", "ndcg@20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ST-HGRec ablation matrix (STHGREC_DESIGN.md §6)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Base YAML config (default: configs/sthgrec.yaml).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ALL_VARIANTS),
        help=(
            "Comma-separated list of variants to run "
            "(default: full,-sentiment,-trust,-both)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs (useful for smoke/testing).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output CSV path (default: results/ablations/summary.csv).",
    )
    return parser.parse_args()


def _load_base_yaml(config_path: Path) -> dict:
    """Load the base YAML file as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _write_variant_yaml(
    base_yaml: dict,
    flags: Dict[str, bool],
    tmp_dir: str,
    variant_name: str,
    epochs_override: Optional[int],
) -> Path:
    """Write a temp YAML for one variant with the ablation flags patched in.

    All keys from the base YAML are preserved; only ``use_sentiment_edges`` and
    ``use_trust_edges`` are overridden. An optional ``epochs`` override flows
    through the same YAML so the config path exercises the YAML-wins semantics
    (not a dict-level side-channel).
    """
    variant_yaml = dict(base_yaml)
    variant_yaml.update(flags)
    if epochs_override is not None:
        variant_yaml["epochs"] = epochs_override

    safe_name = variant_name.lstrip("-") or "both"
    out_path = Path(tmp_dir) / f"variant_{safe_name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(variant_yaml, f, default_flow_style=False)
    return out_path


def _append_ablation_row(row: dict, out_path: Path) -> None:
    """Append one variant row to the ablation summary CSV.

    Header is written exactly once (when the file does not exist yet); on
    subsequent calls rows are appended with no duplicate header.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(out_path, index=False)


def _flatten_variant_row(
    variant: str, model_name: str, seed: int, results: dict
) -> dict:
    """Build one CSV row dict in the ablation schema."""
    row: dict = {"variant": variant, "model": model_name, "seed": seed}
    # Normalise metric keys to lower-case so they match the expected schema.
    for k, v in results.items():
        row[k.lower() if k.lower() in _SCHEMA_COLS else k.lower()] = v
    return row


def run_variant(
    variant: str,
    base_yaml: dict,
    tmp_dir: str,
    epochs_override: Optional[int],
) -> dict:
    """Build, train, and evaluate one ablation variant.

    Returns a dict row ready for the summary CSV.

    Steps mirror run_sthgrec.main() exactly — one source of truth for the
    harness flow:
      1. Write a temp YAML with the variant flags patched in (temp-YAML route
         exercises the same YAML-wins load_config path).
      2. Call load_config(yaml_path) → Config.
      3. Re-run init_seed so every row is seed-2020 reproducible.
      4. create_dataset + data_preparation.
      5. build_model (same function as the runner + smoke test).
      6. Trainer.fit + evaluate.
      7. Return the flattened metric dict.

    The ``graph_id_map_json`` key is taken from the base YAML and passed through
    unchanged so all variants share the same raw<->index contract (the
    stale-map assertion in build_graph_tensors enforces this).
    """
    flags = _VARIANT_FLAGS[variant]

    # Step 1 — write temp variant YAML.
    yaml_path = _write_variant_yaml(base_yaml, flags, tmp_dir, variant, epochs_override)

    # Step 2 — load config via the SAME path as run_sthgrec.main().
    config = load_config(yaml_path)
    model_name = base_yaml.get("model", "STHGRec")

    # Step 3 — re-seed before each variant build for row-level reproducibility.
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    # Steps 4-5 — dataset + model.
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model, _ = build_model(config, train_data.dataset)

    # Step 6 — train + evaluate.
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, show_progress=False)
    test_result = trainer.evaluate(test_data, show_progress=False)

    return _flatten_variant_row(variant, model_name, config["seed"], dict(test_result))


def _print_comparison_table(rows: List[dict]) -> None:
    """Print a small comparison table: variant rows, delta vs 'full'."""
    if not rows:
        return

    # Locate the full-variant row (may not exist if user ran a subset).
    full_row = next((r for r in rows if r.get("variant") == "full"), None)
    metric_cols = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]

    header = f"{'variant':<14}  " + "  ".join(f"{m:>12}" for m in metric_cols)
    if full_row:
        header += "  " + "  ".join(f"{'Δ '+m:>13}" for m in metric_cols)
    print()
    print("=== Ablation comparison ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row.get('variant','?'):<14}  "
        line += "  ".join(f"{row.get(m, float('nan')):>12.4f}" for m in metric_cols)
        if full_row:
            deltas = [row.get(m, float("nan")) - full_row.get(m, float("nan")) for m in metric_cols]
            line += "  " + "  ".join(f"{d:>+13.4f}" for d in deltas)
        print(line)
    print()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in _VARIANT_FLAGS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {unknown}. "
            f"Must be one of: {list(_VARIANT_FLAGS.keys())}"
        )

    base_yaml = _load_base_yaml(config_path)

    print(f"Base config : {config_path}")
    print(f"Variants    : {variants}")
    print(f"Output CSV  : {out_path}")
    if args.epochs is not None:
        print(f"Epochs      : {args.epochs} (override)")
    print()

    collected_rows: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="sthgrec_ablate_") as tmp_dir:
        for variant in variants:
            print(f"--- Running variant: {variant} ---")
            row = run_variant(
                variant=variant,
                base_yaml=base_yaml,
                tmp_dir=tmp_dir,
                epochs_override=args.epochs,
            )
            _append_ablation_row(row, out_path)
            collected_rows.append(row)
            print(f"    → Appended row to {out_path}")

    _print_comparison_table(collected_rows)
    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    main()
