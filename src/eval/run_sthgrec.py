"""Dedicated runner for ST-HGRec (the heterogeneous LightGCN, PRD step 8 / Option D).

Mirrors ``src/eval/run_baselines.py``'s harness flow (STHGREC_DESIGN.md §5): a
RecBole ``Config`` from the YAML drives ``init_seed`` + ``create_dataset`` +
``data_preparation`` so ST-HGRec evaluates on the IDENTICAL official 5-core
benchmark split (``benchmark_filename`` machinery) with the same metrics. Unlike
the baselines, ST-HGRec builds its own typed sparse adjacency from the source
CSVs (``graph_tensors``) and is constructed as ``STHGRec(config, dataset,
graph_tensors=...)`` — hence a separate entrypoint that leaves the
reproducibility-critical baseline runner untouched (STHGREC_DESIGN.md §5).

It reuses RecBole's ``Trainer``/``evaluate`` verbatim (Option B harness) and
appends one row to ``results/baselines/summary.csv`` in the exact existing
schema ``model,seed,recall@10,recall@20,ndcg@10,ndcg@20`` so ST-HGRec slots in
alongside Pop / BPR / LightGCN / NGCF.

Usage:
    python -m src.eval.run_sthgrec --config configs/sthgrec.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from src.models.sthgrec import STHGRec
from src.models.graph_tensors import build_graph_tensors

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "baselines"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

# Custom config keys ST-HGRec reads; declared so RecBole's Config accepts them.
_CUSTOM_KEYS = {
    "use_sentiment_edges": True,
    "use_trust_edges": True,
    "sentiment_weight": 1.0,
    "aspect_weight": 1.0,
    "trust_weight": 1.0,
    "require_pow": False,
    "graph_inter_csv": None,
    "graph_sentiment_csv": None,
    "graph_trust_csv": None,
    "graph_id_map_json": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sthgrec.yaml")
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help=(
            "Build the model and run two seeded forward+backward passes on the "
            "resolved device, asserting bit-identical embeddings and gradients. "
            "Run this once on a new GPU before trusting a headline number."
        ),
    )
    return parser.parse_args()


def resolve_device(config) -> torch.device:
    """Resolve (and, for MPS, override) the training device.

    RecBole's ``_init_device`` already downgrades ``cuda`` -> ``cpu`` when no
    CUDA device is visible. It never selects Apple MPS, but a YAML could set
    ``device: mps`` directly -- and ST-HGRec's ``torch.sparse.mm`` propagation
    has no MPS kernel, so we force CPU there with a loud warning. The resolved
    device is written back into ``config`` so the Trainer and ``.to(...)`` calls
    all agree.
    """
    dev = config["device"]
    if not isinstance(dev, torch.device):
        dev = torch.device(dev)

    if dev.type == "mps":
        logging.getLogger().warning(
            "device 'mps' requested but Apple MPS has no sparse-tensor kernels; "
            "ST-HGRec propagation would crash. Falling back to device: cpu. "
            "Use a CUDA host for GPU acceleration."
        )
        dev = torch.device("cpu")
    elif dev.type == "cuda" and not torch.cuda.is_available():
        logging.getLogger().warning("device 'cuda' requested but unavailable; using cpu.")
        dev = torch.device("cpu")

    config["device"] = dev
    return dev


def check_determinism(config, dataset) -> None:
    """Two seeded forward+backward passes must match bit-for-bit on this device."""

    def _one_pass():
        init_seed(config["seed"], config["reproducibility"])
        model, _ = build_model(config, dataset)
        model.train()
        emb = model.forward()
        # A fixed, structure-touching scalar loss so the spmm backward runs.
        loss = (emb * emb).sum()
        loss.backward()
        grad = model.embedding.weight.grad.detach().clone()
        return emb.detach().clone(), grad

    emb_a, grad_a = _one_pass()
    emb_b, grad_b = _one_pass()

    emb_ok = torch.equal(emb_a, emb_b)
    grad_ok = torch.equal(grad_a, grad_b)
    dev = config["device"]
    print(f"=== determinism check on {dev} ===")
    print(f"  forward embeddings identical : {emb_ok}")
    print(f"  backward gradients identical : {grad_ok}")
    if emb_ok and grad_ok:
        print("  PASS - seeded runs are bit-stable on this device.")
    else:
        max_e = (emb_a - emb_b).abs().max().item()
        max_g = (grad_a - grad_b).abs().max().item()
        print(f"  FAIL - max |Δemb|={max_e:.3e}, max |Δgrad|={max_g:.3e}")
        raise SystemExit(1)


def _resolve(path_str, default: Path) -> Path:
    """Resolve a repo-relative config path, falling back to a default."""
    if path_str is None:
        return default
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def build_model(config, dataset):
    """Construct the graph tensors + ST-HGRec model from a RecBole config.

    Shared by the runner and the end-to-end smoke test so both exercise the
    identical construction + RecBole-id translation path.
    """
    inter_csv = _resolve(
        config["graph_inter_csv"],
        REPO_ROOT / "data" / "processed" / "video_games" / "video_games.train.inter",
    )
    sentiment_csv = _resolve(
        config["graph_sentiment_csv"],
        REPO_ROOT / "data" / "interim" / "absa_normalized.csv",
    )
    trust_csv = _resolve(
        config["graph_trust_csv"],
        REPO_ROOT / "data" / "interim" / "trust_signals.csv",
    )
    id_map_json = _resolve(
        config["graph_id_map_json"],
        REPO_ROOT / "data" / "processed" / "video_games" / "sthgrec_id_maps.json",
    )

    use_sentiment = bool(config["use_sentiment_edges"])
    use_trust = bool(config["use_trust_edges"])

    gt = build_graph_tensors(
        inter_csv=inter_csv,
        sentiment_csv=sentiment_csv,
        trust_csv=trust_csv,
        use_sentiment_edges=use_sentiment,
        use_trust_edges=use_trust,
        id_map_json=id_map_json,
    )
    model = STHGRec(config, dataset, graph_tensors=gt).to(config["device"])
    return model, gt


def flatten_results(model_name: str, seed: int, results: dict) -> dict:
    row = {"model": model_name, "seed": seed}
    row.update(results)
    return row


def append_summary(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        existing = pd.read_csv(SUMMARY_PATH)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(SUMMARY_PATH, index=False)


def load_config(config_path: Path) -> "Config":
    """Build a RecBole Config where YAML ablation flags win over code defaults.

    RecBole 1.2.1 merge order: config_dict values OVERRIDE config_file_list
    values — the opposite of what a comment in the old code claimed. To ensure
    YAML ablation flags (e.g. use_sentiment_edges: false) reach the model, we
    manually extract any custom keys present in the YAML and fold them into
    config_dict on top of the defaults. YAML-defined custom values therefore
    always win; defaults only fill keys absent from the YAML.
    """
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f) or {}

    # Extract whichever of the custom keys the YAML explicitly sets.
    yaml_custom = {k: raw_yaml[k] for k in _CUSTOM_KEYS if k in raw_yaml}

    # YAML values override defaults (right-side wins in {**a, **b}).
    merged_custom = {**_CUSTOM_KEYS, **yaml_custom}

    return Config(
        model=STHGRec,
        config_file_list=[str(config_path)],
        config_dict=merged_custom,
    )


def main() -> None:
    args = parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    model_name = raw_config.get("model", "STHGRec")

    # YAML custom-key values win: load_config() merges them above _CUSTOM_KEYS
    # defaults before constructing Config (see load_config docstring).
    config = load_config(config_path)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    device = resolve_device(config)
    logging.getLogger().info(f"ST-HGRec device: {device}")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if args.check_determinism:
        check_determinism(config, train_data.dataset)
        return

    model, _ = build_model(config, train_data.dataset)

    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, show_progress=False)
    test_result = trainer.evaluate(test_data, show_progress=False)

    print(f"=== {model_name} test results ===")
    for k, v in test_result.items():
        print(f"  {k}: {v}")

    row = flatten_results(model_name, config["seed"], dict(test_result))
    append_summary(row)
    print(f"Appended results to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
