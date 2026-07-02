"""Popularity, BPR, NGCF, LightGCN.

Runs each baseline through the one RecBole/RecBole-GNN harness using the configs/
YAMLs, and records Recall@K and NDCG@K to results/baselines/. Every run traces
back to a config file and a seed.
"""

"""Run a single RecBole baseline model from a config file.

Usage:
    python -m src.eval.run_baselines --config configs/popularity.yaml
    python -m src.eval.run_baselines --config configs/bpr.yaml

Writes Recall@K / NDCG@K results to results/baselines/<model>.csv,
appending to a shared summary table so every run traces back to its
config and seed (PRD §9 reproducibility spine).
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.pop import Pop
from recbole.model.general_recommender.bpr import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "baselines"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

MODEL_CLASSES = {
    "Pop": Pop,
    "BPR": BPR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def flatten_results(model_name: str, seed: int, results: dict) -> dict:
    row = {"model": model_name, "seed": seed}
    row.update(results)
    return row


def main() -> None:
    args = parse_args()
    config_path = REPO_ROOT / args.config

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    model_name = raw_config["model"]
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model in this script: {model_name}")

    config = Config(model=model_name, config_file_list=[str(config_path)])
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_class = MODEL_CLASSES[model_name]
    model = model_class(config, train_data.dataset).to(config["device"])

    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, show_progress=False)
    test_result = trainer.evaluate(test_data, show_progress=False)

    print(f"=== {model_name} test results ===")
    for k, v in test_result.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = flatten_results(model_name, config["seed"], dict(test_result))

    row_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        existing = pd.read_csv(SUMMARY_PATH)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(SUMMARY_PATH, index=False)
    print(f"Appended results to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()