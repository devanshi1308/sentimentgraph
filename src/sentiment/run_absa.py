"""PyABSA -> raw open-vocabulary aspect/sentiment table.

Runs in the ABSA env only. Applies PyABSA over all reviews belonging to
the train-split user population — full-scale run, not the 300-review sanity
sample. Batched + checkpointed since this is a long-running job. Never imports
from the recommender side; the two halves meet only through files in data/.

Output policy ("extract all, normalize after")
-----------------------------------------------
This script writes EVERY open-vocabulary aspect PyABSA emits into
data/interim/absa_full.csv (columns: user_id, item_id, aspect, sentiment).
No filtering or canonicalization happens here. A SEPARATE normalization step
(src/sentiment/normalize_aspects.py) maps raw aspects onto the canonical
vocabulary defined in aspects.CANONICAL_ASPECTS and writes
data/interim/absa_normalized.csv. The graph layer consumes ONLY the normalized
file. See fixes.md #6 for the rationale and project-lead decision.
"""

# Workaround: metric_visualizer calls update_checker.UpdateChecker.check()
# with a signature the installed update_checker version doesn't support.
# This crashes on import even though the update-check feature (pinging
# PyPI for newer versions) is unrelated to ABSA functionality — disable it.
import update_checker
update_checker.UpdateChecker.check = lambda self, *args, **kwargs: None

import argparse
from pathlib import Path

import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PARQUET = REPO_ROOT / "data" / "raw" / "reviews_video_games.parquet"
INTER_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "video_games.train.inter"  # leakage guard: sentiment population = train users only
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "absa_full.csv"
CHECKPOINT_PATH = REPO_ROOT / "data" / "interim" / "absa_full_progress.txt"

# Outer chunk size = how many reviews are processed between checkpoint writes.
# PyABSA batches internally on top of this. 32 is a safe CPU default; on a CUDA
# box bump it (e.g. --batch-size 256) so checkpointing overhead is amortised.
DEFAULT_BATCH_SIZE = 32


def load_graph_users() -> set:
    inter = pd.read_csv(INTER_PATH, sep="\t")
    inter.columns = [c.split(":")[0] for c in inter.columns]
    return set(inter["user_id"].unique())


def load_target_reviews(user_ids: set) -> pd.DataFrame:
    reviews = pd.read_parquet(
        REVIEWS_PARQUET, columns=["user_id", "parent_asin", "text"]
    )
    reviews = reviews[reviews["user_id"].isin(user_ids)]
    reviews = reviews.dropna(subset=["text"])
    reviews = reviews[reviews["text"].str.len() > 20]
    return reviews.reset_index(drop=True)


def get_start_index() -> int:
    if CHECKPOINT_PATH.exists():
        return int(CHECKPOINT_PATH.read_text().strip())
    return 0


def save_checkpoint(index: int) -> None:
    CHECKPOINT_PATH.write_text(str(index))


def run_absa_batched(reviews: pd.DataFrame, extractor, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    start = get_start_index()
    print(f"Resuming from row {start:,} / {len(reviews):,}")

    write_header = not OUTPUT_PATH.exists() or start == 0

    for batch_start in range(start, len(reviews), batch_size):
        batch = reviews.iloc[batch_start:batch_start + batch_size]
        texts = batch["text"].tolist()

        results = extractor.predict(texts, print_result=False)
        if not isinstance(results, list):
            results = [results]

        rows = []
        for (_, row), result in zip(batch.iterrows(), results):
            aspects = result.get("aspect", [])
            sentiments = result.get("sentiment", [])
            for aspect, sentiment in zip(aspects, sentiments):
                rows.append({
                    "user_id": row["user_id"],
                    "item_id": row["parent_asin"],
                    "aspect": aspect,
                    "sentiment": sentiment,
                })

        if rows:
            pd.DataFrame(rows).to_csv(
                OUTPUT_PATH, mode="a", header=write_header, index=False
            )
            write_header = False

        save_checkpoint(batch_start + batch_size)

        if batch_start % (batch_size * 50) == 0:
            print(f"  ...processed {batch_start:,} / {len(reviews):,} reviews")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-scale PyABSA extraction (checkpointed).")
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto (let PyABSA pick CUDA if present), cpu, cuda, cuda:N, or "
            "allcuda (all visible GPUs). Default: auto."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Reviews per checkpoint chunk (default: {DEFAULT_BATCH_SIZE}; use ~256 on GPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading graph users...")
    users = load_graph_users()

    print("Loading target reviews (5-core filtered population only)...")
    reviews = load_target_reviews(users)
    print(f"  {len(reviews):,} reviews to process")

    # PyABSA's `auto_device`: True = auto-pick (CUDA when available), or an
    # explicit device string ("cpu", "cuda:0", "allcuda", ...).
    auto_device = True if args.device == "auto" else args.device
    print(f"Loading PyABSA model (device={args.device})...")
    extractor = ATEPC.AspectExtractor("multilingual", auto_device=auto_device)

    print(f"Running batched extraction (batch_size={args.batch_size}, checkpointed)...")
    run_absa_batched(reviews, extractor, batch_size=args.batch_size)

    print(f"\nDone. Results in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()