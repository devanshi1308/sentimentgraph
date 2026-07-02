"""PyABSA -> aspect/sentiment table.

Runs in the ABSA env only. Applies PyABSA over the review text and writes an
aspect→sentiment table to a file in data/. Never imports from the recommender
side; the two halves meet only through files in data/.
"""

# Workaround: metric_visualizer calls update_checker.UpdateChecker.check()
# with a signature the installed update_checker version doesn't support.
# This crashes on import even though the update-check feature (pinging
# PyPI for newer versions) is unrelated to ABSA functionality — disable it.
import update_checker
update_checker.UpdateChecker.check = lambda self, *args, **kwargs: None

from pathlib import Path

import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC

from src.sentiment.aspects import ASPECTS

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PARQUET = REPO_ROOT / "data" / "raw" / "reviews_video_games.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "absa_sample.csv"

SAMPLE_SIZE = 300


def load_sample() -> pd.DataFrame:
    reviews = pd.read_parquet(REVIEWS_PARQUET, columns=["user_id", "parent_asin", "text"])
    reviews = reviews.dropna(subset=["text"])
    reviews = reviews[reviews["text"].str.len() > 20]
    return reviews.sample(n=SAMPLE_SIZE, random_state=2020).reset_index(drop=True)


def run_absa(sample: pd.DataFrame) -> pd.DataFrame:
    extractor = ATEPC.AspectExtractor("multilingual", auto_device=True)

    rows = []
    for i, row in sample.iterrows():
        result = extractor.predict(row["text"], print_result=False)
        aspects = result.get("aspect", [])
        sentiments = result.get("sentiment", [])
        for aspect, sentiment in zip(aspects, sentiments):
            rows.append({
                "review_index": i,
                "user_id": row["user_id"],
                "item_id": row["parent_asin"],
                "aspect": aspect,
                "sentiment": sentiment,
                "text_snippet": row["text"][:150],
            })
        if i % 50 == 0:
            print(f"  ...processed {i}/{len(sample)} reviews")

    return pd.DataFrame(rows)


def filter_to_fixed_aspects(results: pd.DataFrame) -> pd.DataFrame:
    mask = results["aspect"].str.lower().apply(
        lambda a: any(fixed in a for fixed in ASPECTS)
    )
    return results[mask].reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Sampling reviews...")
    sample = load_sample()

    print("Running PyABSA...")
    raw_results = run_absa(sample)

    print(f"\nRaw aspect mentions extracted: {len(raw_results)}")
    results = filter_to_fixed_aspects(raw_results)
    print(f"Mentions matching fixed ASPECTS list: {len(results)}")

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(results)} aspect-sentiment pairs to {OUTPUT_PATH}")

    print("\n=== Coverage per fixed aspect ===")
    for aspect in ASPECTS:
        count = (results["aspect"].str.lower().str.contains(aspect)).sum()
        print(f"  {aspect}: {count} mentions")

    print("\n=== First 20 examples for eyeballing ===")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()