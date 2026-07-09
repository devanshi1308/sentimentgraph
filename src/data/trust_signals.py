"""Verified-purchase, helpfulness, consistency.
Computes per-review / per-reviewer trust signals from the raw review text and
fields: verified_purchase, helpful_vote, and reviewer consistency. Feeds the
trust edges in the graph.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PARQUET = REPO_ROOT / "data" / "raw" / "reviews_video_games.parquet"
INTER_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "video_games.inter"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "trust_signals.csv"


def load_graph_users() -> set:
    inter = pd.read_csv(INTER_PATH, sep="\t")
    inter.columns = [c.split(":")[0] for c in inter.columns]
    return set(inter["user_id"].unique())


def load_reviews_for_users(user_ids: set) -> pd.DataFrame:
    reviews = pd.read_parquet(
        REVIEWS_PARQUET,
        columns=["user_id", "parent_asin", "rating", "verified_purchase", "helpful_vote"],
    )
    return reviews[reviews["user_id"].isin(user_ids)]


def compute_trust_signals(reviews: pd.DataFrame) -> pd.DataFrame:
    grouped = reviews.groupby("user_id")

    verified_share = grouped["verified_purchase"].mean()
    avg_helpful_votes = grouped["helpful_vote"].mean()
    rating_std = grouped["rating"].std().fillna(0)
    review_count = grouped.size()

    consistency = 1 / (1 + rating_std)

    trust = pd.DataFrame({
        "user_id": verified_share.index,
        "verified_purchase_share": verified_share.values,
        "avg_helpful_votes": avg_helpful_votes.values,
        "rating_consistency": consistency.values,
        "review_count": review_count.values,
    })

    max_helpful = trust["avg_helpful_votes"].max() or 1
    normalized_helpful = trust["avg_helpful_votes"] / max_helpful

    trust["trust_score"] = (
        trust["verified_purchase_share"]
        + normalized_helpful
        + trust["rating_consistency"]
    ) / 3

    return trust.sort_values("trust_score", ascending=False).reset_index(drop=True)


def main() -> None:
    print("Loading graph's user population (5-core filtered)...")
    graph_users = load_graph_users()
    print(f"  {len(graph_users):,} users")

    print("Loading their reviews from raw data...")
    reviews = load_reviews_for_users(graph_users)
    print(f"  {len(reviews):,} reviews matched")

    print("Computing trust signals...")
    trust = compute_trust_signals(reviews)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trust.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(trust):,} user trust profiles to {OUTPUT_PATH}")

    print("\n=== Sample: top 10 most trusted users ===")
    print(trust.head(10).to_string(index=False))

    print("\n=== Sample: bottom 10 least trusted users ===")
    print(trust.tail(10).to_string(index=False))

    print("\n=== Summary stats ===")
    print(trust[["verified_purchase_share", "avg_helpful_votes",
                 "rating_consistency", "trust_score"]].describe())


if __name__ == "__main__":
    main()
