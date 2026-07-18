"""Verified-purchase, helpfulness, consistency -> per-user trust scores.

Computes per-reviewer trust signals from the raw review fields
(``verified_purchase``, ``helpful_vote``, and rating consistency) and emits one
``trust_score`` per user. This feeds ``build_graph.py``, which attaches the
score as a user-node attribute (typed-edge conversion deferred to P4).

Wiring (P1, test-leakage guard):
  * The user population is read from ``video_games.train.inter`` ONLY (same
    rationale as the graph: valid/test-only users must not enter the structure
    ST-HGRec trains over). See ``load_graph_users``.

DESIGN DECISION — scope of the trust aggregate:
  Trust statistics are computed over ALL of a user's raw reviews (all reviews by
  users present in the TRAIN population), NOT restricted to reviews matching
  train (user, item) interactions. Rationale: trust here is a stable,
  slowly-varying per-user *reputation* aggregate (verified-purchase share,
  helpfulness, rating consistency), not an interaction-specific label; computing
  it over the user's full review history maximizes the sample per user and makes
  the score robust and reproducible from the raw parquet without depending on the
  exact split. The cost is mild temporal leakage (a user's aggregate reflects
  some test-period reviews), but this is a per-user prior, not a per-(user,item)
  supervision signal, so it does not leak the held-out labels ST-HGRec is
  evaluated on; restricting to train items would shrink per-user samples and make
  low-activity users' scores noisier for negligible leakage benefit. If a
  stricter train-only variant is later required for an ablation, join reviews on
  the train (user, item) pairs before ``compute_trust_signals``.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PARQUET = REPO_ROOT / "data" / "raw" / "reviews_video_games.parquet"
# TRAIN split only — leakage guard: the trust user population must match the
# users ST-HGRec trains on, not valid/test-only users. Combined .inter is dead.
INTER_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "video_games.train.inter"
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
    print("Loading graph's user population (5-core, TRAIN split only)...")
    graph_users = load_graph_users()
    print(f"  {len(graph_users):,} users")

    print("Loading their reviews from raw data (all reviews per user)...")
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
