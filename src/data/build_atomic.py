"""5-core split -> .inter/.item/.user.

Converts the official 5-core benchmark split into RecBole atomic files under
data/processed/video_games/: video_games.inter (user, item, rating, timestamp),
video_games.item (item-side features), video_games.user (user-side features).
These atomic files are the only thing the recommender reads.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
