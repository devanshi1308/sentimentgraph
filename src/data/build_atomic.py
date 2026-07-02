"""5-core split -> .inter/.item/.user.

Converts the official 5-core benchmark split into RecBole atomic files under
data/processed/video_games/: video_games.inter (user, item, rating, timestamp),
video_games.item (item-side features), video_games.user (user-side features).
These atomic files are the only thing the recommender reads.
"""

from pathlib import Path

import pandas as pd

from src.data.download import stream_download, RAW_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "video_games"

BENCHMARK_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/timestamp/Video_Games"
SPLIT_FILES = {
    "train": f"{BENCHMARK_BASE}.train.csv",
    "valid": f"{BENCHMARK_BASE}.valid.csv",
    "test": f"{BENCHMARK_BASE}.test.csv",
}

META_PARQUET = RAW_DIR / "meta_video_games.parquet"


def download_splits() -> dict[str, Path]:
    paths = {}
    for name, url in SPLIT_FILES.items():
        dest = RAW_DIR / f"Video_Games.{name}.csv"
        paths[name] = stream_download(url, dest)
    return paths


def build_inter(split_paths: dict[str, Path]) -> pd.DataFrame:
    dfs = []
    for name, path in split_paths.items():
        df = pd.read_csv(path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # RecBole atomic .inter columns: user_id:token  item_id:token  rating:float  timestamp:float
    inter = pd.DataFrame({
        "user_id:token": combined["user_id"],
        "item_id:token": combined["parent_asin"],
        "rating:float": combined["rating"].astype(float),
        "timestamp:float": combined["timestamp"].astype(float),
    })
    return inter


def build_item(inter: pd.DataFrame) -> pd.DataFrame:
    item_ids = inter["item_id:token"].unique()
    meta = pd.read_parquet(META_PARQUET)
    meta = meta[meta["parent_asin"].isin(item_ids)].drop_duplicates(subset="parent_asin")

    item = pd.DataFrame({
        "item_id:token": meta["parent_asin"],
        "title:token_seq": meta["title"].fillna("").astype(str),
        "average_rating:float": pd.to_numeric(meta.get("average_rating"), errors="coerce"),
        "price:token": meta.get("price", pd.Series(dtype=str)).fillna("unknown").astype(str),
    })
    return item


def build_user(inter: pd.DataFrame) -> pd.DataFrame:
    user_ids = inter["user_id:token"].unique()
    user = pd.DataFrame({"user_id:token": user_ids})
    return user


def write_atomic_file(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"  wrote {path} ({len(df):,} rows)")


def main() -> None:
    print("Downloading official 5-core timestamp split...")
    split_paths = download_splits()

    print("Building .inter ...")
    inter = build_inter(split_paths)
    write_atomic_file(inter, PROCESSED_DIR / "video_games.inter")

    print("Building .item ...")
    item = build_item(inter)
    write_atomic_file(item, PROCESSED_DIR / "video_games.item")

    print("Building .user ...")
    user = build_user(inter)
    write_atomic_file(user, PROCESSED_DIR / "video_games.user")

    print("=== Atomic Files Summary ===")
    print(f"Interactions: {len(inter):,}")
    print(f"Items:        {inter['item_id:token'].nunique():,}")
    print(f"Users:        {inter['user_id:token'].nunique():,}")


if __name__ == "__main__":
    main()