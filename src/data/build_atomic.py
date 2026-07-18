"""5-core split -> .train/.valid/.test .inter + .item/.user.

Converts the official Amazon-2023 5-core timestamp benchmark split into RecBole
atomic files under data/processed/video_games/. We emit THREE interaction files
(video_games.train.inter, video_games.valid.inter, video_games.test.inter) so
RecBole's `benchmark_filename` mechanism honors the official temporal split
verbatim instead of re-splitting randomly. No combined video_games.inter is
written: downstream consumers (graph, trust, ABSA) read train.inter only, as a
test-leakage guard. video_games.item / video_games.user are built from the union
of ids across all three splits so RecBole's id map covers valid/test too.
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


def build_inter(path: Path) -> pd.DataFrame:
    """Convert one official split CSV into a RecBole atomic .inter DataFrame."""
    df = pd.read_csv(path)

    # RecBole atomic .inter columns: user_id:token  item_id:token  rating:float  timestamp:float
    inter = pd.DataFrame({
        "user_id:token": df["user_id"],
        "item_id:token": df["parent_asin"],
        "rating:float": df["rating"].astype(float),
        "timestamp:float": df["timestamp"].astype(float),
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

    print("Building per-split .inter files ...")
    # One atomic .inter per official split; RecBole's benchmark_filename reads
    # these directly (train/valid/test) and skips its own eval_args splitter.
    split_inters: dict[str, pd.DataFrame] = {}
    for name in ("train", "valid", "test"):
        inter = build_inter(split_paths[name])
        split_inters[name] = inter
        write_atomic_file(inter, PROCESSED_DIR / f"video_games.{name}.inter")

    # .item / .user must span all three splits so RecBole's id remapping covers
    # valid/test users and items that never appear in train.
    union = pd.concat(split_inters.values(), ignore_index=True)

    print("Building .item ...")
    item = build_item(union)
    write_atomic_file(item, PROCESSED_DIR / "video_games.item")

    print("Building .user ...")
    user = build_user(union)
    write_atomic_file(user, PROCESSED_DIR / "video_games.user")

    print("=== Atomic Files Summary ===")
    for name, inter in split_inters.items():
        print(f"{name:>5} interactions: {len(inter):,}")
    print(f"Total interactions: {len(union):,}")
    print(f"Items (union):      {union['item_id:token'].nunique():,}")
    print(f"Users (union):      {union['user_id:token'].nunique():,}")


if __name__ == "__main__":
    main()