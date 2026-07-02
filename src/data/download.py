"""Pull HF parquet configs.

Downloads the Amazon Reviews 2023 Video Games configs from HuggingFace
(McAuley-Lab/Amazon-Reviews-2023): raw_review_Video_Games and
raw_meta_Video_Games. Load the Parquet configs directly to dodge the
deprecated load-script / trust_remote_code issue. Writes to data/raw/.
"""

from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

REVIEWS_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Video_Games.jsonl"
META_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl"

REVIEWS_JSONL = RAW_DIR / "Video_Games.jsonl"
META_JSONL = RAW_DIR / "meta_Video_Games.jsonl"


def stream_download(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> Path:
    if dest.exists():
        print(f"  already have {dest}, skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  ...{downloaded:,} / {total:,} bytes ({pct:.1f}%)", end="\r")
        print()
    return dest


def write_parquet_from_jsonl(
    path: Path,
    output_path: Path,
    chunksize: int = 200_000,
    dtype: dict | None = None,
) -> None:
    writer = None
    reader = pd.read_json(path, lines=True, chunksize=chunksize, dtype=dtype)
    for i, chunk in enumerate(reader):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        print(f"  ...wrote {((i + 1) * chunksize):,} rows so far")
    if writer is not None:
        writer.close()


def print_stats(reviews: pd.DataFrame, meta: pd.DataFrame) -> None:
    n_users = reviews["user_id"].nunique()
    n_items = reviews["parent_asin"].nunique()
    n_interactions = len(reviews)
    density = n_interactions / (n_users * n_items)

    print("=== Video Games — Basic Stats ===")
    print(f"Users:         {n_users:,}")
    print(f"Items:         {n_items:,}")
    print(f"Interactions:  {n_interactions:,}")
    print(f"Density:       {density:.6f}")
    print(f"Meta rows:     {len(meta):,}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = stream_download(REVIEWS_URL, REVIEWS_JSONL)
    meta_path = stream_download(META_URL, META_JSONL)

    print("Parsing reviews jsonl...")
    write_parquet_from_jsonl(reviews_path, RAW_DIR / "reviews_video_games.parquet")

    print("Parsing meta jsonl...")
    write_parquet_from_jsonl(
        meta_path,
        RAW_DIR / "meta_video_games.parquet",
        dtype={"price": str},
    )

    reviews = pd.read_parquet(RAW_DIR / "reviews_video_games.parquet")
    meta = pd.read_parquet(RAW_DIR / "meta_video_games.parquet")

    print_stats(reviews, meta)


if __name__ == "__main__":
    main()