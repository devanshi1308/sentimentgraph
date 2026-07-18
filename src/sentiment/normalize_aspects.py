"""Normalize raw PyABSA aspects onto the canonical vocabulary.

Reads data/interim/absa_full.csv (open-vocabulary output from run_absa.py) and maps
each raw aspect term to a canonical aspect defined in aspects.CANONICAL_ASPECTS, then
writes data/interim/absa_normalized.csv. Rows whose raw aspect term cannot be mapped to
any canonical are DROPPED and counted in the coverage report.

Matching algorithm
------------------
1. Lowercase and strip the raw aspect string.
2. Iterate CANONICAL_ASPECTS in its insertion order (dict is ordered in Python 3.7+).
3. For each canonical, check whether ANY of its variant substrings appears inside the
   lowercased raw aspect (``variant in raw_lower``).
4. The FIRST canonical whose any variant matches wins. This is the tie-break rule:
   canonical insertion order in aspects.py takes priority, so the four PRD-locked
   aspects (gameplay, graphics, price, difficulty) always beat the additional ones.
5. If no canonical matches, the row is dropped.

Environment note
----------------
Pure pandas — no pyabsa, no recbole, no torch imports. This script is deliberately
runnable from EITHER the ABSA env OR the recommender env, as long as pandas is installed.

Usage
-----
    python normalize_aspects.py                          # defaults
    python normalize_aspects.py --in /path/to/full.csv --out /path/to/norm.csv
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from aspects import CANONICAL_ASPECTS

# ---------------------------------------------------------------------------
# Paths (overridable via CLI)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_IN = _REPO_ROOT / "data" / "interim" / "absa_full.csv"
_DEFAULT_OUT = _REPO_ROOT / "data" / "interim" / "absa_normalized.csv"


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _build_canonical_map(canonical_aspects: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """Return canonicals as an ordered list of (canonical, variants) pairs.

    Using a list preserves the deterministic tie-break order even if the caller
    passes an arbitrary dict (though in practice CANONICAL_ASPECTS is already ordered).
    """
    return list(canonical_aspects.items())


def map_to_canonical(raw_aspect: str, ordered_canonicals: list[tuple[str, list[str]]]) -> str | None:
    """Return the canonical name for *raw_aspect*, or None if no match.

    Matching is case-insensitive substring: the first canonical (in insertion order)
    whose any variant string is a substring of the lowercased+stripped raw aspect wins.
    """
    raw_lower = raw_aspect.lower().strip()
    for canonical, variants in ordered_canonicals:
        for variant in variants:
            if variant in raw_lower:
                return canonical
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def normalize(in_path: Path, out_path: Path) -> None:
    """Read absa_full, normalize aspects, write absa_normalized, print coverage report."""
    print(f"Reading  : {in_path}")
    df = pd.read_csv(in_path)

    required_cols = {"user_id", "item_id", "aspect", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: input CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    total_rows = len(df)
    ordered_canonicals = _build_canonical_map(CANONICAL_ASPECTS)

    # Map each raw aspect to a canonical (or None)
    df["canonical"] = df["aspect"].apply(
        lambda a: map_to_canonical(str(a), ordered_canonicals)
    )

    unmatched_mask = df["canonical"].isna()
    unmatched_df = df[unmatched_mask].copy()
    matched_df = df[~unmatched_mask].copy()

    matched_rows = len(matched_df)
    dropped_rows = len(unmatched_df)

    # Build output with canonical aspect replacing raw aspect
    out_df = matched_df[["user_id", "item_id", "canonical", "sentiment"]].rename(
        columns={"canonical": "aspect"}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Writing  : {out_path}")

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    coverage_pct = 100 * matched_rows / total_rows if total_rows > 0 else 0.0

    print()
    print("=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"Total rows        : {total_rows:>10,}")
    print(f"Matched rows      : {matched_rows:>10,}  ({coverage_pct:.1f}%)")
    print(f"Dropped (unmatched): {dropped_rows:>9,}  ({100 - coverage_pct:.1f}%)")
    print()

    # Per-canonical counts
    print("Per-canonical counts (matched rows):")
    canonical_counts = matched_df["canonical"].value_counts()
    for canonical, count in canonical_counts.items():
        print(f"  {canonical:<15s}: {count:>8,}")
    print()

    # Top-20 unmatched raw aspects (feed the lead's curation loop)
    if dropped_rows > 0:
        top_unmatched = (
            Counter(unmatched_df["aspect"].str.lower().str.strip())
            .most_common(20)
        )
        print(f"Top-20 unmatched raw aspect terms (out of {dropped_rows:,} dropped rows):")
        for term, count in top_unmatched:
            print(f"  {count:>6,}  {term!r}")
    else:
        print("No unmatched rows — 100% coverage.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw PyABSA aspects to canonical vocabulary."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=_DEFAULT_IN,
        help=f"Input CSV (default: {_DEFAULT_IN})",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output CSV (default: {_DEFAULT_OUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize(args.in_path, args.out_path)
