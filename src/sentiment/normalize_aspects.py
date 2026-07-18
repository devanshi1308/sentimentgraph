"""Normalize raw PyABSA aspects onto the canonical vocabulary.

Reads data/interim/absa_full.csv (open-vocabulary output from run_absa.py) and maps
each raw aspect term to a canonical aspect defined in aspects.CANONICAL_ASPECTS, then
writes data/interim/absa_normalized.csv. Rows whose raw aspect term cannot be mapped to
any canonical are DROPPED and counted in the coverage report.

Matching algorithm
------------------
Variants are precompiled into regexes at setup time using two modes (see aspects.py):

  * PREFIX variant (ends with '*'):
    Stem matched as a word-start-anchored prefix: ``(?i)(?<!\\w)<stem>``
    e.g. ``"graphic*"`` matches "graphics" but not "autographic".

  * WHOLE-WORD / WHOLE-PHRASE variant (no '*'):
    Matched with word boundaries on both sides: ``(?i)(?<!\\w)<variant>(?!\\w)``
    e.g. ``"hard"`` only matches standalone "hard", not "hardware".

  * Special case — '$':
    The dollar sign is treated as a plain substring (``"$" in raw_lower``) rather than
    a regex-bounded variant, because boundary anchors make it unmatchable when flanked
    by digits (e.g. "$60"). This is the only variant that bypasses the regex path.

Tie-break: the FIRST canonical (in insertion order of CANONICAL_ASPECTS) whose any
variant matches wins. The four PRD-locked aspects (gameplay, graphics, price, difficulty)
always beat the additional ones.

Performance: to handle millions of rows efficiently, we build a term->canonical lookup
from df['aspect'].unique() first, then map the full column — O(unique_terms) regex work
instead of O(rows).

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
import re
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

# Sentinel used to tag the special-case plain-substring '$' variant.
_DOLLAR_SENTINEL = "$"


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------

def _compile_variant(variant: str) -> re.Pattern | None:
    """Compile a single variant string into a regex pattern.

    Returns None for the '$' special case — handled separately as plain substring.
    """
    if variant == _DOLLAR_SENTINEL:
        return None  # special-cased as plain substring match

    if variant.endswith("*"):
        # PREFIX mode: word-start-anchored, no trailing boundary
        stem = re.escape(variant[:-1])
        return re.compile(rf"(?i)(?<!\w){stem}")
    else:
        # WHOLE-WORD/PHRASE mode: boundaries on both sides
        escaped = re.escape(variant)
        return re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")


def _build_compiled_canonicals(
    canonical_aspects: dict[str, list[str]],
) -> list[tuple[str, list[tuple[re.Pattern | None, str]]]]:
    """Return an ordered list of (canonical, [(pattern_or_None, raw_variant), ...]) tuples.

    pattern_or_None is None for the '$' special case, in which case the raw_variant
    string '$' is used for a plain substring test instead.
    """
    result = []
    for canonical, variants in canonical_aspects.items():
        compiled = [((_compile_variant(v), v)) for v in variants]
        result.append((canonical, compiled))
    return result


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _map_term(
    raw_term: str,
    compiled_canonicals: list[tuple[str, list[tuple[re.Pattern | None, str]]]],
) -> str | None:
    """Return the canonical name for *raw_term*, or None if no match.

    Iterates canonicals in insertion order; first match wins.
    """
    raw_lower = raw_term.lower().strip()
    for canonical, patterns in compiled_canonicals:
        for pattern, raw_variant in patterns:
            if pattern is None:
                # '$' special case: plain substring
                if _DOLLAR_SENTINEL in raw_lower:
                    return canonical
            else:
                if pattern.search(raw_lower):
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

    # Precompile regexes once
    compiled_canonicals = _build_compiled_canonicals(CANONICAL_ASPECTS)

    # Performance optimisation: build term->canonical map from UNIQUE raw aspects only,
    # then map the full column. Avoids re-running regex matching for repeated terms.
    unique_terms = df["aspect"].unique()
    term_map: dict[str, str | None] = {
        t: _map_term(str(t), compiled_canonicals) for t in unique_terms
    }

    df["canonical"] = df["aspect"].map(term_map)

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
