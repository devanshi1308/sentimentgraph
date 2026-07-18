"""Graph-grounded explanation — the *honest* study condition (PRD step 9, T5.1).

The honest condition renders **only** facts already extracted from the ST-HGRec
graph via the model's explanation hooks (STHGREC_DESIGN.md §7): per-aspect signed
``has_aspect`` weights, the supporting ``absa_normalized`` sentiment rows behind
the item, and the user's trust context (Option 4A). No new aspect, sentiment, or
claim is ever invented — every rendered field is traceable to hook output.

POLICY (orchestrator + lead): TEMPLATE-FIRST. ``render`` is a deterministic,
dependency-free template. LLM rephrasing exists *only* as an injectable hook — an
optional ``rephrase_fn: Callable[[str], str]``. When provided, the caller supplies
the LLM call and we hand it a facts block formatted per
``prompts/honest_rephrase.txt``'s ``Facts:``/``Explanation:`` contract. When
``None`` (the default) the output is pure templates. **No LLM SDK is imported
anywhere in src/** — this module has zero non-stdlib imports.

Kept apart from ``ungrounded.py`` so study conditions never bleed together.
"""

from __future__ import annotations

from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Fact extraction — every field traceable to a model hook (STHGREC_DESIGN.md §7)
# ---------------------------------------------------------------------------

def extract_facts(model, user_raw_id, item_raw_id, top_k: int = 3) -> dict:
    """Pull a structured, hook-traceable fact set for one ``(user, item)`` pair.

    Reads exclusively from ``model.explain_pair`` (+ ``aspect_item_polarity`` for
    the item-side polarity of each cited aspect). Nothing here derives content the
    hooks did not already return.

    Returned dict::

        {
          "user_id": str, "item_id": str,
          "score": float | None,
          "aspects": [                       # top-K by |contribution|, ties by name
            {"aspect": str, "contribution": float, "polarity": "positive"|
                                                    "negative"|"neutral"},
            ...
          ],
          "supporting_sentiment_edges": {    # counts + examples per aspect
            "counts": {aspect: {"Positive": int, "Negative": int, "Neutral": int}},
            "examples": [{"aspect": str, "sentiment": str}, ...],  # verbatim rows
            "total": int,
          },
          "user_trust": {"trust_score": float | None, "role": str},
        }

    HARD GUARANTEE: ``aspects`` contains no aspect absent from
    ``model.explain_pair(...)["aspect_contributions"]``.
    """
    pair = model.explain_pair(user_raw_id, item_raw_id)

    contributions = dict(pair["aspect_contributions"])  # aspect -> signed weight
    polarity = model.aspect_item_polarity(item_raw_id)  # same source, defensive

    # Stable ordering: |contribution| desc, ties broken by aspect name asc.
    ordered = sorted(contributions.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    top = ordered[: top_k] if top_k is not None else ordered

    aspects = []
    for aspect, contrib in top:
        # Prefer the polarity-hook value if present; fall back to the contribution.
        signed = float(polarity.get(aspect, contrib))
        aspects.append(
            {
                "aspect": str(aspect),
                "contribution": float(contrib),
                "polarity": _polarity_label(signed),
            }
        )

    # Supporting sentiment edges: verbatim (aspect, sentiment) rows for this item.
    raw_edges = pair["supporting_sentiment_edges"]
    counts: dict = {}
    for e in raw_edges:
        a = str(e["aspect"])
        s = str(e["sentiment"])
        bucket = counts.setdefault(a, {"Positive": 0, "Negative": 0, "Neutral": 0})
        if s in bucket:
            bucket[s] += 1
        else:
            bucket[s] = 1
    # Deterministic example ordering: aspect asc, then sentiment asc.
    examples = sorted(
        ({"aspect": str(e["aspect"]), "sentiment": str(e["sentiment"])} for e in raw_edges),
        key=lambda d: (d["aspect"], d["sentiment"]),
    )

    supporting = {
        "counts": counts,
        "examples": examples,
        "total": len(raw_edges),
    }

    return {
        "user_id": str(user_raw_id),
        "item_id": str(item_raw_id),
        "score": pair["score"],
        "aspects": aspects,
        "supporting_sentiment_edges": supporting,
        "user_trust": dict(pair["user_trust"]),
    }


def _polarity_label(signed: float, eps: float = 1e-9) -> str:
    if signed > eps:
        return "positive"
    if signed < -eps:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Deterministic template renderer — cites ONLY facts (STHGREC_DESIGN.md §7)
# ---------------------------------------------------------------------------

def render(facts: dict) -> str:
    """Deterministic 2-4 sentence plain-language explanation citing only ``facts``.

    Mentions no aspect name absent from ``facts["aspects"]``. Ordering is stable
    (facts already come sorted by |contribution| desc, ties by name). Trust context
    is mentioned only when a trust_score is present and trust is active.
    """
    aspects = facts.get("aspects", [])
    sentences: list[str] = []

    if not aspects:
        sentences.append(
            f"We recommend this item ({facts['item_id']}) based on how users "
            "similar to you have interacted with it."
        )
    else:
        positives = [a["aspect"] for a in aspects if a["polarity"] == "positive"]
        negatives = [a["aspect"] for a in aspects if a["polarity"] == "negative"]
        neutrals = [a["aspect"] for a in aspects if a["polarity"] == "neutral"]

        clauses: list[str] = []
        if positives:
            clauses.append(f"reviews from users like you are positive about {_join(positives)}")
        if negatives:
            clauses.append(f"there are concerns about {_join(negatives)}")
        if neutrals and not positives and not negatives:
            clauses.append(f"users discuss {_join(neutrals)}")
        elif neutrals:
            clauses.append(f"opinions on {_join(neutrals)} are mixed")

        lead = "Recommended because " + _join_clauses(clauses) + "."
        sentences.append(lead)

    # Supporting-evidence sentence (only when there is real evidence).
    total = facts.get("supporting_sentiment_edges", {}).get("total", 0)
    if total:
        noun = "review comment" if total == 1 else "review comments"
        sentences.append(
            f"This draws on {total} {noun} about the aspects above from the review graph."
        )

    # Trust sentence — only when we actually have a trust fact in play.
    trust = facts.get("user_trust", {})
    tscore = trust.get("trust_score")
    role = str(trust.get("role", ""))
    if tscore is not None and "disabled" not in role:
        sentences.append(
            f"Your reviewing history was weighted by a trust score of {tscore:.2f} "
            "when forming this recommendation."
        )

    return " ".join(sentences)


def _join(items: list[str]) -> str:
    """Oxford-comma join with stable input ordering preserved."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _join_clauses(clauses: list[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]}, while {clauses[1]}"
    return "; ".join(clauses)


# ---------------------------------------------------------------------------
# Facts-block formatting (matches prompts/honest_rephrase.txt {facts} contract)
# ---------------------------------------------------------------------------

def format_facts_block(facts: dict) -> str:
    """Render ``facts`` into the ``Facts:`` payload the honest prompt expects.

    Mirrors ``prompts/honest_rephrase.txt`` (READ-ONLY) which contains::

        Facts:
        {facts}

        Explanation:

    We fill only the ``{facts}`` region: a compact, deterministic, human-readable
    block. We deliberately reproduce the prompt's ``Facts:``/``Explanation:``
    scaffold here (rather than editing the prompt file) so ``rephrase_fn`` receives
    a fully-formed prompt string.
    """
    lines = [f"Recommended item: {facts['item_id']}"]
    if facts.get("score") is not None:
        lines.append(f"Match score: {facts['score']:.4f}")

    lines.append("Aspects (most influential first):")
    for a in facts.get("aspects", []):
        lines.append(
            f"  - {a['aspect']}: {a['polarity']} (signed contribution {a['contribution']:+.3f})"
        )
    if not facts.get("aspects"):
        lines.append("  - (no aspect-level sentiment available)")

    total = facts.get("supporting_sentiment_edges", {}).get("total", 0)
    lines.append(f"Supporting review comments: {total}")
    for ex in facts.get("supporting_sentiment_edges", {}).get("examples", []):
        lines.append(f"  - {ex['aspect']}: {ex['sentiment']}")

    trust = facts.get("user_trust", {})
    tscore = trust.get("trust_score")
    if tscore is not None:
        lines.append(f"User trust score: {tscore:.3f} ({trust.get('role', '')})")
    else:
        lines.append(f"User trust: {trust.get('role', 'n/a')}")

    facts_body = "\n".join(lines)
    return f"Facts:\n{facts_body}\n\nExplanation:"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def explain(
    model,
    user_raw_id,
    item_raw_id,
    top_k: int = 3,
    rephrase_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Return ``{"facts": <dict>, "text": <str>}`` for a ``(user, item)`` pair.

    - ``rephrase_fn is None`` (DEFAULT): ``text`` is the deterministic template
      render — pure, dependency-free, no LLM.
    - ``rephrase_fn`` given: ``text = rephrase_fn(format_facts_block(facts))`` — the
      caller owns the LLM call; we only hand it the facts block formatted per
      ``prompts/honest_rephrase.txt``. The rephraser must not introduce content
      absent from the block (enforced upstream by prompt + facts contract).

    HARD GUARANTEE: ``facts["aspects"]`` cite only aspects present in the model's
    hook output, and the template ``render`` mentions no aspect outside ``facts``.
    """
    facts = extract_facts(model, user_raw_id, item_raw_id, top_k=top_k)
    if rephrase_fn is not None:
        text = rephrase_fn(format_facts_block(facts))
    else:
        text = render(facts)
    return {"facts": facts, "text": text}
