"""Fluent-but-ungrounded explanation — STUDY VARIABLE ONLY (PRD step 9, T5.2).

.. warning::

    QUARANTINE. This module is a **human-study stimulus**, not a product feature.
    It produces fluent, persuasive text that deliberately references NO graph
    facts — no aspects, no sentiment weights, no trust, no scores. It is the
    *ungrounded* condition in the honest-vs-ungrounded comparison and must NEVER be
    wired into the live recommender or any serving/model/eval/graph/data/sentiment
    code path. ``tests/test_explain.py`` enforces this by scanning ASTs: no module
    under ``src/models``, ``src/eval``, ``src/graph``, ``src/data`` or
    ``src/sentiment`` may import ``src.explain.ungrounded``.

Mirrors ``prompts/ungrounded.txt`` (READ-ONLY): given only an item title/id, write
a generic persuasive blurb. Selection from a fixed template pool is deterministic
given a seed, so study stimuli are reproducible. Like ``honest.py`` it exposes an
optional injectable ``rephrase_fn`` hook for study-time LLM use — but never
imports an LLM SDK.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional


# Generic, persuasive, GRAPH-FREE templates. Each references only ``{item}`` and
# makes no aspect/sentiment/trust/score claim (that is the whole point — this is
# the ungrounded stimulus). ``{item}`` is the sole substitution.
_TEMPLATES = [
    "You're going to love {item} — it's exactly the kind of pick that people "
    "keep coming back to, and we think it'll be a great fit for you.",
    "{item} is a fantastic choice. It has that special something that makes it "
    "stand out, and we're confident it'll quickly become a favorite of yours.",
    "Trust us on this one: {item} is a hidden gem. Once you give it a try, "
    "you'll wonder how you ever went without it.",
    "We picked {item} just for you. It's the sort of standout everyone's "
    "talking about, and something tells us it's right up your alley.",
    "{item} really delivers. It's polished, satisfying, and the kind of thing "
    "that leaves a lasting impression — you won't be disappointed.",
    "Say hello to your next favorite: {item}. It checks all the boxes for a "
    "memorable experience, and we can't wait for you to dive in.",
]


def _seed_index(seed, n: int) -> int:
    """Deterministic template index from an arbitrary seed (or item identity)."""
    key = str(seed).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % n


def explain(
    item_title_or_id,
    seed=None,
    rephrase_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Return a fluent, GRAPH-FREE persuasive blurb for ``item_title_or_id``.

    - ``seed`` selects a template deterministically. ``None`` falls back to the
      item identity itself, so the same item always yields the same stimulus
      unless a seed is supplied to vary it.
    - ``rephrase_fn`` (optional, study-time only): when given, the selected
      template is handed to it as ``rephrase_fn(prompt_block)`` where the block
      mirrors ``prompts/ungrounded.txt`` (``Item:``/``Explanation:``). The caller
      owns any LLM call; this module imports no LLM SDK.

    By construction the output cites NOTHING from the graph — that is what makes it
    the ungrounded condition and makes ``grounding_score`` for it definitionally 0.
    """
    item = str(item_title_or_id)
    idx = _seed_index(seed if seed is not None else item, len(_TEMPLATES))
    template = _TEMPLATES[idx]
    text = template.format(item=item)

    if rephrase_fn is not None:
        block = f"Item: {item}\n\nExplanation:"
        return rephrase_fn(block)
    return text
