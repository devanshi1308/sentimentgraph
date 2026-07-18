"""Explanation-layer acceptance tests (PRD step 9 / T5.1-T5.3).

Run:  /tmp/p4venv/bin/python -m pytest tests/test_explain.py -q

Covers (letters map to the P5-EXPLAIN task spec):
  a. honest facts match the model hook output exactly on the synthetic build.
  b. rendered honest text cites only fact aspects (no hallucinated aspect).
  c. determinism: same inputs -> identical text.
  d. rephrase_fn injection: stub receives a block containing the facts; its
     return becomes text (no LLM anywhere).
  e. ungrounded quarantine: AST scan proves no recommender module imports it.
  f. ungrounded text makes no graph-derived aspect-contribution claim.
  g. faithfulness: removing a CITED aspect's evidence changes score/contribution
     (honest explanation scores 1.0); uncited aspects are not required to change.
  h. grounding_score: honest facts -> 1.0, a fabricated aspect list -> < 1.0.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explain import honest, ungrounded, faithfulness  # noqa: E402
from tests.fixtures import build_toy_recbole  # noqa: E402


# ---------------------------------------------------------------------------
# Session-scoped toy model (one expensive build shared across the honest tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("explain_toy")
    bundle = build_toy_recbole(tmp_path)
    return bundle


def _pick_pair_with_aspects(model, gt, tr):
    """Find a (user, item) train pair whose item has >=1 aspect contribution."""
    for r in tr.itertuples():
        u, i = r.u, r.i
        if model.aspect_item_polarity(i):
            return u, i
    # Fallback: any train pair.
    r0 = tr.iloc[0]
    return r0.u, r0.i


# ---------------------------------------------------------------------------
# a. honest facts match hook output exactly
# ---------------------------------------------------------------------------

def test_facts_match_hook_output(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    facts = honest.extract_facts(model, u, i, top_k=3)
    hook = model.explain_pair(u, i)

    # score is passed straight through.
    assert facts["score"] == hook["score"]
    assert facts["user_id"] == str(u) and facts["item_id"] == str(i)

    # Every fact aspect must exist in the hook's aspect_contributions (HARD GUARANTEE).
    hook_aspects = set(hook["aspect_contributions"])
    fact_aspects = {a["aspect"] for a in facts["aspects"]}
    assert fact_aspects <= hook_aspects, (
        f"facts invented aspects not in hook: {fact_aspects - hook_aspects}"
    )

    # Contributions equal the hook values.
    for a in facts["aspects"]:
        assert a["contribution"] == hook["aspect_contributions"][a["aspect"]]

    # Top-K ordering: sorted by |contribution| desc, ties by name asc.
    ordered = sorted(
        hook["aspect_contributions"].items(), key=lambda kv: (-abs(kv[1]), kv[0])
    )[:3]
    assert [a["aspect"] for a in facts["aspects"]] == [k for k, _ in ordered]

    # supporting edges preserved verbatim (as a set of (aspect, sentiment)).
    hook_edges = {(e["aspect"], e["sentiment"]) for e in hook["supporting_sentiment_edges"]}
    fact_edges = {(e["aspect"], e["sentiment"]) for e in facts["supporting_sentiment_edges"]["examples"]}
    assert fact_edges == hook_edges
    assert facts["supporting_sentiment_edges"]["total"] == len(hook["supporting_sentiment_edges"])

    # trust context passed through.
    assert facts["user_trust"] == hook["user_trust"]


# ---------------------------------------------------------------------------
# b. rendered text cites only fact aspects (no hallucination)
# ---------------------------------------------------------------------------

def test_render_cites_only_fact_aspects(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    facts = honest.extract_facts(model, u, i, top_k=3)
    text = honest.render(facts)
    assert isinstance(text, str) and len(text) > 0

    # Canonical aspect vocabulary = the model's full aspect map.
    vocab = set(gt.id_maps.aspect2idx)
    fact_aspects = {a["aspect"] for a in facts["aspects"]}

    # Any aspect from the vocabulary appearing in the text must be a fact aspect.
    lower = text.lower()
    for aspect in vocab:
        if _word_in_text(aspect.lower(), lower):
            assert aspect in fact_aspects, (
                f"rendered text mentions non-fact aspect {aspect!r}: {text!r}"
            )


def _word_in_text(needle: str, haystack: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


# ---------------------------------------------------------------------------
# c. determinism
# ---------------------------------------------------------------------------

def test_determinism(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    out1 = honest.explain(model, u, i)
    out2 = honest.explain(model, u, i)
    assert out1["text"] == out2["text"]
    assert out1["facts"] == out2["facts"]


# ---------------------------------------------------------------------------
# d. rephrase_fn injection (no LLM anywhere)
# ---------------------------------------------------------------------------

def test_rephrase_fn_injection(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    seen = {}

    def stub_rephrase(block: str) -> str:
        seen["block"] = block
        return "REPHRASED::" + block

    out = honest.explain(model, u, i, rephrase_fn=stub_rephrase)

    # The stub received a facts block matching the prompt contract.
    block = seen["block"]
    assert block.startswith("Facts:")
    assert block.rstrip().endswith("Explanation:")
    # The block must actually contain the fact aspects (traceable content).
    facts = honest.extract_facts(model, u, i)
    for a in facts["aspects"]:
        assert a["aspect"] in block

    # The stub's return became the text.
    assert out["text"] == "REPHRASED::" + block


# ---------------------------------------------------------------------------
# e. ungrounded quarantine: AST scan of the recommender path
# ---------------------------------------------------------------------------

QUARANTINED_DIRS = ["src/models", "src/eval", "src/graph", "src/data", "src/sentiment"]


def test_ungrounded_is_quarantined():
    offenders = []
    for rel in QUARANTINED_DIRS:
        d = REPO_ROOT / rel
        if not d.exists():
            continue
        for py in d.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    mods = [base] + [f"{base}.{n.name}" for n in node.names]
                for m in mods:
                    if "src.explain.ungrounded" in m or m.endswith("explain.ungrounded"):
                        offenders.append(str(py))
    assert not offenders, f"ungrounded imported by recommender modules: {offenders}"


# ---------------------------------------------------------------------------
# f. ungrounded text makes no graph-derived aspect claim
# ---------------------------------------------------------------------------

def test_ungrounded_no_graph_aspects(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    text = ungrounded.explain(i, seed=7)
    assert isinstance(text, str) and len(text) > 0

    # It may mention the item id, but no canonical aspect from the graph.
    lower = text.lower()
    for aspect in gt.id_maps.aspect2idx:
        assert not _word_in_text(aspect.lower(), lower), (
            f"ungrounded text leaked a graph aspect {aspect!r}: {text!r}"
        )

    # Determinism of the stimulus.
    assert ungrounded.explain(i, seed=7) == text
    # Different seed -> (typically) different template; at minimum still graph-free.
    other = ungrounded.explain(i, seed=99)
    for aspect in gt.id_maps.aspect2idx:
        assert not _word_in_text(aspect.lower(), other.lower())


# ---------------------------------------------------------------------------
# g. faithfulness: cited-reason removal is sensitive; score 1.0
# ---------------------------------------------------------------------------

def test_faithfulness_cited_reason_sensitive(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    facts = honest.extract_facts(model, u, i, top_k=3)
    # Only meaningful when the item actually has cited aspects.
    assert facts["aspects"], "toy item unexpectedly has no aspects"

    report = faithfulness.faithfulness_score(model, gt_args, facts)

    # Every cited reason must be sensitive on this connected toy graph.
    for pr in report["per_reason"]:
        assert pr["sensitive"], f"cited reason {pr['aspect']} was not sensitive: {pr}"
        # Contribution collapses toward absence (0.0) when evidence removed.
        assert pr["perturbed_contribution"] == 0.0 or (
            pr["perturbed_contribution"] != pr["base_contribution"]
        )
        if pr["delta_score"] is not None:
            assert pr["delta_score"] > 1e-9

    assert report["score"] == 1.0


def test_faithfulness_uncited_not_required(toy):
    """Removing an UNcited aspect's evidence may change the score but need not.

    We only assert the metric ignores uncited aspects: a facts set trimmed to a
    single cited reason still scores by that reason alone.
    """
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    facts = honest.extract_facts(model, u, i, top_k=3)
    trimmed = dict(facts)
    trimmed["aspects"] = facts["aspects"][:1]  # one cited reason only

    report = faithfulness.faithfulness_score(model, gt_args, trimmed)
    assert report["n_reasons"] == 1
    assert report["score"] == 1.0  # the single cited reason is load-bearing


# ---------------------------------------------------------------------------
# h. grounding_score: honest facts -> 1.0; fabricated -> < 1.0
# ---------------------------------------------------------------------------

def test_grounding_score(toy):
    config, dataset, td, vd, ted, model, gt, splits, gt_args = toy
    tr = splits[0]
    u, i = _pick_pair_with_aspects(model, gt, tr)

    facts = honest.extract_facts(model, u, i, top_k=3)
    real_aspects = [a["aspect"] for a in facts["aspects"]]

    # Honest: the real cited aspects are all grounded.
    g_honest = faithfulness.grounding_score(real_aspects, facts)
    assert g_honest["grounding"] == 1.0

    # Fabricated: inject an aspect that isn't real evidence for this pair.
    fabricated = real_aspects + ["teleportation"]
    g_fab = faithfulness.grounding_score(fabricated, facts)
    assert g_fab["grounding"] < 1.0
    assert "teleportation" in g_fab["ungrounded"]

    # Ungrounded text cites no graph aspect -> grounding 0.0.
    blurb = ungrounded.explain(i, seed=3)
    g_ung = faithfulness.grounding_score(blurb, facts)
    assert g_ung["grounding"] == 0.0
