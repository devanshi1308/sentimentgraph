"""Remove-a-reason faithfulness check (PRD step 9, T5.3; STHGREC_DESIGN.md §7).

METRIC DEFINITION (faithfulness, verbatim). For an honest explanation of a
``(user, item)`` recommendation, each cited aspect is a *reason*. We test whether
each reason is load-bearing by deleting exactly its supporting evidence — the
``absa_normalized`` sentiment rows for ``(item, aspect)`` — rebuilding the graph
tensors, and re-propagating the model's *already-trained* embeddings through the
perturbed adjacency (embeddings are held fixed; only graph structure changes, so
the delta isolates that reason's structural contribution). Per-reason sensitivity
is ``|Δscore| = |score_perturbed − score_original|`` together with the collapse of
that aspect's signed ``has_aspect`` contribution toward zero/absence. The
explanation-level ``faithfulness_score`` is the fraction of cited reasons whose
removal produces a change greater than ``eps`` (default 1e-9); on a connected toy
graph an honest explanation scores 1.0. For the ungrounded side — which cites
nothing graph-real — ``grounding_score`` is the fraction of the text's cited
aspects that exist as real supporting evidence for that ``(user, item)``; an
ungrounded blurb (no graph aspects) is definitionally 0.

All public functions return plain-python report dicts. No LLM anywhere.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

from src.models.graph_tensors import build_graph_tensors


# ---------------------------------------------------------------------------
# Remove-a-reason: rebuild graph tensors with one aspect's evidence deleted
# ---------------------------------------------------------------------------

def remove_reason(gt_build_args: dict, item_raw_id, reason_aspect):
    """Rebuild :class:`GraphTensors` with the ``(item, reason_aspect)`` evidence gone.

    ``gt_build_args`` is the kwargs dict originally passed to
    :func:`build_graph_tensors` (``inter_csv``, ``sentiment_csv``, ``trust_csv``,
    the ablation flags, and — crucially — ``id_map_json`` so the node index space
    stays identical to the original build). We read the sentiment CSV, drop every
    row where ``item_id == item_raw_id and aspect == reason_aspect``, write a
    filtered temp CSV, and rebuild. Operating at the CSV/dataframe level is the
    cleanest hook given ``build_graph_tensors`` reads CSVs, and the shared
    ``id_map_json`` guarantees the adjacency shapes line up for a fair re-score.

    Returns the perturbed ``GraphTensors``. The temp CSV is cleaned up before
    return (its contents are already baked into the returned tensors).
    """
    args = dict(gt_build_args)
    sentiment_csv = args.get("sentiment_csv")
    if sentiment_csv is None:
        # Nothing to remove — sentiment is off / absent. Return an unperturbed build.
        return build_graph_tensors(**args)

    df = pd.read_csv(sentiment_csv)
    keep = ~(
        (df["item_id"].astype(str) == str(item_raw_id))
        & (df["aspect"].astype(str) == str(reason_aspect))
    )
    filtered = df[keep].reset_index(drop=True)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="rmreason_"
    )
    try:
        filtered.to_csv(tmp.name, index=False)
        tmp.close()
        args["sentiment_csv"] = tmp.name
        # id_map_json is preserved from gt_build_args so the index space is stable.
        gt = build_graph_tensors(**args)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return gt


# ---------------------------------------------------------------------------
# Re-score the model's fixed embeddings through a (possibly perturbed) adjacency
# ---------------------------------------------------------------------------

@torch.no_grad()
def _propagate(model, gt) -> torch.Tensor:
    """Propagate ``model``'s trained ego embeddings through ``gt``'s adjacencies.

    Replicates :meth:`STHGRec.forward` exactly but with the adjacencies taken from
    ``gt`` (which may be perturbed), so the *only* thing that differs between the
    original and perturbed scores is graph structure — the embeddings are held
    fixed. Returns the final per-node embedding table (no PAD row appended).
    """
    all_emb = model.get_ego_embeddings()
    emb_list = [all_emb]

    adj_int = gt.adj_interacts.coalesce().to(all_emb.dtype)
    adj_sent = gt.adj_sentiment.coalesce().to(all_emb.dtype) if gt.adj_sentiment is not None else None
    adj_asp = gt.adj_has_aspect.coalesce().to(all_emb.dtype) if gt.adj_has_aspect is not None else None

    for _ in range(model.n_layers):
        agg = torch.sparse.mm(adj_int, all_emb)
        if adj_sent is not None:
            agg = agg + model.sentiment_weight * torch.sparse.mm(adj_sent, all_emb)
        if adj_asp is not None:
            agg = agg + model.aspect_weight * torch.sparse.mm(adj_asp, all_emb)
        all_emb = agg
        emb_list.append(all_emb)

    return torch.stack(emb_list, dim=1).mean(dim=1)


@torch.no_grad()
def _score_pair(model, gt, user_raw_id, item_raw_id) -> Optional[float]:
    """dot-product score for ``(user, item)`` using fixed embeddings + ``gt`` adj."""
    u_node = model._node_of_user(user_raw_id)
    i_node = model._node_of_item(item_raw_id)
    if u_node is None or i_node is None:
        return None
    final = _propagate(model, gt)
    return float(torch.dot(final[u_node], final[i_node]).item())


def _aspect_contribution(gt, item_raw_id, aspect) -> float:
    """The signed ``has_aspect`` contribution for ``(item, aspect)`` in ``gt`` (0 if gone)."""
    return float(gt.has_aspect_weights.get((str(item_raw_id), str(aspect)), 0.0))


# ---------------------------------------------------------------------------
# Explanation-level faithfulness score (honest side)
# ---------------------------------------------------------------------------

def faithfulness_score(
    model,
    gt_build_args: dict,
    facts: dict,
    eps: float = 1e-9,
) -> dict:
    """Remove-a-reason faithfulness report for an honest explanation.

    For each cited aspect in ``facts["aspects"]``: delete its supporting evidence
    (:func:`remove_reason`), re-propagate the model's fixed embeddings through the
    perturbed adjacency, and record ``|Δscore|`` and the contribution collapse.
    ``score`` = fraction of cited reasons whose removal moves the pair score by
    more than ``eps``.

    Returns::

        {
          "user_id": str, "item_id": str,
          "base_score": float | None,
          "per_reason": [
            {"aspect": str, "base_contribution": float,
             "perturbed_contribution": float, "contribution_collapsed": bool,
             "base_score": float|None, "perturbed_score": float|None,
             "delta_score": float|None, "sensitive": bool},
            ...
          ],
          "score": float,               # explanation-level faithfulness in [0,1]
          "n_reasons": int,
        }
    """
    user_raw_id = facts["user_id"]
    item_raw_id = facts["item_id"]

    base_gt = build_graph_tensors(**gt_build_args)
    base_score = _score_pair(model, base_gt, user_raw_id, item_raw_id)

    per_reason = []
    sensitive_count = 0
    reasons = facts.get("aspects", [])

    for a in reasons:
        aspect = a["aspect"]
        base_contrib = _aspect_contribution(base_gt, item_raw_id, aspect)

        pert_gt = remove_reason(gt_build_args, item_raw_id, aspect)
        pert_contrib = _aspect_contribution(pert_gt, item_raw_id, aspect)
        pert_score = _score_pair(model, pert_gt, user_raw_id, item_raw_id)

        if base_score is None or pert_score is None:
            delta = None
            sensitive = base_contrib != pert_contrib
        else:
            delta = abs(pert_score - base_score)
            sensitive = delta > eps or abs(base_contrib - pert_contrib) > eps

        if sensitive:
            sensitive_count += 1

        per_reason.append(
            {
                "aspect": str(aspect),
                "base_contribution": base_contrib,
                "perturbed_contribution": pert_contrib,
                "contribution_collapsed": abs(base_contrib - pert_contrib) > eps,
                "base_score": base_score,
                "perturbed_score": pert_score,
                "delta_score": delta,
                "sensitive": bool(sensitive),
            }
        )

    n = len(reasons)
    return {
        "user_id": str(user_raw_id),
        "item_id": str(item_raw_id),
        "base_score": base_score,
        "per_reason": per_reason,
        "score": (sensitive_count / n) if n else 0.0,
        "n_reasons": n,
    }


# ---------------------------------------------------------------------------
# Grounding score (ungrounded side)
# ---------------------------------------------------------------------------

def grounding_score(cited_aspects, facts: dict) -> dict:
    """Fraction of ``cited_aspects`` that are REAL supporting evidence in ``facts``.

    ``cited_aspects`` is either a list of aspect names or a free-text string (we
    then match the canonical fact aspects appearing as whole words in the text).
    The "real" set is the aspects the model actually cited for this ``(user,
    item)`` — ``facts["aspects"]``. Honest facts fed back in score 1.0; a fabricated
    aspect list scores < 1.0; ungrounded text (no graph aspects) scores 0.0.

    Returns ``{"grounding": float, "cited": [...], "grounded": [...],
    "ungrounded": [...], "n_cited": int}``.
    """
    real = {a["aspect"] for a in facts.get("aspects", [])}

    if isinstance(cited_aspects, str):
        text = cited_aspects.lower()
        cited = [a for a in sorted(real) if _word_in(a.lower(), text)]
        # An ungrounded blurb mentions none of the real aspects -> cited == [].
    else:
        cited = [str(a) for a in cited_aspects]

    if not cited:
        return {
            "grounding": 0.0,
            "cited": [],
            "grounded": [],
            "ungrounded": [],
            "n_cited": 0,
        }

    grounded = [c for c in cited if c in real]
    ungrounded = [c for c in cited if c not in real]
    return {
        "grounding": len(grounded) / len(cited),
        "cited": cited,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "n_cited": len(cited),
    }


def _word_in(needle: str, haystack: str) -> bool:
    """Whole-word-ish membership test (avoids 'price' matching 'priceless')."""
    import re

    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None
