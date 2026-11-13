"""Aggregator (Task 13): per-item bands → page report, no DB/engines/network.

``aggregate_verdict(items, settings)`` folds a list of per-item dicts
``{kind, status, verdict, confidence}`` into an ``AggregatedReport``:
page ``verdict``, page ``confidence``, the counts ``headline``, and the
``per_modality`` split. Pure functions over plain data — the Task 14 seam in
``jobs/manager.py`` reads each Item's engine results, assigns per-item
verdicts with ``verdict_from_score``, and calls this with the resulting dicts.

Voting rules (§11.13 / design §7.4, decided with the human partner):

- "Analyzed" means the item dict's ``verdict`` is not None. Items that ended
  unsupported/degraded/unavailable/fetch_failed/blocked_by_site/too_large
  carry verdict None, do not vote, and do not count as analyzed.
- The page classification is taken over ``synthetic`` vs ``authentic``
  verdict classes only: ``unclear`` items ABSTAIN from the page vote.
  Zero analyzed items → ``insufficient_data``; synthetic AND authentic both
  present → ``mixed``; only synthetic → ``synthetic_detected``; only
  authentic → ``authentic``; neither → ``insufficient_data``. A page where
  every analyzed item is unclear lands on ``insufficient_data`` — never
  ``authentic`` for anything that was not positively judged human.
- Confidence is the confidence of the determining evidence (§7.4):
  the max item confidence among synthetic (≥ band) items for
  ``synthetic_detected``/``mixed`` pages; the min item confidence among
  analyzed items (unclear included) for ``authentic`` pages; for all-unclear
  pages the max item confidence among analyzed items; None when nothing was
  analyzed at all. None confidences never win a max/min — an analyzed item
  with no confidence is simply skipped in the reduction, so a page of
  analyzed items without any confidence still reports None.
- Per-modality verdicts use the same rule per ``kind`` subset with the
  modality vocabulary ``synthetic|mixed|authentic|insufficient``: the page
  tokens ``synthetic_detected``/``insufficient_data`` exist at page level
  only, and ``insufficient`` is the modality's only abstain token (used both
  when nothing was analyzed for that modality and when all of its analyzed
  items were unclear).
- The headline is counts, never a single blended label (§7.4):
  ``"{syn} of {total} items AI-generated ({syn} text blocks, {img} images) ·
  {human} of {total} human · {unclear} unclear"`` — the breakdown lists only
  counts > 0 (text before images, singular when 1; omitted entirely, with no
  empty "()", when nothing is synthetic), and the trailing human/unclear
  segment is always present, ``· 0 unclear`` included at zero. ``total`` is
  every item handed over, ``human`` counts ``authentic`` verdicts, and
  ``unclear`` counts verdict-class ``unclear`` items; items with verdict None
  abstain from those counts too and surface through warnings/item cards.

P3 (out of scope v1): v1 assigns one per-item verdict from the winning
engine's score via ``verdict_from_score``; a future multi-engine vote per
item could replace that single-engine mapping without changing this module's
report shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Page-level classification vocabulary (§11.13) — derived, DB-filter tokens.
PAGE_SYNTHETIC = "synthetic_detected"
PAGE_MIXED = "mixed"
PAGE_AUTHENTIC = "authentic"
PAGE_INSUFFICIENT = "insufficient_data"

# Per-modality vocabulary (§11.13): note "synthetic" (not the page's
# "synthetic_detected") and "insufficient" (not "insufficient_data").
MOD_SYNTHETIC = "synthetic"
MOD_MIXED = "mixed"
MOD_AUTHENTIC = "authentic"
MOD_INSUFFICIENT = "insufficient"

_MOD_TO_PAGE = {
    MOD_SYNTHETIC: PAGE_SYNTHETIC,
    MOD_MIXED: PAGE_MIXED,
    MOD_AUTHENTIC: PAGE_AUTHENTIC,
    MOD_INSUFFICIENT: PAGE_INSUFFICIENT,
}

_VERDICT_SYNTHETIC = "synthetic"
_VERDICT_AUTHENTIC = "authentic"
_VERDICT_UNCLEAR = "unclear"


@dataclass
class AggregatedReport:
    """The aggregation product (§7.4): page classification + counts + evidence."""

    verdict: str
    confidence: float | None
    headline: str
    per_modality: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


def verdict_from_score(score: float, settings: Any) -> str:
    """Band one engine score into a per-item verdict (§7.4 / §3.4).

    ``score`` is the 0..100 synthetic-likelihood; the boundaries come from
    settings (``band_unclear_min``/``band_synthetic_min``, spec 35/65) so the
    thresholds are configuration, never hardcoded: ``score <
    band_unclear_min`` → ``authentic``, ``score >= band_synthetic_min`` →
    ``synthetic``, in between → ``unclear``.
    """
    if score >= settings.band_synthetic_min:
        return _VERDICT_SYNTHETIC
    if score >= settings.band_unclear_min:
        return _VERDICT_UNCLEAR
    return _VERDICT_AUTHENTIC


def _vote(analyzed: list[dict[str, Any]]) -> str:
    """Classify one analyzed set over synthetic vs authentic only (§11.13).

    Unclear items abstain. Returns a modality-vocabulary token; the page
    level maps it through ``_MOD_TO_PAGE``.
    """
    synthetic = any(i["verdict"] == _VERDICT_SYNTHETIC for i in analyzed)
    authentic = any(i["verdict"] == _VERDICT_AUTHENTIC for i in analyzed)
    if synthetic and authentic:
        return MOD_MIXED
    if synthetic:
        return MOD_SYNTHETIC
    if authentic:
        return MOD_AUTHENTIC
    return MOD_INSUFFICIENT


def _confidence_of(analyzed: list[dict[str, Any]], vote: str) -> float | None:
    """Determining-evidence confidence (§7.4): max synthetic / min analyzed.

    None confidences never win a reduction; a set without any confidence
    reports None even when items were analyzed.
    """
    if vote in (MOD_SYNTHETIC, MOD_MIXED):
        pool = [
            i["confidence"]
            for i in analyzed
            if i["verdict"] == _VERDICT_SYNTHETIC and i["confidence"] is not None
        ]
        return max(pool) if pool else None
    pool = [i["confidence"] for i in analyzed if i["confidence"] is not None]
    if vote == MOD_AUTHENTIC:
        return min(pool) if pool else None
    # MOD_INSUFFICIENT with analyzed items = all-unclear page: max analyzed.
    return max(pool) if pool else None


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _headline(items: list[dict[str, Any]]) -> str:
    """Counts headline (§7.4): synthetic breakdown only > 0, human/unclear always."""
    synthetic = [i for i in items if i["verdict"] == _VERDICT_SYNTHETIC]
    syn_text = sum(1 for i in synthetic if i["kind"] == "text")
    syn_images = len(synthetic) - syn_text
    human = sum(1 for i in items if i["verdict"] == _VERDICT_AUTHENTIC)
    unclear = sum(1 for i in items if i["verdict"] == _VERDICT_UNCLEAR)
    total = len(items)

    breakdown: list[str] = []
    if syn_text:
        breakdown.append(f"{syn_text} {_plural(syn_text, 'text block', 'text blocks')}")
    if syn_images:
        breakdown.append(f"{syn_images} {_plural(syn_images, 'image', 'images')}")
    breakdown_part = f" ({', '.join(breakdown)})" if breakdown else ""

    return (
        f"{len(synthetic)} of {total} {_plural(total, 'item', 'items')} AI-generated"
        f"{breakdown_part} · {human} of {total} human · {unclear} unclear"
    )


def aggregate_verdict(items: list[dict[str, Any]], settings: Any) -> AggregatedReport:
    """Aggregate per-item dicts into the page report (§7.4 / §11.13).

    ``items`` entries are ``{kind, status, verdict, confidence}`` with
    ``kind`` in ``text|image`` and ``verdict`` a per-item band token
    (``authentic|unclear|synthetic``) or None for never-analyzed items.
    ``status`` is carried in the shape for seam convenience; aggregation
    reads only ``kind``/``verdict``/``confidence``. Returns the page verdict,
    determining-evidence confidence, counts headline, per-modality split,
    and an empty ``warnings`` list (the seam fills warnings from the job
    manager's in-memory list).
    """
    analyzed = [i for i in items if i["verdict"] is not None]
    text_analyzed = [i for i in analyzed if i["kind"] == "text"]
    image_analyzed = [i for i in analyzed if i["kind"] != "text"]

    page_vote = _vote(analyzed)
    per_modality: dict[str, dict[str, Any]] = {}
    for key, subset in (("text", text_analyzed), ("images", image_analyzed)):
        mod_vote = _vote(subset)
        per_modality[key] = {
            "verdict": mod_vote,
            "counts": {
                "synthetic": sum(1 for i in subset if i["verdict"] == _VERDICT_SYNTHETIC),
                "authentic": sum(1 for i in subset if i["verdict"] == _VERDICT_AUTHENTIC),
            },
        }

    return AggregatedReport(
        verdict=_MOD_TO_PAGE[page_vote],
        confidence=_confidence_of(analyzed, page_vote),
        headline=_headline(items),
        per_modality=per_modality,
    )
