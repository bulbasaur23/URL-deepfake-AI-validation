"""§7.6 payload shaping (Task 14): persisted rows → the scan report JSON.

``serialize_scan_report`` is pure: it takes the loaded Scan row, its id-ordered
ScanItem rows, that scan's EngineResultORM rows grouped by item id, the set of
item ids holding a Snapshot row, the manager's warning tokens, and Settings —
and returns the exact §7.6 JSON dict. No DB access, no HTTP, no app state.

Design authority (verified against the design doc):

- The report is RECOMPUTED from the persisted per-item verdicts via
  ``aggregate_verdict`` (no headline column exists — recompute is the design;
  the seam stored ``ScanItem.verdict``/``confidence`` and the page-level
  ``Scan.verdict``/``confidence``). ``summary`` is the counts headline,
  ``verdict``/``confidence``/``per_modality`` come from the recomputed report.
- §7.6's per-modality counts include an ``unclear`` key that the Task 13
  aggregator deliberately omits (its per-modality counts are
  ``synthetic|authentic``); the API layer derives ``unclear`` from the
  persisted item verdicts here (count of ``kind``-subset items with verdict
  ``unclear`` — the band token, §3.4).
- ``excerpt`` is the best available PERSISTED text. The pipeline does not
  persist item text by design (``ScanItem`` has no text column — full text is
  in-memory only, dropped after the scan): the persisted free text on a row is
  its ``label`` (extractor labels: "Paragraph N"/"List item N" for text
  blocks, the ≤ 60-char alt text for images). So ``excerpt = label``.
- ``thumb`` is forward-compat with Task 15's media route:
  ``/media/{item_id}`` exactly when a Snapshot row exists for the item, else
  None (§7.6 marks it optional; a stable null key keeps client code simple).
- ``scanned_at`` is ``Scan.created_at`` in ISO-8601 with an explicit ``Z``:
  SQLite stores naive UTC clocks (``db.models._utcnow``), and the ``Z`` tells
  consumers the offset instead of letting them guess local time.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from datetime import datetime
from typing import Any

from app.settings import Settings
from db.models import EngineResultORM, Scan, ScanItem
from pipeline.aggregate import aggregate_verdict

# The per-item band token that abstains from the page vote (§3.4 bands,
# §11.13 voting). Literal here because aggregate.py keeps it private.
_UNCLEAR = "unclear"


def _iso_utc(value: datetime) -> str:
    """ISO-8601 with an explicit UTC marker (SQLite stores naive UTC)."""
    text = value.isoformat()
    return text if value.tzinfo is not None else text + "Z"


def serialize_scan_report(
    scan: Scan,
    items: Sequence[ScanItem],
    results_by_item: Mapping[int, Sequence[EngineResultORM]],
    snapshot_item_ids: Container[int],
    warnings: Sequence[str],
    settings: Settings,
) -> dict[str, Any]:
    """Build the exact §7.6 payload for one scan from persisted rows.

    ``items`` must be in ScanItem.id order (the response's item order);
    ``results_by_item`` maps item id → that item's EngineResultORM rows in id
    order; ``warnings`` are the manager's tokens (``[]`` after a restart).
    """
    report = aggregate_verdict(
        [
            {
                "kind": item.kind,
                "status": item.status,
                "verdict": item.verdict,
                "confidence": item.confidence,
            }
            for item in items
        ],
        settings,
    )

    # §7.6 wants counts{synthetic, unclear, authentic} per modality; add the
    # unclear derivation using the aggregator's own subset convention
    # (text = kind "text", images = everything else).
    subsets: dict[str, list[ScanItem]] = {
        "text": [item for item in items if item.kind == "text"],
        "images": [item for item in items if item.kind != "text"],
    }
    for key, subset in subsets.items():
        counts = report.per_modality[key]["counts"]
        report.per_modality[key]["counts"] = {
            "synthetic": counts["synthetic"],
            "unclear": sum(1 for item in subset if item.verdict == _UNCLEAR),
            "authentic": counts["authentic"],
        }

    item_payloads = [
        {
            "kind": item.kind,
            "label": item.label,
            # Best available persisted text (see module docstring): the row's
            # label. Full block text is never persisted by design.
            "excerpt": item.label,
            # Forward-compat thumb for Task 15's GET /media/{item_id}.
            "thumb": f"/media/{item.id}" if item.id in snapshot_item_ids else None,
            "status": item.status,
            "engines": [
                {
                    "id": result.engine_id,
                    "score": result.score,
                    "confidence": result.confidence,
                    "rationale": result.rationale,
                }
                for result in results_by_item.get(item.id, ())
            ],
        }
        for item in items
    ]
    engines_used = sorted(
        {result.engine_id for results in results_by_item.values() for result in results}
    )

    return {
        "status": scan.status,
        "summary": report.headline,
        "confidence": report.confidence,
        "verdict": report.verdict,
        "per_modality": report.per_modality,
        "meta": {
            "title": scan.title,
            "domain": scan.domain,
            "scanned_at": _iso_utc(scan.created_at),
        },
        "items": item_payloads,
        "warnings": list(warnings),
        "engines_used": engines_used,
    }
