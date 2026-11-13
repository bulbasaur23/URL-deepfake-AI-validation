"""Task 13: aggregator tests — pure functions over plain dicts, fully offline.

No DB, no engines, no network: ``aggregate_verdict()`` and the band helper are
exercised directly over hand-built per-item dicts
``{kind, status, verdict, confidence}``, so nothing here needs the job
manager, the models, or a database file. The brief pins the eight tests below;
extra assertions inside them pin the controller's judgment calls:

- headline singular/plural beyond the pinned example ("1 text block",
  "1 item", "3 images"), the empty breakdown (no "()" when nothing is
  synthetic), and the always-present trailing "· {human} of {total} human ·
  {unclear} unclear" segment at zero;
- the boundary helper is settings-driven (one monkeypatch proves 35/65 are
  not hardcoded);
- per-modality tokens are ``synthetic|mixed|authentic|insufficient`` while
  the page vocabulary is ``synthetic_detected|mixed|authentic|
  insufficient_data`` — never the page tokens below the page level.

The headline's ``unclear`` count is the number of verdict=="unclear" items;
items that never ran an engine (verdict None) abstain from the vote and are
surfaced by warnings/item cards, not recounted as unclear here.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.settings import Settings
from pipeline.aggregate import AggregatedReport, aggregate_verdict, verdict_from_score

_SETTINGS = Settings()

PAGE_SYNTHETIC = "synthetic_detected"
PAGE_MIXED = "mixed"
PAGE_AUTHENTIC = "authentic"
PAGE_INSUFFICIENT = "insufficient_data"
MOD_INSUFFICIENT = "insufficient"


def _item(
    kind: str, verdict: str | None, confidence: float | None = None, status: str = "ok"
) -> dict[str, Any]:
    """One per-item dict exactly as the Task 14 seam will feed the aggregator."""
    return {"kind": kind, "status": status, "verdict": verdict, "confidence": confidence}


def test_counts_headline_synthetic_image_human_page() -> None:
    """1 synthetic image + 3 authentic text → the pinned counts headline."""
    items = [
        _item("text", "authentic", 0.91),
        _item("text", "authentic", 0.95),
        _item("text", "authentic", 0.93),
        _item("image", "synthetic", 0.80),
    ]
    report = aggregate_verdict(items, _SETTINGS)
    assert isinstance(report, AggregatedReport)
    # §11.13: synthetic present AND authentic present → mixed, regardless of
    # which side dominates (the controller's test 6 pins "both orderings /
    # mixed dominant counts"); "synthetic_detected" needs ONLY synthetic.
    # The counts headline carries the nuance for this mostly-human page.
    assert report.verdict == PAGE_MIXED
    assert report.headline == "1 of 4 items AI-generated (1 image) · 3 of 4 human · 0 unclear"
    # Determining evidence on a synthetic page: max confidence among synthetic items.
    assert report.confidence == pytest.approx(0.80)
    assert report.per_modality == {
        "text": {"verdict": PAGE_AUTHENTIC, "counts": {"synthetic": 0, "authentic": 3}},
        "images": {"verdict": "synthetic", "counts": {"synthetic": 1, "authentic": 0}},
    }
    assert report.warnings == []


def test_no_items_insufficient_data_never_authentic() -> None:
    """Empty page → insufficient_data (never authentic), None confidence, zero headline."""
    report = aggregate_verdict([], _SETTINGS)
    assert report.verdict == PAGE_INSUFFICIENT
    assert report.confidence is None
    assert report.headline == "0 of 0 items AI-generated · 0 of 0 human · 0 unclear"
    assert report.per_modality == {
        "text": {"verdict": MOD_INSUFFICIENT, "counts": {"synthetic": 0, "authentic": 0}},
        "images": {"verdict": MOD_INSUFFICIENT, "counts": {"synthetic": 0, "authentic": 0}},
    }
    assert report.warnings == []


def test_band_boundaries_35_65(monkeypatch: pytest.MonkeyPatch) -> None:
    """The band helper is pinned at the exact 35/65 boundaries from settings."""
    assert verdict_from_score(34.9, _SETTINGS) == "authentic"
    assert verdict_from_score(35.0, _SETTINGS) == "unclear"
    assert verdict_from_score(64.9, _SETTINGS) == "unclear"
    assert verdict_from_score(65.0, _SETTINGS) == "synthetic"
    # The boundaries come from settings, not hardcoded constants: move the
    # thresholds and the same scores re-band accordingly.
    monkeypatch.setattr(_SETTINGS, "band_unclear_min", 10)
    monkeypatch.setattr(_SETTINGS, "band_synthetic_min", 90)
    assert verdict_from_score(34.9, _SETTINGS) == "unclear"
    assert verdict_from_score(65.0, _SETTINGS) == "unclear"
    assert verdict_from_score(89.9, _SETTINGS) == "unclear"
    assert verdict_from_score(90.0, _SETTINGS) == "synthetic"


def test_confidence_max_on_synthetic_min_on_authentic() -> None:
    """Max confidence among synthetic items on synthetic/mixed; min among analyzed on authentic."""
    # synthetic page: max among the two synthetic items, unclear never decides.
    report = aggregate_verdict(
        [
            _item("text", "synthetic", 0.70),
            _item("text", "synthetic", 0.95),
            _item("image", "unclear", 0.50),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_SYNTHETIC
    assert report.confidence == pytest.approx(0.95)

    # mixed page: still the max among synthetic items only (0.60, not 0.99).
    report = aggregate_verdict(
        [
            _item("text", "synthetic", 0.60),
            _item("image", "authentic", 0.99),
            _item("image", "authentic", 0.98),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_MIXED
    assert report.confidence == pytest.approx(0.60)

    # authentic page: min among analyzed items — unclear analyzed items and
    # None confidences participate per the None-filtering rule (0.85 wins).
    report = aggregate_verdict(
        [
            _item("text", "authentic", 0.90),
            _item("text", "authentic", 0.85),
            _item("image", "unclear", 0.70),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_AUTHENTIC
    assert report.confidence == pytest.approx(0.70)


def test_all_unclear_page_insufficient_data_max_confidence() -> None:
    """Analyzed but all unclear → insufficient_data with max analyzed confidence (not None)."""
    report = aggregate_verdict(
        [
            _item("text", "unclear", 0.40),
            _item("image", "unclear", 0.60),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_INSUFFICIENT
    assert report.confidence == pytest.approx(0.60)
    # Nothing synthetic → the breakdown is empty: no "()" is emitted, and the
    # trailing human/unclear segment is always present.
    assert report.headline == "0 of 2 items AI-generated · 0 of 2 human · 2 unclear"
    assert report.per_modality == {
        "text": {"verdict": MOD_INSUFFICIENT, "counts": {"synthetic": 0, "authentic": 0}},
        "images": {"verdict": MOD_INSUFFICIENT, "counts": {"synthetic": 0, "authentic": 0}},
    }


def test_mixed_page_both_directions() -> None:
    """synthetic+authentic present in either direction/dominance → mixed."""
    # Synthetic first, dominant authentic.
    report = aggregate_verdict(
        [_item("image", "synthetic", 0.80), _item("text", "authentic", 0.99)],
        _SETTINGS,
    )
    assert report.verdict == PAGE_MIXED
    # Authentic first, dominant synthetic.
    report = aggregate_verdict(
        [
            _item("text", "authentic", 0.91),
            _item("text", "authentic", 0.92),
            _item("image", "synthetic", 0.88),
            _item("image", "synthetic", 0.89),
            _item("image", "synthetic", 0.90),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_MIXED
    assert report.headline == "3 of 5 items AI-generated (3 images) · 2 of 5 human · 0 unclear"
    assert report.confidence == pytest.approx(0.90)
    # Both synthetic kinds present: text is listed before images (design §7.4).
    report = aggregate_verdict(
        [
            _item("text", "synthetic", 0.75),
            _item("image", "synthetic", 0.85),
            _item("text", "authentic", 0.95),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_MIXED
    assert (
        report.headline
        == "2 of 3 items AI-generated (1 text block, 1 image) · 1 of 3 human · 0 unclear"
    )


def test_synthetic_decides_despite_unclear() -> None:
    """1 synthetic + 3 unclear → synthetic_detected: unclear items abstain."""
    report = aggregate_verdict(
        [
            _item("text", "synthetic", 0.66),
            _item("text", "unclear", 0.30),
            _item("text", "unclear", 0.40),
            _item("image", "unclear", 0.50),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_SYNTHETIC
    assert report.confidence == pytest.approx(0.66)
    # Singular breakdown ("1 text block"), the zero-count image entry omitted,
    # unclear counted (3), trailing segment present at zero human.
    assert report.headline == "1 of 4 items AI-generated (1 text block) · 0 of 4 human · 3 unclear"
    assert report.per_modality["text"]["verdict"] == "synthetic"


def test_per_modality_split_independent() -> None:
    """One modality's outcome never leaks into the other's per_modality entry."""
    # Text never analyzed (verdict None) → text insufficient while images vote.
    report = aggregate_verdict(
        [
            _item("text", None, None, status="unsupported"),
            _item("image", "synthetic", 0.82),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_SYNTHETIC
    assert report.per_modality["text"] == {
        "verdict": MOD_INSUFFICIENT,
        "counts": {"synthetic": 0, "authentic": 0},
    }
    assert report.per_modality["images"] == {
        "verdict": "synthetic",
        "counts": {"synthetic": 1, "authentic": 0},
    }

    # Mirror: images never analyzed, text authentic → page authentic, images insufficient.
    report = aggregate_verdict(
        [
            _item("text", "authentic", 0.88),
            _item("image", None, None, status="unavailable"),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_AUTHENTIC
    assert report.confidence == pytest.approx(0.88)
    assert report.per_modality["images"] == {
        "verdict": MOD_INSUFFICIENT,
        "counts": {"synthetic": 0, "authentic": 0},
    }
    assert report.per_modality["text"]["verdict"] == PAGE_AUTHENTIC

    # Stronger no-leak case: all-unclear text must NOT inherit the page's
    # synthetic_detected — the modality abstains as "insufficient".
    report = aggregate_verdict(
        [
            _item("text", "unclear", 0.45),
            _item("image", "synthetic", 0.91),
        ],
        _SETTINGS,
    )
    assert report.verdict == PAGE_SYNTHETIC
    assert report.per_modality["text"]["verdict"] == MOD_INSUFFICIENT
    assert report.per_modality["images"]["verdict"] == "synthetic"
