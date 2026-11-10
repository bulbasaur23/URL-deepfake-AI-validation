"""Task 6: language gate tests — pure in-process, fully offline.

No fetching, no files, no ML models: ``detect()`` and ``mark_text_items()``
are exercised directly over hand-built ``Item`` objects, so network and GPU
availability are irrelevant here. The brief pins three tests (english ok,
hindi marked unsupported, unsupported items carried not dropped); the
supporting tests pin the controller rulings the brief left open: a None
verdict never marks (uncertain text flows to the engines rather than being
silently dropped), image items are skipped by kind, the seeded factory makes
repeated calls agree, and the plan-pinned page-warning constant is exported
verbatim for Task 12 to import.

Settings follow the extractor-test convention: plain ``Settings()`` defaults,
no ``backend/.env`` required. The gate reads nothing from settings (langdetect
has no threshold concept worth a P1 tunable); passing it keeps the plan's
signature shape so a later task could add a tunable without call-site churn.
"""

from __future__ import annotations

import pytest

from app.settings import Settings
from pipeline import langid
from pipeline.langid import NON_ENGLISH_WARNING, detect, mark_text_items
from pipeline.types import Item

_SETTINGS = Settings()

# > 80 chars, clear English — comfortably past the extractor's block gate.
_ENGLISH_TEXT = (
    "This opening paragraph describes the scan pipeline and carries enough "
    "ordinary filler text to comfortably pass the eighty character minimum "
    "in the content extractor tests"
)

# > 80 chars, clear Hindi — the canonical "English-trained engines must not
# judge this" case for the P1 detector stack.
_HINDI_TEXT = (
    "यह हिंदी पैराग्राफ एक समाचार लेख के बारे में है जो कृत्रिम बुद्धिमत्ता और "
    "मशीनी अनुवाद के विषय पर चर्चा करता है और इसमें पर्याप्त विस्तार है"
)

# No removable language features whatsoever: the detector cannot classify.
_EMOJI_TEXT = "😀 😃 😄 😁 😆 😅 🤣 😂 🙃 🥳 🎉 🎊"
_DIGIT_TEXT = "1234567890 " * 5


def _text_item(text: str | None, label: str = "Paragraph 1") -> Item:
    """A kind=text item as Task 5's extractor produces it (status ok, clean links)."""
    return Item(kind="text", label=label, source_ref="article:p[0]", text=text)


def test_english_ok() -> None:
    """English stays status="ok" and gains language="en"; nothing else changes."""
    item = _text_item(_ENGLISH_TEXT)
    assert mark_text_items([item], _SETTINGS) == 0  # nothing marked unsupported
    assert item.status == "ok"
    assert item.language == "en"
    assert item.verdict is None and item.confidence is None
    assert item.engine_results == []  # gating only; engines attach results (Task 7)
    assert detect(_ENGLISH_TEXT, _SETTINGS) == "en"


def test_hindi_marked_unsupported() -> None:
    """Non-English gets status="unsupported", language=<code>, and returns count 1."""
    item = _text_item(_HINDI_TEXT, label="Paragraph 2")
    assert mark_text_items([item], _SETTINGS) == 1  # exactly the unsupported count
    assert item.status == "unsupported"
    assert item.language == "hi"
    assert item.verdict is None and item.confidence is None
    assert item.engine_results == []  # no engine may run on unsupported text
    assert detect(_HINDI_TEXT, _SETTINGS) == "hi"


def test_unsupported_items_carried_not_dropped() -> None:
    """Unsupported items survive in the list, mutated in place, never removed.

    Task 12 owns the page-level warning; the gate's only job is marking (and
    counting) — deleting the evidence from ``text_items`` would strand that
    warning and shift every downstream index.
    """
    english = _text_item(_ENGLISH_TEXT)
    hindi = _text_item(_HINDI_TEXT, label="Paragraph 2")
    items = [english, hindi]
    marked = mark_text_items(items, _SETTINGS)
    assert len(items) == 2  # nothing removed, nothing appended
    assert items[0] is english and items[1] is hindi  # same objects, in order
    assert hindi.status == "unsupported" and hindi.language == "hi"
    assert hindi.engine_results == []  # "survive into the report with no engine results"
    assert english.status == "ok" and english.language == "en"
    assert marked == 1


# --- ruling: detect None means "cannot classify" — unmarked, never removed -------


def test_unclassifiable_text_yields_none() -> None:
    """Empty/whitespace and feature-less strings (emoji/digits) detect as None."""
    for text in ("", "   \t\n  ", _EMOJI_TEXT, _DIGIT_TEXT):
        assert detect(text, _SETTINGS) is None, repr(text[:12])


def test_unclassifiable_item_stays_ok_and_unmarked() -> None:
    """Detection failure is not proof of non-English: None never marks an item."""
    item = _text_item(_EMOJI_TEXT)
    assert mark_text_items([item], _SETTINGS) == 0
    assert item.status == "ok"
    assert item.language is None  # stamped with nothing — no guess, no drop
    assert item.engine_results == []


def test_blank_text_item_stays_ok_and_unmarked() -> None:
    """A whitespace-only block (defensive beyond the extractor's caps) is left alone."""
    item = _text_item("   \t\n  ")
    assert mark_text_items([item], _SETTINGS) == 0
    assert item.status == "ok"
    assert item.language is None


def test_unknown_token_maps_to_uncertain_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """langdetect's literal "unknown" token is NOT an ISO-639-1 code — map to None.

    When no candidate clears langdetect's probability threshold, its Detector
    returns the sentinel ``UNKNOWN_LANG == "unknown"`` instead of raising.
    Passing that token through would mark such text ``status="unsupported"``
    with ``language="unknown"``, letting Task 12's non-English page warning
    fire on text whose language cannot be attributed at all. Same uncertain
    semantics as every other None path: unmarked, flows to the engines.
    """
    monkeypatch.setattr(langid, "_langdetect", lambda text: "unknown")
    assert langid.detect(_ENGLISH_TEXT, _SETTINGS) is None


# --- ruling: image items are gated nowhere here ----------------------------------


def test_image_items_skipped_entirely() -> None:
    """kind="image" is skipped by kind — even a text-bearing image item is untouched.

    Language is a text-detector concern; image gating is Task 7's business,
    and ImageRef carries no text the gate could or should judge.
    """
    image = Item(
        kind="image",
        label="A hero image",
        source_ref="article:img[0]",
        image_path="media/img-1.png",
        mime="image/png",
        text=_ENGLISH_TEXT,  # defensively setable; still skipped by kind
    )
    assert mark_text_items([image], _SETTINGS) == 0
    assert image.status == "ok" and image.language is None


def test_mixed_kinds_only_text_marked() -> None:
    """A realistic page list: only the text item's language is stamped."""
    image = Item(
        kind="image",
        label="Hero",
        source_ref="article:img[0]",
        image_path="media/img-1.png",
        mime="image/png",
    )
    english = _text_item(_ENGLISH_TEXT)
    hindi = _text_item(_HINDI_TEXT, label="Paragraph 2")
    emoji = _text_item(_EMOJI_TEXT, label="Paragraph 3")
    items = [image, english, hindi, emoji]
    assert mark_text_items(items, _SETTINGS) == 1  # only the hindi item marked
    assert image.language is None and image.status == "ok"
    assert english.language == "en" and english.status == "ok"
    assert hindi.status == "unsupported" and hindi.language == "hi"
    assert emoji.status == "ok" and emoji.language is None


# --- ruling: seeded detector is deterministic ------------------------------------


def test_repeated_calls_deterministic() -> None:
    """DetectorFactory.seed = 0 pins the RNG: same text, same verdict, every call."""
    for text, expected in ((_ENGLISH_TEXT, "en"), (_HINDI_TEXT, "hi")):
        verdicts = {detect(text, _SETTINGS) for _ in range(5)}
        assert verdicts == {expected}  # exactly one verdict across repeats


# --- ruling: the page warning lives exactly once, word for word ------------------


def test_non_english_warning_matches_plan_verbatim() -> None:
    """Task 12 imports this constant; a paraphrase would break plan compliance."""
    assert NON_ENGLISH_WARNING == "text is not English; those blocks were not analyzed"
