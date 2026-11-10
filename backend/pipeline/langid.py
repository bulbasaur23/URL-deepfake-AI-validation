"""Language gate (Task 6): which text blocks may the detector models judge?

``langid.detect(text, settings)`` maps one text block to its ISO-639-1 code
(``"en"``, ``"hi"``, ...) or ``None``; ``langid.mark_text_items(items,
settings)`` applies that verdict to Task 5's text ``Item``s in place:

- English text keeps ``status="ok"`` and gains ``language="en"``;
- text detected as another language is marked ``status="unsupported"``
  (and counted) but NEVER removed from the list — Task 12 owns the
  page-level warning built on these markers, and dropping items would
  strand that warning and shift downstream indexes;
- text that cannot be classified at all is left completely unmarked:
  detection failure is not proof of non-English, and the anti-goal is
  wrongly excluding legitimate English text, so uncertain text keeps
  flowing to the engines (§11.9: the gate never translates).

Pure in-process gate: no network, no files, no ML models — just seeded
langdetect over strings. The Task 9 text detectors are English-trained, so
every "unsupported" block is one the models must not judge.

Not for images: language is a text-detector concern, so media items are
skipped by ``kind`` and the image story is untouched (ImageRef carries no
text for this gate to judge).
"""

from __future__ import annotations

from typing import Any

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect as _langdetect
from langdetect.detector_factory import init_factory as _init_factory

from pipeline.types import Item

# langdetect is non-deterministic unless seeded (upstream: "if you try to run
# it on a text which is either too short or too ambiguous, you might get
# different results everytime you run it"); seed 0 enforces consistent
# results. Unseeded, each Detector reseeds its private random.Random() from
# system entropy, so the same block could be judged a different language
# across calls — also making parallel/scanned outcomes non-repeatable. Each
# call builds its own Detector, so no RNG state is shared between concurrent
# calls.
DetectorFactory.seed = 0
# Warm the profile factory now instead of off the scan clock: langdetect loads
# its profiles lazily on the first detect() call, and that lazy init both
# costs startup latency inside the 120 s scan deadline and is not thread-safe
# (a second concurrent first-detect could observe a half-initialized global).
# Import time is single-threaded, so building it here removes both problems.
_init_factory()

# Plan-pinned page warning (§11.9), defined here once so Task 12 imports the
# exact string rather than re-typing it. Export/bootstrap only — the gate's
# marking logic below deliberately never references it.
NON_ENGLISH_WARNING = "text is not English; those blocks were not analyzed"


def detect(text: str, settings: Any) -> str | None:
    """Return the ISO-639-1 code of ``text`` (``"en"``, ``"hi"``, ...) or ``None``.

    ``None`` is returned — never a guess — when langdetect cannot classify:
    blank input (empty/whitespace), strings with no removable language
    features (emoji/digit/symbol-only), anything else rejected with
    ``LangDetectException``, or langdetect's own ``"unknown"`` sentinel
    (no candidate cleared its probability threshold — not an ISO-639-1
    code). A ``None`` caller contract: treat it as "flows to the engines
    unmarked"; a failed detection is not evidence of non-English.

    ``settings`` is part of the plan's gate signature and is deliberately
    unused: langdetect exposes no threshold concept worth adopting, so no
    tunable is honoured in P1. It stays a parameter so a later task can add
    one without changing call sites.
    """
    stripped = (text or "").strip()  # blank input fast-path (also: dataclass tolerance)
    if not stripped:
        return None
    try:
        code = _langdetect(stripped)
    except LangDetectException:
        return None
    # langdetect signals "no profile cleared its probability threshold" by
    # returning the sentinel ``UNKNOWN_LANG == "unknown"`` rather than
    # raising — not an ISO-639-1 code. Map it to ``None`` (the same
    # uncertain semantics as every failure path above): unattributable
    # text is not evidence of non-English, so it stays unmarked and keeps
    # flowing to the engines.
    return None if code == "unknown" else code


def mark_text_items(items: list[Item], settings: Any) -> int:
    """Gate every text ``Item`` in place; return how many were marked unsupported.

    Per item: non-text items (kind image) and blank text are skipped;
    ``detect() == "en"`` stamps ``language="en"`` with ``status`` left
    "ok"; any other code stamps ``language=<code>`` +
    ``status="unsupported"``; ``detect()`` returning ``None`` stamps
    nothing at all (uncertain text is not proof of non-English). The list
    is only read, and the items are mutated in place — unsupported items
    are carried, never dropped.
    """
    marked = 0
    for item in items:
        if item.kind != "text" or not (item.text or "").strip():
            continue
        code = detect(item.text, settings)
        if code is None:
            continue  # unclassified: stay "ok", language stays None, unmarked
        item.language = code
        if code != "en":
            item.status = "unsupported"  # judged elsewhere never; warned at page level
            marked += 1
    return marked
