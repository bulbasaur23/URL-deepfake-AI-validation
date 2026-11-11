"""Pipeline value objects (Task 3 slice: FetchResult + Item).

The Task 2 review pin froze the status vocabularies; every token lives here
exactly once and later tasks reuse these aliases instead of re-declaring:

- ``FetchStatus`` — scan fetch stage outcomes: ``ok | fetch_failed``.
- ``FetchErrorCode`` — ``bad_scheme | private_ip | redirect_loop |
  redirect_too_many | body_too_large | timeout | dns_failure | http_4xx |
  http_5xx | http_429 | robots_disallowed``. ``robots_disallowed`` is defined
  here only as a token; robots.txt behaviour belongs to Task 4.
- ``ItemStatus`` — ``ok | degraded | unavailable | blocked_by_site |
  fetch_failed | too_large | unsupported``. Item.status defaults to ``"ok"``.

Never persist a token outside its set; typing catches drift in downstream
tasks. ``EngineResult`` joined here with the Task 8 engine contract (so
``engine_results`` is typed); ``ItemInput`` follows with Task 12's runner.
``ImageRef`` joins here with the Task 5 extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FetchStatus = Literal["ok", "fetch_failed"]
FetchErrorCode = Literal[
    "bad_scheme",
    "private_ip",
    "redirect_loop",
    "redirect_too_many",
    "body_too_large",
    "timeout",
    "dns_failure",
    "http_4xx",
    "http_5xx",
    "http_429",
    "robots_disallowed",
]
ItemStatus = Literal[
    "ok",
    "degraded",
    "unavailable",
    "blocked_by_site",
    "fetch_failed",
    "too_large",
    "unsupported",
]
ItemKind = Literal["text", "image"]


@dataclass
class EngineResult:
    """One engine's verdict on one Item (Task 8 contract; design §7.2).

    ``score`` is the 0..100 synthetic-likelihood, ``confidence`` the 0..1
    engine self-assessment, ``rationale`` one human-readable sentence for the
    evidence panel, and ``latency_ms`` this analyze() call's wall-clock cost.
    Task 12's engine runner appends these to ``Item.engine_results``; Task 13
    aggregates them into per-item verdicts.
    """

    engine_id: str
    engine_version: str
    score: float  # 0..100 synthetic-likelihood
    confidence: float  # 0..1
    rationale: str
    latency_ms: int


@dataclass
class FetchResult:
    """Outcome of one SSRF-guarded fetch (``fetcher.fetch``).

    ``content``/``content_type`` carry payload only when ``status == "ok"``.
    On failure, ``http_status`` is the last seen HTTP status (``None`` when the
    fetch never got a response, e.g. bad scheme / DNS / dial timeout) and
    ``final_url`` is the last attempted URL.
    """

    status: FetchStatus
    content: bytes = b""
    content_type: str = ""
    http_status: int | None = None
    final_url: str | None = None
    error_code: FetchErrorCode | None = None


@dataclass
class Item:
    """One extractable content payload belonging to a scanned page.

    Task 5's extractor produces items (text + label, images with paths); Task
    7's engine runner appends to ``engine_results``. Required fields come
    first (dataclass rule), so optional fields are keyword-only in practice —
    semantics follow the brief's listing exactly, only positional order moved
    to keep the class constructible.
    """

    kind: ItemKind
    label: str
    source_ref: str
    text: str | None = None
    image_path: str | None = None
    mime: str | None = None
    status: ItemStatus = "ok"
    language: str | None = None
    verdict: str | None = None
    confidence: float | None = None
    # Filled by the engine runner (Task 12); each entry is an EngineResult.
    engine_results: list[EngineResult] = field(default_factory=list)


@dataclass
class ImageRef:
    """One extractable image candidate (Task 5) — Task 7 turns it into an Item.

    ``url_candidates`` lists sources in extraction order — ``src`` first (when
    present), then ``srcset`` entries in attribute order — each as
    ``(absolute_url, width_or_None)`` where ``width`` is the ``w`` descriptor
    in natural pixels (``None`` for plain srcs and ``x`` descriptors). The
    extractor excludes ``data:`` URIs and non-http(s) candidates, so Task 7
    never sees them. Task 7 resolves the winner: highest ``width``; last entry
    when no widths exist (yielding ``src`` when srcset is absent outright).
    """

    label: str
    url_candidates: list[tuple[str, int | None]]
    source_ref: str
