"""Fetcher politeness (GATE 1 supplement): robots.txt, Retry-After, identity.

``fetcher`` consults this module instead of embedding politeness rules:

- ``check_allowed(url)`` fetches the target origin's ``robots.txt`` through
  ``fetch()`` — the one SSRF-guarded channel (same resolve/refuse/pin rules)
  — and answers whether our product token may crawl ``url`` (stdlib
  ``urllib.robotparser``, following RFC 9309). A robots.txt that cannot be
  retrieved (404, auth wall, DNS or transport failure) is a *permissive*
  outcome: an unanswerable robots endpoint must not become a site-wide veto,
  and the content fetch itself still reports the real status downstream.
- ``retry_after_wait(headers)`` reads ``Retry-After`` (delay-seconds or
  HTTP-date, RFC 9110 §10.2.1) in seconds, hedged at
  ``Settings.engine_timeout_s``, so a hostile header cannot park the scan
  deadline.
- ``USER_AGENT`` is the one identity string the fetcher puts on the wire;
  ``check_allowed`` parses robots.txt under the same product token.

Re-entrancy: a robots.txt fetch consults ``fetch()``, so the robots check for
*that* request must not re-enter ``check_allowed``. The suppression travels in
a ``ContextVar`` rather than a thread-local deliberately: the robots fetch
runs on yet another worker thread (``fetch()`` → ``asyncio.to_thread``), and
``ContextVar`` copies flow into that thread, where thread-locals visibly
would not.
"""

from __future__ import annotations

import asyncio
import contextvars
import email.utils
import urllib.parse
import urllib.robotparser
from datetime import UTC, datetime
from typing import Any

from app.settings import settings as _app_settings
from pipeline.types import FetchResult

# The identity servers rate-limit, robots.txt names, and log against. The
# brief pins the prefix "AIContentDetector/0.1 ("; the parenthesized contact
# block follows the RFC 9309 bot-announcement convention (info page + contact).
USER_AGENT = (
    "AIContentDetector/0.1 "
    "(https://ai-content-detector.invalid/bot-info; contact=admin@ai-content-detector.invalid)"
)

# robots.txt groups name us by the product token — the part before the
# version (e.g. "User-agent: AIContentDetector").
_PRODUCT_TOKEN = USER_AGENT.split("/", 1)[0]

# True while a robots.txt itself is being fetched: fetcher skips consulting.
_CONSULTING = contextvars.ContextVar("pipeline_robots_consulting", default=False)


def robots_fetch_active() -> bool:
    """True when the current fetch IS a robots.txt fetch (no nested consults)."""
    return _CONSULTING.get()


def robots_txt_url(url: str) -> str:
    """The robots.txt URL for ``url``'s origin (same scheme/host/port)."""
    return urllib.parse.urljoin(url, "/robots.txt")


def check_allowed(url: str) -> bool:
    """True when ``url``'s origin robots.txt permits our UA to crawl it."""
    # Lazy import: fetcher imports this module at its top, so a top-level
    # round-trip import would loop. By call time fetcher is fully loaded.
    from pipeline.fetcher import fetch

    token = _CONSULTING.set(True)
    try:
        # Legal here: check_allowed runs on the blocking fetch worker, which
        # owns no running event loop. asyncio.run and asyncio.to_thread both
        # copy the context, so the suppression flag reaches the very worker
        # thread that fetches robots.txt and stops the nested consult there.
        result = asyncio.run(fetch(robots_txt_url(url), _app_settings))
    finally:
        _CONSULTING.reset(token)
    return _permitted(result, url)


def _permitted(result: FetchResult, url: str) -> bool:
    """robots.txt retrievable → parse and match our token; otherwise permissive."""
    if result.status != "ok":  # 404 / auth wall / DNS / dial failure → allow
        return True
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(result.content.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(_PRODUCT_TOKEN, url)


def retry_after_wait(response_headers: Any) -> float | None:
    """Seconds to wait per a ``Retry-After`` header; None when unusable.

    ``response_headers`` is any ``.get(name, default)`` mapping (http.client's
    ``HTTPMessage`` and plain dicts both). Per RFC 9110 §10.2.1 the value is
    non-negative delay-seconds or an HTTP-date; a date in the past means the
    wait is already over (0.0). The result never exceeds
    ``Settings.engine_timeout_s``.
    """
    raw = response_headers.get("Retry-After")
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    if value.isdigit():
        wait = float(value)
    else:
        try:
            deadline = email.utils.parsedate_to_datetime(value)
        except TypeError, ValueError:
            return None
        if deadline is None:  # very old stdlib returned None instead of raising
            return None
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        wait = max((deadline - datetime.now(UTC)).total_seconds(), 0.0)
    return min(wait, float(_app_settings.engine_timeout_s))
