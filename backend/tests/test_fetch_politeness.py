"""Task 4: fetcher politeness tests (robots, Retry-After, identity) — all offline.

Same offline levers as ``test_fetcher.py``:

1. "patched dial" — the ``fetcher.is_forbidden_ip`` seam is monkeypatched to
   lift the loopback guard *only* (fixture origins bind to 127.0.0.1);
   everything else about the SSRF guard stays production-exact.
2. a real ``http.server`` fixture serves robots.txt + pages, records every
   request (arrival time, path, User-Agent) so tests pin *that no disallowed
   page was ever dialed*, that 429s waited for the requested seconds, and the
   User-Agent the wire carried.

robots.txt content is per-test module state (an origin has exactly one
robots.txt route, so tests install their own body before fetching).
"""

import asyncio
import email.utils
import http.server
import ipaddress
import threading
import time
import urllib.parse

import pytest

from app.settings import Settings
from app.settings import settings as app_settings
from pipeline import fetcher
from pipeline.fetcher import fetch

_PAGE_BODY = b"<html><body>politeness fixture</body></html>"
_PAGE_CTYPE = "text/html; charset=utf-8"
_UA_PREFIX = "AIContentDetector/0.1 ("  # the brief's pinned identity prefix

# "for our UA": the specific entry matches our product token only; a wildcard
# entry exists so a pass proves the *specific* rule was the one consulted.
_ROBOTS_FORBID_OUR_UA = "User-agent: AIContentDetector\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
_ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /\n"

# Fixture origin state; each test installs what it needs before fetching.
_ROBOTS_TXT: str = _ROBOTS_ALLOW_ALL  # default: permissive robots.txt
_ROBOTS_STATUS: int = 200  # default: a healthy robots.txt endpoint
_RETRY_AFTER: str = "2"  # default: the /rate-limited 429's requested wait
_REQUESTS: list[tuple[float, str, str]] = []  # (time.monotonic(), path, User-Agent)
_HITS: dict[str, int] = {}  # per-path request counter (the 429-then-200 route)


def _install(robots_txt: str, *, robots_status: int = 200, retry_after: str = "2") -> None:
    """Install fixture state (robots body/status + 429 Retry-After) + clears."""
    global _ROBOTS_TXT, _ROBOTS_STATUS, _RETRY_AFTER
    _ROBOTS_TXT = robots_txt
    _ROBOTS_STATUS = robots_status
    _RETRY_AFTER = retry_after
    _REQUESTS.clear()
    _HITS.clear()


def _mk_settings(**overrides: object) -> Settings:
    """Settings pinned to the brief's defaults so backend/.env cannot skew tests."""
    values: dict[str, object] = {
        "fetch_timeout_s": 15,
        "body_cap_mb": 10,
        "max_redirects": 3,
        "respect_robots": True,
    }
    values.update(overrides)
    return Settings(**values)


class _Server(http.server.ThreadingHTTPServer):
    """Fixture origin; serves robots.txt + politeness routes below."""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Path-programmable loopback origin that records every request."""

    def log_message(self, format: str, *args: object) -> None:
        """Silence per-request logging; failures live in the client assertions."""

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        ua = self.headers.get("User-Agent", "")
        _REQUESTS.append((time.monotonic(), path, ua))
        try:
            if path == "/robots.txt":
                if _ROBOTS_STATUS == 200:
                    body = _ROBOTS_TXT.encode()
                else:  # the unreachable-robots fixtures (500 / 404 buckets)
                    body = b"fixture robots unavailable"
                self._send(_ROBOTS_STATUS, [("Content-Type", "text/plain")], body)
            elif path == "/page":
                self._send(200, [("Content-Type", _PAGE_CTYPE)], _PAGE_BODY)
            elif path == "/rate-limited":
                hit = _HITS.get(path, 0) + 1
                _HITS[path] = hit
                if hit == 1:  # first attempt: 429 with the requested wait
                    self._send(
                        429,
                        [("Content-Type", "text/plain"), ("Retry-After", _RETRY_AFTER)],
                        b"fixture rate limit",
                    )
                else:  # the one retry: success with intact data
                    self._send(200, [("Content-Type", _PAGE_CTYPE)], _PAGE_BODY)
            else:
                self._send(404, [("Content-Type", "text/plain")], b"no such fixture")
        except OSError:
            # Client aborted: outcome already recorded client-side; server-side
            # broken pipes are noise (same rationale as test_fetcher's handler).
            pass

    def _send(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def allow_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched dial for the local fixtures: lift the guard for loopback only.

    Copied from test_fetcher's fixture of the same name: ``http://127.0.0.1`` is
    refused by the production guard, so the wrapped seam keeps every other
    rejection identical to production while letting the fixture dial.
    """
    real = fetcher.is_forbidden_ip

    def fixture_guard(ip: str) -> bool:
        try:
            addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
                ip.split("%", 1)[0]
            )
        except ValueError:
            return True
        if addr.version == 6 and addr.ipv4_mapped is not None:  # type: ignore[union-attr]
            addr = addr.ipv4_mapped  # type: ignore[assignment]
        if addr.is_loopback:
            return False
        return real(ip)

    monkeypatch.setattr(fetcher, "is_forbidden_ip", fixture_guard)


@pytest.fixture()
def server() -> str:
    """Loopback origin URL; thread dies with the test, shutdown politely."""
    httpd = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _paths() -> list[str]:
    """Paths of every request the fixture origin recorded, in order."""
    return [path for _, path, _ in _REQUESTS]


# --- the brief's four pinned tests ----------------------------------------------


def test_robots_disallowed(server: str, allow_loopback: None) -> None:
    """robots.txt Disallow: / for our UA → robots_disallowed, page never dialed."""
    _install(_ROBOTS_FORBID_OUR_UA)
    r = asyncio.run(fetch(f"{server}/page", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "robots_disallowed"
    # only the robots.txt consult may have happened; /page must be untouched
    assert _paths() == ["/robots.txt"]


def test_robots_allowed(server: str, allow_loopback: None) -> None:
    """Permissive robots.txt → straight success; the consult runs before dialing."""
    _install(_ROBOTS_ALLOW_ALL)
    r = asyncio.run(fetch(f"{server}/page", _mk_settings()))
    assert r.status == "ok"
    assert r.content == _PAGE_BODY
    assert r.error_code is None
    assert r.http_status == 200
    # the consult must precede the content dial (that is the politeness point)
    assert _paths() == ["/robots.txt", "/page"]


def test_429_retry_after(server: str, allow_loopback: None) -> None:
    """429 once with Retry-After: 2 → wait ≈2 s (never more), retry once → 200."""
    _install(_ROBOTS_ALLOW_ALL)
    hits = asyncio.run(fetch(f"{server}/rate-limited", _mk_settings()))
    assert hits.status == "ok"
    assert hits.content == _PAGE_BODY  # data intact: the eventual 200's body
    assert hits.error_code is None
    assert _HITS.get("/rate-limited") == 2  # exactly one retry

    rate_hits = [t for t, path, _ in _REQUESTS if path == "/rate-limited"]
    waited = rate_hits[1] - rate_hits[0]  # declared in the 429: Retry-After: 2
    assert waited >= 2.0 - 0.1  # the wait was actually honored
    assert waited <= 2.0 + 0.25  # ≤ requested (scheduler jitter slack only)


def test_identifiable_user_agent(server: str, allow_loopback: None) -> None:
    """Every request (robots.txt consult and page alike) carries our UA prefix."""
    _install(_ROBOTS_ALLOW_ALL)
    r = asyncio.run(fetch(f"{server}/page", _mk_settings()))
    assert r.status == "ok"
    assert _paths() == ["/robots.txt", "/page"]
    for _, _, ua in _REQUESTS:
        assert ua.startswith(_UA_PREFIX)


# --- supporting politeness tests (Retry-After semantics, respect flag) -----------


def test_retry_after_seconds_and_garbage() -> None:
    """Retry-After: delay-seconds parses; absent/garbage headers → no wait."""
    from pipeline.robots import retry_after_wait

    assert retry_after_wait({"Retry-After": "2"}) == 2.0
    assert retry_after_wait({"Retry-After": " 3 "}) == 3.0  # trimmed, below the cap
    assert retry_after_wait({}) is None
    assert retry_after_wait({"Retry-After": "soon-ish"}) is None
    # No-category exotic digits ("²", "①") must classify as garbage → None; the
    # pre-hardening isdigit() accepted them and float() then raised a raw
    # ValueError (fix round 1, finding #2).
    assert retry_after_wait({"Retry-After": "²"}) is None
    assert retry_after_wait({"Retry-After": "①"}) is None
    assert retry_after_wait({"Retry-After": "② seconds"}) is None


def test_retry_after_http_date() -> None:
    """Retry-After as an HTTP-date (RFC 9110 §10.2.1) → delta seconds; past → 0."""
    from pipeline.robots import retry_after_wait

    soon = email.utils.formatdate(time.time() + 4, usegmt=True)  # whole-second RFC dates
    wait = retry_after_wait({"Retry-After": soon})
    assert wait is not None and 3.0 <= wait <= 4.5  # floor(t+4) - now ∈ [3, ~4)
    past = email.utils.formatdate(time.time() - 5, usegmt=True)
    assert retry_after_wait({"Retry-After": past}) == 0.0


def test_retry_after_wait_is_capped() -> None:
    """A computed wait never exceeds Settings.engine_timeout_s."""
    from pipeline.robots import retry_after_wait

    cap = float(app_settings.engine_timeout_s)
    assert retry_after_wait({"Retry-After": "999999"}) == cap


def test_respect_robots_false_skips_consult(server: str, allow_loopback: None) -> None:
    """The binding flag: respect_robots=False fetches without a robots consult."""
    _install(_ROBOTS_FORBID_OUR_UA)  # robots.txt would forbid us — flag must win
    r = asyncio.run(fetch(f"{server}/page", _mk_settings(respect_robots=False)))
    assert r.status == "ok"
    assert r.content == _PAGE_BODY
    assert _paths() == ["/page"]  # no robots.txt consult happened at all


# --- fix round 1: strict robots-unreachable policy (RFC 9309 bucket split) -------


def test_robots_unreachable_5xx_disallow(server: str, allow_loopback: None) -> None:
    """RFC 9309 §2.3.1.4: unreachable robots (5xx) → complete disallow, no dial."""
    _install(_ROBOTS_ALLOW_ALL, robots_status=500)
    r = asyncio.run(fetch(f"{server}/page", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "robots_disallowed"
    assert _paths() == ["/robots.txt"]  # the page is provably never dialed

    # Consult-level pin of the ruling: the bucket split lives in check_allowed.
    # (Lazy import also lets this file collect RED while pipeline.robots is
    # still unhardened; running it after the fetch keeps the log assert clean.)
    from pipeline.robots import check_allowed

    assert check_allowed(f"{server}/page") is False


def test_robots_404_permissive(server: str, allow_loopback: None) -> None:
    """RFC 9309 §2.3.1.3: 4xx robots.txt = unavailable → permissive fetch.

    Hardening guard: the must-stay-permissive bucket while 5xx/DNS/timeouts
    moved to complete disallow. (Also passes pre-hardening under the old
    unconditional fail-open; it pins that hardening 5xx must NOT drag the
    4xx bucket along.)
    """
    _install("cannot serve robots", robots_status=404)
    r = asyncio.run(fetch(f"{server}/page", _mk_settings()))
    assert r.status == "ok"
    assert r.content == _PAGE_BODY
    assert _paths() == ["/robots.txt", "/page"]  # consult ran, page still served


def test_429_hostile_retry_after_header_nonfatal(server: str, allow_loopback: None) -> None:
    """A hostile Retry-After digit must fold to http_429, never raise.

    "²" (latin-1 wire-safe, unlike circled "②" which http.server cannot even
    send) passes the old isdigit() check but crashes float(); with the
    hardened parser it is garbage → no wait → the existing single-attempt
    http_429 FetchResult fallback (fix round 1, finding #2).
    """
    _install(_ROBOTS_ALLOW_ALL, retry_after="²")  # No-category digit, latin-1
    r = asyncio.run(fetch(f"{server}/rate-limited", _mk_settings()))  # must not raise
    assert r.status == "fetch_failed"
    assert r.error_code == "http_429"
    assert r.http_status == 429
    assert _HITS.get("/rate-limited") == 1  # garbage header → the single attempt
