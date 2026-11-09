"""Task 3: SSRF-safe fetcher tests — all offline.

Two offline levers keep the suite from touching the network:

1. "patched dial" — the module seams ``fetcher.resolve_ip`` /
   ``fetcher.is_forbidden_ip`` are monkeypatched. The loopback guard is lifted
   *only* for ``127.0.0.1`` (every real private/loopback/metadata range still
   rejects), because the local test servers bind there.
2. a real ``http.server`` fixture over loopback drives the happy paths:
   success roundtrip, streamed 10 MB cap, redirect hops, gzip payload/bomb,
   status mapping, and timeout — patched to fail fast (1 s) instead of the
   default 15 s.
"""

import asyncio
import gzip
import http.server
import ipaddress
import re
import socket
import threading
import time
import urllib.parse
from collections.abc import Iterator

import pytest

from app.settings import Settings
from pipeline import fetcher
from pipeline.fetcher import fetch
from pipeline.types import FetchResult, Item

_BODY_CAP = 10 * 1024 * 1024  # Settings.body_cap_mb default, in bytes
_BIG_STREAM_TOTAL = _BODY_CAP + 1  # one byte over the cap: the streaming page
_OK_BODY = b"<html><body>detector fixture</body></html>"
_OK_CTYPE = "text/html; charset=utf-8"
_SLOW_SECONDS = 3  # > the 1 s timeout the /slow test installs


def _mk_settings(**overrides: object) -> Settings:
    """Settings pinned to the brief's defaults so backend/.env cannot skew tests."""
    values: dict[str, object] = {
        "fetch_timeout_s": 15,
        "body_cap_mb": 10,
        "max_redirects": 3,
        "respect_robots": True,  # Task 4 consumes this; fetcher ignores it for now
    }
    values.update(overrides)
    return Settings(**values)


class _Server(http.server.ThreadingHTTPServer):
    """Fixture origin; every route answers from module constants below."""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Path-programmable loopback origin (kept on the HTTP/1.0 default)."""

    def log_message(self, format: str, *args: object) -> None:
        """Silence per-request logging; failures live in the client assertions."""

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        hop = re.fullmatch(r"/redir/(\d+)", path)
        code = re.fullmatch(r"/status/(\d+)", path)
        try:
            if path == "/ok":
                self._send(200, [("Content-Type", _OK_CTYPE)], _OK_BODY)
            elif path == "/big":
                self._stream(_BIG_STREAM_TOTAL)
            elif path == "/bomb":
                payload = gzip.compress(b"\x00" * (_BODY_CAP + 1), mtime=0)
                self._send(
                    200,
                    [("Content-Type", "application/octet-stream"), ("Content-Encoding", "gzip")],
                    payload,
                )
            elif path == "/gzip-ok":
                payload = gzip.compress(_OK_BODY, mtime=0)
                self._send(
                    200,
                    [("Content-Type", _OK_CTYPE), ("Content-Encoding", "gzip")],
                    payload,
                )
            elif hop:
                hops = int(hop.group(1))
                if hops == 0:
                    self._send(200, [("Content-Type", "text/plain")], b"chain end")
                else:
                    self._redirect(302, f"/redir/{hops - 1}")
            elif path == "/redirect-file":
                self._redirect(302, "file:///etc/passwd")
            elif code:
                self._send(int(code.group(1)), [("Content-Type", "text/plain")], b"fixture failure")
            elif path == "/slow":
                time.sleep(_SLOW_SECONDS)
                self._send(200, [("Content-Type", "text/plain")], b"finally")
            else:
                self._send(404, [("Content-Type", "text/plain")], b"no such fixture")
        except OSError:
            # Client aborted (cap trimmed / timed out): outcome was already
            # recorded client-side; a broken pipe on the server thread is noise.
            pass

    def _stream(self, total: int) -> None:
        """Write ``total`` bytes with no Content-Length: read-until-close on 1.0.

        Omitting Content-Length deliberately — with one the client's header
        precheck would abort before the streaming path is ever exercised.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        blob = b"\x00" * 65_536
        remaining = total
        while remaining > 0:
            self.wfile.write(blob[: min(65_536, remaining)])
            remaining -= 65_536

    def _send(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, status: int, location: str) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def allow_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched dial for the local fixtures: lift the guard for loopback only.

    ``http://127.0.0.1:<port>`` would be rejected by the production guard
    (even loopback is a non-routable address), so the wrapped seam keeps every
    other rejection identical to production while letting the fixture dial.
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
def server() -> Iterator[str]:
    """Loopback origin URL; threads die with the test process, shutdown politely."""
    httpd = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --- the brief's pinned test (verbatim assertions) -------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://192.168.1.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "ftp://x.io/a",
    ],
)
def test_private_and_metadata_ips_rejected(url: str) -> None:
    """Loopback/private/link-local/metadata IPs and off-allowlist schemes never dial."""
    r = asyncio.run(fetch(url, _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code in {"private_ip", "bad_scheme"}


# --- the brief's named fixture tests --------------------------------------------


def test_success_roundtrip(server: str, allow_loopback: None) -> None:
    """Local 200 page → ok, body + content type kept, final_url set."""
    r = asyncio.run(fetch(f"{server}/ok", _mk_settings()))
    assert isinstance(r, FetchResult)
    assert r.status == "ok"
    assert r.content == _OK_BODY
    assert r.content_type == _OK_CTYPE
    assert r.http_status == 200
    assert r.final_url == f"{server}/ok"
    assert r.error_code is None


def test_body_cap_10mb(server: str, allow_loopback: None) -> None:
    """Streaming cap: 10 MB + 1 bytes → abort with body_too_large, nothing else."""
    r = asyncio.run(fetch(f"{server}/big", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "body_too_large"


def test_three_redirects_ok_fourth_rejected(server: str, allow_loopback: None) -> None:
    """Hop-by-hop re-validation: 3 hops pass, the 4th redirect is refused."""
    ok = asyncio.run(fetch(f"{server}/redir/3", _mk_settings()))
    assert ok.status == "ok"
    assert ok.content == b"chain end"
    assert ok.final_url == f"{server}/redir/0"  # chain resolved end-to-end

    too_many = asyncio.run(fetch(f"{server}/redir/4", _mk_settings()))
    assert too_many.status == "fetch_failed"
    assert too_many.error_code == "redirect_too_many"


# --- supporting fixture tests ---------------------------------------------------


def test_gzip_payload_is_decompressed(server: str, allow_loopback: None) -> None:
    """Content-Encoding: gzip responses are transparently decompressed."""
    r = asyncio.run(fetch(f"{server}/gzip-ok", _mk_settings()))
    assert r.status == "ok"
    assert r.content == _OK_BODY
    assert r.content_type == _OK_CTYPE


def test_gzip_bomb_rejected(server: str, allow_loopback: None) -> None:
    """Decompression cap: ~10 KB gzip that inflates to 10 MB + 1 → body_too_large."""
    r = asyncio.run(fetch(f"{server}/bomb", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "body_too_large"


def test_redirect_to_forbidden_scheme(server: str, allow_loopback: None) -> None:
    """Re-validation applies to every hop: a redirect to file:// rejected."""
    r = asyncio.run(fetch(f"{server}/redirect-file", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "bad_scheme"


def test_http_status_error_mapping(server: str, allow_loopback: None) -> None:
    """4xx/5xx/429 map to their pinned taxonomy tokens with the status kept."""
    for path, expected, http_status in [
        ("/status/404", "http_4xx", 404),
        ("/status/403", "http_4xx", 403),
        ("/status/500", "http_5xx", 500),
        ("/status/503", "http_5xx", 503),
        ("/status/429", "http_429", 429),
    ]:
        r = asyncio.run(fetch(f"{server}{path}", _mk_settings()))
        assert r.status == "fetch_failed", path
        assert r.error_code == expected, path
        assert r.http_status == http_status, path


def test_dns_failure_maps_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolver failures become dns_failure (patched seam: no real lookups)."""

    def unresolvable(host: str, port: int) -> str:
        raise socket.gaierror(-2, "Name or service not known (test stub)")

    monkeypatch.setattr(fetcher, "resolve_ip", unresolvable)
    r = asyncio.run(fetch("http://offline-unreachable.test/x", _mk_settings()))
    assert r.status == "fetch_failed"
    assert r.error_code == "dns_failure"


def test_timeout_maps_code(server: str, allow_loopback: None) -> None:
    """Origin slower than fetch_timeout_s → timeout (~1 s test runtime)."""
    r = asyncio.run(fetch(f"{server}/slow", _mk_settings(fetch_timeout_s=1)))
    assert r.status == "fetch_failed"
    assert r.error_code == "timeout"


def test_item_defaults_and_pinned_vocabularies() -> None:
    """Item ships status="ok" + a per-instance engine_results list; the pinned
    Task-2 item statuses all construct without extra tokens."""
    item = Item(kind="text", label="main article", source_ref="body:text:main")
    assert item.status == "ok"
    assert item.text is None and item.image_path is None and item.mime is None
    assert item.language is None and item.verdict is None and item.confidence is None
    assert item.engine_results == []

    other = Item(kind="image", label="hero", source_ref="img[0]")
    item.engine_results.append({"engine_id": "text-ai-roberta"})  # shared default?
    assert other.engine_results == []  # default_factory must isolate instances

    for status in (
        "ok",
        "degraded",
        "unavailable",
        "blocked_by_site",
        "fetch_failed",
        "too_large",
        "unsupported",
    ):
        assert Item(kind="text", label="l", source_ref="r", status=status).status == status
