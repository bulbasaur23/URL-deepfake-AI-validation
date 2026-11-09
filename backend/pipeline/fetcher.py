"""SSRF-safe HTTP fetcher (GATE 1) — every scan's fetch stage runs through here.

Owned rules (§7.6 / Global Constraints): http/https scheme allowlist; DNS
resolve then reject private/loopback/link-local/metadata IPs (127/8, 10/8,
172.16/12, 192.168/16, 169.254/16, ::1, fc00::/7) — if *any* resolved address
is forbidden the request is refused; otherwise the first candidate is pinned
and dialed directly (no rebinding window between resolution and connection,
while SNI/Host still name the real host); every redirect hop is re-validated from
scratch, at most ``Settings.max_redirects`` follows with loop detection; the
body is streamed under ``Settings.body_cap_mb`` and, when compressed,
re-checked after decoding (gzip-bomb guard); socket work uses
``Settings.fetch_timeout_s``.

Explicitly deferred to Task 4 robots politeness: robots.txt enforcement, the
identifiable User-Agent string (nothing identifying is sent yet), and
Retry-After handling. ``robots_disallowed`` exists only as a token here.

Transport is stdlib ``http.client`` dialed straight at the pinned IP — no
proxy support (a proxy would defeat IP pinning). Tests never touch the
network: they patch the module seams ``resolve_ips`` / ``is_forbidden_ip``
("patched dial") and serve fixture pages from a loopback ``http.server``.

Network errors that fit no taxonomy token (refused / reset / TLS handshake
failures, corrupt decode, malformed redirect) fail with ``error_code=None``:
the vocabulary is exact, so inventing a token is worse than letting Task 4's
orchestrator layer decide. Genuinely unexpected exceptions (bugs) are not
swallowed — they propagate.
"""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import ssl
import urllib.parse
import zlib
from collections.abc import Iterator
from typing import NamedTuple

from app.settings import Settings
from pipeline.types import FetchErrorCode, FetchResult


class FetchError(Exception):
    """Internal control flow: aborts a fetch with a taxonomy code attached.

    Public so Task 4's robots module can raise the same shape when
    enforcement lands; ``error_code=None`` is the escape hatch for real-world
    failures that fit no exact token.
    """

    def __init__(
        self,
        error_code: FetchErrorCode | None,
        message: str,
        *,
        http_status: int | None = None,
        final_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status
        self.final_url = final_url


class _Hop(NamedTuple):
    """One validated redirect hop: everything the dial needs, already checked."""

    scheme: str
    hostname: str
    port: int  # effective port (scheme default filled in when omitted)
    request_target: str  # path?query — what goes on the wire
    host_header: str  # Host value; includes an explicit non-default port
    explicit_port: int | None  # None when the URL used the scheme default


_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_REDIRECTABLE = frozenset({301, 302, 303, 307, 308})
_CHUNK = 65_536  # streaming read granularity (64 KiB); also bounds inflate spikes
_COMPRESSED_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate"})

# The brief's binding rejection list, verbatim: IPv4 private/loopback/
# link-local (incl. the 169.254.169.254 metadata endpoint), IPv6 loopback and
# unique-local. Plus two same-category tightenings named by the rule's prose:
# fe80::/10 (IPv6 link-local — the link-local category's IPv6 form) and
# 0.0.0.0/8 ("this network", whose literal dial lands on the local host).
_FORBIDDEN_NETS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("0.0.0.0/8"),
)


def is_forbidden_ip(ip: str) -> bool:
    """True when ``ip`` must never be dialed.

    IPv4-mapped IPv6 (``::ffff:a.b.c.d``) and IPv6 zone ids are normalized so
    the famiy-specific ranges above catch them. Unparseable addresses are
    refused: the guard fails closed.
    """
    try:
        addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
            ip.split("%", 1)[0]
        )
    except ValueError:
        return True
    if addr.version == 6 and (mapped := addr.ipv4_mapped) is not None:
        addr = mapped
    return any(addr in net for net in _FORBIDDEN_NETS)


def resolve_ips(host: str, port: int) -> tuple[str, ...]:
    """Resolve ``host`` once to *every* dialable candidate address.

    Module-level seam so tests can replace the dial ("patched dial") without
    touching any network. The caller validates the whole list — one forbidden
    candidate refuses the fetch (reject-if-any) — and the first candidate is
    the pin. The lookup repeats at every redirect hop, so a pin lives exactly
    one connection: a rebinding DNS answer can only mismatch a fresh
    validation, never a live socket.
    """
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    candidates: list[str] = []
    seen: set[str] = set()
    for info in infos:
        if (address := info[4][0]) and address not in seen:
            seen.add(address)
            candidates.append(address)
    if not candidates:
        raise socket.gaierror(f"hostname {host!r} resolved to no usable address")
    return tuple(candidates)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection dialing the pinned IP while SNI and Cert stay on the real host."""

    def __init__(self, pinned_ip: str, sni_host: str, port: int, timeout: float) -> None:
        super().__init__(
            pinned_ip, port=port, timeout=timeout, context=ssl.create_default_context()
        )
        self._sni_host = sni_host

    def connect(self) -> None:
        """Dial the pinned IP; TLS wraps it with the real hostname.

        Base-class ``connect()`` would use ``self.host`` — the pinned IP — as
        the SNI name, which the real server rejects. Proxies/tunnels are
        intentionally unsupported (a proxy defeats IP pinning).
        """
        raw = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self._sni_host)


def _connect(hop: _Hop, ip: str, timeout: float) -> http.client.HTTPConnection:
    """Connection object dialing the pinned ``ip``; Host/SNI name the real host."""
    if hop.scheme == "https":
        return _PinnedHTTPSConnection(ip, hop.hostname, hop.port, timeout)
    return http.client.HTTPConnection(ip, hop.port, timeout=timeout)


def _parse_url(url: str) -> _Hop:
    """Parse and validate one URL against the allowlist → dialable plan."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _SCHEMES:
        raise FetchError("bad_scheme", f"scheme {scheme!r} is not http/https: {url}")
    hostname = parts.hostname or ""
    if not hostname:
        raise FetchError("bad_scheme", f"URL without a network host: {url}")
    try:
        explicit_port = parts.port
    except ValueError as exc:
        raise FetchError("bad_scheme", f"unusable port in URL: {url}") from exc
    try:
        if not hostname.isascii():
            hostname = hostname.encode("idna").decode("ascii")  # Unicode hosts → punycode
    except UnicodeError as exc:
        raise FetchError("bad_scheme", f"hostname is not IDN-encodable: {url}") from exc
    port = explicit_port if explicit_port is not None else _DEFAULT_PORTS[scheme]
    request_target = parts.path or "/"
    if parts.query:
        request_target += f"?{parts.query}"
    host_header = f"[{hostname}]" if ":" in hostname else hostname  # RFC 5952 IPv6 literal
    if explicit_port is not None and explicit_port != _DEFAULT_PORTS[scheme]:
        host_header += f":{explicit_port}"
    return _Hop(scheme, hostname, port, request_target, host_header, explicit_port)


def _error_code_for_status(status: int) -> FetchErrorCode | None:
    """Map a non-followed, non-2xx response status onto the exact vocabulary."""
    if status == 429:
        return "http_429"
    if 400 <= status < 500:
        return "http_4xx"
    if 500 <= status < 600:
        return "http_5xx"
    return None  # unfollowable 3xx / exotic statuses: no invented token


def _new_decoder(encoding: str, head: bytes) -> zlib._Decompress:
    """Decompressor for a claimed Content-Encoding (sniffs deflate framing).

    ``deflate`` is the zlib-wrapped stream per RFC 7230, but origins in the
    wild send bare deflate; a leading 0x78 byte discriminates the two.
    """
    if encoding == "deflate":
        wbits = zlib.MAX_WBITS if head == b"\x78" else -zlib.MAX_WBITS
        return zlib.decompressobj(wbits)
    return zlib.decompressobj(16 + zlib.MAX_WBITS)  # gzip / x-gzip members


def _read_body(response: http.client.HTTPResponse, encoding: str | None, cap: int) -> bytes:
    """Stream the (possibly compressed) body while enforcing an absolute cap.

    The cap counts *decoded* bytes: a gzip bomb — tiny compressed stream, huge
    flag content — aborts exactly like an oversized plain body. Reads in 64
    KiB chunks, so a lying undersized Content-Length cannot buffer-bomb either
    and a single chunk's inflate output stays bounded by ``decompress(…, room)``.
    """
    out = bytearray()
    header = response.headers.get("Content-Length")
    if header is not None:
        try:
            declared = int(header)
        except ValueError:
            declared = None
        if declared is not None and declared > cap:
            raise FetchError("body_too_large", f"Content-Length {declared} exceeds cap {cap}")
    claimed = (encoding or "").strip().lower()
    if claimed not in _COMPRESSED_ENCODINGS | {"", "identity"}:
        # Advertised "gzip, deflate" only; an unexpected decode would hand the
        # extractor mojibake — fail closed instead.
        raise FetchError(None, f"unsupported Content-Encoding {claimed!r}")
    if claimed not in _COMPRESSED_ENCODINGS:  # identity / absent payload
        total = 0
        while chunk := response.read(_CHUNK):
            total += len(chunk)
            if total > cap:
                raise FetchError("body_too_large", f"streamed body exceeds cap {cap}")
            out += chunk
        return bytes(out)
    _stream_inflated(response, claimed, cap, out)
    return bytes(out)


def _stream_inflated(
    response: http.client.HTTPResponse, encoding: str, cap: int, out: bytearray
) -> None:
    """Decode gzip/deflate chunks across member boundaries into ``out``.

    ``out`` is the caller's accumulator, so the cap spans chunk and member
    boundaries; decoding anything past the cap aborts the whole fetch.
    """
    decoder: zlib._Decompress | None = None
    received = False
    for chunk in _iter_response(response):
        received = True
        current = chunk
        while current:
            if decoder is None:
                decoder = _new_decoder(encoding, current[:1])
            room = cap - len(out) + 1
            piece = decoder.decompress(current, room)
            out += piece
            if len(out) > cap:
                raise FetchError("body_too_large", f"decoded body exceeds cap {cap}")
            if decoder.eof:  # next gzip member (or trailing bytes) in unused_data
                current, decoder = decoder.unused_data, None
                continue
            current = decoder.unconsumed_tail
            if current and not piece:
                raise FetchError(None, "decompressor stalled without progress")
    if not received:  # zero compressed bytes: no valid stream could have arrived
        raise FetchError(None, "compressed response ended without any payload")


def _iter_response(response: http.client.HTTPResponse) -> Iterator[bytes]:
    """Yield raw wire chunks until EOF."""
    while chunk := response.read(_CHUNK):
        yield chunk


def _hop_loop(url: str, settings: Settings) -> FetchResult:
    """Validate → resolve → dial → (redirect?) loop; FetchError when unsafe."""
    cap = settings.body_cap_mb * 1_048_576
    timeout = float(settings.fetch_timeout_s)
    budget = int(settings.max_redirects)
    current = url
    visited: set[tuple[str, str, int, str]] = set()

    try:
        while True:
            hop = _parse_url(current)
            endpoint = (hop.scheme, hop.hostname, hop.port, hop.request_target)
            if endpoint in visited:  # fragment-only differences still loop
                raise FetchError("redirect_loop", f"already visited {current}", final_url=current)
            visited.add(endpoint)

            resolved = resolve_ips(hop.hostname, hop.port)
            forbidden = [ip for ip in resolved if is_forbidden_ip(ip)]
            if forbidden:  # reject-if-any: one bad candidate poisons the host
                raise FetchError(
                    "private_ip",
                    f"refusing {hop.hostname}: forbidden dial candidates {forbidden}",
                    final_url=current,
                )
            pinned_ip = resolved[0]

            connection = _connect(hop, pinned_ip, timeout)
            try:
                connection.request(
                    "GET",
                    hop.request_target,
                    headers={
                        "Host": hop.host_header,
                        "Accept-Encoding": "gzip, deflate",  # only advertised codes
                        "Connection": "close",
                    },
                )
                response = connection.getresponse()
                if 200 <= response.status < 300:
                    content = _read_body(response, response.headers.get("Content-Encoding"), cap)
                    return FetchResult(
                        status="ok",
                        content=content,
                        content_type=response.headers.get("Content-Type", "").strip(),
                        http_status=response.status,
                        final_url=current,
                    )
                if response.status in _REDIRECTABLE:
                    location = response.headers.get("Location", "").strip()
                    if not location:
                        raise FetchError(
                            None,
                            f"redirect without Location: {current}",
                            http_status=response.status,
                        )
                    if budget <= 0:
                        raise FetchError(
                            "redirect_too_many",
                            f"more than {settings.max_redirects} redirects",
                            http_status=response.status,
                            final_url=current,
                        )
                    budget -= 1
                    try:
                        current = urllib.parse.urljoin(current, location)
                    except ValueError as exc:
                        raise FetchError(
                            None,
                            f"unusable redirect Location {location!r}",
                            http_status=response.status,
                        ) from exc
                    continue  # finally closes this; the next hop re-validates fully

                raise FetchError(
                    _error_code_for_status(response.status),
                    f"HTTP {response.status} for {current}",
                    http_status=response.status,
                )
            finally:
                connection.close()
    except TimeoutError as exc:  # socket.timeout == TimeoutError (py ≥ 3.10)
        raise FetchError(
            "timeout",
            f"fetch timed out after {settings.fetch_timeout_s}s: {exc}",
            final_url=current,
        ) from exc
    except (socket.gaierror, UnicodeError) as exc:
        raise FetchError("dns_failure", f"could not resolve: {exc}", final_url=current) from exc
    except (zlib.error, http.client.HTTPException, OSError) as exc:
        # Refused/reset/unreachable, TLS failures, corrupt streams — real
        # failures that fit no exact token; fetch_failed + error_code=None.
        raise FetchError(None, f"network failure: {exc!r}", final_url=current) from exc


def _fetch_sync(url: str, settings: Settings) -> FetchResult:
    """Blocking fetch core: run the guarded loop, fold FetchError into a result.

    Exceptions other than FetchError are bugs and propagate on purpose.
    """
    try:
        return _hop_loop(url, settings)
    except FetchError as exc:
        return FetchResult(
            status="fetch_failed",
            content=b"",
            content_type="",
            http_status=exc.http_status,
            final_url=exc.final_url or url,
            error_code=exc.error_code,
        )


async def fetch(url: str, settings: Settings) -> FetchResult:
    """Fetch ``url`` through the SSRF-guarded channel (async surface).

    The blocking transport runs in a worker thread so callers stay on the
    event loop; the timeout/cap/redirect guards all apply unchanged.
    """
    return await asyncio.to_thread(_fetch_sync, url, settings)
