"""Task 7: image pipeline tests (GATE 2) — all offline.

Two offline levers, reused verbatim from ``tests/test_fetcher.py``:

1. "patched dial" — the fetcher seam ``is_forbidden_ip`` is monkeypatched and
   the loopback guard lifted *only* for ``127.0.0.1`` (every other
   private/loopback/metadata range still rejects), because the local fixture
   server binds there. ``process_all`` never bypasses the fetcher, so every
   fetch here travels the fully guarded channel.
2. a real loopback ``ThreadingHTTPServer`` answering per route from the
   inline PNG/JPEG/GIF/WebP byte tables generated just below (deterministic
   PIL fixtures: same arguments → same bytes on every run), plus deliberate
   fake/corrupt bodies. ``/robots.txt`` answers 404 (permissive-by-policy)
   until a test flips the flag, which pins the robots mapping.

The seven brief-pinned tests: srcset best-variant resolution, magic-byte
validation, natural-size skip, content-hash dedupe, the 20-image cap, the
Snapshot row with its 30-day TTL, and the unfetchable mapping — plus the
GATE-2 downscale pin, the robots mapping, and format coverage (JPEG/GIF/WebP).
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import io
import ipaddress
import threading
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image, features
from sqlalchemy import select

from app.settings import Settings
from db.models import Base, Scan, ScanItem, Snapshot
from db.session import get_session
from pipeline import fetcher
from pipeline.fetcher import fetch
from pipeline.images import process_all  # Task 7 module (RED until it exists)
from pipeline.types import ImageRef

SNAPSHOT_TTL = timedelta(days=30)  # spec: snapshot_ttl_s default = 2_592_000 s


# --- inline fixture byte tables (deterministic: PIL re-runs give same bytes) ----


def _png(w: int, h: int, seed: int) -> bytes:
    color = ((37 * seed) % 255 + 1, (91 * seed) % 255 + 1, (53 * seed) % 255 + 1)
    return _saved(Image.new("RGB", (w, h), color), "PNG")


def _jpeg(w: int, h: int, seed: int) -> bytes:
    color = ((13 * seed) % 255 + 1, (29 * seed) % 255 + 1, (71 * seed) % 255 + 1)
    return _saved(Image.new("RGB", (w, h), color), "JPEG")


def _gif(w: int, h: int, seed: int) -> bytes:
    color = ((17 * seed) % 255 + 1, (43 * seed) % 255 + 1, (83 * seed) % 255 + 1)
    return _saved(Image.new("RGB", (w, h), color), "GIF")


def _saved(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


_WEBP_SUPPORTED = features.check("webp")  # graceful-skip gate for the WebP table
_WEBP_OK = _saved(Image.new("RGB", (150, 150), (9, 9, 200)), "WEBP") if _WEBP_SUPPORTED else b""
_PNG_50 = _png(50, 50, 3)  # valid file with both natural sides < image_min_px
_PNG_THIN = _png(50, 300, 4)  # ONE natural side < 100 → still skipped (either side)
_PNG_300 = _png(300, 200, 5)  # highest-width srcset entry (must win)
_PNG_100 = _png(100, 100, 6)  # lower-width srcset entry (never fetched)
_PNG_B40 = _png(40, 40, 7)  # no-widths entries; not the last one (never fetched)
_PNG_B120 = _png(200, 120, 8)  # no-widths entries; last → winner (both sides ≥ 100)
_PNG_OK = _png(220, 120, 9)  # ordinary ok image
_PNG_GIANT = _png(3200, 1000, 10)  # downscale pin: longest side forced to 1600
_PNG_IDENTICAL = _png(180, 180, 11)  # served under two URLs and re-served once
_FAKE_PNG = b"<html>plain text masquerading as a png</html>"
_JPEG_OK = _saved(Image.new("RGB", (160, 200), (200, 30, 60)), "JPEG")
_GIF_OK = _saved(Image.new("RGB", (140, 140), (30, 200, 120)), "GIF")
_CAP_PNGS = {n: _png(128, 128, 100 + n) for n in range(1, 25)}  # 24 distinct bodies

_ROBOTS_DISALLOW = b"User-agent: *\nDisallow: /\n"

_ROUTES: dict[str, bytes] = {
    "/pixels/a50.png": _PNG_50,
    "/pixels/a300.png": _PNG_300,
    "/pixels/a100.png": _PNG_100,
    "/pixels/b40.png": _PNG_B40,
    "/pixels/b120.png": _PNG_B120,
    "/pixels/ok.png": _PNG_OK,
    "/pixels/giant.png": _PNG_GIANT,
    "/pixels/fake.png": _FAKE_PNG,
    "/pixels/icon50.png": _png(50, 50, 21),
    "/pixels/icon-thin.png": _PNG_THIN,
    "/pixels/same-a.png": _PNG_IDENTICAL,
    "/pixels/same-b.png": _PNG_IDENTICAL,
    "/pixels/photo.jpeg": _JPEG_OK,
    "/pixels/picture.gif": _GIF_OK,
}
if _WEBP_OK:
    _ROUTES["/pixels/render.webp"] = _WEBP_OK
for n, body in _CAP_PNGS.items():
    _ROUTES[f"/pixels/cap{n}.png"] = body

HITS: list[str] = []  # every GET path the fixture origin saw, in request order
_ROBOTS_DISALLOW_ALL = {"on": False}  # flipped per-test for the robots mapping pin


class _Server(http.server.ThreadingHTTPServer):
    """Fixture origin; every route answers from the tables above."""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        """Silence per-request logging; assertions live in the tests."""

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        HITS.append(path)
        if path == "/robots.txt":
            if _ROBOTS_DISALLOW_ALL["on"]:
                self._send(200, _ROBOTS_DISALLOW)
            else:
                self._send(404, b"no robots here")  # http_4xx consult → permissive
            return
        if path == "/status/500":
            self._send(500, b"fixture failure")
            return
        body = _ROUTES.get(path)
        if body is None:
            self._send(404, b"no such fixture")
            return
        self._send(200, body)

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "image/jpeg" if status == 200 else "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def allow_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched dial for the local fixtures: guard lifted for loopback only.

    ``http://127.0.0.1:<port>`` is rejected by the production guard, so this
    wrapped seam keeps every other rejection identical to production while
    letting the fixture site be dialed (same lever as test_fetcher).
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
def site(allow_loopback: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Loopback origin URL; resets the hit log and the robots flag per test."""
    HITS.clear()
    monkeypatch.setitem(_ROBOTS_DISALLOW_ALL, "on", False)
    httpd = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Brief-default Settings with a throwaway in-memory DB + tmp media dir."""
    return Settings(db_path=":memory:", media_dir=str(tmp_path / "media"))


@pytest.fixture()
def mem_db(settings: Settings) -> Iterator[None]:
    """Clean tables around each test on the shared in-memory engine (T2 style)."""
    session = get_session(settings)
    try:
        Base.metadata.drop_all(session.get_bind())
        Base.metadata.create_all(session.get_bind())
        yield
    finally:
        Base.metadata.drop_all(session.get_bind())
        session.close()


def _new_scan(settings: Settings) -> int:
    """Insert the Scan row that ScanItem FKs hang from; return its id."""
    session = get_session(settings)
    try:
        scan = Scan(url="http://scan.test/page", url_canonical="http://scan.test/page")
        session.add(scan)
        session.commit()
        return int(scan.id)
    finally:
        session.close()


def _image_hits() -> list[str]:
    """Fixture requests, robots.txt consults filtered out (one per fetch)."""
    return [hit for hit in HITS if hit != "/robots.txt"]


def _scan_items(settings: Settings) -> list[tuple[str, str, str | None, str]]:
    """All ScanItem rows as (kind, source_ref, label, status), insertion order."""
    session = get_session(settings)
    try:
        return [
            (row.kind, row.source_ref, row.label, row.status)
            for row in session.scalars(select(ScanItem).order_by(ScanItem.id)).all()
        ]
    finally:
        session.close()


def _snapshots(settings: Settings) -> list[tuple[str, str, datetime, datetime]]:
    """All Snapshot rows as (path, mime, created_at, expires_at)."""
    session = get_session(settings)
    try:
        return [
            (row.path, row.mime, row.created_at, row.expires_at)
            for row in session.scalars(select(Snapshot).order_by(Snapshot.id)).all()
        ]
    finally:
        session.close()


def _snapshot_paths(settings: Settings) -> list[str]:
    """Just the stored snapshot paths."""
    return [row[0] for row in _snapshots(settings)]


def _run(refs: list[ImageRef], settings: Settings, scan_id: int):
    """The pinned call shape: the real Task-3 fetch through process_all."""
    return asyncio.run(process_all(refs, fetch, settings, scan_id))


def _ref(site: str, path: str, label: str, index: int) -> ImageRef:
    """One plain-src ImageRef over the fixture site."""
    return ImageRef(
        label=label,
        url_candidates=[(f"{site}{path}", None)],
        source_ref=f"article:img[{index}]",
    )


# --- the brief's pinned tests ----------------------------------------------------


def test_best_srcset_variant_selected(site: str, settings: Settings, mem_db: None) -> None:
    """Highest-width candidate wins; the last entry wins when no widths exist."""
    scan_id = _new_scan(settings)
    with_widths = ImageRef(
        label="hero",
        url_candidates=[
            (f"{site}/pixels/a50.png", None),
            (f"{site}/pixels/a300.png", 300),
            (f"{site}/pixels/a100.png", 100),
        ],
        source_ref="article:img[0]",
    )
    without_widths = ImageRef(
        label="stamp",
        url_candidates=[
            (f"{site}/pixels/b40.png", None),
            (f"{site}/pixels/b120.png", None),
        ],
        source_ref="article:img[1]",
    )
    items = _run([with_widths, without_widths], settings, scan_id)

    assert [item.status for item in items] == ["ok", "ok"]
    assert _image_hits() == ["/pixels/a300.png", "/pixels/b120.png"]  # one fetch per ref
    assert Image.open(items[0].image_path).size == (300, 200)  # highest width, not a100
    assert Image.open(items[1].image_path).size == (200, 120)  # last entry, not b40


def test_magic_bytes_reject_fake_png(site: str, settings: Settings, mem_db: None) -> None:
    """Text bytes named .png are rejected after the ok fetch → unsupported."""
    scan_id = _new_scan(settings)
    items = _run([_ref(site, "/pixels/fake.png", "pretend", 0)], settings, scan_id)

    assert len(items) == 1
    assert items[0].status == "unsupported"
    assert items[0].image_path is None and items[0].mime is None
    assert _scan_items(settings) == [("image", "article:img[0]", "pretend", "unsupported")]
    assert _snapshots(settings) == []  # nothing was snapshotted, nothing on disk
    assert list((Path(settings.media_dir)).glob("*.jpg")) == []


def test_small_icon_skipped(site: str, settings: Settings, mem_db: None) -> None:
    """Natural size < image_min_px on either side → no Item, nothing stored."""
    scan_id = _new_scan(settings)
    items = _run(
        [
            _ref(site, "/pixels/icon50.png", "icon", 0),
            _ref(site, "/pixels/icon-thin.png", "thin", 1),
        ],
        settings,
        scan_id,
    )

    assert items == []  # skipped pipeline: no Items at all
    assert _image_hits() == ["/pixels/icon50.png", "/pixels/icon-thin.png"]  # fetched first
    assert _scan_items(settings) == [] and _snapshots(settings) == []


def test_dedupe_by_content_hash(site: str, settings: Settings, mem_db: None) -> None:
    """sha256 of the re-encoded bytes keeps the FIRST occurrence only.

    Same body behind two URLs, plus the first URL repeated again on the page:
    three fetches still yield exactly one Item / one file / one Snapshot row.
    """
    scan_id = _new_scan(settings)
    refs = [
        ImageRef(
            label="hero",
            url_candidates=[(f"{site}/pixels/same-a.png", None)],
            source_ref="article:img[0]",
        ),
        ImageRef(
            label="dupe",
            url_candidates=[(f"{site}/pixels/same-b.png", None)],
            source_ref="article:img[1]",
        ),
        ImageRef(
            label="repeat",
            url_candidates=[(f"{site}/pixels/same-a.png", None)],
            source_ref="article:img[2]",
        ),
    ]
    items = _run(refs, settings, scan_id)

    assert _image_hits() == ["/pixels/same-a.png", "/pixels/same-b.png", "/pixels/same-a.png"]
    assert len(items) == 1  # dedupe losers produce NO Item
    assert items[0].source_ref == "article:img[0]" and items[0].label == "hero"
    assert [status for _, _, _, status in _scan_items(settings)] == ["ok"]
    assert len(_snapshots(settings)) == 1  # the first occurrence's row — no second row
    assert len(_snapshot_paths(settings)) == 1  # ... and exactly one file on disk


def test_cap_20_marks_too_large(site: str, settings: Settings, mem_db: None) -> None:
    """max_images=20: positions 1..20 process; 21+ mark too_large, never fetched."""
    scan_id = _new_scan(settings)
    refs = [
        ImageRef(
            label=f"cap {n}",
            url_candidates=[(f"{site}/pixels/cap{n}.png", None)],
            source_ref=f"article:img[{n - 1}]",
        )
        for n in range(1, 23)  # 22 refs, two beyond the cap
    ]
    items = _run(refs, settings, scan_id)

    assert len(items) == 22  # document order kept for the excess Items
    assert [item.status for item in items[:20]] == ["ok"] * 20
    assert [item.status for item in items[20:]] == ["too_large", "too_large"]
    assert all(item.image_path is None and item.mime is None for item in items[20:])
    hit_positions = {f"/pixels/cap{n}.png" for n in range(1, 23)} & set(_image_hits())
    assert hit_positions == {f"/pixels/cap{n}.png" for n in range(1, 21)}  # NO fetch past 20
    assert len(_snapshots(settings)) == 20
    assert [status for _, _, _, status in _scan_items(settings)] == ["ok"] * 20 + ["too_large"] * 2


def test_snapshot_written_with_ttl_30d(
    site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ok image: file at sha256 path, Snapshot row with +30 days, exact TTL.

    Uses the default ``media_dir="media/"`` (chdir'd into tmp) so the pinned
    content-addressed row path form ``media/<sha256>.jpg`` is asserted verbatim.
    """
    monkey_settings = Settings(db_path=":memory:", media_dir="media/")
    monkeypatch.chdir(tmp_path)
    session = get_session(monkey_settings)
    try:
        Base.metadata.drop_all(session.get_bind())
        Base.metadata.create_all(session.get_bind())
    finally:
        session.close()
    scan_id = _new_scan(monkey_settings)
    items = _run([_ref(site, "/pixels/ok.png", "main", 0)], monkey_settings, scan_id)

    assert len(items) == 1 and items[0].status == "ok"
    row = _snapshots(monkey_settings)[0]
    path, mime, created_at, expires_at = row
    assert items[0].image_path == path and items[0].mime == mime
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == Path(path).stem
    assert mime == "image/jpeg"
    assert expires_at == created_at + SNAPSHOT_TTL  # Task-2 semantics, Python-side
    assert expires_at - datetime.now(UTC).replace(tzinfo=None) <= SNAPSHOT_TTL
    assert Image.open(path).size == (220, 120)  # re-encoded JPEG, natural size kept
    session = get_session(monkey_settings)
    try:
        loaded = session.scalars(select(Snapshot)).one()
        assert loaded.item.scan_id == scan_id  # FK chain Scan → ScanItem → Snapshot
    finally:
        session.close()


def test_unfetchable_image_gets_fetch_failed(site: str, settings: Settings, mem_db: None) -> None:
    """Non-2xx image fetches (404, 500) → fetch_failed (NOT blocked_by_site)."""
    scan_id = _new_scan(settings)
    for index, path in enumerate(["/pixels/absent.png", "/status/500"]):
        items = _run([_ref(site, path, f"broken {index}", index)], settings, scan_id)
        assert len(items) == 1
        assert items[0].status == "fetch_failed", path
        assert items[0].image_path is None and items[0].mime is None
    assert [status for _, _, _, status in _scan_items(settings)] == ["fetch_failed"] * 2
    assert _snapshots(settings) == []


# --- supporting pins (controller rulings / GATE 2 ) --------------------------------


def test_robots_disallow_maps_blocked_by_site(
    site: str, settings: Settings, mem_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """robots_disallowed ⇔ blocked_by_site: the ONLY error that maps that way."""
    monkeypatch.setitem(_ROBOTS_DISALLOW_ALL, "on", True)
    scan_id = _new_scan(settings)
    items = _run([_ref(site, "/pixels/ok.png", "hero", 0)], settings, scan_id)

    assert len(items) == 1 and items[0].status == "blocked_by_site"
    assert items[0].image_path is None
    assert _snapshots(settings) == [] and list(Path(settings.media_dir).rglob("*")) == []
    assert [status for _, _, _, status in _scan_items(settings)] == ["blocked_by_site"]


def test_downscale_long_side_1600(site: str, settings: Settings, mem_db: None) -> None:
    """GATE 2 memory bound: a 3200x1000 input re-encodes to 3200→1600x500.

    Aspect kept, shrink-only, and applied BEFORE the content hash — the
    snapshot digest is of the bounded re-encoded bytes.
    """
    scan_id = _new_scan(settings)
    items = _run([_ref(site, "/pixels/giant.png", "giant", 0)], settings, scan_id)

    assert len(items) == 1 and items[0].status == "ok"
    assert (
        Path(items[0].image_path).stem
        == hashlib.sha256(Path(items[0].image_path).read_bytes()).hexdigest()
    )
    assert Image.open(items[0].image_path).size == (1600, 500)


def test_jpeg_gif_accepted_and_reencoded_to_jpeg(
    site: str, settings: Settings, mem_db: None
) -> None:
    """Accepted raster families (JPEG, GIF) all re-encode to image/jpeg."""
    scan_id = _new_scan(settings)
    items = _run(
        [_ref(site, "/pixels/photo.jpeg", "shot", 0), _ref(site, "/pixels/picture.gif", "art", 1)],
        settings,
        scan_id,
    )
    assert [item.status for item in items] == ["ok", "ok"]
    assert [item.mime for item in items] == ["image/jpeg", "image/jpeg"]
    assert Image.open(items[0].image_path).size == (160, 200)
    assert Image.open(items[1].image_path).size == (140, 140)
    assert len({Path(item.image_path).name for item in items}) == 2  # distinct digests


@pytest.mark.skipif(not _WEBP_SUPPORTED, reason="Pillow built without WebP support")
def test_webp_served_ok(site: str, settings: Settings, mem_db: None) -> None:
    """WebP (RIFF…WEBP) validates and re-encodes to image/jpeg like the rest."""
    scan_id = _new_scan(settings)
    items = _run([_ref(site, "/pixels/render.webp", "render", 0)], settings, scan_id)

    assert len(items) == 1 and items[0].status == "ok"
    assert items[0].mime == "image/jpeg"
    assert Image.open(items[0].image_path).size == (150, 150)
