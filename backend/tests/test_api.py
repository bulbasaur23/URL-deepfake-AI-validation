"""Task 14: Scan API tests — submit, poll, rate limit, errors. All offline.

Reuses Task 12's offline pattern (``tests/test_jobs.py``): instrumented fake
engines (never real weights), a monkeypatched ``pipeline.fetcher.fetch``
serving canned HTML/PNG bytes, and a throwaway file-backed SQLite database +
media dir under ``tmp_path``. The app under test is the real ``create_app()``
with its lifespan; fake engines enter it by monkeypatching the registry seam
(``registry.get_engine``), exactly the seam the lifespan consults.

The 5 brief-pinned tests:

1. ``test_submit_202_and_roundtrip_shape`` — POST → ``202 {"scan_id": int}``,
   poll to ``done``, then assert EVERY §7.6 key (incl.
   ``per_modality.text.verdict``) plus the seam-wired DB verdicts
   (``Scan.verdict`` = the aggregate token verbatim).
2. ``test_invalid_url_422`` — §7.5 scheme/host violations → generic
   ``{"detail": "invalid url"}``; validation runs BEFORE rate limiting, so
   invalid submissions never consume rate budget.
3. ``test_unknown_scan_404`` — generic 404 payload, no internals.
4. ``test_rate_limit_429_after_30`` — 30 POSTs pass the window, the 31st
   gets the generic 429.
5. ``test_generic_error_payloads_only`` — 422/429/404 bodies are exactly
   ``{"detail": <token>}`` (incl. a non-integer scan id → 404).

Disclosed extras beyond the brief:

- ``test_get_polling_not_rate_limited`` — pins the POST-only scope (ruling 3):
  40 GETs against an exhausted submission window never 429.
- ``test_warnings_graceful_after_restart`` — warnings are manager memory
  (ruling 2): the live app reports the tokens, a fresh app on the same DB
  reports ``[]``.
"""

from __future__ import annotations

import io
import time
import urllib.parse
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.main import create_app
from app.settings import settings
from db.models import EngineResultORM, Scan, ScanItem
from db.session import get_session
from pipeline import fetcher
from pipeline.engines import registry
from pipeline.types import EngineResult, FetchResult, Item

_PAGE_URL = "http://page.test/article"
_IMAGE_URL = "http://page.test/pic.png"

# > 80 chars, clear English — past the extractor's block gate (canonical
# fixtures reused from tests/test_jobs.py; seeded langdetect is deterministic
# so the Hindi block is marked unsupported when the gate sees it).
_EN_1 = (
    "This opening paragraph describes the scan pipeline and carries enough "
    "ordinary filler text to comfortably pass the eighty character minimum "
    "in the content extractor tests"
)
_EN_2 = (
    "A second plain English block, also well past the extractor's minimum "
    "length, so the status-progression and degradation tests have two "
    "independent text items to work with"
)
_HINDI_TEXT = (
    "यह हिंदी पैराग्राफ एक समाचार लेख के बारे में है जो कृत्रिम बुद्धिमत्ता और "
    "मशीनी अनुवाद के विषय पर चर्चा करता है और इसमें पर्याप्त विस्तार है"
)


def _page_html(*paragraphs: str, image_src: str | None = None) -> bytes:
    """A minimal page whose <article> holds the given paragraphs (+1 img)."""
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    if image_src:
        body += f'<img src="{image_src}">'
    html = (
        f"<html><head><title>Test Page</title></head><body><article>{body}</article></body></html>"
    )
    return html.encode("utf-8")


def _png_bytes(width: int, height: int) -> bytes:
    """Deterministic PNG raster the fake image fetch can serve."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 240)).save(buffer, "PNG")
    return buffer.getvalue()


def _fake_fetch(image_src: str | None = None, paragraphs: tuple[str, ...] = (_EN_1, _EN_2)):
    """Canned page/image fetch factory (no network), Task 12's patch seam."""

    async def fetch(url: str, _settings: object) -> FetchResult:
        if url == _IMAGE_URL and image_src is not None:
            return FetchResult(status="ok", content=_png_bytes(300, 200), content_type="image/png")
        # The manager fetches Scan.url — stored AS SUBMITTED (scheme case and
        # fragments included), so match on the parsed identity, not the string.
        parts = urllib.parse.urlsplit(url)
        assert (parts.hostname, parts.path) == ("page.test", "/article"), f"unexpected fetch: {url}"
        return FetchResult(
            status="ok",
            content=_page_html(*paragraphs, image_src=image_src),
            content_type="text/html",
        )

    return fetch


class _FakeEngine:
    """Offline engine stub: one fixed reading per call, calls recorded."""

    def __init__(self, kind: str, score: float, confidence: float) -> None:
        self.id = f"fake-{kind}"
        self.version = "fake/1.0"
        self._kind = kind
        self._score = score
        self._confidence = confidence
        self.calls: list[Item] = []

    def supports(self, item: Item) -> bool:
        return item.kind == self._kind

    async def analyze(self, item: Item) -> EngineResult:
        self.calls.append(item)
        return EngineResult(
            engine_id=self.id,
            engine_version=self.version,
            score=self._score,
            confidence=self._confidence,
            rationale=f"fake {self._kind} reading",
            latency_ms=1,
        )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Real ``create_app()`` + lifespan, fake engines, throwaway DB + media."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    # 2 synthetic text + 1 authentic image → page verdict `mixed` (§11.13).
    engines = {
        "text": _FakeEngine("text", score=80.0, confidence=0.9),
        "image": _FakeEngine("image", score=10.0, confidence=0.7),
    }

    def fake_get_engine(modality: str, _settings: object) -> _FakeEngine:
        return engines[modality]

    monkeypatch.setattr(registry, "get_engine", fake_get_engine)
    monkeypatch.setattr(fetcher, "fetch", _fake_fetch(image_src=_IMAGE_URL))
    with TestClient(create_app()) as test_client:
        yield test_client


def _poll_until_terminal(client: TestClient, scan_id: int, timeout_s: float = 15.0) -> dict:
    """GET-polled §7.6 payload until the scan reaches done | failed."""
    deadline = time.monotonic() + timeout_s
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/scan/{scan_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.05)
    pytest.fail(f"scan {scan_id} never reached a terminal status: {payload}")


def _db_snapshot(scan_id: int) -> tuple[Scan, list[ScanItem], list[EngineResultORM]]:
    """Read one scan's row, items (id order), engine results (id order)."""
    session = get_session(settings)
    try:
        scan = session.get(Scan, scan_id)
        assert scan is not None
        items = list(
            session.scalars(
                select(ScanItem).where(ScanItem.scan_id == scan_id).order_by(ScanItem.id)
            )
        )
        results = list(
            session.scalars(
                select(EngineResultORM)
                .join(ScanItem, EngineResultORM.item_id == ScanItem.id)
                .where(ScanItem.scan_id == scan_id)
                .order_by(EngineResultORM.id)
            )
        )
    finally:
        session.close()
    return scan, items, results


def test_submit_202_and_roundtrip_shape(client: TestClient) -> None:
    """POST → 202; poll to done; every §7.6 key present + seam-wired verdicts."""
    post = client.post("/api/scan", json={"url": "HTTP://Page.test/article#section"})
    assert post.status_code == 202
    body = post.json()
    assert set(body) == {"scan_id"}
    scan_id = body["scan_id"]
    assert isinstance(scan_id, int)

    payload = _poll_until_terminal(client, scan_id)
    assert payload["status"] == "done"
    # §7.6 top-level keys, exactly.
    assert set(payload) == {
        "status",
        "summary",
        "confidence",
        "verdict",
        "per_modality",
        "meta",
        "items",
        "warnings",
        "engines_used",
    }
    # 2 synthetic text + 1 authentic image → mixed (§11.13: both present).
    assert payload["verdict"] == "mixed"
    assert payload["confidence"] == 0.9  # max confidence among synthetic items
    assert payload["summary"] == (
        "2 of 3 items AI-generated (2 text blocks) · 1 of 3 human · 0 unclear"
    )
    assert set(payload["per_modality"]) == {"text", "images"}
    text = payload["per_modality"]["text"]
    assert set(text) == {"verdict", "counts"}
    assert set(text["counts"]) == {"synthetic", "unclear", "authentic"}
    assert text["verdict"] == "synthetic"
    assert text["counts"] == {"synthetic": 2, "unclear": 0, "authentic": 0}
    images = payload["per_modality"]["images"]
    assert images["verdict"] == "authentic"
    assert images["counts"] == {"synthetic": 0, "unclear": 0, "authentic": 1}
    assert set(payload["meta"]) == {"title", "domain", "scanned_at"}
    assert payload["meta"]["title"] == "Test Page"
    assert payload["meta"]["domain"] == "page.test"
    from datetime import datetime

    datetime.fromisoformat(payload["meta"]["scanned_at"])  # ISO-parseable
    assert payload["warnings"] == []
    assert payload["engines_used"] == ["fake-image", "fake-text"]

    assert [item["kind"] for item in payload["items"]] == ["text", "text", "image"]
    for item in payload["items"]:
        assert set(item) == {"kind", "label", "excerpt", "thumb", "status", "engines"}
        assert item["status"] == "ok"
        assert item["excerpt"] == item["label"]  # best persisted text (no text column)
        for engine in item["engines"]:
            assert set(engine) == {"id", "score", "confidence", "rationale"}
    text_items = [item for item in payload["items"] if item["kind"] == "text"]
    image_items = [item for item in payload["items"] if item["kind"] == "image"]
    for item in text_items:
        assert item["engines"] == [
            {
                "id": "fake-text",
                "score": 80.0,
                "confidence": 0.9,
                "rationale": "fake text reading",
            }
        ]
        assert item["thumb"] is None
    assert image_items[0]["engines"] == [
        {
            "id": "fake-image",
            "score": 10.0,
            "confidence": 0.7,
            "rationale": "fake image reading",
        }
    ]

    # DB side: the seam filled verdicts (ruling 1) — aggregate token verbatim.
    scan_row, item_rows, results = _db_snapshot(scan_id)
    assert scan_row.status == "done"
    assert scan_row.verdict == "mixed"  # never a per-item band token
    assert scan_row.confidence == 0.9
    assert [row.verdict for row in item_rows] == ["synthetic", "synthetic", "authentic"]
    assert [row.confidence for row in item_rows] == [0.9, 0.9, 0.7]
    assert scan_row.url == "HTTP://Page.test/article#section"  # stored as submitted
    assert scan_row.url_canonical == "http://page.test/article"  # computed pre-insert
    assert len(results) == 3
    for payload_item, row in zip(payload["items"], item_rows, strict=True):
        expected_thumb = f"/media/{row.id}" if row.kind == "image" else None
        assert payload_item["thumb"] == expected_thumb  # forward-compat with T15
        assert payload_item["label"] == row.label


def test_invalid_url_422(client: TestClient) -> None:
    """§7.5 violations → generic 422; validation precedes rate limiting."""
    for url in ["ftp://x.io/a", "file:///etc/passwd", "not a url", "https://", "http:///no"]:
        response = client.post("/api/scan", json={"url": url})
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid url"}
    # Malformed/absent bodies are invalid urls too (generic payloads only).
    assert client.post("/api/scan", json={"nope": 1}).json() == {"detail": "invalid url"}
    no_body = client.post("/api/scan", content=b"", headers={"content-type": "application/json"})
    assert no_body.status_code == 422
    assert no_body.json() == {"detail": "invalid url"}
    # Binding flow order (ruling 5): validate BEFORE the limiter, so invalid
    # submissions never consume budget — 30+ invalid POSTs still 422, not 429.
    for _ in range(30):
        response = client.post("/api/scan", json={"url": "ftp://x.io/a"})
        assert response.status_code == 422


def test_unknown_scan_404(client: TestClient) -> None:
    """Unknown scan id → 404 generic payload, no internals."""
    response = client.get("/api/scan/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "not found"}


def test_rate_limit_429_after_30(client: TestClient) -> None:
    """30 submissions fit the window; the 31st POST gets the generic 429."""
    for i in range(30):
        response = client.post("/api/scan", json={"url": f"{_PAGE_URL}?p={i}"})
        assert response.status_code == 202, (i, response.json())
    response = client.post("/api/scan", json={"url": _PAGE_URL})
    assert response.status_code == 429
    assert response.json() == {"detail": "rate limited"}


def test_generic_error_payloads_only(client: TestClient) -> None:
    """Every client error body is exactly {"detail": <generic token>}."""
    assert client.post("/api/scan", json={"url": "ftp://x"}).json() == {"detail": "invalid url"}
    for i in range(30):
        assert client.post("/api/scan", json={"url": f"{_PAGE_URL}?i={i}"}).status_code == 202
    assert client.post("/api/scan", json={"url": _PAGE_URL}).json() == {"detail": "rate limited"}
    unknown = client.get("/api/scan/424242")
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "not found"}
    # A non-integer scan id is an unknown scan — generic 404, not FastAPI's
    # verbose validation payload (GATE 4: generic client errors).
    for bad_id in ("not-a-number", "1.5"):
        response = client.get(f"/api/scan/{bad_id}")
        assert response.status_code == 404
        assert response.json() == {"detail": "not found"}


def test_get_polling_not_rate_limited(client: TestClient) -> None:
    """Disclosed extra (ruling 3): the cap is POST-only — GET polling is free.

    The submission window is exhausted (30 POSTs), then 40 GETs fly against
    the created scans: every one must answer (200), never 429 — a 30/min GET
    cap would 429 legitimate 2 s polling of a 120 s scan.
    """
    for i in range(30):
        assert client.post("/api/scan", json={"url": f"{_PAGE_URL}?i={i}"}).status_code == 202
    for scan_id in range(1, 41):  # scans 1..30 exist; 31..40 are unknown
        response = client.get(f"/api/scan/{scan_id}")
        assert response.status_code != 429
        assert response.status_code in (200, 404)


def test_warnings_graceful_after_restart(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disclosed extra (ruling 2): warnings are manager memory.

    The live app reports the in-memory tokens; a fresh app (new manager) on
    the same DB gracefully reports ``[]`` — a restart drops them by design.
    """
    monkeypatch.setattr(fetcher, "fetch", _fake_fetch(paragraphs=(_EN_1, _HINDI_TEXT)))
    scan_id = client.post("/api/scan", json={"url": _PAGE_URL}).json()["scan_id"]
    payload = _poll_until_terminal(client, scan_id)
    assert payload["warnings"] == ["language_limited", "x_escaped_analysis"]
    with TestClient(create_app()) as restarted:
        restarted_payload = restarted.get(f"/api/scan/{scan_id}").json()
    assert restarted_payload["status"] == "done"
    assert restarted_payload["warnings"] == []  # graceful empty after restart
