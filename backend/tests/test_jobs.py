"""Task 12: ScanJobManager tests — queue, deadline, per-item resilience. All offline.

Every engine below is an instrumented fake driven by a per-call script (never
loads real weights), the page fetch is monkeypatched on the
``pipeline.fetcher.fetch`` seam, and each test owns a throwaway file-backed
SQLite database + media dir under ``tmp_path`` — no shared in-memory engine
state, no network.

The seven brief-pinned tests:

1. ``test_status_progression_persisted`` — transitions exactly
   queued→fetching→parsing→analyzing→aggregating→done, written between stages
   (observed both by a recorder on the write seam and by mid-flight DB reads).
2. ``test_engine_fault_degrades_item_ships_partial`` — EngineFault on one
   item degrades JUST that item; the partial report still ships as done.
3. ``test_deadline_gone_mid_scan_partial_done`` — the wall clock cuts a slow
   engine: ≥1 analyzed item → done + ``deadline_exceeded``; stragglers
   become ``unavailable``.
4. ``test_deadline_before_any_analysis_fails_timeout`` — zero analyzed →
   ``failed`` with the closed fetch-vocab token ``timeout``.
5. ``test_unsupported_items_skip_engines`` — language-gated text never
   reaches an engine (call count 0) and is persisted ``unsupported``.
6. ``test_concurrency_cap_serializes_second_scan`` — the semaphore
   serializes scans (cap monkeypatched to 1) and ``queue_depth`` tracks
   queued+running.
7. ``test_fetch_failed_scan_fails_with_error_code`` — a ``robots_disallowed``
   fetch short-circuits: failed + the fetch token verbatim, no later stages.

Plus one disclosed integration test (``test_image_rows_bind_engine_results``)
running the REAL Task 7 image pipeline so the positional row zip (controller
ruling 1) is proven against real persisted rows, and a real ``langid`` gate
(test 5) so the unsupported marker is the genuine Task 6 signal.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import select

from app.settings import Settings
from db.models import EngineResultORM, Scan, ScanItem, Snapshot
from db.session import get_session
from jobs.manager import ScanJobManager
from pipeline import fetcher
from pipeline.engines import registry
from pipeline.engines.base import EngineFault
from pipeline.types import EngineResult, FetchResult, Item

pytestmark = pytest.mark.asyncio

_PAGE_URL = "http://page.test/article"

# > 80 chars, clear English — comfortably past the extractor's block gate
# (canonical fixtures reused from tests/test_langid.py; seeded langdetect is
# deterministic, so the gate marks exactly the Hindi block).
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


class _FakeEngine:
    """Instrumented offline engine: per-call script + call recording.

    Script entries (per analyze call, in call order): ``None`` → default
    reading; ``float``/``int`` → sleep that long, then the default reading;
    an ``Exception`` instance → raise it (the degrade paths); an
    ``EngineResult`` → return it verbatim.
    """

    def __init__(self, kind: str = "text", script: tuple[object, ...] = ()) -> None:
        self.id = f"fake-{kind}"
        self.version = "fake/1.0"
        self._kind = kind
        self._script = list(script)
        self.calls: list[Item] = []
        self.intervals: list[tuple[float, float]] = []
        self.depth_samples: list[int] = []
        self.statuses_seen: list[object] = []
        # Optional hooks the tests set after construction:
        self.probe: Callable[[], object] | None = None  # DB status at analyze time
        self.manager: ScanJobManager | None = None  # queue_depth sampling

    def supports(self, item: Item) -> bool:
        return item.kind == self._kind

    async def analyze(self, item: Item) -> EngineResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        self.calls.append(item)
        if self.manager is not None:
            self.depth_samples.append(self.manager.queue_depth())
        if self.probe is not None:
            self.statuses_seen.append(self.probe())
        index = len(self.calls) - 1
        step = self._script[index] if index < len(self._script) else None
        if isinstance(step, (int, float)):
            await asyncio.sleep(step)
            step = None
        if isinstance(step, Exception):
            raise step
        self.intervals.append((started, loop.time()))
        if isinstance(step, EngineResult):
            return step
        return EngineResult(
            engine_id=self.id,
            engine_version=self.version,
            score=42.0,
            confidence=0.8,
            rationale="fake reading",
            latency_ms=1,
        )


class _ExplodingSupportsEngine(_FakeEngine):
    """A supports() that raises — a contract violation the manager must contain."""

    def supports(self, item: Item) -> bool:
        raise RuntimeError("supports exploded")


class _RecordingManager(ScanJobManager):
    """Test double: records every persisted status transition (ruling 3)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.transitions: list[str] = []

    def _write_scan(self, scan_id: int, **fields: object) -> None:
        status = fields.get("status")
        if isinstance(status, str):
            self.transitions.append(status)
        super()._write_scan(scan_id, **fields)


def _make_manager(
    tmp_path: Path, engines: dict[str, _FakeEngine], **overrides: object
) -> tuple[_RecordingManager, Settings]:
    """Manager on a throwaway file DB + media dir under tmp_path."""
    test_settings = Settings(
        db_path=str(tmp_path / "app.db"),
        media_dir=str(tmp_path / "media"),
        **overrides,  # type: ignore[arg-type]
    )
    return _RecordingManager(test_settings, engines), test_settings


def _html_fetch(
    *paragraphs: str, image_src: str | None = None
) -> Callable[[str, Settings], Awaitable[FetchResult]]:
    """Fake page fetch: always ok, always the same HTML (no network)."""

    async def fake_fetch(url: str, settings: Settings) -> FetchResult:
        return FetchResult(
            status="ok",
            content=_page_html(*paragraphs, image_src=image_src),
            content_type="text/html",
        )

    return fake_fetch


def _snapshot_db(
    settings: Settings, scan_id: int
) -> tuple[Scan | None, list[ScanItem], list[EngineResultORM]]:
    """Read one scan's row, items (id order), engine results (id order)."""
    session = get_session(settings)
    try:
        scan = session.get(Scan, scan_id)
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


async def _run_one(
    manager: ScanJobManager, url: str = _PAGE_URL, ip_hash: str = "test-ip-hash"
) -> int:
    """Submit one URL and block until the scan reaches a terminal status."""
    scan_id = await manager.submit(url, ip_hash)
    await manager.wait(scan_id)
    return scan_id


async def test_status_progression_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 3: transitions written between stages, exactly the pinned vocab."""
    engine = _FakeEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})
    state: dict[str, int] = {}
    seen_at_fetch: list[str | None] = []

    async def fake_fetch(url: str, settings: Settings) -> FetchResult:
        scan, _, _ = _snapshot_db(test_settings, state["scan_id"])
        seen_at_fetch.append(scan.status if scan else None)  # persisted BEFORE the stage ran
        return FetchResult(status="ok", content=_page_html(_EN_1, _EN_2), content_type="text/html")

    monkeypatch.setattr(fetcher, "fetch", fake_fetch)
    engine.probe = lambda: _snapshot_db(test_settings, state["scan_id"])[0].status  # type: ignore[union-attr]

    def _no_scan_time_resolution(*args: object, **kwargs: object) -> None:
        raise AssertionError("no engine may be resolved during a scan (ruling 8: boot-built)")

    monkeypatch.setattr(registry, "get_by_id", _no_scan_time_resolution)

    state["scan_id"] = await manager.submit("HTTP://Page.test/article#section", "test-ip-hash")
    scan, _, _ = _snapshot_db(test_settings, state["scan_id"])
    assert scan.status == "queued"  # persisted at submit, before the task ran
    assert manager.queue_depth() == 1  # submitted and running counts as depth 1
    await manager.wait(state["scan_id"])

    assert manager.transitions == ["fetching", "parsing", "analyzing", "aggregating", "done"]
    assert seen_at_fetch == ["fetching"]  # the fetching write landed before fetch ran
    assert engine.statuses_seen == ["analyzing", "analyzing"]  # both items analyzed mid-stage
    scan, items, results = _snapshot_db(test_settings, state["scan_id"])
    assert scan.status == "done"
    # Task 14 wired the aggregate seam: the fake engine scored both items 42.0
    # (→ unclear, confidence 0.8), an all-unclear page → insufficient_data with
    # max-analyzed confidence (§11.13) — never a per-item band token on Scan.
    assert scan.verdict == "insufficient_data" and scan.confidence == 0.8
    assert [row.verdict for row in items] == ["unclear", "unclear"]
    assert [row.confidence for row in items] == [0.8, 0.8]
    assert scan.title == "Test Page" and scan.domain == "page.test"
    assert scan.url == "HTTP://Page.test/article#section"  # stored as submitted
    assert scan.url_canonical == "http://page.test/article"  # fragment gone, host lowered
    assert scan.ip_hash == "test-ip-hash"  # stored as given (ruling 10)
    assert [row.status for row in items] == ["ok", "ok"]
    assert len(results) == 2  # one reading per text item
    assert manager.warnings_for(state["scan_id"]) == []


async def test_engine_fault_degrades_item_ships_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EngineFault degrades just that item; the partial report still ships."""
    engine = _FakeEngine("text", script=(None, EngineFault("fake engine exploded")))
    manager, test_settings = _make_manager(tmp_path, {"text": engine})
    monkeypatch.setattr(fetcher, "fetch", _html_fetch(_EN_1, _EN_2))

    scan_id = await _run_one(manager)

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "done" and scan.error_code is None  # never failed the scan
    assert [row.status for row in items] == ["ok", "degraded"]  # engine RAN → degraded
    assert len(results) == 1  # only the healthy item carries a reading
    assert results[0].item_id == items[0].id
    assert manager.warnings_for(scan_id) == ["x_escaped_analysis"]


async def test_deadline_gone_mid_scan_partial_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wall clock cuts a slow engine: done + deadline_exceeded + straggler unavailable."""
    engine = _FakeEngine("text", script=(None, 5.0))  # item 2 hangs past the clock
    manager, test_settings = _make_manager(
        tmp_path, {"text": engine}, scan_deadline_s=1, engine_timeout_s=10
    )
    monkeypatch.setattr(fetcher, "fetch", _html_fetch(_EN_1, _EN_2))

    started = time.monotonic()
    scan_id = await _run_one(manager)
    assert time.monotonic() - started < 3.0  # the 5 s engine sleep was cut at ~1 s

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "done" and scan.error_code is None  # ≥1 analyzed → done
    assert [row.status for row in items] == ["ok", "unavailable"]  # straggler never ran
    assert len(results) == 1 and results[0].item_id == items[0].id
    assert manager.warnings_for(scan_id) == ["deadline_exceeded", "x_escaped_analysis"]
    assert scan.verdict is None and scan.confidence is None
    # The clock fired mid-analyzing, so aggregating is never reached; the
    # deadline handler writes the terminal done straight from the handler.
    assert manager.transitions == ["fetching", "parsing", "analyzing", "done"]


async def test_deadline_before_any_analysis_fails_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero analyzed at wall-clock fire → failed with the vocab token timeout."""
    manager, test_settings = _make_manager(
        tmp_path, {"text": _FakeEngine("text")}, scan_deadline_s=1
    )

    async def slow_fetch(url: str, settings: Settings) -> FetchResult:
        await asyncio.sleep(2.0)  # the wall clock fires long before this returns
        return FetchResult(status="ok", content=_page_html(_EN_1), content_type="text/html")

    monkeypatch.setattr(fetcher, "fetch", slow_fetch)

    started = time.monotonic()
    scan_id = await _run_one(manager)
    assert time.monotonic() - started < 2.0

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "failed" and scan.error_code == "timeout"  # closed-vocab token
    assert items == [] and results == []  # zero analyzed, nothing persisted
    assert manager.transitions == ["fetching", "failed"]
    assert manager.warnings_for(scan_id) == ["deadline_exceeded"]


async def test_unsupported_items_skip_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Language-gated text reaches NO engine (0 calls) and persists unsupported."""
    engine = _FakeEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})
    monkeypatch.setattr(fetcher, "fetch", _html_fetch(_EN_1, _HINDI_TEXT))

    scan_id = await _run_one(manager)

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "done"
    assert [row.status for row in items] == ["ok", "unsupported"]  # gated, never dropped
    assert len(results) == 1 and results[0].item_id == items[0].id
    assert len(engine.calls) == 1  # the Hindi item produced zero engine calls
    assert manager.warnings_for(scan_id) == ["language_limited", "x_escaped_analysis"]


async def test_concurrency_cap_serializes_second_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap (monkeypatched to 1) serializes scans; queue_depth tracks it."""
    engine = _FakeEngine("text", script=(0.15, 0.15, 0.15))
    manager, test_settings = _make_manager(tmp_path, {"text": engine}, max_concurrent_scans=1)
    engine.manager = manager
    monkeypatch.setattr(fetcher, "fetch", _html_fetch(_EN_1))

    ids = [await manager.submit(f"http://page.test/s{i}", "ip") for i in range(3)]
    assert manager.queue_depth() == 3  # all submitted, none finished yet
    for scan_id in ids:
        await manager.wait(scan_id)

    scan, _, _ = _snapshot_db(test_settings, ids[0])
    assert scan.status == "done"
    assert len(engine.calls) == 3  # one text item per scan
    assert engine.depth_samples == [3, 2, 1]  # running + queued while the cap holds
    for (_, previous_end), (start, _) in zip(engine.intervals, engine.intervals[1:]):
        assert start >= previous_end  # strictly serialized: no overlapping analysis
    assert manager.queue_depth() == 0  # everything finished


async def test_fetch_failed_scan_fails_with_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fetch_failed result short-circuits: failed + the fetch token verbatim."""
    engine = _FakeEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})

    async def blocked_fetch(url: str, settings: Settings) -> FetchResult:
        return FetchResult(status="fetch_failed", error_code="robots_disallowed")

    monkeypatch.setattr(fetcher, "fetch", blocked_fetch)

    scan_id = await _run_one(manager)

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "failed"
    assert scan.error_code == "robots_disallowed"  # never "fetch_blocked_by_robots"
    assert items == [] and results == []  # no later stage ran
    assert engine.calls == []
    assert manager.transitions == ["fetching", "failed"]  # never reached parsing
    assert manager.warnings_for(scan_id) == []


async def test_image_rows_bind_engine_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disclosed extra: the REAL Task 7 pipeline + the ruling-1 positional zip.

    ``process_all`` persists its own kind=image rows; the manager must bind
    engine results through the persisted rows (positionally, document order)
    and never create a second image row — asserted here by row counts.
    """
    text_engine = _FakeEngine("text")
    image_engine = _FakeEngine("image")
    manager, test_settings = _make_manager(tmp_path, {"text": text_engine, "image": image_engine})

    async def fake_fetch(url: str, settings: Settings) -> FetchResult:
        if url == _PAGE_URL:
            html = _page_html(_EN_1, image_src="http://page.test/pic.png")
            return FetchResult(status="ok", content=html, content_type="text/html")
        return FetchResult(status="ok", content=_png_bytes(300, 200), content_type="image/png")

    monkeypatch.setattr(fetcher, "fetch", fake_fetch)

    scan_id = await _run_one(manager)

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "done"
    assert [row.kind for row in items] == ["text", "image"]  # document order
    assert len(items) == 2  # row counts: exactly one text row + one image row
    assert len(results) == 2  # one reading per item, each bound to the right row
    by_item = {result.item_id: result for result in results}
    assert by_item[items[0].id].engine_id == text_engine.id
    assert by_item[items[1].id].engine_id == image_engine.id
    session = get_session(test_settings)
    try:
        snapshots = list(session.scalars(select(Snapshot).where(Snapshot.item_id == items[1].id)))
    finally:
        session.close()
    assert len(snapshots) == 1  # no duplicate snapshot rows either
    assert snapshots[0].path.startswith(str(tmp_path / "media"))


async def test_wait_returns_when_task_dies_with_stuck_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Double fault (run() raises AND the failed-write raises) must not hang wait().

    The orchestrator's catch-all writes ``failed`` via ``_write_scan``; if even
    that write fails (simulated SQLite lock), the job task dies with the Scan
    row stuck mid-ladder. ``wait()`` must resolve through the TRACKED TASK in
    that case: awaiting it surfaces the chained terminal error (the original
    orchestrator error as ``__cause__``, never masked by the write failure)
    instead of polling a row that can never move. Wrapped in ``asyncio.timeout``
    so a RED regression fails loudly rather than hanging the suite.
    """
    engine = _FakeEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})

    def locked_write(scan_id: int, **fields: object) -> None:
        if fields.get("status") == "failed":
            raise RuntimeError("sqlite locked: terminal write refused")
        ScanJobManager._write_scan(manager, scan_id, **fields)  # non-terminal writes land

    monkeypatch.setattr(manager, "_write_scan", locked_write)  # instance-level override

    async def exploding_fetch(url: str, settings: Settings) -> FetchResult:
        raise RuntimeError("fetch exploded")

    monkeypatch.setattr(fetcher, "fetch", exploding_fetch)

    scan_id = await manager.submit(_PAGE_URL, "test-ip-hash")
    with pytest.raises(RuntimeError, match="terminal write failed") as excinfo:
        async with asyncio.timeout(2.0):
            await manager.wait(scan_id)
    # The write failure must not mask the original: chained as __cause__.
    assert str(excinfo.value.__cause__) == "fetch exploded"
    # Exactly the stuck state DB-polling alone can never resolve.
    scan, _, _ = _snapshot_db(test_settings, scan_id)
    assert scan.status == "fetching" and scan.error_code is None


async def test_canonicalize_malformed_url_falls_back_to_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``urlsplit('http://[::1/x')`` raises ValueError — submit must survive it.

    The unclosed IPv6 literal used to crash ``submit()`` BEFORE any Scan row
    existed. The fix falls back to the RAW URL as ``url_canonical`` (minimal
    degradation, documented in the docstring); Task 14's route validation
    still 422s this shape for HTTP callers — unchanged here.
    """
    engine = _FakeEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})

    async def blocked_fetch(url: str, settings: Settings) -> FetchResult:
        return FetchResult(status="fetch_failed", error_code="bad_scheme")

    monkeypatch.setattr(fetcher, "fetch", blocked_fetch)

    malformed = "http://[::1/x"  # unclosed IPv6 literal: urlsplit raises ValueError
    scan_id = await _run_one(manager, url=malformed)

    scan, items, _ = _snapshot_db(test_settings, scan_id)
    assert scan.url == malformed
    assert scan.url_canonical == malformed  # raw fallback — submit did not crash
    assert scan.status == "failed" and scan.error_code == "bad_scheme"
    assert items == []


async def test_supports_raise_marks_item_unavailable_scan_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising supports() degrades JUST the item; the scan still ships.

    supports() used to run OUTSIDE the per-item guard, so a raise failed the
    whole scan through the gather re-raise loop. Fixed: the raise is contained
    per item — the engine never ran, so the item ends ``unavailable`` (not
    ``degraded``) with zero engine calls, and the scan continues to done.
    """
    engine = _ExplodingSupportsEngine("text")
    manager, test_settings = _make_manager(tmp_path, {"text": engine})
    monkeypatch.setattr(fetcher, "fetch", _html_fetch(_EN_1, _EN_2))

    scan_id = await _run_one(manager)

    scan, items, results = _snapshot_db(test_settings, scan_id)
    assert scan.status == "done" and scan.error_code is None  # never failed the scan
    assert [row.status for row in items] == ["unavailable", "unavailable"]  # never ran
    assert len(results) == 0 and engine.calls == []  # zero engine calls
    # Zero analyzed → the wired aggregate seam reports insufficient_data (§7.4).
    assert scan.verdict == "insufficient_data" and scan.confidence is None
    assert manager.warnings_for(scan_id) == ["x_escaped_analysis"]
