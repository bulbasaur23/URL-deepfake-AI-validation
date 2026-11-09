"""Task 2: four-table schema tests (Task 2 brief §1), all on in-memory SQLite."""

import threading
from collections.abc import Generator
from datetime import timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.settings import Settings
from db import session as db_session
from db.models import Base, EngineResultORM, Scan, ScanItem, Snapshot
from db.session import get_session

SNAPSHOT_TTL = timedelta(days=30)  # spec: snapshot_ttl_s default = 2_592_000 s


def _memory_settings() -> Settings:
    """Task 1's Settings type, pointed at a throwaway in-memory database."""
    return Settings(db_path=":memory:")


@pytest.fixture()
def memory_session() -> Generator[Session]:
    """Session on the shared in-memory engine; tables wiped after each test."""
    session = get_session(_memory_settings())
    yield session
    session.close()
    Base.metadata.drop_all(session.get_bind())


def test_round_trip(memory_session: Session) -> None:
    """Scan + 2 items + engine result + snapshot survive an expire + refetch."""
    session = memory_session
    scan = Scan(
        url="https://example.com/watch?v=1",
        url_canonical="https://example.com/watch?v=1",
        status="done",
        verdict="insufficient_data",
        confidence=42.5,
        ip_hash="0f1e2d3c4b5a",
        title="Example Page",  # prefetched metadata (Task 12 stores it)
        domain="example.com",
    )
    text_item = ScanItem(
        scan=scan,
        kind="text",
        source_ref="body:text:main",
        label="main article",
        status="ok",
        verdict="human",
        confidence=3.25,
    )
    image_item = ScanItem(
        scan=scan,
        kind="image",
        source_ref="img[0]",
        label=None,  # image alt/caption may be absent
        status="ok",
        verdict="unclear",
        confidence=44.0,
    )
    EngineResultORM(
        item=text_item,
        engine_id="text-ai-roberta",
        engine_version="1.0.0",
        score=3.4,
        confidence=0.91,
        rationale="prose looks human-written",
        latency_ms=120,
    )
    snapshot = Snapshot(item=image_item, path="media/ab12cd.jpg", mime="image/jpeg")
    session.add_all([scan, text_item, image_item, snapshot])
    session.commit()
    assert text_item.id and image_item.id

    session.expire_all()  # force the re-read to come from SQLite, not identity map
    loaded = session.get(Scan, scan.id)
    assert loaded is not None
    assert loaded.url_canonical == "https://example.com/watch?v=1"
    assert loaded.status == "done" and loaded.verdict == "insufficient_data"
    assert loaded.title == "Example Page" and loaded.domain == "example.com"

    by_kind = {item.kind: item for item in loaded.items}  # scan.items round-trips
    assert set(by_kind) == {"text", "image"}
    result = by_kind["text"].engine_results[0]
    assert (result.engine_id, result.score, result.latency_ms) == (
        "text-ai-roberta",
        3.4,
        120,
    )
    snap = by_kind["image"].snapshots[0]
    assert (snap.path, snap.mime) == ("media/ab12cd.jpg", "image/jpeg")
    assert snap.expires_at == snap.created_at + SNAPSHOT_TTL


def test_foreign_keys_reject_orphan_rows(memory_session: Session) -> None:
    """PRAGMA foreign_keys=ON: a ScanItem cannot dangle without its Scan."""
    orphan = ScanItem(
        scan_id=99_999,
        kind="text",
        source_ref="body:text",
        label=None,
        status="fetch_failed",
    )
    memory_session.add(orphan)
    with pytest.raises(IntegrityError):
        memory_session.commit()
    memory_session.rollback()


def test_file_db_is_created_and_persists_across_sessions(tmp_path) -> None:
    """get_session builds the parent dirs and reopens the same on-disk data."""
    settings = Settings(db_path=str(tmp_path / "nested" / "app.db"))
    writer = get_session(settings)
    writer.add(Scan(url="https://example.org", url_canonical="https://example.org/"))
    writer.commit()
    writer.close()
    assert (tmp_path / "nested" / "app.db").exists()

    reader = get_session(settings)
    row = reader.scalar(select(Scan).order_by(Scan.id))
    assert row is not None
    assert row.url == "https://example.org"
    assert row.status == "queued"  # model default; no fixture sets a status here
    assert row.verdict is None  # queued scans carry no verdict yet
    reader.close()


def test_concurrent_get_session_shares_one_memory_engine() -> None:
    """Fix round 1 #2: two threads racing a cold engine cache get one engine.

    Regression canary for the check-then-act unlock: on the pre-fix code both
    threads miss the empty cache and build two distinct engines, failing the
    identity assertion. Races `_engine_for` directly since that lock is the
    fix; the private cache is reached only to reset it for a cold start.
    """
    url = db_session._sqlite_url(":memory:")
    db_session._ENGINES.pop(url, None)
    barrier = threading.Barrier(2)
    engines: list[Engine] = []

    def worker() -> None:
        barrier.wait()
        engines.append(db_session._engine_for(url))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(engines) == 2
    assert engines[0] is engines[1]  # the lock turns the double-miss into a hit
