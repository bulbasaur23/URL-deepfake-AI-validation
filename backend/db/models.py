"""SQLAlchemy ORM models for the four backend tables (Task 2 brief).

Scans → ScanItems → EngineResultORM rows form the verdict tree; Snapshots
attach to ScanItems and expire after the configured TTL.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Default snapshot TTL in seconds (30 days). db.session.get_session() refreshes
# this from Settings.snapshot_ttl_s on every call, so a `.env` override stays
# authoritative without the models importing the app settings singleton.
SNAPSHOT_TTL_S = 2_592_000


class Base(DeclarativeBase):
    """Declarative base shared by all backend tables."""


def _utcnow() -> datetime:
    """Naive UTC timestamp; SQLite has no tz type, so every clock stays naive."""
    return datetime.now(UTC).replace(tzinfo=None)


class Scan(Base):
    """One analyzed URL: lifecycle status, final verdict, and provenance."""

    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048))
    url_canonical: Mapped[str] = mapped_column(String(2048))
    # Filled in after the fetch (Tasks 12/14/16); unknown until then.
    title: Mapped[str | None] = mapped_column(String(512))
    domain: Mapped[str | None] = mapped_column(String(255))
    # queued | fetching | parsing | analyzing | aggregating | done | failed —
    # written by the scan orchestrator.
    status: Mapped[str] = mapped_column(String(32), default="queued")
    # authentic | unclear | synthetic | insufficient_data (§3.4 verdict bands);
    # None while the scan is still running or failed.
    verdict: Mapped[str | None] = mapped_column(String(32))
    # 0..100 final band score; per-item scores live on ScanItem.
    confidence: Mapped[float | None] = mapped_column(Float)
    # Stable error taxonomy (e.g. fetch_blocked_by_robots) when status = failed.
    error_code: Mapped[str | None] = mapped_column(String(64))
    # Salted hash of the caller IP (never the IP itself; §11.7).
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    items: Mapped[list[ScanItem]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
    )


class ScanItem(Base):
    """One content payload extracted from a Scan (text block, image, …)."""

    __tablename__ = "scan_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"))
    # text | image | metadata — the extractor's payload taxonomy (GATE 2).
    kind: Mapped[str] = mapped_column(String(32))
    # Locator in the source document, e.g. "body:text:main" or "img[0]".
    source_ref: Mapped[str] = mapped_column(String(512))
    # Image alt text / caption when present; text blocks use it for context.
    label: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32))
    verdict: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)

    scan: Mapped[Scan] = relationship(back_populates="items")
    engine_results: Mapped[list[EngineResultORM]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
    )
    snapshots: Mapped[list[Snapshot]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
    )


class EngineResultORM(Base):
    """One engine's individual reading of a ScanItem (pre-aggregation)."""

    __tablename__ = "engine_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("scan_items.id", ondelete="CASCADE"))
    # Settings-driven ids like "text-ai-roberta"; the Task 11 bake-off may swap them.
    engine_id: Mapped[str] = mapped_column(String(64))
    engine_version: Mapped[str] = mapped_column(String(64))
    # 0..100 synthetic-likeness score from the engine itself.
    score: Mapped[float] = mapped_column(Float)
    # The engine's own confidence; separate from the aggregated band score.
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)

    item: Mapped[ScanItem] = relationship(back_populates="engine_results")


class Snapshot(Base):
    """Persisted media file for one ScanItem, purged after its TTL expires."""

    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("scan_items.id", ondelete="CASCADE"))
    # Path under Settings.media_dir; mime keeps the purge decision cheap.
    path: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)

    item: Mapped[ScanItem] = relationship(back_populates="snapshots")


@event.listens_for(Snapshot, "before_insert")
def _stamp_snapshot_times(_mapper: object, _connection: object, target: Snapshot) -> None:
    """Derive snapshot timestamps from a single clock reading.

    Setting both columns inside one event keeps the persisted row invariant
    `expires_at == created_at + SNAPSHOT_TTL_S` exact — two independent
    `datetime.now()` defaults could drift apart by microseconds. Python-side
    column defaults still cover plain Core inserts that bypass the ORM.
    """
    if target.created_at is None:
        target.created_at = _utcnow()
    if target.expires_at is None:
        target.expires_at = target.created_at + timedelta(seconds=SNAPSHOT_TTL_S)
