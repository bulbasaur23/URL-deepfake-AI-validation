"""SQLite session factory: one engine per database path, create-all on open."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.settings import Settings
from db import models

_MEMORY_URL = "sqlite://"

# Engines are cached per URL: session-per-request must not rebuild a pool, and
# repeated get_session() calls with db_path=":memory:" must hit the same DB.
_ENGINES: dict[str, Engine] = {}


def _sqlite_url(db_path: str) -> str:
    """Map Settings.db_path to a SQLite URL; ':memory:' is the bare URL."""
    if db_path == ":memory:":
        return _MEMORY_URL
    return f"sqlite:///{db_path}"


def _enable_sqlite_fk(dbapi_connection: sqlite3.Connection, _record: object) -> None:
    """SQLite skips FK enforcement unless the pragma is set per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _mkdir_for(url: str) -> None:
    """Create the parent directory of a file-based SQLite URL."""
    path = url.removeprefix("sqlite:///")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _new_engine(url: str) -> Engine:
    """Build one engine; in-memory uses a StaticPool so sessions share one DB."""
    if url == _MEMORY_URL:
        engine = create_engine(
            url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        # FastAPI runs sync handlers in a threadpool, so pooled connections may
        # be used from different threads than the one that opened them.
        engine = create_engine(url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _enable_sqlite_fk)
    return engine


def _engine_for(url: str) -> Engine:
    engine = _ENGINES.get(url)
    if engine is None:
        if url != _MEMORY_URL:
            _mkdir_for(url)
        engine = _new_engine(url)
        _ENGINES[url] = engine
    return engine


def get_session(settings: Settings) -> Session:
    """Open a Session on settings.db_path, creating tables when missing."""
    # Refresh the snapshot TTL from the active Settings before any flush.
    models.SNAPSHOT_TTL_S = int(settings.snapshot_ttl_s)
    engine = _engine_for(_sqlite_url(settings.db_path))
    models.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
