"""Scan job manager (Task 12): queue, 120 s deadline, per-item resilience.

``ScanJobManager`` owns the background orchestrator for one submitted URL:
persist a queued ``Scan`` row → fetch (GATE 1) → extract + language gate +
image pipeline (parsing) → engines (analyzing) → the aggregating seam
(Task 13) → done. Every status transition is one ``db.session.get_session``
write between stages; stage mapping (controller ruling 3): ``fetching`` = the
fetch, ``parsing`` = extract + langid gate + ``images.process_all``,
``analyzing`` = the two engine gather groups, ``aggregating`` = Task 13's
seam stage.

Resilience (design §7.3/§7.4, controller rulings):

- the orchestrator body (fetch+extract+analyze+aggregate; startup preloads
  excluded, §11.7) runs under ``asyncio.timeout(settings.scan_deadline_s)``.
  On wall-clock fire: ≥1 item holding an engine result → ``done`` +
  ``deadline_exceeded`` (never-analyzed items → status ``unavailable``,
  persisted); zero analyzed → ``failed`` with the closed fetch-vocab token
  ``timeout``.
- each engine call is ``asyncio.wait_for(engine.analyze(item),
  settings.engine_timeout_s)``; EngineFault / TimeoutError / OSError /
  torch.OutOfMemoryError mark JUST that item ``degraded`` and the scan
  continues — one item never fails the whole scan (any other engine
  exception is a contract violation and degrades the item too). Text and
  image items run as two concurrent gather groups; within a group items are
  sequential in document order (determinism + engine thread-safety).
- a ``fetch_failed`` result short-circuits: ``failed`` + the fetch
  ``error_code`` token verbatim, no later stage (§7.4 edge case 1).
- kind=image ``ScanItem`` rows are created by Task 7's ``process_all`` —
  never here; results bind through a POSITIONAL zip of the persisted rows
  against the returned Items (both deterministic document order, T7
  guarantee). A mismatch is a pipeline bug: loud RuntimeError, never a blind
  bind. Text rows are created here (one batch window after extraction).

Warnings (controller ruling 7) are IN-MEMORY, keyed by scan_id, tokens
exactly ``deadline_exceeded | x_escaped_analysis | text_cap_truncated |
language_limited`` — plain tokens, no counts embedded (counts are derivable
from item statuses at report time). P1 is single-process: persistence and
report-side derivation are Task 13/14 territory; a restart drops them by
design. ``warnings_for(scan_id)`` reads them.

Event-loop caveat: ``submit()`` schedules the scan task on the running loop
that called it, and every manager method must be used from that one loop
(P1 runs a single loop; multi-worker deployments get one manager + SQLite DB
per worker, each with its own queue depth).
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from collections.abc import Mapping, Sequence

from sqlalchemy import select

from app.settings import Settings
from db.models import EngineResultORM, Scan, ScanItem
from db.session import get_session
from pipeline import fetcher, images
from pipeline.engines.base import Engine, EngineFault
from pipeline.extract import TEXT_CAP_WARNING, ExtractionResult, extract
from pipeline.langid import mark_text_items
from pipeline.types import Item

try:  # torch ships with the engines; guarded so the manager imports without it
    import torch

    _TORCH_ERRORS: tuple[type[BaseException], ...] = (torch.OutOfMemoryError,)
except ImportError:  # pragma: no cover - torch is a hard engine dependency
    _TORCH_ERRORS = ()

# GATE 3 degrade paths (§11.14): per-call timeout, the engine contract fault,
# and the two host-level failure classes the brief names. Anything else an
# engine raises is a contract violation — it degrades the item too (the broad
# catch in _analyze_group), never the scan.
_DEGRADE_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    OSError,
    EngineFault,
    *_TORCH_ERRORS,
)

logger = logging.getLogger(__name__)


def _canonicalize(url: str) -> str:
    """Minimal canonicalization for ``Scan.url_canonical`` (ruling 2).

    Nothing upstream canonicalizes (verified T3/T5): strip the fragment,
    lowercase the host (``urlsplit`` already lowercases scheme and hostname),
    and drop userinfo so credentials never reach stored URLs (GATE 4). Path,
    query, and explicit ports are preserved verbatim.
    """
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    if ":" in host:  # IPv6 literal stays bracketed in the rebuilt netloc
        netloc = f"[{host}]" if port is None else f"[{host}]:{port}"
    else:
        netloc = host if port is None else f"{host}:{port}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def _hostname(url: str) -> str | None:
    """Lowercase hostname of ``url`` — the ``Scan.domain`` fallback (ruling 2)."""
    try:
        hostname = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return None
    return hostname.lower() if hostname else None


class ScanJobManager:
    """Background orchestrator: queue scans, run them capped, ship partials."""

    def __init__(self, settings: Settings, engines: Mapping[str, Engine]) -> None:
        """Store the boot-built engine instances; never re-resolve per scan.

        ``engines`` maps modality → live engine instance as built ONCE at
        startup (§11.7: the first scan must never reload weights — instances
        come in, ``registry`` is not consulted here). A modality missing from
        the map means its engine failed to load at startup: its items end
        ``unavailable`` rather than crashing the scan.
        """
        self._settings = settings
        self._engines: dict[str, Engine] = dict(engines)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_scans)
        # Tasks live for the process lifetime (each is bounded by the 120 s
        # deadline): queue_depth() reads them; nothing else needs to.
        self._tasks: dict[int, asyncio.Task[None]] = {}
        # In-memory warning tokens per scan (ruling 7; schema has no column).
        self._warnings: dict[int, list[str]] = {}

    def queue_depth(self) -> int:
        """Scans submitted but not finished: queued (waiting on the cap) + running."""
        return sum(1 for task in self._tasks.values() if not task.done())

    def warnings_for(self, scan_id: int) -> list[str]:
        """The in-memory warning tokens for one scan (empty when unknown)."""
        return list(self._warnings.get(scan_id, ()))

    async def wait(self, scan_id: int) -> None:
        """Block until the scan reaches a terminal status (``done`` | ``failed``).

        DB-polled on purpose: the terminal status is the one persisted fact
        callers care about, and polling keeps the public surface free of task
        plumbing (P1: single process, SQLite-local reads).
        """
        while True:
            if self._load_scan(scan_id).status in ("done", "failed"):
                return
            await asyncio.sleep(0.01)

    async def submit(self, url: str, client_ip_hash: str) -> int:
        """Persist ``Scan(status="queued")`` and schedule the orchestrator run.

        ``client_ip_hash`` is the CALLER's salted hash (Task 14's API computes
        it); it is stored verbatim (ruling 10). Returns the scan id
        immediately — the work happens on the scheduled task, which acquires
        the concurrency-cap semaphore before running (ruling 9).
        """
        session = get_session(self._settings)
        try:
            scan = Scan(
                url=url,
                url_canonical=_canonicalize(url),
                status="queued",
                ip_hash=client_ip_hash,
            )
            session.add(scan)
            session.commit()
            scan_id = scan.id
        finally:
            session.close()
        self._warnings[scan_id] = []
        self._tasks[scan_id] = asyncio.create_task(
            self._guarded_run(scan_id), name=f"scan-{scan_id}"
        )
        return scan_id

    async def run(self, scan_id: int) -> None:
        """Orchestrate one scan: fetch → parse → analyze → aggregate → done.

        The deadline clock covers fetch through the aggregating seam only —
        startup preloads are excluded (§11.7). On wall-clock fire the partial
        result ships (ruling 5, ``_on_deadline``); a ``fetch_failed`` result
        short-circuits into ``failed`` with the fetch token (§7.4 edge 1).
        """
        settings = self._settings
        scan = self._load_scan(scan_id)
        self._write_scan(scan_id, status="fetching")
        # Deadline-handler inputs, declared before the clock starts: the
        # in-memory Items and their id(Item) → persisted-row-id bindings.
        items: list[Item] = []
        bindings: dict[int, int] = {}
        extraction: ExtractionResult | None = None
        try:
            async with asyncio.timeout(settings.scan_deadline_s):
                fetched = await fetcher.fetch(scan.url, settings)
                if fetched.status != "ok":
                    # §7.4 edge 1: fetch_failed short-circuits the scan. The
                    # error_code is the closed fetch-vocab token verbatim —
                    # never a paraphrase like "fetch_blocked_by_robots".
                    self._write_scan(scan_id, status="failed", error_code=fetched.error_code)
                    return
                self._write_scan(scan_id, status="parsing")
                extraction = extract(
                    fetched.content, url=fetched.final_url or scan.url, settings=settings
                )
                mark_text_items(extraction.text_items, settings)  # §11.9: gate, never translate
                items = [*extraction.text_items]  # deadline-handler visible from here on
                bindings = self._persist_text_items(scan_id, extraction.text_items)
                image_items = await images.process_all(
                    extraction.image_refs, fetcher.fetch, settings, scan_id
                )
                items.extend(image_items)
                bindings.update(self._bind_image_rows(scan_id, image_items))
                # The extractor emits no metadata items (ruling 2): title is
                # ExtractionResult.title (<title>/og:title), domain its
                # hostname — with the URL netloc as the domain fallback.
                self._write_scan(
                    scan_id,
                    status="analyzing",
                    title=extraction.title,
                    domain=extraction.domain or _hostname(scan.url),
                )
                outcomes = await asyncio.gather(
                    self._analyze_group("text", extraction.text_items),
                    self._analyze_group("image", image_items),
                    return_exceptions=True,
                )
                for outcome in outcomes:  # a group bug fails the scan loudly
                    if isinstance(outcome, BaseException) and not isinstance(
                        outcome, asyncio.CancelledError
                    ):
                        raise outcome
                self._persist_analysis(items, bindings)
                self._write_scan(scan_id, status="aggregating")
                # --- Task 13 seam -------------------------------------------------
                # Task 13 fills aggregate_verdict here: read each item's
                # engine_results, fill ScanItem.verdict/confidence and
                # Scan.verdict/confidence in the final write below. Task 12
                # leaves every verdict/confidence None.
                self._write_scan(scan_id, status="done")
        except TimeoutError:
            self._on_deadline(scan_id, items, bindings)
        self._final_warnings(scan_id, extraction, items)

    async def _guarded_run(self, scan_id: int) -> None:
        """Run one scan under the concurrency cap; a crash marks it failed.

        The broad catch is the last-resort net for orchestrator bugs (the
        vocab has no crash token — the status alone says failed, error_code
        stays None, nothing is invented). Cancellation is never swallowed.
        """
        async with self._semaphore:
            try:
                await self.run(scan_id)
            except Exception:
                logger.exception("scan %s: unexpected orchestrator error", scan_id)
                self._write_scan(scan_id, status="failed")

    def _on_deadline(
        self, scan_id: int, items: Sequence[Item], bindings: Mapping[int, int]
    ) -> None:
        """Wall-clock fire (ruling 5): partial results ship; stragglers unavailable."""
        analyzed = any(item.engine_results for item in items)
        self._warn(scan_id, "deadline_exceeded")
        for item in items:
            if item.status == "ok" and not item.engine_results:
                item.status = "unavailable"  # the engine never ran to completion
        self._persist_analysis(items, bindings)  # whatever rows exist get the truth
        if analyzed:
            self._write_scan(scan_id, status="done")
        else:
            self._write_scan(scan_id, status="failed", error_code="timeout")

    def _final_warnings(
        self, scan_id: int, extraction: ExtractionResult | None, items: Sequence[Item]
    ) -> None:
        """Ruling 7 tokens derivable from the extraction + final item statuses."""
        if extraction is not None and TEXT_CAP_WARNING in extraction.warnings:
            self._warn(scan_id, "text_cap_truncated")
        if any(item.kind == "text" and item.status == "unsupported" for item in items):
            self._warn(scan_id, "language_limited")
        if any(item.status in ("unsupported", "degraded", "unavailable") for item in items):
            self._warn(scan_id, "x_escaped_analysis")

    async def _analyze_group(self, modality: str, group_items: Sequence[Item]) -> None:
        """Analyze one modality's items sequentially in document order (ruling 6).

        Items whose status is not ``ok`` (language-gated text, failed/blocked
        images) skip engines entirely. With no loaded engine for the modality
        (startup skip) or an engine that does not claim the item, the item
        ends ``unavailable`` — the engine never ran on it.
        """
        engine = self._engines.get(modality)
        for item in group_items:
            if item.status != "ok":
                continue  # unsupported / fetch_failed / ... never reach an engine
            if engine is None or not engine.supports(item):
                item.status = "unavailable"  # engine missing or not claiming it
                logger.warning("scan item %s: no engine ran on it", item.source_ref)
                continue
            try:
                outcome = await asyncio.wait_for(
                    engine.analyze(item), self._settings.engine_timeout_s
                )
            except _DEGRADE_ERRORS as exc:  # GATE 3 (§11.14) degrade paths
                logger.warning("scan item %s degraded: %s", item.source_ref, exc)
                item.status = "degraded"
            except Exception as exc:
                # One item never fails the scan: a broken-engine-contract
                # exception degrades just this item (logged, then continue).
                logger.warning(
                    "scan item %s: engine %s broke the contract; item degraded",
                    item.source_ref,
                    engine.id,
                    exc_info=exc,
                )
                item.status = "degraded"
            else:
                item.engine_results.append(outcome)

    def _persist_text_items(self, scan_id: int, text_items: Sequence[Item]) -> dict[int, int]:
        """Create the text ScanItem rows in one batch window (ruling 2).

        The extractor emits no kind=metadata items (verified T5: metadata is
        ``ExtractionResult.title``/``.domain``, persisted on the Scan row in
        the analyzing transition), so this batch is text-only. Returns the
        id(item) → row-id binding for the analysis write.
        """
        bindings: dict[int, int] = {}
        if not text_items:
            return bindings
        session = get_session(self._settings)
        try:
            for item in text_items:
                row = ScanItem(
                    scan_id=scan_id,
                    kind=item.kind,
                    source_ref=item.source_ref,
                    label=item.label,
                    status=item.status,
                )
                session.add(row)
                session.flush()  # materialize the id for the binding
                bindings[id(item)] = row.id
            session.commit()
        finally:
            session.close()
        return bindings

    def _bind_image_rows(self, scan_id: int, image_items: Sequence[Item]) -> dict[int, int]:
        """Ruling 1: T7 already persisted the image rows — zip POSITIONALLY.

        Both sequences are deterministic document order (T7 guarantee): one
        row per returned Item, same order. A mismatch is a pipeline bug:
        refuse loudly instead of binding blind (the RuntimeError fails the
        scan through the orchestrator's catch-all).
        """
        session = get_session(self._settings)
        try:
            rows = list(
                session.scalars(
                    select(ScanItem)
                    .where(ScanItem.scan_id == scan_id, ScanItem.kind == "image")
                    .order_by(ScanItem.id)
                )
            )
        finally:
            session.close()
        if len(rows) != len(image_items):
            raise RuntimeError(
                f"scan {scan_id}: image row zip mismatch — {len(rows)} kind=image rows "
                f"vs {len(image_items)} Items from process_all; refusing to bind blindly"
            )
        return {id(item): row.id for item, row in zip(image_items, rows)}

    def _persist_analysis(self, items: Sequence[Item], bindings: Mapping[int, int]) -> None:
        """One batch: EngineResultORM rows + final item statuses (one session)."""
        session = get_session(self._settings)
        try:
            for item in items:
                row_id = bindings.get(id(item))
                if row_id is None:
                    continue  # deadline before its row existed — nothing to update
                for outcome in item.engine_results:
                    session.add(
                        EngineResultORM(
                            item_id=row_id,
                            engine_id=outcome.engine_id,
                            engine_version=outcome.engine_version,
                            score=outcome.score,
                            confidence=outcome.confidence,
                            rationale=outcome.rationale,
                            latency_ms=outcome.latency_ms,
                        )
                    )
                row = session.get(ScanItem, row_id)
                if row is not None:
                    row.status = item.status
            session.commit()
        finally:
            session.close()

    def _write_scan(self, scan_id: int, **fields: object) -> None:
        """One persisted Scan-row write — one get_session per write (ruling 3)."""
        session = get_session(self._settings)
        try:
            scan = session.get(Scan, scan_id)
            if scan is not None:
                for key, value in fields.items():
                    setattr(scan, key, value)
                session.commit()
        finally:
            session.close()

    def _warn(self, scan_id: int, token: str) -> None:
        """Append one warning token to the in-memory per-scan list (ruling 7)."""
        self._warnings.setdefault(scan_id, []).append(token)

    def _load_scan(self, scan_id: int) -> Scan:
        """Load the Scan row (raises ``LookupError`` for an unknown id)."""
        session = get_session(self._settings)
        try:
            scan = session.get(Scan, scan_id)
        finally:
            session.close()
        if scan is None:
            raise LookupError(f"scan {scan_id} not found")
        return scan
