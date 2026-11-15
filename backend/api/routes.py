"""Public scan API (Task 14): submit, poll, rate limit, generic errors.

``POST /api/scan`` follows the §7.5 flow exactly: validate (scheme http/https
+ host present) → 422 ``{"detail": "invalid url"}`` → rate limit (sliding
window, salted-hashed IP) → 429 ``{"detail": "rate limited"}`` →
``manager.submit(url, ip_hash)`` → 202 ``{"scan_id": <int>}``. Validation and
rate limiting are ROUTE-LEVEL dependencies in declaration order (ruling 3's
sanctioned alternative to an ASGI middleware): FastAPI resolves dependencies
sequentially, so a 422 short-circuits before the limiter runs — an outer
middleware could not honor that order without duplicating URL validation.

``GET /api/scan/{id}`` returns the exact §7.6 payload (see
``api.serializers``): status from the Scan row, the report recomputed from
persisted per-item verdicts, warnings from the live manager (gracefully
``[]`` after a restart — they are in-memory by design), ``engines_used`` =
distinct engine ids across the scan's EngineResultORM rows. Unknown ids (and
non-integer ids, which cannot exist) → 404 ``{"detail": "not found"}`` —
generic client errors only (spec §7.7), never FastAPI's verbose validation
payloads, which is why the body is parsed manually and the path id is taken
as a string.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from api.ratelimit import SlidingWindowRateLimiter, hash_ip
from api.serializers import serialize_scan_report
from app.settings import settings
from db.models import EngineResultORM, Scan, ScanItem, Snapshot
from db.session import get_session

router = APIRouter()


def _invalid_url() -> HTTPException:
    """The generic §7.5 rejection (every bad request shape funnels here)."""
    return HTTPException(status_code=422, detail="invalid url")


async def _validated_url(request: Request) -> str:
    """§7.5 validation dependency: scheme http/https + host present.

    The body is parsed manually (FastAPI's own parser would 422 malformed
    payloads with its verbose internal error list — GATE 4 wants generic
    client errors). Any malformed body, missing key, or scheme/host violation
    funnels into the same generic 422. The URL is returned AS SUBMITTED:
    ``manager.submit`` stores it verbatim and computes ``url_canonical``
    before the row insert (Task 12 behavior, untouched by ruling 5).
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - malformed/absent body == invalid url
        raise _invalid_url() from None
    url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(url, str):
        raise _invalid_url()
    try:
        parts = urllib.parse.urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        raise _invalid_url() from None
    if parts.scheme.lower() not in ("http", "https") or not hostname:
        raise _invalid_url()
    return url


async def _rate_limited(request: Request) -> None:
    """Sliding-window submission limiter dependency (ruling 3/4).

    Keyed by the salted-hashed client IP — the plaintext address is never
    stored or keyed. Fail-open if no limiter is on app state (only possible
    for an app built without ``create_app``; the limiter is always wired
    there, so this branch is unreachable in practice).
    """
    limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    client = request.client
    # The direct peer address only: honoring spoofable X-Forwarded-For headers
    # would let one client rotate its way around the window (P1: no proxy tier).
    ip_hash = hash_ip(client.host if client is not None else "", settings)
    if not limiter.allow(ip_hash):
        raise HTTPException(status_code=429, detail="rate limited")


def _load_scan_bundle(
    scan_id: int,
) -> tuple[Scan, list[ScanItem], dict[int, list[EngineResultORM]], set[int]] | None:
    """One session: the Scan row + id-ordered items + grouped results + snapshot ids.

    Returns None for an unknown id. All reads complete before the session
    closes — column attributes stay usable on the detached objects.
    """
    session = get_session(settings)
    try:
        scan = session.get(Scan, scan_id)
        if scan is None:
            return None
        items = list(
            session.scalars(
                select(ScanItem).where(ScanItem.scan_id == scan_id).order_by(ScanItem.id)
            )
        )
        item_ids = [item.id for item in items]
        results_by_item: dict[int, list[EngineResultORM]] = {}
        snapshot_ids: set[int] = set()
        if item_ids:
            for result in session.scalars(
                select(EngineResultORM)
                .where(EngineResultORM.item_id.in_(item_ids))
                .order_by(EngineResultORM.id)
            ):
                results_by_item.setdefault(result.item_id, []).append(result)
            snapshot_ids = set(
                session.scalars(select(Snapshot.item_id).where(Snapshot.item_id.in_(item_ids)))
            )
    finally:
        session.close()
    return scan, items, results_by_item, snapshot_ids


@router.post("/api/scan", status_code=202)
async def submit_scan(
    request: Request,
    url: str = Depends(_validated_url),
    _rate: None = Depends(_rate_limited),
) -> dict[str, int]:
    """§7.5: validate → rate limit → submit → ``202 {"scan_id": int}``.

    The IP is hashed (``settings.ip_salt`` salt) before it reaches the
    manager, which persists it verbatim as ``Scan.ip_hash`` — no plaintext IP
    is ever stored. ``async`` on purpose: ``manager.submit`` schedules the
    scan task on the loop that calls it, and this handler runs on that loop.
    """
    manager = request.app.state.manager
    ip_hash = hash_ip(request.client.host if request.client is not None else "", settings)
    scan_id = await manager.submit(url, ip_hash)
    return {"scan_id": scan_id}


@router.get("/api/scan/{scan_id}")
async def get_scan(scan_id: str, request: Request) -> dict[str, Any]:
    """§7.6 payload for one scan; 404 generic for anything unknown.

    The path id is a string parsed here so a non-integer id (which can never
    exist) lands in the same generic 404 instead of FastAPI's verbose 422.
    Warnings come from the live manager's in-memory store — ``[]`` after a
    restart (graceful empty, ruling 2).
    """
    try:
        identifier = int(scan_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found") from None
    bundle = _load_scan_bundle(identifier)
    if bundle is None:
        raise HTTPException(status_code=404, detail="not found")
    scan, items, results_by_item, snapshot_ids = bundle
    manager = getattr(request.app.state, "manager", None)
    warnings = list(manager.warnings_for(identifier)) if manager is not None else []
    return serialize_scan_report(scan, items, results_by_item, snapshot_ids, warnings, settings)
