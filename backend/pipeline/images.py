"""Image pipeline (GATE 2 pick side): extractor ImageRefs → analyzable Items.

``process_all(image_refs, fetch, settings, scan_id)`` turns the extractor's
image candidates into ``kind="image"`` Items the engines can judge:

1. **Pick (before any fetch):** every ImageRef resolves to exactly one URL —
   the candidate with the highest ``w`` descriptor; the LAST candidate when
   none declares a width (plain ``src``-only refs resolve that way by the
   extractor's construction). Width ties keep the first occurrence, and the
   losing variants are never fetched: one deterministic fetch per ref, no
   politeness spend on silent fallbacks. If the winner fails to fetch, the
   smaller variants do NOT rescue it.
2. **Fetch:** through the passed callable — the Task-3 SSRF-guarded
   ``fetcher.fetch`` with its robots politeness, so every hop inherits GATE 1.
   Outcomes map strictly: ``error_code == "robots_disallowed"`` →
   ``blocked_by_site``; every other failure (``http_4xx``/``http_5xx``/
   ``http_429``, ``timeout``, ``dns_failure``, ``private_ip``, ``bad_scheme``,
   redirect bookkeeping, untokened transport errors) → ``fetch_failed``.
3. **Validate:** magic bytes only — JPEG ``FFD8``, PNG ``89504E47``, GIF
   ``474946``, WebP ``RIFF…WEBP`` — then a full PIL decode (``load()`` forces
   truncated bodies / non-decodable variants to fail). A body that passes the
   fetch but not validation is not a fetch failure: it becomes an Item with
   ``status="unsupported"``.
4. **Size gate + re-encode:** the natural size is checked BEFORE anything
   else — ``min(width, height) < Settings.image_min_px`` skips the whole
   pipeline for that ref (no Item, no fetch-spend already made is sunk).
   Surviving images are flattened onto RGB (alpha-on-white) and, when the
   longest side exceeds 1600 px, downscaled to fit 1600 px (aspect kept,
   shrink-only) — the GATE 2 memory bound for the engines' tensors — then
   re-encoded to JPEG (deterministically, quality 85), which strips all
   metadata and produces the dedupe key: ``sha256`` of the re-encoded bytes.
   A repeated digest is a dedupe loser: no Item, no second row, no second
   file ("keep first").
5. **Cap:** ``Settings.max_images`` bounds the refs that are processed, by
   document order and regardless of outcome — a skipped/deduped ref within
   the cap consumed its slot (the fetch/decode work happened; skipping again
   would unbound it). Refs beyond the processing order become Items with
   ``status="too_large"`` without any fetch — no wasted politeness.

Deterministic output: Items are returned as a document-order subset of the
input refs (monotonically ordered, smaller by the skip/dedupe rules above).

Presistence contract (ruling 3): snapshot files are content-addressed under
``Settings.media_dir`` — ``media/<sha256-of-re-encoded-bytes>.jpg``, written
(mkdir -p, only when the digest's file is not already on disk) BEFORE any DB
row. In one ``db.session.get_session(settings)`` batch window at the end,
each produced Item's ``ScanItem`` row is created and, for ``status="ok"``
items, one ``Snapshot`` row pointing at that file with
``expires_at = created_at + Settings.snapshot_ttl_s`` (Task 2's computed
semantics). Ownership note carried for Task 12: the ScanItem rows for
**image** items are created here (scan_id FK requires an existing Scan row,
and the Snapshot FK requires an existing ScanItem row — impossible to satisfy
without creating the shells), so the orchestrator must not re-persist them;
text items remain fully the job manager's.

``ImageRef`` (declared in ``pipeline/types.py`` alongside ``Item``) never
carries ``data:`` URLs from the extractor; a defensive filter drops any that
slip through so they can never produce an Item (ruling 2).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from app.settings import Settings
from db.models import ScanItem, Snapshot
from db.session import get_session
from pipeline.types import FetchResult, ImageRef, Item

_MAX_ENCODE_PX = 1600  # (call)-pinned long-side downscale — bounds engine tensor memory
_JPEG_QUALITY = 85  # deterministic re-encode knobs of the snapshot writer
_JPEG_FORMAT = "JPEG"

FetchCallable = Callable[[str, Settings], Awaitable[FetchResult]]


def _utcnow() -> datetime:
    """Naive UTC timestamp — the same clock shape ``db.models`` persists."""
    return datetime.now(UTC).replace(tzinfo=None)


def _magic_ok(payload: bytes) -> bool:
    """True when the leading bytes declare one of the accepted raster families.

    Deliberately byte-prefix checks of the brief's constants (JPEG ``FFD8``,
    PNG ``89504E47``, GIF ``474946``, WebP ``RIFF…WEBP``): claimed file names
    and Content-Type headers are attacker-controlled, the first bytes are not.
    """
    if payload[:2] == b"\xff\xd8":  # JPEG
        return True
    if payload[:4] == b"\x89PNG":  # PNG
        return True
    if payload[:3] == b"GIF":  # GIF (any of GIF87a/GIF89a)
        return True
    return len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"


def _best_candidate_url(ref: ImageRef) -> str | None:
    """Resolve the one URL this ref will spend its fetch on.

    Highest ``w`` descriptor wins (``max()`` keeps the first maximal entry on
    ties — ruling 5); with no width-bearing candidate at all the LAST
    candidate wins (plain ``src`` when srcset is absent outright). ``data:``
    candidates — never emitted by the extractor — are filtered first so they
    can neither win nor force a fetch; None means "nothing usable", i.e. skip.
    """
    usable = [
        (url, width) for url, width in ref.url_candidates if not url.lower().startswith("data:")
    ]
    if not usable:
        return None
    widest = [candidate for candidate in usable if candidate[1] is not None]
    if widest:
        return max(widest, key=lambda candidate: candidate[1])[0]
    return usable[-1][0]


def _decode(payload: bytes) -> Image.Image | None:
    """Fully decode ``payload`` with PIL; None when PIL cannot read it.

    ``load()`` forces the whole decode immediately instead of letting the
    error surface lazily at ``save()``/``size``-read time: truncated bodies,
    corrupt samples, and over-Pillow's decompression-bomb guard all fail here.
    """
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception:  # noqa: BLE001
        # Any PIL/Open/Decode exc means "this body is not a usable raster":
        # content-shaped failure → unsupported Item, never a raised error.
        return None
    return image


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Mode-safe RGB conversion; alpha composites onto white, not black."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.convert("RGBA").split()[3])
        return background
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image) -> bytes:
    """Re-encode to JPEG with fixed knobs — the metadata-stripping dedupe key."""
    buffer = io.BytesIO()
    image.save(buffer, _JPEG_FORMAT, quality=_JPEG_QUALITY)
    return buffer.getvalue()


def _snapshot_path(settings: Settings, digest: str) -> str:
    """Content-addressed media_dir-internal path: ``media/<sha256>.jpg``."""
    return (Path(settings.media_dir) / f"{digest}.jpg").as_posix()


def _write_if_missing(path: str, jpeg_bytes: bytes) -> None:
    """Content-addressed write: create it once, reuse it forever after."""
    target = Path(path)
    if target.exists():
        return  # same content → same path; the first write is the whole truth
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(jpeg_bytes)


def _persist(items: list[Item], settings: Settings, scan_id: int) -> None:
    """Write every Item's ScanItem row (+ Snapshot for ok) in one batch window.

    Files were written before this runs (ruling 3, file-before-row). One
    ``get_session`` session covers all rows so a scan's media rows commit
    together; ``expires_at`` is computed in Python from the same
    ``created_at`` reading (Task 2's invariant, re-derived from Settings).
    """
    session = get_session(settings)
    try:
        created_at = _utcnow()
        expires_at = created_at + timedelta(seconds=settings.snapshot_ttl_s)
        for item in items:
            scan_item = ScanItem(
                scan_id=scan_id,
                kind=item.kind,
                source_ref=item.source_ref,
                label=item.label,
                status=item.status,
            )
            session.add(scan_item)
            session.flush()  # Snapshot's FK needs the materialized ScanItem id
            if item.status == "ok" and item.image_path is not None:
                session.add(
                    Snapshot(
                        item_id=scan_item.id,
                        path=item.image_path,
                        mime=item.mime,
                        created_at=created_at,
                        expires_at=expires_at,
                    )
                )
        session.commit()
    finally:
        session.close()


async def process_all(
    image_refs: list[ImageRef],
    fetch: FetchCallable,
    settings: Settings,
    scan_id: int,
) -> list[Item]:
    """Image pipeline over Task-5 refs, in one politeness-safe pass.

    Sequence per ref, in document order: pick one candidate URL → fetch
    through the guarded channel → validate magic bytes → PIL-decode → natural
    size gate (skip whole pipeline below ``image_min_px`` on either side) →
    flatten/downscale/re-encode → sha256 dedupe (first occurrence wins) →
    content-addressed snapshot file → Items returned as a document-order
    subset; persistence (ScanItem shells + ok-image Snapshots) happens in one
    batch window at the end. See the module docstring for the exact mapping
    and the persistence ownership note.
    """
    items: list[Item] = []
    seen_digests: set[str] = set()

    for position, ref in enumerate(image_refs):
        if position >= settings.max_images:
            items.append(
                Item(
                    kind="image", label=ref.label, source_ref=ref.source_ref, status="too_large"
                )  # known before any fetch: no work wasted
            )
            continue

        url = _best_candidate_url(ref)
        if url is None:
            continue  # defensive: data:-only candidates → no Item (ruling 2)

        result = await fetch(url, settings)
        if result.status != "ok":
            blocked = result.error_code == "robots_disallowed"
            items.append(
                Item(
                    kind="image",
                    label=ref.label,
                    source_ref=ref.source_ref,
                    status="blocked_by_site" if blocked else "fetch_failed",
                )
            )
            continue

        if not _magic_ok(result.content):
            items.append(
                Item(
                    kind="image", label=ref.label, source_ref=ref.source_ref, status="unsupported"
                )  # fetch ok, body is no accepted raster
            )
            continue

        image = _decode(result.content)
        if image is None:
            items.append(
                Item(
                    kind="image", label=ref.label, source_ref=ref.source_ref, status="unsupported"
                )  # magic ok, PIL cannot decode it
            )
            continue

        if min(image.size) < settings.image_min_px:
            continue  # natural size too small on either side: skip, NO Item

        flattened = _flatten_to_rgb(image)
        if max(flattened.size) > _MAX_ENCODE_PX:
            flattened.thumbnail((_MAX_ENCODE_PX, _MAX_ENCODE_PX), Image.Resampling.LANCZOS)
        jpeg_bytes = _encode_jpeg(flattened)
        digest = hashlib.sha256(jpeg_bytes).hexdigest()

        if digest in seen_digests:
            continue  # dedupe loser: keep first occurrence, no Item, no row

        seen_digests.add(digest)
        path = _snapshot_path(settings, digest)
        _write_if_missing(path, jpeg_bytes)  # file BEFORE any DB row (ruling 3)
        items.append(
            Item(
                kind="image",
                label=ref.label,
                source_ref=ref.source_ref,
                image_path=path,
                mime="image/jpeg",
                status="ok",
            )
        )

    if items:
        _persist(items, settings, scan_id)
    return items
