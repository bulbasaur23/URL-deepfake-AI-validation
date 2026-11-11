"""Startup model preload (Task 8): winners instantiated once, off the scan clock.

Design §11.7 splits preload from analysis: model download/load happens here,
driven by the startup hook (``app.main.lifespan``), so per-scan time
(``scan_deadline_s = 120``) never includes model loading.

Today the winners are the not-yet-written Task 9/10 classes, so preload is a
no-op-tolerant scan: an entry whose class cannot import/load logs a warning
and is skipped, keeping ``models_loaded`` short (Task 1's health pin stays
green). When Tasks 9/10 land, this same code loads the real weights — no
edits needed here. Typo'd settings ids are the one non-tolerated failure:
``UnknownEngineError`` is re-raised so a bad id fails startup loudly
(design §11.6), not per scan.
"""

from __future__ import annotations

import logging

from app.settings import Settings
from pipeline.engines import registry

logger = logging.getLogger(__name__)


def preload(settings: Settings) -> dict[str, str]:
    """Import + instantiate each winner engine once; return ``{engine_id: version}``.

    One instance per modality winner (``settings.text_engine_id`` /
    ``settings.image_engine_id``; duplicate ids are deduped, iteration is
    sorted so the health payload order is deterministic). A winner whose
    class cannot import/load yet is logged + skipped — startup must not
    crash just because weights are missing.
    """
    loaded: dict[str, str] = {}
    for engine_id in sorted({settings.text_engine_id, settings.image_engine_id}):
        try:
            engine = registry.get_by_id(engine_id, settings)
        except registry.UnknownEngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - tolerated at preload only (missing weights)
            logger.warning(
                "preload: engine %r not loadable yet, skipped (models_loaded stays short): %s",
                engine_id,
                exc,
            )
            continue
        loaded[engine_id] = engine.version
    return loaded
