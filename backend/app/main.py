"""FastAPI application factory.

Startup wiring (Tasks 8 + 12 + 14): a FastAPI ``lifespan`` hook — the modern
lifespan protocol, chosen over deprecated ``on_event("startup")`` — builds
the two winner engine instances ONCE at boot (design §11.7: model load
happens off the per-scan clock) and hands the live instances to the
``ScanJobManager`` on ``app.state``; ``GET /api/health`` reports the
instances' versions and the manager's real queued+running count. Task 14
mounts the scan API (``api.routes``) and puts the submission rate limiter on
``app.state``; Task 12 no longer calls ``pipeline.preload.preload`` here:
preload constructs-and-discards its instances, so calling it too would load
every checkpoint twice.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import routes
from api.ratelimit import SlidingWindowRateLimiter
from app.settings import settings
from jobs.manager import ScanJobManager
from pipeline.engines import registry
from pipeline.engines.base import Engine
from pipeline.engines.registry import UnknownEngineError

logger = logging.getLogger(__name__)

_MODALITIES = ("text", "image")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the winner engines once at boot; store the live instances.

    §11.6/§11.7 semantics at the call site (Task 12 ruling 8): a typo'd
    settings id raises ``UnknownEngineError`` — re-raised so startup crashes
    loudly; an engine class that cannot import/load logs a warning and is
    skipped, so ``models_loaded`` stays short and health stays answerable
    (the Task 8 tolerance, now owned here instead of inside preload).
    """
    engines: dict[str, Engine] = {}
    built: dict[str, Engine] = {}
    for modality in _MODALITIES:
        try:
            engine = registry.get_engine(modality, settings)
        except UnknownEngineError:
            raise  # typo'd settings id: fail startup (§11.6), never per scan
        except Exception as exc:  # noqa: BLE001 - tolerated at startup only (missing weights)
            logger.warning(
                "startup: %s engine not loadable, skipped (models_loaded stays short): %s",
                modality,
                exc,
            )
            continue
        if engine.id in built:
            # Both modality ids point at one engine: one shared instance. With a
            # degenerate multi-modal config the two concurrent gather groups would
            # call that one instance simultaneously (unreachable with the real
            # winners — noted per Task 12 review).
            engines[modality] = built[engine.id]
            continue
        built[engine.id] = engines[modality] = engine
    app.state.engines = engines
    # Sorted by engine id — the same deterministic payload order preload used.
    app.state.models_loaded = {
        engine_id: engine.version for engine_id, engine in sorted(built.items())
    }
    app.state.manager = ScanJobManager(settings, engines)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AI Content Detector", version="0.1.0", lifespan=lifespan)

    # Task 14: the public scan API. Rate limiting is enforced at the POST
    # route (a route-level dependency, ruling 3's sanctioned alternative to an
    # ASGI middleware) so §7.5 URL validation runs BEFORE the limiter and
    # invalid submissions never consume budget; the limiter itself lives on
    # app state, keyed by salted-hashed client IPs (no plaintext IP anywhere).
    app.include_router(routes.router)
    app.state.rate_limiter = SlidingWindowRateLimiter(
        settings.rate_window_s, settings.rate_max_requests
    )

    # Defaults so /api/health answers even before (or without) the lifespan
    # hook having run; the lifespan overwrite carries the real values.
    app.state.models_loaded = {}

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        """Liveness + model-preload probe (Task 1 payload shape; Task 12 wiring).

        ``models_loaded`` lists the LIVE winner engine versions (empty when a
        winner class cannot import/load at startup); ``queue_depth`` is the
        ScanJobManager's real count of queued+running scans (0 before the
        lifespan has run). async on purpose: it reads the manager's task
        table on the loop instead of racing it from a threadpool.
        """
        manager = getattr(app.state, "manager", None)
        return {
            "models_loaded": list(app.state.models_loaded.values()),
            "queue_depth": manager.queue_depth() if manager is not None else 0,
        }

    return app
