"""FastAPI application factory.

Startup wiring (Task 8): a FastAPI ``lifespan`` hook — the modern lifespan
protocol, chosen over deprecated ``on_event("startup")`` — runs
``pipeline.preload.preload(settings)`` once at boot, so model load happens
off the per-scan clock (design §11.7) and the loaded engine versions land on
``app.state.models_loaded`` for the health probe.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.settings import settings
from pipeline.preload import preload


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Preload the winner engines at boot; overwrite the create_app defaults."""
    app.state.models_loaded = preload(settings)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AI Content Detector", version="0.1.0", lifespan=lifespan)

    # Defaults so /api/health answers even before (or without) the lifespan
    # hook having run; the lifespan overwrite carries the real preload result.
    app.state.models_loaded = {}
    # Placeholder counter Task 12's ScanJobManager owns (always 0 until then).
    app.state.queue_depth = 0

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Liveness + model-preload probe.

        Payload pinned by Task 1: ``models_loaded`` lists winner engine
        versions (empty until a winner class import+loads); ``queue_depth``
        is the placeholder owned by Task 12's ScanJobManager (0 until then).
        """
        return {
            "models_loaded": list(app.state.models_loaded.values()),
            "queue_depth": app.state.queue_depth,
        }

    return app
