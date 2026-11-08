"""FastAPI application factory."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="AI Content Detector", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Liveness probe; payload shape is refactored at Task 8's model preload."""
        return {"models_loaded": [], "queue_depth": 0}

    return app
