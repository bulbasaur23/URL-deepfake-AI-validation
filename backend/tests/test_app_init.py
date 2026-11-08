from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_empty_engines():
    r = TestClient(create_app()).get("/api/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == []  # engines plug in at Task 8
