"""Task 8: engine seam tests — contract, registry, preload, health. All offline.

Dummy engines are registered by monkeypatching the static registry map
(``registry._ENGINE_SOURCES`` via ``monkeypatch.setitem`` — restored per
test) and pointing a fresh ``Settings`` at the dummy id; no real model is
downloaded or loaded. Structural assertions (``id``/``version``/behavior)
are used instead of ``isinstance`` on purpose: pytest imports this file as
top-level ``test_engines`` while the registry resolves the dotted namespace
path ``tests.test_engines``, so two distinct class objects share one name.

The not-yet-written Task 10 winner (``image-ai-vit``) stays tolerated at
preload, and the no-weights state is simulated (both winner ids pointed at
nonexistent modules) where tests pin preload tolerance / the Task 1 health
payload — Task 9's text weights are real now and load only via the
network-marked smoke tests, never offline.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import settings as settings_module
from app.main import create_app
from app.settings import Settings
from pipeline.engines import registry
from pipeline.engines.base import EngineFault
from pipeline.engines.registry import UnknownEngineError
from pipeline.preload import preload
from pipeline.types import EngineResult, Item

_DUMMY_ID = "test-dummy-text"
_DUMMY_VERSION = "dummy/1.0"


class _TextDummyEngine:
    """Offline stand-in satisfying the Engine contract structurally (§7.2)."""

    id = _DUMMY_ID
    version = _DUMMY_VERSION

    def supports(self, item: Item) -> bool:
        return item.kind == "text"

    async def analyze(self, item: Item) -> EngineResult:
        return EngineResult(
            engine_id=self.id,
            engine_version=self.version,
            score=42.0,
            confidence=0.5,
            rationale="dummy verdict",
            latency_ms=1,
        )


class _FaultDummyEngine:
    """Contract probe: analyze() raising EngineFault is the broken-item signal."""

    id = "test-dummy-fault"
    version = _DUMMY_VERSION

    def supports(self, item: Item) -> bool:
        return item.kind == "text"

    async def analyze(self, item: Item) -> EngineResult:
        raise EngineFault("model exploded on this item")


def _register_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the static map at the in-file dummy for the duration of one test."""
    monkeypatch.setitem(registry._ENGINE_SOURCES, _DUMMY_ID, "tests.test_engines:_TextDummyEngine")


def test_get_engine_returns_settings_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The modality winner comes from the settings id, resolved through the map."""
    _register_dummy(monkeypatch)
    engine = registry.get_engine("text", Settings(text_engine_id=_DUMMY_ID))
    assert engine.id == _DUMMY_ID
    assert engine.version == _DUMMY_VERSION
    text_item = Item(kind="text", label="block 1", source_ref="p[0]", text="hello")
    assert engine.supports(text_item) is True
    assert engine.supports(Item(kind="image", label="", source_ref="img[0]")) is False


def test_get_engine_unknown_settings_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A settings id absent from the map fails at the startup path (§11.6)."""
    with pytest.raises(UnknownEngineError):
        registry.get_engine("text", Settings(text_engine_id="nope"))


def test_get_engine_rejects_unknown_modality() -> None:
    """Only the two modalities with settings fields are legal."""
    with pytest.raises(ValueError):
        registry.get_engine("nope", Settings())


def test_get_by_id_reaches_non_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bake-off (Task 11) resolves non-winner ids through the same map."""
    _register_dummy(monkeypatch)
    engine = registry.get_by_id(_DUMMY_ID, Settings())
    assert engine.id == _DUMMY_ID
    assert engine.version == _DUMMY_VERSION


def test_get_by_id_unknown_id_raises() -> None:
    with pytest.raises(UnknownEngineError):
        registry.get_by_id("nope", Settings())


def test_get_by_id_unimportable_class_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tolerance for unimportable entries lives in preload ONLY.

    Task 9's text module exists now (its real-weight load is exercised by the
    network-marked smoke tests, never offline), so the boundary is pinned
    with a synthetic registry entry whose module does not exist: direct
    resolution must surface the ImportError instead of hiding it.
    """
    monkeypatch.setitem(
        registry._ENGINE_SOURCES, "missing-engine", "tests._no_such_engine:_MissingEngine"
    )
    with pytest.raises(ImportError):
        registry.get_by_id("missing-engine", Settings())


def test_analyze_engine_fault_is_the_broken_item_signal() -> None:
    """Raising EngineFault inside analyze() is the documented 'broken here' contract."""
    engine = _FaultDummyEngine()
    item = Item(kind="text", label="block 1", source_ref="p[0]", text="hello")
    assert engine.supports(item) is True
    with pytest.raises(EngineFault):
        asyncio.run(engine.analyze(item))


def test_preload_tolerates_unimportable_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preload skips winner classes that cannot import/load: warning + skip.

    Task 9's text module now exists, so the no-weights state is simulated by
    pointing both default winner ids at nonexistent modules (image-ai-vit
    stays unwritten until Task 10 anyway): both are tolerated, and
    ``models_loaded`` stays short (Task 1's health pin).
    """
    for engine_id in ("text-ai-roberta", "image-ai-vit"):
        monkeypatch.setitem(
            registry._ENGINE_SOURCES, engine_id, "tests._no_such_engine:_MissingEngine"
        )
    assert preload(Settings()) == {}


def test_preload_returns_versions_for_loadable_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_dummy(monkeypatch)
    loaded = preload(Settings(text_engine_id=_DUMMY_ID))
    assert loaded == {_DUMMY_ID: _DUMMY_VERSION}


def test_preload_unknown_settings_id_fails_startup() -> None:
    """Typo detection survives preload: unknown ids crash startup on purpose."""
    with pytest.raises(UnknownEngineError):
        preload(Settings(image_engine_id="nope"))


def test_health_no_weights_state_matches_task1_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real lifespan wiring reproduces Task 1's exact no-weights payload.

    Default settings, with both winner ids pointed at nonexistent modules
    (the no-weights state; Task 9's text weights are real now and only load
    via the network-marked tests / a booted app): preload tolerates both,
    and health answers ``{"models_loaded": [], "queue_depth": 0}``.
    """
    for engine_id in ("text-ai-roberta", "image-ai-vit"):
        monkeypatch.setitem(
            registry._ENGINE_SOURCES, engine_id, "tests._no_such_engine:_MissingEngine"
        )
    with TestClient(create_app()) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"models_loaded": [], "queue_depth": 0}


def test_health_reports_preloaded_winner_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a winner class loads, its version lands in models_loaded."""
    _register_dummy(monkeypatch)
    monkeypatch.setattr(settings_module.settings, "text_engine_id", _DUMMY_ID)
    with TestClient(create_app()) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"models_loaded": [_DUMMY_VERSION], "queue_depth": 0}
