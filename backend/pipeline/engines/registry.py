"""Engine registry (Task 8): static id -> class map, resolved lazily.

The map is a STATIC dict of engine id -> ``"module.path:ClassName"`` string
covering ALL engine ids — winners and bake-off candidates alike — so the
Task 11 offline bake-off can reach non-winner engines via ``get_by_id``.
Entries are resolved with importlib at call time: importing this module never
imports an engine module, and nothing here touches model weights.

Rules (design §11.6 + §11.15, Task 8 brief):
- a settings id ABSENT from this map raises ``UnknownEngineError`` at the
  startup path (typo detection — ``preload`` re-raises it, it does not
  swallow it);
- a map entry whose class cannot import/load is tolerated ONLY by
  ``preload`` (warning + skip so ``models_loaded`` stays short and health
  stays answerable before Task 9/10 weights exist); direct ``get_engine`` /
  ``get_by_id`` calls surface the import error to the caller.

The Task 9/10 classes are registered before they exist on purpose: the
strings pin the module paths those tasks must create, and preload tolerates
their absence until then.
"""

from __future__ import annotations

import importlib
from typing import Literal

from app.settings import Settings
from pipeline.engines.base import Engine

# Static engine map (design §7.2: "adding an engine = new class + registration").
_ENGINE_SOURCES: dict[str, str] = {
    "text-ai-roberta": "pipeline.engines.roberta_text:RobertaTextEngine",
    "image-ai-vit": "pipeline.engines.vit_image:ViTImageEngine",
    "image-ai-smaug": "pipeline.engines.smaug_image:SmaugImageEngine",
}

# Modality -> the Settings field that names that modality's winner.
_MODALITY_FIELD: dict[str, str] = {"text": "text_engine_id", "image": "image_engine_id"}


class UnknownEngineError(LookupError):
    """An engine id is absent from the static map (typo detection, §11.6)."""


def _engine_class(engine_id: str) -> type:
    """Import and return the engine class registered under ``engine_id``.

    Raises ``UnknownEngineError`` for ids absent from the map; import /
    attribute failures propagate to the caller (preload is the only
    tolerating consumer).
    """
    source = _ENGINE_SOURCES.get(engine_id)
    if source is None:
        raise UnknownEngineError(
            f"unknown engine id {engine_id!r}: not in the registry map "
            f"(known ids: {sorted(_ENGINE_SOURCES)})"
        )
    module_path, _, class_name = source.partition(":")
    if not module_path or not class_name:
        raise UnknownEngineError(f"malformed registry entry for {engine_id!r}: {source!r}")
    return getattr(importlib.import_module(module_path), class_name)


def get_by_id(engine_id: str, settings: Settings) -> Engine:
    """Resolve + instantiate the engine registered under ``engine_id``.

    ``settings`` is part of the dispatch-pinned signature for parity with
    ``get_engine`` (and future construction needs such as device selection);
    it is deliberately unused today — the Task 6 ``langid.detect`` precedent
    for a typed-but-unused parameter. Import errors on not-yet-written
    engine classes propagate (only ``preload`` tolerates them).
    """
    return _engine_class(engine_id)()


def get_engine(modality: Literal["text", "image"], settings: Settings) -> Engine:
    """Return the single winner engine for ``modality`` from the settings id.

    Unknown settings ids raise here (and in preload) so a typo'd id fails at
    startup — never per scan (§11.6); a non-modal ``modality`` raises
    ``ValueError``.
    """
    field_name = _MODALITY_FIELD.get(modality)
    if field_name is None:
        raise ValueError(f"unsupported modality {modality!r}; expected 'text' or 'image'")
    engine_id = getattr(settings, field_name)
    if engine_id not in _ENGINE_SOURCES:
        raise UnknownEngineError(
            f"settings.{field_name}={engine_id!r} is not a registered engine id "
            f"(known ids: {sorted(_ENGINE_SOURCES)})"
        )
    return get_by_id(engine_id, settings)
