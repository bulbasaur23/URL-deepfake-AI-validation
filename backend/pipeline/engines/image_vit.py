"""Task 10: the default image engine — ``umm-maybe/AI-image-detector``.

An AutoTrain-trained Swin image-classification head (the checkpoint's config
declares ``SwinForImageClassification`` — the "AI-image-detector" name is a
ViT-era leftover; id2label = {0: "artificial", 1: "human"}). The only network
touch this engine ever makes is downloading its weights at first load (§11.7:
preload instantiates once at boot, off the 120 s scan clock); ``analyze`` is
pure inference. ``__init__`` loads the real checkpoint when called with no
arguments (the preload/registry path does exactly that) and accepts injected
model/processor fakes for offline tests and future re-wiring.

Scoring (brief-pinned):
- the item's re-encoded snapshot JPEG (Task 7 wrote ``media/<sha256>.jpg``)
  is opened with PIL and converted to RGB;
- the image processor renders it to the model's input tensors, which are
  moved to the model's device at inference — load device is ``cuda`` when
  available, else cpu (ruling 4 bakes in the T9 carry-forward: CPU-created
  tensors + GPU model would fault per-item);
- one forward pass; softmax over the head yields the synthetic probability
  ``p`` of the AI class, whose index is derived from the loaded config's
  ``id2label`` (never hardcoded): an AI-hint label
  (artificial|ai|synthetic|diffusion|generated, without a real-side hint)
  wins, else the lone non-real label of a binary head; unresolvable →
  ``ValueError`` at init so a wrong-side scorer cannot boot silently
  (preload warns + skips instead);
- score = 100·p; confidence = |p − 0.5|·2 — exactly the Task 9 convention,
  kept consistent for cross-engine comparability;
- rationale is the brief-exact template
  ``"vit image classifier: diffusion-style artifacts, p={p:.0%}"``.

``EngineFault`` is raised only when inference genuinely fails on an item
(unsupported item, unreadable snapshot, forward-pass failure) — never for
import problems; preload tolerates those separately.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline.engines.base import Engine, EngineFault
from pipeline.types import EngineResult, Item

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.image_processing_utils import BaseImageProcessor

CHECKPOINT = "umm-maybe/AI-image-detector"

# Substrings identifying the AI class in a config's id2label values (labels
# are lowercased before matching); a real-side hint disqualifies the match.
_AI_LABEL_HINTS = ("artificial", "ai", "synthetic", "diffusion", "generated")
_HUMAN_LABEL_HINTS = ("human", "real", "natural", "authentic")

# The rationale's "{model name}" slot (ruling 6), byte-pinned by the tests.
MODEL_NAME = "vit image classifier"


def _ai_label_index(config: Any) -> int:
    """Index of the AI class in the loaded head, derived from ``config.id2label``.

    Matches an AI hint (and no real-side hint) first; falls back to the lone
    non-real label of a binary head. Raises ``ValueError`` when neither
    resolves — an unidentifiable head must fail at init (preload logs + skips
    the engine) rather than silently score the wrong class.
    """
    id2label = getattr(config, "id2label", None) or {}
    labels: dict[int, str] = {}
    for key, value in id2label.items():
        try:
            labels[int(key)] = str(value).strip().lower()
        except TypeError, ValueError:
            continue
    for index, label in sorted(labels.items()):
        is_ai = any(hint in label for hint in _AI_LABEL_HINTS)
        is_real = any(hint in label for hint in _HUMAN_LABEL_HINTS)
        if is_ai and not is_real:
            return index
    real_indexes = [
        index for index, label in labels.items() if any(h in label for h in _HUMAN_LABEL_HINTS)
    ]
    if len(labels) == 2 and len(real_indexes) == 1:
        return next(index for index in labels if index != real_indexes[0])
    raise ValueError(f"cannot identify the AI class in id2label={id2label!r}")


def _load_device() -> torch.device:
    """Where a freshly loaded checkpoint goes: cuda when available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_device(model: Any) -> torch.device:
    """The device the model's weights live on (its first parameter).

    Models with no parameters (injected offline fakes) resolve to CPU, keeping
    the offline tests zero-GPU regardless of the host's CUDA state.
    """
    try:
        first = next(model.parameters())
    except AttributeError, StopIteration, TypeError, RuntimeError:
        return torch.device("cpu")
    device = getattr(first, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


class ViTImageEngine(Engine):
    """The §7.2 default image candidate: Swin classification head over snapshots."""

    id = "image-ai-vit"
    version = "AI-image-detector/1.0"  # Task 9 convention: "<checkpoint-name>/1.0"

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        processor: BaseImageProcessor | None = None,
    ) -> None:
        """Load the real checkpoint once (preload path) or take injected fakes.

        With no arguments the HF checkpoint downloads (first run ~80 MB) and
        loads onto ``cuda`` when available, else cpu; inference inputs move to
        the model's own device either way (ruling 4). Tests inject both fakes
        for zero-download, zero-GPU runs.
        """
        if model is None or processor is None:
            model = AutoModelForImageClassification.from_pretrained(CHECKPOINT).to(_load_device())
            processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
        self._model = model
        self._model.eval()
        self._processor = processor
        self._ai_index = _ai_label_index(model.config)
        self._device = _model_device(model)

    def supports(self, item: Item) -> bool:
        """True for image items carrying a re-encoded snapshot path; nothing else."""
        return item.kind == "image" and item.image_path is not None

    async def analyze(self, item: Item) -> EngineResult:
        """Score one image item; wall-clock latency measured inside this call."""
        started_ns = time.perf_counter_ns()
        if not self.supports(item):
            raise EngineFault(
                f"{self.id}: unsupported item (kind={item.kind!r}, image_path required)"
            )
        try:
            p = self._infer(item.image_path or "")
        except EngineFault:
            raise
        except Exception as exc:  # broken on THIS item -> degrade it, never the scan
            raise EngineFault(f"{self.id}: inference failed on this item: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
        return EngineResult(
            engine_id=self.id,
            engine_version=self.version,
            score=100.0 * p,
            confidence=min(1.0, max(0.0, abs(p - 0.5) * 2)),
            rationale=f"{MODEL_NAME}: diffusion-style artifacts, p={p:.0%}",
            latency_ms=latency_ms,
        )

    def _infer(self, path: str) -> float:
        """Snapshot JPEG → RGB → processor → forward → softmax probability of the AI class.

        Every tensor the processor returns is moved to the model's device
        (ruling 4: never assume the load device still matches at inference).
        """
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        inputs = {
            key: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for key, value in self._processor(images=image, return_tensors="pt").items()
        }
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return float(torch.softmax(logits, dim=-1)[0, self._ai_index].item())
