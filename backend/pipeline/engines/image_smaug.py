"""Task 10: the Smaug image-engine candidate — ``abacusai/Smaug-AI-Image-Detector``.

A 439M-parameter EmbeddingGemma-2 fine-tune (vision-only, Apache-2.0,
abacus.ai) with a linear classification head, distributed as a
sentence-transformers model. Its output contract is unusual among the
candidates: there is NO id2label to derive — the model card pins the output
as the probability that the image is AI-generated. The repo ships an
inference recipe (``smaug_image_detect.py``, Apache-2.0); the scoring flow
below replicates it faithfully and self-contained (ruling 1: no shared
helper module):

- the snapshot is EXIF-transposed to RGB, the shorter side is scaled to
  576 px (upscaled to 528 when smaller), and three 528×528 crops are taken
  at native resolution (start/centre/end of the long side) — pixel-level
  traces are what the detector reads, so images are never squeezed;
- the SentenceTransformer encodes the crop batch (the model's own processor
  tokenizes images; the training-time vision budget of 140 tokens is pinned
  on the processor at load, mirroring the shipped recipe);
- the per-crop logits are averaged and squashed through a sigmoid — that
  single number is ``p``;
- score = 100·p; confidence = |p − 0.5|·2 — exactly the Task 9 convention,
  kept consistent for cross-engine comparability;
- rationale is the brief-exact template
  ``"smaug image classifier: diffusion-style artifacts, p={p:.0%}"``.

Device policy (ruling 4): the checkpoint is loaded straight onto ``cuda``
when available (bfloat16 there, float32 elsewhere — the model is unstable in
fp16), so every encoded input lands on the model's device inside the encode
call; CPU-created tensors + GPU model cannot fault per-item by construction.

Dependency note (reported, verified post-recovery): the checkpoint declares
model type ``embedding_gemma2`` (Google EmbeddingGemma-2 base) and was authored
with sentence-transformers 6.1.0 — transformers 4.57.6 (this venv's pin)
predates that architecture, so the no-argument load fails at ``AutoConfig``
before any weights move ("Transformers does not recognize this architecture").
The transformers 5.x line that knows the model removed the generic
``prepare_for_model`` text_roberta calls, and the repo ships no remote code
(no ``configuration_*.py``/``modeling_*.py`` — ``trust_remote_code`` is not an
escape hatch), so the two real loads cannot share this venv. The
sentence-transformers import is therefore LAZY (inside the no-argument
branch): the module imports and the offline tests run dependency-free on
injected fakes, while a no-argument construction surfaces exactly the "class
cannot load" state preload tolerates (warning + skip). The network smoke
skips on precisely this evidence (see test_image_engine.py).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image, ImageOps

from pipeline.engines.base import Engine, EngineFault
from pipeline.types import EngineResult, Item

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

CHECKPOINT = "abacusai/Smaug-AI-Image-Detector"

# Must match training (the shipped smaug_image_detect.py): shorter side -> 576,
# 528×528 crops, 140 vision tokens per crop.
_CROP, _SIDE, _BUDGET = 528, 576, 140

# The rationale's "{model name}" slot (ruling 6), byte-pinned by the tests.
MODEL_NAME = "smaug image classifier"


def _views(image: Image.Image, crops: int = 3) -> list[Image.Image]:
    """The shipped recipe's views: shorter side to 576 px (upscaled to 528 when
    smaller), then ``crops`` 528×528 crops at native resolution.

    ``crops=3`` covers start, centre and end of the long side; ``crops=1`` is
    centre only. Replicated from the repo's ``smaug_image_detect.py`` so this
    engine stays self-contained.
    """
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    width, height = prepared.size
    if min(width, height) > _SIDE:
        scale = _SIDE / min(width, height)
        prepared = prepared.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    elif min(width, height) < _CROP:
        scale = _CROP / min(width, height)
        prepared = prepared.resize(
            (math.ceil(width * scale), math.ceil(height * scale)), Image.BICUBIC
        )
    width, height = prepared.size
    center_x, center_y = (width - _CROP) // 2, (height - _CROP) // 2
    if crops == 1:
        return [prepared.crop((center_x, center_y, center_x + _CROP, center_y + _CROP))]
    if width >= height:
        return [
            prepared.crop((x, center_y, x + _CROP, center_y + _CROP))
            for x in (0, center_x, width - _CROP)
        ]
    return [
        prepared.crop((center_x, y, center_x + _CROP, center_y + _CROP))
        for y in (0, center_y, height - _CROP)
    ]


class SmaugImageEngine(Engine):
    """The §7.2 Smaug bake-off candidate: EmbeddingGemma-2 vision scorer over crops."""

    id = "image-ai-smaug"
    version = "Smaug-AI-Image-Detector/1.0"  # Task 9 convention: "<checkpoint-name>/1.0"

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        processor: Any | None = None,
    ) -> None:
        """Load the real checkpoint once (preload path) or take injected fakes.

        ``model`` is the SentenceTransformer-like scorer (``encode([{"image":
        view}, ...], convert_to_tensor=True)`` → per-view logits);
        ``processor`` is the multimodal processor whose image_processor gets
        the training-time vision budget pinned. With ``model=None`` the real
        checkpoint downloads (first run ~1 GB) and the processor comes from
        it; a fake model needs no processor (offline tests inject none), and
        an injected one is budget-pinned like the real one.
        """
        if model is None:
            from sentence_transformers import SentenceTransformer  # lazy: optional dep

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(
                CHECKPOINT,
                device=device,
                config_kwargs={"audio_config": None},  # audio tower removed upstream
                model_kwargs={
                    "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32
                },
            )
            processor = model[0].processor
        self._model = model
        self._processor = processor
        if processor is not None:
            self._pin_vision_budget(processor)

    @staticmethod
    def _pin_vision_budget(processor: Any) -> None:
        """Pin the training-time vision budget on the processor (140 tokens).

        Mirrors the shipped recipe: the encoded image must produce exactly the
        token budget the model was trained with.
        """
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return
        image_processor.max_soft_tokens = _BUDGET
        image_processor.image_seq_length = _BUDGET

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
        """Snapshot JPEG → RGB → native-resolution crops → mean logit → sigmoid p.

        The model was loaded ON its device, so the encode call owns input
        placement (ruling 4); the sigmoid output is the model-card-pinned
        probability that the image is AI-generated — no id2label exists here
        to misread.
        """
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        views = _views(image, crops=3)
        flat = [{"image": view} for view in views]
        with torch.no_grad():
            logits = self._model.encode(flat, batch_size=16, convert_to_tensor=True)
        crop_logits = logits.double().view(1, len(views))
        return float(torch.sigmoid(crop_logits.mean(1))[0].item())
