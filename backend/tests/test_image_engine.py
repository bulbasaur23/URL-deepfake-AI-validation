"""Task 10 unit tests: the two image engines over injected fakes (offline).

Every offline test injects model/processor fakes — zero downloads, zero GPU.
The fixture images are real tiny JPEGs written to tmp_path via PIL, so the
engines' real file-load path (``PIL open → RGB``) is genuinely exercised; the
model fakes return chosen probabilities, so ``p``/score/confidence/rationale
are pinned exactly.

Seam per engine (constructor injection, the Task 9 pattern):
- ``ViTImageEngine(model, processor)``: model = HF image-classification model
  (``eval()`` + ``__call__(pixel_values=...) -> output.logits``; its config
  carries ``id2label`` — the fakes mirror the real checkpoint's mapping
  ``{"0": "artificial", "1": "human"}``); processor = image processor
  (``__call__(images=..., return_tensors="pt") -> {"pixel_values": tensor}``).
- ``SmaugImageEngine(model, processor)``: model = SentenceTransformer-like
  scorer (``encode([{"image": view}, ...], convert_to_tensor=True) ->
  per-view logits``); processor = optional multimodal processor whose
  image_processor gets the training-time vision budget pinned. Smaug has no
  id2label to derive — its output IS the AI probability (sigmoid of the
  crop-mean logit, pinned by the model card), so nothing can be misread.

The four brief-pinned tests run for BOTH engines (ids ``vit``/``smaug``).
The ``network``-marked smoke tests at the bottom load the real checkpoints on
the RTX 3050 (device asserted); the fixture images are the official Smaug
Space examples shipped by the model author (a real photograph and an AI
collage — sources documented inline). The photographic image doubles as the
inverted-mapping guard for the default engine (score < 65 ⇔ p < 0.65,
brief-pinned). The Smaug smoke is written and ready but skips on this venv:
the checkpoint declares model type ``embedding_gemma2`` (Google
EmbeddingGemma-2 base), which the pinned transformers 4.57.6 predates, and
upgrading transformers is blocked by text_roberta's 4.x-only tokenizer API —
the smoke detects that state and skips with the evidence (dependency
conflict, documented in task-10-report.md).
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import torch
from PIL import Image

from pipeline.engines.base import EngineFault
from pipeline.engines.image_smaug import SmaugImageEngine
from pipeline.engines.image_vit import ViTImageEngine
from pipeline.types import Item

# The real umm-maybe checkpoint's id2label (string keys, HF-native shape):
# the AI class ("artificial") sits on index 0.
_REAL_ID2LABEL = {"0": "artificial", "1": "human"}
_SWAPPED_ID2LABEL = {"0": "human", "1": "artificial"}

_ENGINE_IDS = {"vit": "image-ai-vit", "smaug": "image-ai-smaug"}

# Rationale templates at p = 0.4249 -> "{p:.0%}" = "42%", one per engine,
# pinned byte-for-byte (ruling 6: concrete "{model name}" per engine).
_ENGINE_RATIONALES = {
    "vit": "vit image classifier: diffusion-style artifacts, p=42%",
    "smaug": "smaug image classifier: diffusion-style artifacts, p=42%",
}


class _FakeViTOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeViTModel:
    """Binary image-classification head emitting queued probabilities on ``ai_index``."""

    def __init__(
        self,
        probs: list[float],
        id2label: dict[str, str] | None = None,
        ai_index: int = 0,
    ) -> None:
        self.config = SimpleNamespace(id2label=dict(id2label or _REAL_ID2LABEL))
        self.probs = list(probs)
        self.ai_index = ai_index
        self.seen: list[torch.Tensor] = []

    def eval(self) -> _FakeViTModel:
        return self

    def __call__(self, pixel_values: torch.Tensor | None = None, **_: object) -> _FakeViTOutput:
        self.seen.append(pixel_values)
        p = self.probs[len(self.seen) - 1]
        pair = [p, 1.0 - p] if self.ai_index == 0 else [1.0 - p, p]
        return _FakeViTOutput(torch.log(torch.tensor([pair], dtype=torch.float64)))


class _FakeViTProcessor:
    """Image-processor stand-in encoding the decoded image's (width, height).

    The values only exist after PIL has actually opened + decoded the fixture
    JPEG, so asserting on ``model.seen`` proves the real file-load path.
    """

    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []

    def __call__(
        self, images: Image.Image, return_tensors: str | None = None, **_: object
    ) -> dict[str, torch.Tensor]:
        self.seen_sizes.append((images.width, images.height))
        return {"pixel_values": torch.tensor([[float(images.width), float(images.height)]])}


class _FakeSmaugModel:
    """SentenceTransformer stand-in: one logit per 528×528 crop view.

    ``encode`` receives the ``[{"image": view}, ...]`` batch the engine builds
    from the real fixture image and emits the queued probability as identical
    per-view log-probability logits, so sigmoid(mean(logit)) recovers it.
    """

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.seen_view_counts: list[int] = []

    def encode(
        self,
        inputs: list[dict[str, Image.Image]],
        batch_size: int = 16,
        convert_to_tensor: bool = True,
        **_: object,
    ) -> torch.Tensor:
        self.seen_view_counts.append(len(inputs))
        p = self.probs[len(self.seen_view_counts) - 1]
        return torch.full((len(inputs),), math.log(p / (1.0 - p)), dtype=torch.float64)


class _FakeSmaugProcessor:
    """Multimodal processor stand-in carrying a budgetable image_processor."""

    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(max_soft_tokens=0, image_seq_length=0)


class _ExplodingViTModel(_FakeViTModel):
    def __call__(self, pixel_values: torch.Tensor | None = None, **_: object) -> _FakeViTOutput:
        raise RuntimeError("forward pass exploded")


class _ExplodingSmaugModel(_FakeSmaugModel):
    def encode(
        self,
        inputs: list[dict[str, Image.Image]],
        batch_size: int = 16,
        convert_to_tensor: bool = True,
        **_: object,
    ) -> torch.Tensor:
        raise RuntimeError("encode exploded")


def _build(engine_id: str, probs: list[float], *, explode: bool = False) -> tuple[object, object]:
    """One engine + its fake model for ``engine_id``; fakes make this offline."""
    if engine_id == "vit":
        model = _ExplodingViTModel(probs) if explode else _FakeViTModel(probs)
        return ViTImageEngine(model=model, processor=_FakeViTProcessor()), model
    model = _ExplodingSmaugModel(probs) if explode else _FakeSmaugModel(probs)
    return SmaugImageEngine(model=model), model


def _fixture_jpeg(tmp_path: Path) -> str:
    """A real tiny JPEG on disk (96×64) — exercises the engines' PIL load path."""
    path = tmp_path / "snapshot.jpg"
    Image.new("RGB", (96, 64), (120, 30, 200)).save(path, format="JPEG", quality=90)
    return str(path)


def _image_item(image_path: str | None = None) -> Item:
    return Item(kind="image", label="hero shot", source_ref="img[0]", image_path=image_path)


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_score_maps_label_to_synthetic_prob(engine_id: str, tmp_path: Path) -> None:
    """Chosen model output → p on the synthetic (AI) side → score = 100·p."""
    engine, model = _build(engine_id, [0.9])
    result = asyncio.run(engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    # Both fake paths are fp64 end to end: ViT softmaxes the fake's fp64
    # log-probabilities directly; Smaug casts its logits with .double()
    # (image_smaug.py:198) before the mean + sigmoid. abs=1e-9 pins the exact
    # fp64 round-trip (error ~1e-14 here) — an fp32 cast misses it by
    # ~2.4e-6, so a regression back to .float() fails this test.
    assert result.score == pytest.approx(90.0, abs=1e-9)
    assert result.engine_id == _ENGINE_IDS[engine_id]
    if engine_id == "vit":
        # [96.0, 64.0] proves the real fixture JPEG was opened + decoded and
        # its pixel values flowed through the processor into the model.
        assert model.seen[0].tolist() == [[96.0, 64.0]]
    else:
        # Smaug scores the real image as three 528×528 native-resolution crops.
        assert model.seen_view_counts == [3]


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_confidence_convention_same_as_text(engine_id: str, tmp_path: Path) -> None:
    """confidence = |p − 0.5|·2 — exactly the Task 9 text-engine convention."""
    engine, _ = _build(engine_id, [0.9])
    result = asyncio.run(engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    # fp64 exact on both fake paths — Smaug's logits are .double()-cast
    # (image_smaug.py:198), ViT softmaxes fp64 directly — so abs=1e-9 pins the
    # convention: an fp32 cast misses it by ~4.8e-8 and would fail.
    assert result.confidence == pytest.approx(0.8, abs=1e-9)

    engine_coin, _ = _build(engine_id, [0.5])
    coin = asyncio.run(engine_coin.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert coin.confidence == 0.0  # p = 0.5 is the coin flip
    assert coin.score == pytest.approx(50.0, abs=1e-9)


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_rationale_template(engine_id: str, tmp_path: Path) -> None:
    """Rationale is the brief-exact template with the byte-pinned model name."""
    engine, _ = _build(engine_id, [0.4249])
    result = asyncio.run(engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert result.rationale == _ENGINE_RATIONALES[engine_id]
    # fp64 exact on both fake paths — Smaug's logits are .double()-cast
    # (image_smaug.py:198) — so abs=1e-9 pins the template's %: an fp32 cast
    # misses it by ~2.5e-6 and would fail.
    assert result.score == pytest.approx(42.49, abs=1e-9)  # the template's % is the score

    engine_high, _ = _build(engine_id, [0.994])
    high = asyncio.run(engine_high.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert high.rationale.endswith("p=99%")


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_supports_rejects_text_items(engine_id: str, tmp_path: Path) -> None:
    """supports(): kind == "image" with a non-null image_path — nothing else."""
    engine, _ = _build(engine_id, [0.5])
    assert engine.supports(_image_item(_fixture_jpeg(tmp_path))) is True
    assert engine.supports(Item(kind="text", label="l", source_ref="r", text="hello")) is False
    assert engine.supports(Item(kind="text", label="l", source_ref="r")) is False
    assert engine.supports(Item(kind="image", label="l", source_ref="r")) is False


def test_engine_identity_is_pinned() -> None:
    """Registry ids + the Task 9 "<checkpoint-name>/1.0" version convention."""
    vit, _ = _build("vit", [0.5])
    assert vit.id == "image-ai-vit"
    assert vit.version == "AI-image-detector/1.0"
    smaug, _ = _build("smaug", [0.5])
    assert smaug.id == "image-ai-smaug"
    assert smaug.version == "Smaug-AI-Image-Detector/1.0"


def test_vit_label_mapping_derived_from_config(tmp_path: Path) -> None:
    """The AI class index comes from the loaded config's id2label, never hardcoded.

    With the checkpoint's real mapping ({0: artificial, 1: human}) the queued
    probability sits on index 0; with a swapped mapping it sits on index 1 —
    a hardcoded index would score the swapped case 20 instead of 80.
    """
    engine, model = _build("vit", [0.8])
    result = asyncio.run(engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert result.score == pytest.approx(80.0, abs=1e-6)
    assert model.ai_index == 0

    swapped_model = _FakeViTModel([0.8], id2label=_SWAPPED_ID2LABEL, ai_index=1)
    swapped_engine = ViTImageEngine(model=swapped_model, processor=_FakeViTProcessor())
    swapped = asyncio.run(swapped_engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert swapped.score == pytest.approx(80.0, abs=1e-6)
    assert swapped_model.ai_index == 1


def test_vit_unresolvable_labels_raise_valueerror() -> None:
    """No AI label derivable from id2label → loud ValueError at init.

    Both labels are real-side, so neither the hint match nor the lone
    non-real binary fallback resolves — a wrong-side scorer must never boot
    silently (preload warns + skips instead).
    """
    model = _FakeViTModel([0.5], id2label={"0": "real", "1": "human"})
    with pytest.raises(ValueError):
        ViTImageEngine(model=model, processor=_FakeViTProcessor())


def test_smaug_pins_training_vision_budget() -> None:
    """An injected processor gets the training-time vision budget pinned (140)."""
    processor = _FakeSmaugProcessor()
    SmaugImageEngine(model=_FakeSmaugModel([0.5]), processor=processor)
    assert processor.image_processor.max_soft_tokens == 140
    assert processor.image_processor.image_seq_length == 140


def test_smaug_needs_no_processor_for_fakes() -> None:
    """A fake model alone is a valid construction (nothing to budget-pin)."""
    engine = SmaugImageEngine(model=_FakeSmaugModel([0.5]))
    assert engine.supports(_image_item("x.jpg")) is True


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_missing_snapshot_raises_engine_fault(engine_id: str, tmp_path: Path) -> None:
    """A missing snapshot file is 'broken on this item' → EngineFault, per task."""
    engine, _ = _build(engine_id, [0.5])
    with pytest.raises(EngineFault) as excinfo:
        asyncio.run(engine.analyze(_image_item(str(tmp_path / "missing.jpg"))))
    assert "inference failed" in str(excinfo.value)


@pytest.mark.parametrize("engine_id", ["vit", "smaug"], ids=["vit", "smaug"])
def test_forward_failure_raises_engine_fault(engine_id: str, tmp_path: Path) -> None:
    """A failing forward pass surfaces as EngineFault, not as RuntimeError."""
    engine, _ = _build(engine_id, [0.5], explode=True)
    with pytest.raises(EngineFault) as excinfo:
        asyncio.run(engine.analyze(_image_item(_fixture_jpeg(tmp_path))))
    assert "inference failed" in str(excinfo.value)


# --- network smoke (real checkpoints on the RTX 3050; run `pytest -m network -v`) --

# Both fixtures are the official Smaug Space examples shipped by the model
# author (https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector —
# that demo scores these very images with this very model): a real photograph
# and an AI-generated collage, cached locally so re-runs don't re-download.
_FIXTURE_SOURCES = {
    "real_photo.jpg": (
        "https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector/"
        "resolve/main/examples/real_photo.jpg"
    ),
    "ai_image.jpg": (
        "https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector/"
        "resolve/main/examples/ai_image.jpg"
    ),
}


def _smoke_fixture(name: str) -> str:
    """Fetch a fixture image (once) into a temp cache; source is documented."""
    cache_dir = Path(tempfile.gettempdir()) / "task10-image-fixtures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if not target.exists():
        response = requests.get(_FIXTURE_SOURCES[name], timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
    return str(target)


@pytest.mark.network
def test_real_model_smoke_vit_fixtures_on_cuda() -> None:
    """Real umm-maybe checkpoint on the RTX 3050; photographic image < 65.

    Doubles as the inverted-mapping guard: id2label says {0: "artificial",
    1: "human"} and the clearly-photographic fixture must score LOW — a high
    score means the label derivation is wrong and must be reported, never
    silently flipped. The AI-collage fixture has NO brief-pinned threshold:
    its score is produced and printed for the Task 11 bake-off.
    """
    engine = ViTImageEngine()  # loads the real weights (downloaded once, §11.7)
    assert next(engine._model.parameters()).device.type == "cuda", (
        "the real image checkpoint must run on the RTX 3050 (ruling 4)"
    )
    for name in ("real_photo.jpg", "ai_image.jpg"):
        result = asyncio.run(engine.analyze(_image_item(_smoke_fixture(name))))
        print(
            f"\n[smoke] vit {name}: score={result.score:.2f} "
            f"confidence={result.confidence:.3f} latency_ms={result.latency_ms}"
        )
        if name == "real_photo.jpg":
            assert result.score < 65, (
                f"clearly-photographic image scored {result.score:.1f} — the id2label "
                "mapping may be inverted; investigate and report instead of flipping "
                "any logic"
            )


def _smoke_smaug_unloadable_reason() -> str | None:
    """Why the real Smaug checkpoint cannot load in this venv, or None if it can.

    Evidence-based (verified post-recovery): the checkpoint declares model type
    ``embedding_gemma2`` (Google EmbeddingGemma-2 base, authored against
    sentence-transformers 6.1.0), which transformers 4.57.6 predates — the real
    load fails at ``AutoConfig`` with "Transformers does not recognize this
    architecture". The transformers 5.x line that knows the model removed the
    generic ``prepare_for_model`` text_roberta calls, so upgrading is blocked;
    the constraint is documented in task-10-report.md. A probe that cannot run
    returns None on purpose: the smoke then attempts the real load and fails
    honestly with the actual error instead of skipping on a guess.
    """
    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            return "sentence-transformers is not installed in this venv"
        import transformers
        from transformers.models.auto import configuration_auto

        if "embedding_gemma2" not in configuration_auto.CONFIG_MAPPING:
            return (
                f"transformers {transformers.__version__} does not recognize the Smaug "
                "checkpoint's model type 'embedding_gemma2' (needs the transformers 5.x "
                "line, whose removal of prepare_for_model breaks text_roberta) — "
                "dependency conflict documented in task-10-report.md"
            )
    except Exception:  # noqa: BLE001 - a broken probe must not fake a skip: real load runs
        return None
    return None


@pytest.mark.network
def test_real_model_smoke_smaug_fixtures_on_cuda() -> None:
    """Real Smaug checkpoint on the RTX 3050; both fixture images score.

    The output IS the AI probability (model-card-pinned direction; no
    id2label exists to misread). No threshold is brief-pinned for either
    fixture: both scores are produced and printed for the Task 11 bake-off.
    Skips with the environmental evidence when this venv cannot recognize the
    checkpoint's architecture (see _smoke_smaug_unloadable_reason).
    """
    reason = _smoke_smaug_unloadable_reason()
    if reason:
        pytest.skip(reason)
    engine = SmaugImageEngine()  # loads the real weights (downloaded once, §11.7)
    assert engine._model.device.type == "cuda", (
        "the real Smaug model must run on the RTX 3050 (ruling 4)"
    )
    for name in ("real_photo.jpg", "ai_image.jpg"):
        result = asyncio.run(engine.analyze(_image_item(_smoke_fixture(name))))
        print(
            f"\n[smoke] smaug {name}: score={result.score:.2f} "
            f"confidence={result.confidence:.3f} latency_ms={result.latency_ms}"
        )
        assert 0.0 <= result.score <= 100.0
        assert 0.0 <= result.confidence <= 1.0
