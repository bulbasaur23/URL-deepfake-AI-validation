"""Task 9 unit tests: the roberta text engine over injected fakes (offline).

Every unit test builds ``RobertaTextEngine(tokenizer=..., model=...)`` with
fakes — zero downloads, zero GPU. The fakes pin the two interfaces the engine
consumes:

- fake tokenizer: ``model_max_length`` + ``num_special_tokens_to_add()`` (the
  real 512-token limit minus special-token overhead defines the chunk
  budget), ``encode(text, add_special_tokens=False)`` (1 token per character,
  so chunk boundaries are exact), and ``prepare_for_model(ids, ...)`` (adds
  the <s>/</s> overhead like the real RoBERTa tokenizer).
- fake model: a binary classification head whose config carries ``id2label``
  (string keys, mirroring the real checkpoint); every forward call pops the
  next probability from a queue and returns log-probability logits, so
  softmax recovers that probability exactly on the configured AI index.

The two ``network``-marked smoke tests at the bottom load the real
``Hello-SimpleAI/chatgpt-detector-roberta`` checkpoint (~500 MB first run)
and pin the scoring bands: a hand-written paragraph must land in the
authentic band (score < 35) and a canned ChatGPT-style paragraph in the
synthetic band (>= 65). The hand-written assertion doubles as the
inverted-mapping guard: a high score there means the label derivation is
wrong and must be reported — the logic is never silently flipped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from pipeline.engines.base import EngineFault
from pipeline.engines.text_roberta import RobertaTextEngine
from pipeline.types import Item

# Fake RoBERTa special-token ids (bos=0, eos=2 mirror the real checkpoint).
_BOS_ID = 0
_EOS_ID = 2

# id2label shapes seen in the wild: the real checkpoint's mapping, and a
# swapped one proving the engine derives the AI index instead of hardcoding.
_HUMAN_ID2LABEL = {"0": "Human", "1": "ChatGPT"}
_SWAPPED_ID2LABEL = {"0": "ChatGPT", "1": "Human"}


class _FakeTokenizer:
    """1-token-per-character stand-in with configurable special-token overhead."""

    def __init__(
        self,
        model_max_length: int = 512,
        special_count: int = 0,
        token_ids: list[int] | None = None,
    ) -> None:
        self.model_max_length = model_max_length
        self._special_count = special_count
        self._token_ids = token_ids  # when set, encode() returns it verbatim

    def num_special_tokens_to_add(self) -> int:
        return self._special_count

    def encode(self, text: str, add_special_tokens: bool = True, verbose: bool = True) -> list[int]:
        if self._token_ids is not None:
            return list(self._token_ids)
        ids = [ord(char) % 500 + 1 for char in text]
        if add_special_tokens and self._special_count:
            return [_BOS_ID, *ids, _EOS_ID]
        return ids

    def prepare_for_model(
        self, ids: list[int], add_special_tokens: bool = True, **_: object
    ) -> dict[str, torch.Tensor]:
        seq = list(ids)
        if add_special_tokens and self._special_count:
            seq = [_BOS_ID, *seq, _EOS_ID]
        return {"input_ids": torch.tensor([seq], dtype=torch.long)}


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Binary head emitting queued per-chunk AI probabilities on ``ai_index``."""

    def __init__(
        self,
        chunk_probs: list[float],
        id2label: dict[str, str] | None = None,
        ai_index: int = 1,
    ) -> None:
        self.config = SimpleNamespace(id2label=dict(id2label or _HUMAN_ID2LABEL))
        self.chunk_probs = list(chunk_probs)
        self.ai_index = ai_index
        self.seen: list[list[int]] = []

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, input_ids: torch.Tensor | None = None, **_: object) -> _FakeOutput:
        self.seen.append(input_ids.tolist()[0])
        p = self.chunk_probs[len(self.seen) - 1]
        probs = [p, 1.0 - p] if self.ai_index == 0 else [1.0 - p, p]
        return _FakeOutput(torch.log(torch.tensor([probs], dtype=torch.float64)))


class _ExplodingModel(_FakeModel):
    """Forward pass that always fails — EngineFault (not RuntimeError) escapes."""

    def __call__(self, input_ids: torch.Tensor | None = None, **_: object) -> _FakeOutput:
        raise RuntimeError("forward pass exploded")


def _text(chars: int) -> str:
    """Deterministic filler of exactly ``chars`` characters (1 token per char)."""
    return ("lorem ipsum dolor sit amet " * (chars // 27 + 1))[:chars]


def _engine(
    chunk_probs: list[float],
    *,
    special_count: int = 0,
    model_max_length: int = 512,
    id2label: dict[str, str] | None = None,
    ai_index: int = 1,
    token_ids: list[int] | None = None,
) -> tuple[RobertaTextEngine, _FakeModel]:
    tokenizer = _FakeTokenizer(
        model_max_length=model_max_length, special_count=special_count, token_ids=token_ids
    )
    model = _FakeModel(chunk_probs, id2label=id2label, ai_index=ai_index)
    return RobertaTextEngine(tokenizer=tokenizer, model=model), model


def _item(text: str) -> Item:
    return Item(kind="text", label="p[0]", source_ref="p[0]", text=text)


def test_chunking_length_weighted_mean() -> None:
    """3 chunks (512/512/100 content tokens) with known probs -> the exact score."""
    engine, model = _engine([0.2, 0.6, 0.9])
    result = asyncio.run(engine.analyze(_item(_text(1124))))
    expected = 100.0 * (0.2 * 512 + 0.6 * 512 + 0.9 * 100) / (512 + 512 + 100)
    assert result.score == pytest.approx(expected, abs=1e-6)
    # confidence = mean(|p - 0.5| * 2) = (0.6 + 0.2 + 0.8) / 3
    assert result.confidence == pytest.approx(1.6 / 3, abs=1e-9)
    assert [len(ids) for ids in model.seen] == [512, 512, 100]  # exact budget, no overlap
    assert result.engine_id == "text-ai-roberta"
    assert result.engine_version == "chatgpt-detector-roberta/1.0"
    assert isinstance(result.latency_ms, int) and result.latency_ms >= 0


def test_multiple_chunks_mean() -> None:
    """Two equal-length chunks (0.1, 0.7) -> plain mean 0.4 -> score 40."""
    engine, model = _engine([0.1, 0.7])
    result = asyncio.run(engine.analyze(_item(_text(1024))))
    assert result.score == pytest.approx(40.0, abs=1e-9)
    assert result.confidence == pytest.approx((0.8 + 0.4) / 2, abs=1e-9)
    assert [len(ids) for ids in model.seen] == [512, 512]


def test_rationale_template() -> None:
    """Rationale is the brief-exact template showing the item-level weighted p."""
    engine, _ = _engine([0.4249])
    result = asyncio.run(engine.analyze(_item(_text(300))))
    assert result.rationale == (
        "roberta classifier: 42% probability ChatGPT-generated (HC3-trained, v1.0)"
    )
    assert result.score == pytest.approx(42.49, abs=1e-6)  # the template's % is the score

    engine, _ = _engine([0.994])
    result = asyncio.run(engine.analyze(_item(_text(300))))
    assert result.rationale == (
        "roberta classifier: 99% probability ChatGPT-generated (HC3-trained, v1.0)"
    )


def test_confidence_monotonic_edges() -> None:
    """p = 0.5 -> confidence 0.0 (coin flip); p = 0.99 -> confidence ~= 1.0."""
    coin_engine, _ = _engine([0.5])
    coin = asyncio.run(coin_engine.analyze(_item(_text(300))))
    assert coin.confidence == 0.0
    assert coin.score == pytest.approx(50.0, abs=1e-9)

    sure_engine, _ = _engine([0.99])
    sure = asyncio.run(sure_engine.analyze(_item(_text(300))))
    assert sure.confidence == pytest.approx(1.0, abs=0.03)  # |0.99 - 0.5| * 2 = 0.98
    assert sure.confidence > coin.confidence


def test_chunk_spans_respect_max_length() -> None:
    """Special-token overhead eats into the budget: real limit 512 -> 510 content.

    2000 content tokens split into [510, 510, 510, 470]; every prepared span
    (with <s>/</s> added) stays <= 512, and the content total is preserved —
    no overlap, no loss. The score weights by CONTENT tokens, not by the
    special-token-padded span length.
    """
    engine, model = _engine([0.3, 0.5, 0.7, 0.9], special_count=2)
    result = asyncio.run(engine.analyze(_item(_text(2000))))
    assert [len(ids) for ids in model.seen] == [512, 512, 512, 472]
    assert all(len(ids) <= 512 for ids in model.seen)
    assert sum(len(ids) - 2 for ids in model.seen) == 2000
    # (0.3*510 + 0.5*510 + 0.7*510 + 0.9*470) / 2000 = 0.594
    assert result.score == pytest.approx(59.4, abs=1e-6)


def test_supports_contract() -> None:
    """supports(): kind == "text" with non-empty text — nothing else."""
    engine, _ = _engine([0.5])
    assert engine.supports(Item(kind="text", label="l", source_ref="r", text="hello")) is True
    assert engine.supports(Item(kind="text", label="l", source_ref="r", text="")) is False
    assert engine.supports(Item(kind="text", label="l", source_ref="r")) is False
    assert engine.supports(Item(kind="image", label="l", source_ref="r")) is False


def test_engine_identity_is_pinned() -> None:
    engine, _ = _engine([0.5])
    assert engine.id == "text-ai-roberta"
    assert engine.version == "chatgpt-detector-roberta/1.0"


def test_unsupported_item_raises_engine_fault() -> None:
    """The broken-item contract: unsupported items fault, never crash the scan."""
    engine, _ = _engine([0.5])
    with pytest.raises(EngineFault):
        asyncio.run(engine.analyze(Item(kind="image", label="img", source_ref="img[0]")))
    with pytest.raises(EngineFault):
        asyncio.run(engine.analyze(Item(kind="text", label="p", source_ref="p[0]")))


def test_empty_tokenization_raises_engine_fault() -> None:
    """A tokenizer yielding zero tokens is an engine fault on this item."""
    engine, _ = _engine([0.5], token_ids=[])
    with pytest.raises(EngineFault):
        asyncio.run(engine.analyze(_item(_text(300))))


def test_inference_failure_raises_engine_fault() -> None:
    """A failing forward pass surfaces as EngineFault, not as RuntimeError."""
    tokenizer = _FakeTokenizer()
    engine = RobertaTextEngine(tokenizer=tokenizer, model=_ExplodingModel([0.5]))
    with pytest.raises(EngineFault) as excinfo:
        asyncio.run(engine.analyze(_item(_text(300))))
    assert "inference failed" in str(excinfo.value)


def test_label_mapping_derived_from_config() -> None:
    """The AI class index comes from the loaded config's id2label, never hardcoded.

    With the checkpoint's real mapping ({0: Human, 1: ChatGPT}) the queued
    probability sits on index 1; with a swapped mapping it sits on index 0 —
    the engine must follow the config either way (a hardcoded index would
    score the swapped case 20 instead of 80).
    """
    engine, model = _engine([0.8])
    result = asyncio.run(engine.analyze(_item(_text(300))))
    assert result.score == pytest.approx(80.0, abs=1e-6)
    assert model.ai_index == 1

    swapped_engine, swapped_model = _engine([0.8], id2label=_SWAPPED_ID2LABEL, ai_index=0)
    swapped = asyncio.run(swapped_engine.analyze(_item(_text(300))))
    assert swapped.score == pytest.approx(80.0, abs=1e-6)
    assert swapped_model.ai_index == 0


# --- network smoke (real checkpoint; run with `pytest -m network -v`) --------

_HANDWRITTEN_PARAGRAPH = (
    "I've been meaning to fix the porch step for weeks now. Every time I walk out "
    "with the toolbox it either starts raining or I remember I still haven't bought "
    "the right screws — the old ones stripped years ago and dad always said to just "
    "use slightly bigger ones and they'd bite. Didn't listen, obviously. Anyway the "
    "dog nearly went through it yesterday, so this weekend it's getting done no "
    "matter what the weather says."
)

_CHATGPT_STYLE_PARAGRAPH = (
    "Maintaining your outdoor living space requires consistent attention and a "
    "proactive approach. First, assess the structural integrity of each element, "
    "including steps and railings, to identify potential hazards before they "
    "escalate. Second, gather the appropriate tools and materials before beginning "
    "any repair to ensure an efficient workflow. Finally, establish a regular "
    "maintenance schedule, as timely interventions not only extend the lifespan of "
    "your property but also enhance its overall safety and aesthetic appeal. By "
    "following these steps, homeowners can enjoy a well-maintained environment "
    "year-round."
)


@pytest.mark.network
def test_real_model_smoke_handwritten_lands_authentic_band() -> None:
    """Real checkpoint: hand-written English scores < 35 (authentic band).

    Doubles as the inverted-mapping guard — a high score here means the label
    derivation is wrong and must be reported, never silently flipped.
    """
    engine = RobertaTextEngine()  # loads the real weights (downloaded once, §11.7)
    result = asyncio.run(
        engine.analyze(
            Item(kind="text", label="hand-written", source_ref="test", text=_HANDWRITTEN_PARAGRAPH)
        )
    )
    print(
        f"\n[smoke] hand-written: score={result.score:.2f} "
        f"confidence={result.confidence:.3f} latency_ms={result.latency_ms}"
    )
    assert result.score < 35, (
        f"hand-written paragraph scored {result.score:.1f} — the label mapping may be "
        "inverted; investigate and report instead of flipping any logic"
    )


@pytest.mark.network
def test_real_model_smoke_chatgpt_style_lands_synthetic_band() -> None:
    """Real checkpoint: canned ChatGPT-style text scores >= 65 (synthetic band)."""
    engine = RobertaTextEngine()
    result = asyncio.run(
        engine.analyze(
            Item(
                kind="text",
                label="chatgpt-style",
                source_ref="test",
                text=_CHATGPT_STYLE_PARAGRAPH,
            )
        )
    )
    print(
        f"\n[smoke] chatgpt-style: score={result.score:.2f} "
        f"confidence={result.confidence:.3f} latency_ms={result.latency_ms}"
    )
    assert result.score >= 65, f"canned ChatGPT-style paragraph scored only {result.score:.1f}"
