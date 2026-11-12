"""Task 9: the text engine — ``Hello-SimpleAI/chatgpt-detector-roberta``.

HC3-trained RoBERTa sequence classifier (Apache-2.0). The only network touch
this engine ever makes is downloading its weights at first load (§11.7:
preload instantiates once at boot, off the 120 s scan clock); ``analyze`` is
pure inference. ``__init__`` loads the real checkpoint when called with no
arguments (the preload/registry path does exactly that) and accepts injected
tokenizer/model fakes for offline tests and future re-wiring.

Scoring (brief-pinned):
- the item text is encoded without special tokens and split into ordered,
  non-overlapping chunks of at most ``model_max_length -
  num_special_tokens_to_add()`` content tokens, so every prepared span (with
  <s>/</s> added back) respects the tokenizer's REAL max length — 512 for
  this checkpoint;
- each chunk gets one forward pass; softmax over the binary head yields a
  synthetic probability ``p`` of the AI class, whose index is derived from
  the loaded config's ``id2label`` (never hardcoded);
- item score = 100 · Σ(chunk_p · chunk_len) / Σ chunk_len — the
  length-weighted mean over chunks, ``chunk_len`` counted in content tokens
  (special-token overhead excluded so padding never skews the weighting);
- confidence = mean over chunks of |p − 0.5| · 2, clipped to [0, 1] — the
  binary head has no calibrated confidence, so distance from the 0.5 coin
  flip is the honest self-assessment;
- rationale is the brief-exact template
  ``"roberta classifier: {p:.0%} probability ChatGPT-generated (HC3-trained, v1.0)"``
  where the shown ``p`` is deliberately the item-level length-weighted
  probability, so the evidence panel and the numeric score always agree.

``EngineFault`` is raised only when inference genuinely fails on an item
(unsupported item, tokenization that yields zero tokens, forward-pass
failure) — never for import problems; preload tolerates those separately.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.engines.base import Engine, EngineFault
from pipeline.types import EngineResult, Item

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

CHECKPOINT = "Hello-SimpleAI/chatgpt-detector-roberta"

# Substrings identifying the AI class in a config's id2label values (labels
# are lowercased before matching); HUMAN_LABEL_HINTS excludes false hits.
_AI_LABEL_HINTS = ("chatgpt", "machine", "synthetic", "fake", "generated", "bot")
_HUMAN_LABEL_HINTS = ("human", "real", "natural", "authentic")

# Fallback when a tokenizer ships no usable model_max_length (some HF
# tokenizers carry an unset-sentinel ~1e30 instead of the true limit).
_FALLBACK_MAX_TOKENS = 512


def _ai_label_index(config: Any) -> int:
    """Index of the AI class in the loaded head, derived from ``config.id2label``.

    Matches an AI hint (and no human hint) first; falls back to the lone
    non-human label of a binary head. Raises ``ValueError`` when neither
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
        is_human = any(hint in label for hint in _HUMAN_LABEL_HINTS)
        if is_ai and not is_human:
            return index
    human_indexes = [
        index for index, label in labels.items() if any(h in label for h in _HUMAN_LABEL_HINTS)
    ]
    if len(labels) == 2 and len(human_indexes) == 1:
        return next(index for index in labels if index != human_indexes[0])
    raise ValueError(f"cannot identify the AI class in id2label={id2label!r}")


def _content_budget(tokenizer: Any, model: Any) -> int:
    """Content tokens per chunk: the real max length minus special-token overhead.

    Prefers the tokenizer's own ``model_max_length`` (512 for this
    checkpoint); an unset sentinel or nonsense value falls back to the
    model's ``max_position_embeddings``, then to 512. Subtracting
    ``num_special_tokens_to_add()`` guarantees a prepared span (specials
    included) never exceeds the real limit.
    """
    max_len = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_len, int) or not 0 < max_len <= 100_000:
        max_len = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
    if max_len <= 0:
        max_len = _FALLBACK_MAX_TOKENS
    try:
        specials = int(tokenizer.num_special_tokens_to_add())
    except AttributeError, TypeError:
        specials = 0
    return max(1, max_len - specials)


def _chunks(ids: list[int], size: int) -> list[list[int]]:
    """Split ``ids`` into ordered, non-overlapping chunks of at most ``size``."""
    return [ids[i : i + size] for i in range(0, len(ids), size)]


class RobertaTextEngine(Engine):
    """The §7.2 text winner: HC3-trained RoBERTa over chunked item text."""

    id = "text-ai-roberta"
    version = "chatgpt-detector-roberta/1.0"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: PreTrainedModel | None = None,
    ) -> None:
        """Load the real checkpoint once (preload path) or take injected fakes.

        With no arguments the HF checkpoint downloads (first run ~500 MB) and
        loads here — at boot, never inside a scan (§11.7). Tests inject both
        fakes for zero-download, zero-GPU runs. Weights stay on their load
        device (CPU); GPU placement is the Task 11 bake-off's call.
        """
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
            model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
        self._tokenizer = tokenizer
        self._model = model
        self._model.eval()
        self._ai_index = _ai_label_index(model.config)
        self._content_budget = _content_budget(tokenizer, model)

    def supports(self, item: Item) -> bool:
        """True for text items with non-empty text; nothing else."""
        return item.kind == "text" and bool(item.text)

    async def analyze(self, item: Item) -> EngineResult:
        """Score one text item; wall-clock latency measured inside this call."""
        started_ns = time.perf_counter_ns()
        if not self.supports(item):
            raise EngineFault(
                f"{self.id}: unsupported item (kind={item.kind!r}, text non-empty required)"
            )
        try:
            weighted_p, confidence = self._infer(item.text or "")
        except EngineFault:
            raise
        except Exception as exc:  # broken on THIS item -> degrade it, never the scan
            raise EngineFault(f"{self.id}: inference failed on this item: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
        return EngineResult(
            engine_id=self.id,
            engine_version=self.version,
            score=100.0 * weighted_p,
            confidence=confidence,
            rationale=(
                f"roberta classifier: {weighted_p:.0%} probability ChatGPT-generated "
                "(HC3-trained, v1.0)"
            ),
            latency_ms=latency_ms,
        )

    def _infer(self, text: str) -> tuple[float, float]:
        """Return the length-weighted AI probability and the mean confidence.

        The full text is encoded in one call (``verbose=False`` silences the
        tokenizer's misleading "longer than model max length" notice — no
        sequence ever reaches the model over budget; the split happens right
        here), then chunked at the content budget.
        """
        probabilities: list[float] = []
        lengths: list[int] = []
        for content_ids in _chunks(
            self._tokenizer.encode(text, add_special_tokens=False, verbose=False),
            self._content_budget,
        ):
            probabilities.append(self._chunk_probability(content_ids))
            lengths.append(len(content_ids))
        if not probabilities:
            raise EngineFault(f"{self.id}: tokenizer produced no tokens for this item")
        weighted_p = sum(p * n for p, n in zip(probabilities, lengths)) / sum(lengths)
        confidence = min(
            1.0, max(0.0, sum(abs(p - 0.5) * 2 for p in probabilities) / len(probabilities))
        )
        return weighted_p, confidence

    def _chunk_probability(self, content_ids: list[int]) -> float:
        """One forward pass per chunk; softmax probability of the AI class.

        ``prepare_for_model`` re-adds the tokenizer's own special tokens, so
        the span never exceeds the real max length (the content budget above
        is sized exactly for that); no truncation is passed — if the budget
        math ever broke, an oversized span fails loudly as an EngineFault
        instead of silently scoring a truncated chunk.
        """
        encoding = self._tokenizer.prepare_for_model(
            list(content_ids), add_special_tokens=True, return_tensors="pt"
        )
        # Fast tokenizers return UNBATCHED 1-D tensors here (the slow one
        # returns [1, seq]); the model wants a batch, so lift any 1-D tensor.
        encoding = {
            key: value.unsqueeze(0)
            if isinstance(value, torch.Tensor) and value.dim() == 1
            else value
            for key, value in encoding.items()
        }
        with torch.no_grad():
            logits = self._model(**encoding).logits
        return float(torch.softmax(logits, dim=-1)[0, self._ai_index].item())
