# Bake-off results + winner selection (Task 11 — offline spike)

**Status:** done, throwaway-quality by design (spike philosophy — this documents
decisions, it is not a product feature). Harness: `backend/tools/bakeoff.py`;
fixture generator: `backend/tests/fixtures/make_fixtures.py`; fixtures:
`data/eval/` (tracked, provenance in `data/eval/MANIFEST.md`).

**Metric definitions:** the design doc (`§8`, `§10`, `§11`) names the bake-off
but defines no metric formulas; the pinned metrics below are computed from the
app's own verdict bands (`app/settings.py:53-54`, design §11.13):
`accuracy = (human items < 35 + AI items ≥ 65)/N`; `FP-rate = human items ≥ 65 /
human N`; `unclear-rate = items in [35,65)/N`; `mean latency` = mean
`EngineResult.latency_ms` over successful items. Per-item `EngineFault` →
counted as incorrect and disclosed (none occurred on either pass).

**Runtime (both passes):** torch 2.15.0+cu132,
`torch.cuda.is_available()=True` — NVIDIA GeForce RTX 3050 6GB Laptop GPU.
Image engines self-place on cuda (their committed device policy); the text
engine runs on CPU — its committed load path keeps weights on CPU and never
moves inputs, so CUDA placement would require engine changes (out of scope for
a spike that consumes the seam, ruling 8).

## Environment strategy (two passes — why)

- **Pass 1 — NATIVE pinned venv** (`backend/.venv`: transformers 4.57.6,
  sentence-transformers 5.7.0): measures `text-ai-roberta` +
  `image-ai-vit`. `image-ai-smaug` is expected to fail to load: the checkpoint
  declares `model_type embedding_gemma2` (needs the transformers 5.x line) and
  the venv's pin predates it — recorded as a **blocked row with the real
  failure reason**, never a fabricated number.
- **Pass 2 — OVERLAY** (process-local `PYTHONPATH=/tmp/opencode/tf5`, recipe in
  `task-10-report.md` addendum: transformers 5.22.2 + tokenizers 0.23.3 +
  huggingface_hub 2.4.0 + httpx2 + sentence-transformers 6.3.0 +
  torchvision 0.29.2; `backend/.venv` untouched): measures `image-ai-smaug`
  ONLY — the overlay breaks the text engine (transformers 5.x removed
  `prepare_for_model`, which `text_roberta.py:205` requires), so pass 2 and
  pass 1 never share a process. Smaug's rows are **overlay-measured**.
- The overlay stack lives in `/tmp` (transient). Re-create it with the T10
  recipe if it is gone; model weights are already in `~/.cache/huggingface/hub`
  (sha-verified during Tasks 9–10).

## Commands (exact)

```
# pass 1 — native pinned venv
cd /home/alanjopaul/Projects/project1/backend
.venv/bin/python -m tools.bakeoff

# pass 2 — overlay (T10 recipe)
cd /home/alanjopaul/Projects/project1/backend
PYTHONPATH=/tmp/opencode/tf5 .venv/bin/python -m tools.bakeoff \
    --candidates image-ai-smaug --env-label overlay
```

Fixture set (built by the generator, see provenance below): **text 20 human +
20 AI-style; images 10 real + 10 AI-generated**. Human truth = "not
AI-generated"; real photos carry truth `human`.

## Results (one table, both passes)

| engine | modality | environment | status | accuracy | FP-rate (human) | unclear-rate | mean latency | notes |
|---|---|---|---|---|---|---|---|---|
| `text-ai-roberta` | text | native | measured | 50.00% (20/40) | 25.00% (5/20) | 0.00% | 109 ms | |
| `image-ai-vit` | image | native | measured | 65.00% (13/20) | 10.00% (1/10) | 10.00% | 38 ms | |
| `image-ai-smaug` | image | native | blocked | — | — | — | — | `ValueError: The checkpoint you are trying to load has model type 'embedding_gemma2' but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.` |
| `image-ai-smaug` | image | overlay | measured | 90.00% (18/20) | 0.00% (0/10) | 0.00% | 279 ms | overlay-measured (transformers 5.22.2 / ST 6.3.0); not runnable in-app under the pinned venv |

Notes: 0 per-item EngineFaults on either pass; latencies are wall-clock
`EngineResult.latency_ms` (first-call warmup included, e.g. ViT's first item
205 ms, Smaug's 588 ms — warm calls are ~25–35 ms ViT / ~250–270 ms Smaug /
~70–160 ms text-CPU). Scores were stable across the run (engines are eval-mode;
deterministic given the same weights).

**Winner (measured): `image-ai-smaug`** — 90% accuracy, 0% FP-rate, 0%
unclear-rate vs ViT's 65% / 10% / 10%. It missed exactly two AI images (the
DALL-E lady `ai_06` and the DALL-E cancer-cell `ai_07` — ViT scored those two
≈100) and had zero false positives on all 10 real photos; ViT inverted on one
real photo (`real_08`, the SDO solar-flare observation image, 72.48).

## Winner selection

- **Text winner: `text-ai-roberta`** — the only registered text candidate
  (`registry._ENGINE_SOURCES`, `registry.py:32-36`), already smoke-verified in
  Task 9 (0.24 human / 99.63 ChatGPT). Winner by default; nothing to flip.
- **Image winner (measured): `image-ai-smaug`** — per the table above,
  consistent with the T10 smoke numbers.

**Shipping note (why `backend/.env` keeps `image-ai-vit`):** Smaug cannot run
in-app under the pinned venv (the blocked row above; and the transformers-5.x
line that recognizes the model removed the generic `prepare_for_model` that the
committed text engine depends on — a flip to Smaug breaks both the image engine
preload AND the text engine in the shared process). The in-app shipping
decision — keep ViT as-is, or coordinate a transformers-5 + sentence-transformers-6
+ text_roberta tokenizer-API migration to ship Smaug — **belongs to the human at
the Milestone 3 gate** (implementation plan: verification gate "after Task 11
(bake-off winner locked)").

**Winner-config mechanism (`backend/.env`):** `backend/.env` is gitignored
(root `.gitignore` `\.env`) and does not exist in this tree; it is therefore
documented here rather than force-added. Mechanism verified:
`app/settings.py:22-26` (pydantic-settings `SettingsConfigDict(env_file=
backend/.env, extra=ignore)` — env-file keys are case-insensitive) plus
case-insensitive process env vars (`TEXT_ENGINE_ID=some-other-id
.venv/bin/python -c "from app.settings import Settings; …"` → the value is
picked up). Intended winner values (identical to the committed defaults at
`app/settings.py:61-62`, so **no `.env` file is needed today**):

```
TEXT_ENGINE_ID=text-ai-roberta   # text winner (already the default)
IMAGE_ENGINE_ID=image-ai-vit     # default KEPT deliberately — see shipping note above
```

If the Milestone 3 gate later chooses Smaug in-app, that requires the
coordinated dependency migration AND `IMAGE_ENGINE_ID=image-ai-smaug` in
`backend/.env` (never a bare-default flip).

## Fixture provenance (full tables: `data/eval/MANIFEST.md`)

- **Text — human (20):** single-paragraph excerpts (280–620 chars,
  whitespace-collapsed) from public-domain Project Gutenberg plain-text
  editions, all pre-1930 (therefore pre-AI): Pride and Prejudice (1813), Moby-Dick
  (1851), Frankenstein (1818), Alice in Wonderland (1865), Sherlock Holmes
  (1892), Dracula (1897), The Time Machine (1895), The Great Gatsby (1925),
  Wizard of Oz (1900), Call of the Wild (1903) — 2 excerpts per book, per-file
  book/author/year/ebook-id in the manifest.
- **Text — AI-style (20):** agent-composed ChatGPT-style outputs, written by
  the coding agent for this bake-off in an assistant register (polite opener,
  structured steps, closing offer). NOT collected ChatGPT traffic — stated in
  the manifest per file.
- **Images — real (10):** 9 NASA image-library photographs (public domain:
  Blue Marble, ISS spacewalk, ISS aurora, Perseverance selfie, Mars rover
  close-up, Hurricane Helene from ISS, X-59 in flight, SDO solar flare, Super
  Blood Moon) + 1 model-author demo photograph (Smaug Space `real_photo.jpg`).
- **Images — AI (10):** 9 Wikimedia Commons AI-generated images from per-tool
  categories (CC0/PD/CC BY-SA as stated per file page: 2× Midjourney, 2× FLUX,
  2× DALL-E, 2× Stable Diffusion, plus one FLUX 1.1 Pro Raw webp) + 1
  model-author demo AI image (Smaug Space `ai_image.jpg`).
- Every image was EXIF-transposed, alpha-composited onto white, converted to
  RGB, capped at a 1280 px long side and re-encoded JPEG q88 (deterministic;
  sha256 per file in the manifest). Commons sources are fetched through the
  CDN-cached 1280px thumbnail endpoints.

### Honest caveats about the text numbers (read before drawing conclusions)

1. **Human-text register:** the PD literature register (1813–1925 prose,
   children's literature) differs from HC3's human sources (Reddit/Wikipedia
   Q&A). Four of the five text FPs are Baum/Carroll/Fitzgerald/London prose
   (91–99); this is a distribution note about the eval set, not evidence of an
   engine bug.
2. **AI-text register:** the agent-composed outputs are one register of
   "modern assistant style"; 15/20 were called authentic by the HC3-trained
   head. Both directions are honest generalization observations for the
   Milestone 3 discussion and the P2 fine-tune; the text winner call is
   unaffected (single candidate).
3. `real_08` (SDO solar flare) is a real false-color scientific observation
   image — ViT called it synthetic (72.48); Smaug did not (1.13).

### Cross-check vs the Task 10 smoke numbers

`real_10`/`ai_10` are the same Smaug-Space demo files used in T10 (re-encoded
by this generator). ViT: 33.75 / 25.28 here vs 37.63 / 19.81 in T10 — same
direction, both < 65 (the photographic guard still holds). Smaug (overlay):
0.06 / 99.94 here vs 0.11 / 99.96 in T10 — consistent.

## Reproduction

```
# fixtures (network for image sources; text is embedded, offline)
cd /home/alanjopaul/Projects/project1/backend
.venv/bin/python tests/fixtures/make_fixtures.py        # resumable; --force re-fetches

# harness (from backend/, module-style imports like the app itself)
.venv/bin/python -m tools.bakeoff                       # pass 1 (native)
PYTHONPATH=/tmp/opencode/tf5 .venv/bin/python -m tools.bakeoff \
    --candidates image-ai-smaug --env-label overlay     # pass 2 (overlay)
```

The harness is not a pytest module; the offline suite never touches the network
and stays fully offline. Suite integrity at commit time: **108 passed, 4
deselected** (`pytest -m "not network"`), ruff check + format clean.
