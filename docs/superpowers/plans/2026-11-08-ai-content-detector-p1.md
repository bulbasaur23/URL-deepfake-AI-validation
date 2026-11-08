# AI Content Detector — P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Harness note:** This plan was written in OpenCode PLAN mode; project files may only be touched by the Build agent. First step of Task 1: make `/home/alanjopaul/Projects/project1` a git repo and copy this file into `docs/superpowers/plans/2026-11-08-ai-content-detector-p1.md`.

**Goal:** Build the P1 vertical slice — paste a URL, background scan fetches/extracts text + images, scans both through pretrained detectors, and serves an evidence-backed per-item verdict page and history.

**Architecture:** FastAPI backend with a strict pipeline (Fetcher → Extractor → EngineRunner → Aggregator), pluggable Engine seam (one winner per modality in production), SQLite job state, background job manager with a 120 s deadline. Next.js frontend with landing, live verdict page, history.

**Tech Stack:** Python 3.14, torch ≥ 2.15, transformers 4.57.x, FastAPI + uvicorn, httpx, BeautifulSoup4, Pillow, langdetect, SQLAlchemy + SQLite, pytest. Node 22 + Next.js + Tailwind.

**Spec:** `/home/alanjopaul/.opencode/plan/2026-11-08-ai-content-detector-design.md` (read it first; §7 is the binding system design, §11 the binding resolution list).

## Global Constraints (verbatim values from the spec)

- Python 3.14, `torch >= 2.15`, `transformers == 4.57.x` — the pin is authoritative; do not adopt transformers 5.x.
- Node 22 + Next.js + Tailwind frontend. SQLite via SQLAlchemy (single file, upgradeable to Postgres).
- Open-source Hugging Face models run locally. **No paid APIs.** Text model default: `Hello-SimpleAI/chatgpt-detector-roberta`. Image candidates: `umm-maybe/AI-image-detector`, `abacusai/Smaug-AI-Image-Detector`.
- Scan deadline **120 s** (models preloaded at startup are excluded from the clock). Fetch: **15 s** timeout, **10 MB** body cap, **≤ 3** redirects re-validated at every hop, gzip-bomb guard.
- Extractor: **≤ 20 images/scan**, **≤ 60 kB** text total (~61_440 bytes), blocks < **80 chars** skipped, blocks > **512 model tokens** chunked, images < **100 px** natural dimension skipped, `data:` URIs and CSS backgrounds never, content-hash dedupe.
- Verdict bands: score < 35 → `authentic`, 35–64 → `unclear`, ≥ 65 → `synthetic`. No single blended page label — headline is counts. `insufficient_data` never reported as authentic.
- Snapshot TTL 30 days (auto-purge). robots.txt honored by default; identifiable User-Agent; 429/Retry-After respected.
- Engine faults never fail a scan. Fetch failure → `failed` + fetch `error_code` (§7.4 edge case 1); after a successful fetch, no analyzable items → `insufficient_data`, never `authentic`. Non-English text items → `unsupported` + warning (no translation).
- All limits live in `backend/app/settings.py` (pydantic-settings over `.env`) as tunable fields, never hard-coded constants in logic.
- Scan statuses exactly: `queued|fetching|parsing|analyzing|aggregating|done|failed`. Item statuses exactly: `ok|degraded|unavailable|blocked_by_site|fetch_failed|too_large|unsupported`. Scan verdicts exactly: `synthetic_detected|mixed|authentic|insufficient_data`.
- Tests: pytest, `TestClient` for API; network-touching tests marked `@pytest.mark.network` and excluded by default (`-m "not network"`).

## Push Protocol (human-gated — decided with the human partner 2026-11-08)

- Remote: `https://github.com/bulbasaur23/URL-deepfake-AI-validation.git` (public, currently empty; the human's own account → direct pushes, no fork, no PR flow).
- Auth: ✅ done 2026-11-08 — ed25519 keypair at `~/.ssh/id_ed25519` (no passphrase) generated manually by the human, public key added to the GitHub account, `ssh -T git@github.com` verified ("Hi bulbasaur23!"). Remaining at Build start: `git config --global user.name "Alan Jopaul" && git config --global user.email "alanjopaul30@gmail.com"` (human-confirmed 2026-11-08) — without a configured identity, git commits fail, so this comes before Task 1. Note: `alanjopaul30@gmail.com` must also be listed under GitHub → Settings → Emails for commits to be attributed to the profile's contribution graph; if it isn't, ask the human to either add it or switch to the account's noreply email.
- **Implementer subagents commit locally and never push.** Pushes are controller work, only at the milestones below, and only after the human (1) verifies the milestone result themselves and (2) answers "yes" to the push question:
  after Task 1 (scaffold + health) · after Task 8 (engines wired, models in `/api/health`) · after Task 11 (bake-off winner locked) · after Task 14 (scan API end-to-end via curl) · after Task 17 (frontend live) · after Task 19 (acceptance sweep — push `main` + tag `v0.1-p1`).
- Push only `main`; never force-push; never push other branches.

## Reporting Protocol (standing instructions from the human partner, added 2026-11-09)

Valid for the whole build. The human never has to ask for these — the report comes with the work, every time:

1. **Explain the code.** Use plain English pitched at a Python-comfortable student (they can read code; they want to follow it, not decode it). After every task: a short digest — what the task adds, its key functions/types with their signatures, one short excerpt of the most interesting piece and why it's written that way.
2. **Explain purpose at each milestone.** At each of the 6 Push Protocol checkpoints, a fuller walkthrough: (a) purpose — what this slice of the product now does end-to-end; (b) **what changed since the last milestone** — files added/modified + the commit list; (c) a proper tour of the important code parts (what each does, in what order, and why); (d) exactly how to verify (commands to run, URL to open, what to expect); then the milestone push yes/no question.
3. **Explain every change.** Nothing lands silently — every diff introduced in a fix loop, rule change, or cowboy fix gets stated in the same message that reports it, in the same plain-English style.

## Review Focus

The spec implies these but only the named tasks pin them to tests:

1. One decorative/stock image flagged synthetic on an otherwise human page → expect counts headline ("1 of 4 items AI-generated (1 image) · 3 of 4 human · 0 unclear"), page never labeled one word "FAKE". Pinned: Task 13 `test_counts_headline_synthetic_image_human_page`.
2. Paywalled / JS-only page → zero analyzable items → expect `done` + `insufficient_data` + warning, never `authentic`. Pinned: Task 13 `test_no_items_insufficient_data`.
3. SSRF: `http://127.0.0.1`, `http://192.168.1.1`, `http://169.254.169.254` → expect fetch rejected locally with a generic error, no request leaves the machine. Pinned: Task 3 `test_private_and_metadata_ips_rejected`.
4. A detector crashing/OOMing mid-scan → expect that item `degraded`, partial report still ships, status `done`. Pinned: Task 12 `test_engine_fault_degrades_item_ships_partial`.
5. robots.txt-disallowed site and an upstream 429 with Retry-After → expect honest `fetch_failed` with `robots_disallowed` / `http_429` error codes and a Retry-After backoff (no silent retry storms); the scan-level outcome of those failures (`failed` + error_code, not `insufficient_data`) is pinned by Task 12 `test_fetch_failed_scan_fails_with_error_code`. Pinned: Task 4 `test_robots_disallowed` and `test_429_retry_after`.

---

## File map (P1)

```
/home/alanjopaul/Projects/project1/
  docs/superpowers/{specs,plans}/     design doc copy + this plan (Task 1)
  backend/
    app/{main,settings}.py            FastAPI factory · pydantic-settings config
    db/{models,session}.py            4 SQLAlchemy tables · session/engine
    pipeline/
      types.py                        Item · FetchResult · EngineResult · ItemInput
      fetcher.py                      SSRF-safe fetch (GATE 1)
      robots.py                       robots.txt + Retry-After politeness
      extract.py                      main-content extraction + text itemization
      langid.py                       language gate → unsupported
      images.py                       image pick/fetch/validate/re-encode/snapshot
      preload.py                      startup model preload + health payload
      aggregate.py                    bands · counts headline · confidence rules
      engines/{base,registry,text_roberta,image_vit,image_smaug}.py
    jobs/manager.py                   queue · concurrency cap · 120 s deadline
    api/{routes,serializers,ratelimit}.py
    tools/bakeoff.py                  offline bake-off harness (writes docs/bakeoff.md)
    tests/                            1 file per component
  frontend/
    lib/api.ts                        typed client for the 6 endpoints
    app/page.tsx                      landing: input + recent scans
    app/scan/[id]/page.tsx            polling + verdict page
    app/history/page.tsx              history + delete
  models/  media/ .env.example        reserved for v2 fine-tune · runtime snapshots
```

---

### Task 1: Repo scaffold, config, health endpoint

**Files:** Create `docs/superpowers/plans/…` (copy), `backend/{pyproject.toml,app/main.py,app/settings.py,tests/conftest.py,tests/test_app_init.py}`, `.env.example`, `models/.gitkeep`, `media/.gitkeep`.

**Interfaces (produced):**
- `app.settings.Settings` pydantic-settings class; instance `settings = Settings()` reading `backend/.env`. Every cap from Global Constraints is a field here with its spec default. Extra fields: `db_path` (default `data/app.db`), `media_dir` (default `media/`), `rate_window_s=60`, `rate_max_requests=30`, `engine_timeout_s=30`, `max_concurrent_scans=2`, `purge_interval_s=21600`, `respect_robots=True`. Remaining fields must **boot without `backend/.env`** (§11.15): `ip_salt` → `secrets.token_hex(16)` generated at boot when unset (set it in `.env` for stable `ip_hash` across restarts); `text_engine_id="text-ai-roberta"`, `image_engine_id="image-ai-vit"` (planned winners; confirm/flip after Task 11).
- `app.main.create_app() -> FastAPI` with `GET /api/health`.

- [ ] **Step 1:** Create git repo in `/home/alanjopaul/Projects/project1` (`git init`), wire the remote per Push Protocol (`git remote add origin https://github.com/bulbasaur23/URL-deepfake-AI-validation.git` then `git branch -M main`), copy design doc + this plan into `docs/superpowers/`, commit. (First push happens only at the Task 1 milestone, after human verification — never from this task.)
- [ ] **Step 2:** Create Python 3.14 venv (`backend/.venv`), install: `torch>=2.15`, `transformers==4.57.*`, `fastapi`, `uvicorn`, `httpx`, `beautifulsoup4`, `Pillow`, `langdetect`, `sqlalchemy`, `pydantic-settings`, `pytest`, `pytest-asyncio`, `ruff`. Pin exact resolved versions in `backend/requirements.txt` (`pip freeze`).
- [ ] **Step 3:** Write `.env.example` (every fixed-default Settings field; `ip_salt` documented as boot-generated when unset) and a `README.md` quickstart (3 lines: venv → uvicorn → next dev).
- [ ] **Step 4:** Write failing test `tests/test_app_init.py`:

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_health_reports_empty_engines():
    r = TestClient(create_app()).get("/api/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == []          # engines plug in at Task 8
```

- [ ] **Step 5:** Run: `cd backend && python -m pytest tests/test_app_init.py -v` → expect FAIL ("models_loaded" KeyError / 404). Implement `create_app()` with stub `GET /api/health` → `{"models_loaded": [], "queue_depth": 0}` (payload shape refactored at Task 8's preload). Lint gate: `ruff check app tests` passes.
- [ ] **Step 6:** Tests pass. Commit: `git commit -m "feat(backend): scaffold app, settings, health"`.

### Task 2: Data model (4 tables)

**Files:** Create `backend/db/models.py`, `backend/db/session.py`, `backend/tests/test_db.py`.

**Interfaces (produced):** ORM classes `Scan(id, url, url_canonical, status, verdict, confidence, error_code, ip_hash, created_at)`, `ScanItem(id, scan_id FK, kind, source_ref, label, status, verdict, confidence)`, `EngineResultORM(id, item_id FK, engine_id, engine_version, score, confidence, rationale, latency_ms)`, `Snapshot(item_id FK, path, mime, created_at, expires_at)`; `db.session.get_session(settings) -> Session` factory with `create_all`.

- [ ] **Step 1:** Failing test `tests/test_db.py`: create in-memory sqlite, insert a Scan + two ScanItems + EngineResultORM + Snapshot; assert `scan.items` round-trips, `snapshot.expires_at == snapshot.created_at + 30d`, and `Scan.verdict` accepts `"insufficient_data"`.
- [ ] **Step 2:** Run → FAIL. Implement the 4 ORM classes + session factory. Run → PASS. Commit: `feat(db): four-table schema`.

### Task 3: Fetcher — SSRF-safe core (GATE 1)

**Files:** Create `backend/pipeline/types.py`, `backend/pipeline/fetcher.py`, `backend/tests/test_fetcher.py`.

**Interfaces (produced):**
- `pipeline.types.FetchResult` dataclass: `status` `Literal["ok","fetch_failed"]`, `content: bytes`, `content_type: str`, `http_status: int|None`, `final_url: str|None`, `error_code: str|None` — codes exactly: `bad_scheme|private_ip|redirect_loop|redirect_too_many|body_too_large|timeout|dns_failure|http_4xx|http_5xx|http_429|robots_disallowed`.
- `pipeline.types.Item` dataclass: `kind: Literal["text","image"]`, `label: str`, `text: str|None`, `image_path: str|None`, `source_ref: str`, `mime: str|None`, `status == "ok"` default, `language: str|None`, `verdict: str|None`, `confidence: float|None`, `engine_results: list` (filled later).
- `pipeline.fetcher.fetch(url: str, settings: Settings) -> FetchResult`.

Rules implemented here: scheme allowlist http/https; DNS-resolve then reject private/loopback/link-local/metadata (`127/8,10/8,172.16/12,192.168/16,169.254/16,::1,fc00::/7`) IPs; pin the resolved IP for connection (no rebinding); re-validate at every hop, max 3 redirects; 10 MB body cap streamed; 15 s timeout; decompression cap (reject if decompressed > 10 MB — gzip-bomb guard).

- [ ] **Step 1:** Failing tests `tests/test_fetcher.py` (all offline via patched dial + real local http.server fixtures where needed):

```python
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://10.0.0.1/x", "http://192.168.1.1/x",
    "http://169.254.169.254/latest/meta-data/", "file:///etc/passwd", "ftp://x.io/a"])
def test_private_and_metadata_ips_rejected(url):
    r = asyncio.run(fetch(url, settings))
    assert r.status == "fetch_failed"
    assert r.error_code in {"private_ip", "bad_scheme"}

def test_success_roundtrip(): local http.server 200 page → status "ok", content_type kept, final_url set.
```

Additional fixture tests (same file, described):

- `test_body_cap_10mb` — local server streams 10 MB + 1 → `error_code == "body_too_large"`.
- `test_three_redirects_ok_fourth_rejected` — hop-by-hop IP re-validation through 3 hops passes; a 4th hop → `error_code == "redirect_too_many"`.
```

- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(pipeline): ssrf-safe fetcher`.

### Task 4: Fetcher politeness — robots, retries, identity

**Files:** Create `backend/pipeline/robots.py`, `backend/tests/test_fetch_politeness.py`. Modify `pipeline/fetcher.py` to consult it (flag `respect_robots`).

**Interfaces (produced):**
- `robots.check_allowed(url: str) -> bool` (parses robots.txt through `fetch()`'s own validated channel).
- `robots.retry_after_wait(response_headers) -> float|None` (seconds; cap at `engine_timeout_s`).

**Interfaces (consumed):** `fetch(url: str, settings: Settings) -> FetchResult` (Task 3, re-used verbatim to fetch robots.txt through the same validated channel).

- [ ] **Step 1:** Failing tests: `test_robots_disallowed` (robots.txt `Disallow: /` for our UA → fetch returns `error_code=="robots_disallowed"`), `test_robots_allowed`, `test_429_retry_after` (local server returns 429 once with `Retry-After: 2` then 200 → fetcher waits ≤ requested seconds, succeeds, data intact), `test_identifiable_user_agent` (local server asserts UA string starts `"AIContentDetector/0.1 ("`).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(pipeline): robots + retry-after politeness`.

### Task 5: Extractor — main content + text itemization

**Files:** Create `backend/pipeline/extract.py`, `backend/tests/test_extract.py`.

**Interfaces (consumed):** `Item` (Task 3 fields).
**Interfaces (produced):** `extract.ExtractionResult` dataclass: `title: str|None`, `domain: str|None`, `text_items: list[Item]` (kind=text, status ok), `image_refs: list[ImageRef]` dataclass `{label, url_candidates: list[(url, width or None)], source_ref}`, `warnings: list[str]`, `analyzable_count: int`.
Rules: `<article>`/`<main>` or BeautifulSoup readability fallback; strip nav/header/footer/aside/script/comment boilerplate; one text item per top-level `<p>`/`<li>` of main content; skip blocks < 80 chars; enforce 60 kB text total (set truncated flag + warning when exceeded); label = `Paragraph N (first 40 chars…)` style.
**(call) tie-break:** if no `<article>`/`<main>`, use the DOM subtree with the highest paragraph-density score.

- [ ] **Step 1:** Failing tests (HTML fixture strings, fully offline): `test_boilerplate_excluded` (nav/footer text absent), `test_paragraph_itemization_and_labels`, `test_short_blocks_skipped` (< 80 chars), `test_text_cap_truncates_with_warning`, `test_no_content_yields_no_items_not_crash`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(pipeline): content extractor + itemization`.

### Task 6: Language gate (no translation)

**Files:** Create `backend/pipeline/langid.py`, `backend/tests/test_langid.py`.

**Interfaces (consumed):** `Item` (kind=text).
**Interfaces (produced):** `langid.detect(text: str, settings) -> str|None`; §11.9: non-English → item `status="unsupported"`, page-level warning `"text is not English; those blocks were not analyzed"` assembled by the job (Task 12). English text keeps `status="ok"`, `language="en"`.

- [ ] **Step 1:** Failing tests: `test_english_ok`, `test_hindi_marked_unsupported`, `test_unsupported_items_carried_not_dropped` (they survive into the report as items with no engine results).
- [ ] **Step 2:** Run → FAIL. Implement with `langdetect` (seeded deterministic). Run → PASS. Commit: `feat(pipeline): language gate, no translation`.

### Task 7: Image pipeline — pick, fetch, validate, snapshot (GATE 2)

**Files:** Create `backend/pipeline/images.py`, `backend/tests/test_images.py`.

**Interfaces (consumed):** `image_refs` + `fetch()` from Tasks 3/5.
**Interfaces (produced):** `images.process_all(image_refs, fetch, settings, scan_id) -> list[Item]` (kind=image). Per image: resolve best `srcset` candidate (highest width; last entry when no widths); fetch bytes via the Task-3 fetcher (inherits every SSRF rule); magic-byte validate (JPEG `FFD8`, PNG `89504E47`, GIF `474946`, WebP `RIFF…WEBP`); PIL open → re-encode to JPEG (strip metadata; **(call)** also downscale longest side to ≤ 1600 px to bound tensor memory); skip entire pipeline for natural size < 100 px, `data:` URIs (never present — extractor excludes), any URL reused on the same page (dedupe by sha256 of re-encoded bytes, keep first); hard cap 20 (excess marked `too_large`); save Media snapshot {..., path, mime, expires_at=+30d}; statuses `ok|fetch_failed|blocked_by_site|too_large|unsupported` carried on the Item.

- [ ] **Step 1:** Failing tests with locally-served PNG/JPEG bytes (table fixtures): `test_best_srcset_variant_selected`, `test_magic_bytes_reject_fake_png`, `test_small_icon_skipped`, `test_dedupe_by_content_hash`, `test_cap_20_marks_too_large`, `test_snapshot_written_with_ttl_30d`, `test_unfetchable_image_gets_fetch_failed`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(pipeline): image pipeline with gate 2 + snapshots`.

### Task 8: Engine seam — contract, registry, preload, health

**Files:** Create `backend/pipeline/engines/base.py`, `backend/pipeline/engines/registry.py`, `backend/pipeline/preload.py`; modify `app/main.py` (startup preloads, real health), `app/settings.py` (engine ids).

**Interfaces (produced/consumed):**
- `engines.base.Engine` Protocol exactly as spec §7.2: fields `id: str`, `version: str`; `def supports(item: Item) -> bool`; `async def analyze(item: Item) -> EngineResult` where `types.EngineResult(dataclass): engine_id, engine_version, score 0..100, confidence 0..1, rationale str, latency_ms int`.
- `engines.registry.get_engine(modality: Literal["text","image"], settings) -> Engine` — single winner per modality from settings ids; unknown id → raise at startup, not per scan.
- `preload.preload(settings) -> dict[engine_id, version]` — imports and instantiates each winner's model once; startup hook calls it; `GET /api/health` returns `{"models_loaded": [...versions], "queue_depth": <int>}`.
- `engines.base.EngineFault(Exception)` — raising it inside `analyze` is the contract for "this engine is broken on this item".

- [ ] **Step 1:** Failing test `tests/test_engines.py`: register a dummy engine by monkeypatching settings ids → `get_engine("text")` returns it; `get_engine("nope")` raises; a dummy raising `EngineFault` documents the expected symptom (assertion: `pytest.raises(EngineFault)`).
- [ ] **Step 2:** Run → FAIL. Implement contract + registry as a static map `engine_id -> "module.path:ClassName"` string for **all** engine ids (incl. the not-yet-written Task 9/10 classes), resolved lazily so Task 11's bake-off reaches non-winners via `registry.get_by_id(any_id)`. Rules: a settings id absent from the map → raise at startup (typo detection, §11.6); a map entry whose class cannot yet import/load → tolerated at preload only (log warning, skip, `models_loaded` stays short) so the Task 1 health test stays green before Task 9's weights exist.
- [ ] **Step 3:** Run full suite `python -m pytest -m "not network" -v` → PASS (Health test from Task 1 still green, `models_loaded` still `[]` until a real engine sets settings ids — keep `preload` tolerant of missing weights). Commit: `feat(engines): seam + registry + preload + health`.

### Task 9: Text engine — chatgpt-detector-roberta

**Files:** Create `backend/pipeline/engines/text_roberta.py`, `backend/tests/test_text_engine.py` (+ one `network`-marked integration test).

**Interfaces (consumed):** Engine contract (Task 8), `Item` (kind=text, text non-empty).
**Interfaces (produced):** `class RobertaTextEngine(Engine)` — `id="text-ai-roberta"`, `version="chatgpt-detector-roberta/1.0"`. `analyze()`: tokenizer truncates-aware chunking at 512 tokens (no overlap); per-chunk model softmax → synthetic-probability; item `score = 100 · Σ(chunk_p · chunk_len) / Σ chunk_len` (length-weighted mean); `confidence` = mean of per-chunk `|p − 0.5| · 2` clipped to [0,1] (binary head has no calibrated confidence); templated rationale exactly: `"roberta classifier: {p:.0%} probability ChatGPT-generated (HC3-trained, v1.0)"`. Chunk spans must respect the tokenizer's real max length (spec's 512).

- [ ] **Step 1:** Failing unit tests with a monkeypatched tokenizer/model fixture (no download in unit tests): `test_chunking_length_weighted_mean` (3 chunks with known probs/lengths → exact score), `test_multiple_chunks_mean`, `test_rationale_template`, `test_confidence_monotonic_edges` (p = 0.5 → confidence 0.0; p = 0.99 → confidence ≈ 1.0).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS (`-m "not network"`).
- [ ] **Step 3:** `@pytest.mark.network test_real_model_smoke`: loads the actual HF checkpoint (first run downloads ~500 MB), scores a hand-written English paragraph → score < 35; a canned ChatGPT-style paragraph → score ≥ 65. Run with `-m network`. Expect: 2 PASS (bake-off later re-verified on GPU; tolerance documented if borderline — flag result in commit message).
- [ ] **Step 4:** Commit: `feat(engines): roberta text engine`.

### Task 10: Image engines — ViT + Smaug

**Files:** Create `backend/pipeline/engines/image_vit.py`, `backend/pipeline/engines/image_smaug.py`, `backend/tests/test_image_engine.py` (+ network smoke test).

**Interfaces (consumed):** Engine contract; `Item.image_path` (re-encoded JPEG from Task 7).
**Interfaces (produced):** `class ViTImageEngine(Engine)` (`id="image-ai-vit"`, `umm-maybe/AI-image-detector`) and `class SmaugImageEngine(Engine)` (`id="image-ai-smaug"`, `abacusai/Smaug-AI-Image-Detector`). Both: image → tensor → model classes mapped to synthetic-probability `p`; `score = 100·p`; confidence `|p−0.5|·2` (same convention as Task 9 — keep consistent); templated rationale: `"{model name}: diffusion-style artifacts, p={p:.0%}"`. `supports()` gates on `item.kind == "image"` and non-null `image_path`.

- [ ] **Step 1:** Failing unit tests (fixture image path + monkeypatched model returning a chosen logit): `test_score_maps_label_to_synthetic_prob`, `test_confidence_convention_same_as_text`, `test_rationale_template`, `test_supports_rejects_text_items`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 3:** Network smoke (`-m network`): both models load on the RTX 3050; two fixture images (one clearly-photographic, one obviously diffusion-style) → photographic p < 0.65 for the *default* engine. Commit: `feat(engines): image engines (two candidates for bake-off)`.

### Task 11: Offline bake-off harness → winner config

**Files:** Create `backend/tools/bakeoff.py`, `docs/bakeoff.md`, fixture generator `backend/tests/fixtures/make_fixtures.py` (writes `out/`-generated eval images into `data/eval/` — tracked).

**Interfaces (consumed):** `registry.get_by_id(any_engine_id)` (Task 8), both engines (Tasks 9–10), fixture sets.
**Interfaces (produced):** `python -m backend.tools.bakeoff` → evaluates every registered candidate for a modality on the eval set; writes `docs/bakeoff.md` with per-candidate accuracy/FP-rate on human fixtures, mean ppt latency; then the *builder* picks winners. Fixture sets: text — 20 short human-written notes + 20 ChatGPT-style outputs (hand-assembled, provenance listed in the file); images — 10 real photos (from an existing public-domain folder the builder already has; provenance listed) + 10 diffusion outputs generated with a free local Stable Diffusion variant or collected from the same public-domain AI-sample folders. This task is throwaway-quality per the design's spike philosophy — it documents decisions, it ships as a script not a product feature.

- [ ] **Step 1:** Write fixtures + `bakeoff.py` (score both image candidates + the text model; output one markdown table).
- [ ] **Step 2:** Run on GPU, capture pass → `docs/bakeoff.md` committed with the numbers and the picked winners (`settings.text_engine_id`, `settings.image_engine_id` set in `backend/.env`, impossible ones left default with a note).
- [ ] **Step 3:** Commit: `docs: bakeoff results + winner selection`.

### Task 12: Job manager — queue, deadline, resilience

**Files:** Create `backend/jobs/manager.py`, `backend/tests/test_jobs.py`. Modify `app/main.py` (instantiate manager as app state; startup preloads; `GET /api/health` reads real queue depth).

**Interfaces (consumed):** fetch (T3/T4), extract (T5), langid (T6), images (T7), `get_engine` (T8), db (T2).
**Interfaces (produced):** `class ScanJobManager` with `async def submit(url: str, client_ip_hash: str) -> int` (creates `Scan(status="queued")`), `async def run(scan_id: int)` internal orchestrator, asyncio.Semaphore(`settings.max_concurrent_scans=2`), per-scan `asyncio.timeout(120 s, covers fetch+extract+analyze+aggregate; model preloads excluded — Task 8 does them at startup)`. Status transitions exactly `queued→fetching→parsing→analyzing→aggregating→done|failed` persisted between stages. Per-item engine execution: `asyncio.wait_for(engine.analyze(item), settings.engine_timeout_s)`; catching `EngineFault`/`TimeoutError`/`OSError`/`torch.OutOfMemoryError` → item `status="degraded"` (or `"unavailable"` if never ran), continue; warnings list: `deadline_exceeded`, `x_escaped_analysis` (per unsupported/degraded items), `text_cap_truncated`, `language_limited`. On wall-clock fire: ≥1 item with engine result → `done` + `deadline_exceeded` warning; else `failed` with `error_code="timeout"`. A `fetch_failed` FetchResult ends the scan immediately: `failed` + the fetch `error_code`, no later stages (§7.4 edge case 1). Text and image items run as two concurrent gather groups (§7.4 parallel inference); per-item `asyncio.wait_for` unchanged. (call, §11.14) GATE 3 "memory cap" in P1 = `engine_timeout_s` per engine call + catching `torch.OutOfMemoryError`/`OSError` → degrade; no memory-ceiling setting in v1.

- [ ] **Step 1:** Failing tests (all offline: monkeypatch engine classes with instrumented fakes; never load real models): `test_status_progression_persisted`, `test_engine_fault_degrades_item_ships_partial`, `test_deadline_gone_mid_scan_partial_done`, `test_deadline_before_any_analysis_fails_timeout`, `test_unsupported_items_skip_engines`, `test_concurrency_cap_serializes_second_scan`, `test_fetch_failed_scan_fails_with_error_code` (monkeypatched fetch returns a `robots_disallowed` FetchResult → scan `failed` with that code).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(jobs): queue, deadline, degraded-item resilience`.

### Task 13: Aggregator — bands, headline, confidence

**Files:** Create `backend/pipeline/aggregate.py`, `backend/tests/test_aggregate.py`.

**Interfaces (consumed):** list of per-item dicts `{kind, status, verdict, confidence}` (verdicts assigned here from per-item EngineResults by band).
**Interfaces (produced):** `aggregate.aggregate_verdict(items, settings) -> AggregatedReport` dataclass: `verdict` ∈ `synthetic_detected|mixed|authentic|insufficient_data` (§11.13: unclear items abstain from the page vote — zero analyzed items → `insufficient_data`; synthetic present AND authentic present → `mixed`; only synthetic → `synthetic_detected`; only authentic → `authentic`; neither → `insufficient_data`, never authentic); `confidence: float|None` — max item confidence among ≥65 items when verdict ∈ {synthetic_detected, mixed}; min item confidence among analyzed items when verdict == authentic; **all-unclear pages**: max item confidence among analyzed items; `headline: str` exactly `"{syn} of {total} items AI-generated ({syn} text blocks, {img} images) · {human} of {total} human · {unclear} unclear"` (breakdown lists only counts > 0, text before images, singular when 1 — hence "(1 image)" when zero text blocks are synthetic; the trailing `· {human} of {total} human · {unclear} unclear` is always present, §7.4's example shows `· 0 unclear` at zero); `per_modality: {"text": {"verdict": …, "counts": {"synthetic":…"authentic":…}}, "images": …}` (same rule per modality; per-modality verdict vocab exactly `synthetic|mixed|authentic|insufficient`; `insufficient` when none analyzed for that modality); `warnings: list[str]`.

- [ ] **Step 1:** Failing tests pinning Review Focus items 1–2 plus the confidence rules:
  `test_counts_headline_synthetic_image_human_page` (1 synthetic image + 3 authentic text → headline exactly "1 of 4 items AI-generated (1 image) · 3 of 4 human · 0 unclear"), `test_no_items_insufficient_data_never_authentic`, `test_band_boundaries_35_65` (64.9 → unclear, 65 → synthetic, 34.9 → authentic), `test_confidence_max_on_synthetic_min_on_authentic`, `test_all_unclear_page_insufficient_data_max_confidence`, `test_mixed_page_both_directions`, `test_synthetic_decides_despite_unclear` (1 synthetic + 3 unclear → `synthetic_detected`), `test_per_modality_split_independent`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(pipeline): aggregation with bands + headline`.

### Task 14: Scan API — submit, poll, rate limit, errors

**Files:** Create `backend/api/routes.py`, `backend/api/serializers.py`, `backend/api/ratelimit.py`, `backend/tests/test_api.py`. Modify `app/main.py` (mount routes, middleware).

**Interfaces (consumed):** manager (T12), serialize (beyond T2 rows).
**Interfaces (produced):**
- `POST /api/scan {"url": str}` → `202 {"scan_id": int}`; `422` generic `{"detail":"invalid url"}` for scheme/host violations; `429` `{"detail":"rate limited"}`.
- `GET /api/scan/{id}` → exactly spec §7.6 JSON: `{status, summary, confidence?, verdict, per_modality{text,images}, meta{title,domain,scanned_at}, items[{kind,label,excerpt,thumb?,status,engines[{id,score,confidence,rationale}]}], warnings[], engines_used[]}`; unknown id → 404 generic.
- `api.ratelimit` sliding-window per-IP middleware using `rate_window_s`/`rate_max_requests`; keyed by hashed IP.
- `engine_results` DB rows persist per item; `engines_used` = distinct engine ids that produced results.
- `POST /api/scan` computes `url_canonical` before `manager.submit`: lowercase scheme + host, strip default ports, drop fragment (spec §7.5).
- IP privacy: the middleware hashes client IP with `settings.ip_salt` (sha256) before any storage or rate-limit use — spec: no plaintext IP storage.

- [ ] **Step 1:** Failing tests: `test_submit_202_and_roundtrip_shape` (monkeypatched fast fake engine; assert every §7.6 key present incl. `per_modality.text.verdict`), `test_invalid_url_422`, `test_unknown_scan_404`, `test_rate_limit_429_after_30`, `test_generic_error_payloads_only`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(api): submit + poll + rate-limit middleware`.

### Task 15: Media route + snapshot TTL purge

**Files:** Create `backend/api/media.py` (router), `backend/jobs/purge.py`, `backend/tests/test_media.py`.

**Interfaces (consumed):** Snapshot rows (T2), `media_dir` files (T7).
**Interfaces (produced):** `GET /media/{item_id}` streams the snapshot bytes with `Content-Type` = stored mime, served under a path-exact route with the same `Content-Security-Policy` + `no cookie` headers the frontend uses, plus `Cache-Control: private, max-age=<seconds until expires_at>` entirely derived from expiration; 404 when expired/missing. `purge.purge_expired(settings)` deletes rows + files where `expires_at < now`; startup hook runs it, background task re-runs every 6 h (tunable constant `purge_interval_s=21600` in settings).

- [ ] **Step 1:** Failing tests: `test_media_streams_bytes_with_csp_headers`, `test_expired_snapshot_404s_and_file_removed`, `test_purge_removes_expired_only`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(api): sandboxed media + ttl purge`.

### Task 16: History API — redacted list, delete

**Files:** Modify `backend/api/routes.py`; create `backend/tests/test_history.py`.

**Interfaces (produced):**
- `GET /api/scans` → `{"scans": [{id, domain, title, verdict, created_at}]}` — **no URL, no path in the response** (redaction rule); newest first.
- `DELETE /api/scan/{id}` → 204; cascades ScanItems/EngineResultORMs/Snapshots (+media files removed); unknown → 404.

- [ ] **Step 1:** Failing tests: `test_history_redacts_urls`, `test_history_newest_first`, `test_delete_cascades_and_removes_media`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit: `feat(api): redacted history + delete`.

### Task 17: Frontend — client, landing, verdict page

**Files:** Create `frontend/{package.json,next.config.js,tailwind.config.ts,lib/api.ts,app/layout.tsx,app/page.tsx,app/scan/[id]/page.tsx,components/ItemCard.tsx,components/HeadlineBanner.tsx}` + `frontend/tests/api.test.ts`.

**Interfaces (consumed):** the exact 6 endpoints (Tasks 14–16). `lib/api.ts` mirrors §7.6 with types `ScanStatus`, `ItemStatus`, `Verdict` etc.
**Behavior:** Landing (`/`): URL input → `submitScan` → `router.push(/scan/{id})`; "Recent scans" card → `listScans()`. Verdict page (`/scan/[id]`): poll `getScan` every 1 s while status ∈ {queued, fetching, parsing, analyzing, aggregating} — while polling, render a visible 5-stage status stepper driven by `scan.status`; then render: headline banner (from `summary`), per-modality lines, warnings banner, item cards — kind badge, band-colored score bar (authentic=green, unclear=amber, synthetic=red), confidence %, rationale line, `thumb` via `/media/…` for images, status chip for degraded/unsupported items. Tailwind classes only.

- [ ] **Step 1:** `npx create-next-app`, Tailwind scaffold, `npm i` once; commit immediately (`chore(frontend): scaffold`).
- [ ] **Step 2:** Write `frontend/tests/api.test.ts` (Vitest): `lib/api.ts` request-builder functions under test — assert GET `/api/scan/{id}` URL, header, and that the type definitions pin the §7.6 response keys. Run `npm test` → FAIL → implement `lib/api.ts` → PASS → commit `test(frontend): typed api client`.
- [ ] **Step 3:** Implement the three pages/components; `npm run build` must pass. Commit: `feat(frontend): landing + verdict page`.
- [ ] **Step 4:** Manual check (`network`): run both servers, scan `https://en.wikipedia.org/wiki/Computer`, screenshot-verdict page renders headline + cards + thumbnails.

### Task 18: Frontend — history page

**Files:** Create `frontend/app/history/page.tsx`; modify `frontend/lib/api.ts` (`deleteScan`).

- [ ] **Step 1:** Extend `api.test.ts`: `deleteScan(id)` builds DELETE method — FAIL → implement → PASS → commit `test(frontend): delete client`.
- [ ] **Step 2:** History page: list from `listScans()` (domain + title + verdict badge + date), Delete button with confirm, link rows to `/scan/[id]`. `npm run build` passes. Commit: `feat(frontend): history page`.

### Task 19: Acceptance sweep — 10 diverse URLs + security pins

**Files:** Create `backend/tests/test_acceptance.py` (all `@pytest.mark.network`), `docs/acceptance.md`.

**Interfaces (consumed):** everything.
**Acceptance list (10, adjust freely):** (1) Wikipedia article, (2) a longform blog, (3) a news homepage, (4) a low-content landing page, (5) a non-English article, (6) an image-heavy gallery page, (7) a page whose images come from a third-party CDN, (8) a very old plain-HTML static site, (9) a page with a discussion thread (boilerplate comments), (10) `http://localhost:8000` → rejected locally by the fetcher (GATE 1), no request sent. Each scan must: finish ≤ 120 s; produce the §7.6 shape; never display a band-less synthetic claim; handle `insufficient_data` correctly where it hits.

- [ ] **Step 1:** `pytest -m "not network"` full suite green; security fixture review: GATE 1 rules re-exercised by test_fetcher set, GATE 2 by test_images, GATE 3 by test_jobs, GATE 4 by test_media/test_history/feed tests (each gate has at least 2 assertions deliberately re-checked).
- [ ] **Step 2:** Run `pytest -m network -k acceptance` on the 10-URL list → all 10 `done`; any URL that honestly fails (site blocks us, page gone, robots-disallowed) is swapped for an equivalent from the same category and the swap recorded → write per-URL results into `docs/acceptance.md` (headline, per-modality, warnings, wall time). This file is the P1 acceptance bar of §11.8 and the builder checklist's "10 diverse real URLs" line.
- [ ] **Step 3:** Commit: `test: 10-url acceptance sweep + docs` → tag `v0.1-p1` (local only — the final push of `main` + tag is controller work at the last Push Protocol milestone, after human verification).

---

## Self-review notes

- **Spec coverage:** §7.1–7.8 mapped to Tasks 1–18; §11 resolution 2 (templated rationale) in T9; §11.3 counts headline in T13; §11.5 pin in Global Constraints; §11.6 one winner via `get_engine` + bake-off T11; §11.7 preload/deadline split T12; §11.8 bar = T19; §11.9 language gate T6; §11.10 image selection T7; §11.11 confidence rules T13; §11.12 politeness T4; §11.13 classification (unclear abstains, `mixed` on both directions) in T13; §11.14 GATE-3 cap call in T12; §11.15 bootable defaults in T1/T8. P2/P3 (fine-tune, video slot, fact-check slot) are outside this plan per the spec scope-check.
- **Step scan:** every step names one file/artifact, one run command, one expected result; algorithm-only bodies (fetcher SSRF loop, chunk-mean math) carry exact formulas; everything else is signature + a pointer to the owning test.
- **Type consistency:** `Item`/`FetchResult`/`EngineResult`/`ImageRef`/`ExtractionResult` signatures introduced once (T3/T5/T7/T8) and consumed by exact name downstream; `EngineResultORM` only in the DB layer (name collision with the pipeline dataclass avoided by a separator suffix, documented in T2).
- **Review Focus:** five failure modes, each pinned to its named test in T3/T4/T12/T13.
- **Proportion:** one compact task per component (19 tasks), tests carry deciding assertions only; no transcription of engine/pipeline bodies beyond the formulas the spec fixes.
- **Push protocol:** human-gated pushes (SSH auth set up pre-Task 1; milestone checkpoints after T1/T8/T11/T14/T17/T19, each requiring human verification + yes, subagents never push) — wiring folded into Task 1 Step 1 (remote + main) and Task 19 Step 3 (final push note).
