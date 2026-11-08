# AI Content Detector — Design Document

- **Date:** 2026-11-08
- **Status:** Adopted — builder review complete, resolutions folded in (§11)
- **Author:** Alan Jopaul (with agent assistance)

---

## 1. Intent and why

Build a web tool that takes a URL, fetches the page it points to, analyzes its
content, and reports whether the content is **AI-generated / deepfaked** or
**authentic** — with an evidence-backed report per content item.

The project serves two purposes:

1. **Portfolio piece** — demonstrates end-to-end ML systems engineering:
   model selection, a pluggable inference architecture, an explainable verdict
   UI, and (in v2) a personally fine-tuned detector with documented experiments.
2. **Project work** — substantial enough to run locally and demo, scoped so it
   can actually be finished.

The tool is **not** a fact-checker. It detects synthetic media and AI-written
text. It never claims a story's claims are true or false.

---

## 2. Goals

- Paste a URL → get a verdict within a bounded time (≤120 s per scan).
- Per-content-item verdicts: each text block and each image gets its own score.
- Explainable output: score, confidence, engine name/version, and a one-line
  rationale per item.
- Pluggable engine architecture: adding a video engine (v2/v3) requires no
  pipeline redesign.
- Runs entirely on the developer's machine with open-source models; no paid
  APIs required.
- Clean, tested codebase suitable for portfolio review.

## 3. Non-goals

- Fact-checking / misinformation classification against news corpora.
- Video deepfake detection in v1 (architecture leaves a slot for it).
- Multi-user accounts, authentication, or team features.
- Production-scale deployment (portfolio demo deployment only).
- Real-time streaming scans.
- Detecting *whether claims are true*; only whether content appears synthetic.

## 4. Anti-goals (what would make this a failure even if it works)

- A verdict page that says "FAKE" without evidence or confidence.
- A detector that flags ordinary human content as synthetic without any
  "unclear" band (false-positive embarrassment).
- A codebase so tangled that it cannot be explained in an interview.
- A demo that depends on paid API keys or cloud accounts to function.
- Security holes (SSRF, XSS) that would embarrass the portfolio.

---

## 5. Constraints

- **Hardware:** 12 CPU cores, 14 GB RAM, NVIDIA RTX 3050 6 GB, 180 GB free disk.
- **Stack pinned for compatibility:**
  - Python 3.14, `torch >= 2.15`, `transformers == 4.57.x` (**authoritative
    pin** — 4.57.6 wheels verified for Python 3.14; the 2023-era AI-text model
    was built on 4.x semantics. transformers 5.x is the latest line but is
    not adopted until model compatibility is verified).
  - FastAPI + uvicorn, httpx, BeautifulSoup4, Pillow.
  - Node 22 + Next.js + Tailwind.
  - SQLite via SQLAlchemy (single file, upgradeable to Postgres).
- **Network:** outbound HTTP/HTTPS works from the dev machine (verified against
  a major news site, HTTP/2 200).
- **No paid services required.**

---

## 6. Approaches considered and decisions

### 6.1 Content scope

| Option | Description | Trade-off | Decision |
|---|---|---|---|
| A | Text only | Cheapest, but cannot claim deepfake detection | Rejected |
| B | Text + images | Solid range, no video compute pain | **Picked** |
| C | Text + images + video | Most impressive, but GPU-heavy and risky to finish | Deferred to v2/v3 via engine slot |

### 6.2 Misinformation / news verification

The original idea included "misinformation" and "actual news" classes. That
requires a claim-extraction → retrieval → NLI pipeline against trusted-news /
fact-check corpora. The user explicitly dropped this: the tool focuses only on
URL review for **deepfake / AI-generated vs. valid**.

Dropped with it: retrieval index, news corpora, source-credibility scoring.

### 6.3 ML depth

| Option | Description | Trade-off | Decision |
|---|---|---|---|
| A | Integrate pretrained detectors only | Fast, honest, systems-focused | v1 |
| B | Fine-tune one model yourself | Gives a genuine "I trained this" story | v2 |
| C | Train all three branches | Months of GPU work, risky | Rejected |

**Picked: both A and B.** v1 integrates pretrained models; v2 fine-tunes one
detector (likely a text or image classifier on a public dataset) with a
documented baseline → improved evaluation.

### 6.4 Stack

- Backend: **Python + FastAPI** (user writes Python/ML code comfortably).
- Frontend: **Next.js + Tailwind** (verdict/evidence page is the portfolio
  centerpiece).
- DB: **SQLite** (zero infra, Postgres upgrade path).
- Fetching: **httpx + BeautifulSoup**; Playwright optional and **off by
  default** (reduces attack surface).
- Analysis: open-source Hugging Face models run locally.

---

## 7. System design

### 7.1 Component architecture

```
FRONTEND (Next.js + Tailwind)
  /              → URL input + recent scans
  /scan/[id]     → live progress + verdict page + evidence + history

BACKEND (FastAPI / uvicorn)
  api/        routes: /scan · /scan/{id} · /scans · /health
  jobs/       job manager: queue + concurrency cap + 120 s per-scan deadline
  pipeline/   Fetcher → Extractor → EngineRunner → Aggregator
  engines/    registry: TEXT · IMAGE · (video · fact-check: future slots)
  db/         SQLite via SQLAlchemy
  media/      sandboxed snapshot route (strict CSP, no cookies, TTL headers)
```

### 7.2 Engine contract (the single seam)

Every detector implements:

```python
class Engine(Protocol):
    id: str                 # e.g. "text-ai-1"
    version: str            # model + code version, shown in evidence panel
    def supports(item: Item) -> bool
    async def analyze(item: Item) -> EngineResult:
        score: float        # 0–100 synthetic-likelihood
        confidence: float   # 0–1
        rationale: str      # one human-readable sentence
```

Adding video later = new class + registration. No pipeline edits.

### 7.3 Scan lifecycle

Statuses: `queued → fetching → parsing → analyzing → aggregating → done | failed`

Per-item statuses: `ok | degraded | unavailable | blocked_by_site |
fetch_failed | too_large | unsupported`

- `POST /api/scan` returns `202 {scan_id}` immediately; work happens in the
  background job.
- Frontend polls `GET /api/scan/{id}`.
- Engine-level faults never fail the scan: the item is marked degraded/
  unavailable and the report still ships.

### 7.4 Aggregation rules (decided with the human partner)

Text and images are always separated at extraction time and analyzed by their
respective engines **in parallel**. Verdicts are reported **per modality** —
there is no single blended page label. The page headline is **counts**:

```
"2 of 5 items AI-generated (1 text block, 1 image) · 3 of 5 human · 0 unclear"
```

Per-modality verdict lines sit below the headline (e.g. `Text: 1 of 3 blocks
synthetic · Images: 1 of 2 synthetic`), then the per-item detail.

Per-item verdict bands: score < 35 → `authentic` · 35–64 → `unclear` ·
≥ 65 → `synthetic`.

Edge cases:

- **No analyzable items** (JS-only page, blocked images, empty page):
  verdict `insufficient_data`, status `done` with a warning — **never
  `authentic`** for a page that was never analyzed. If the fetch itself
  failed, status is `failed` with the fetch error code.
- **Deadline fired mid-scan:** with ≥ 1 analyzed item → `done` + partial
  report + `deadline_exceeded` warning; without any analyzed item → `failed`
  / `timeout`.

Derived page classification (DB filtering only, **not** the UI headline):
`synthetic_detected | mixed | authentic | insufficient_data`, derived from
the majority of item verdict classes.

`scans.confidence` = confidence of the determining evidence: the **max** item
confidence among ≥ 65 items (when synthetic/mixed signals exist) or the
**min** item confidence among analyzed items (for `authentic` pages). Pages
where every item is `unclear` (no item ≥ 65, no authentic majority):
confidence = max item confidence among analyzed items.

#### 7.4.1 Itemization rules

- **Text blocks:** top-level paragraphs and list items from extracted main
  content; nav/header/footer/aside/comment boilerplate excluded; blocks
  shorter than 80 characters are not analyzed; blocks longer than ~512 model
  tokens are chunked and the item score is the length-weighted mean of chunk
  scores.
- **Images:** content images only (`<img>` in main content, best `srcset`
  variant). Skipped: icons/logos/avatars via size heuristic (< 100 px natural
  dimension), `data:` URIs, CSS-background images; deduplicated by content
  hash; hard cap 20 per scan.

### 7.5 Data model (SQLite, 4 tables)

```
scans
  id (PK) · url · url_canonical · status · verdict · confidence
  error_code · ip_hash · created_at

scan_items
  id (PK) · scan_id (FK → scans) · kind (text|image) · source_ref
  label · status · verdict · confidence

engine_results
  id (PK) · item_id (FK → scan_items) · engine_id · engine_version
  score · confidence · rationale · latency_ms

snapshots
  item_id (FK → scan_items) · blob (on disk) · mime
  created_at · expires_at (+30 days, auto-purge)
```

Relationships: one scan → many items → many engine results per item.
`scans.verdict` stores the derived classification; the UI headline is
counts-based (see §7.4).

### 7.6 API contract

```
POST /api/scan          {url} → 202 {scan_id}
                        (429 rate-limit, 422 invalid input)
GET  /api/scan/{id}     → {status, summary (counts headline), confidence?,
                          verdict (derived classification), per_modality{
                            text  {verdict, counts{synthetic, unclear, authentic}},
                            images{verdict, counts{synthetic, unclear, authentic}}},
                          meta{title, domain, scanned_at},
                          items[{kind, label, excerpt, thumb?, status,
                                 engines[{id, score, confidence, rationale}]}],
                          warnings[], engines_used[]}
GET  /api/scans         → history list (redacted URLs)
DELETE /api/scan/{id}
GET  /media/{item_id}   → snapshot bytes (sandboxed origin, CSP, TTL)
GET  /api/health        → models loaded, queue depth
```

### 7.7 Security gates

```
GATE 1 — FETCHER
  • http/https only (no file://, ftp://, javascript://)
  • resolve DNS → reject private/loopback/link-local/metadata IPs
  • re-validate on every redirect; max 3 redirects
  • 10 MB body cap, 15 s timeout; DNS pinning against rebinding

GATE 2 — EXTRACTOR
  • ≤ 20 images per page, ≤ 60 kB text total
  • magic-byte checks on images
  • every image re-encoded through PIL before inference
  • SVG rasterized-or-rejected

GATE 3 — ENGINES
  • per-engine timeout and memory cap
  • one engine crashing never kills the scan

GATE 4 — STORAGE/UI
  • scraped text rendered as text (no XSS)
  • snapshots served from sandboxed origin with strict CSP
  • URLs redacted in history; no plaintext IP storage
  • 30-day auto-purge job for snapshots
```

Additional defaults (tunable config): per-IP rate limits, request budgets,
generic client errors, pinned dependencies, identifiable User-Agent,
robots.txt honored by default (configurable) with 429/Retry-After respected
(§11.12).

### 7.8 Deployment

- **Dev:** `uvicorn --reload` + `next dev`, SQLite file + `media/` dir.
- **Prod (optional):** one `docker-compose` with backend, frontend, and
  `/data` volume — deployable anywhere for a live portfolio link.

---

## 8. Feasibility findings (verified 2026-11-08)

| Pillar | Result |
|---|---|
| Compute | PASS — RTX 3050 6 GB runs v1 inference comfortably; v2 fine-tune fits with small batches (Colab T4 fallback) |
| Models exist | PASS — text: `Hello-SimpleAI/chatgpt-detector-roberta` (HC3, Apache-2.0); image: `umm-maybe/AI-image-detector` and `abacusai/Smaug-AI-Image-Detector` for a bake-off |
| Datasets (v2) | PASS — GenImage mirrors on Hugging Face (100K–1M images, imagefolder) |
| Stack installable | PASS with pin — torch 2.15 / transformers 5.22 / fastapi / bs4 have Python 3.14 wheels; **pin transformers 4.57.x** for model compatibility |
| Network | PASS — outbound fetch verified (HTTP/2 200) |
| Node toolchain | PASS — Node 22.22, npm 9.2 |

**Remaining unknowns, to be resolved by a runtime spike after design approval:**

1. Real per-item inference latency on the RTX 3050 (expect 0.1–0.5 s/item).
2. Extraction quality across 4–5 diverse real pages.
3. Actual accuracy of the two image detectors on modern fakes (bake-off numbers).

**Effort estimate:** P1 vertical slice ≈ 3–4 weeks part-time; P2 fine-tune ≈
1–2 weeks of experiments.

---

## 9. Phases

| Phase | Deliverable |
|---|---|
| **P1** | Fetcher + extractor + text engine + image engine (pretrained) + verdict page + history |
| **P2** | Fine-tuned model: pick public dataset, baseline → fine-tune → documented eval |
| **P3** | Video engine slot · fact-check engine slot (only if revisited) · model bake-off results surfaced in UI |

---

## 10. What is left to the builder

- Choose exact model checkpoints after the bake-off.
- Choose dataset subset and fine-tuning recipe for P2.
- Decide exact UI styling/layout within the verdict-page concept.
- Tune config defaults (caps, timeouts, rate limits) based on the spike.
- Decide whether/when to add Playwright JS rendering.
- Choose hosting for the optional live demo.

---

## 11. Builder-check resolutions (2026-11-08)

The builder review returned 12 questions. Resolved as follows —
(theirs) = decided by the human partner, (call) = definite call by the
designer, safe to override.

1. *(call)* **Text block granularity** — top-level paragraphs and list items
   from main content; boilerplate excluded; < 80 chars skipped; > 512-token
   blocks chunk-averaged with length-weighted mean. §7.4.1.
2. *(call)* **Rationale source** — v1 rationale is templated from model output
   (engine, score, top signals). Token/attention attribution (SHAP, heatmaps)
   is a v2+ enhancement, not required for P1.
3. *(theirs)* **Page-level verdict semantics** — separate branches, parallel
   inference, counts headline, per-modality verdicts, no blended single
   label. §7.4.
4. *(call)* **Zero analyzable items** — `insufficient_data`, never
   `authentic`; `done` + warning if fetch succeeded, `failed` +
   error_code otherwise. §7.4 edge cases.
5. *(call)* **Dependency pin authority** — `transformers == 4.57.x` is the
   authoritative pin (wheels verified on Python 3.14); 5.22 is the latest
   line, not adopted until model compatibility is verified. §5.
6. *(call)* **Multi-engine in production** — exactly one winning engine per
   modality runs in production; the bake-off (2–3 candidates per modality,
   including text) happens offline and is documented; the schema/API keep
   `engines[]` for future multi-engine use.
7. *(call)* **Deadline behavior** — the 120 s deadline covers the scan only
   (models preloaded at startup, excluded from the clock; `/api/health`
   reports loaded models). ≥ 1 item analyzed → partial `done` +
   `deadline_exceeded` warning; otherwise `failed`.
8. *(call)* **v1 acceptance bar** — end-to-end flow works on 10 diverse real
   URLs with honest confidence bands; no accuracy threshold gates P1;
   measured accuracy is a P2 deliverable; a quick text bake-off is included.
9. *(call — confirmed with the human partner)* **Non-English pages** —
   language detection on extracted text (fastText lid or langdetect);
   non-English text items → status `unsupported` + page warning; images
   unaffected. Machine translation before detection was considered and
   **rejected for v1**: translated text is fluent and regular, which biases
   detectors toward "synthetic" — the top false-positive risk. Translate-
   then-detect may be revisited as a P2 experiment, always reported with a
   "translated, low trust" label.
10. *(call)* **Which images become items** — content images only per §7.4.1
    (size heuristics, `data:` exclusion, content-hash dedupe, cap 20).
11. *(call)* **`scans.confidence` derivation** — max item confidence among
    ≥ 65 items (synthetic/mixed), min item confidence among analyzed items
    (authentic). §7.4.
12. *(call)* **robots.txt / politeness** — fetcher honors robots.txt by
    default (configurable), respects 429/Retry-After, uses an identifiable
    User-Agent.
13. *(call)* **Page classification algorithm** — the §7.4 "majority of item
    verdict classes" is taken over `synthetic` vs `authentic` only; `unclear`
    items abstain. Zero analyzed items → `insufficient_data`; synthetic and
    authentic both present → `mixed`; only synthetic → `synthetic_detected`;
    only authentic → `authentic`; neither → `insufficient_data` (pages where
    every item is `unclear` land here, never `authentic`). Per-modality
    verdicts use the same rule with vocab
    `synthetic | mixed | authentic | insufficient`.
14. *(call)* **GATE 3 memory cap** — v1 enforces the per-engine
    `engine_timeout_s` timeout and catches `torch.OutOfMemoryError`/`OSError`
    (item degrades, scan ships); no explicit memory-ceiling setting in v1.
    A real cap is a P2+ hardening item.
15. *(call)* **Bootable defaults** — Settings must boot with no `backend/.env`:
    `ip_salt` is boot-generated (`secrets.token_hex`) unless set in the env;
    `text_engine_id`/`image_engine_id` default to the planned winners
    (`text-ai-roberta`, `image-ai-vit`); the engine registry resolves ids
    lazily, tolerating not-yet-importable engine classes at preload and
    raising only on unknown ids.
