# Granite Local Stack: Plan v18 (M2 Max, 32 GB)

A fully local, commercially usable (Apache-2.0) pipeline built on IBM Granite models:

| Model | Role |
|---|---|
| Granite Speech 5.0 470M TurboCTC | Audio → transcript |
| Granite-Docling 258M | Page image → DocTags → Markdown |
| Granite Vision 4.1 4B | Tables, charts and forms → HTML / CSV / JSON |
| Granite 4.2 8B | Reasoning, RAG, summarization, structured output |
| Granite Embedding English R2 (149M) | Text → 768-d vectors for semantic search (hybrid with BM25) |
| Granite Embedding Reranker English R2 (149M) | Re-scores the top search candidates (question + passage read together) |
| Granite Guardian 4.1 8B | Yes/no judge: groundedness and relevance of answers, custom checks (evaluation in v1; batch verify job in v1.1) |

Status: **approved plan (v18), not started**. Sizes and dependency versions checked on Hugging Face / PyPI on 2026-10-03.
Speeds are estimates and get measured in M1.

**Changes from v1:** removed "deep" mode (30B q4) and the 3B draft model. Granite Vision now runs on
**mlx-vlm** instead of llama.cpp/GGUF, so the stack is all MLX. Added a uv / ruff / ty / pytest toolchain
and a Streamlit UI. Voice input deferred to a later version.
**Changes in v4:** back to **phase switching** (ingest and Q&A take turns), replacing v3's "keep everything loaded".
Vision runs from IBM's official full-precision (bf16) weights, so there's no local quantization, and the LLM context returns to 32K.
**Changes in v5:** added **Granite Embedding English R2** for **hybrid search** (BM25 + vectors, combined with Reciprocal Rank Fusion), plus
a data storage section (§2.3).
**Changes in v6:** added **Granite Embedding Reranker English R2** as a third retrieval stage (hybrid top-30 → rerank → top-8 to the LLM).
**Changes in v7:** storage details in §2.3: SQLite connection and write rules, code-friendly FTS5 tokenizer + trigram index for IDs,
vector-cache lifecycle and when to move to sqlite-vec, alternatives considered.
**Changes in v8:** added **Granite Guardian 4.1 8B** as a judge: **M7 evaluation in v1** (groundedness / relevance scoring) and an
optional **v1.1 batch "verify" job** in a third phase (C), never loaded alongside the Q&A LLM. Added `qa_turns` and `verdicts` tables (§2.4).
**Changes in v9:** added **Claude Code hooks** (§4.1) that enforce this plan's rules during development: format/lint after edits,
protected files, command guards (uv only, license, memory), and a quality gate before Claude finishes.
**Changes in v10:** added a **GitHub Actions CI workflow** (§4.2): lockfile + lint on Linux, type checking + unit tests on an Apple Silicon
macOS runner, no model downloads in CI; model golden tests stay on the M2 Max.
**Changes in v11:** added **automatic releases** (§4.3): when the version in `pyproject.toml` has no matching tag, a green build on `main`
is tagged and published as a GitHub Release (wheel + sdist + generated notes). Releases are created only by CI.
**Changes in v12:** granit is **open source under Apache-2.0** (§4.4): `LICENSE`, `NOTICE`, standard license metadata in `pyproject.toml`, a CI
**dependency-license guard** (no GPL/AGPL/SSPL/unknown), third-party notices attached to every release, and audio decoding via `miniaudio`
instead of ffmpeg.
**Changes in v13:** added a project-specific **`.gitignore`** (§4.5), tested against the plan's file layout: ignores user data, model weights,
secrets and caches; keeps `uv.lock`, hooks, CI, license files and test fixtures. Committed first, before any other file.
**Changes in v14:** added **GitHub repository setup** (§4.6): public repo, description + 20 topics, merge settings, secret scanning,
and a **ruleset** protecting `main` (PR required, `lint` + `test` must pass, no force-push or deletion), set up in an order that avoids
an accidental first release.
**Changes in v15:** added the **Streamlit theme** (§4.7): "granite" stone neutrals + one indigo accent, matching light and dark modes
(same hues, mirrored lightness), status colors reserved for status, bundled fonts (works offline), every color pair checked to WCAG AA
and the config validated on Streamlit 1.65. The same file turns off Streamlit telemetry and binds to localhost.
**Changes in v16:** added **adaptive layout** (§4.8): wide pages with width-capped content and wrapping side panels, tested at 7 display
widths (760–3440 px); per-page patterns; `safe_md()` to stop dollar amounts rendering as LaTeX; layout screenshots in M6.
**Changes in v17:** added **voice activity detection** (§3.5): Silero VAD v6 (MIT, 1.2 MB, via the already-pinned mlx-audio) splits long
recordings at pauses instead of fixed windows. Also fixes an **audio-format gap**: miniaudio can't read M4A/AAC/AIFF, so those
go through macOS's built-in `afconvert`.
**Changes in v18:** added the **evaluation design** (§4.9): four levels (unit → golden → benchmarks → quality); a **public synthetic eval
set** (committed) and a **private eval set** (your data, in `data/`); labeled relevance; metrics for retrieval, answers, extraction,
summaries and ASR; **1.0 pass criteria**; result history; a Guardian agreement check; per-stage retrieval logging; golden-test fixes.
Arize Phoenix considered and rejected.

---

## 1. Scope (v1)

1. **Meeting / call transcripts**: audio file → transcript → summary, decisions, action items (JSON).
2. **Document Q&A**: PDFs and scans → Markdown with accurate tables and chart data → answers with citations.
3. **Form / invoice extraction**: document image + JSON Schema → validated JSON.
4. **Cross-source questions** over everything ingested.

Out of scope for v1: live microphone / voice questions, non-English content, multi-user deployment,
live (immediate) answer verification. **v1.1:** batch verification of answers and summaries with Guardian (§2.4).

## 2. Architecture

```
┌──────────────────────── Streamlit UI (app/) ────────────────────────┐
│  Ingest  │  Library  │  Ask (chat + citations)  │  Extract (schema) │
│        status banner: "Ready" / "Ingesting: Q&A paused (3 left)"    │
└────┬─────────────┬──────────────────┬───────────────────┬───────────┘
     │ queue job   │ reads            │ HTTP (OpenAI API) │ queue job
     ▼             ▼                  ▼                   ▼
┌───────────────────────── phase manager ─────────────────────────────┐
│  owns which phase is loaded; only one at a time (A, B, or C*)       │
│  * PHASE C (v1.1): verify worker, Guardian 4.1 8B q8, ~10.5 GB      │
└──────────────┬────────────────────────────────────┬─────────────────┘
               ▼                                    ▼
┌─── PHASE A: ingest worker ──────────┐   ┌─── PHASE B: mlx_lm.server ───┐
│ ~12 GB peak                         │   │ ~15.8 GB peak                │
│ Speech TurboCTC   (mlx-audio)       │   │ granite-4.2-8b-q8-mlx        │
│ Granite-Docling   (Docling→mlx-vlm) │   │ 32K context                  │
│ Granite Vision    (mlx-vlm, bf16)   │   │ + Embedding R2 (queries,     │
│ Embedding R2      (torch/mps)       │   │   torch/mps, in UI backend)  │
│                                     │   │ + Reranker R2 (fp16, same)   │
└──────────────┬──────────────────────┘   └──────────────┬───────────────┘
               ▼                                         │
  SQLite: sources · chunks (FTS5/BM25) · chunk_vectors · extractions · jobs · qa_turns · verdicts
                                          ◄── hybrid retrieve (BM25 ∪ vectors → RRF → rerank) ──┘
```

### 2.1 Phase switching

- **Default phase is B (Q&A):** `mlx_lm.server` is running and the Ask page is live. Phase C (v1.1 verify) switches the same way as A.
- **Queuing work:** uploads and extraction requests are written to a `jobs` table and don't start immediately.
- **Switching to A:** when jobs are waiting and no chat request is in progress (or the user clicks **"Process now"**),
  the phase manager:
  1. stops `mlx_lm.server` and waits until the process exits, which frees its memory;
  2. starts the ingest worker, which loads Speech (+ VAD), Docling and Vision once;
  3. runs **every** queued job in one batch;
  4. exits the worker;
  5. restarts `mlx_lm.server` and health-checks it.
- **Cost:** each switch takes about 5–15 s (loading ~9–10 GB of weights from SSD), measured in M1. Batching jobs
  means you pay it once per batch, not once per file.
- **In the UI:** a status banner shows the current phase. During ingest the Ask page shows "Q&A paused" and
  keeps your typed question until Q&A is back. Extract results appear in the Library when their job finishes.
- **Memory is freed by ending processes.** Stopping a process reliably returns all of its memory, which makes
  phase switching safer than unloading models inside a long-running process.

### 2.2 Models stay out of the Streamlit process

Streamlit re-runs the script on every interaction and runs each session in its own thread. Keeping MLX
models in the ingest worker and in `mlx_lm.server` keeps the UI responsive and avoids cross-thread MLX
use. The UI only calls `granit` library functions, sends HTTP requests and queues jobs.

**One exception: the query embedder and reranker.** Together about 0.6 GB, each call well under a second, so they run in the
Q&A backend (loaded once with `st.cache_resource`, used behind a lock) rather than as another server.
M0 checks that PyTorch on MPS behaves under Streamlit's threads. If it doesn't, it moves into a small sidecar process.

### 2.3 Data storage

Everything is local. The only network traffic is the one-time model downloads from Hugging Face.

```
data/                              # gitignored; path set in config.py
├── granit.db                      # SQLite (WAL mode): metadata, text, vectors, extractions, job queue
├── files/<sha256>.<ext>           # originals, named by content hash (duplicates stored once)
└── derived/<source_id>/
    ├── document.json · document.md        # DoclingDocument + Markdown export
    ├── transcript.json                    # segments with start/end timestamps
    └── crops/p3_table1.png                # regions sent to Granite Vision
```

| Table | Holds |
|---|---|
| `sources` | one row per file: sha256, name, kind (audio/document/image), added_at, status |
| `jobs` | queued / running / done / failed work: task, params (e.g. a JSON Schema), error, timestamps |
| `chunks` | text pieces with **citation details**: source_id, page or start/end seconds, element type |
| `chunks_fts` | FTS5 full-text index over `chunks.text` (BM25 ranking), with a code-friendly tokenizer (below) |
| `chunks_trigram` | FTS5 `trigram` index for **exact substring** search on IDs and codes (invoice numbers, part numbers) |
| `chunk_vectors` | `chunk_id`, `model_revision`, `embedding` (768 × float16 = 1.5 KB BLOB, **normalized to length 1**) |
| `extractions` | Vision output per table / chart / form: format, content, schema, valid flag, model + revision |

**Verified on this Mac (2026-10-03, uv-managed Python 3.12):** SQLite 3.49.1, FTS5 + `bm25()` work,
`enable_load_extension` is available (unlike macOS's built-in Python), and sqlite-vec 0.1.9 loads.

#### Connection and write rules (`store/db.py`)

- Set once per connection: `PRAGMA journal_mode=WAL; busy_timeout=5000; foreign_keys=ON; synchronous=NORMAL`.
  WAL mode lets the Streamlit UI read (Library, job banner) while the ingest worker writes, across processes.
- **One transaction per job:** chunks + FTS rows + vectors + extractions + `jobs.status='done'` are committed together.
  A worker crash leaves no half-ingested source, and the job is simply retried.
- Only the ingest worker writes heavily (Phase A). The UI writes small rows only (new jobs, `qa_turns`).

#### Tokenizer choice for codes and IDs

- The default FTS5 tokenizer splits on punctuation: in testing, the query `INV` matched `INV-2026-0042`, and a
  search for the full ID only matches it as a sequence of separate pieces.
- `chunks_fts` uses `tokenize="unicode61 tokenchars '-_./'"`, which keeps `INV-2026-0042`, `v1.2.3` and `ab_12` as single tokens.
- `chunks_trigram` (`tokenize="trigram"`) handles partial-ID lookups (`2026-004`). The search layer sends ID-like queries
  (regex: letters/digits with `-_./`) there and merges the results into hybrid search.
- Unit tests cover exact-ID, partial-ID and plain-word queries.

#### Vector search

- **Now:** load `chunk_vectors` into a float16 NumPy matrix **once when Phase B starts**, and reload it when the jobs
  table changes. Rank with one matrix product. Measured: **150K chunks × 768 → top-50 in 11.6 ms, 230 MB of memory**
  (fp16), which fits the Phase B budget (§3.3). M1 measures the real amount.
- **Upgrade trigger:** move to **sqlite-vec** (vectors searched on disk, not in memory) when the matrix gets big enough to matter
  in the Phase B memory budget (about 1M chunks ≈ 1.5 GB), not when search gets slow (≈ 80 ms at 1M).
- **Re-embedding:** `model_revision` is stored per vector, so after a model upgrade we re-embed only the outdated rows.

#### Operations

- Backup = copy `data/` (or `sqlite3 granit.db ".backup ..."` while running). Encryption at rest = macOS FileVault.

#### Alternatives considered

| Option | Why not (for v1) |
|---|---|
| LanceDB | Strong vector + hybrid search, but not built for a transactional job queue or relational data, so we'd still need SQLite: two stores and no atomic per-job writes |
| DuckDB | Built for analytics; only one process can have the file open for writing, so the worker and the UI would block each other |
| Chroma | Vector-first; we'd still need SQLite, and it adds little over NumPy at this scale |
| Qdrant | A server, or a single-process local mode; overkill for one user |
| Postgres + pgvector | Technically excellent but a server process with setup and memory cost. **Upgrade path if granit ever becomes multi-user.** |

### 2.4 Verification with Granite Guardian

Guardian 4.1 8B is a **judge**: given a conversation plus one criterion, it answers `<score>yes|no</score>`. IBM positions the 8B
for "model assessment, observability and monitoring, and spot-checking", which matches how we use it.

**Never loaded alongside the Q&A LLM.** Guardian q8 (~8.9 GB) next to the 8B q8 would put Phase B at about 25 GB, over the ~21 GB GPU limit.
It runs only in M7 evaluation (v1) and in a separate **Phase C** batch job (v1.1).

| Use | Criterion | Where | When |
|---|---|---|---|
| Are answers supported by their cited chunks? | `groundedness` (built-in) | M7 eval → v1.1 verify job | **v1** / v1.1 |
| Does the answer address the question? | `answer_relevance` (built-in) | M7 eval → v1.1 verify job | **v1** / v1.1 |
| Were the retrieved chunks relevant? | `context_relevance` (built-in) | M7 eval (compare retrieval setups) | **v1** |
| Meeting summary quality | custom: e.g. *"Every action item names an owner"* | v1.1 verify job | v1.1 |
| Jailbreak, harm, function-calling hallucination | built-in | not used (single user; no tool calls in v1) | n/a |

**How it's called (`verify/guardian.py`):**
- The last user message is the **guardian block**: the no-think instruction, `### Criteria: …`, `### Scoring Schema: …`.
  Retrieved chunks go in through the chat template's `documents=` argument.
- **No-think mode by default.** It's about as accurate as think mode on IBM's benchmarks (groundedness 0.760 vs 0.764; function
  calling 0.79 vs 0.78) and much cheaper. The card warns that reasoning traces "may not be faithful", so think mode is for debugging only
  and its text is never shown as an explanation.
- Use `temperature 0`. Parse by stripping `<think>…</think>` and reading `<score>`. Anything unparseable is stored as `error`, never as a pass.
- **Usage scope (from the card):** only this yes/no scoring format, English only, and custom criteria "require testing".

**Yes/no polarity (`verify/criteria.py`):** each criterion is stored with `yes_means: "risk" | "pass"`.
Built-in risk definitions ("A text is considered **ungrounded** if…") → *yes = problem*. Positively worded requirements
("Every action item names an owner") → *yes = met*. The UI and reports only ever show the derived `passed` flag. A unit test covers both polarities.

**v1.1 verify job (Phase C):** the Ask page records each turn in `qa_turns`. A queued `verify` job makes the phase manager
stop Q&A, start the verify worker (Guardian loaded once), judge every unverified turn against the enabled criteria, write `verdicts`
in one transaction per batch, exit, and restart Q&A. Verdicts appear as badges in the Ask history and Library
(✅ grounded / ⚠️ unsupported claims / ❓ error).

| Table | Holds |
|---|---|
| `qa_turns` | question, answer, cited `chunk_ids`, LLM model + revision, retrieval config (BM25/hybrid/rerank flags), **`retrieval_trace`** (JSON: per-stage chunk IDs + scores for BM25, vectors, RRF, rerank; §4.9), latency, created_at (**v1**: also gives the Ask page persistent history) |
| `verdicts` | target (turn / summary / extraction id), `criterion_id`, raw `yes/no/error`, `yes_means`, derived `passed`, mode (think/no-think), Guardian model + revision, created_at |

## 3. Models and variants (32 GB, M2 Max ≈ 400 GB/s)

| Component | Variant | Runtime | Weights | Phase | Notes |
|---|---|---|---|---|---|
| Speech | `ibm-granite/granite-speech-5.0-470m-turboctc` | mlx-audio 0.5.7 | 0.95 GB | A | Apache-2.0. **Not** `-nc` (CC-BY-NC-SA). |
| VAD | **`mlx-community/silero-vad-v6`** (MIT) | `mlx_audio.vad` (mlx-audio 0.5.7) | 1.2 MB | A | Splits audio at pauses. **Not** `mlx-community/silero-vad` (no license metadata). See §3.5. |
| Docling | `ibm-granite/granite-docling-258M-mlx` | docling[vlm] → mlx-vlm | 0.63 GB | A | Official MLX build. |
| Vision | `ibm-granite/granite-vision-4.1-4b` (**official bf16**) | mlx-vlm 0.7.4 | 8.0 GB | A | No conversion, no third-party weights. See §3.1. |
| LLM | **`ibm-granite/granite-4.2-8b-q8-mlx`** | mlx-lm 0.32.0 server | 9.34 GB | B | 32K context; ~30–40 tok/s estimated on M2 Max. |
| Embedding | `ibm-granite/granite-embedding-english-r2` | sentence-transformers 6.1.0 (torch, `mps`) | 0.30 GB | A + B | Chunks in A, queries in B. See §3.2. |
| Reranker | `ibm-granite/granite-embedding-reranker-english-r2` | sentence-transformers `CrossEncoder` (torch, `mps`, **fp16**) | 0.30 GB (0.60 GB fp32 on disk) | B | See §3.2. |
| Judge | `ibm-granite/granite-guardian-4.1-8b` → **8-bit MLX built locally** (`mlx_lm.convert -q --q-bits 8`) | mlx-lm 0.32.0 (Python API) | ~8.9 GB (est.; 16.8 GB bf16 source) | M7 / C | No official MLX build; IBM's GGUF Q8_0 (8.91 GB) is the fallback. See §2.4. |

Pin every model to a Hub **revision** (commit SHA) in `src/granit/config.py`.

### 3.1 Granite Vision on mlx-vlm (from the model card)

The card's "Usage with MLX VLM" section:

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git   # card says: install from source
mlx_vlm generate --model ibm-granite/granite-vision-4.1-4b --image chart.jpg --prompt "<chart2csv>"
```

What I checked:
- The card's "install from source" advice is **out of date**. mlx-vlm added `granite4_vision` support for 4.1-4b
  in commit #1104 (2026-05-05), and released versions now include it. We pin **`mlx-vlm==0.7.4`** from PyPI;
  M0 confirms that it loads.
- mlx-vlm runs the **official repo directly** (bf16, 8.0 GB). With phase switching it fits in Phase A,
  so there's no need to quantize it.
- Task tags (`<chart2csv>`, `<tables_html>`, …) are expanded by the model's own chat template, so we send the tag only.
- Use `temperature 0` for every extraction call.

### 3.2 Embeddings: Granite Embedding English R2

- **Why this model:** our pipeline only ingests English. IBM's own multilingual card recommends the English R2 model for
  English data. It scores higher on English retrieval (56.4 vs 52.6 on MTEB-v2 Retrieval), is half the size (149M), and
  scores well on table retrieval (78.5) and multi-turn RAG (MTRAG 57.6), which match our tables and chat.
- **Licensing:** Apache 2.0, no third-party tokenizer terms, and trained **without MS-MARCO** (non-commercial). The multilingual
  311M R2 is the upgrade path if non-English content appears, **after** a legal check of its Gemma-derived tokenizer terms.
- **⚠️ Its output isn't normalized** (`modules.json` has no Normalize step). Always call
  `encode(..., normalize_embeddings=True)`. A unit test checks that stored vectors have length 1.
- **No query/passage prefixes.** Context is 8K tokens; our chunks are a few hundred tokens. Matryoshka truncation isn't
  supported, so storage is saved by storing float16 instead.
- **Hybrid retrieval:** take BM25 top-50 and vector top-50 and combine them with **Reciprocal Rank Fusion** (score = Σ 1/(60 + rank)).
  BM25 catches exact strings (invoice numbers, names, codes); vectors catch paraphrases.
- **Reranking (Reranker English R2):** a cross-encoder reads (question, chunk) together and scores the **RRF top ~30**.
  The **top ~8** go to the LLM. It's fine-tuned from the same English R2 base, so it pairs naturally with our embedder. Apache 2.0, no MS-MARCO.
  - IBM's results on top-20 dense candidates: BEIR 53.1 → 55.8, MLDR 41.6 → 45.8, MIRACL 43.6 → 55.2.
  - gte-reranker-modernbert-base scores higher on BEIR and long documents. We keep Granite for licensing clarity, and M7 tells us
    whether long-document reranking is a weakness on our data.
  - Weights are stored as **float32** (598 MB). Load as fp16 (~0.3 GB). sentence-transformers applies a sigmoid (scores 0–1).
  - Scores are for **ordering within one question** only. No fixed "nothing relevant" threshold unless it's tuned from the M7 set.
  - The reranker can be disabled via config, so M7 can compare with and without it.
- **Runtime: sentence-transformers, not mlx-embeddings** (tested 2026-10-03, mlx-embeddings 0.1.0, reference = sentence-transformers 6.1.0):
  - Embedder via mlx-embeddings' default output: **wrong**. Cosine similarity with the reference embeddings is only 0.77–0.82.
    The config's `classifier_pooling: "mean"` is used instead of the CLS pooling the model was trained with (`1_Pooling/config.json`).
    Pooling the CLS token manually from the encoder output matches the reference (0.9999).
  - Reranker via mlx-embeddings: loads and gives the same ordering. Scores differ from the reference by up to 0.026.
  - Speed: about the same (256 chunks: 0.60 s with MLX vs 0.69 s with PyTorch on MPS, fp16).
  - PyTorch is installed by `docling[vlm]` anyway, so switching to MLX saves no dependencies. Decision: sentence-transformers
    (the reference implementation, correct by default). If we revisit, a golden test must compare against sentence-transformers output.

### 3.3 Memory budget

| Phase | Loaded | Weights | Cache / activations | Peak |
|---|---|---|---|---|
| A: Ingest | Speech + VAD + Docling + Vision bf16 + Embedding | 9.9 GB | ~1–2 GB (Vision: up to ~1.6K image tokens) | **~12 GB** |
| B: Q&A | 8B q8 + Embedding (queries) + Reranker fp16 | 9.94 GB | 32K ctx × 160 KB = 5.2 GB | **~15.8 GB** |
| C: Verify (v1.1) / M7 eval | Guardian 4.1 8B q8 | ~8.9 GB | 8K ctx × 160 KB = 1.3 GB (chunks + answer + guardian block) | **~10.5 GB** |

All phases stay well under the ~21 GB macOS lets the GPU use by default on a 32 GB Mac, with no system tuning
and room left over for macOS, the browser and Streamlit.

**The 32K context and long transcripts:** a one-hour meeting is about 12–14K tokens, so meetings up to about 2 hours fit in one
pass. Longer ones are summarized in sections first, then combined. RAG answers use top-k chunks.

### 3.4 Dependencies (dry-run resolved on Python 3.12)

`mlx 0.32.3 · mlx-lm 0.32.0 · mlx-vlm 0.7.4 · mlx-audio 0.5.7 · docling 2.133.0 · transformers 5.18.0 · sentence-transformers 6.1.0 · torch 2.14.1 · streamlit 1.65.0`

The constraints overlap narrowly: mlx-vlm / mlx-audio need `transformers>=5.14`, and Docling on macOS excludes
5.9–5.15, so the lock lands on **≥ 5.16**. `docling[vlm]` also pulls in **torch** (~2.14), a large
install that is required by Docling's VLM extra. The embedder uses that same PyTorch, so it adds only sentence-transformers.

### 3.5 Audio pipeline: decoding + voice activity detection

**Why VAD:** fixed 30–60 s windows can cut through words. TurboCTC has no context across chunk boundaries, so a cut word gets garbled,
and invoice numbers and names are the words that matter most. VAD finds the pauses, so chunks start and end in silence. Its main
value here is **choosing where to split**. TurboCTC's CTC decoder outputs "blank" on silence, so it's far less prone than Whisper-style
models to inventing text during silence; that's why the settings favor generous padding over aggressive trimming.

**Tested (2026-10-03, mlx-audio 0.5.7):** 12.7 s of macOS `say` speech with silences of 2.0 / 3.0 / 1.5 s plus light noise; expected speech 2.00–5.23 s and 8.23–11.18 s.

| Weights | License | Size | Load | Detect | Segments |
|---|---|---|---|---|---|
| `mlx-community/silero-vad` | ⚠️ none in metadata | 2.2 MB | 2.31 s | 1.015 s | 2.31–5.25 · 8.67–11.20 |
| **`mlx-community/silero-vad-v6`** ✅ | MIT | 1.2 MB | **0.16 s** | **0.067 s** | 2.31–5.31 · 8.67–11.23 |

Both find exactly the two segments and the gap. Starts land ~0.3–0.4 s after the expected times, likely the silence `say` adds at the start of each sentence.
**v6 is chosen:** explicit MIT license (passes `test_config.py`'s approved-license check, which the unlabelled repo would fail even though
mlx-audio's README uses it), ~15× faster, half the size.

**Decoding (fixes a gap in §4.4's "miniaudio instead of ffmpeg"):** miniaudio decodes only **WAV, FLAC, MP3, Ogg Vorbis**. It failed
on AIFF in testing and can't read **M4A/AAC** (Voice Memos and many meeting recorders). macOS's built-in **`afconvert`** converted an M4A/AAC
file to 16 kHz mono WAV in testing. It ships with macOS: nothing to install, bundle or license, and still no GPL ffmpeg.

```
decode(path):  .wav .flac .mp3 .ogg        → miniaudio (in-process) → 16 kHz mono float32
               .m4a .aac .aiff .caf .mp4   → afconvert -f WAVE -d LEI16@16000 -c 1 (temp file) → miniaudio
               anything else               → clear error ("convert to WAV or M4A"); ffmpeg only if the user installed it
```

**Pipeline (`ingest/audio.py`)**
1. **Decode** → 16 kHz mono (above).
2. **VAD:** `mlx_audio.vad.load(<silero-vad-v6, pinned revision>)` → `get_speech_timestamps(..., return_seconds=True)`.
3. **Chunk:** pad each segment by ~0.3 s, then merge neighboring segments into chunks of **at most ~30 s**, **splitting at the longest
   pauses**. If continuous speech runs past 30 s with no pause, fall back to a fixed split with ~1 s overlap and remove the duplicated words.
4. **Transcribe** each chunk with TurboCTC; offset timestamps by the chunk start.
5. **Store** `transcript.json` segments (start / end / text) and index them as `chunks` with `start_s` / `end_s`, which become citations like
   "Tuesday call · 12:40".
6. **Report** total speech vs silence time per file on the Library page (helps spot a silent or broken recording).

**Tests**
- **Unit** (`test_audio_chunking.py`, no models): merge/split logic on made-up segments (longest-pause split, ≤ 30 s cap,
  overlap fallback, padding at the start and end of the file); the format routing table.
- **Golden** (`-m model`): the `say`-generated file → exactly 2 segments, gap detected, boundaries within ±0.5 s; a LibriSpeech clip → WER below a threshold;
  **one file per supported format** (WAV, FLAC, MP3, M4A) → identical 16 kHz mono output length (±1 %).
- Fixtures are generated by `scripts/make_audio_fixtures.py` (macOS `say` + `afconvert`), so no third-party recordings and no licensing questions.

**Later (v2 voice input):** mlx-audio also has `realtime_vad` (streaming endpointing) and `smart_turn` (end-of-turn detection) for
microphone input. Their weights' licenses get checked when v2 starts. Speaker diarization (`sortformer`, `nemotron_diarization`) stays out
of scope (licenses unchecked).

## 4. Development toolchain

| Tool | Use |
|---|---|
| **uv** | Project + lockfile (`uv.lock`), Python 3.12 pinned in `.python-version`, `uv run …` everywhere |
| **ruff** | Lint + format (`ruff check`, `ruff format`); config in `pyproject.toml` |
| **ty** | Type checking (`ty check src tests`) |
| **pytest** | Tests; markers split fast unit tests from model-backed tests |
| **Streamlit** | UI (`uv run streamlit run app/Home.py`) |

### 4.1 Claude Code hooks

Hooks make the plan's rules **automatic** while building granit with Claude Code, instead of relying on the model to remember them.
Static guidance (conventions, architecture) goes in `CLAUDE.md`; hooks handle what must **always** happen or **never** happen.

| # | Event · matcher | Script | What it does | Blocks? |
|---|---|---|---|---|
| H1 | `PostToolUse` · `Edit\|Write` | `format_lint.py` | For edited `*.py`: `uv run ruff format <file>` then `uv run ruff check --fix <file>`. Remaining lint errors → **exit 2**, stderr shown to Claude so it fixes them right away | Feedback only (the edit already happened) |
| H2 | `PreToolUse` · `Edit\|Write` | `protect_paths.py` | Denies edits to `data/**` (user data, `granit.db`), `models/**` (downloaded / converted weights), `uv.lock` (change via `uv add` / `uv lock` only), `.env*` | **Yes** (exit 2 + reason) |
| H3 | `PreToolUse` · `Bash` | `guard_bash.py` | Denies: `pip install` / `python -m pip` (→ "use `uv add`"); `huggingface-cli` (deprecated → `hf`); anything referencing **`turboctc-nc`** (non-commercial license); `rm -rf` on `data/` or `models/`; `sudo sysctl iogpu…` (the plan needs no GPU-limit tuning); `git tag` / `git push --tags` / `gh release create` (releases come only from CI, §4.3) | **Yes** |
| H4 | `PreToolUse` · `Bash` | `guard_memory.py` | If the command loads models (`pytest -m model`, `granit ingest\|eval\|verify`, `mlx_lm.*`, `mlx_vlm.*`) **and** an `mlx_lm.server` / granit worker process is already running → deny with "stop the running phase first": two phases at once would exceed ~21 GB | **Yes** |
| H5 | `Stop` | `quality_gate.py` | If any `*.py` changed (`git diff --name-only` + untracked files): `ruff format --check`, `ruff check`, `ty check`, `pytest -q -m "not model"`. Failures → **exit 2** with a short summary, so Claude keeps working. If `stop_hook_active` is true → exit 0 (no loops) | **Yes** (keeps Claude working) |
| H6 | `SessionStart` | `session_context.py` | Prints a few lines into Claude's context: branch + uncommitted files, current milestone (from `.claude/milestone`), which models are downloaded / converted, "plan: PLAN.md §7 Decisions" | No |

**Configuration** (`.claude/settings.json`, committed so the rules apply to everyone; personal tweaks go in `.claude/settings.local.json`):

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [{ "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/session_context.py", "timeout": 10 }] }
    ],
    "PreToolUse": [
      { "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect_paths.py", "timeout": 5 }] },
      { "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard_bash.py", "timeout": 5 },
          { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard_memory.py", "timeout": 5 }
        ] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/format_lint.py", "timeout": 60 }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/quality_gate.py", "timeout": 300 }] }
    ]
  }
}
```

**Design rules for the hook scripts**
- **Standard library only, run with `python3`** (not `uv run`): H2–H4 run before *every* edit and command, so they must start in milliseconds.
  Only H1 and H5 call `uv run` (ruff, ty, pytest).
- **Input:** JSON on stdin: `tool_name`, `tool_input.file_path` / `tool_input.command`, `cwd`, and `stop_hook_active` for `Stop`.
  **Output:** exit 0 = allow; **exit 2 + one-line stderr reason** = block or feedback (shown to Claude). Never crash: malformed input → exit 0 and a log line.
- **Narrow matching:** H3/H4 match command words, not substrings of file contents (e.g. `grep turboctc-nc PLAN.md` is allowed; `hf download …turboctc-nc` is not).
- **Fast gate:** H5 runs only fast unit tests (`-m "not model"`) and is skipped when no Python changed. Model golden tests stay manual (they load GBs).
- **Tested like any code:** `tests/unit/test_hooks.py` feeds sample stdin JSON to each script and checks the exit code and message.
- **Verify in M0:** hook behavior can change between Claude Code versions. M0 confirms each hook fires with `/hooks` and a deliberate
  violation (e.g. ask Claude to `pip install requests`).
- **Not hooked on purpose:** git commits/pushes (handled by the user), notifications (not essential), auto-running model tests (too heavy for every change).

### 4.2 GitHub Actions CI

CI checks what can be checked **without models**: formatting, linting, types, the lockfile, and the unit tests (including the hook scripts
and the license guard). Model golden tests need GBs of weights and real Apple GPU memory, so they stay on the M2 Max (`uv run pytest -m model`).

| Job | Runner | Steps | Why this runner |
|---|---|---|---|
| `lint` | `ubuntu-latest` | `uv lock --check` (lockfile matches `pyproject.toml`) · ruff format check · ruff check, using the **locked** ruff from a small `lint` dependency group | Seconds, needs no ML deps, cheapest runner |
| `test` | **`macos-26`** (Apple Silicon, matches the dev Mac's macOS 26) | `uv sync --locked` · `ty check src tests` · `pytest -m "not model"` | The lock includes macOS-only MLX packages; ty needs them installed to resolve imports |
| `release` | `ubuntu-latest` | Only on push to `main`, after `lint` + `test`: if tag `v<version>` doesn't exist → `uv build` → GitHub Release. See §4.3 | Pure-Python wheel; no ML deps needed to build |

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
env:
  HF_HUB_OFFLINE: "1"        # any accidental model download fails fast instead of pulling GBs
  UV_PYTHON: "3.12"
jobs:
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
      - run: uv lock --check
      - run: uv run --locked --only-group lint ruff format --check
      - run: uv run --locked --only-group lint ruff check
  test:
    needs: lint
    runs-on: macos-26
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
        with:
          enable-cache: true   # caches uv's download cache keyed on uv.lock (torch, mlx, ...)
      - run: uv sync --locked
      - run: uv run ty check src tests
      - run: uv run pytest -q -m "not model"          # includes the dependency-license guard (§4.4)
      - run: uv run python scripts/third_party_notices.py > THIRD_PARTY_NOTICES.md
      - uses: actions/upload-artifact@v7
        with:
          name: third-party-notices
          path: THIRD_PARTY_NOTICES.md
```

**Design rules**
- **No models in CI.** `HF_HUB_OFFLINE=1` plus the default `-m "not model"` marker. A unit test that accidentally needs weights fails
  loudly instead of downloading 20+ GB.
- **Unit tests must not need the GPU.** GitHub's macOS runners are VMs, not a reliable Metal / GPU environment. So `mlx`, `mlx_lm`, `mlx_vlm`,
  `torch` and `sentence_transformers` are **imported lazily** inside the wrappers, never at module import time. Unit tests use fakes (as the phase
  manager tests already do).
- **License guard in CI too:** `tests/unit/test_config.py` checks that every model ID in `config.py` is on the approved Apache-2.0 list
  (and contains no `-nc`). That's the CI counterpart of hook H3, so the rule holds even for changes made without Claude Code.
- **Same tool versions everywhere:** ruff and ty are pinned in `pyproject.toml` dependency groups (`lint = ["ruff"]`, `dev = ["ty", "pytest", …]`)
  and locked in `uv.lock`. CI never uses `uvx ruff` (which runs the *newest* ruff), so local, hooks and CI can't disagree on formatting.
- **Same commands as local and hooks:** CI runs exactly what hook H5 runs (`ruff`, `ty`, `pytest -m "not model"`). A green Stop gate locally predicts a green CI.
- **Cost and speed:** lint runs first on the cheap Linux runner, and `test` only starts if it passes. GitHub bills macOS minutes at a
  higher rate than Linux on private repos (public repos are free). `concurrency` cancels outdated runs on the same branch.
- **Security:** `permissions: contents: read`, no secrets used, and actions pinned to major versions (`checkout@v7`, `setup-uv@v10`, current as of
  2026-10-03). Pin to commit SHAs if stricter supply-chain control is needed.
- **Branch protection:** a ruleset requires `lint` and `test` to pass before merging to `main` (§4.6).
- **Not in CI on purpose:** model golden tests, benchmarks (M1), Guardian evaluation (M7). Optional later: a `workflow_dispatch` job on a
  **self-hosted runner on the M2 Max** that runs `pytest -m model`. Dependabot / automatic upgrades are left out because dependency upgrades
  are deliberate (narrow transformers window, §3.4).

### 4.3 Automatic releases on version bump

**Trigger rule:** a release happens when the version in `pyproject.toml` has **no matching `v<version>` tag** and the commit on `main`
passed `lint` + `test`. Nothing else triggers a release.

```
you:  uv version --bump minor  →  commit (pyproject.toml + uv.lock)  →  PR  →  merge to main
CI:   lint ─► test ─► release
                        ├─ uv version --short                    → 0.3.0
                        ├─ tag v0.3.0 exists?  yes → skip
                        └─ no → uv build → tag v0.3.0 → GitHub Release (wheel + sdist + generated notes)
```

```yaml
# added to .github/workflows/ci.yml (jobs:)
  release:
    needs: [lint, test]                      # only release a green build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write                        # only this job can create tags / releases
    concurrency:
      group: release
      cancel-in-progress: false              # never cancel a half-finished release
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0                     # need existing tags
      - uses: astral-sh/setup-uv@v10
      - name: Read version
        id: v
        run: echo "version=$(uv version --short)" >> "$GITHUB_OUTPUT"
      - name: Skip if already released
        id: check
        run: |
          if git rev-parse "v${{ steps.v.outputs.version }}" >/dev/null 2>&1; then
            echo "exists=true" >> "$GITHUB_OUTPUT"
          fi
      - name: Build
        if: steps.check.outputs.exists != 'true'
        run: uv build
      - uses: actions/download-artifact@v8
        if: steps.check.outputs.exists != 'true'
        with:
          name: third-party-notices          # generated by `test` on macOS, where all deps are installed
          path: dist
      - name: Create release
        if: steps.check.outputs.exists != 'true'
        env:
          GH_TOKEN: ${{ github.token }}
          VERSION: ${{ steps.v.outputs.version }}
        run: |
          PRE=""; [[ "$VERSION" =~ (a|b|rc|dev)[0-9]+ ]] && PRE="--prerelease"
          gh release create "v$VERSION" dist/* \
            --target "$GITHUB_SHA" --title "granit v$VERSION" --generate-notes $PRE
```

**Design rules**
- **Check the tag, not the diff.** "Does `v<version>` exist?" gives the same answer no matter how it's run: re-runs, squash merges and pushes without a bump are
  harmless no-ops. The version in `pyproject.toml` is the only source of truth.
- **Same workflow, `needs: [lint, test]`:** a red build can never be released (unlike a separate `paths: pyproject.toml` workflow).
- **Bump only with uv:** `uv version --bump patch|minor|major` (prerelease: `--bump minor --bump beta`) updates **both** `pyproject.toml`
  and `uv.lock` (verified with uv 0.12.22). A hand-edited version makes `uv lock --check` fail in `lint`, so the release never runs.
- **Prereleases:** versions containing `a` / `b` / `rc` / `dev` are published as GitHub prereleases automatically.
- **Releases come only from CI:** hook H3 blocks local `git tag` / `git push --tags` / `gh release create`, so every release is a green
  build on `main`.
- **Least privilege:** only `release` has `contents: write`; `lint` / `test` stay read-only. `--target "$GITHUB_SHA"` tags the exact tested commit.
- **Release assets:** wheel + sdist (both carry `LICENSE` and `NOTICE` via `license-files`) + `THIRD_PARTY_NOTICES.md` (§4.4).
- **Known limits:** releases created with the default `GITHUB_TOKEN` don't trigger other workflows (by design, to prevent loops), so any
  future post-release steps go in this job. Code only: model weights are never attached (they come from Hugging Face at pinned revisions).
  No PyPI publishing (granit is a local app). If ever wanted: `uv publish` with trusted publishing (OIDC, no stored token).

**Version plan**

| Version | When |
|---|---|
| `0.1.0.dev0` | Initial version in `pyproject.toml`. The first green merge publishes it as a **prerelease**, which doubles as an end-to-end test of the release job (M0) |
| `0.1.0` … `0.7.x` | Optional minor bumps at milestone ends (M1–M7) |
| `1.0.0` | v1 complete (M0–M7) |
| `1.1.0` | M8 verify job (v1.1) |

**How to cut a release** (also in `CONTRIBUTING.md`): `uv version --bump <part>` → commit both files → PR → merge. CI does the rest.

### 4.4 License: Apache-2.0 (open source)

**Why Apache-2.0:** it matches the Granite models and the most important libraries (transformers, streamlit, sentence-transformers),
includes an **explicit patent grant** (MIT doesn't), and enterprises routinely approve it. *Engineering guidance, not legal advice;
have counsel confirm before commercial distribution.*

**Dependency audit (2026-10-03, all 148 resolved packages):** MIT 79 · BSD 38 (incl. torch) · Apache-2.0 27 · ISC 1 · PSF 1 ·
**MPL-2.0 2** (`certifi`, `tqdm`: weak copyleft that applies only to changes made *to their files*; we use them unmodified) ·
**GPL / LGPL / AGPL / SSPL / unknown: 0**. All 8 Granite models are Apache-2.0 and are downloaded, never redistributed.

| Item | Detail |
|---|---|
| `LICENSE` | Full Apache-2.0 text. Copyright 2026 Daryl Lim |
| `NOTICE` | `granit — Copyright 2026 Daryl Lim`. Notes that IBM Granite models (Apache-2.0) are downloaded separately from Hugging Face at pinned revisions and aren't included |
| `pyproject.toml` | `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]` (standard license metadata), so the wheel and sdist carry both files |
| `README.md` | License badge + a "Models & licenses" table (model, license, source link) |
| **Dependency-license guard** | `tests/unit/test_dependency_licenses.py` reads every installed distribution's license metadata (`importlib.metadata`; offline, fast) |
| **Third-party notices** | `scripts/third_party_notices.py` collects each installed package's name, version, license and license text into `THIRD_PARTY_NOTICES.md`. CI generates it in `test` (macOS, full deps) and `release` attaches it |
| **Audio decoding** | `miniaudio` (MIT, already installed via mlx-vlm) decodes WAV / FLAC / MP3 / Ogg and resamples to 16 kHz mono; **M4A / AAC / AIFF / CAF** go through macOS's built-in `afconvert` first (§3.5). Homebrew's ffmpeg includes GPL components; avoiding it keeps distribution simple. ffmpeg is an optional fallback only if the user installed it, **never bundled** |

**How the dependency-license guard works:**
- **Allowed:** MIT, BSD-2/3-Clause, Apache-2.0, ISC, PSF-2.0, Zlib, HPND, 0BSD, Unlicense, MPL-2.0 (unmodified use only).
- **Rejected:** GPL, LGPL, AGPL, SSPL, BUSL, CC-BY-NC / non-commercial, and **unknown or missing** metadata.
- **Exceptions** live in `licenses_overrides.toml`: package, the license confirmed by hand, and the reason. Each one is reviewed in the PR that adds it.
- **The result:** a future `uv add` can't quietly bring in copyleft or unlicensed code. Combined with `test_config.py` (models on the
  approved list) and hook H3 (`turboctc-nc`), licensing is checked locally, in CI, and for both code and models.
- **Contributions:** Apache-2.0 section 5 covers them (inbound = outbound), so no separate CLA is needed. Optionally add
  `# SPDX-License-Identifier: Apache-2.0` headers to source files.

### 4.5 `.gitignore`

A short **project-specific** file instead of GitHub's ~200-line Python template, which covers irrelevant tools (Django, poetry, Jupyter…)
and knows nothing about this project's biggest risk: user data and multi-GB model weights.

```gitignore
# ── granit: user data & model weights (never commit; see PLAN.md §2.3, §3) ──
# granit.db (+ -wal/-shm), originals, derived/, crops/
/data/
# locally converted weights (e.g. Guardian q8)
/models/
# Safety net: weight / cache files anywhere in the tree
*.safetensors
*.gguf
*.bin
*.npy
*.npz
*.db
*.db-wal
*.db-shm
*.sqlite3

# ── Python / uv ──
.venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# ── Tool caches ──
.ruff_cache/
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.mypy_cache/

# ── Generated at release time (PLAN.md §4.4) ──
THIRD_PARTY_NOTICES.md

# ── Secrets & local config ──
.env
.env.*
!.env.example
.streamlit/secrets.toml
.claude/settings.local.json

# ── OS / editors ──
.DS_Store
.AppleDouble
._*
.idea/
.vscode/*
!.vscode/extensions.json
*.swp
```

**Tested (2026-10-03)** with `git status --ignored` / `git check-ignore -v` on a mock of the plan's layout:

| Ignored ✅ | Committed ✅ |
|---|---|
| `data/` (db + WAL, derived docs, crops) · `models/` · stray `*.gguf` / `*.safetensors` · `.venv/` · `dist/` · `__pycache__/` · ruff/pytest caches · `THIRD_PARTY_NOTICES.md` · `.env` · `.streamlit/secrets.toml` · `.claude/settings.local.json` · `.DS_Store` · `.vscode/settings.json` | `uv.lock` · `.python-version` · `pyproject.toml` · `.claude/settings.json` + `hooks/` + `milestone` · `.github/workflows/ci.yml` · `LICENSE` · `NOTICE` · `licenses_overrides.toml` · `CLAUDE.md` · `PLAN.md` · `.env.example` · `.streamlit/config.toml` · `.vscode/extensions.json` · `tests/fixtures/*` |

**Rules**
- **No comments at the end of a line.** Git treats `/data/   # note` as one literal pattern that matches nothing. The first draft did exactly this and
  would have committed `data/derived/…`. Comments go on their own lines.
- **Two layers for data and weights:** directory rules (`/data/`, `/models/`) plus extension rules anywhere in the tree (`*.safetensors`,
  `*.gguf`, `*.db`, …). A weight file committed even once stays in git history forever.
- **Must stay committed:** `uv.lock` (CI runs `--locked` / `uv lock --check`), `.claude/settings.json` and `hooks/` (shared rules; only
  `settings.local.json` is personal), small test fixtures (audio is ignored only under `data/`, not by extension).
- **Not needed:** Hugging Face downloads live in `~/.cache/huggingface/` (outside the repo, `HF_HOME` unset), and so does the HF token.
- **Trade-off:** `*.npy` / `*.npz` / `*.bin` are ignored everywhere. Golden-test arrays are stored as JSON, or get an explicit `!tests/fixtures/…` exception.
- **Kept honest by a test:** `tests/unit/test_gitignore.py` runs `git check-ignore` on sample paths (`data/x.db`, `models/w.safetensors`,
  `.env` → ignored; `uv.lock`, `.claude/settings.json`, `tests/fixtures/a.wav` → not ignored), so a future edit can't silently change this.

### 4.6 GitHub repository setup

The repo is currently local only (no remote). It becomes **public** on GitHub as `darylalim/granit` during M0. Every step below is
public-facing, so it's done **by the user or with explicit go-ahead**, never automatically by Claude.

**About**

| Field | Value |
|---|---|
| Description | *Private, fully local document & meeting intelligence on Apple Silicon. IBM Granite models (speech, Docling, vision, 8B LLM, embeddings, reranker, Guardian) on MLX, with hybrid search, cited answers and a Streamlit UI.* |
| Website | Empty until there's a docs page or a first stable release |
| Sidebar | ✅ Releases · ❌ Packages · ❌ Deployments |
| License | Detected automatically from `LICENSE` (Apache-2.0) |
| Social preview | Later: 1280×640 screenshot of the Ask page |

**Topics (20, the GitHub maximum)**, chosen by checking how many repos use each one (2026-10-03). Broad topics bring traffic; niche ones are where granit can rank near the top:

| Group | Topics (repos using it) |
|---|---|
| Identity | `ibm-granite` (119) · `mlx` (2.7K) · `apple-silicon` (5.4K) · `streamlit` (54K) |
| Local / private | `local-llm` (7.7K) · `on-device-ai` (2.0K) · `offline-first` (10.8K) · `privacy` (27.9K) |
| LLM / RAG | `llm` (145K) · `rag` (52K) · `hybrid-search` (2.3K) · `reranking` (1.0K) · `embeddings` (9.8K) · `hallucination-detection` (1.0K) |
| Documents | `document-ai` (1.4K) · `docling` (393) · `table-extraction` (220) · `vision-language-model` (1.8K) |
| Speech | `speech-to-text` (11.7K) · `meeting-transcription` (155) |

Left out: `python` (918K; the language bar already shows it), `macos`, `sqlite`, `uv` (implementation details), `granite` (ambiguous),
`guardrails` / voice topics (not in v1). **Rule:** topics describe what's actually built; add new ones when features land (e.g. voice in v2).

**Settings**

| Setting | Value | Why |
|---|---|---|
| Merge methods | **Squash only**; delete branch on merge | One commit per PR, so `--generate-notes` release notes read cleanly |
| Wiki / Projects | Off | Docs live in the repo (`README`, `PLAN.md`, `CONTRIBUTING.md`) |
| Issues | On | Bug reports from users |
| Secret scanning + push protection | On (free for public repos) | Blocks pushes that contain tokens (e.g. an HF token) |
| Dependabot **alerts** | On | Security alerts only; no automatic upgrade PRs (upgrades are deliberate, §3.4) |
| Actions workflow permissions | Read-only default | Only the `release` job asks for `contents: write` (§4.3) |

**Ruleset `main`** (current GitHub feature; replaces classic branch protection):

| Rule | Setting |
|---|---|
| Target | default branch |
| Require pull request | yes; **0 approvals** (solo maintainer can't approve their own PR); dismiss stale reviews |
| Required status checks | `lint`, `test` (the CI job names), **branch must be up to date** |
| Block force pushes / deletion | yes |
| Bypass | none, so the maintainer goes through PRs too and every release is a reviewed, green build |

**Setup order (avoids an accidental first release)**

The `release` job publishes on any green push to `main` whose version has no tag. So `pyproject.toml` and the CI workflow must arrive
through a **PR**, after the ruleset exists:

```bash
# 1. First commit: .gitignore, LICENSE, NOTICE, README.md, PLAN.md only (no pyproject.toml, no workflow → nothing can release)
gh repo create darylalim/granit --public --source . --remote origin --push \
  --description "Private, fully local document & meeting intelligence on Apple Silicon. IBM Granite models (speech, Docling, vision, 8B LLM, embeddings, reranker, Guardian) on MLX, with hybrid search, cited answers and a Streamlit UI."

# 2. About + settings
gh repo edit darylalim/granit \
  --add-topic ibm-granite,mlx,apple-silicon,streamlit,local-llm,on-device-ai,offline-first,privacy,llm,rag,hybrid-search,reranking,embeddings,hallucination-detection,document-ai,docling,table-extraction,vision-language-model,speech-to-text,meeting-transcription \
  --enable-squash-merge --enable-merge-commit=false --enable-rebase-merge=false --delete-branch-on-merge \
  --enable-wiki=false --enable-projects=false \
  --enable-secret-scanning --enable-secret-scanning-push-protection

# 3. Scaffold PR (pyproject.toml at 0.1.0.dev0, ci.yml, hooks, ...) → CI runs on the PR → `lint` and `test` checks now exist
# 4. Create the ruleset (status-check names must have run at least once to be selectable)
gh api -X POST repos/darylalim/granit/rulesets --input .github/rulesets/main.json
# 5. Merge the scaffold PR → green build on main → release job publishes the v0.1.0.dev0 prerelease (M0 exit check)
```

`.github/rulesets/main.json` is committed so the protection is **reviewable and reproducible**:

```json
{
  "name": "main",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "pull_request",
      "parameters": { "required_approving_review_count": 0, "dismiss_stale_reviews_on_push": true,
                      "require_code_owner_review": false, "require_last_push_approval": false,
                      "required_review_thread_resolution": false } },
    { "type": "required_status_checks",
      "parameters": { "strict_required_status_checks_policy": true,
                      "required_status_checks": [ { "context": "lint" }, { "context": "test" } ] } }
  ]
}
```

- `gh` flags verified against gh 2.95.0 (`repo create` / `repo edit --help`). The ruleset API fields should be re-checked in M0
  against GitHub's REST docs, and Dependabot alerts enabled in Settings → Security if not on by default.
- Public repo → GitHub Actions minutes are free, including the macOS `test` job (§4.2's cost note applies only if it ever goes private).

### 4.7 Streamlit theme

**Concept: "granite".** Warm stone-gray neutrals (granite) with **one indigo accent** used only for actions. Status colors
(green / blue / orange / red / gray) are **reserved for status**: the phase banner, verdict badges and validation results. That way a ⚠️
never competes with a button. Streamlit's default red accent is replaced because it reads like an error next to status messages.

**Light and dark are designed as a pair:**
- **Same hues, mirrored lightness.** The neutral ladder flips (light: page `#FFFFFF` → panels `#F5F5F4` → borders `#D6D3D1`; dark: page
  `#1C1917` → panels `#292524` → borders `#44403C`). The **sidebar is one step off the page in both** (light `#F5F5F4`, dark `#0C0A09`).
- **One accent hue (243°) in both modes:** light `#4F46E5`, dark `#6059F1`. Dark mode can't use the light shade. Streamlit draws white
  text on primary buttons (needs ≥ 4.5:1) and the accent must stay visible on the dark page (≥ 3:1), which leaves a narrow
  luminance window (0.132–0.183). `#6059F1` was found by searching inside it: white text 5.0:1, vs page 3.5:1.
- **Status colors:** darker solid colors in light mode, brighter in dark mode, same hue. **Tints carry equal visual weight:** light-mode pale tints sit just above the
  white page (contrast 1.07–1.26); dark tints are 18% of the status color over the page (1.33–1.54). Status text on tints: ≥ 6.4:1 light, ≥ 6.9:1 dark.
- **Charts:** categorical palettes keep the same 8-hue order (indigo, teal, amber, pink, sky, violet, lime, stone), with darker shades in light mode and lighter in dark.
  Sequential indigo ramps are **reversed in dark mode**, so low values fade into the page in both. Diverging: red ↔ indigo, with a neutral
  midpoint that matches each page.
- **Both `[theme.light]` and `[theme.dark]` are defined**, so the app follows the system setting and users can switch in ⋮ → Settings.

**Semantic color map** (used consistently in `app/`):

| UI element | Color | Streamlit API |
|---|---|---|
| Buttons, links, focus, selected page | indigo (primary) | automatic |
| Phase banner: Ready | green | `st.success` |
| Phase banner: Ingesting / Verifying, Q&A paused | blue | `st.info` (a normal state, not a warning) |
| Phase or job failure | red | `st.error` |
| Verdict badges (v1.1): ✅ grounded · ⚠️ unsupported · ❓ error | green · orange · gray | `st.badge(color=…)` |
| Extraction: schema valid · invalid | green · red | `st.badge` |
| Citations (source / page / timestamp) | gray | `st.badge(color="gray")` |
| Chat avatars | neutral icons, never status colors | `st.chat_message(..., avatar=":material/person:")` / `":material/neurology:"` (Streamlit's default user/assistant avatars are red/orange, which read as error/warning) |

**Fonts:** Streamlit's built-in `sans-serif` (Source Sans) and `monospace` (Source Code Pro), which are **bundled with Streamlit and served
locally**. No Google Fonts: loading them would make network requests from a "fully local" app and break offline use.
No extra font files or licenses to manage. Base size 15 px suits dense tables and transcripts.

**`.streamlit/config.toml`** (committed):

```toml
# granit theme: "granite" stone neutrals + one indigo accent; status colors are reserved for status.
# Light and dark share every hue; only lightness changes. Contrast checked to WCAG AA (PLAN.md §4.7).

[browser]
gatherUsageStats = false          # fully local app: no Streamlit telemetry

[server]
address = "localhost"             # never listen on the network

[client]
toolbarMode = "viewer"            # hide developer options (Deploy, rerun); keep Settings (light/dark switch)

[theme]
# Shared by both modes
font = "sans-serif"               # Source Sans, bundled with Streamlit (works offline; no Google Fonts)
codeFont = "monospace"            # Source Code Pro, bundled
baseFontSize = 15
headingFontWeights = [700, 600, 600, 600, 600, 600]
baseRadius = "medium"
buttonRadius = "medium"
showWidgetBorder = true
showSidebarBorder = true
linkUnderline = false

[theme.light]
primaryColor = "#4F46E5"                  # indigo: actions only (buttons, links, focus, selected nav)
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F4"      # stone-100
textColor = "#1C1917"                     # stone-900
linkColor = "#4338CA"
borderColor = "#D6D3D1"                   # stone-300
codeBackgroundColor = "#F5F5F4"
codeTextColor = "#1C1917"
dataframeBorderColor = "#E7E5E4"
dataframeHeaderBackgroundColor = "#F5F5F4"
dataframeHeaderTextColor = "#1C1917"
greenColor = "#15803D"
greenBackgroundColor = "#DCFCE7"
greenTextColor = "#166534"
yellowColor = "#A16207"
yellowBackgroundColor = "#FEF9C3"
yellowTextColor = "#854D0E"
orangeColor = "#C2410C"
orangeBackgroundColor = "#FFEDD5"
orangeTextColor = "#9A3412"
redColor = "#B91C1C"
redBackgroundColor = "#FEE2E2"
redTextColor = "#991B1B"
blueColor = "#1D4ED8"
blueBackgroundColor = "#DBEAFE"
blueTextColor = "#1E40AF"
violetColor = "#7E22CE"
violetBackgroundColor = "#F3E8FF"
violetTextColor = "#6B21A8"
grayColor = "#57534E"
grayBackgroundColor = "#E7E5E4"
grayTextColor = "#44403C"
chartCategoricalColors = ["#4F46E5", "#0D9488", "#B45309", "#BE185D", "#0369A1", "#7C3AED", "#4D7C0F", "#57534E"]
chartSequentialColors = ["#EEF2FF", "#E0E7FF", "#C7D2FE", "#A5B4FC", "#818CF8", "#6366F1", "#4F46E5", "#4338CA", "#3730A3", "#312E81"]
chartDivergingColors = ["#991B1B", "#DC2626", "#F87171", "#FCA5A5", "#FEE2E2", "#E0E7FF", "#A5B4FC", "#818CF8", "#4F46E5", "#3730A3"]

[theme.light.sidebar]
backgroundColor = "#F5F5F4"
secondaryBackgroundColor = "#E7E5E4"
codeBackgroundColor = "#E7E5E4"
borderColor = "#D6D3D1"

[theme.dark]
primaryColor = "#6059F1"                  # same indigo hue, lighter: white button text 5.0:1, vs background 3.5:1
backgroundColor = "#1C1917"               # stone-900
secondaryBackgroundColor = "#292524"      # stone-800
textColor = "#F5F5F4"                     # stone-100
linkColor = "#A5B4FC"
borderColor = "#44403C"                   # stone-700
codeBackgroundColor = "#292524"
codeTextColor = "#F5F5F4"
dataframeBorderColor = "#44403C"
dataframeHeaderBackgroundColor = "#292524"
dataframeHeaderTextColor = "#F5F5F4"
# Status tints = 18% of the status color over the page, the same visual weight as light mode's pale tints
greenColor = "#4ADE80"
greenBackgroundColor = "#243C2A"
greenTextColor = "#86EFAC"
yellowColor = "#FACC15"
yellowBackgroundColor = "#443917"
yellowTextColor = "#FDE68A"
orangeColor = "#FB923C"
orangeBackgroundColor = "#442F1E"
orangeTextColor = "#FDBA74"
redColor = "#F87171"
redBackgroundColor = "#442927"
redTextColor = "#FCA5A5"
blueColor = "#60A5FA"
blueBackgroundColor = "#283240"
blueTextColor = "#93C5FD"
violetColor = "#C084FC"
violetBackgroundColor = "#3A2C40"
violetTextColor = "#D8B4FE"
grayColor = "#A8A29E"
grayBackgroundColor = "#35322F"
grayTextColor = "#D6D3D1"
chartCategoricalColors = ["#818CF8", "#2DD4BF", "#FBBF24", "#F472B6", "#38BDF8", "#A78BFA", "#A3E635", "#A8A29E"]
chartSequentialColors = ["#312E81", "#3730A3", "#4338CA", "#4F46E5", "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE", "#E0E7FF", "#EEF2FF"]
chartDivergingColors = ["#FCA5A5", "#F87171", "#DC2626", "#991B1B", "#450A0A", "#1E1B4B", "#3730A3", "#4F46E5", "#818CF8", "#A5B4FC"]

[theme.dark.sidebar]
backgroundColor = "#0C0A09"               # stone-950: one step darker than the page, mirroring light mode
secondaryBackgroundColor = "#1C1917"
codeBackgroundColor = "#1C1917"
borderColor = "#292524"
```

**Verified (2026-10-03):**
- **Contrast:** 64 pairs (32 per mode: text on page, panel and sidebar; white on primary; primary vs page and sidebar; links; code;
  dataframe headers; every status text-on-tint and color-on-page; every chart color) → **0 below WCAG AA** (4.5:1 text, 3:1 UI/graphics).
  Lowest chart contrast: light 3.74, dark 5.86.
- **Streamlit 1.65.0:** the server starts with this file (HTTP 200, health `ok`), all options read back via `streamlit.config`, and no warnings in the log.
- **Visual preview:** a page with the banner, badges, chat, buttons, input, table and chart was rendered in both modes (Playwright + Chrome).
  The first pass found three issues, all fixed above: dark status banners too heavy (deep 900-shade fills → 18% tints), default red/orange chat
  avatars (→ neutral icons), and the irrelevant "Deploy" button (→ `toolbarMode = "viewer"`, which keeps the Settings menu for the light/dark switch).
  M6 keeps this as a visual check: screenshots of each page in both modes before the milestone is done.

**Rules**
- **No custom CSS for theming.** Everything is in `config.toml` (it survives Streamlit upgrades). `st.context.theme.type` is only used for things
  config can't reach (e.g. swapping a logo).
- **Kept honest by a test:** `tests/unit/test_theme.py` parses `config.toml` and re-runs the contrast checks (same pairs and thresholds),
  checks that both modes define the same keys and use the same primary hue, and checks that `gatherUsageStats = false`, `address = "localhost"` and `toolbarMode = "viewer"`. A tweak that breaks
  contrast or the light/dark pairing fails CI.
- **Privacy is in the same file:** `[browser] gatherUsageStats = false` (Streamlit sends usage statistics by default) and `[server] address = "localhost"`
  (never listen on the network).

### 4.8 Adaptive layout for different displays

**Constraints**
- **Python can't see the screen size.** `st.context` (1.65) exposes cookies, headers, locale, theme, timezone and URL, but no viewport width.
  So Python can't branch on screen size; layouts must adapt through Streamlit's own sizing, wrapping and collapsing.
- **Target displays (localhost only, so no phones or tablets):** half-screen window ~760 px · 13" MacBook Air 1470 · 14" MacBook Pro 1512 ·
  16" 1728 · external 2560 · ultrawide 3440 (CSS px).
- **`st.columns` stacks only at ≤ 640 px**, so between ~760 and 1280 px side-by-side columns get cramped instead of stacking.

**Measured (Streamlit 1.65 + Chrome, 2026-10-03), characters per line of answer text:**

| Approach | 1280–1512 | 2560–3440 | 760 (half-screen) |
|---|---|---|---|
| `layout="centered"` | ✅ ~94 | ❌ 736 px fixed, 2/3 of the screen empty | ✅ |
| `layout="wide"` | ⚠️ 115–149 | ❌ **300–427**, unreadable | ✅ |
| wide + `st.columns([5, 3])` | ✅ 68–89 | ❌ 184–263 | ❌ cramped (57) |
| **wide + 720 px column + wrapping side panel** ⭐ | ✅ **~96** | ✅ **~96** | ✅ **~96**, panel moves below |

**Core technique: wrapping rows with fixed-width panels.** In `st.container(horizontal=True, wrap=True)`, fixed-width child containers sit
side by side when there's room and wrap to the next line otherwise. The panel widths decide where that switch happens. This gives a real
breakpoint without JavaScript or screen-size detection.

**Per-page patterns** (all pages use `st.set_page_config(layout="wide", initial_sidebar_state="auto")`, so the sidebar and header
stay in the same place and each page limits its own content):

| Page | Pattern | Tested behavior |
|---|---|---|
| **Ask** | Wrapping row: **720 px** conversation (left-aligned) + **380 px** sources panel; `st.chat_input(..., width=720)` | Sources **beside** from **1470 px** (13" MacBook Air) up, **below** on smaller windows; the input lines up with the conversation (same x and width) at every width |
| **Extract** | Wrapping row, centered: **520 px** document image │ **520 px** extracted fields | **Side by side from 1512 px** (14" MacBook Pro) up; stacked and centered below that |
| **Library** | Tables `width="stretch"`, fixed `height` with scrolling | Uses the full width: more columns visible on big monitors |
| **Ingest** | 720 px column: upload area + job queue | Same reading width as Ask |
| **All** | `initial_sidebar_state="auto"` | The sidebar collapses at ~760 px, giving the content full width |

```python
# app/layout.py: the shared helpers every page uses
READING_WIDTH = 720      # ~96 chars/line at 15 px Source Sans
SIDE_PANEL_WIDTH = 380   # Ask: sources beside the answer from ~1470 px
EXTRACT_PANEL_WIDTH = 520  # Extract: side by side from ~1512 px

def reading_with_side_panel():
    row = st.container(horizontal=True, wrap=True, gap="large")
    return row.container(width=READING_WIDTH), row.container(width=SIDE_PANEL_WIDTH, border=True)

def safe_md(text: str) -> str:
    """Model output is plain prose: dollar signs are currency, never LaTeX."""
    return text.replace("$", r"\$")
```

**Rules**
- **Answers are capped at the reading width, not stretched.** Past ~100 characters per line, text gets hard to read. Big screens get the *extra panel*
  (sources), not longer lines. Tables are the exception (Library stretches).
- **The chat input is left-aligned on purpose.** Streamlit keeps it at the bottom of the screen, full-width and left-aligned, so it can't be
  centered from Python. Left-aligning the reading column keeps the input under it, and the space to the right holds the sources panel.
- **⚠️ All model output goes through `safe_md()` before `st.markdown`.** Found in the screenshots: two dollar amounts in one paragraph
  (`$4,980 … $1,200`) rendered the text between them as **LaTeX** (`code.language-math` in the page). The bundled docs only cover a single `$5`.
  Escaped output renders correctly (verified). `tests/unit/test_layout.py` checks that `safe_md` escapes every `$`, and that the
  Ask / Library / Extract rendering code calls `safe_md` (grep-style check) so new code can't skip it.
- **Width constants live in one place** (`app/layout.py`). Changing them moves the switch points, so the M6 screenshot check runs again.
- **Display scaling caveat:** switch points are CSS pixels; browser zoom and macOS "Larger Text / More Space" change the effective width,
  so switch points move a little but the behavior stays the same.

**M6 layout check:** `scripts/ui_screenshots.py` (Playwright + installed Chrome, dev-only dependency) captures every page at
**760, 1512 and 2560 px** in **light and dark** mode (24 screenshots) and reports panel positions (beside or below) and characters per line.
It's a manual pre-merge check for UI changes, not part of CI (it needs a running app).

### 4.9 Evaluation design

**Four levels**, from cheap and frequent to expensive and occasional:

| Level | Question | When | Where |
|---|---|---|---|
| 1. **Unit tests** | Is the logic correct? | every Claude stop (H5) + CI | `tests/unit/` (no models) |
| 2. **Golden tests** | Does each model behave correctly in *our* setup? | manually on the M2 Max, after model / dependency changes | `tests/models/` (`-m model`) |
| 3. **Benchmarks** | Fast enough, within memory? | M1, then after model / dependency changes | `granit bench` |
| 4. **Quality evaluation** | Good answers on real content? | M7, then before every release and after model / prompt / retrieval changes | `granit eval` |

#### Two eval sets: public (committed) and private (never committed)

The repo is public, and real eval data (your invoices, meetings, expected values) is private. Both sets use **the same format**:

| Set | Content | Location | Results |
|---|---|---|---|
| **public** | **Synthetic**, generated by `scripts/make_eval_fixtures.py`: ~10 invoices / forms as PDFs (HTML templates → Playwright `page.pdf()` with installed Chrome; verified 2026-10-03), ~3 scripted "meetings" (macOS `say` voices, known action items), ~30 questions | `eval/public/` (committed; generated files are regenerated, not hand-edited) | `eval/results/` (committed: metrics only) |
| **private** | Your real documents and recordings, ~30 questions, hand-written references | `data/eval/` (ignored by `.gitignore`, protected by hook H2) | `data/eval/results/` (local only) |

The public set makes evaluation **reproducible by anyone** and safe to discuss in PRs. The private set is what actually decides whether granit
works for *you*. Both must pass the 1.0 criteria.

**Question format** (`questions.yaml`):

```yaml
- id: inv-total-001
  question: "What is the corrected total on invoice INV-2026-0042?"
  category: document          # document | meeting | cross-source | unanswerable
  expected_facts: ["$4,980"]  # normalized match: numbers, dates, case, whitespace
  gold_refs:                  # filled in by `granit eval label`
    - {source_sha256: "9f2c…", page: 1}
- id: mtg-actions-001
  question: "What are the action items from the Tuesday call?"
  category: meeting
  expected_facts: ["Finance", "revised invoice", "Friday"]
  gold_refs: [{source_sha256: "41ab…", start_s: 720, end_s: 790}]
- id: none-001
  question: "What is the warranty period for Widget A?"
  category: unanswerable      # not in any document → must decline, not invent
```

Extraction and summary references live next to it: `extraction/<doc>.json` (expected field values for a schema) and
`summaries/<recording>.yaml` (reference action items with owner + task).

**Labeling: `granit eval label`.** For each question, it shows the hybrid + rerank top-20 chunks; you tick the relevant ones, which become `gold_refs`
(by source hash + page or time range, so labels survive re-chunking). About 2 minutes per question, ~1 hour per set.

#### Metrics

| Area | Metric | How |
|---|---|---|
| **Retrieval** | recall@8, MRR@8 | gold refs vs retrieved chunks (a chunk matches a ref if it overlaps the page or time range); for each setup: BM25 · vectors · hybrid · hybrid + rerank |
| **Answers** | fact coverage | share of `expected_facts` found in the answer (normalized) |
| | citation precision | share of cited chunks that match a gold ref |
| | groundedness, answer relevance | Guardian pass rates (§2.4), run after answering, in its own phase |
| | **declines when unanswerable** | `unanswerable` questions must get "not found in your documents", never an invented answer |
| **Extraction** | field accuracy | exact match per field after normalization (ISO dates, numbers, whitespace); share of documents with **all** fields correct |
| | table accuracy | cell-level F1 on the normalized table grid (our own code; no TEDS dependency) |
| **Summaries** | action-item recall / precision | a reference item matches if the owner matches and the task wording overlaps (keyword match), with a Guardian custom check as a second opinion |
| **ASR** | WER | `jiwer` (Apache-2.0, dev dependency) after normalization (lowercase, punctuation, number words) |
| **Speed** (recorded, not a pass criterion in v1) | time to first token, answer time p50 / p90 | from `qa_turns.latency` |

#### Pass criteria for 1.0 (starting points)

The first full run calibrates these. Any later change to a threshold needs a one-line reason in the PR.

| Metric | Public set | Private set |
|---|---|---|
| Retrieval recall@8 (hybrid + rerank) | ≥ 0.90 | ≥ 0.85 |
| Fact coverage | ≥ 0.90 | ≥ 0.85 |
| Guardian groundedness pass rate | ≥ 0.90 | ≥ 0.90 |
| Unanswerable questions declined | ≥ 0.90 | ≥ 0.80 |
| Extraction field accuracy | ≥ 0.98 | ≥ 0.95 |
| Table cell F1 | ≥ 0.95 | ≥ 0.90 |
| Action-item recall | ≥ 0.90 | ≥ 0.80 |
| WER | ≤ 0.05 (TTS speech) | ≤ 0.10 (your recordings) |

**`1.0.0` is released only when both sets pass** (§4.3's version plan). Until then, versions stay `0.x`.

#### Trusting the judge

Guardian is a Granite model judging a Granite model, so it may share the LLM's blind spots. Before its scores count toward pass
criteria (and before v1.1 badges are turned on): hand-label **50 verdicts** (grounded yes/no) and compute agreement. **Required: ≥ 85 % agreement**, with
Cohen's κ reported. Until then, Guardian metrics are shown as **informational only**. The check is repeated after any Guardian or prompt change.

#### Running and comparing

- `granit eval run --set public|private [--retrieval bm25|vectors|hybrid|hybrid+rerank]` runs through the phase manager:
  Phase B answers every question (logging `retrieval_trace`), then Phase C runs Guardian over the answers, then metrics are computed.
- Results go to `results/<date>-<git-sha>-<set>.json`: metrics, config and model revisions only, **no document content**.
- `granit eval compare <a> <b>` prints per-metric changes and **flags regressions of 2 points or more**. Run before merging any change to models,
  prompts, chunking or retrieval.
- **Per-stage retrieval logging** (`qa_turns.retrieval_trace`) is the debugging tool: for a bad answer, it shows which stage lost the right chunk.

#### Golden-test fixes

- **Speech:** the WER test uses `say`-generated speech (committed fixture, **WER ≤ 0.05**). LibriSpeech is optional: downloaded at test time
  (`hf-internal-testing/librispeech_asr_dummy`, CC BY 4.0, credited in `tests/README.md`), never committed. This resolves the conflict with §3.5's
  "no third-party recordings".
- **Every golden test has an explicit threshold** (VAD ± 0.5 s, formats ± 1 %, WER ≤ 0.05). A test without a number can't fail.

#### Alternatives considered

- **Arize Phoenix:** not used. Core server and evals are **Elastic License 2.0** (fails the dependency-license check); telemetry (FullStory,
  Scarf) is **on by default**; traces would copy document contents outside `data/`; file-based evaluation covers v1's scale. Revisit only if
  per-stage retrieval logging in `qa_turns` proves insufficient (then: OpenTelemetry spans via Apache-2.0 packages, any compatible viewer).
- **Grading answers with the same 8B LLM:** not used. A model grading its own answers is biased; Guardian is trained for judging, and its
  agreement is measured.

**Repo layout**

```
granit/
├── .gitignore                # project-specific, tested (§4.5)
├── .streamlit/config.toml    # theme (light + dark), telemetry off, localhost only (§4.7)
├── .github/workflows/ci.yml  # CI (§4.2): lint on Linux, ty + unit tests on macOS arm64
├── .github/rulesets/main.json  # branch ruleset for main (§4.6), applied with gh api
├── .claude/settings.json     # Claude Code hooks (§4.1), committed
├── .claude/hooks/            # format_lint · protect_paths · guard_bash · guard_memory · quality_gate · session_context
├── CLAUDE.md                 # conventions + pointers to PLAN.md (static guidance)
├── CONTRIBUTING.md           # dev setup, quality checks, how to cut a release (§4.3)
├── LICENSE · NOTICE          # Apache-2.0 (§4.4)
├── licenses_overrides.toml   # hand-reviewed license exceptions for the guard (§4.4)
├── scripts/third_party_notices.py   # builds THIRD_PARTY_NOTICES.md for releases
├── pyproject.toml            # deps, [tool.ruff], [tool.ty], [tool.pytest.ini_options]
├── uv.lock · .python-version
├── src/granit/
│   ├── config.py             # model IDs + pinned revisions, paths, context sizes
│   ├── models/server.py      # start/stop/health-check mlx_lm.server
│   ├── models/phases.py      # phase manager: A / B / C switching, job batching, status
│   ├── models/convert.py     # build local MLX q8 of Guardian + record source revision
│   ├── ingest/audio.py       # decode (miniaudio / afconvert) → Silero VAD → pause-aware chunks → TurboCTC → timestamps (§3.5)
│   ├── ingest/documents.py   # Docling VLM pipeline → Markdown + element crops
│   ├── ingest/vision.py      # Granite Vision tag tasks + schema KVP
│   ├── ingest/worker.py      # Phase A process: load models once, drain job queue, exit
│   ├── store/db.py           # SQLite schema, connection settings (WAL etc.), one transaction per job
│   ├── search/embed.py       # Embedding R2 wrapper (normalized, float16, revision-tagged)
│   ├── search/vectors.py     # fp16 vector matrix cache (load at Phase B start, reload on change)
│   ├── search/hybrid.py      # BM25 + trigram (IDs) + vector search, RRF, optional rerank stage
│   ├── search/rerank.py      # Reranker R2 CrossEncoder wrapper (fp16, top-k)
│   ├── reason/llm.py         # OpenAI-compatible client, thinking controls
│   ├── reason/prompts.py     # RAG, meeting summary, cross-source templates
│   ├── verify/criteria.py    # criterion registry: text, scoring schema, yes_means (risk|pass)
│   ├── verify/guardian.py    # guardian block builder, no-think call, <score> parser
│   ├── verify/worker.py      # Phase C process (v1.1): judge unverified turns, write verdicts, exit
│   ├── evaluate/             # M7 harness: retrieval setups × questions → recall@k + Guardian scores
│   └── cli.py                # `granit ingest|ask|meeting|extract|eval` (+ `verify` in v1.1)
├── app/Home.py + app/pages/  # Streamlit: Ingest, Library, Ask, Extract
├── app/layout.py             # width constants, wrapping-row helpers, safe_md() (§4.8)
├── scripts/ui_screenshots.py # M6 check: every page × 3 widths × light/dark (§4.8)
├── scripts/make_audio_fixtures.py  # test audio via macOS `say` + `afconvert` (§3.5)
├── scripts/make_eval_fixtures.py   # public eval set: invoice PDFs (Playwright) + scripted meetings (`say`) + questions (§4.9)
├── eval/public/              # committed synthetic eval set: questions.yaml, extraction/, summaries/ (§4.9)
├── eval/results/             # committed eval results for the public set: metrics only
└── tests/
    ├── unit/                 # no models: chunking, DB, prompts, parsing, phase state machine, RRF,
    │                         #   exact/partial-ID search, per-job transaction rollback,
    │                         #   guardian block builder, <score> parsing, yes_means polarity, hook scripts,
    │                         #   approved-license model list (test_config.py),
    │                         #   dependency-license guard (test_dependency_licenses.py),
    │                         #   theme contrast + light/dark parity + privacy settings (test_theme.py),
    │                         #   safe_md escaping + every page uses it (test_layout.py),
    │                         #   audio merge/split logic + format routing (test_audio_chunking.py),
    │                         #   eval metrics: recall@k, MRR, normalization, field match, cell F1, WER (test_eval_metrics.py)
    └── models/               # @pytest.mark.model: golden tests, need downloaded weights
```

**Test policy**
- `uv run pytest` runs unit tests only (default `-m "not model"`). Should take seconds.
  - The phase manager is unit-tested as a state machine with fake processes: no overlapping phases,
    jobs batched, server restarted after a failed job.
- `uv run pytest -m model` runs golden tests on the Mac: Vision on the card's `chart.jpg` / `table.png` / `invoice.png`,
  Docling on a sample page, Speech on `say`-generated speech (**WER ≤ 0.05**; LibriSpeech optional, downloaded at test time), VAD on the generated `say` file (2 segments, ±0.5 s) and
  each audio format (WAV / FLAC / MP3 / M4A → same 16 kHz mono length), the LLM returning valid JSON, and the embedder
  (vectors have length 1; a known query ranks its matching passage first), the reranker (the card's water/H2O example ranks the H2O passages first),
  and Guardian q8 on the card's examples: function-call hallucination → `yes`, RAG groundedness (the *Eat* film) → `yes`,
  requirement check (the 4-line poem) → `no`. These also check that the local q8 build matches the card.
- Quality checks before commit: `ruff format --check && ruff check && ty check && pytest`.

## 5. Milestones

| # | Milestone | Deliverable | Effort |
|---|---|---|---|
| M0 | Scaffold | **`.gitignore` committed first** (§4.5), **public GitHub repo + About/topics/settings, then ruleset after the first CI run** (§4.6), uv project, **`LICENSE` / `NOTICE` / license metadata + dependency-license guard**, ruff/ty/pytest config, **Claude Code hooks + `CLAUDE.md`** (tested with deliberate violations), **CI workflow green on a first PR, and the first merge publishes the `v0.1.0.dev0` prerelease**, pinned deps, `granit models download` (~20.5 GB + 16.8 GB Guardian bf16 source, which can be deleted once converted), **build Guardian q8**, smoke-load each model | 1–1½ days |
| M1 | Benchmark | tok/s, peak memory per phase, **phase switch time (A→B, B→A)**, embed + rerank latency (30 pairs), **Guardian no-think time per check**; confirms the §3.3 budget | ½ day |
| M2 | Audio ingest | decode (**miniaudio** + **`afconvert`** for M4A/AAC/AIFF; ffmpeg only if the user installed it) → **Silero VAD v6** → pause-aware chunks (≤ 30 s, split at the longest pauses, overlap fallback) → TurboCTC → timestamped segments; fixture script; chunking unit tests; VAD / format / WER golden tests | 1½ days |
| M3 | Document ingest | Docling → Markdown; table/chart crops → Vision; schema KVP + `jsonschema` validation; golden tests | 2 days |
| M4 | Store + search | SQLite schema + WAL + per-job transactions; chunking with citations; FTS5/BM25 (code tokenizer) + trigram ID index + **Embedding R2 vectors + RRF hybrid search + Reranker R2** | 2–2½ days |
| M5 | Reasoning + phases | `mlx_lm.server` manager, **phase manager**; RAG, meeting summary and cross-source prompts; JSON-output tests; record `qa_turns` | 2 days |
| M6 | Streamlit UI | Ingest (upload + queue + "Process now"), Library, Ask (chat + citations + phase banner), Extract (schema editor); **theme + semantic color map (§4.7)**, **adaptive layout + `safe_md()` (§4.8)**, screenshot check of every page at 760 / 1512 / 2560 px in light and dark | 2–2½ days |
| M7 | Evaluation (§4.9) | `granit eval` harness + metrics; public synthetic set (fixture script); private set (your ~30 questions, references); `granit eval label`; Guardian agreement check (50 verdicts, ≥ 85 %); first full run → calibrate thresholds; retrieval comparison (4 setups); **1.0 gate: both sets pass** | 2½–3 days, then before every release |
| M8 (v1.1) | Verify job | Phase C in the phase manager, `verify` job type, `verdicts` table, criterion registry, custom summary criteria, verdict badges in Ask / Library | 1½ days |

About 13½–15½ days for v1, plus 1½ days for M8 (v1.1). Every milestone ends with ruff + ty + unit tests passing.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Phase switches feel slow if jobs trickle in one at a time | Batch queued jobs; "Process now" button; M1 measures switch time |
| A crashed ingest job leaves Q&A down | Phase manager always restarts `mlx_lm.server` (`try/finally`); failed jobs marked and kept for retry |
| The mlx-vlm Vision port could differ from the reference Transformers output (DeepStack injection) | Golden tests against the card's example outputs; compare one sample with Transformers on MPS if needed |
| Tight transformers version window (≥ 5.16, Docling excludes some 5.x) | Commit `uv.lock`; upgrade deliberately |
| MLX in Streamlit threads | Models only in the worker process / `mlx_lm.server` (by design) |
| Thinking mode slows answers | `reasoning_effort: "low"` by default; a per-question toggle in the UI |
| Model cards' "≥ 16 GB" claims are template text | Rely on M1 measurements |
| ID / code searches miss with the default tokenizer | Code-friendly `tokenchars` + trigram index; unit tests for exact and partial IDs |
| Vector matrix grows in Phase B memory | Measure in M1; switch to sqlite-vec at about 1M chunks |
| Unnormalized embeddings distort dot-product ranking | `normalize_embeddings=True` enforced in one wrapper; unit test checks vector length |
| Guardian yes/no polarity misread (yes = risk vs yes = met) | `yes_means` per criterion; UI shows only derived `passed`; unit tests for both polarities |
| Local Guardian q8 build judges differently from bf16 | Golden tests on the card's examples + compare q8 vs bf16 on a sample of M7 verdicts; fallback: IBM GGUF Q8_0 via llama.cpp |
| Private eval data (real invoices, expected values) leaked to the public repo | Separate private set in `data/eval/` (ignored, hook H2); public set is synthetic; results files hold metrics only |
| Tuning to a small eval set | Two sets (public + private); add questions over time; thresholds change only with a reason in the PR |
| No objective "done" for 1.0 | Pass criteria table (§4.9); `1.0.0` only when both sets pass |
| Guardian verdicts trusted blindly (custom criteria "require testing"; traces may be unfaithful) | No-think by default; traces never shown as explanations; hand-check a sample in M7 before enabling v1.1 badges |
| CI unit tests accidentally import MLX/torch or download models | Lazy imports in wrappers; `HF_HUB_OFFLINE=1`; `-m "not model"` by default |
| Dollar amounts in answers rendered as LaTeX (garbled finance answers) | All model output through `safe_md()`; `test_layout.py` enforces it |
| Layout unreadable on large monitors or cramped in half-screen windows | 720 px reading column + wrapping side panels (§4.8); M6 screenshot check at 3 widths |
| App phones home or is reachable from the network | `gatherUsageStats = false`, `server.address = "localhost"`, no Google Fonts; asserted by `test_theme.py` |
| Theme tweak breaks contrast or the light/dark pairing | `test_theme.py` re-checks WCAG AA pairs and parity in CI |
| First push to `main` publishes an unintended release | Initial push has no `pyproject.toml` / workflow; scaffold arrives via PR after the ruleset (§4.6 order) |
| Unreviewed or red code merged to `main` | Ruleset: PR required, `lint` + `test` required and up to date, no force-push / deletion, no bypass |
| Secrets pushed to the public repo | Secret scanning + push protection; `.env` / `secrets.toml` ignored (§4.5) |
| User data, secrets or multi-GB weights committed to git (stays in history forever) | Tested `.gitignore` with directory + extension rules, committed before any other file; `test_gitignore.py`; hook H2 blocks edits to `data/` and `models/` |
| A new dependency brings in GPL / AGPL / unknown-license code | Dependency-license guard in CI fails the build; exceptions only via reviewed `licenses_overrides.toml` |
| GPL obligations from ffmpeg if bundled | Decode with `miniaudio` + macOS `afconvert`; ffmpeg optional and never bundled |
| Words cut at chunk boundaries garble names and numbers | Silero VAD: chunks split at pauses; overlap + de-duplication only when there's no pause |
| Common recordings (M4A from Voice Memos / meeting apps) can't be decoded | `afconvert` (macOS built-in) before miniaudio; format golden tests |
| Using the VAD weights with no license metadata | Pin `silero-vad-v6` (MIT); `test_config.py` approved list |
| Missing attribution when distributing | `LICENSE` + `NOTICE` in wheel/sdist via `license-files`; `THIRD_PARTY_NOTICES.md` attached to every release |
| Unreviewed code released by a direct push to `main` | Branch protection: PRs + required `lint` / `test`; the release job only runs after both pass |
| Version bumped by hand without updating `uv.lock` | `uv lock --check` fails in `lint`, which blocks the release; bump only with `uv version --bump` |
| macOS CI minutes cost more (private repo) | Lint on Linux first; `test` only after lint passes; cancel outdated runs; uv cache |
| Hooks slow down every step, or block legitimate work | Standard-library scripts for pre-tool hooks; command-word matching; unit tests; personal overrides in `settings.local.json` |
| Stop gate loops on a test Claude can't fix | Honors `stop_hook_active`; Claude Code also stops forcing continuation after repeated blocks |
| PyTorch/MPS query embedder misbehaves in Streamlit threads | Lock around calls; fallback to a small sidecar process (checked in M0) |

## 7. Decisions (2026-10-03)

| Decision | Choice |
|---|---|
| Hardware | M2 Max, 32 GB (~400 GB/s) |
| Licensing (models) | Commercial use → Apache-2.0 models only (no `turboctc-nc`; multilingual embedding deferred pending Gemma tokenizer terms review) |
| GitHub repo | **Public** `darylalim/granit` (§4.6): description + 20 topics, squash-only merges, secret scanning + push protection, Dependabot alerts (no auto-upgrade PRs), ruleset on `main` (PR + `lint`/`test` required, no force-push/deletion, no bypass) stored in `.github/rulesets/main.json` |
| `.gitignore` | Project-specific (§4.5): ignore `data/`, `models/`, weight/db extensions, venv/caches/dist, secrets, OS/editor files; commit `uv.lock`, hooks, CI, license files, fixtures; guarded by `test_gitignore.py` |
| License (granit) | **Open source, Apache-2.0**: `LICENSE` + `NOTICE`, `license = "Apache-2.0"` metadata, CI dependency-license guard (permissive + MPL-2.0 only), `THIRD_PARTY_NOTICES.md` on releases, miniaudio instead of ffmpeg |
| LLM | Granite 4.2 8B q8 only; no 30B deep mode, no 3B draft |
| Memory strategy | **Phase switching**: ingest (A: Speech + Docling + Vision bf16, ~12 GB), Q&A (B: 8B q8, 32K, ~15.8 GB) and verify (C, v1.1: Guardian q8, ~10.5 GB) take turns |
| Audio | Decode: miniaudio (WAV/FLAC/MP3/Ogg) + macOS `afconvert` (M4A/AAC/AIFF/CAF), no ffmpeg; **Silero VAD v6 (MIT, via mlx-audio)** splits at pauses into ≤ 30 s chunks; TurboCTC per chunk; timestamped segments |
| Search | **Hybrid + rerank**: SQLite FTS5 BM25 + Granite Embedding English R2 vectors → RRF → Granite Reranker English R2 → top-8 |
| Storage | **SQLite** (WAL, one transaction per job) + FTS5 (code tokenizer) + trigram ID index + float16 vectors in NumPy (sqlite-vec later); files on disk by sha256. Postgres + pgvector only if multi-user |
| Verification | **Granite Guardian 4.1 8B (local MLX q8)**: M7 evaluation in v1; batch verify job (Phase C) in v1.1; never loaded with the Q&A LLM; no-think; `yes_means` per criterion |
| UI | Streamlit, 4 pages: Ingest, Library, Ask, Extract, plus a phase status banner (verdict badges in v1.1) |
| Layout | **Wide pages, width-capped content (§4.8):** Ask 720 px column + 380 px sources panel that wraps beside it from ~1470 px; Extract 2 × 520 px panels side by side from ~1512 px; Library tables stretch; sidebar collapses automatically; `safe_md()` for all model output |
| Theme | **"Granite"**: stone neutrals + indigo accent (light `#4F46E5` / dark `#6059F1`, same hue), status colors reserved for status, matching light/dark modes, bundled fonts, WCAG AA verified; telemetry off, localhost only |
| Toolchain | uv, ruff, ty, pytest |
| CI | GitHub Actions: `lint` (ubuntu: `uv lock --check`, ruff) → `test` (macos-26 arm64: `uv sync --locked`, ty, `pytest -m "not model"`) → `release` (main only); no models in CI; model tests on the M2 Max |
| Releases | Automatic: a version in `pyproject.toml` with no `v<version>` tag + green build on `main` → tag + GitHub Release (wheel, sdist, generated notes); bump with `uv version --bump`; releases only from CI; `1.0.0` = v1 |
| Claude Code hooks | H1 format/lint after edits · H2 protected paths · H3 command guards (uv only, no `turboctc-nc`, no GPU-limit tuning) · H4 one phase at a time · H5 quality gate on Stop · H6 session context; shared via `.claude/settings.json` |
| Voice input | Deferred to v2 (mlx-audio `realtime_vad` / `smart_turn` as candidates, licenses checked then) |
| Evaluation | **Four levels** (unit, golden, benchmarks, quality); **public synthetic + private** eval sets in the same format; labeled `gold_refs`; metrics for retrieval, answers (incl. unanswerable), extraction, summaries, ASR; **1.0 pass criteria on both sets**; Guardian scores count only after ≥ 85 % agreement; results history + `eval compare`; per-stage `retrieval_trace`; **no Arize Phoenix** (ELv2, telemetry on by default) |
| Next step | Plan only; implementation not started |
