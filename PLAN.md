# Granite Local Stack: Plan v37 (M2 Max, 32 GB)

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

Status: **v37: M0–M7 done; **the public set passes every 1.0 criterion**; Guardian judge check passed; real-world private set built and run; Vision page pass for photos and scans, merged table headers (private fact coverage 0.648 → 0.741 → 0.796 → 0.833); action-item scoring and the summary prompt fixed (private action items 0.283 → 0.467, §4.9 *Private set*, item 5); receipt sums checked (item 6); private WER triaged (item 4); form values checked against the PDF text layer (public extraction 0.935 → 0.984); the library's names and terms spell recordings (public WER 0.055 → 0.029). The private set does not pass yet (§4.9 *Private set*).** Sizes and dependency versions checked on Hugging Face / PyPI on 2026-10-03;
speeds and memory measured with `granit bench` on 2026-10-04 (§3.3).
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

**Changes in v19 (M0 + M1 results):** M0 found and fixed: `setup-uv` has no floating `@v10` tag (pinned to a SHA, §4.2); ruff
excludes `*.md` (§4); three permissive licenses added to the allow-list plus one reviewed override (§4.4). **M1 measured everything**
(§3.3): the GPU limit is **26.8 GB**, not ~21; Phase B needs a **2 GB prompt-cache cap and a 1 GB MLX cache limit** (it reached
28.8 GB with mlx-lm defaults); **prefill (215–345 tok/s) dominates time to first token** (~10 s for a RAG answer, new risk in §6);
the vector matrix moves to **fp16 on MPS** (NumPy fp16 is 180× slower, §2.3); phase switches take 6.5–9 s (§2.1); the server needs a
warm-up request after `/health`.
**Changes in v20:** **one LLM request is capped at 16K tokens** (was 32K), decided after M1: Phase B peaks at **19.3 GB** at the cap
(23.5 GB at 30K), and a 30K prompt took 2.3 min to read. Meetings over about an hour are summarized in sections (§3.3); 8-bit KV is
the fallback if M7 finds sectioned summaries miss things.
**Changes in v21 (M2):** audio ingest built (§3.5). **Word timestamps from the CTC frames** (TurboCTC returns text only), so
segments are per sentence rather than per 30 s chunk and overlap windows are stitched by time; WER normalization handles number
words, contractions and `%`; format decoding is unit-tested in CI. Measured: fixtures WER 0.0, LibriSpeech WER 4.3 %, RTF ~0.01.
**Changes in v22 (M3):** document ingest built (§3.6). **PDFs render through pypdfium2** (Docling's default renderer made
Granite-Docling miss a scanned page); **picture crops come from our own render** (Docling's were offset); charts are detected by asking
Vision yes/no and their CSV is attached as `meta.tabular_chart`, so chart data appears in the Markdown under the chart; **tables stay
Docling's by default** (equal accuracy on the card's table, 8× faster); M7 decides with hard tables and a pre-agreed rule (switch
only if Vision wins by ≥ 3 points of cell F1, §4.9), and M6 adds a per-upload "Accurate tables (slower)" option; form extraction
normalizes dates to ISO 8601 and treats nulls as missing.
**Changes in v23 (M4):** store, chunking, hybrid search and the ingest worker built (§2.3 *As built in M4*). **The FTS tokenizer
choice was corrected:** `tokenchars '-_./'` glued punctuation to ordinary words ("finance." ≠ "finance"), so the index and queries now
share a normalization that keeps IDs whole and splits everything else. Tables are chunked as Markdown; failed jobs wait for the next
run instead of being replayed; Phase A peaked at 12.8 GB with the MLX cache cap; hybrid + rerank ≈ 90 ms per question. M3's
`document.json` no longer embeds base64 page images (175 → 19 KB for a 2-page scan).
**Changes in v24 (M5):** the phase manager, Q&A with citations and meeting summaries built (§2.1, §2.1.1). **Thinking "low" is
confirmed as the default** (off rambled to the token limit, on never answered); `mlx_lm.server` has no structured output, so JSON
comes from strict instructions + schema validation + one repair turn; `qa_turns` records every answer with its retrieval trace;
measured switch: stop 0.09 s, Q&A back and warm in 7.3 s.

**Changes in v25 (M6):** the Streamlit app built (`uv run granit ui`): Ingest, Library, Ask, Extract (§2.2, §4.8). **Top navigation
instead of a sidebar:** the screenshot check found the sidebar's ~280 px pushed the Ask sources and the Extract fields *below* at
1512 px; without it they sit beside as planned. A shared `Backend` (one per server process) owns the phase manager and runs
`tick()` / "Process now" on its own thread. Pages open a SQLite connection per run (a connection belongs to its thread). Deep links
`?source=ID`, per-upload "Accurate tables", re-ingest from the Library, and an AST check that every Markdown call in `app/` escapes
what it renders (it found 14 unescaped calls on its first run). Measured in Chrome with the real models: Q&A ready 13 s after the
first visit; an upload with accurate tables → Q&A paused → back in 27 s; answers ~97 characters per line at every width.

**Changes in v26 (M7):** the evaluation harness (`granit eval run|compare|init|label|agreement`) and the public eval set
(`scripts/make_eval_fixtures.py` → `eval/public/`: 24 files, 31 questions) are built (§4.9). **The first run found that
Granite-Docling finds some tables but transcribes none of their cells** (an empty `<otsl>`: 6 of 21 tables, including two invoices'
line items), so their content never reached search; Vision now fills those tables at ingest, and "Accurate tables" now puts
Vision's cells into the chunks (it only stored them before). Text-layer words missing from the chunks: 33.6 % → 1.3 %. Results
and the open findings are in §4.9 *First results*; the table default was decided (§4.9 *Tables*: Docling, Vision for empty tables).

**Changes in v27:** **control answers** for checking the judge (§4.9 *Trusting the judge*): real answers mostly pass, so 50 hand
labels of real verdicts alone couldn't tell a good judge from one that always says "pass". Each run now also judges 12 deliberately
wrong answers; Guardian caught 12 / 12 on the public set.

**Changes in v28:** M7's first finding fixed. A **text-layer safety net** (§3.6) adds back what Granite-Docling dropped from
digital PDFs, and **every chunk's context now starts with its document's title**, because Docling marks every heading level 1 and
chunks lost which document they were from. Public set: fact coverage 0.885 → **0.962**, false declines 7.7 % → **0 %**, text-layer
words missing from the chunks 1.3 % → 0.5 %; vector MRR@8 0.859 → 0.919.

**Changes in v29:** the **private set** is built from real published content (`scripts/fetch_real_eval.py`: 5 US federal PDFs, a 1964
scan, 5 receipt photos, 2 AMI meetings, 31 questions; all sha-pinned) and its first run triaged (§4.9 *Private set*). Its biggest
finding is fixed: Granite-Docling **looped or read nothing** on receipt photos and the scan, putting junk (`loc>loc>201` ×100) or nothing
in the index. A **Vision page pass** (§3.6 item 6) reads such pages again, and a **loop check** keeps looping output out of the index.
Private set: fact coverage 0.648 → **0.741**, retrieval recall@8 0.889 → **1.000**; public set unchanged (tables 0.997, facts 0.962).

**Changes in v30:** the private set's second finding is fixed. Docling kept the Fed projections table's **two-row header**, but
Markdown has one header row, so the chunks repeated `Median 1` ×4 and lost the years. **Multi-level headers are merged** into one
row (`Median 1 / 2024`, §3.6 item 7), and **captioned tables are chunked as tables** (caption + header repeated in every piece)
instead of as sentences. Private set: fact coverage 0.759 → **0.796**; public set unchanged (no change of 2 points or more).
Action-item recall (0.283) is triaged (§4.9 *Private set*, item 5).

**Changes in v31:** action-item scoring, from the triage. The keyword **stemmer** maps word forms together (`splitting` /
`split`, `arrangement` / `arranging`, `offices` / `office`), and an **owner the transcript never says** (`metrics.mentions`: the
name's words in a row, with the usual spelling slips) is met by no owner, because nothing in the audio could give it; a guessed
name still misses. The public `peak-season` meeting had the same gap: its reference owner "Marcus" was only a speaker label in
the fixture script. Its line now says the name ("Marcus, can you start the hiring?"), so the public set still tests owners.
Private action-item recall 0.283 → **0.383**; public 0.833 → **1.000** (Marcus found), fact coverage 0.962 → **0.981** (mtg-04
answerable). **Public decision recall 0.667 → 0.333** (not a gate metric): with the request in the transcript, the summary
puts every point in `action_items` and returns no decisions (same transcript with the old wording: the Tacoma lease decision
found, at temperature 0). A summary-prompt weakness the fixed fixture exposed; next: the summary prompt (item 5).

**Changes in v32:** the **summary prompt** (§4.9 *Private set*, item 5): decisions are listed alongside action items (one that
creates a task goes in both); tasks handed out as a list get one item per person; an arranged next meeting is an item; an owner
can be a role when people are addressed by role; an unknown owner is `null`, and placeholder owners the model still writes
(`someone`, `the team`) are set to `null` in code (`meetings.clean_owners`). Public decision recall 0.333 → **1.000**, action
items 1.000; private action-item recall 0.383 → **0.467**, precision 0.475 → 0.542, decisions unchanged (0.455). **Measuring
prompts:** at temperature 0 the output still depends on what the server ran before (prompt-cache reuse changes the numerics):
the same prompt scored 0.467 or 0.383 by run order. Compare prompts each in a fresh server process; the eval's own order
(questions, then summaries) is what the results files report.

**Changes in v33:** private extraction triaged (§4.9 *Private set*, item 6): the receipts print commas, and Granite Vision
**misreads digits** on faint or crumpled print (325,600 → `325.400`, 17,908 → `17.708`, 194,000 → `174,000`) and writes `.` for
`,`. Its page reading makes the same misreads, so the page text can't correct them (v29's "every amount right" was wrong for
cord-020), and a 2× / 3× upscale doesn't fix them reliably. A schema can now declare **sums** (`x-sums`, §3.6 *Form
extraction*); a sum that doesn't add up marks the extraction invalid with the reason, instead of returning misread amounts as
valid. Field accuracy is unchanged (0.857): the remaining gap is the model's reading.

**Changes in v34:** private WER triaged (§4.9 *Private set*, item 4). Number formatting is ~1.5 % of the reference words; the
gap is **dropped words** (659 of 928 errors on ES2008b, 1,027 of 1,307 on IB4003), in runs inside speech segments where
speakers overlap or backchannel (VAD keeps 72–89 % of the audio, no long gaps). One formatting bug is fixed at the word level:
TurboCTC glues "o'clock" to the hour as a 0 (`20 clock`, `110 clock`); `audio.fix_oclock` splits an hour 1–12 + `0` before
"clock" back into `2 o'clock`, timestamps shared. WER 0.178 → 0.178; mtg-01's transcript now says "2 o'clock", but the answer
takes the first time proposed (11) over the one agreed (2): a Q&A finding, not ASR.

**Changes in v35:** **form values are checked against the PDF's text layer** (§3.6 *Form extraction*), which is independent of
Vision, unlike its page reading. An ambiguous date takes the one reading the text has (`09-05-2026` → `2026-09-05`), and a
misread name takes the single closest run of words in the text (`Sam Okator` → `Sam Okafor`). Public extraction field accuracy
0.935 → **0.984** (passes), invalid extractions 1 → 0; the one field left is `approved_by`, printed under "Manager approval"
(not guessed from labels). Public WER (0.055) is the last failing public gate.

**Changes in v36:** **names and terms in recordings** (§3.5 *Vocabulary*): the library keeps a list of the names people expect
(set on the Ingest page), and a recording's words are spelled that way after transcription (`north beam` → Northbeam, `pria`
→ Priya). The list used is recorded per recording; **Re-transcribe** in the Library applies a changed list. Eval sets may
carry a `vocabulary.txt` (the public one: the four people, Northbeam, Tacoma, Atlas). Public WER 0.055 → **0.029**
(vendor-review 0.112 → 0.034), fact coverage 0.981 → **1.000**: **the public set passes**. Layout check: 24 + 24 screenshots,
0 problems (Library with a recording: Re-transcribe beside Summarize at 760 px).

**Changes in v37:** **table section rows repeat in split chunks** (§4.9 *Private set*, item 3): a row that is one repeated,
non-numeric cell (`| Not Adjusted | Not Adjusted | … |`) labels the rows under it, and every later part of a split table now
starts with its latest section row after the header. doc-11 answered (322,862, not adjusted); private fact coverage 0.796 →
**0.833**, false declines 0.037 → 0; public unchanged (passes). **Tried and reverted:** a Q&A rule "in a recording, answer
with what a later part settles" left mtg-01 unchanged (the answer still gives the first time proposed, with both passages in
the prompt), so the prompt stays as it was.
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
│ Granite-Docling   (Docling→mlx-vlm) │   │ 16K context cap              │
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
- **Cost (measured in M1, weights in the OS file cache):** **B→A 7.6–8.9 s** (server stop 0.1 s + worker start and
  Phase A load ~7.5–8.9 s, of which ~3 s is Python imports: torch, Docling) and **A→B 6.5–6.7 s** (worker exit 0.7 s + server
  `/health` 3.3 s + first request 2.5 s). Batching jobs means you pay it once per batch, not once per file.
- **"Healthy" isn't "loaded":** `mlx_lm.server` answers `/health` before its weights are in memory (1.4 GB footprint), and loads
  them on the first request. The phase manager sends a 1-token **warm-up request** before reporting Q&A as ready.
- **In the UI:** a status banner shows the current phase. During ingest the Ask page shows "Q&A paused" and
  keeps your typed question until Q&A is back. Extract results appear in the Library when their job finishes.
- **Memory is freed by ending processes.** Stopping a process reliably returns all of its memory, which makes
  phase switching safer than unloading models inside a long-running process.
- **As built in M5 (`models/phases.py`):** each answer holds a `chat()` lease; no switch starts while one is open, and questions
  during a switch get `PhaseBusy` with the banner text ("Q&A paused: ingesting 2 files"). `tick()` (the UI backend's thread calls it every 2 s) switches when
  jobs are queued and Q&A has been idle for **60 s**; **"Process now"** switches as soon as the answer in progress finishes. The
  worker runs as its own process (`python -m granit.ingest.worker`) and drains the whole queue; Q&A is **always** restarted,
  health-checked and warmed up afterwards, even when the batch fails. The manager refuses to start a second LLM server if one is
  already answering on the port. **Measured (2 files):** stop 0.09 s → worker 13.0 s (process start, model loads, both files;
  peak 13.8 GB) → Q&A back and warm in 7.3 s. Unit tests run it as a state machine with fake processes (never two phases at
  once, one batch per queue, Q&A back after a crash); a golden test does a real B → A → B switch and answers about the new file.

### 2.1.1 Reasoning as built (M5, `reason/`)

- **Answers:** search (hybrid + rerank, top 8) → a prompt with as many sources as fit the **16K budget** (exact counts with the
  LLM's own tokenizer; lower-ranked sources are dropped first, a smaller one may still fit) → Granite 4.2 with thinking **low** →
  citations (`[2]`, `[1, 3]`, `[2–4]`) mapped back to chunks → the turn stored in **`qa_turns`** with its `retrieval_trace`.
  The rules: only facts from the sources, cite every statement, copy numbers / IDs exactly, **decline with exactly "Not found in your
  documents."** when unanswerable (no LLM call at all when retrieval finds nothing), say when sources disagree.
- **Meetings:** transcript lines with timestamps → JSON (`summary`, `decisions`, `action_items` with owner / task / due in the
  transcript's own words), validated against a JSON Schema with one repair turn. Over the section budget: section by section, then a
  combine step that keeps the final version of anything changed later. Stored as a `summary` extraction (replaced on re-run,
  removed on re-ingest).
- **Measured on the fixture library:** "Q2 revenue" → 145 thousand USD [chart]; "draft to Legal" → Marcus by Wednesday [meeting, memo];
  invoice total and due date → exact; a cross-source question cites report, memo and meeting; questions outside the documents are
  declined. 2–5 s per answer. The 31 s meeting: valid JSON on the first try (6.9 s): Marcus / Elena action items, Tuesday decision.
- CLI: `granit ask "…" [--thinking off|low|on]` (streams the answer) and `granit meeting SOURCE`; both use the running Q&A server,
  or start one for the command (never while an ingest worker holds Phase A).

### 2.2 Models stay out of the Streamlit process

Streamlit re-runs the script on every interaction and runs each session in its own thread. Keeping MLX
models in the ingest worker and in `mlx_lm.server` keeps the UI responsive and avoids cross-thread MLX
use. The UI only calls `granit` library functions, sends HTTP requests and queues jobs.

**One exception: the query embedder and reranker.** Together about 0.6 GB, each call well under a second, so they run in the
Q&A backend (loaded once with `st.cache_resource`, used behind a lock) rather than as another server.

**As built in M6 (`src/granit/ui/`):** `Backend` is created once per Streamlit server (`st.cache_resource`) and shared by every
browser tab. Its thread waits for any granit ingest process left running, loads Embedding R2 + Reranker R2, starts Q&A, then
calls `tick()` every 2 s or `process_now()` when "Process now" is pressed. Answers and summaries run under a chat lease and one
lock (MPS models, and writes on the backend's shared connection). It stops the LLM server when the cache entry is released and
at exit (verified: no server left after the app stops). Pages never touch models or processes: they call `ask`, `summarize`,
`request_processing`, `status`. A question asked while Q&A is paused is kept and asked when Q&A is back.
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

#### Tokenizer choice for codes and IDs (corrected in M4)

- The default FTS5 tokenizer splits on punctuation: in testing, the query `INV` matched `INV-2026-0042`, and a
  search for the full ID only matches it as a sequence of separate pieces.
- `chunks_fts` uses `tokenize="unicode61 tokenchars '-_./'"`, which keeps `INV-2026-0042`, `v1.2.3` and `ab_12` as single tokens.
- **M4 finding:** on its own, that also glues punctuation to ordinary words. "approved by Finance." was indexed as `finance.` and
  "Legal/Procurement" as one token, so a search for "finance" or "legal" found nothing, which would affect most sentence-final words.
  **Fix (`store/text.py`):** the index sees a normalized `chunks.search_text` and queries go through the same function: a token
  **with a digit** keeps its `-_./` (an ID, version, date or amount), any other token is split at them (`finance.` → `finance`,
  `Legal/Procurement` → `legal procurement`).
- `chunks_trigram` (`tokenize="trigram"`) handles partial-ID lookups (`2026-004`). The search layer sends ID-like query tokens
  (letters/digits with `-_./`, at least 3 characters, not a bare number) there and merges the results into hybrid search with RRF.
- Unit tests cover exact-ID, partial-ID, plain-word and sentence-final-word queries.

#### Vector search

- **Now:** load `chunk_vectors` into a **float16 PyTorch tensor on MPS** (in the same process as the query embedder) **once when
  Phase B starts**, and reload it when the jobs table changes. Rank with one matrix product + `torch.topk`.
- **Measured in M1** (top-50, p50): NumPy has no BLAS path for float16, so `fp16 @ fp16` is slow; the earlier "11.6 ms" was fp32 math.

  | Vectors | Matrix (fp16) | NumPy fp16 | NumPy fp32 (2× memory) | **MPS fp16** |
  |---|---|---|---|---|
  | 150K | 230 MB | 182 ms | 9.5 ms | **1.0 ms** |
  | 1M | 1.5 GB | 1,269 ms | 61 ms | **5.2 ms** |

- **Upgrade trigger:** move to **sqlite-vec** (vectors searched on disk, not in memory) when the matrix gets big enough to matter
  in the Phase B memory budget (about 1M chunks ≈ 1.5 GB), not when search gets slow (5 ms at 1M on MPS).
- **Re-embedding:** `model_revision` is stored per vector, so after a model upgrade we re-embed only the outdated rows.

#### Operations

- Backup = copy `data/` (or `sqlite3 granit.db ".backup ..."` while running). Encryption at rest = macOS FileVault.

#### As built in M4 (`store/`, `search/`, `ingest/worker.py`)

- **Schema** is versioned (`PRAGMA user_version`, migrations in `store/db.py`): `sources`, `jobs`, `chunks` (+ `search_text`),
  `chunks_fts` and `chunks_trigram` (external-content FTS5, kept in sync by triggers), `chunk_vectors`, `extractions` and `meta`
  (`corpus_version`, bumped by every completed job or deletion so Phase B reloads its vector matrix). `qa_turns` and `verdicts` come
  with M5 / M8 as later migrations.
- **Files** are stored once by sha256 (`files/<sha256><ext>`, written atomically); adding the same file again is a no-op.
- **Jobs:** claimed atomically (`UPDATE … RETURNING`); a failure is recorded and **re-queued for the next worker run** (not replayed
  in the same run, which could cost minutes of Vision time while Q&A is paused) up to 3 attempts; jobs left `running` by a crashed
  worker are re-queued when the next worker starts. Re-ingesting a source replaces its chunks.
- **Chunking (`store/chunking.py`):** Docling's `HierarchicalChunker` with **Markdown tables** (its default "North, 1 = 1,240"
  triplets drop the column names questions use); consecutive text under one heading merged to ~300 tokens (1,100 characters at
  3.75 characters per token); tables and charts kept as their own chunks; oversized text split at sentences and tables by rows with
  the header repeated. Transcripts: consecutive segments up to the same size, with start / end seconds. The heading path is stored as
  `context` and prepended for BM25, embeddings and the reranker. Citations: `report.pdf · p. 2`, `pp. 3–4`, `call.m4a · 12:40`.
- **Worker:** loads models lazily by kind (an audio-only batch never loads Vision) and caps MLX's buffer cache at 1 GB.
  **Phase A peaked at 12.8 GB** ingesting all M2 / M3 fixtures (M1: 15.6 GB without the cap).
- **Measured on the fixtures (5 files, 12 chunks):** query embedding ~12 ms, vector top-50 0.7 ms, reranking ~75 ms; **hybrid + rerank
  ≈ 90 ms per question** after a one-time warm-up (`Searcher.warm_up()`, ~100 ms, for the UI to call at start). Every expected answer
  was in the top 3: the chart for "Q2 revenue", the table for "North Q1 units", the meeting for "who sends the draft to Legal", the memo
  table for "who approves the invoice", the invoice for its own ID; the partial ID `2026-004` reaches results only through trigrams.
- **Known, for M7:** the reranker scored the Markdown chart table just below the report's intro paragraph (0.881 vs 0.893) although
  RRF ranked it first. M7's retrieval comparison should test **linearizing tables / charts for the reranker** (e.g. "Q2: Revenue
  145 thousand USD") before changing anything.

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

**Never loaded alongside the Q&A LLM.** Guardian q8 (12.4 GB peak, M1) next to Phase B (~16 GB) would exceed the 26.8 GB GPU limit (§3.3).
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
| LLM | **`ibm-granite/granite-4.2-8b-q8-mlx`** | mlx-lm 0.32.0 server | 9.34 GB | B | **16K context cap** (prompt ≤ 14,336 + output ≤ 2,048); **36 tok/s** decode (22 at 16K), prefill 264–345 tok/s (M1, §3.3). |
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

**Measured in M1 (2026-10-04, `granit bench`; results in `bench/results/`).** Memory is each process's **physical footprint**
(what Activity Monitor shows; it includes Metal buffers, which RSS misses). The GPU limit is Metal's recommended working set:
**26.8 GB** on this M2 Max with macOS 26 and no tuning (`iogpu.wired_limit_mb = 0`). The plan's earlier "~21 GB" figure was out of date.

| Phase | Loaded | Plan estimate | **Measured peak** | What drives it |
|---|---|---|---|---|
| A: Ingest | Speech + VAD + Docling + Vision bf16 + Embedding | ~12 GB | **15.6 GB** (11.7 GB after loading) | MLX keeps freed buffers cached between stages (Docling, Vision); the worker should set an MLX cache limit (M2/M3) |
| B: Q&A, RAG prompts (~3K tokens) | 8B q8 server + embedder + reranker + 150K-vector matrix | ~15.8 GB | **~15.3 GB** (server ~12.5 + retrieval 2.8) | Weights 9.3 GB; the prompt cache is capped at 2 GB |
| B: Q&A, prompt at the **16K cap** | same | ~15.8 GB | **19.3 GB** (server 16.5 + retrieval 2.8) | 16K × 160 KB of KV = 2.6 GB, plus prefill buffers and the capped prompt cache |
| *B at 30K (not allowed, for reference)* | same | — | *23.5 GB* (server 20.7 + retrieval 2.8) | *why the cap exists* |
| C: Verify / M7 eval | Guardian 4.1 8B q8 | ~10.5 GB | **12.4 GB** (8.2K-token check) | Weights 8.9 GB + KV + buffers |

**Phase B needs two limits** (defaults in `config.py`, applied by `ServerConfig` / `granit.models.mlx_server`). With mlx-lm's defaults it reached
**28.8 GB, over the GPU limit**:
- **`--prompt-cache-bytes 2GB`:** mlx-lm keeps up to 10 prompt KV caches with **no byte limit**. Over 10 distinct RAG prompts the server grew
  10.8 → 16.2 GB with the default, and stayed at ~12.5 GB with the cap.
- **MLX cache limit 1 GB:** MLX caches the old buffers each time the KV cache grows. One 30K request on a fresh server peaked at 24.8 GB
  with defaults and **20.9 GB** with both limits (in-process: 19.8 → 16.3 GB). Same speed either way.
- `--prefill-step-size` (512 vs 2048) made **no** difference to the peak.

**Decision (after M1): cap one LLM request at 16K tokens.** At 30K, Phase B used 23.5 GB, leaving ~8 GB of the 32 GB for macOS, the
browser and Streamlit. At the 16K cap it peaks at **19.3 GB** (7.5 GB below the GPU limit, ~12.7 GB left for the system), and one
capped request on a fresh server peaks at 15.1 GB. Meetings over about an hour are summarized in sections (below). 8-bit KV
(`--kv-bits 8`) stays untested, as the fallback if M7 shows sectioned summaries miss things.

**Speed (M2 Max, measured):**

| | Result |
|---|---|
| LLM decode | **35.8 tok/s** short context → 27 tok/s at 8K → **21.6 tok/s at the 16K cap** (16 at 30K) |
| LLM prefill (compute-bound) | 345 tok/s at 1K → 304 at 8K → 263 at 16K (215 at 30K). **TTFT: 2.9 s (1K), 12.5 s (4K), 26 s (8K), 61 s (16K cap)**; 137 s at 29K |
| Server start | `/health` in 3.7 s; weights load on the first request (warm-up 1.1–3.2 s) |
| Query embedding | 13 ms p50 |
| Rerank 30 pairs (~300 tokens each) | **555 ms** p50 |
| Vector search, top-50 | 1.0 ms at 150K, 5.1 ms at 1M (fp16 on MPS; §2.3) |
| Phase A load | 5.5 s (Speech 0.4, VAD 0.02, Docling 1.5, Vision 2.8, Embedding 0.8) |
| Speech | real-time factor **0.009** (112 s of audio: VAD 0.56 s + TurboCTC 0.45 s) |
| Docling | 2.4 s per page |
| Vision bf16 | prefill ~660 tok/s, decode ~44 tok/s; `<chart2csv>` 3.0 s, `<tables_html>` (800 tokens) 20 s |
| Chunk embedding (ingest) | 57 chunks/s (~300 tokens each) |
| Guardian no-think check | **9.0 s** p50 for 3K tokens (8 chunks + answer); 26 s at 8.2K. All 10 scores parsed |
| Phase switch | B→A 7.6–8.9 s · A→B 6.5–6.7 s (§2.1) |

A RAG answer's time to first token is therefore about **embed 13 ms + rerank 0.55 s + prefill ~9 s ≈ 10 s**, dominated by prefill (§6).

**The 16K context cap and long transcripts (decided after M1).** One LLM request is at most **16,384 tokens**: a prompt of up to
**14,336** plus up to **2,048** of output (`LLM_CONTEXT_TOKENS`, `LLM_MAX_OUTPUT_TOKENS` in `config.py`). `mlx_lm.server` doesn't
enforce a limit, so the M5 prompt builder must count tokens and refuse or split anything larger.
- **RAG answers** (~3K tokens) are far below the cap.
- **Meetings:** an hour is about 12–14K tokens, so meetings up to **about an hour** are summarized in one pass. Longer ones are split
  into sections at transcript segment boundaries (≤ ~12K tokens each), each section is summarized, then the section summaries are
  combined. Each step reports progress in the UI.
- **Why not 32K:** a 30K prompt takes ~2.3 min to read before the first token (§3.3) and pushed Phase B to 23.5 GB. Two ~14K sections
  read faster in total and keep Phase B at 19.3 GB (table above).
- **Fallback if M7 shows sectioned summaries miss things** (e.g. action-item recall < 0.90 on long meetings): measure
  `--kv-bits 8` (halves KV) at 32K, then decide.

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

**Vocabulary (v36).** CTC decoding can't be prompted with expected words, so names are corrected afterwards, at the word level:
the library's names and terms (`Store.vocabulary`, the Ingest page's *Names and terms in recordings*, one per line or
comma-separated) are applied by `audio.apply_vocabulary` in the worker. A run of up to 3 words that joins to a term becomes it
(`north beam` → Northbeam, `tacoma wa` → Tacoma WA); a single word ≥ 0.85 similar to a term of 5+ letters does too (`pria` →
Priya). Shorter terms match only exactly, so `same` never becomes Sam. The new word spans the run's time; punctuation after it
is kept. The list used is saved in the recording's info; Library's **Re-transcribe** (enabled when the list has changed) queues
it again. Changing the list never edits existing transcripts on its own.

**Pipeline (`ingest/audio.py`, built in M2)**
1. **Decode** → 16 kHz mono (above). A WAV whose codec miniaudio can't read (e.g. μ-law) falls back to `afconvert`.
2. **VAD:** `mlx_audio.vad.load(<silero-vad-v6, pinned revision>)` → `get_speech_timestamps(..., return_seconds=True)` with the model's
   defaults (threshold 0.5, min silence 100 ms). Silero splits continuous speech about every 10 s with ~0.1 s gaps; padding merges them.
3. **Chunk:** pad each segment by 0.3 s, then merge neighboring segments into chunks of **at most 30 s**, **splitting at the longest
   pauses** (recursively). If continuous speech runs past 30 s with no pause, fall back to fixed windows with a 1 s overlap.
4. **Transcribe** each chunk with TurboCTC, **with word timestamps.** TurboCTC's `generate()` returns text only, so `AudioTranscriber.words()`
   runs the same greedy CTC decode and keeps each token's frame (**80 ms per frame**); tokens are grouped into words at the BPE
   word-start marker (`Ġ`). A golden test asserts the text is identical to `generate()` (the code uses mlx-audio 0.5.7's
   `_tokenizer` and `compute_features`).
5. **Stitch overlaps by time:** each word in an overlap is kept from the window whose half contains its start; a word repeated within 0.4 s
   across the seam is dropped (CTC timing is least precise at a window's edge: the continuous fixture produced "and the the printer").
6. **Segments** (the citation unit) break at pauses and are at most 30 s: one per spoken sentence on the meeting fixture.
   `transcript.json` (via `Transcript.to_json()`) keeps every word's time, so citations like "Tuesday call · 12:40" can point at the word.
7. **Report** total speech vs silence time per file (`Transcript.speech_s` / `silence_s`) for the Library page.

`granit transcribe FILE… [--json DIR]` runs the pipeline from the command line (M4 wires it into the ingest worker and the store).

**Measured in M2:** every fixture (TTS) **WER 0.0**; **LibriSpeech (73 clips, 8 min of read speech) WER 4.3 %**, transcribed in 4.5 s
(real-time factor ~0.01). VAD boundaries within 0.1 s of the truth.

**WER normalization (`evaluate/asr.py`)** so formatting isn't counted as error: lowercase, punctuation removed (digits kept together:
`4,980` → `4980`), `%` → "percent", common contractions expanded ("let's" → "let us", which TurboCTC writes), possessive apostrophes
dropped, and **number words → digits** ("twelve percent" = "12%"; "twenty twenty six" → `20 26`, since a number ends when the next word
can't extend it). Without it, the meeting fixture scored 6.2 % on formatting alone.

**Tests**
- **Unit** (`test_audio_chunking.py`, `test_asr_metrics.py`, no models): format routing; **decoding every format** (WAV, FLAC, MP3, Ogg,
  M4A, AIFF, CAF → identical 16 kHz mono length, within 1 %; these need only miniaudio and `afconvert`, so they run in CI rather than as
  golden tests); the μ-law fallback; clear decode errors; padding / merging; longest-pause split; the 30 s cap; overlap windows; CTC
  collapse and word grouping; seam stitching; segments; JSON round trip; number words and normalization.
- **Golden** (`-m model`, `tests/models/test_audio.py`): VAD finds both phrases with boundaries within ±0.5 s; our timed decode equals
  `generate()`; word times fall inside speech; meeting WER ≤ 0.05 with one segment per sentence; continuous speech uses the overlap with
  WER ≤ 0.05 and no duplicated word; every format WER ≤ 0.05; LibriSpeech WER ≤ 0.10 (downloaded at test time, skipped offline).
- Fixtures are generated by `scripts/make_audio_fixtures.py` (macOS `say` voice *Samantha* + `afconvert`; MP3 / Ogg via ffmpeg at
  generation time only), so no third-party recordings and no licensing questions. `manifest.json` holds references and expected boundaries.
- **Phase A note for the ingest worker (M4):** set an MLX cache limit there, as `mlx_server` does for Phase B: M1 measured Phase A
  peaking at 15.6 GB from cached buffers.

**Later (v2 voice input):** mlx-audio also has `realtime_vad` (streaming endpointing) and `smart_turn` (end-of-turn detection) for
microphone input. Their weights' licenses get checked when v2 starts. Speaker diarization (`sortformer`, `nemotron_diarization`) stays out
of scope (licenses unchecked).

### 3.6 Documents: Docling + Granite Vision (built in M3)

**Pipeline (`ingest/documents.py`, `ingest/vision.py`)**
1. **Docling VLM pipeline** (Granite-Docling 258M on MLX) → `DoclingDocument` (text, tables with cells, pictures; page provenance for
   citations) → `document.json` + `document.md`. Inputs: PDF (digital or scanned) and images (PNG, JPEG, TIFF, WebP, BMP).
   **PDF pages are rendered by pypdfium2** (`PyPdfiumDocumentBackend`): with Docling's default backend (docling-parse) a scanned page
   rendered slightly darker and softer, and Granite-Docling then output only the logo and stopped (direct model calls on the same page
   rendered by pypdfium2 read it fully). Digital PDFs convert identically with either backend.
2. **Pictures:** each picture at least **72 page units** on both sides (one inch on a PDF; logos and icons are skipped) is **cropped from
   our own page render** (pypdfium2 at 3×, 216 dpi; images at native size; 1.5 % padding). Docling's own picture crops came out offset
   at `images_scale` > 1. Granite Vision then answers *"Is this image a chart …? yes or no"* (0.6–1.6 s; no extra classifier model).
3. **Charts** → `<chart2csv>` → the rows are attached to the picture as `meta.tabular_chart` (with `created_by` = Vision + revision), so
   the **chart's data appears as a table right under it in the Markdown** and in `document.json`: searchable and citable by page.
4. **Tables stay as Docling extracted them** by default (`DOCUMENT_VISION_TABLES = False` in `config.py`). On the card's table both Docling
   and Vision `<tables_html>` got **96/96 values right**, but Docling took 3.4 s for the whole page and Vision 29 s for the one table.
   The Vision path is built and tested (`granit convert --vision-tables`). **Decision (after M3): Docling stays the default until M7
   compares both on hard tables**; **M7 decided (2026-10-04): Docling by default, Vision for every table Docling leaves empty** (rule in §4.9: *Tables: Docling vs Vision*). Until then, the M6 Ingest page offers
   **"Accurate tables (slower)"** per upload, which turns the Vision path on for that document. **M7 found and fixed silent table
   loss:** Granite-Docling sometimes returns a table with no cells (an empty `<otsl>`: dense and long tables, and the line items of
   two invoice layouts). Such tables now always go to Vision, and Vision's cells replace Docling's in the document (and so in the
   chunks); Docling's own grids are kept in `derived/<id>/docling_tables.json` for the eval.
5. **Text-layer safety net (v28).** Granite-Docling also drops plain text on some layouts: on one invoice layout it kept the labels
   ("Invoice number:") and dropped their values ("GS-2026-0117"), and it misread `PO-48502` as `PO48502`. For a **digital PDF**, each
   line of the PDF's own text layer is compared with what Docling captured on that page (as canonical tokens: `$9,360.00` = `9360`,
   any date format = ISO); a line with an uncaptured token containing a digit (an ID, amount or date), or with half its words
   uncaptured, is added back under a top-level heading *"Text found only in the PDF's text layer (page N)"* with that page's
   provenance, so it's searchable and cited by page. Headers and footers count as captured. Scans and images have no text layer
   and are unchanged (they would need a Vision page pass). Public set: 4 + 4 field lines recovered on the two affected invoices, the
   correct `PO-48502`, and the contacts Vision had glued together; text-layer words missing from the chunks 1.3 % → 0.5 %.
   **Chunks also say which document they're from:** Docling marks every heading level 1, so a chunk kept only its nearest heading
   ("Invoice") and lost the vendor; every chunk's context now starts with the document's first heading
   ("Cedar & Pine Catering > Invoice").
6. **Vision page pass (v29).** On the private set Granite-Docling **looped** on receipt photos (counting `1 2 3 … 242`, or
   `loc>loc>loc>201` lines, with no provenance), took a whole receipt for one picture (stored as the word "Other"), and read only the
   header of a typewritten scan. A page **without a text layer** is read again by Granite Vision when Docling's reading of it
   **loops** (one line or word ≥ 30 % of the output) or has **fewer than 40 distinct words**, unless the page has a filled table or a
   chart with data (a table page has few words, and Docling's structure is what §4.9 scores: tables stayed 0.997). Vision's reading
   (prompt *"Read all the text in this image, line by line…"*; the model card has no tag for it) replaces everything Docling put on
   the page if it found more distinct words, under *"Text read from the page image by Granite Vision (page N)"*. Vision loops too
   (`$0.21` ×400 on one receipt): a looping reading is **retried with a repetition penalty of 1.1**, which isn't the default because
   it changes digits on clean receipts (325.400 → 325.600); a reading that still loops is dropped. A **looping page is cleared
   either way**, and a digital page Docling looped on is cleared and refilled by the text-layer safety net. Docling's items without
   provenance belong to the page before them in reading order. Cost: 3–6 s per receipt, ~45 s for a looping page (both attempts).
   Measured: 4 / 5 receipts read with every amount right (cord-004, crumpled, also picks up handwriting behind it; v33: on
   faint cord-020 two digits are misread, §4.9 *Private set*, item 6); the scan's
   subject, sender, recipients and comments, but not its date stamp.
7. **Merged table headers (v30).** A table's leading header rows with spanning cells are merged with the leaf header row into one
   (`Median 1` over `2024` → `Median 1 / 2024`): from Docling's cell spans, or from Vision's HTML (`colspan` rows). A caption next to
   a table no longer turns the pair into plain text: the table is split by rows, and every piece repeats the caption and the header.
   Spaced numeric ranges keep their dash in the search text (`3.9 – 4.3`). Fixed: the year columns (cross-01) and the central
   tendency range (doc-03). Not fixed: Docling **shifted the row labels** of one block (March values under the *Memo* row), so
   doc-02 still reads the wrong row: a partly wrong table that no header rule catches.
8. **Crops** sent to Vision are saved as `crops/p<page>_<kind><n>.png`; every Vision result is an `Extraction` (kind, format, content,
   valid, errors, page, crop, model + revision), ready for the `extractions` table (M4).

**Form extraction (`VisionModel.extract_fields`, `granit extract FILE --schema S.json`)**
- The model card's **VAREX prompt** with the user's JSON Schema; the schema is checked before any model time is spent (must be an object).
- **Null = not found:** null fields are reported as `missing`, not errors; a missing `required` field is an error.
- **Dates:** Granite Vision rewrites dates in its own formats (`2026-09-14` came back as `14/09/2026`, `2026-10-14` as `10-14-2026`;
  a schema hint didn't help). For `"format": "date"` fields, unambiguous dates are normalized to ISO 8601 (day/month order only when a
  part is > 12; month names understood) and then **format-checked**: an ambiguous `03/04/2026` is kept and flagged, never guessed.
- **PDFs:** up to 4 pages (`--pages`) are extracted page by page and merged field by field (first non-null wins).
- **Text layer (v35):** a digital PDF's own text (`documents.text_layer_lines`, the pages extracted) checks the values before
  validation (`vision.resolve_against_text`). A `"format": "date"` value that's ambiguous takes the one day/month reading found
  in the text (both or neither: kept and flagged). A string with letters that the text doesn't contain takes the single closest
  run of the same number of words (similarity ≥ 0.85, no tie), so a misread name is corrected and a correct value is never
  moved. Numbers and amounts are left alone (the nearest similar number may be another one); images and scans have no text.
- **Sums (v33):** a schema may list arithmetic that must hold, `"x-sums": [{"total": "total", "parts": ["subtotal",
  "-discount", "service_charge", "tax"]}]` (`-` subtracts; a missing part counts as 0; a rule needs its total and one part).
  Amounts are parsed as printed (`377,859` = `377.859` = 377859; `1.234,56` = 1234.56). A sum off by more than 0.01 is an
  error, so a misread digit makes the extraction invalid with the reason (`total: subtotal - discount = 154600, but total is
  174,600`). `x-` keys are never shown to the model.

**Measured in M3:** the card's chart → CSV exact (10/10 values), table 96/96 (Docling and Vision), invoice fields exact; the scanned
2-page report: text, table and chart (4/4 values) exact, logo skipped; the digital memo exact; the generated invoice (PNG and PDF):
all 6 fields exact with ISO dates, absent field reported missing. Docling ≈ 1.2–2.7 s per page; chart check + extraction ≈ 4.5–5.6 s.

**Tests:** unit (`test_vision_outputs.py`, `test_documents.py`, no models: CSV / HTML / JSON parsing, the VAREX prompt, schema checks,
null handling, required fields, date normalization + format checks, `x-sums` and amount parsing, page merging, crop geometry, page rendering, chart data in Markdown);
golden (`tests/models/test_documents.py`): the card's three examples plus the generated scanned report, digital memo and invoice.
Fixtures: `scripts/make_document_fixtures.py` (Pillow + macOS Helvetica; the memo is printed by headless Chrome).

## 4. Development toolchain

| Tool | Use |
|---|---|
| **uv** | Project + lockfile (`uv.lock`), Python 3.12 pinned in `.python-version`, `uv run …` everywhere |
| **ruff** | Lint + format (`ruff check`, `ruff format`); config in `pyproject.toml`. `*.md` is excluded: ruff 0.16 also formats Python code blocks inside Markdown and rewrote this plan's snippets in M0 |
| **ty** | Type checking (`ty check src tests app`) |
| **pytest** | Tests; markers split fast unit tests from model-backed tests |
| **Streamlit** | UI (`uv run granit ui`, which runs `streamlit run app/Home.py` from the project root) |

### 4.1 Claude Code hooks

Hooks make the plan's rules **automatic** while building granit with Claude Code, instead of relying on the model to remember them.
Static guidance (conventions, architecture) goes in `CLAUDE.md`; hooks handle what must **always** happen or **never** happen.

| # | Event · matcher | Script | What it does | Blocks? |
|---|---|---|---|---|
| H1 | `PostToolUse` · `Edit\|Write` | `format_lint.py` | For edited `*.py`: `uv run ruff format <file>` then `uv run ruff check --fix <file>`. Remaining lint errors → **exit 2**, stderr shown to Claude so it fixes them right away | Feedback only (the edit already happened) |
| H2 | `PreToolUse` · `Edit\|Write` | `protect_paths.py` | Denies edits to `data/**` (user data, `granit.db`), `models/**` (downloaded / converted weights), `uv.lock` (change via `uv add` / `uv lock` only), `.env*` | **Yes** (exit 2 + reason) |
| H3 | `PreToolUse` · `Bash` | `guard_bash.py` | Denies: `pip install` / `python -m pip` (→ "use `uv add`"); `huggingface-cli` (deprecated → `hf`); anything referencing **`turboctc-nc`** (non-commercial license); `rm -rf` on `data/` or `models/`; `sudo sysctl iogpu…` (the plan needs no GPU-limit tuning); `git tag` / `git push --tags` / `gh release create` (releases come only from CI, §4.3) | **Yes** |
| H4 | `PreToolUse` · `Bash` | `guard_memory.py` | If the command loads models (`pytest -m model`, `granit ingest\|eval\|verify`, `mlx_lm.*`, `mlx_vlm.*`) **and** an `mlx_lm.server` / granit worker process is already running → deny with "stop the running phase first": two phases at once would exceed the 26.8 GB GPU limit | **Yes** |
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
| `test` | **`macos-26`** (Apple Silicon, matches the dev Mac's macOS 26) | `uv sync --locked` · `ty check src tests app` · `pytest -m "not model"` | The lock includes macOS-only MLX packages; ty needs them installed to resolve imports |
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
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7  # v10.2.0
      - run: uv lock --check
      - run: uv run --locked --only-group lint ruff format --check
      - run: uv run --locked --only-group lint ruff check
  test:
    needs: lint
    runs-on: macos-26
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7  # v10.2.0
        with:
          enable-cache: true   # caches uv's download cache keyed on uv.lock (torch, mlx, ...)
      - run: uv sync --locked
      - run: uv run ty check src tests app
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
- **Security:** `permissions: contents: read`, no secrets used. GitHub's own actions are pinned to major versions (`checkout@v7`,
  `upload-artifact@v7`, `download-artifact@v8`). **`setup-uv` is pinned to the v10.2.0 commit SHA**: astral-sh publishes exact version
  tags only, so `@v10` doesn't resolve (found in M0; the first CI run failed on it).
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
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7  # v10.2.0
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
- **Allowed:** MIT, MIT-0, BSD-2/3-Clause, Apache-2.0, ISC, PSF-2.0, Zlib, HPND, 0BSD, Unlicense, CC0-1.0, MPL-2.0 (unmodified use only),
  plus three permissive OSI licenses found in the M0 audit: **MIT-CMU** (Pillow), **CNRI-Python** (regex), **BSL-1.0** (Boost, inside torch).
  M0 found 146 packages and one hand-reviewed override (`pypdfium2`: free-text metadata; BSD-3-Clause OR Apache-2.0, bundled PDFium
  build licenses all permissive).
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
showEmailPrompt = false           # no first-run prompt (it would block `granit ui` without a terminal)
maxUploadSize = 2048              # MB: an hour of 44.1 kHz stereo WAV is ~600 MB (Streamlit's default is 200)

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

**Per-page patterns** (all pages use `st.set_page_config(layout="wide")` and **top navigation**, so the header stays in the same
place and each page limits its own content):

| Page | Pattern | Tested behavior |
|---|---|---|
| **Ask** | Wrapping row: **720 px** conversation (left-aligned) + **380 px** sources panel; `st.chat_input(..., width=720)` | Sources **beside** from **1470 px** (13" MacBook Air) up, **below** on smaller windows; the input lines up with the conversation (same x and width) at every width |
| **Extract** | Wrapping row, centered: **520 px** document image │ **520 px** extracted fields | **Side by side from 1512 px** (14" MacBook Pro) up; stacked and centered below that |
| **Library** | Tables `width="stretch"`, fixed `height` with scrolling | Uses the full width: more columns visible on big monitors |
| **Ingest** | 720 px column: upload area + job queue | Same reading width as Ask |
| **All** | `st.navigation(..., position="top")`, no sidebar | **Changed in M6:** with navigation in a sidebar, its ~280 px pushed the Ask sources and Extract fields below at 1512 px. At 760 px the top navigation collapses into a menu |

```python
# src/granit/ui/layout.py: the shared helpers every page uses (each container gets a key → CSS class st-key-<key>)
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
  Ask / Library / Extract rendering code calls `safe_md` (grep-style check) so new code can't skip it. **As built:** an AST check of
  every page: each call that renders Markdown (`markdown`, `caption`, `info`, `error`, `badge`, `subheader`, …) may only interpolate
  escaped values (`safe_md`, `answer_md`, `document_md`, `transcript_md`) or numbers (`{n:d}`). Its first run found 14 unescaped
  calls (file names, error messages, model names). Document Markdown needs it too: the invoice's table has `$620.00 … $3,720.00`.
- **Width constants live in one place** (`src/granit/ui/layout.py`). Changing them moves the switch points, so the M6 screenshot check runs again.
- **Display scaling caveat:** switch points are CSS pixels; browser zoom and macOS "Larger Text / More Space" change the effective width,
  so switch points move a little but the behavior stays the same.

**M6 layout check:** `scripts/ui_screenshots.py` (Playwright + installed Chrome, dev-only dependency) captures every page at
**760, 1512 and 2560 px** in **light and dark** mode (24 screenshots) and reports panel positions (beside or below) and characters per line.
It's a manual pre-merge check for UI changes, not part of CI (it needs a running app).
`uv run --group screenshots python scripts/ui_screenshots.py --data DIR [--source ID] [--form ID]` (Playwright is in its own
dependency group, so CI doesn't install it). **M6 result (2026-10-04):** 24 screenshots, 0 problems: no LaTeX, no sideways scroll,
the chat input aligned with the conversation at every width; Ask sources and Extract fields **beside** at 1512 and 2560 px,
**below** at 760 px; answers **~97 characters per line** at 15 px (wrapped prose measured in a chat message) at every width.

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
| **public** | **Synthetic**, generated by `scripts/make_eval_fixtures.py`: ~10 invoices / forms as PDFs (HTML templates → Playwright `page.pdf()` with installed Chrome; verified 2026-10-03), **~8 hard tables** (see *Tables: Docling vs Vision*), ~3 scripted "meetings" (macOS `say` voices, known action items), ~30 questions | `eval/public/` (committed; generated files are regenerated, not hand-edited) | `eval/results/` (committed: metrics only) |
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

#### Tables: Docling vs Vision (decided after M3)

M3 found Docling and Vision `<tables_html>` **equally accurate (96/96) on one clean, ruled table**, with Vision ~8× slower (~30 s per
table, while ingest pauses Q&A). One easy table can't settle it, so M7 tests the cases where a 4B vision model could beat the 258M
Granite-Docling.
- **Hard tables in the public set (~8):** merged / multi-level headers, a borderless table, a skewed and noisy scan, a table split across
  two pages, a dense numeric table (≥ 15 rows × 8 columns), cells with line breaks, and two ordinary tables as a control.
  Each is generated with known cell values (HTML → PDF via Playwright; the scan variants are rasterized and distorted with Pillow).
- **Scoring:** table cell F1 for **both paths on every table**, plus seconds per table. Reported per table type and overall.
- **Decision rule (agreed before the run):**
  - Vision beats Docling by **≥ 3 points of cell F1 overall** → `DOCUMENT_VISION_TABLES = True`.
  - Vision wins by ≥ 3 points only on **some table types** (e.g. scans) → send only those to Vision, if the type can be detected
    reliably at ingest (e.g. scanned pages); otherwise keep Docling.
  - Otherwise → keep Docling as the default.
- The result and the decision are recorded here and in §3.6; until then the per-upload "Accurate tables (slower)" option covers hard cases.
- **M7 result (2026-10-04, `eval/results/2026-10-04-b59bdd1-public.json`), mean cell F1 over the 8 tables:**

  | Path | Cell F1 | Seconds per table | Notes |
  |---|---|---|---|
  | Docling alone | 0.747 | 1.3 | 0.0 on the dense and the split tables: it found them but emitted no cells |
  | Vision on every table | 0.925 | 10.6 | 0.50 on line breaks (`1800 Port WayTacoma`), 0.92 borderless |
  | **Docling, Vision for the tables Docling left empty** (built in M7) | **0.997** | Vision only where needed | only merged headers below 1.0 (0.98, both paths) |

- **Decision (2026-10-04): Docling by default, Vision for every table Docling leaves empty; `DOCUMENT_VISION_TABLES = False`.**
  This is the rule's second branch: Vision's overall lead (0.925 vs 0.747, +17.8 points) comes entirely from the empty tables, which
  ingest detects reliably; on every table Docling did transcribe, Vision was equal or worse (line breaks glued: `1800 Port WayTacoma`).
  Best measured result (0.997) at ~1/8 of the cost of Vision everywhere (10.6 s vs 1.3 s per table, while ingest pauses Q&A).
  - **Known gap:** a table Docling transcribes *partly wrong* isn't sent to Vision. The public set barely tests this (merged headers:
    0.98 on both paths). "Accurate tables" stays available per upload for such documents.
  - **Revisit when the private set runs:** it scores the same three paths on your tables. If Vision beats Docling by ≥ 3 points on
    tables Docling *did* transcribe, the rule's first branch applies and the default switches to Vision.

**`1.0.0` is released only when both sets pass** (§4.3's version plan). Until then, versions stay `0.x`.

#### First results (M7, 2026-10-04, public set)

`eval/results/2026-10-04-b59bdd1-public.json`, reproduced exactly after a fresh re-ingest. 9 minutes end to end (A 220 s, B 203 s, C 110 s).

| Metric | Result | Pass at | |
|---|---|---|---|
| Retrieval recall@8 (hybrid + rerank) | **1.000** | ≥ 0.90 | ✅ all four setups 1.000; MRR@8: BM25 0.936, hybrid 0.923, hybrid + rerank 0.914, vectors 0.859 |
| Fact coverage | 0.885 | ≥ 0.90 | ❌ 2 questions need field values Docling dropped (below), 1 a misheard name ("Pria"), 1 an owner said in the first person |
| Unanswerable questions declined | **1.000** | ≥ 0.90 | ✅ (2 of 26 answerable questions were declined, both for the dropped values) |
| Citation precision | 0.957 | | |
| Guardian groundedness / answer relevance | 0.958 / 0.958 | ≥ 0.90 | informational until the judge check (below) |
| Extraction field accuracy | 0.935 | ≥ 0.98 | ❌ 4 of 62 fields: `Sam Okator`, a missed `approved_by`, two ISO dates rewritten as `09-05-2026` (flagged invalid, never guessed). **v35: 0.984 ✅** (text layer; `approved_by` left) |
| Table cell F1 | **0.997** | ≥ 0.95 | ✅ with the empty-table fallback (Docling alone: 0.747) |
| Action-item recall | 0.833 | ≥ 0.90 | ❌ "*I'll* post the job ads": no speaker diarization, so no owner (v31: the name was never said; the fixture now says it, 1.000) |
| WER | 0.056 | ≤ 0.05 | ✅ **v36: 0.029** (vocabulary). Was ❌ vendor-review 0.112: "Northbeam" → "north beam", "Priya" → "pria"; the other two 0.018 and 0.037 |
| Answer time p50 / p90 | 4.6 / 6.1 s | (recorded) | first token 2.8 / 4.6 s |

**Thresholds are not changed:** every failure above is a real gap, not a measurement problem. Measurement problems found on the way
were fixed in the harness instead (written amounts like `$16,500` vs spoken words in WER; a misheard owner name counted twice).

**Open findings (next steps, for decision):**
1. ~~**Granite-Docling drops field values next to their labels**~~ **Fixed in v28** (§3.6 item 5): the text-layer safety net, plus
   document titles in chunk context (with the recovered lines in view, the model had declined a question whose answer line didn't
   say which vendor it came from). Public set `eval/results/2026-10-04-b6dc4e6-public.json`: **fact coverage 0.962** (passes;
   document and cross-source questions 1.000), false declines **0**, MRR@8 up in every setup (vectors 0.859 → 0.919), Guardian
   12 / 12 controls. Scans would still need a Vision page pass.
2. **No speaker diarization:** action items said in the first person have no owner. A v1 limitation (diarization is not in the plan's
   model set); the summary prompt could ask for "unknown" explicitly.
3. ~~**Vision rewrites ISO dates**~~ into an ambiguous `MM-DD-YYYY`: validation flags it (as designed). **Fixed in v35** for
   digital PDFs: the ambiguity is resolved against the text layer (§3.6 *Form extraction*).

#### Trusting the judge

Guardian is a Granite model judging a Granite model, so it may share the LLM's blind spots. Before its scores count toward pass
criteria (and before v1.1 badges are turned on): hand-label **50 verdicts** (grounded yes/no) and compute agreement. **Required: ≥ 85 % agreement**, with
Cohen's κ reported. Until then, Guardian metrics are shown as **informational only**. The check is repeated after any Guardian or prompt change.

**Control answers (v27, `evaluate/controls.py`).** Real answers mostly pass (23 of 24 on the public set), so a judge that always said
"pass" would agree with a person ~96 % of the time on real verdicts. Each `granit eval run` therefore also judges **12 deliberately
wrong answers** built from real ones: a **number** changed (`$12,640.00` → `$17,317.00`: groundedness should fail) or **another
question's answer** under this question (answer relevance should fail). They serve twice:
- **Automatically, every run:** `guardian.controls.caught`, the share of planted errors caught. **Public set (2026-10-04): 12 / 12**
  (numbers 6 / 6 ungrounded; swaps 6 / 6 irrelevant), while relevance correctly still passed 4 of the 5 number controls that
  answered the question. Real-answer pass rates never include controls.
- **In the hand check:** `granit eval agreement` mixes them in **blind**, up to a third of the 50, so the labels contain failures
  to catch: a judge that missed every control could agree on at most ~67 % and fails the 85 % bar. Agreement and κ are reported for
  real answers and controls separately.

Planted errors are blunter than real ones, so the hand check (on real answers too) still decides whether Guardian counts.

**Judge check result (2026-10-04, public set, `eval/public/judge_agreement.json`): passed. Guardian's groundedness now counts
toward the pass criteria.** 50 verdicts labeled blind: **agreement 0.92** (≥ 0.85), **Cohen's κ 0.81**; real answers 0.92
(κ 0.54, few real failures to agree on), controls 0.92 (κ 0, uninformative: Guardian failed every control). Of the 4 disagreements,
2 came from the labeling tool, which cut sources at 300 characters and hid the evidence for two correct answers (a table's week 7,
a transcript's later line), so the person said "not supported" and Guardian, seeing everything, passed them. The tool now shows
sources whole, so the true agreement is at least 0.92. The other 2: a correct answer whose reasoning narration was judged
unsupported, and a planted "12 %" (source: 9 %) labeled supported. Repeat after any Guardian or prompt change.

#### Private set (2026-10-06, real published content)

Built by `scripts/fetch_real_eval.py` into `data/eval/` (never committed): 5 US federal PDFs (Fed projections, FOMC minutes, two Census
releases, IRS W-9), a 1964 NARA routing sheet (scan, no text layer), 5 CORD receipt photos (Indonesian amounts), 2 AMI meetings
(~35 min each, four speakers, room mix) and 31 questions whose references were written by others (AMI annotators, CORD labelers).

| Metric | First run | v29 | v30 | Pass at | Cause of the remaining gap |
|---|---|---|---|---|---|
| Retrieval recall@8 | 0.889 | **1.000** | **1.000** | ≥ 0.85 | ✅ (BM25 0.852 → vectors 0.833 → hybrid 0.870 → + rerank 0.889 in the first run) |
| Fact coverage | 0.648 | 0.741 | 0.796 | ≥ 0.85 | ❌ v37: **0.833** (doc-11 fixed). Left: doc-02 row labels, the scan (Vision misreads), mtg-01 / mtg-04 answers, mtg-06 merged number |
| Unanswerable declined | 1.000 | 1.000 | 1.000 | ≥ 0.80 | ✅ |
| Guardian groundedness | 1.000 | 1.000 | 0.962 | ≥ 0.90 | informational (no private hand labels); 12 / 12 controls caught |
| Extraction field accuracy | 0.857 | 0.857 | 0.857 | ≥ 0.95 | ❌ item 6 below: Vision misreads digits on faint print (both receipts now flagged invalid by `x-sums`) |
| Action-item recall | 0.283 | 0.283 | 0.283 | ≥ 0.80 | ❌ item 5 below; v31 scoring fix 0.383, v32 summary prompt **0.467** |
| WER | 0.178 | 0.178 | 0.178 | ≤ 0.10 | ❌ item 4 below: dropped words in overlapping speech (number formatting ≈ 1.5 % of words) |

**Triaged causes (read from the stored chunks and answers):**
1. ~~**Docling on photos and scans:** loops or nothing~~ **Fixed in v29** (§3.6 item 6): both receipt questions now answered;
   the scan questions 0 / 2 → 0.5 / 2 (its date stamp is still missed).
2. ~~**Merged table headers:**~~ **Fixed in v30** (§3.6 item 7), except doc-02 (shifted row labels). The Fed projections table has
   a two-row header; Docling kept both rows, but Markdown has one header row, so split chunks repeated `Median 1 | Median 1 | …`
   (years lost) and 3 questions read the wrong column (the range 3.7–4.3 instead of the central tendency 3.9–4.3; 4.9 instead of 4.6) or
   declined. The §4.9 *Tables* rule asks for Vision on such tables when the private set shows it winning.
3. **Table captions missing from chunk context:** the Census release's seasonally adjusted and not-adjusted tables are identical
   chunks; the answer took 283,293 (adjusted) for 322,862 (not adjusted). v30 finding: the label is not a caption but a section
   row inside the table (`Adjusted` / `Not adjusted`, one cell spanning every column); the question now declines.
   **Fixed in v37:** split table parts repeat their latest section row (`chunking.section_row`); doc-11 answered.
4. **Spoken numbers and WER (v34 triage):** TurboCTC's number formatting turned "eleven o'clock" into `110 clock` (~~fixed in
   v34~~, `audio.fix_oclock`) and merges adjacent or repeated numbers: a stuttered "twelve… twelve fifty" → `€1212.5`, "12 13"
   → `1213`, "3 3" → `33` (not fixed: `1213` can be a real number). Scoring gaps (`first` / `1st`, a lone `hundred` / `100`)
   are worth 0.001. Most of the WER is **dropped words** where the four speakers overlap or backchannel: the reference
   interleaves every speaker, one CTC stream can't. Reaching ≤ 0.10 needs a stronger speech model, not formatting fixes.
   mtg-01 (v34): the transcript says "2 o'clock", the answer still gives the first proposal (11): answers should prefer what
   was finally agreed.
5. **Action items (v30 triage, 3 of 11 matched):** the AMI references name **roles** as owners ("Project Manager") for 6 of 11
   items, which the annotators knew from the corpus metadata; with no speaker labels the summary gives `null` or a first name, so
   2 items with the right task still miss on the owner. Ignoring owners, recall is 0.467, so most of the gap is real: one
   person's task merged into another's (the interface concept under *industrial design*), a role read as an owner (*market trend
   watcher*), the next-meeting time missed, two "poll others" items folded into one. The keyword matcher is also strict: its
   stemmer doesn't map `splitting` → `split` or `arrangement` → `arrang`. The summary also writes `someone` for unnamed owners
   (the prompt asks for `null`). ~~Matcher stemming; owners the transcript never says~~ **fixed in v31** (0.383; one owner is a
   guessed first name, one task matches only in meaning). ~~The summary prompt~~ **v32**: 0.467. Left (IB4003 1 / 3 of its 6,
   ES2008b 3 / 5): owners given as first names where the reference has a role (likely the same person; nothing in the audio
   says so), tasks that match only in meaning (no shared keywords), an ASR-garbled role ("market trend wa watching"). Prompt
   wording alone is unlikely to reach 0.80 on two 35-minute room-mix meetings; open: a semantic matcher, more meetings, or
   recalibrating the private threshold (with a reason, §4.9).
6. **Receipt amounts (v33 triage):** the receipts print commas. cord-020 (faint dot-matrix): Vision writes `.` for `,` in all
   four amounts and misreads two digits (325,**6**00 → `325.400`, 17,**9**08 → `17.708`); cord-004 (crumpled): 1**9**4,000 →
   `174,000`. The page-pass reading has the same misreads on cord-020; on cord-004 it reads 194,000, but `174,000` is one digit
   from both 194,000 and 174,600, so a correction would be a guess. Upscaling (2× fixes cord-004, 3× doesn't; contrast is
   unstable) is noise, not a fix. `x-sums` now flags both receipts (and none of the 3 correct ones or the references). Without a
   better reader, field accuracy tops out near 0.89–0.94 (separators counted as wrong, like v28's rewritten dates).

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
├── app/Home.py + app/app_pages/  # Streamlit: Ingest, Library, Ask, Extract (page scripts only)
├── src/granit/ui/            # layout.py (widths, safe_md), views.py (page data), backend.py (phases + Q&A), session.py
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
| M6 | Streamlit UI | Ingest (upload + queue + "Process now" + **"Accurate tables (slower)"** per upload, §3.6), Library, Ask (chat + citations + phase banner), Extract (schema editor); **theme + semantic color map (§4.7)**, **adaptive layout + `safe_md()` (§4.8)**, screenshot check of every page at 760 / 1512 / 2560 px in light and dark | 2–2½ days |
| M7 | Evaluation (§4.9) | `granit eval` harness + metrics; public synthetic set (fixture script); private set (your ~30 questions, references); `granit eval label`; Guardian agreement check (50 verdicts, ≥ 85 %); first full run → calibrate thresholds; retrieval comparison (4 setups); **tables: Docling vs Vision on hard tables → decide `DOCUMENT_VISION_TABLES` by the §4.9 rule**; **1.0 gate: both sets pass** | 2½–3 days, then before every release |
| M8 (v1.1) | Verify job | Phase C in the phase manager, `verify` job type, `verdicts` table, criterion registry, custom summary criteria, verdict badges in Ask / Library | 1½ days |

About 13½–15½ days for v1, plus 1½ days for M8 (v1.1). Every milestone ends with ruff + ty + unit tests passing.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Phase switches feel slow if jobs trickle in one at a time | Batch queued jobs; "Process now" button. **M1: B→A 7.6–8.9 s, A→B 6.5–6.7 s** (§2.1) |
| A crashed ingest job leaves Q&A down | Phase manager always restarts `mlx_lm.server` (`try/finally`); failed jobs marked and kept for retry |
| The mlx-vlm Vision port could differ from the reference Transformers output (DeepStack injection) | Golden tests against the card's example outputs; compare one sample with Transformers on MPS if needed |
| Chart detection misses a chart (Vision answers "no") or charts a photo | Yes/no check measured on the card's chart and logo; misses leave the image as a placeholder (no wrong data); M7 public set adds charts of several types |
| Vision rewrites field values (dates seen in M3) | `format: date` normalization + format check; ambiguous dates flagged, never guessed; M7 field accuracy |
| Granite-Docling silently drops content (M7: empty tables in 6 of 21; field values on one invoice layout) | Empty tables → Vision at ingest (M7); digital PDFs: text-layer safety net (v28, 0.5 % of words still uncaptured on the public set); scans and photos: Vision page pass (v29) when Docling loops or reads < 40 distinct words; looping output never indexed |
| Tight transformers version window (≥ 5.16, Docling excludes some 5.x) | Commit `uv.lock`; upgrade deliberately |
| MLX in Streamlit threads | Models only in the worker process / `mlx_lm.server` (by design) |
| Thinking mode slows answers | **`reasoning_effort: "low"` by default (confirmed in M5)**; per-question toggle in the UI. On a 3K-token RAG prompt: off **rambled to the token limit** (32 s), low gave a short answer in 11 s, full thinking used its whole budget without answering. On the fixture library, low answers in 2–5 s. Known: on some questions low still narrates its reasoning in the answer ("…So answer: 145"); M7's answer checks track it |
| No structured output in `mlx_lm.server` (no JSON schema / `response_format`) | Strict JSON instructions, parse + JSON Schema validation, one repair turn with the error (`LLMClient.chat_json`); meeting summaries were valid on the first try in M5 |
| **Long prompts are slow to start (M1)**: prefill is compute-bound at ~260–345 tok/s, so TTFT is ~10 s for a 3K-token RAG prompt and ~1 min at the 16K cap | Keep RAG prompts small (top-8 chunks ≈ 3K tokens); stream answers; show "reading N chunks…" progress; reuse the prompt cache for follow-up questions on the same document; 16K cap with sectioned meeting summaries |
| Sectioned meeting summaries miss links across sections (e.g. a decision reversed later in the meeting) | Combine step sees every section summary; M7 action-item recall on meetings over an hour; fallback: `--kv-bits 8` at 32K (§3.3) |
| **Phase B memory grows past the GPU limit (M1)**: mlx-lm keeps up to 10 prompt KV caches with no byte limit, and MLX caches freed KV buffers; a 30K request reached 24.7 GB alone | `--prompt-cache-bytes 2GB` + MLX cache limit 1 GB via `granit.models.mlx_server` (defaults in `config.py`); 16K context cap; `granit bench` re-checks the budget |
| Model cards' "≥ 16 GB" claims are template text | Rely on M1 measurements |
| ID / code searches miss with the default tokenizer | Code-friendly `tokenchars` + trigram index; unit tests for exact and partial IDs |
| Vector matrix grows in Phase B memory | fp16 on MPS: 230 MB at 150K chunks (M1); switch to sqlite-vec at about 1M chunks (1.5 GB) |
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
| Memory strategy | **Phase switching**: ingest (A: Speech + Docling + Vision bf16, 15.6 GB measured), Q&A (B: 8B q8, **16K context cap**; ~15 GB for RAG, 19.3 GB at the cap, with the 2 GB prompt-cache cap + 1 GB MLX cache limit) and verify (C, v1.1: Guardian q8, 12.4 GB) take turns, under the 26.8 GB GPU limit (§3.3) |
| LLM context | **16K tokens per request** (prompt ≤ 14,336 + output ≤ 2,048), enforced by the M5 prompt builder; meetings over ~1 hour summarized in sections; fallback `--kv-bits 8` at 32K if M7 shows sectioned summaries miss things |
| Audio | Decode: miniaudio (WAV/FLAC/MP3/Ogg) + macOS `afconvert` (M4A/AAC/AIFF/CAF), no ffmpeg; **Silero VAD v6 (MIT, via mlx-audio)** splits at pauses into ≤ 30 s chunks; TurboCTC per chunk; timestamped segments |
| Documents | Granite-Docling via Docling (pypdfium2 page rendering) → Markdown + `document.json`; pictures ≥ 72 pt → Vision yes/no chart check → `<chart2csv>` into `meta.tabular_chart`; tables from Docling unless `DOCUMENT_VISION_TABLES` (per-upload "Accurate tables"), **with Vision for any table Docling leaves empty (M7)**; decided in M7: Docling by default, revisit if the private set shows Vision winning on transcribed tables (§4.9); **pages without a text layer that Docling loops on or barely reads → Vision page pass (v29, §3.6)**; forms: VAREX prompt + JSON Schema, dates → ISO 8601 (§3.6) |
| Search | **Hybrid + rerank**: SQLite FTS5 BM25 + Granite Embedding English R2 vectors (fp16 matrix on MPS, 1 ms at 150K) → RRF → Granite Reranker English R2 (555 ms for 30 pairs) → top-8; ≈ 90 ms per question on the fixture library (M4) |
| Storage | **SQLite** (WAL, one transaction per job, versioned schema) + FTS5 over normalized `search_text` (IDs kept whole, words split from punctuation) + trigram ID index + float16 vectors on MPS (sqlite-vec later); files on disk by sha256. Postgres + pgvector only if multi-user |
| Verification | **Granite Guardian 4.1 8B (local MLX q8)**: M7 evaluation in v1; batch verify job (Phase C) in v1.1; never loaded with the Q&A LLM; no-think; `yes_means` per criterion |
| UI | Streamlit, 4 pages: Ingest, Library, Ask, Extract, plus a phase status banner (verdict badges in v1.1) |
| Layout | **Wide pages, width-capped content (§4.8):** Ask 720 px column + 380 px sources panel that wraps beside it from ~1470 px; Extract 2 × 520 px panels side by side from ~1512 px; Library tables stretch; top navigation, no sidebar; `safe_md()` for all model output |
| Navigation | **Top navigation, no sidebar** (decided 2026-10-04, M6). With the page links in a sidebar, its ~280 px pushed the Ask sources and the Extract fields *below* at 1512 px (found by the §4.8 screenshot check); without it they sit beside as planned (~97 characters per line at every width). Four pages fit across the top and fold into a menu at 760 px; Streamlit recommends top navigation for 3–7 pages; the sidebar would hold nothing else (the phase banner is at the top of each page). Rejected: narrower panels (answers ~86 characters per line, a smaller invoice image, and still borderline on a 13" MacBook Air at 1470 px); a sidebar collapsed by default (page links hidden behind a toggle). Revisit only if app-wide controls (e.g. global filters) need a home |
| Theme | **"Granite"**: stone neutrals + indigo accent (light `#4F46E5` / dark `#6059F1`, same hue), status colors reserved for status, matching light/dark modes, bundled fonts, WCAG AA verified; telemetry off, localhost only |
| Toolchain | uv, ruff, ty, pytest |
| CI | GitHub Actions: `lint` (ubuntu: `uv lock --check`, ruff) → `test` (macos-26 arm64: `uv sync --locked`, ty, `pytest -m "not model"`) → `release` (main only); no models in CI; model tests on the M2 Max |
| Releases | Automatic: a version in `pyproject.toml` with no `v<version>` tag + green build on `main` → tag + GitHub Release (wheel, sdist, generated notes); bump with `uv version --bump`; releases only from CI; `1.0.0` = v1 |
| Claude Code hooks | H1 format/lint after edits · H2 protected paths · H3 command guards (uv only, no `turboctc-nc`, no GPU-limit tuning) · H4 one phase at a time · H5 quality gate on Stop · H6 session context; shared via `.claude/settings.json` |
| Voice input | Deferred to v2 (mlx-audio `realtime_vad` / `smart_turn` as candidates, licenses checked then) |
| Evaluation | **Four levels** (unit, golden, benchmarks, quality); **public synthetic + private** eval sets in the same format; labeled `gold_refs`; metrics for retrieval, answers (incl. unanswerable), extraction, summaries, ASR; **1.0 pass criteria on both sets**; Guardian scores count only after ≥ 85 % agreement; results history + `eval compare`; per-stage `retrieval_trace`; **no Arize Phoenix** (ELv2, telemetry on by default) |
| Next step | The private set's remaining causes (§4.9 *Private set*): merged table headers (Vision for such tables), table captions in chunk context, then spoken numbers and action items; 1.0 when both sets pass |
