# Demo video plan: "granit in 3 minutes"

A 3-minute YouTube overview, recorded automatically with Playwright from the public eval library, narrated by a local
text-to-speech voice. Nothing in it comes from `data/` user files.

## Storyline

> A project manager has a status report and a recording of the check-in meeting. Do they agree?

Files (synthetic, from `eval/public/files/`): `reports/atlas-status.pdf`, `meetings/atlas-checkin.m4a`, plus
`invoices/inv-01.pdf` for the closing Extract shot. The video ends on the eval set's cross-source question:

**"Do the Atlas status report and the check-in meeting agree on the go-live date?"**

The answer (recording test, 2026-11-08): "Yes. Both the Atlas status report and the check-in meeting state the go-live
date is November 3, 2026", citing report p. 1 and the recording. That shows documents, audio and search working in one
answer. The narration should say "cites the report and the recording", not "the moment it was said": the check-in is
25 s long and one chunk, so its citation is always 0:00.

## Shot list (~180 s)

| # | Time | Page | On screen | Caption |
|---|---|---|---|---|
| 1 | 0:00–0:10 | Title card | Name + tagline | "Private document & meeting intelligence. Runs entirely on your Mac." |
| 2 | 0:10–0:30 | Ingest | Queue with both files at "done" | "Drop in PDFs, scans and recordings" |
| 3 | 0:30–0:55 | Library → `atlas-status.pdf` | Content tab, then "Tables and charts" | "Documents become Markdown, tables included" |
| 4 | 0:55–1:30 | Library → `atlas-checkin.m4a` | Speakers panel → type a name → Transcript tab | "Who spoke when. Name the people you recognize" |
| 5 | 1:30–1:55 | Same recording | Summary tab: decisions + action items | "Summaries with decisions and action items" |
| 6 | 1:55–2:40 | Ask | Type the go-live question; answer streams; Sources panel | "Cited answers from the report *and* the meeting · offline" |
| 7 | 2:40–2:55 | Extract → `inv-01.pdf` | Invoice beside its JSON fields | "Forms → validated JSON" |
| 8 | 2:55–3:00 | End card | Repo URL, "Apache-2.0 · IBM Granite on MLX" | |

## Pipeline

Four steps, each its own process (one phase at a time, PLAN.md §2.1):

```bash
uv run --group screenshots python scripts/demo_video.py prepare  # library copy: names set, reindexed (phase A)
uv run --group screenshots python scripts/demo_video.py record   # demo-raw.webm + beats.json (phase B)
uv run python scripts/demo_voice.py                              # voiceover.md → one WAV per shot (macOS say)
uv run --group screenshots python scripts/demo_edit.py           # cut, captions, cards, audio → demo.mp4
```

Output goes to a scratch directory (`--out`, default `build/demo/`; `build/` is already gitignored). The finished `demo.mp4` is uploaded by
hand; it isn't committed.

### 1. Prepare (phase A)

- Copy `data/eval/libraries/public` to a temporary directory, the same way `scripts/readme_screenshots.py` does. The eval
  library itself is never changed.
- **Speaker naming, option (a):** set the speaker name that shot 4 types (e.g. `1='Project Manager'`) on the copy with
  `set_speaker_names`, then run the queued `reindex` with the ingest worker. Ask in shot 6 then sees the name. On camera,
  shot 4 selects the field's text and types the same name over it, then saves once. `set_speaker_names` compares the
  saved names with the stored ones (`store/db.py`) and queues nothing when they match. Saving an empty field first would
  be a real change and queue a `reindex`. **Check:** the summary must not ask to be regenerated after the save.
- Clear `qa_turns` so Ask starts with no history.
- Generate the meeting summary for `atlas-checkin.m4a` beforehand if the copy has none, so shot 5 doesn't wait on it.

### 2. Record (phase B)

- `scripts/demo_video.py` imports `settle`, `idle`, `wait_for_http`, `wait_until_ready` and `scroll_to` from the
  screenshot scripts; if they need changes, move them to a small shared module rather than copying them.
- Start Streamlit on the copy with `GRANIT_DATA_DIR` and `HF_HUB_OFFLINE=1` (as `readme_screenshots.py` does).
- **Warm-up:** ask one throwaway question off camera (before the video context opens), then clear `qa_turns` again. The
  on-camera answer then skips the LLM server start and model load. Measured: Q&A ready 15–19 s after launch, warm-up
  answer 11–13 s, then the on-camera answer in **5.1 s** (two takes, identical).
- The warm-up also clears the "Loading the search models…" banner that every page shows until the query models are
  loaded. `settle` doesn't wait for it, so no recorded page may open before the warm-up finishes.
- **Recording context: viewport and `record_video_size` both 1536×864,** light theme (dark as a second take, if wanted).
  The edit step upscales to 1920×1080 (lanczos), which makes text ~25% larger and still sharp. Tested and rejected:
  - 1920×1080 at scale 1: sharp, but the app's maximum content width leaves the right ~35% of the frame empty.
  - `device_scale_factor=1.25`: Chromium's screencast records CSS pixels, so the page lands in the corner of the frame
    with grey padding.
  - CSS `zoom: 1.25`: clicks still land correctly, but Streamlit's canvas data grid (Library) overflows the frame.
- Playwright records VP8 WebM at 25 fps, ≈ 375 kbit/s. Frames are sharp both still and in motion (typing, answer
  streaming).
- Natural-looking input:
  - typing with `page.keyboard.type(text, delay=60)`
  - `page.mouse.move(x, y, steps=25)` to an element before clicking it
  - a short-lived highlight ring injected around the clicked element (CSS `outline` on a `data-demo` attribute, removed
    after 600 ms), only in the demo context
- Scroll with `element.scrollIntoView({behavior: "smooth"})`: the app scrolls an inner container, so mouse-wheel scrolling
  on the page doesn't move it.
- **Beat log:** `beats.json` records each shot's start and end time (seconds since the video started), plus `wait`
  spans (e.g. Ask's "Running…" stretch) that the editor cuts out or speeds up.

```json
{"shots": [{"id": 6, "start": 101.2, "end": 158.9, "waits": [[107.0, 131.5]]}]}
```

### 3. Voiceover (local TTS)

- **Script:** `docs/demo/voiceover.md`, one section per shot, about 400 words in total (≈ 140 words a minute). Each
  section must fit its shot's length; `demo_voice.py` warns when a clip runs longer than its shot.
- **Voice: macOS `say`** with a Premium voice. It's part of the system: no dependency, model or license to add, and
  no GPU, so it can run at any time.
  - One-time setup by hand: System Settings → Accessibility → Spoken Content → System Voice → Manage Voices, download
    e.g. "Zoe (Premium)". Only the basic voices (Samantha, Daniel) are installed by default.
  - `demo_voice.py` takes the voice as `--voice` (default "Zoe (Premium)") and checks `say -v '?'` first, stopping
    with the setup steps above if the voice isn't installed. `say` itself exits 0 with a missing voice and silently
    uses the default one, so this check is the only safeguard.
  - Per shot: `say -v VOICE -r RATE --file-format=WAVE --data-format=LEI16@24000 -o voice/shot-N.wav -f shot-N.txt`.
    `-r` (words per minute, default ≈ 175) is the knob for fitting a clip to its shot.
  - Pronunciation: `voiceover.md` is written for the ear ("granite" for granit, "M L X" for MLX). Captions come from
    the shot list, so they still show the real names. Check each clip by ear.
- Output: `voice/shot-N.wav`, 24 kHz mono, loudness normalized to −16 LUFS in the edit step.

**Rejected: Kokoro-82M via mlx-audio** (spike, 2026-11-08). The weights are Apache-2.0 and quality was good (≈ 8× faster
than real time once warm, 2.8 GB peak), but its English front end (`misaki[en]`) fails `test_dependency_licenses.py`:
`num2words` (LGPL) is imported every time, and the espeak fallback brings in `phonemizer-fork` (GPL-3.0+) and
`espeakng-loader` (no license metadata; bundles GPL-3.0 espeak-ng). Without the fallback, words not in the dictionary
are silently skipped. spaCy also installed `en_core_web_sm` into `.venv` by itself at runtime, outside `uv.lock`.

### 4. Edit

`scripts/demo_edit.py` calls `ffmpeg` (Homebrew; not a Python dependency):

1. Cut each shot from `demo-raw.webm` using `beats.json`. Remove each `wait` span, or speed it up ×8 with a small
   "⏩ thinking" caption if the gap needs explaining.
2. Fit the video to the voiceover: a shot that's shorter than its clip holds its last frame, and one that's longer is
   trimmed.
3. Title and end cards: render small HTML pages (in the app theme's colors) with Playwright to 1920×1080 PNGs, and show
   each for its shot's length.
4. Captions: `drawtext` lower-thirds from the shot list, or burn in a `.srt`. Also export `demo.srt` to upload to
   YouTube as captions.
5. Audio: the voiceover clips at their shot start times, plus optional royalty-free music at −28 dB with ducking under
   the voice.
6. Encode: upscale 1536×864 → 1920×1080 (`scale=1920:1080:flags=lanczos`), H.264 High, CRF 18, 25 fps (Playwright's
   rate; no frame-rate conversion), `yuv420p`, AAC 192 kbps, `-movflags +faststart`, which suits YouTube.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Length / audience | 3-minute overview | YouTube discovery; the README covers the details |
| Structure | One storyline (Project Atlas) | The cross-source question shows every feature working together |
| Recording | Playwright, automated | Repeatable after UI changes; reuses the screenshot tooling |
| Speaker naming | (a) name pre-set on the copy, retyped on camera | Ask sees the name without a phase-A reindex mid-recording |
| Ingest shot | Pre-ingested library, queue shown at "done" | Live ingest is minutes long and would need phase A on camera |
| Offline claim | Caption on shot 6 | Playwright can't switch Wi-Fi off; `HF_HUB_OFFLINE=1` really is set |
| Voice | Local TTS: macOS `say`, Premium voice | Local like the app, no new dependency; Kokoro fails the license test (see Voiceover) |
| Data | Public eval library copy only | Synthetic files; never `data/` user files |

## Risks

- **Answer latency:** 5.1 s after the warm-up in both test takes, so shot 6 needs no speed-up. The beat log still lets
  the editor trim waits if a later take is slower.
- **Retyping the name requeues work:** if `set_speaker_names` sees a change (e.g. whitespace), a `reindex` is queued and
  the summary is marked stale on camera. Type the exact pre-set name, and check the copy's job queue after recording.
- **UI changes break the script:** selectors follow the screenshot scripts (`data-testid`, headings). Re-run after UI
  changes and watch the result before uploading.
- **Shot 6 answer wording:** temperature 0 can still vary with the prompt cache (CLAUDE.md, prompt A/B). The warm-up
  question is always the same one, so the on-camera answer should be stable; check it in the raw take.
- **Memory:** the recording test measured Playwright's Chrome at 1.1 GB and Streamlit (query embedder + reranker) at
  1.0 GB peak RSS. The LLM server (`granit.models.mlx_server`) wasn't sampled; PLAN.md §3.3 puts Phase B at ~16 GB
  (19.3 GB at the 16K cap), so the total stays well inside 32 GB. The real script should sample by process tree, not by
  name: a name match also counts your own Chrome windows.

## Work items

1. `scripts/demo_video.py` (`prepare`, `record`) + shared helpers with the screenshot scripts
2. `docs/demo/voiceover.md` (narration, timed per shot)
3. `scripts/demo_voice.py` (`say`, voice check, per-shot WAVs, warns when a clip is longer than its shot)
4. `scripts/demo_edit.py` (ffmpeg pipeline, cards, captions, `demo.srt`)
5. CLAUDE.md: one command line for re-recording the demo
