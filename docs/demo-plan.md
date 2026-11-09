# Demo video plan: "granit in 90 seconds"

A 90-second YouTube overview, recorded automatically with Playwright from the public eval library, told in on-screen
captions over CC0 music. Nothing in it comes from `data/` user files.

## Storyline

> A project manager has a status report and a recording of the check-in meeting. Do they agree?

Files (synthetic, from `eval/public/files/`): `reports/atlas-status.pdf`, `meetings/atlas-checkin.m4a`, plus
`invoices/inv-01.pdf` for the closing Extract shot. The video ends on a question across both:

**"When does Project Atlas go live, according to the status report and the check-in meeting?"**

The answer (same in every take): "According to the status report, Project Atlas go-live has moved to November 3, 2026
because the data migration took longer than planned [1]. In the check-in meeting Sam says "the data migration slipped
so go live moves to november 3" [3]. Both sources agree the go-live date is November 3, 2026." It cites report p. 1
and the recording, and quotes a named speaker, which shows documents, audio, speaker names and search working in one
answer. Captions should say "cites its sources", not "the moment it was said": the check-in is 25 s long and one
chunk, so its citation is always 0:00.

## Shot list (88 s)

First planned at 3:00, cut to ~2:00 (2026-11-08) to fit the measured footage, then narrated by macOS `say` (#51). The
synthetic voice was dropped the same day (see Voiceover, below): captions now tell the story, so each shot is its
footage plus the time to read its captions, and the video is 88 s. The lengths and captions live in
`docs/demo/captions.md`, which `demo_edit.py` checks.

| # | Time | Page | On screen | Caption (headline) |
|---|---|---|---|---|
| 1 | 0:00–0:05 | Title card | Name + tagline | "Private document & meeting intelligence · Runs entirely on your Mac" |
| 2 | 0:05–0:15 | Ingest | Queue with both files at "done" | "Drop in PDFs, scans and recordings" |
| 3 | 0:15–0:27 | Library → `atlas-status.pdf` | Content tab, then "Tables and charts" | "Documents become Markdown, tables included" |
| 4 | 0:27–0:44 | Library → `atlas-checkin.m4a` | Speakers panel → type a name → Transcript tab | "Name the people you recognize", then at 11 s "Who spoke when" |
| 5 | 0:44–0:55 | Same recording | Summary tab: decisions + action items | "Every meeting gets a summary" |
| 6 | 0:55–1:13 | Ask | Type the go-live question; answer streams; Sources panel | "Answers that cite their sources" |
| 7 | 1:13–1:22 | Extract → `inv-01.pdf` | Invoice beside its JSON fields | "Forms → validated JSON" |
| 8 | 1:22–1:28 | End card | Repo URL, "Open source, private by design", "Apache-2.0 · IBM Granite on MLX" | |

Captions only say what's on screen. Granite Guardian, which the narration mentioned, never appears in the footage
(`record` doesn't run `granit verify`), so it goes in the YouTube description instead.

## Pipeline

Three steps, each its own process (one phase at a time, PLAN.md §2.1):

```bash
uv run --group screenshots python scripts/demo_video.py prepare  # library copy: speaker named, reindexed, summarized
uv run --group screenshots python scripts/demo_video.py record   # demo-raw.webm + beats.json (phase B)
uv run --group screenshots python scripts/demo_edit.py --music build/demo/music/city-loop.mp3  # cut, captions, cards, music → demo.mp4
```

Output goes to a scratch directory (`--out`, default `build/demo/`; `build/` is already gitignored). The finished `demo.mp4` is uploaded by
hand; it isn't committed.

### 1. Prepare (phases A, then B; ~20 s)

- Copy `data/eval/libraries/public` to `build/demo/library`. The eval library itself is never changed.
- The eval run already named the check-in's speakers 2 and 3 (Sam, Marcus); speaker 1, who runs the meeting, is unnamed.
- **Speaker naming, option (a):** `granit speakers … 1=Priya` on the copy, then `granit ingest` runs the queued
  `reindex` (phase A), so Ask in shot 6 sees the name. Then `granit meeting` writes a new summary (phase B) that uses
  it ("Priya requested Sam to update the barcode scanner firmware…"), so shot 5 isn't marked out of date.
- Clear `qa_turns` so Ask starts with no history.
- `prepare` fails if speaker 1 isn't named, the summary predates the names, a job is queued, or Ask has history.
- On camera, shot 4 clears speaker 1's field (before the shot's logged start), types the name and saves once. The saved
  names equal the stored ones, so `set_speaker_names` (`store/db.py`) queues nothing; `record` fails if the take left a
  job queued, because the app would pause Q&A for it.

### 2. Record (phase B; ~2 min)

- `record` copies `build/demo/library` to a temporary directory each take, so retakes start from the same state.
- `scripts/demo_video.py` imports `settle`, `idle`, `wait_for_http`, `wait_until_ready` and `clear_history` from the
  screenshot scripts.
- Start Streamlit on the copy with `GRANIT_DATA_DIR` and `HF_HUB_OFFLINE=1` (as `readme_screenshots.py` does).
- **Warm-up:** ask one throwaway question off camera (before the video context opens), then clear `qa_turns` again. The
  on-camera answer then skips the LLM server start and model load. Measured: Q&A ready 15–19 s after launch, warm-up
  answer 11–13 s, then the on-camera answer in **5.1 s** (two takes, identical).
- The warm-up also clears the "Loading the search models…" banner that every page shows until the query models are
  loaded. `settle` doesn't wait for it, so no recorded page may open before the warm-up finishes.
- **Recording context: viewport and `record_video_size` both 1536×864,** dark theme (chosen after watching a light take; `--theme light`
  records one).
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
  - a drawn cursor and a 600 ms ring at each click, injected only in the recording context: headless Chrome draws
    neither, so clicks would come out of nowhere
- Scroll with `element.scrollIntoView({behavior: "smooth"})`: the app scrolls an inner container, so mouse-wheel scrolling
  on the page doesn't move it.
- **Beat log:** `beats.json` records each shot's start and end time (seconds since the video started), plus `wait`
  spans (e.g. Ask's "Running…" stretch) that the editor cuts out or speeds up.

```json
{"shots": [{"id": 6, "start": 101.2, "end": 158.9, "waits": [[107.0, 131.5]]}]}
```

### 3. Voiceover (removed)

A narration read by macOS `say` (Zoe (Premium), `scripts/demo_voice.py`, #51) was built and dropped on 2026-11-08: it
sounded synthetic. Captions plus music replaced it. Its lessons, in case a voice comes back: Zoe reads ≈ 185 words a
minute; `[[slnc N]]` adds exact silence, but `[[emph +]]` changes nothing and `-r` can't go slower than her default;
`say` exits 0 with a missing voice and silently reads in the default one. A human recording per shot is the better
route: one WAV per shot, mixed under the footage the way `demo_edit.py` mixed the `say` clips (see git history, #52).

**Rejected: Kokoro-82M via mlx-audio** (spike, 2026-11-08). The weights are Apache-2.0 and quality was good (≈ 8× faster
than real time once warm, 2.8 GB peak), but its English front end (`misaki[en]`) fails `test_dependency_licenses.py`:
`num2words` (LGPL) is imported every time, and the espeak fallback brings in `phonemizer-fork` (GPL-3.0+) and
`espeakng-loader` (no license metadata; bundles GPL-3.0 espeak-ng). Without the fallback, words not in the dictionary
are silently skipped. spaCy also installed `en_core_web_sm` into `.venv` by itself at runtime, outside `uv.lock`.

### 4. Music (CC0, not in the repo)

The track is passed as `--music FILE` and never committed, so it needs no notice in the repo.

**Chosen: [City Loop](https://opengameart.org/content/city-loop-0) by wipics** (CC0 1.0 in its page's license field;
upbeat synths and drums; 56 s, so it plays about 1.5 times). Download it to `build/demo/music/city-loop.mp3` from
`https://opengameart.org/sites/default/files/city-loop.mp3`.

How it was picked (2026-11-08), each candidate rendered under the full video at the same loudness:
- Calm ambient and piano (First Light Particles, The Budding of Consciousness, Calm Ambient 1, Disturbingly Calm):
  all rejected, too still for a product overview.
- Upbeat electronic: City Loop (chosen), Oldskool, Scaling Up, Space Synth Wave, Electronic Synth. "Scaling up" and
  "Oldskool" also exist as CC-BY 4.0 remixes on OpenGameArt: use the CC0 originals. "Melodic EDM Loops" is only
  short loops (6.9 s), too repetitive over 88 s.

Even CC0 music can draw a YouTube Content ID claim if someone registered it: check YouTube Studio's Copyright tab while
the upload is still private.

### 5. Edit (~20 s)

`scripts/demo_edit.py` calls `ffmpeg` (Homebrew; not a Python dependency). Everything about a shot (length, captions)
is in its section of `docs/demo/captions.md`; the cut points are in `beats.json`.

1. Checks first: the raw take, a section for every recorded shot, one caption line for every card, caption times that
   start at 0 and increase inside the shot, and every caption on screen long enough to read (3 words a second: viewers
   also watch the footage).
2. A recorded shot is cut from `demo-raw.webm` (`trim`), ending 0.5 s before its logged end: the video runs a little
   behind `beats.json`, and the next page starts loading right after, so the held frame was once a blank white page.
   Footage shorter than its shot holds its last frame (`tpad`); longer footage loses the middle of its waits first
   (each keeps 1 s), then its tail. Shot 6's ~10 s wait for the answer is trimmed to ~3 s: without narration, a long
   spinner looks frozen.
3. Upscale 1536×864 → 1920×1080 (lanczos), 25 fps (Playwright's rate).
4. Captions: `headline | detail`, rendered by Playwright as a transparent 1920×1080 PNG (Source Sans, dark-theme
   colors; a 42 px headline over a 30 px muted detail line) and overlaid at the bottom right. Each fades in at its
   start (0.6 s into the shot for the first) and out as the next starts or 0.5 s before the cut. At most 820 px wide,
   so it stays right of the app's main column; keep headlines to one line (shot 6's two-line headline covered the
   Sources panel).
5. Title and end cards: the caption's lines (split at `|`) under "granit", same font and colors; the title fades in
   from black, the end card out to black.
6. Segments are video only. `join` concatenates them (video copied) and adds the music once: looped or cut to the
   video's length, faded in (1.5 s) and out (3 s), normalized to −18 LUFS (−1.5 dBTP) in two passes (measure, then
   one linear gain: one pass landed tracks between −16.1 and −18.0), then AAC 192 kbps. Without `--music` the video
   has no audio track.
7. `demo.mp4`: H.264 High, CRF 18, `yuv420p`, `+faststart`. Measured: 1:28.0, −18 ± 0.2 LUFS across the shortlist.

Not built: the "⏩ thinking" speed-up (no shot needs it).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Length / audience | 90-second overview (first 3:00, then 2:00) | YouTube discovery; the README covers the details |
| Structure | One storyline (Project Atlas) | The cross-source question shows every feature working together |
| Recording | Playwright, automated | Repeatable after UI changes; reuses the screenshot tooling |
| Speaker naming | (a) name pre-set on the copy, retyped on camera | Ask sees the name without a phase-A reindex mid-recording |
| Ingest shot | Pre-ingested library, queue shown at "done" | Live ingest is minutes long and would need phase A on camera |
| Offline claim | Caption on shot 6 | Playwright can't switch Wi-Fi off; `HF_HUB_OFFLINE=1` really is set |
| Sound | CC0 music, captions tell the story (no voice) | macOS `say` sounded synthetic; Kokoro fails the license test (see Voiceover). Captions also reach viewers who watch muted |
| Data | Public eval library copy only | Synthetic files; never `data/` user files |

## Risks

- **Answer latency:** 5.1 s after the warm-up in both test takes, so shot 6 needs no speed-up. The beat log still lets
  the editor trim waits if a later take is slower.
- **Retyping the name requeues work:** if `set_speaker_names` sees a change (e.g. whitespace), a `reindex` is queued and
  the summary is marked stale on camera. Type the exact pre-set name, and check the copy's job queue after recording.
- **UI changes break the script:** selectors follow the screenshot scripts (`data-testid`, headings). Re-run after UI
  changes and watch the result before uploading.
- **Shot 6 answer wording:** the same in every take (same warm-up question), so the question was chosen by its answer.
  With speaker 1 named, the eval set's "Do the Atlas status report and the check-in meeting agree on the go-live date?"
  answered "(source [1] and source [3/4/5])": a slash-separated citation that the badge renderer and
  `parse_citations` didn't recognize, so "[3/4/5]" showed as raw text and sources 3–5 weren't marked cited. Both now
  accept slashes (one shared `prompts.CITATION`). Of three rewordings (2026-11-08), all answered cleanly; the one used
  quotes Sam by name. A change to the prompt, retrieval or model can change it: check shot 6's last frame.
- **Memory:** the recording test measured Playwright's Chrome at 1.1 GB and Streamlit (query embedder + reranker) at
  1.0 GB peak RSS. The LLM server (`granit.models.mlx_server`) wasn't sampled; PLAN.md §3.3 puts Phase B at ~16 GB
  (19.3 GB at the 16K cap), so the total stays well inside 32 GB. The real script should sample by process tree, not by
  name: a name match also counts your own Chrome windows.

## Work items

Done (2026-11-08):

1. `scripts/demo_video.py` (`prepare`, `record`), reusing the screenshot scripts' helpers (#50)
2. A voiceover (`docs/demo/voiceover.md`, `scripts/demo_voice.py`, #51) and its edit (`scripts/demo_edit.py`, #52),
   then replaced: `docs/demo/captions.md` and captions + music in `demo_edit.py`
3. CLAUDE.md: a command line per step

Published 2026-11-09 (uploaded private, no Content ID claims, then public): <https://youtu.be/MBoblMP9Siw>, linked from the README with
`docs/screenshots/demo-thumbnail.png` (the title card, 1280×720, with a play button; also the YouTube thumbnail).
