# Demo captions

On-screen text for the demo video (docs/demo-plan.md), rendered by `scripts/demo_edit.py`. There's no narration: music
plays under the footage, and the captions carry the story. One section per shot; the number after the last `·` in each
heading is the shot's length in the finished video.

Each non-blank line is a caption, written `headline | detail`: a bold headline that works on its own, and a smaller line
for whoever reads on. A caption starts with its shot (fading in just after), or at `@N`, N seconds into the shot, and stays
until the next one starts or the shot ends. The edit fails when a caption is on screen for less time than it takes to
read (`READ_WPS` words a second). A shot with no footage (title, end) is a card: its one line, split by `|`, under the
granit name. Lines starting with `>` are notes.

Captions say only what's on screen: Granite Guardian isn't in the footage, so it's in the YouTube description instead.

## 1 · Title · 5 s

Private document & meeting intelligence | Runs entirely on your Mac

## 2 · Ingest · 10 s

Drop in PDFs, scans and recordings | Processed in the background by IBM Granite models on Apple MLX. Nothing leaves your Mac.

## 3 · Document · 12 s

Documents become Markdown, tables included | Each table is also kept as data, next to an image of the original.

## 4 · Speakers · 17 s

> 0–11 s: speaker 1 is named and saved, and the summary updates; 11 s on: the Transcript tab (frames, 2026-11-08).

Name the people you recognize | granit only uses the names you give it. It never guesses.
@11 Who spoke when | The transcript shows who said what, with timestamps.

## 5 · Summary · 11 s

Every meeting gets a summary | Decisions and action items, with owners from the names you gave.

## 6 · Ask · 18 s

> The answer takes about ten seconds; the edit keeps a few of them (the wait is trimmed to fit the shot).

Answers that cite their sources | Across documents and meetings, fully offline.

## 7 · Extract · 9 s

Forms → validated JSON | Give it a JSON schema, get the fields back, checked and ready to download.

## 8 · End · 6 s

github.com/darylalim/granit | Open source, private by design | Apache-2.0 · IBM Granite on MLX
