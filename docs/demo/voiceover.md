# Demo voiceover

Narration for the 2-minute demo (docs/demo-plan.md), read by macOS `say` through `scripts/demo_voice.py`. One section
per shot; the number after the last `·` in each heading is the shot's length in the finished video, and the script fails
when a clip runs longer.

Written for the ear, not the eye: "granite" for granit, "M L X" for MLX (`say` reads "PDFs", "IBM" and "JSON" well as
they are). Captions come from the shot list, so they show the real names. Lines starting with `>` are notes and aren't
read. A blank line inside a section is a pause (`--pause`, 450 ms): put one where a speaker would breathe.

Written the way people talk: short sentences, contractions, one idea at a time. Zoe ignores `[[emph +]]` and won't read
slower than her default, so the rhythm has to come from the text and the pauses (spike, 2026-11-08).

## 1 · Title · 8 s

This is granite.

Private document and meeting intelligence, running entirely on your Mac.

## 2 · Ingest · 17 s

Start by dropping in your files. PDFs, scans, meeting recordings.

granite queues them up, and works through them in the background, whenever you're not asking questions. It all runs
locally, with IBM Granite models on Apple's M L X. Nothing leaves your Mac.

## 3 · Document · 18 s

Here's a project status report. It's turned into clean Markdown, with its headings, text and tables intact.

Each table is also kept as data, right next to a picture of where it came from. And for tricky tables, an accurate mode
reads them again with Granite Vision.

## 4 · Speakers · 22 s

Recordings are transcribed with timestamps, and granite works out who spoke when.

For each speaker, you get a few of their lines, and a clip you can play. Name the ones you recognize. Here, speaker one
is Priya.

granite only uses the names you give it. It never guesses who someone is. And now, the transcript shows who said what.

## 5 · Summary · 15 s

Every meeting gets a summary. What was discussed, what was decided, and who's doing what. The owners come from the names
you gave.

You can even write your own rules, like "every action item has an owner", and have Granite Guardian check them.

## 6 · Ask · 28 s

> The question is typed on screen while this plays; the answer arrives about six seconds after it's sent.

Now for a question that needs both sources. When does Project Atlas go live, according to the status report and the
check-in meeting?

granite searches the whole library, and answers in seconds.

Every claim is cited. The report, and the meeting, where Sam says the date has moved to November third. The sources
panel shows exactly what the answer was based on. And Granite Guardian can check that each answer is backed by its
sources.

## 7 · Extract · 10 s

Forms work too. Give it an invoice and a JSON schema, and you get the fields back, checked against the schema, and ready
to download.

## 8 · End · 5 s

granite.

Open source, and private by design.
