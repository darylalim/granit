"""Demo video, final cut (docs/demo-plan.md): raw take + voiceover + captions + cards → ``OUT/demo.mp4`` + ``demo.srt``.

    uv run --group screenshots python scripts/demo_edit.py [--out DIR] [--script FILE]

Needs ``demo_video.py record`` (``demo-raw.webm``, ``beats.json``) and ``demo_voice.py`` (``voice/shot-N.wav``) in
``OUT``. Each section of ``docs/demo/voiceover.md`` is one shot of its stated length: a shot recorded in ``beats.json`` is
cut from the raw take, upscaled to 1920×1080 and held on its last frame to fill its length (waits are trimmed only when
the footage is longer), with its ``> Caption:`` at the bottom right; a shot that wasn't recorded (title, end) is a card
of its caption's lines. Its voice clip starts ``LEAD_S`` in. Segments keep PCM audio, so the AAC is encoded once, in the
final pass, after loudness normalization to −16 LUFS. ``demo.srt`` holds the narration as subtitles, with the spellings
for the ear turned back ("granite" → granit). Captions and cards are HTML rendered by Playwright in the app's dark-theme
colors and font. No models.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from demo_voice import Section, parse_script, wav_seconds

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1920, 1080, 25
LEAD_S = 0.5  # the voice starts this far into its shot
CAPTION_IN_S, CAPTION_FADE_S = 0.6, 0.4
TITLE_FADE_S, END_FADE_S = 0.6, 0.8
KEEP_WAIT_S = 1.0  # what's left of a wait (e.g. for an answer) when a shot must be shortened
END_MARGIN_S = 0.5  # dropped from each shot's end: the video runs a little behind beats.json, and the next shot's
# page starts loading (a blank frame, then held to fill the shot) right after
SAMPLE_RATE = 48_000
# .streamlit/config.toml, [theme.dark]
BACKGROUND, TEXT, MUTED, ACCENT = "#1C1917", "#F5F5F4", "#D6D3D1", "#6059F1"
FONT_GLOB = "SourceSansVF-Upright*.woff2"  # Streamlit's bundled font, as in the app
WRITTEN = [
    (re.compile(r"\bgranite\b"), "granit"),
    (re.compile(r"\bM L X\b"), "MLX"),
]  # spoken → written
SRT_MAX_CHARS = 84  # two lines of ~42


@dataclass(frozen=True)
class Shot:
    section: Section
    start_s: float  # in the finished video
    clip_s: float  # voice clip length
    footage: dict | None  # its beats.json entry, or None for a card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build/demo")
    parser.add_argument("--script", type=Path, default=ROOT / "docs/demo/voiceover.md")
    args = parser.parse_args(argv)
    out: Path = args.out
    shots, problems = plan_shots(parse_script(args.script.read_text()), out)
    if problems:
        print("can't edit yet:\n" + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 1

    work = out / "edit"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    render_graphics(shots, work)
    segments = []
    for shot in shots:
        segment = work / f"seg-{shot.section.shot}.mkv"
        encode_segment(shot, out, work, segment, first=shot is shots[0], last=shot is shots[-1])
        segments.append(segment)
        print(f"shot {shot.section.shot} {shot.section.name:<9} {describe(shot)}", flush=True)
    final = out / "demo.mp4"
    join(segments, work, final)
    (out / "demo.srt").write_text(srt(subtitles(shots)))
    total = sum(s.section.slot_s for s in shots)
    print(f"{final} ({total // 60:.0f}:{total % 60:04.1f}), {out / 'demo.srt'}")
    return 0


# ── planning (pure) ───────────────────────────────────────────────────────────────────────────────────────────────────


def plan_shots(sections: list[Section], out: Path) -> tuple[list[Shot], list[str]]:
    beats_path = out / "beats.json"
    problems = []
    if not (out / "demo-raw.webm").exists() or not beats_path.exists():
        problems.append(f"no raw take in {out}: run `demo_video.py record`")
        beats: dict[int, dict] = {}
    else:
        beats = {s["id"]: s for s in json.loads(beats_path.read_text())["shots"]}
    shots, start = [], 0.0
    for section in sections:
        clip = out / "voice" / f"shot-{section.shot}.wav"
        if not clip.exists():
            problems.append(f"no {clip.relative_to(out)}: run `demo_voice.py`")
            clip_s = 0.0
        else:
            clip_s = wav_seconds(clip)
            if LEAD_S + clip_s > section.slot_s:
                problems.append(
                    f"shot {section.shot}: {clip_s:.1f} s of voice from {LEAD_S} s in runs past its {section.slot_s:g} s"
                )
        footage = beats.get(section.shot)
        if beats and footage is None and not section.caption:
            problems.append(
                f"shot {section.shot} has no footage, so it's a card: give it a `> Caption:`"
            )
        shots.append(Shot(section, start, clip_s, footage))
        start += section.slot_s
    unused = sorted(set(beats) - {s.shot for s in sections})
    if unused:
        problems.append(f"recorded shots with no section in the voiceover: {unused}")
    return shots, problems


def keep_ranges(
    start: float, end: float, waits: list[list[float]], slot: float
) -> list[tuple[float, float]]:
    """The parts of [start, end] to keep so the footage fits ``slot``: shorter footage is kept whole (and held later);
    longer footage loses the middle of its waits first (each keeps ``KEEP_WAIT_S``), then its tail."""
    excess = (end - start) - slot
    if excess <= 0:
        return [(start, end)]
    cuts = []
    for a, b in sorted((max(a, start), min(b, end)) for a, b in waits):
        take = min(max(0.0, (b - a) - KEEP_WAIT_S), excess)
        if take > 0:
            middle = (a + b) / 2
            cuts.append((middle - take / 2, middle + take / 2))
            excess -= take
        if excess <= 0:
            break
    ranges, cursor = [], start
    for a, b in cuts:
        ranges.append((cursor, a))
        cursor = b
    ranges.append((cursor, end - max(0.0, excess)))
    return [(round(a, 3), round(b, 3)) for a, b in ranges if b - a > 0.001]


def footage_ranges(footage: dict, slot: float) -> list[tuple[float, float]]:
    return keep_ranges(footage["start"], footage["end"] - END_MARGIN_S, footage["waits"], slot)


def written(text: str) -> str:
    for pattern, replacement in WRITTEN:
        text = pattern.sub(replacement, text)
    return text


def subtitles(shots: list[Shot]) -> list[tuple[float, float, str]]:
    """Narration cues: each shot's sentences, grouped up to two lines, spread over its clip by length."""
    cues = []
    for shot in shots:
        sentences = re.split(
            r"(?<=[.?!])\s+", written(shot.section.text.replace("\n\n", " ")).strip()
        )
        groups: list[str] = []
        for sentence in sentences:
            if groups and len(groups[-1]) + 1 + len(sentence) <= SRT_MAX_CHARS:
                groups[-1] += " " + sentence
            else:
                groups.append(sentence)
        total = sum(len(g) for g in groups)
        t = shot.start_s + LEAD_S
        for g in groups:
            length = shot.clip_s * len(g) / total
            cues.append((round(t, 3), round(t + length, 3), g))
            t += length
    return cues


def srt(cues: list[tuple[float, float, str]]) -> str:
    def stamp(t: float) -> str:
        ms = round(t * 1000)
        return (
            f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        )

    return "".join(
        f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1)
    )


def describe(shot: Shot) -> str:
    if shot.footage is None:
        return f"card, {shot.section.slot_s:g} s"
    f = shot.footage
    ranges = footage_ranges(f, shot.section.slot_s)
    kept = sum(b - a for a, b in ranges)
    held = shot.section.slot_s - kept
    return (
        f"{kept:.1f} s of footage"
        + (f" + {held:.1f} s held" if held > 0.05 else "")
        + f" = {shot.section.slot_s:g} s"
    )


# ── graphics ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def font_face() -> str:
    import streamlit

    fonts = sorted((Path(streamlit.__file__).parent / "static").rglob(FONT_GLOB))
    if not fonts:
        return ""
    data = base64.b64encode(fonts[0].read_bytes()).decode()
    return f"@font-face {{ font-family: 'Source Sans'; src: url(data:font/woff2;base64,{data}) format('woff2'); }}"


def caption_html(caption: str, font: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font}
html, body {{ margin: 0; width: {WIDTH}px; height: {HEIGHT}px; background: transparent; }}
.caption {{ position: absolute; right: 96px; bottom: 64px; max-width: 760px; padding: 20px 32px;
  background: rgba(28, 25, 23, 0.9); border-left: 6px solid {ACCENT}; border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45); color: {TEXT};
  font: 600 42px/1.25 'Source Sans', -apple-system, sans-serif; }}
</style></head><body><div class="caption">{html.escape(caption)}</div></body></html>"""


def card_html(caption: str, font: str) -> str:
    lines = "".join(f"<p>{html.escape(line.strip())}</p>" for line in caption.split("|"))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font}
html, body {{ margin: 0; width: {WIDTH}px; height: {HEIGHT}px; background: {BACKGROUND}; }}
body {{ display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-family: 'Source Sans', -apple-system, sans-serif; color: {TEXT}; }}
h1 {{ margin: 0; font-size: 168px; font-weight: 700; letter-spacing: -2px; }}
.bar {{ width: 120px; height: 8px; margin: 28px 0 40px; border-radius: 4px; background: {ACCENT}; }}
p {{ margin: 6px 0; font-size: 46px; font-weight: 500; color: {MUTED}; }}
</style></head><body><h1>granit</h1><div class="bar"></div>{lines}</body></html>"""


def render_graphics(shots: list[Shot], work: Path) -> None:
    from playwright.sync_api import sync_playwright

    font = font_face()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        for shot in shots:
            card = shot.footage is None
            if not shot.section.caption:
                continue
            page.set_content((card_html if card else caption_html)(shot.section.caption, font))
            page.evaluate("document.fonts.ready")
            name = "card" if card else "caption"
            page.screenshot(
                path=str(work / f"{name}-{shot.section.shot}.png"), omit_background=not card
            )
        browser.close()


# ── ffmpeg ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def encode_segment(
    shot: Shot, out: Path, work: Path, segment: Path, *, first: bool, last: bool
) -> None:
    slot, n = shot.section.slot_s, shot.section.shot
    inputs: list[str] = []
    graph: list[str] = []
    if shot.footage is None:
        inputs += [
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-t",
            f"{slot}",
            "-i",
            str(work / f"card-{n}.png"),
        ]
        fades = []
        if first:
            fades.append(f"fade=t=in:st=0:d={TITLE_FADE_S}")
        if last:
            fades.append(f"fade=t=out:st={slot - END_FADE_S}:d={END_FADE_S}")
        graph.append(
            f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1,{','.join([*fades, 'format=yuv420p'])}[v]"
        )
        audio_index = 1
    else:
        f = shot.footage
        ranges = footage_ranges(f, slot)
        inputs += ["-i", str(out / "demo-raw.webm")]
        for i, (a, b) in enumerate(ranges):
            graph.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[r{i}]")
        joined = "".join(f"[r{i}]" for i in range(len(ranges)))
        graph.append(
            f"{joined}concat=n={len(ranges)}:v=1:a=0,fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={slot},trim=duration={slot},setpts=PTS-STARTPTS[base]"
        )
        caption = work / f"caption-{n}.png"
        if caption.exists():
            inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{slot}", "-i", str(caption)]
            out_at = slot - CAPTION_FADE_S - 0.5
            graph.append(
                f"[1:v]format=rgba,fade=t=in:st={CAPTION_IN_S}:d={CAPTION_FADE_S}:alpha=1,"
                f"fade=t=out:st={out_at}:d={CAPTION_FADE_S}:alpha=1[cap]"
            )
            graph.append("[base][cap]overlay=0:0,format=yuv420p[v]")
            audio_index = 2
        else:
            graph.append("[base]format=yuv420p[v]")
            audio_index = 1
    inputs += ["-i", str(out / "voice" / f"shot-{n}.wav")]
    delay_ms = round(LEAD_S * 1000)
    graph.append(
        f"[{audio_index}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"afade=t=in:d=0.02,adelay={delay_ms}:all=1,apad=whole_dur={slot},atrim=duration={slot}[a]"
    )
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph)]
    command += ["-map", "[v]", "-map", "[a]", "-t", f"{slot}", "-r", str(FPS)]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
    ]
    command += ["-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), str(segment)]
    subprocess.run(command, check=True)


def join(segments: list[Path], work: Path, final: Path) -> None:
    """Concatenate (video copied) and encode the audio once: loudness to −16 LUFS, then AAC."""
    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))
    command = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    command += ["-c:v", "copy", "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={SAMPLE_RATE}"]
    command += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    sys.exit(main())
