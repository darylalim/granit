"""Demo video, final cut (docs/demo-plan.md): raw take + captions + cards + music → ``OUT/demo.mp4``.

    uv run --group screenshots python scripts/demo_edit.py [--music FILE] [--out DIR] [--script FILE]

Needs ``demo_video.py record`` (``demo-raw.webm``, ``beats.json``) in ``OUT``. Each section of ``docs/demo/captions.md``
is one shot of its stated length: a shot recorded in ``beats.json`` is cut from the raw take, upscaled to 1920×1080 and
held on its last frame to fill its length (waits are trimmed only when the footage is longer), with its captions at the
bottom right, each from its ``@N`` until the next; a shot that wasn't recorded (title, end) is a card of its caption's
lines. ``--music`` is looped or cut to the video's length, faded in and out, and normalized to ``MUSIC_LUFS`` in the one
pass that encodes the audio; without it the video is silent. The track never goes in the repo: pass a CC0 file from
outside it (docs/demo-plan.md). Captions and cards are HTML rendered by Playwright in the app's dark-theme colors and
font. No models.
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

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1920, 1080, 25
CAPTION_IN_S, CAPTION_FADE_S, CAPTION_END_S = (
    0.6,
    0.4,
    0.5,
)  # a shot's last caption fades out this far before its end
READ_WPS = 3.0  # words a second a caption must stay readable for (a relaxed pace: viewers also watch the footage)
TITLE_FADE_S, END_FADE_S = 0.6, 0.8
MUSIC_FADE_IN_S, MUSIC_FADE_OUT_S = 1.5, 3.0
MUSIC_LUFS = (
    -18
)  # music only, so a little under YouTube's −14 reference: background, not foreground
KEEP_WAIT_S = 1.0  # what's left of a wait (e.g. for an answer) when a shot must be shortened
END_MARGIN_S = 0.5  # dropped from each shot's end: the video runs a little behind beats.json, and the next shot's
# page starts loading (a blank frame, then held to fill the shot) right after
SAMPLE_RATE = 48_000
# .streamlit/config.toml, [theme.dark]
BACKGROUND, TEXT, MUTED, ACCENT = "#1C1917", "#F5F5F4", "#D6D3D1", "#6059F1"
FONT_GLOB = "SourceSansVF-Upright*.woff2"  # Streamlit's bundled font, as in the app
HEADING = re.compile(r"^## (\d+) · (.+?) · (\d+(?:\.\d+)?) s$")
AT = re.compile(r"^@(\d+(?:\.\d+)?)\s+(.+)$")


@dataclass(frozen=True)
class Caption:
    at_s: float  # into its shot
    text: str  # "headline | detail" (a card: its lines)

    @property
    def words(self) -> int:
        return len(self.text.replace("|", " ").split())


@dataclass(frozen=True)
class Section:
    shot: int
    name: str
    slot_s: float
    captions: tuple[Caption, ...]


@dataclass(frozen=True)
class Shot:
    section: Section
    start_s: float  # in the finished video
    footage: dict | None  # its beats.json entry, or None for a card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--music", type=Path, help="audio file to play under the video (CC0; not in the repo)"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "build/demo")
    parser.add_argument("--script", type=Path, default=ROOT / "docs/demo/captions.md")
    args = parser.parse_args(argv)
    out: Path = args.out
    shots, problems = plan_shots(parse_script(args.script.read_text()), out)
    if args.music is not None and not args.music.is_file():
        problems.append(f"no music file {args.music}")
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
    total = sum(s.section.slot_s for s in shots)
    join(segments, work, final, total, args.music)
    music = f", music: {args.music.name}" if args.music else ", silent"
    print(f"{final} ({total // 60:.0f}:{total % 60:04.1f}{music})")
    return 0


# ── the script (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────


def parse_script(markdown: str) -> list[Section]:
    """``## N · Name · S s`` sections, each with its caption lines (``@N`` to start one later; ``>`` lines are notes)."""
    sections: list[Section] = []
    current: tuple[int, str, float] | None = None
    captions: list[Caption] = []

    def close() -> None:
        if current is not None:
            if not captions:
                raise ValueError(f"shot {current[0]} has no caption")
            sections.append(Section(*current, captions=tuple(captions)))

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            close()
            m = HEADING.match(line)
            if m is None:
                raise ValueError(f"not a `## N · Name · S s` heading: {line!r}")
            current, captions = (int(m.group(1)), m.group(2), float(m.group(3))), []
        elif current is None or not line or line.startswith(">"):
            continue
        elif m := AT.match(line):
            captions.append(Caption(float(m.group(1)), m.group(2).strip()))
        else:
            captions.append(Caption(0.0, line))
    close()
    shots = [s.shot for s in sections]
    if not sections or shots != sorted(set(shots)):
        raise ValueError(f"shots must be numbered in order, once each: {shots}")
    return sections


def caption_spans(section: Section) -> list[tuple[float, float]]:
    """When each caption is fully or partly on screen: from its fade-in to the end of its fade-out."""
    spans = []
    for i, caption in enumerate(section.captions):
        start = caption.at_s + (CAPTION_IN_S if caption.at_s == 0 else 0.0)
        later = section.captions[i + 1 :]
        end = later[0].at_s if later else section.slot_s - CAPTION_END_S
        spans.append((round(start, 3), round(end, 3)))
    return spans


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
        footage = beats.get(section.shot)
        problems += caption_problems(section, card=bool(beats) and footage is None)
        shots.append(Shot(section, start, footage))
        start += section.slot_s
    unused = sorted(set(beats) - {s.shot for s in sections})
    if unused:
        problems.append(f"recorded shots with no section in the script: {unused}")
    return shots, problems


def caption_problems(section: Section, *, card: bool) -> list[str]:
    n, captions = section.shot, section.captions
    if card:
        return [] if len(captions) == 1 else [f"shot {n} is a card: give it one caption line"]
    problems = []
    times = [c.at_s for c in captions]
    if times[0] != 0 or times != sorted(set(times)) or times[-1] >= section.slot_s:
        problems.append(
            f"shot {n}: captions must start at 0, then at later @times inside its {section.slot_s:g} s"
        )
        return problems
    for caption, (a, b) in zip(captions, caption_spans(section), strict=True):
        need = caption.words / READ_WPS
        if b - a < need:
            problems.append(
                f"shot {n}: {caption.text.split('|')[0].strip()!r} is on screen {b - a:.1f} s,"
                f" but its {caption.words} words need {need:.1f} s"
            )
    return problems


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
        + f" = {shot.section.slot_s:g} s, {len(shot.section.captions)} caption(s)"
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
    headline, _, detail = (part.strip() for part in caption.partition("|"))
    more = f'<div class="detail">{html.escape(detail)}</div>' if detail else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font}
html, body {{ margin: 0; width: {WIDTH}px; height: {HEIGHT}px; background: transparent; }}
.caption {{ position: absolute; right: 96px; bottom: 64px; max-width: 820px; padding: 22px 32px 24px;
  background: rgba(28, 25, 23, 0.9); border-left: 6px solid {ACCENT}; border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45); color: {TEXT}; font-family: 'Source Sans', -apple-system, sans-serif; }}
.headline {{ font-size: 42px; line-height: 1.2; font-weight: 600; }}
.detail {{ margin-top: 10px; font-size: 30px; line-height: 1.3; font-weight: 400; color: {MUTED}; }}
</style></head><body><div class="caption"><div class="headline">{html.escape(headline)}</div>{more}</div></body></html>"""


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
            n, card = shot.section.shot, shot.footage is None
            for i, caption in enumerate(shot.section.captions):
                page.set_content((card_html if card else caption_html)(caption.text, font))
                page.evaluate("document.fonts.ready")
                path = work / (f"card-{n}.png" if card else f"caption-{n}-{i}.png")
                page.screenshot(path=str(path), omit_background=not card)
        browser.close()


# ── ffmpeg ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def looped_image(path: Path, seconds: float) -> list[str]:
    return ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds}", "-i", str(path)]


def encode_segment(
    shot: Shot, out: Path, work: Path, segment: Path, *, first: bool, last: bool
) -> None:
    """One shot as video only (the music is added once, over the whole video, in ``join``)."""
    slot, n = shot.section.slot_s, shot.section.shot
    inputs: list[str] = []
    graph: list[str] = []
    if shot.footage is None:
        inputs += looped_image(work / f"card-{n}.png", slot)
        fades = []
        if first:
            fades.append(f"fade=t=in:st=0:d={TITLE_FADE_S}")
        if last:
            fades.append(f"fade=t=out:st={slot - END_FADE_S}:d={END_FADE_S}")
        graph.append(
            f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1,{','.join([*fades, 'format=yuv420p'])}[v]"
        )
    else:
        ranges = footage_ranges(shot.footage, slot)
        inputs += ["-i", str(out / "demo-raw.webm")]
        for i, (a, b) in enumerate(ranges):
            graph.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[r{i}]")
        joined = "".join(f"[r{i}]" for i in range(len(ranges)))
        graph.append(
            f"{joined}concat=n={len(ranges)}:v=1:a=0,fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={slot},trim=duration={slot},setpts=PTS-STARTPTS[o0]"
        )
        spans = caption_spans(shot.section)
        for i, (a, b) in enumerate(spans):
            inputs += looped_image(work / f"caption-{n}-{i}.png", slot)
            graph.append(
                f"[{i + 1}:v]format=rgba,fade=t=in:st={a}:d={CAPTION_FADE_S}:alpha=1,"
                f"fade=t=out:st={b - CAPTION_FADE_S}:d={CAPTION_FADE_S}:alpha=1[c{i}]"
            )
            graph.append(f"[o{i}][c{i}]overlay=0:0[o{i + 1}]")
        graph.append(f"[o{len(spans)}]format=yuv420p[v]")
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph)]
    command += ["-map", "[v]", "-t", f"{slot}", "-r", str(FPS)]
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
        str(segment),
    ]
    subprocess.run(command, check=True)


LOUDNORM = f"loudnorm=I={MUSIC_LUFS}:TP=-1.5:LRA=11"


def music_filter(total_s: float, measured: dict[str, str] | None = None) -> str:
    """The music, looped (by ``-stream_loop``) and cut to the video, faded, then normalized.

    Without ``measured`` it's the first pass, which only prints loudnorm's measurement. One pass alone misses quiet,
    slow-building tracks by up to 2 LU (−16.1 for a −18 target, 2026-11-08), so the second applies the measured values as
    one linear gain.
    """
    shaped = (
        f"atrim=duration={total_s},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={MUSIC_FADE_IN_S},afade=t=out:st={total_s - MUSIC_FADE_OUT_S}:d={MUSIC_FADE_OUT_S}"
    )
    if measured is None:
        return f"{shaped},{LOUDNORM}:print_format=json"
    m = measured
    return (
        f"{shaped},{LOUDNORM}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
        f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
        f"aresample={SAMPLE_RATE},aformat=channel_layouts=stereo"
    )


def loudness_json(stderr: str) -> dict[str, str]:
    """loudnorm's measurement: the last ``{…}`` block it prints to stderr."""
    return json.loads(stderr[stderr.rindex("{") : stderr.rindex("}") + 1])


def measure_music(music: Path, total_s: float) -> dict[str, str]:
    command = ["ffmpeg", "-hide_banner", "-nostats", "-stream_loop", "-1", "-i", str(music)]
    command += ["-af", music_filter(total_s), "-t", f"{total_s}", "-f", "null", "-"]
    return loudness_json(subprocess.run(command, capture_output=True, text=True, check=True).stderr)


def join(segments: list[Path], work: Path, final: Path, total_s: float, music: Path | None) -> None:
    """Concatenate (video copied), with the music encoded once to AAC, or no audio track without it."""
    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))
    command = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if music is None:
        command += ["-map", "0:v", "-c:v", "copy"]
    else:
        audio = music_filter(total_s, measure_music(music, total_s))
        command += ["-stream_loop", "-1", "-i", str(music), "-map", "0:v", "-map", "1:a"]
        command += ["-c:v", "copy", "-af", audio, "-c:a", "aac", "-b:a", "192k"]
        command += ["-t", f"{total_s}"]
    command += ["-movflags", "+faststart", str(final)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    sys.exit(main())
