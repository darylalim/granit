"""Demo voiceover (docs/demo-plan.md): docs/demo/voiceover.md → one WAV per shot, read by macOS `say`.

    uv run python scripts/demo_voice.py [--voice "Zoe (Premium)"] [--rate WPM] [--pause MS] [--script FILE] [--out DIR]

Each ``## N · Name · S s`` section of the script becomes ``OUT/voice/shot-N.wav`` (24 kHz, 16-bit mono), with a pause
between its paragraphs. The voice must
be installed: ``say`` exits 0 with a missing voice and quietly reads in the default one, so the script checks
``say -v '?'`` first. Fails when a clip runs longer than its shot (S seconds): shorten the text, or raise ``--rate``.
No models, so it can run alongside any phase.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = "Zoe (Premium)"
SAMPLE_RATE = 24_000
HEADING = re.compile(r"^## (\d+) · (.+?) · (\d+(?:\.\d+)?) s$")
# "Zoe (Premium)       en_US    # Hello! …"; a long name gets one space: "Daniel (English (UK)) en_GB    # …"
VOICE_LINE = re.compile(r"^(.+?)\s+[a-z]{2,3}_[A-Za-z0-9]+\s+#")
SETUP = (
    "Download it once: System Settings → Accessibility → Spoken Content → System Voice (ⓘ) → pick the voice → download"
    " (or search Settings for “Spoken Content”)."
)


@dataclass(frozen=True)
class Section:
    shot: int
    name: str
    slot_s: float
    text: str


@dataclass(frozen=True)
class Fit:
    shot: int
    name: str
    words: int
    clip_s: float
    slot_s: float

    @property
    def fits(self) -> bool:
        return self.clip_s <= self.slot_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument(
        "--rate", type=int, help="words per minute (default: the voice's own; Zoe won't go slower)"
    )
    parser.add_argument(
        "--pause", type=int, default=450, help="silence between paragraphs, ms (default: 450)"
    )
    parser.add_argument("--script", type=Path, default=ROOT / "docs/demo/voiceover.md")
    parser.add_argument("--out", type=Path, default=ROOT / "build/demo")
    args = parser.parse_args(argv)

    voices = installed_voices(
        subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    )
    if args.voice not in voices:
        premium = sorted(v for v in voices if "(Premium)" in v or "(Enhanced)" in v)
        print(f"voice {args.voice!r} isn't installed. {SETUP}", file=sys.stderr)
        if premium:
            print(f"Installed: {', '.join(premium)}", file=sys.stderr)
        return 1

    sections = parse_script(args.script.read_text())
    out = args.out / "voice"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sections:
        text_path = out / f"shot-{s.shot}.txt"
        wav_path = out / f"shot-{s.shot}.wav"
        text_path.write_text(spoken(s.text, args.pause) + "\n")
        command = [
            "say",
            "-v",
            args.voice,
            "--file-format=WAVE",
            f"--data-format=LEI16@{SAMPLE_RATE}",
        ]
        if args.rate:
            command += ["-r", str(args.rate)]
        subprocess.run([*command, "-o", str(wav_path), "-f", str(text_path)], check=True)
        rows.append(fit(s, wav_seconds(wav_path)))
    report = {
        "voice": args.voice,
        "rate": args.rate,
        "pause_ms": args.pause,
        "shots": [asdict(r) | {"fits": r.fits} for r in rows],
    }
    (out / "voice-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        mark = "ok" if r.fits else "TOO LONG"
        print(
            f"shot {r.shot} {r.name:<9} {r.words:>3} words  {r.clip_s:5.1f} s of {r.slot_s:4.0f} s  {mark}"
        )
    print(
        f"{sum(r.words for r in rows)} words, {sum(r.clip_s for r in rows):.0f} s of speech → {out}"
    )
    long = [r for r in rows if not r.fits]
    if long:
        print(
            f"{len(long)} clip(s) longer than their shot: shorten the text or raise --rate",
            file=sys.stderr,
        )
        return 1
    return 0


def installed_voices(listing: str) -> set[str]:
    """Voice names from ``say -v '?'``: the name is everything before the locale column (names can contain spaces)."""
    return {m.group(1).strip() for line in listing.splitlines() if (m := VOICE_LINE.match(line))}


def parse_script(markdown: str) -> list[Section]:
    """``## N · Name · S s`` sections. Their text is the non-note lines (notes start with ``>``): lines of a paragraph are
    joined by spaces, paragraphs (split by blank lines) by a blank line, where ``spoken`` puts a pause."""
    sections: list[Section] = []
    current: tuple[int, str, float] | None = None
    paragraphs: list[list[str]] = []

    def close() -> None:
        if current is not None:
            text = "\n\n".join(" ".join(p) for p in paragraphs if p)
            if not text:
                raise ValueError(f"shot {current[0]} has no text")
            sections.append(Section(*current, text=text))

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            close()
            m = HEADING.match(line)
            if m is None:
                raise ValueError(f"not a `## N · Name · S s` heading: {line!r}")
            current, paragraphs = (int(m.group(1)), m.group(2), float(m.group(3))), [[]]
        elif current is None or line.startswith(">"):
            continue
        elif line:
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    close()
    shots = [s.shot for s in sections]
    if not sections or shots != sorted(set(shots)):
        raise ValueError(f"shots must be numbered in order, once each: {shots}")
    return sections


def spoken(text: str, pause_ms: int) -> str:
    """The text for ``say``: a silence of ``pause_ms`` between paragraphs (Premium voices honour ``[[slnc]]``)."""
    return (
        f" [[slnc {pause_ms:d}]] ".join(text.split("\n\n"))
        if pause_ms
        else " ".join(text.split("\n\n"))
    )


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def fit(section: Section, clip_s: float) -> Fit:
    words = len(section.text.split())
    return Fit(section.shot, section.name, words, round(clip_s, 2), section.slot_s)


if __name__ == "__main__":
    sys.exit(main())
