"""Generate the audio test fixtures in tests/fixtures/audio/ (PLAN.md §3.5).

Speech comes from macOS ``say`` (pinned voice) and ``afconvert``, so there are no third-party recordings and no
licensing questions. MP3 and Ogg Vorbis need an encoder macOS doesn't have; they're made with ffmpeg if it's installed
(a dev-time tool only: the committed files are just test data). Re-run after changing a fixture, then commit the output:

    uv run python scripts/make_audio_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from granit.ingest.audio import SAMPLE_RATE, decode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "audio"
VOICE = "Samantha"  # en_US; pinned because the system default voice differs between Macs
SILENCE_THRESHOLD = 1e-3  # |sample| below this counts as silence when trimming `say` output

PHRASE_A = "Please send the revised invoice to Finance before Friday."
PHRASE_B = "Marcus will confirm the delivery date with the warehouse team."

# Scripted meeting with pauses; the reference is the text without `say` commands.
MEETING = [
    "Good morning everyone, let's start with the budget review.",
    "Priya, can you share the numbers from last quarter?",
    "Sure, revenue was up twelve percent and costs stayed flat.",
    "Great. The main open item is the contract with the shipping vendor.",
    "Legal still needs to review the termination clause.",
    "Marcus, please send the draft to Legal by Wednesday.",
    "Elena will update the customer forecast before the next meeting.",
    "Let's meet again on Tuesday at ten to make a final decision.",
]

# Run-on speech with no sentence breaks: longer than 30 s, so the chunker must use overlapping windows.
CONTINUOUS = (
    "and then the team reviewed the budget and the vendor list and the shipping plan and the contract terms "
    "and the travel schedule and the hiring targets and the office move and the security audit and the "
    "customer survey and the product roadmap and the training sessions and the quarterly forecast and the "
    "pricing changes and the support backlog and the marketing calendar and the partner agreements and the "
    "release checklist and the warranty claims and the inventory counts and the safety review and the "
    "onboarding guide and the data retention policy and the expense reports and the cloud migration and the "
    "network upgrade and the printer contracts and the parking permits and the holiday rota"
)

FORMAT_PHRASE = (
    "The quarterly invoice for twelve server racks is due on Friday, and Legal must approve the "
    "contract before the warehouse can ship the order to the customer."
)


def say(text: str, tmp: Path, name: str, rate: int | None = None) -> Path:
    """`say` → AIFF at its native sample rate (decoding then exercises resampling to 16 kHz)."""
    aiff = tmp / f"{name}.aiff"
    cmd = ["say", "-v", VOICE, "-o", str(aiff)]
    cmd += ["-r", str(rate)] if rate else []
    subprocess.run([*cmd, text], check=True)
    return aiff


def trimmed(audio: np.ndarray) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > SILENCE_THRESHOLD)
    return audio[loud[0] : loud[-1] + 1]


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def convert(source: Path, target: Path, file_format: str, data_format: str) -> None:
    subprocess.run(
        ["afconvert", "-f", file_format, "-d", data_format, str(source), str(target)], check=True
    )


def ffmpeg(source: Path, target: Path, *codec: str) -> bool:
    if not shutil.which("ffmpeg"):
        print(f"  skipped {target.name}: ffmpeg not installed", file=sys.stderr)
        return False
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source), *codec, str(target)],
        check=True,
    )
    return True


def vad_pauses(tmp: Path) -> dict[str, Any]:
    """2.0 s silence · phrase · 3.0 s · phrase · 1.5 s, plus light noise. Expected speech is known exactly."""
    a = trimmed(decode(say(PHRASE_A, tmp, "a")))
    b = trimmed(decode(say(PHRASE_B, tmp, "b")))
    gap = lambda s: np.zeros(int(s * SAMPLE_RATE), dtype=np.float32)  # noqa: E731
    audio = np.concatenate([gap(2.0), a, gap(3.0), b, gap(1.5)])
    audio += np.random.default_rng(0).normal(0, 0.002, audio.shape).astype(np.float32)
    write_wav(OUT / "vad_pauses.wav", audio)
    a_end = 2.0 + len(a) / SAMPLE_RATE
    b_start = a_end + 3.0
    return {
        "reference": f"{PHRASE_A} {PHRASE_B}",
        "expected_speech": [
            [2.0, round(a_end, 3)],
            [round(b_start, 3), round(b_start + len(b) / SAMPLE_RATE, 3)],
        ],
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
    }


def meeting(tmp: Path) -> dict[str, Any]:
    text = " [[slnc 700]] ".join(MEETING)
    aiff = say(text, tmp, "meeting")
    convert(aiff, OUT / "meeting.flac", "flac", "flac@16000")
    return {"reference": " ".join(MEETING), "duration_s": round(len(decode(aiff)) / SAMPLE_RATE, 3)}


def continuous(tmp: Path) -> dict[str, Any]:
    aiff = say(CONTINUOUS, tmp, "continuous", rate=165)
    convert(aiff, OUT / "continuous.flac", "flac", "flac@16000")
    return {"reference": CONTINUOUS, "duration_s": round(len(decode(aiff)) / SAMPLE_RATE, 3)}


def formats(tmp: Path) -> dict[str, Any]:
    """The same ~10 s phrase in every supported container, each at `say`'s native rate."""
    aiff = say(FORMAT_PHRASE, tmp, "phrase")
    out = OUT / "formats"
    out.mkdir(exist_ok=True)
    shutil.copy(aiff, out / "sample.aiff")
    convert(aiff, out / "sample.wav", "WAVE", "LEI16")
    convert(aiff, out / "sample.flac", "flac", "flac")
    convert(aiff, out / "sample.m4a", "m4af", "aac")
    convert(aiff, out / "sample.caf", "caff", "aac")
    files = ["sample.aiff", "sample.wav", "sample.flac", "sample.m4a", "sample.caf"]
    if ffmpeg(aiff, out / "sample.mp3", "-codec:a", "libmp3lame", "-q:a", "4"):
        files.append("sample.mp3")
    if ffmpeg(aiff, out / "sample.ogg", "-codec:a", "libvorbis", "-q:a", "4"):
        files.append("sample.ogg")
    return {
        "reference": FORMAT_PHRASE,
        "files": sorted(files),
        "duration_s": round(len(decode(aiff)) / SAMPLE_RATE, 3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="granit-fixtures-") as tmp:
        manifest = {
            "voice": VOICE,
            "sample_rate": SAMPLE_RATE,
            "vad_pauses.wav": vad_pauses(Path(tmp)),
            "meeting.flac": meeting(Path(tmp)),
            "continuous.flac": continuous(Path(tmp)),
            "formats": formats(Path(tmp)),
        }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1000:.0f} KB")


if __name__ == "__main__":
    main()
