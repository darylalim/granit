"""Generated inputs for smoke checks and benchmarks: macOS ``say`` audio, rendered pages, synthetic text.

Everything is produced on the fly (no third-party recordings or documents, so no licensing questions)
and is deterministic for a given seed.
"""

from __future__ import annotations

import random
import subprocess
from collections.abc import Sequence
from pathlib import Path


def say_wav(text: str, directory: Path, name: str = "speech", voice: str | None = None) -> Path:
    """Synthesize ``text`` with macOS ``say`` and convert it to 16 kHz mono WAV with ``afconvert``."""
    aiff, wav = directory / f"{name}.aiff", directory / f"{name}.wav"
    subprocess.run(["say", *(["-v", voice] if voice else []), "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True
    )
    return wav


# Two voices the diarization spike told apart in every meeting (PLAN.md §3.5); Karen merged with Samantha.
DIALOGUE = (
    (
        "Daniel",
        "Thanks for coming. The budget review is due on Friday, so we need the numbers this week.",
    ),
    ("Samantha", "I can pull the sales figures together by Wednesday afternoon."),
    ("Daniel", "Good. Then I will check the travel costs and send the summary to finance."),
    ("Samantha", "Sounds good. I will also ask the regional teams for their forecasts."),
)


def dialogue_wav(
    lines: Sequence[tuple[str, str]], directory: Path, name: str = "dialogue", pause_s: float = 0.7
) -> tuple[Path, list[tuple[float, float, str]]]:
    """``say`` each (voice, text) line, trimmed and ``pause_s`` apart, into one 16 kHz WAV; returns each line's (start, end, voice)."""
    import wave

    import numpy as np

    rate = 16_000
    pause = np.zeros(int(pause_s * rate), dtype=np.int16)
    parts, turns, t = [pause], [], pause_s
    for i, (voice, text) in enumerate(lines):
        with wave.open(str(say_wav(text, directory, f"{name}-{i}", voice)), "rb") as f:
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        loud = np.flatnonzero(np.abs(audio) > 32)
        audio = audio[loud[0] : loud[-1] + 1] if loud.size else audio
        turns.append((round(t, 2), round(t + len(audio) / rate, 2), voice))
        parts += [audio, pause]
        t += len(audio) / rate + pause_s
    path = directory / f"{name}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.concatenate(parts).tobytes())
    return path, turns


def text_page_png(lines: tuple[str, ...], path: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(lines):
        draw.text((100, 120 + i * 90), line, fill="black", font=font)
    image.save(path)
    return path


_PEOPLE = ("Priya", "Marcus", "Elena", "Tomás", "Aiko", "Sam", "Fatima", "Jonas")
_TEAMS = ("Finance", "Procurement", "Legal", "Operations", "Sales", "Engineering")
_ITEMS = ("steel brackets", "server racks", "cable trays", "hydraulic pumps", "sensor kits")
_VERBS = ("review", "approve", "send", "revise", "confirm", "reconcile")
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def sentence(rng: random.Random) -> str:
    """One plausible business sentence mixing names, IDs, amounts and dates (like real ingested text)."""
    invoice = f"INV-2026-{rng.randint(1, 9999):04d}"
    amount = f"${rng.randint(100, 99_999):,}"
    templates = (
        f"{rng.choice(_PEOPLE)} asked {rng.choice(_TEAMS)} to {rng.choice(_VERBS)} invoice {invoice} "
        f"for {amount} by {rng.choice(_DAYS)}.",
        f"The order of {rng.randint(2, 500)} {rng.choice(_ITEMS)} under {invoice} totals {amount}, "
        f"including a {rng.randint(2, 15)}% discount.",
        f"{rng.choice(_TEAMS)} will {rng.choice(_VERBS)} the contract terms with {rng.choice(_PEOPLE)} "
        f"before the {rng.choice(_DAYS)} meeting.",
        f"Action item: {rng.choice(_PEOPLE)} owns the follow-up on {invoice} and reports back on "
        f"{rng.choice(_DAYS)}.",
    )
    return rng.choice(templates)


def synthetic_text(n_chars: int, seed: int = 0) -> str:
    """Deterministic English business prose of roughly ``n_chars`` characters (~4 chars per token)."""
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    while size < n_chars:
        paragraph = " ".join(sentence(rng) for _ in range(rng.randint(3, 6)))
        parts.append(paragraph)
        size += len(paragraph) + 2
    return "\n\n".join(parts)


def synthetic_chunks(n: int, chars: int = 1200, seed: int = 0) -> list[str]:
    """``n`` distinct chunks of about ``chars`` characters (~300 tokens, the plan's chunk size)."""
    return [synthetic_text(chars, seed=seed * 100_003 + i)[:chars] for i in range(n)]
