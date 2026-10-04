"""Generated inputs for smoke checks and benchmarks: macOS ``say`` audio, rendered pages, synthetic text.

Everything is produced on the fly (no third-party recordings or documents, so no licensing questions)
and is deterministic for a given seed.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path


def say_wav(text: str, directory: Path, name: str = "speech") -> Path:
    """Synthesize ``text`` with macOS ``say`` and convert it to 16 kHz mono WAV with ``afconvert``."""
    aiff, wav = directory / f"{name}.aiff", directory / f"{name}.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True
    )
    return wav


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
