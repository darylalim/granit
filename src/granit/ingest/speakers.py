"""Speakers (PLAN.md §3.7): Nemotron 3 Diarization → a speaker for every transcript word, numbered by first appearance.

The model gives per-frame activity probabilities for up to 8 speaker channels. Each word takes the channel most active
within ``PAD_S`` of it; a word with no one active there takes the nearest active channel within ``NEAREST_S``, else the
previous word's. Channels are renumbered 1, 2, … in the order they first speak. Turns (consecutive words of one speaker)
are what the Library shows and what summaries read once speakers are named.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from granit.config import LOCAL_MODELS
from granit.ingest.audio import SAMPLE_RATE, Segment, Transcript, Word

ACTIVE = (
    0.5  # a channel is speaking when its probability is above this (the model card's threshold)
)
PAD_S = 0.1  # CTC word times are tight: look this far around each word
NEAREST_S = 1.0  # no one active at a word: the nearest channel active within this distance
MAX_TURN_S = 60.0  # a long monologue is split so each line stays readable and cites a nearby time


# ── words → speakers (pure; ``probs`` is frames × channels) ──


def _channel(probs: Any, start: float, end: float, frame_s: float) -> int | None:
    a, b = max(int(start / frame_s), 0), int(end / frame_s) + 1
    window = probs[a:b]
    if not len(window) or not (window > ACTIVE).any():
        return None
    return int(window.sum(axis=0).argmax())


def word_channels(words: Sequence[Word], probs: Any, frame_s: float) -> list[int | None]:
    """The model's channel for each word (None only before anyone has spoken)."""
    channels: list[int | None] = []
    for w in words:
        channel = _channel(probs, w.start - PAD_S, w.end + PAD_S, frame_s)
        if channel is None:
            channel = _channel(probs, w.start - NEAREST_S, w.end + NEAREST_S, frame_s)
        if channel is None and channels:
            channel = channels[-1]
        channels.append(channel)
    return channels


def number_speakers(channels: Sequence[int | None]) -> list[int | None]:
    """Channel ids → speakers 1, 2, … in order of first appearance."""
    order: dict[int, int] = {}
    for c in channels:
        if c is not None:
            order.setdefault(c, len(order) + 1)
    return [None if c is None else order[c] for c in channels]


def with_speakers(transcript: Transcript, speakers: Sequence[int | None]) -> Transcript:
    """The transcript with ``speakers`` (one per word, in order) set on its words."""
    it = iter(speakers)
    segments = tuple(
        replace(s, words=tuple(replace(w, speaker=next(it)) for w in s.words))
        for s in transcript.segments
    )
    return replace(transcript, segments=segments)


def speaker_count(transcript: Transcript) -> int:
    return len({w.speaker for s in transcript.segments for w in s.words if w.speaker is not None})


def has_speakers(transcript: Transcript) -> bool:
    return any(w.speaker is not None for s in transcript.segments for w in s.words)


# ── turns ──


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int | None
    text: str


def turns(segments: Sequence[Segment], max_s: float = MAX_TURN_S) -> list[Turn]:
    """Consecutive words of one speaker, split before ``max_s``. Segments without words are kept whole."""
    out: list[Turn] = []
    words: list[Word] = []

    def flush() -> None:
        if words:
            text = " ".join(w.text for w in words)
            out.append(Turn(words[0].start, words[-1].end, words[0].speaker, text))
            words.clear()

    for segment in segments:
        if not segment.words:
            flush()
            out.append(Turn(segment.start, segment.end, None, segment.text))
            continue
        for w in segment.words:
            if words and (w.speaker != words[0].speaker or w.end - words[0].start > max_s):
                flush()
            words.append(w)
    flush()
    return out


def speaker_label(speaker: int | None, names: Mapping[int, str]) -> str:
    """What a turn shows: the user's name for the speaker, else ``Speaker N`` (``""`` for no speaker)."""
    if speaker is None:
        return ""
    return names.get(speaker) or f"Speaker {speaker}"


# ── model ──


class Diarizer:
    """Nemotron 3 Diarization on mlx-audio (fp32 local build), loaded once by the Phase A worker."""

    def __init__(self) -> None:
        self._model: Any = None

    @staticmethod
    def available() -> bool:
        return LOCAL_MODELS["diarization"].is_built()

    def load(self) -> Diarizer:
        if self._model is None:
            from mlx_audio.vad import load

            self._model = load(str(LOCAL_MODELS["diarization"].path))
            self._model.set_streaming_config("offline")
        return self

    def probabilities(self, audio: Any) -> tuple[Any, float]:
        """(frames × channels activity, seconds per frame) for 16 kHz mono ``audio``."""
        import mlx.core as mx
        import numpy as np

        self.load()
        result = self._model.generate(mx.array(audio), sample_rate=SAMPLE_RATE, threshold=ACTIVE)
        probs = np.array(result.speaker_probs)
        return probs, (len(audio) / SAMPLE_RATE / len(probs) if len(probs) else 0.01)

    def label(self, transcript: Transcript, audio: Any) -> Transcript:
        probs, frame_s = self.probabilities(audio)
        words = [w for s in transcript.segments for w in s.words]
        return with_speakers(transcript, number_speakers(word_channels(words, probs, frame_s)))
