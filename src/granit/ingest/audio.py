"""Audio ingest (PLAN.md §3.5): decode → Silero VAD v6 → pause-aware chunks → TurboCTC → timestamped segments.

- **Decode** to 16 kHz mono float32: miniaudio in-process (WAV / FLAC / MP3 / Ogg); M4A / AAC / AIFF / CAF / MP4 go
  through macOS's built-in ``afconvert`` first; ffmpeg only if the user installed it.
- **VAD** finds speech; segments are padded by ~0.3 s and merged into chunks of at most 30 s, **split at the longest
  pauses**. Continuous speech longer than 30 s falls back to fixed windows with a 1 s overlap.
- **Word timestamps** come from the CTC frames (80 ms each): TurboCTC's ``generate()`` returns text only, so we run
  the same greedy decode and keep each token's frame. Overlapping windows are stitched by time, not by text.
- **Segments** (the citation unit) break at pauses and are at most 30 s long.

The pure functions here need no models; ``AudioTranscriber`` imports MLX lazily (PLAN.md §4.2).
"""

from __future__ import annotations

import bisect
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from granit.config import HUB_MODELS

SAMPLE_RATE = 16_000
MAX_CHUNK_S = 30.0  # longest audio sent to TurboCTC at once
PAD_S = 0.3  # padding around each VAD speech segment (generous: CTC outputs blanks on silence)
OVERLAP_S = 1.0  # overlap between fixed windows when speech runs past MAX_CHUNK_S without a pause
MAX_SEGMENT_S = 30.0  # longest transcript segment (one citation)
SEAM_REPEAT_S = 0.4  # a word repeated this close across an overlap seam is a duplicate
WORD_START = "Ġ"  # byte-level BPE marker for a token that starts a new word

MINIAUDIO_FORMATS = frozenset({".wav", ".flac", ".mp3", ".ogg"})
AFCONVERT_FORMATS = frozenset({".m4a", ".aac", ".aiff", ".aif", ".caf", ".mp4"})


class AudioError(ValueError):
    """The file can't be decoded as audio."""


class UnsupportedAudioFormat(AudioError):
    pass


# ── decoding ──


def route(path: Path, ffmpeg: str | None = None) -> str:
    """Which decoder handles ``path``: ``miniaudio``, ``afconvert`` or (only if installed) ``ffmpeg``."""
    ext = path.suffix.lower()
    if ext in MINIAUDIO_FORMATS:
        return "miniaudio"
    if ext in AFCONVERT_FORMATS:
        return "afconvert"
    if ffmpeg:
        return "ffmpeg"
    raise UnsupportedAudioFormat(
        f"{path.name}: unsupported audio format {ext or '(no extension)'}; convert it to WAV or M4A"
    )


def decode(path: str | Path) -> Any:
    """Decode an audio file to a 16 kHz mono float32 NumPy array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    backend = route(path, shutil.which("ffmpeg"))
    if backend == "miniaudio":
        try:
            return _miniaudio(path)
        except AudioError:
            # e.g. a WAV with a codec miniaudio doesn't read (ADPCM, μ-law): macOS usually can.
            return _convert_then_decode(path, "afconvert")
    return _convert_then_decode(path, backend)


def _miniaudio(path: Path) -> Any:
    import miniaudio
    import numpy as np

    try:
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
        )
    except miniaudio.DecodeError as exc:
        raise AudioError(f"{path.name}: could not decode audio ({exc})") from exc
    return np.frombuffer(decoded.samples, dtype=np.float32).copy()


def converter_command(backend: str, source: Path, target: Path) -> list[str]:
    if backend == "afconvert":
        # Ships with macOS: nothing to install, bundle or license (PLAN.md §3.5).
        return [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{SAMPLE_RATE}",
            "-c",
            "1",
            str(source),
            str(target),
        ]
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(target),
    ]  # fmt: skip


def _convert_then_decode(path: Path, backend: str) -> Any:
    with tempfile.TemporaryDirectory(prefix="granit-audio-") as tmp:
        wav = Path(tmp) / "decoded.wav"
        result = subprocess.run(
            converter_command(backend, path, wav), capture_output=True, text=True
        )
        if result.returncode != 0 or not wav.is_file():
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
            raise AudioError(f"{path.name}: {backend} could not convert it ({detail[0]})")
        return _miniaudio(wav)


# ── chunk planning (pure) ──


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Chunk(Span):
    overlaps_previous: bool = False


def pad_and_merge(segments: Sequence[Span], duration: float, pad: float = PAD_S) -> list[Span]:
    """Pad each speech segment, clamp to the file, and merge segments that now touch or overlap."""
    merged: list[Span] = []
    for seg in sorted(segments, key=lambda s: s.start):
        span = Span(max(0.0, seg.start - pad), min(duration, seg.end + pad))
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def plan_chunks(
    spans: Sequence[Span], max_s: float = MAX_CHUNK_S, overlap_s: float = OVERLAP_S
) -> list[Chunk]:
    """Group speech spans into chunks of at most ``max_s``, splitting at the longest pauses first."""
    return _split(list(spans), max_s, overlap_s) if spans else []


def _split(spans: list[Span], max_s: float, overlap_s: float) -> list[Chunk]:
    if spans[-1].end - spans[0].start <= max_s:
        return [Chunk(spans[0].start, spans[-1].end)]
    if len(spans) == 1:
        return fixed_windows(spans[0], max_s, overlap_s)
    gaps = [b.start - a.end for a, b in pairwise(spans)]
    k = max(range(len(gaps)), key=gaps.__getitem__)  # the longest pause
    return _split(spans[: k + 1], max_s, overlap_s) + _split(spans[k + 1 :], max_s, overlap_s)


def fixed_windows(
    span: Span, max_s: float = MAX_CHUNK_S, overlap_s: float = OVERLAP_S
) -> list[Chunk]:
    """Continuous speech with no pause: windows of ``max_s`` that overlap by ``overlap_s``."""
    windows: list[Chunk] = []
    start = span.start
    while True:
        end = min(start + max_s, span.end)
        windows.append(Chunk(start, end, overlaps_previous=bool(windows)))
        if end >= span.end:
            return windows
        start = end - overlap_s


# ── words from CTC frames (pure) ──


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


def ctc_words(
    frame_ids: Sequence[int],
    frame_s: float,
    offset: float,
    blank_id: int,
    piece: Callable[[int], str],
    decode_ids: Callable[[list[int]], str],
) -> list[Word]:
    """Greedy CTC collapse that keeps timing: a token is emitted on the first frame of each non-blank run.

    Tokens are grouped into words at the BPE word-start marker. A word spans from its first token's frame to
    the end of its last token's frame.
    """
    groups: list[list[tuple[int, int]]] = []
    previous: int | None = None
    for frame, token in enumerate(frame_ids):
        if token != previous and token != blank_id:
            if not groups or piece(token).startswith(WORD_START):
                groups.append([(frame, token)])
            else:
                groups[-1].append((frame, token))
        previous = token
    words = []
    for group in groups:
        text = decode_ids([token for _, token in group]).strip()
        if text:
            start = offset + group[0][0] * frame_s
            end = offset + (group[-1][0] + 1) * frame_s
            words.append(Word(text, round(start, 2), round(end, 2)))
    return words


def stitch(
    chunk_words: Sequence[tuple[Chunk, list[Word]]], repeat_window_s: float = SEAM_REPEAT_S
) -> list[Word]:
    """Join chunk transcripts. Where fixed windows overlap, each word is kept once: from the window whose half of
    the overlap contains the word's start (the overlap's midpoint is the boundary).

    CTC timing is least precise at a window's edge, so the same word can land on both sides of the midpoint
    (M2 fixture: "and the the printer"). At each seam, a word that repeats the previous word within
    ``repeat_window_s`` is dropped.
    """
    words: list[Word] = []
    for i, (chunk, chunk_words_) in enumerate(chunk_words):
        low, high = float("-inf"), float("inf")
        if chunk.overlaps_previous and i > 0:
            low = (chunk.start + chunk_words[i - 1][0].end) / 2
        if i + 1 < len(chunk_words) and chunk_words[i + 1][0].overlaps_previous:
            high = (chunk_words[i + 1][0].start + chunk.end) / 2
        kept = [w for w in chunk_words_ if low <= w.start < high]
        if chunk.overlaps_previous and words and kept:
            last, first = words[-1], kept[0]
            if (
                first.text.lower() == last.text.lower()
                and first.start - last.start < repeat_window_s
            ):
                kept = kept[1:]
        words.extend(kept)
    return words


_OCLOCK_HOUR = re.compile(r"(1[0-2]|[1-9])0")


def fix_oclock(words: Sequence[Word]) -> list[Word]:
    """TurboCTC writes "o'clock" as a 0 glued to the hour: "two o'clock" → ``20 clock``, "eleven" → ``110 clock``.

    An hour (1–12) followed by 0, then the word "clock", is split back into the hour and "o'clock". The hour keeps the
    first share of the number's time span, in proportion to its digits.
    """
    fixed: list[Word] = []
    i = 0
    while i < len(words):
        word = words[i]
        hour = _OCLOCK_HOUR.fullmatch(word.text)
        if hour and i + 1 < len(words) and words[i + 1].text.lower().strip(".,?!") == "clock":
            split = round(word.start + (word.end - word.start) * len(hour[1]) / len(word.text), 2)
            fixed += [Word(hour[1], word.start, split), Word("o'clock", split, words[i + 1].end)]
            i += 2
        else:
            fixed.append(word)
            i += 1
    return fixed


_EDGE_PUNCTUATION = ".,?!;:"
_FUZZY_MIN_LETTERS = 5  # shorter terms match only exactly: "same" must never become "Sam"
_FUZZY_RATIO = 0.85


def parse_terms(text: str) -> list[str]:
    """Names and terms typed one per line or comma-separated → a clean list (blank and duplicate entries dropped)."""
    terms: list[str] = []
    for part in re.split(r"[,\n]", text):
        term = " ".join(part.split())
        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    return terms


def _match(run: Sequence[Word], terms: Sequence[str]) -> str | None:
    from difflib import SequenceMatcher

    joined = "".join(w.text.strip(_EDGE_PUNCTUATION) for w in run).lower()
    for term in terms:
        key = "".join(term.split()).lower()
        if joined == key:
            return term
    if len(run) == 1:
        for term in terms:
            key = "".join(term.split()).lower()
            fuzzy = len(key) >= _FUZZY_MIN_LETTERS and key.isalpha()
            if fuzzy and SequenceMatcher(None, joined, key).ratio() >= _FUZZY_RATIO:
                return term
    return None


def apply_vocabulary(words: Sequence[Word], terms: Sequence[str], max_words: int = 3) -> list[Word]:
    """Spell the library's expected names and terms the way the user wrote them.

    A run of up to ``max_words`` words that joins to a term ("north beam" → "Northbeam") becomes the term, and a single
    word close to a term of 5+ letters ("pria" → "Priya", similarity ≥ 0.85) does too. The new word spans the run's
    time; punctuation after the run is kept.
    """
    if not terms:
        return list(words)
    fixed: list[Word] = []
    i = 0
    while i < len(words):
        for n in range(min(max_words, len(words) - i), 0, -1):
            run = words[i : i + n]
            if term := _match(run, terms):
                tail = run[-1].text[len(run[-1].text.rstrip(_EDGE_PUNCTUATION)) :]
                fixed.append(Word(term + tail, run[0].start, run[-1].end))
                i += n
                break
        else:
            fixed.append(words[i])
            i += 1
    return fixed


# ── segments and transcript ──


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default=())


def segments_from_words(
    words: Sequence[Word], spans: Sequence[Span], max_s: float = MAX_SEGMENT_S
) -> list[Segment]:
    """One segment per speech span (they're separated by pauses), split further to stay under ``max_s``."""
    starts = [s.start for s in spans]

    def span_of(word: Word) -> int:
        return max(0, bisect.bisect_right(starts, word.start) - 1)

    segments: list[Segment] = []
    current: list[Word] = []
    for word in words:
        if current and (
            span_of(word) != span_of(current[0]) or word.end - current[0].start > max_s
        ):
            segments.append(_segment(current))
            current = []
        current.append(word)
    if current:
        segments.append(_segment(current))
    return segments


def _segment(words: list[Word]) -> Segment:
    return Segment(words[0].start, words[-1].end, " ".join(w.text for w in words), tuple(words))


@dataclass(frozen=True)
class Transcript:
    duration_s: float
    speech_s: float  # time VAD marked as speech (before padding): spots silent or broken recordings
    chunks: int
    segments: tuple[Segment, ...]
    model: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def silence_s(self) -> float:
        return round(self.duration_s - self.speech_s, 2)

    def with_vocabulary(self, terms: Sequence[str]) -> Transcript:
        """The same transcript with ``apply_vocabulary`` on each segment's words (and its text rebuilt from them)."""
        if not terms:
            return self
        segments = tuple(
            _segment(apply_vocabulary(s.words, terms)) if s.words else s for s in self.segments
        )
        return replace(self, segments=segments)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Transcript:
        segments = tuple(
            Segment(s["start"], s["end"], s["text"], tuple(Word(**w) for w in s["words"]))
            for s in data["segments"]
        )
        return cls(
            data["duration_s"], data["speech_s"], data["chunks"], segments, data.get("model", "")
        )


def timestamp(seconds: float) -> str:
    """``754.2`` → ``12:34`` (or ``1:02:34`` past an hour), as citations show it."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# ── models ──


class AudioTranscriber:
    """Silero VAD v6 + Granite Speech TurboCTC, loaded once (by the Phase A worker) and reused for every file."""

    def __init__(self) -> None:
        self._speech: Any = None
        self._vad: Any = None

    def load(self) -> AudioTranscriber:
        if self._speech is None:
            import mlx_audio.vad as vad
            from mlx_audio.stt.utils import load_model

            from granit.models.download import local_snapshot

            self._speech = load_model(local_snapshot(HUB_MODELS["speech"]))
            self._vad = vad.load(local_snapshot(HUB_MODELS["vad"]))
        return self

    def transcribe(self, path: str | Path) -> Transcript:
        return self.transcribe_audio(decode(path))

    def speech_spans(self, audio: Any) -> list[Span]:
        """Raw VAD speech segments in seconds (no padding)."""
        self.load()
        found = self._vad.get_speech_timestamps(audio, sample_rate=SAMPLE_RATE, return_seconds=True)
        return [Span(float(s["start"]), float(s["end"])) for s in found]

    def transcribe_audio(self, audio: Any) -> Transcript:
        duration = len(audio) / SAMPLE_RATE
        raw = self.speech_spans(audio)
        spans = pad_and_merge(raw, duration)
        chunks = plan_chunks(spans)
        chunk_words = [
            (c, self.words(audio[int(c.start * SAMPLE_RATE) : int(c.end * SAMPLE_RATE)], c.start))
            for c in chunks
        ]
        speech = HUB_MODELS["speech"]
        return Transcript(
            duration_s=round(duration, 2),
            speech_s=round(sum(s.duration for s in raw), 2),
            chunks=len(chunks),
            segments=tuple(segments_from_words(fix_oclock(stitch(chunk_words)), spans)),
            model=f"{speech.repo_id}@{speech.revision}",
        )

    def words(self, audio: Any, offset: float = 0.0) -> list[Word]:
        """TurboCTC on one chunk, with word timestamps (the same greedy decode as ``Model.generate``)."""
        import mlx.core as mx
        from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import compute_features

        self.load()
        model = self._speech
        features = compute_features(
            mx.array(audio), num_mel_bins=model.config.encoder_config.num_mel_bins
        )[None]
        # argmax over a (frames, vocab) array is 1-D, so tolist() is a list of ints
        frame_ids = cast(list[int], mx.argmax(model(features)[0], axis=-1).tolist())
        if not frame_ids:
            return []
        tokenizer = (
            model._tokenizer
        )  # mlx-audio 0.5.7 keeps the tokenizer here; golden test checks the decode
        return ctc_words(
            frame_ids,
            frame_s=len(audio) / SAMPLE_RATE / len(frame_ids),
            offset=offset,
            blank_id=model.config.pad_token_id,
            piece=tokenizer.id_to_token,
            decode_ids=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        )
