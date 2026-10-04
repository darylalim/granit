"""Audio ingest without models (PLAN.md §3.5): format routing, decoding, chunk planning, CTC words, stitching.

Decoding uses only miniaudio and macOS ``afconvert``, so it's tested here (and in CI) rather than as a golden test.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from granit.ingest import audio
from granit.ingest.audio import Chunk, Segment, Span, Transcript, Word
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "audio"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="afconvert ships with macOS")


# ── format routing ──


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("a.wav", "miniaudio"),
        ("a.FLAC", "miniaudio"),
        ("a.mp3", "miniaudio"),
        ("a.ogg", "miniaudio"),
        ("a.m4a", "afconvert"),
        ("a.aac", "afconvert"),
        ("a.aiff", "afconvert"),
        ("a.aif", "afconvert"),
        ("a.caf", "afconvert"),
        ("a.mp4", "afconvert"),
    ],
)
def test_route(name: str, backend: str) -> None:
    assert audio.route(Path(name)) == backend


def test_unknown_formats_need_ffmpeg() -> None:
    with pytest.raises(audio.UnsupportedAudioFormat, match="convert it to WAV or M4A"):
        audio.route(Path("talk.wma"))
    assert audio.route(Path("talk.wma"), ffmpeg="/opt/homebrew/bin/ffmpeg") == "ffmpeg"


def test_converter_commands_produce_16k_mono_wav(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.m4a", tmp_path / "out.wav"
    afconvert = audio.converter_command("afconvert", src, dst)
    assert afconvert[:2] == ["afconvert", "-f"] and "LEI16@16000" in afconvert and "-c" in afconvert
    ffmpeg = audio.converter_command("ffmpeg", src, dst)
    assert ffmpeg[0] == "ffmpeg" and ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "1"


# ── decoding (real fixtures; no models) ──


@macos_only
@pytest.mark.parametrize("name", MANIFEST["formats"]["files"])
def test_every_format_decodes_to_the_same_16k_mono_length(name: str) -> None:
    """PLAN.md §3.5: one file per supported format → identical 16 kHz mono output length (±1 %)."""
    reference = audio.decode(FIXTURES / "formats" / "sample.wav")
    decoded = audio.decode(FIXTURES / "formats" / name)
    assert decoded.dtype.name == "float32" and decoded.ndim == 1
    assert abs(len(decoded) / len(reference) - 1) <= 0.01
    assert abs(len(reference) / audio.SAMPLE_RATE - MANIFEST["formats"]["duration_s"]) < 0.05


def test_decode_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such file"):
        audio.decode(tmp_path / "missing.wav")
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all" * 100)
    with pytest.raises(audio.AudioError, match=r"junk\.mp3"):
        audio.decode(junk)


@macos_only
def test_wav_that_miniaudio_cant_read_falls_back_to_afconvert(tmp_path: Path) -> None:
    """A μ-law WAV: miniaudio rejects the codec, macOS decodes it."""
    import subprocess

    ulaw = tmp_path / "ulaw.wav"
    src = FIXTURES / "formats" / "sample.wav"
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "ulaw", str(src), str(ulaw)], check=True)
    decoded = audio.decode(ulaw)
    assert abs(len(decoded) / len(audio.decode(src)) - 1) <= 0.01


@pytest.mark.skipif(shutil.which("ffmpeg") is not None, reason="only meaningful without ffmpeg")
def test_unsupported_format_without_ffmpeg(tmp_path: Path) -> None:
    wma = tmp_path / "a.wma"
    wma.write_bytes(b"x")
    with pytest.raises(audio.UnsupportedAudioFormat):
        audio.decode(wma)


# ── chunk planning ──


def test_pad_and_merge_clamps_to_the_file_and_merges_touching_spans() -> None:
    spans = audio.pad_and_merge(
        [Span(0.1, 1.0), Span(1.4, 2.0), Span(5.0, 9.9)], duration=10.0, pad=0.3
    )
    assert spans == [Span(0.0, 2.3), Span(4.7, 10.0)]
    assert audio.pad_and_merge([], duration=5.0) == []


def test_short_audio_is_one_chunk() -> None:
    spans = [Span(0.0, 4.0), Span(6.0, 12.0)]
    assert audio.plan_chunks(spans) == [Chunk(0.0, 12.0)]


def test_chunks_split_at_the_longest_pause() -> None:
    # 52 s of speech: pauses of 0.5 s, 4 s (the longest) and 1 s.
    spans = [Span(0, 10), Span(10.5, 20), Span(24, 40), Span(41, 52)]
    assert audio.plan_chunks(spans, max_s=30) == [Chunk(0, 20), Chunk(24, 52)]


def test_every_chunk_respects_the_cap() -> None:
    spans = [Span(i * 7.0, i * 7.0 + 6.0) for i in range(20)]  # 139 s, 1 s pauses
    chunks = audio.plan_chunks(spans, max_s=30)
    assert all(c.duration <= 30 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == spans[-1].end
    assert not any(c.overlaps_previous for c in chunks)  # pauses were available everywhere


def test_continuous_speech_falls_back_to_overlapping_windows() -> None:
    chunks = audio.plan_chunks([Span(0, 70)], max_s=30, overlap_s=1)
    assert chunks == [Chunk(0, 30), Chunk(29, 59, True), Chunk(58, 70, True)]


def test_fixed_windows_end_exactly_at_the_span() -> None:
    assert audio.fixed_windows(Span(5, 35.5), max_s=30, overlap_s=1) == [
        Chunk(5, 35),
        Chunk(34, 35.5, True),
    ]


# ── words from CTC frames ──

PIECES = {1: "pri", 2: "ya", 3: "Ġasked", 4: "Ġ", 5: "4", 6: "2", 7: "Ġby"}


def decode_pieces(ids: list[int]) -> str:
    return "".join(PIECES[i] for i in ids).replace("Ġ", " ")


def words(frames: list[int], offset: float = 0.0) -> list[Word]:
    return audio.ctc_words(
        frames, 0.08, offset, blank_id=0, piece=PIECES.__getitem__, decode_ids=decode_pieces
    )


def test_ctc_words_collapse_and_group_at_word_starts() -> None:
    frames = [1, 1, 0, 2, 0, 0, 3, 3, 3, 0, 4, 5, 0, 6, 7]
    assert words(frames) == [
        Word("priya", 0.0, 0.32),  # "pri" + "ya": no word-start marker on "ya"
        Word("asked", 0.48, 0.56),
        Word("42", 0.8, 1.12),  # bare "Ġ" then digits form one word
        Word("by", 1.12, 1.2),
    ]


def test_ctc_repeated_token_after_blank_is_emitted_twice() -> None:
    assert [w.text for w in words([3, 0, 3])] == ["asked", "asked"]
    assert [w.text for w in words([3, 3, 3])] == ["asked"]


def test_ctc_words_are_offset_by_the_chunk_start() -> None:
    assert words([3], offset=29.0) == [Word("asked", 29.0, 29.08)]
    assert words([0, 0, 0]) == []


# ── stitching overlapping windows ──


def test_stitch_keeps_each_overlap_word_once_by_time() -> None:
    first = Chunk(0, 30)
    second = Chunk(29, 40, overlaps_previous=True)  # midpoint 29.5
    a = [Word("one", 28.0, 28.4), Word("two", 29.2, 29.4), Word("three", 29.7, 29.9)]
    b = [Word("two", 29.25, 29.45), Word("three", 29.72, 29.9), Word("four", 31.0, 31.3)]
    assert [w.text for w in audio.stitch([(first, a), (second, b)])] == [
        "one",
        "two",
        "three",
        "four",
    ]


def test_stitch_drops_a_word_repeated_across_the_seam() -> None:
    """Edge timing put the same word on both sides of the midpoint (seen on the continuous fixture)."""
    first, second = Chunk(0, 30), Chunk(29, 40, overlaps_previous=True)
    a = [Word("and", 29.1, 29.3), Word("the", 29.4, 29.48)]
    b = [Word("the", 29.55, 29.65), Word("printer", 29.7, 30.1)]
    assert [w.text for w in audio.stitch([(first, a), (second, b)])] == ["and", "the", "printer"]


def test_stitch_keeps_genuine_repeats_away_from_seams() -> None:
    chunk = Chunk(0, 10)
    a = [Word("very", 1.0, 1.2), Word("very", 1.3, 1.5)]
    assert [w.text for w in audio.stitch([(chunk, a)])] == ["very", "very"]


# ── segments ──


def test_segments_break_at_pauses() -> None:
    spans = [Span(0, 5), Span(8, 12)]
    ws = [Word("hello", 0.3, 0.6), Word("there", 0.7, 1.0), Word("next", 8.4, 8.8)]
    segs = audio.segments_from_words(ws, spans)
    assert [(s.start, s.end, s.text) for s in segs] == [
        (0.3, 1.0, "hello there"),
        (8.4, 8.8, "next"),
    ]


def test_long_segments_split_under_the_cap() -> None:
    ws = [Word(f"w{i}", i * 1.0, i * 1.0 + 0.5) for i in range(70)]
    segs = audio.segments_from_words(ws, [Span(0, 70)], max_s=30)
    assert all(s.end - s.start <= 30 for s in segs)
    assert " ".join(s.text for s in segs) == " ".join(w.text for w in ws)


def test_transcript_json_round_trip() -> None:
    seg = Segment(1.0, 2.0, "hi there", (Word("hi", 1.0, 1.3), Word("there", 1.4, 2.0)))
    transcript = Transcript(10.0, 2.5, 1, (seg,), "repo@rev")
    data = json.loads(json.dumps(transcript.to_json()))
    assert Transcript.from_json(data) == transcript
    assert transcript.text == "hi there"
    assert transcript.silence_s == 7.5


@pytest.mark.parametrize(
    ("seconds", "text"), [(0, "0:00"), (59.9, "0:59"), (754.2, "12:34"), (3754, "1:02:34")]
)
def test_timestamp(seconds: float, text: str) -> None:
    assert audio.timestamp(seconds) == text
