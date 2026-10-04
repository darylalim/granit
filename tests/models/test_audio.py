"""Audio golden tests on the Mac (``uv run pytest -m model``), PLAN.md §3.5 / §4.9. Every test has an explicit threshold."""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from granit.evaluate.asr import normalize, wer
from granit.ingest.audio import SAMPLE_RATE, AudioTranscriber, decode
from tests.conftest import ROOT

pytestmark = pytest.mark.model

FIXTURES = ROOT / "tests" / "fixtures" / "audio"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
VAD_TOLERANCE_S = 0.5
MAX_WER = 0.05  # TTS speech (PLAN.md §4.9)
MAX_WER_LIBRISPEECH = 0.10  # read audiobook speech, harder than TTS
LIBRISPEECH = (
    "hf-internal-testing/librispeech_asr_dummy",
    "5be91486e11a2d616f4ec5db8d3fd248585ac07a",
)


@pytest.fixture(scope="module")
def transcriber() -> Iterator[AudioTranscriber]:
    yield AudioTranscriber().load()


def test_vad_finds_both_phrases_and_the_pauses(transcriber: AudioTranscriber) -> None:
    spans = transcriber.speech_spans(decode(FIXTURES / "vad_pauses.wav"))
    expected = MANIFEST["vad_pauses.wav"]["expected_speech"]
    assert len(spans) == len(expected) == 2
    for span, (start, end) in zip(spans, expected, strict=True):
        assert abs(span.start - start) <= VAD_TOLERANCE_S, (span, start)
        assert abs(span.end - end) <= VAD_TOLERANCE_S, (span, end)


def test_word_timestamps_decode_exactly_like_generate(transcriber: AudioTranscriber) -> None:
    """Our timed CTC decode must equal mlx-audio's own greedy decode (guards the private-API use)."""
    for name in ("formats/sample.wav", "vad_pauses.wav"):
        audio = decode(FIXTURES / name)
        ours = " ".join(w.text for w in transcriber.words(audio))
        assert ours == transcriber._speech.generate(audio).text


def test_word_timestamps_fall_inside_speech(transcriber: AudioTranscriber) -> None:
    transcript = transcriber.transcribe(FIXTURES / "vad_pauses.wav")
    (first_start, first_end), (second_start, second_end) = MANIFEST["vad_pauses.wav"][
        "expected_speech"
    ]
    words = [w for s in transcript.segments for w in s.words]
    friday = next(w for w in words if w.text == "friday")
    marcus = next(w for w in words if w.text == "marcus")
    assert first_start - VAD_TOLERANCE_S <= friday.start <= first_end + VAD_TOLERANCE_S
    assert second_start - VAD_TOLERANCE_S <= marcus.start <= second_end
    assert [s.text.split()[0] for s in transcript.segments] == [
        "please",
        "marcus",
    ]  # split at the pause


def test_scripted_meeting_wer_and_segments(transcriber: AudioTranscriber) -> None:
    transcript = transcriber.transcribe(FIXTURES / "meeting.flac")
    assert wer(MANIFEST["meeting.flac"]["reference"], transcript.text) <= MAX_WER
    # One segment per spoken sentence (they're separated by 0.7 s pauses), in order, inside the file.
    assert len(transcript.segments) == 8
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts) and transcript.segments[-1].end <= transcript.duration_s
    assert 0.6 <= transcript.speech_s / transcript.duration_s <= 0.95


def test_continuous_speech_uses_overlap_without_duplicates(transcriber: AudioTranscriber) -> None:
    transcript = transcriber.transcribe(FIXTURES / "continuous.flac")
    assert transcript.duration_s > 30  # longer than one chunk, with no pause to split at
    assert transcript.chunks >= 2
    assert wer(MANIFEST["continuous.flac"]["reference"], transcript.text) <= MAX_WER
    words = normalize(transcript.text).split()
    assert not [a for a, b in pairwise(words) if a == b], "duplicated word at a seam"


@pytest.mark.parametrize("name", MANIFEST["formats"]["files"])
def test_every_format_transcribes_the_same(transcriber: AudioTranscriber, name: str) -> None:
    transcript = transcriber.transcribe(FIXTURES / "formats" / name)
    assert wer(MANIFEST["formats"]["reference"], transcript.text) <= MAX_WER


@pytest.fixture(scope="module")
def librispeech() -> list[dict[str, Any]]:
    """Optional real-speech check: downloaded at test time (never committed); skipped when offline."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            LIBRISPEECH[0],
            "clean/validation-00000-of-00001.parquet",
            repo_type="dataset",
            revision=LIBRISPEECH[1],
        )
    except Exception as exc:  # offline, HF_HUB_OFFLINE=1, or not cached
        pytest.skip(f"LibriSpeech sample unavailable: {exc}")
    return pq.read_table(path).to_pylist()  # all 73 clips, ~8 min of audio: transcribed in ~5 s


def test_librispeech_wer(
    transcriber: AudioTranscriber, librispeech: list[dict[str, Any]], tmp_path: Path
) -> None:
    references, hypotheses = [], []
    for i, row in enumerate(librispeech):
        clip = tmp_path / f"{i}.flac"
        clip.write_bytes(row["audio"]["bytes"])
        references.append(row["text"])
        hypotheses.append(transcriber.transcribe(clip).text)
    # Pooled WER over the clips (one long reference / hypothesis pair).
    assert wer(" ".join(references), " ".join(hypotheses)) <= MAX_WER_LIBRISPEECH
    assert len(decode(clip)) / SAMPLE_RATE > 1
