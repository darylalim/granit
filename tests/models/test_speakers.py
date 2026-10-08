"""Speakers with the real Nemotron 3 Diarization build (PLAN.md §3.7), on a generated two-voice dialogue.

Daniel and Samantha are the `say` voices the spike told apart in every meeting (§3.5). The test pins what ingest relies on
from mlx-audio's new `nemotron_diarization` module (0.5.7): two speakers, turn edges near the true line times, and words
labelled by the pipeline's own functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from granit.config import LOCAL_MODELS
from granit.fixtures import DIALOGUE, dialogue_wav
from granit.ingest.audio import Segment, Transcript, Word, decode
from granit.ingest.speakers import ACTIVE, Diarizer, turns

pytestmark = pytest.mark.model

EDGE_S = 0.2  # turn starts and ends within this of the true line times
PEAK_GB = 2.0  # spike: 1.3 GB for 37 min of audio (above what the process already holds: model tests share one)


@pytest.fixture(scope="module")
def dialogue(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[tuple[float, float, str]]]:
    return dialogue_wav(DIALOGUE, tmp_path_factory.mktemp("dialogue"))


@pytest.fixture(scope="module")
def diarizer() -> Diarizer:
    assert LOCAL_MODELS["diarization"].is_built(), "run: uv run granit models convert diarization"
    return Diarizer().load()


def test_two_voices_alternate_with_edges_near_the_line_times(
    diarizer: Diarizer, dialogue: tuple[Path, list[tuple[float, float, str]]]
) -> None:
    import mlx.core as mx

    path, lines = dialogue
    held = mx.get_active_memory()
    mx.reset_peak_memory()
    probs, frame_s = diarizer.probabilities(decode(path))
    assert (mx.get_peak_memory() - held) / 1e9 < PEAK_GB
    channels = []
    for start, end, _ in lines:
        window = probs[int(start / frame_s) : int(end / frame_s)]
        channel = int(window.sum(axis=0).argmax())
        channels.append(channel)
        active = (probs[:, channel] > ACTIVE).nonzero()[0] * frame_s
        inside = active[(active > start - 1.0) & (active < end + 1.0)]  # this line's activity
        assert abs(inside.min() - start) <= EDGE_S and abs(inside.max() - end) <= EDGE_S, (
            f"line {start:.2f}–{end:.2f} s heard as {inside.min():.2f}–{inside.max():.2f} s"
        )
    assert channels[0] == channels[2] != channels[1] == channels[3], channels


def test_label_gives_every_word_its_lines_speaker(
    diarizer: Diarizer, dialogue: tuple[Path, list[tuple[float, float, str]]]
) -> None:
    path, lines = dialogue
    # One word per line, at its middle: what TurboCTC's word times look like to the labeller.
    words = tuple(
        Word(f"w{i}", (s + e) / 2, (s + e) / 2 + 0.2) for i, (s, e, _) in enumerate(lines)
    )
    transcript = Transcript(lines[-1][1] + 1, 0.0, 1, (Segment(0.0, lines[-1][1], "w", words),))
    labelled = diarizer.label(transcript, decode(path))
    assert [w.speaker for w in labelled.segments[0].words] == [1, 2, 1, 2]
    assert [t.speaker for t in turns(labelled.segments)] == [1, 2, 1, 2]
