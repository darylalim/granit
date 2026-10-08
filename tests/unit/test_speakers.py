"""Speakers (PLAN.md §3.7) without models: word → speaker labelling, numbering, turns, transcript JSON."""

from __future__ import annotations

import json

import numpy as np
import pytest

from granit.ingest import speakers as sp
from granit.ingest.audio import Segment, Transcript, Word

FRAME = 0.01


def activity(duration_s: float, *spans: tuple[int, float, float]) -> np.ndarray:
    """frames × 4 channels; each (channel, start, end) is active (probability 0.9)."""
    probs = np.zeros((int(duration_s / FRAME), 4))
    for channel, start, end in spans:
        probs[int(start / FRAME) : int(end / FRAME), channel] = 0.9
    return probs


def w(text: str, start: float, end: float, speaker: int | None = None) -> Word:
    return Word(text, start, end, speaker)


def test_word_takes_the_channel_active_around_it() -> None:
    probs = activity(10, (2, 0.0, 3.0), (0, 3.0, 6.0))
    words = [w("hello", 1.0, 1.2), w("there", 4.0, 4.3)]
    assert sp.word_channels(words, probs, FRAME) == [2, 0]


def test_word_at_a_turn_edge_goes_to_whoever_speaks_more_of_it() -> None:
    probs = activity(10, (1, 0.0, 2.05), (3, 2.05, 5.0))
    assert sp.word_channels([w("edge", 1.9, 2.3)], probs, FRAME) == [3]


def test_silent_word_takes_the_nearest_channel_then_the_previous_words() -> None:
    probs = activity(20, (1, 0.0, 2.0), (2, 5.0, 6.0))
    words = [
        w("a", 1.0, 1.1),
        w("b", 4.5, 4.6),  # nothing within 0.1 s; channel 2 starts 0.4 s later
        w("c", 10.0, 10.1),  # nothing within 1 s: keeps the previous word's
    ]
    assert sp.word_channels(words, probs, FRAME) == [1, 2, 2]


def test_words_before_anyone_speaks_have_no_channel() -> None:
    probs = activity(10, (1, 8.0, 9.0))
    assert sp.word_channels([w("um", 0.5, 0.6), w("hi", 8.2, 8.4)], probs, FRAME) == [None, 1]


def test_speakers_are_numbered_by_first_appearance() -> None:
    assert sp.number_speakers([None, 3, 3, 0, 3, 2, 0]) == [None, 1, 1, 2, 1, 3, 2]


def transcript(*segments: Segment) -> Transcript:
    return Transcript(duration_s=100.0, speech_s=50.0, chunks=1, segments=segments)


def seg(*words: Word) -> Segment:
    return Segment(words[0].start, words[-1].end, " ".join(x.text for x in words), words)


def test_with_speakers_sets_one_speaker_per_word_in_order() -> None:
    t = transcript(seg(w("a", 0, 1), w("b", 1, 2)), seg(w("c", 3, 4)))
    labelled = sp.with_speakers(t, [1, 2, 2])
    assert [x.speaker for s in labelled.segments for x in s.words] == [1, 2, 2]
    assert labelled.segments[0].text == "a b"  # text and times unchanged
    assert sp.speaker_count(labelled) == 2
    assert sp.has_speakers(labelled) and not sp.has_speakers(t)


def test_turns_split_at_speaker_changes_across_segments() -> None:
    t = transcript(
        seg(w("thanks", 0, 1, 1), w("for", 1, 2, 1), w("coming", 2, 3, 1), w("sure", 3, 4, 2)),
        seg(w("I'll", 5, 6, 2), w("send", 6, 7, 2), w("it", 7, 8, 2)),
    )
    assert sp.turns(t.segments) == [
        sp.Turn(0, 3, 1, "thanks for coming"),
        sp.Turn(3, 8, 2, "sure I'll send it"),
    ]


def test_long_monologues_are_split() -> None:
    words = [w(f"w{i}", i * 10.0, i * 10.0 + 1, 1) for i in range(8)]  # 71 s of one speaker
    out = sp.turns([seg(*words)], max_s=30)
    assert [(t.start, t.end) for t in out] == [(0, 21), (30, 51), (60, 71)]
    assert all(t.speaker == 1 for t in out)


def test_segments_without_words_stay_whole() -> None:
    bare = Segment(0.0, 2.0, "no word times", ())
    assert sp.turns([bare]) == [sp.Turn(0.0, 2.0, None, "no word times")]


@pytest.mark.parametrize(
    ("speaker", "names", "label"),
    [(None, {}, ""), (2, {}, "Speaker 2"), (2, {2: "Priya"}, "Priya"), (2, {2: ""}, "Speaker 2")],
)
def test_speaker_label(speaker: int | None, names: dict[int, str], label: str) -> None:
    assert sp.speaker_label(speaker, names) == label


def test_transcript_json_round_trips_speakers_and_reads_old_files() -> None:
    t = sp.with_speakers(transcript(seg(w("a", 0, 1), w("b", 1, 2))), [1, None])
    again = Transcript.from_json(json.loads(json.dumps(t.to_json())))
    assert again == t
    old = {
        "duration_s": 1.0,
        "speech_s": 1.0,
        "chunks": 1,
        "segments": [
            {"start": 0, "end": 1, "text": "a", "words": [{"text": "a", "start": 0, "end": 1}]}
        ],
    }
    assert Transcript.from_json(old).segments[0].words[0].speaker is None


def test_vocabulary_keeps_the_speaker_of_a_merged_word() -> None:
    from granit.ingest.audio import apply_vocabulary

    merged = apply_vocabulary([w("north", 0, 1, 2), w("beam", 1, 2, 2)], ["Northbeam"])
    assert [(x.text, x.speaker) for x in merged] == [("Northbeam", 2)]
