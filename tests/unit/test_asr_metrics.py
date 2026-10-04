"""WER normalization (PLAN.md §4.9): formatting differences must not count as recognition errors."""

from __future__ import annotations

import pytest

from granit.evaluate.asr import normalize, wer, words_to_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one hundred and five", "105"),
        ("forty two", "42"),
        (
            "twenty twenty six",
            "20 26",
        ),  # a new number starts when the next word can't extend this one
        ("five twenty", "5 20"),
        ("two thousand and twenty six", "2026"),
        ("one million three hundred thousand", "1300000"),
        ("ten thousand", "10000"),
        ("zero zero four two", "0 0 4 2"),
        ("ninety nine bottles and three", "99 bottles and 3"),
        ("and then", "and then"),
    ],
)
def test_number_words(text: str, expected: str) -> None:
    assert " ".join(words_to_numbers(text.split())) == expected


def test_normalize_punctuation_case_contractions_and_numbers() -> None:
    assert (
        normalize("Let's meet at TEN; revenue was up 12%!")
        == "let us meet at 10 revenue was up 12 percent"
    )
    assert normalize("Invoice INV-2026-0042 totals $4,980.") == "invoice inv 2026 0042 totals 4980"
    assert (
        normalize("It’s Priya’s turn") == "it is priyas turn"
    )  # possessive apostrophes dropped: "priyas" either way


def test_wer_ignores_formatting_but_counts_real_errors() -> None:
    assert wer("Revenue was up twelve percent.", "revenue was up 12%") == 0.0
    assert wer("send the draft to legal", "send a draft to legal") == pytest.approx(0.2)
    assert wer("", "") == 0.0
    assert wer("", "noise") == 1.0
