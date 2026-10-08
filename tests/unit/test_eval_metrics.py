"""Evaluation metrics (PLAN.md §4.9): normalization, retrieval, extraction, table cell F1, action items, agreement."""

from __future__ import annotations

import pytest

from granit.evaluate import metrics as m
from granit.evaluate.metrics import Located, Ref

# ── normalization ──


@pytest.mark.parametrize(
    ("text", "normal"),
    [
        ("Total: $4,980.00", "total 4980"),
        ("USD 1,234,567.50", "usd 1234567.50"),
        ("Due October 14, 2026.", "due 2026-10-14"),
        ("due 14 Oct 2026", "due 2026-10-14"),
        ("due 10/14/2026", "due 2026-10-14"),
        ("2026-9-4", "2026-09-04"),
        ("Invoice INV–2026–0042", "invoice inv-2026-0042"),
        ("Marcus’s draft — by Friday!", "marcus s draft by friday"),
        ("rate 4.5% at 10:30", "rate 4.5% at 10:30"),
        ("  Many   spaces\n", "many spaces"),
    ],
)
def test_normalize(text: str, normal: str) -> None:
    assert m.normalize(text) == normal


def test_facts_match_whole_words_after_normalizing() -> None:
    answer = "The total due is $4,980.00, payable by October 14, 2026 [1]."
    assert m.contains(answer, "$4,980") and m.contains(answer, "4980")
    assert m.contains(answer, "2026-10-14")
    assert not m.contains(answer, "498")  # no partial numbers
    assert not m.contains(answer, "") and not m.contains("16 units", "6")
    assert m.fact_coverage(["$4,980", "Friday"], answer) == 0.5
    assert m.fact_coverage([], answer) is None
    assert m.fact_found("two years | 2 years", "Renew for 2 years [1].")
    assert not m.fact_found("two years | 2 years", "Renew for 3 years.")
    # v30 private set: the answer spaced the en dash of a range
    assert m.fact_found("3.9–4.3 | 3.9 to 4.3", "is 3.9 – 4.3 percent【3】.")


# ── retrieval ──

PDF, WAV = "a" * 64, "b" * 64


def test_chunks_match_refs_by_file_page_and_time() -> None:
    page2 = Ref(PDF, page=2)
    assert m.matches(Located(PDF, 2, 2), page2) and m.matches(Located(PDF, 1, 3), page2)
    assert not m.matches(Located(PDF, 3, 3), page2) and not m.matches(Located(WAV, 2, 2), page2)
    assert not m.matches(Located(PDF), page2)
    talk = Ref(WAV, start_s=60, end_s=90)
    assert m.matches(Located(WAV, start_s=80, end_s=110), talk)
    assert not m.matches(Located(WAV, start_s=95, end_s=120), talk)
    assert m.matches(Located(PDF, 5, 5), Ref(PDF))  # a whole-file ref


def test_recall_mrr_and_citation_precision() -> None:
    refs = [Ref(PDF, page=2), Ref(WAV, start_s=60, end_s=90)]
    ranked = [
        Located(PDF, 1, 1),
        Located(PDF, 2, 2),
        Located(PDF, 3, 3),
        Located(WAV, start_s=70, end_s=75),
    ]
    assert m.recall_at_k(ranked, refs, k=8) == 1.0
    assert m.recall_at_k(ranked, refs, k=2) == 0.5
    assert m.reciprocal_rank(ranked, refs, k=8) == 0.5
    assert m.reciprocal_rank(ranked[:1], refs, k=8) == 0.0
    assert m.recall_at_k(ranked, [], k=8) is None and m.reciprocal_rank(ranked, [], k=8) is None
    assert m.citation_precision([Located(PDF, 2, 2), Located(PDF, 3, 3)], refs) == 0.5
    assert m.citation_precision([], refs) is None


# ── extraction ──


def test_field_values_match_after_normalization() -> None:
    expected = {
        "total": "$4,980.00",
        "due": "2026-10-14",
        "po": None,
        "vendor": "Granite Supply Co.",
    }
    got = {"total": "4980", "due": "October 14, 2026", "po": "", "vendor": "Granite Supply Co"}
    assert m.field_matches(expected, got) == {
        "total": True,
        "due": True,
        "po": True,
        "vendor": True,
    }
    assert m.field_matches({"po": None}, {"po": "PO-1"}) == {"po": False}
    assert m.field_matches({"total": "5"}, {}) == {"total": False}


# ── tables ──

TRUTH = [["Region", "Q1", "Q2"], ["North", "1,240", "1,310"], ["South", "980", "1,045"]]


def test_identical_tables_score_one() -> None:
    assert m.cell_f1(TRUTH, TRUTH) == 1.0
    assert (
        m.cell_f1(
            TRUTH, [["region", "q1", "q2"], ["north", "1240", "1310"], ["south", "980", "1045"]]
        )
        == 1.0
    )


def test_one_wrong_cell_costs_one_cell() -> None:
    wrong = [r[:] for r in TRUTH]
    wrong[2][2] = "1,054"
    assert m.cell_f1(TRUTH, wrong) == pytest.approx(8 / 9)


def test_an_extra_row_or_column_only_costs_precision() -> None:
    extra_row = [["Shipments"], *TRUTH]
    assert m.cell_f1(TRUTH, extra_row) == pytest.approx(2 * 1 * (9 / 10) / (1 + 9 / 10))
    extra_col = [[*r, "x"] for r in TRUTH]
    assert m.cell_f1(TRUTH, extra_col) == pytest.approx(2 * (9 / 12) / (1 + 9 / 12))


def test_missing_rows_cost_recall_and_empty_predictions_score_zero() -> None:
    assert m.cell_f1(TRUTH, TRUTH[:2]) == pytest.approx(2 * (6 / 9) / (1 + 6 / 9))
    assert m.cell_f1(TRUTH, []) == 0.0 and m.cell_f1([], []) == 1.0
    assert m.cell_f1(TRUTH, [["", ""], *TRUTH, ["", ""]]) == 1.0  # empty rows don't count


def test_html_tables_expand_merged_cells() -> None:
    html = (
        "<table><tr><th rowspan='2'>Region</th><th colspan='2'>2026</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>North</td><td>1,240</td><td>1,310<br>(est.)</td></tr></table>"
    )
    assert m.html_grid(html) == [
        ["Region", "2026", "2026"],
        ["Region", "Q1", "Q2"],
        ["North", "1,240", "1,310 (est.)"],
    ]
    assert m.html_grid(
        "<table><tr><td>a</td><td rowspan='3'>b</td></tr><tr><td>c</td></tr></table>"
    ) == [
        ["a", "b"],
        ["c", "b"],
        ["", "b"],
    ]


# ── action items ──


def test_action_items_match_owner_and_task_keywords() -> None:
    expected = [
        {"owner": "Marcus", "task": "send the revised draft to Legal", "due": "Wednesday"},
        {"owner": "Elena", "task": "update the customer forecast", "due": None},
        {"owner": "Priya", "task": "book the venue for the offsite", "due": None},
    ]
    got = [
        {"owner": "marcus", "task": "Send draft to legal", "due": "wednesday"},
        {"owner": "Elena Ruiz", "task": "update customer forecasts", "due": None},
        {"owner": "Tom", "task": "book the venue", "due": None},
        {"owner": None, "task": "order coffee", "due": None},
    ]
    recall, precision = m.action_items(expected, got)
    assert recall == pytest.approx(2 / 3) and precision == pytest.approx(2 / 4)
    assert m.action_items([], []) == (None, None)
    # A misheard name is still the same owner (the mishearing counts in WER), a different person isn't.
    assert m.owner_matches("Priya", "pria") and m.owner_matches("Elena Ruiz", "elena")
    assert not m.owner_matches("Sam", "Pam") and not m.owner_matches("Marcus", None)
    assert m.item_matches(
        {"owner": None, "task": "book venue"}, {"owner": "Tom", "task": "Book the venue"}
    )


def test_keywords_stem_word_forms() -> None:
    # From the AMI triage: the same task worded as "splitting" / "split", "arrangement" / "arranging", "offices" / "office".
    assert m.keywords("splitting the offices") == m.keywords("split an office")
    assert (
        m.keywords("arranging people")
        == m.keywords("arrangement of people")
        == m.keywords("arrange people")
    )
    assert m.keywords("planned passes") == {"plan", "pass"}


def test_mentions_whole_names_in_order() -> None:
    text = "Priya will check. Same as before, the project is late and the manager agreed."
    assert m.mentions(text, "Priya") and m.mentions(
        "so pria said", "Priya"
    )  # a misheard name still counts
    assert not m.mentions("the samples are late", "Sam")  # whole words, not inside "samples"
    # Near-spellings count ("same" for "Sam"): erring towards "said" keeps the owner required, the strict side.
    assert m.mentions(text, "Sam")
    assert not m.mentions(text, "Project Manager")  # both words, but not together
    assert m.mentions("as project manager I'll", "Project Manager")


def test_owner_the_transcript_never_says_is_met_by_no_owner() -> None:
    expected = [
        {"owner": "Project Manager", "task": "post the minutes", "due": None},
        {"owner": "Marcus", "task": "send the draft to legal", "due": None},
    ]
    got = [
        {"owner": None, "task": "Post the meeting minutes", "due": None},
        {"owner": None, "task": "send the draft to legal", "due": None},
    ]
    transcript = "Marcus, can you send the draft to legal? I'll post the minutes."
    # A role nobody says can't be heard; a name that is said still has to be given.
    assert m.action_items(expected, got, transcript) == (0.5, 0.5)
    assert m.action_items(expected, got) == (0.0, 0.0)  # without the transcript, every owner counts
    wrong = [{"owner": "Elena", "task": "post the minutes", "due": None}]
    assert m.action_items(expected[:1], wrong, transcript) == (
        0.0,
        0.0,
    )  # a guessed owner isn't met


def test_cohens_kappa() -> None:
    assert m.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert m.cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert m.cohens_kappa([True, True], [True, True]) is None  # one-sided: chance agreement is 1
    assert m.cohens_kappa([], []) is None


def test_mean_and_percentile() -> None:
    assert m.mean([1.0, None, 0.0]) == 0.5 and m.mean([None]) is None
    assert m.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5 and m.percentile([], 0.9) is None


# ── speakers (M9, PLAN.md §3.7) ──

REFERENCE = [
    {"name": "Elena", "start": 0.0, "end": 4.0},
    {"name": "Priya", "start": 5.0, "end": 6.0},
    {"name": "Elena", "start": 7.0, "end": 9.0},
    {"name": "Marcus", "start": 10.0, "end": 12.0},
]


def test_reference_speaker_skips_silence_and_overlap() -> None:
    assert m.reference_speaker(1.0, REFERENCE) == "Elena"
    assert m.reference_speaker(6.2, REFERENCE) == "Priya"  # within the 0.3 s pad
    assert m.reference_speaker(20.0, REFERENCE) is None
    overlap = [*REFERENCE, {"name": "Sam", "start": 0.5, "end": 1.5}]
    assert m.reference_speaker(1.0, overlap) is None


def test_speakers_are_named_only_when_pure() -> None:
    words = [
        *[(t, t + 0.5, 1) for t in (0.0, 1.0, 2.0, 3.0, 7.0, 8.0)],  # Elena only
        *[(t, t + 0.5, 2) for t in (5.0, 5.5)],  # Priya…
        *[(t, t + 0.5, 2) for t in (10.0, 10.5, 11.0)],  # …and Marcus merged: 60 % Marcus
        (30.0, 30.5, 3),  # nobody in the reference
        (1.0, 1.5, None),
    ]
    assert m.speaker_naming(words, REFERENCE) == {1: "Elena"}
    assert m.speaker_naming(words, REFERENCE, min_purity=0.5) == {1: "Elena", 2: "Marcus"}
    # majority mapping: speaker 2 → Marcus, so Priya's 2 words are wrong: 9 of 11 words right
    assert m.speaker_attribution(words, REFERENCE) == round(9 / 11, 4)
    assert m.speaker_attribution([(30.0, 30.5, 1)], REFERENCE) is None
