"""Chunking with citations (M4), on Docling output committed as fixtures (made by `granit convert` from the M3 fixtures)."""

from __future__ import annotations

import json

import pytest

from granit.ingest.audio import Segment, Transcript, Word
from granit.store import chunking
from granit.store.chunking import chunk_document, chunk_transcript, citation
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "documents"


def load(name: str):
    from docling_core.types.doc import DoclingDocument

    return DoclingDocument.model_validate(json.loads((FIXTURES / name).read_text()))


def test_report_chunks_keep_headings_pages_and_element_types() -> None:
    chunks = chunk_document(load("report.document.json"))
    summary = [(c.element, c.context, c.page_start, c.page_end) for c in chunks]
    assert summary == [
        ("text", "Quarterly Operations Report", 1, 1),
        ("table", "Quarterly Operations Report > Shipments by region (units)", 1, 1),
        ("text", "Quarterly Operations Report > Revenue", 2, 2),
        ("chart", "Quarterly Operations Report > Revenue", 2, 2),
        ("text", "Quarterly Operations Report > Revenue", 2, 2),
    ]
    table = chunks[1].text.splitlines()
    assert table[0].replace(" ", "") == "|Region|Q1|Q2|Q3|"  # column names, not "North, 1 = 1,240"
    assert "1,240" in chunks[1].text
    assert (
        "145" in chunks[3].text and "Revenue (thousand USD)" in chunks[3].text
    )  # chart data from M3
    assert chunks[0].search_body.startswith("Quarterly Operations Report\nThis report")


def test_digital_pdf_chunks() -> None:
    chunks = chunk_document(load("memo.document.json"))
    assert [c.element for c in chunks] == ["text", "table"]
    assert "INV-2026-0042" in chunks[0].text and "Approve the revised invoice" in chunks[1].text


def test_small_text_under_one_heading_is_merged() -> None:
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    doc = DoclingDocument(name="t")
    doc.add_heading("Notes")
    for i in range(5):
        doc.add_text(label=DocItemLabel.TEXT, text=f"Short paragraph number {i}.")
    doc.add_heading("Other")
    doc.add_text(label=DocItemLabel.TEXT, text="Different section.")
    chunks = chunk_document(doc)
    assert [c.context for c in chunks] == [
        "Notes",
        "Notes > Other",
    ]  # the first heading is the document's title
    assert chunks[0].text.count("Short paragraph") == 5


def test_long_text_is_split_at_sentences() -> None:
    sentence = "This sentence is exactly fifty characters long ok. "
    parts = chunking._split_text(sentence * 10, limit=120)
    assert all(len(p) <= 120 for p in parts)
    assert len(parts) == 5 and parts[0].count("This sentence") == 2
    assert chunking._split_text("x" * 250, limit=100) == ["x" * 100, "x" * 100, "x" * 50]


def test_long_tables_are_split_by_rows_with_the_header_repeated() -> None:
    table = "\n".join(["| A | B |", "|---|---|", *[f"| r{i} | {i} |" for i in range(40)]])
    parts = chunking._split_table(table, limit=150)
    assert len(parts) > 1 and all(len(p) <= 150 for p in parts)
    assert all(p.startswith("| A | B |\n|---|---|\n") for p in parts)
    rows = [line for p in parts for line in p.splitlines()[2:]]
    assert rows == [f"| r{i} | {i} |" for i in range(40)]  # every row exactly once, in order


def transcript(n: int, seconds: float = 4.0) -> Transcript:
    segments = tuple(
        Segment(i * seconds, i * seconds + seconds - 1, f"sentence number {i} of the meeting")
        for i in range(n)
    )
    return Transcript(n * seconds, n * seconds * 0.8, 1, segments, "speech@rev")


def test_transcripts_group_segments_and_keep_times() -> None:
    chunks = chunk_transcript(transcript(30), target=200)
    assert all(c.element == "speech" for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].start_s == 0.0 and chunks[-1].end_s == 30 * 4.0 - 1
    assert " ".join(c.text for c in chunks) == " ".join(s.text for s in transcript(30).segments)
    starts = [c.start_s for c in chunks if c.start_s is not None]
    assert len(starts) == len(chunks) and starts == sorted(starts)
    assert chunk_transcript(transcript(0)) == []


@pytest.mark.parametrize(
    ("args", "text"),
    [
        (("report.pdf", 2, 2, None), "report.pdf · p. 2"),
        (("report.pdf", 3, 4, None), "report.pdf · pp. 3–4"),
        (("call.m4a", None, None, 760.4), "call.m4a · 12:40"),
        (("photo.png", None, None, None), "photo.png"),
    ],
)
def test_citation(args: tuple, text: str) -> None:
    assert citation(*args) == text


def test_split_tables_repeat_the_section_row_they_continue() -> None:
    # Census e-commerce release: one table, "Adjusted" then "Not Adjusted" rows with identical columns (doc-11).
    rows = ["| Adjusted | Adjusted | Adjusted |"] + [
        f"| Q{i} | {i}00 | {i}.0 |" for i in range(1, 5)
    ]
    rows += ["| Not Adjusted | Not Adjusted | Not Adjusted |"] + [
        f"| Q{i} | {i}11 | {i}.1 |" for i in range(1, 5)
    ]
    table = "\n".join(["| Quarter | Sales | Share |", "|---|---|---|", *rows])
    parts = chunking._split_table(table, limit=110)
    assert len(parts) > 2
    for part in parts:
        assert chunking.section_row(part.splitlines()[2]), (
            part
        )  # every part names its section first
    data = [line for p in parts for line in p.splitlines()[2:] if not chunking.section_row(line)]
    assert data == [r for r in rows if not chunking.section_row(r)]  # every data row once, in order
    assert any(p.splitlines()[2].startswith("| Not Adjusted") and "| Q4 | 411" in p for p in parts)


def test_section_rows() -> None:
    assert chunking.section_row("| Not Adjusted | Not Adjusted | Not Adjusted |")
    assert not chunking.section_row("| 1,200 | 1,200 |")  # repeated numbers are data
    assert not chunking.section_row("| Q1 | 100 | 1.0 |")
    assert not chunking.section_row("| | | |")
    assert not chunking.section_row("| Total |")  # a one-column table's rows are all "spanning"


def dialogue(lines: list[tuple[int, str]], seconds: float = 3.0) -> Transcript:
    """One segment per line, its words spoken by ``speaker``."""
    segments = []
    for i, (speaker, text) in enumerate(lines):
        start = i * seconds
        words = tuple(
            Word(w, start + j * 0.2, start + j * 0.2 + 0.1, speaker)
            for j, w in enumerate(text.split())
        )
        segments.append(Segment(start, words[-1].end, text, words))
    return Transcript(len(lines) * seconds, len(lines) * seconds, 1, tuple(segments), "speech@rev")


MEETING = [
    (1, "Then let's renew for two years."),
    (2, "Will do."),
    (3, "Yes, I'll set that up."),
    (3, "Drivers rely on it every day."),
]
MEETING_END = 9.0 + 5 * 0.2 + 0.1  # the last line starts at 9 s; its sixth word ends 1.1 s later


def test_named_speakers_start_their_lines_in_chunks() -> None:
    """PLAN.md §3.8: search finds the name and Ask sees who spoke; unnamed speakers stay plain, as in summaries."""
    chunks = chunk_transcript(dialogue(MEETING), {3: "Sam", 2: "Priya"})
    assert [c.text for c in chunks] == [
        "Then let's renew for two years.\n"
        "Priya: Will do.\n"
        "Sam: Yes, I'll set that up. Drivers rely on it every day."
    ]
    assert chunks[0].start_s == 0.0 and chunks[0].end_s == MEETING_END
    assert "Speaker" not in chunks[0].text  # the model never sees an unnamed speaker's number


def test_without_names_chunks_are_todays_exactly() -> None:
    transcript = dialogue(MEETING)
    assert chunk_transcript(transcript, {}) == chunk_transcript(transcript)
    assert chunk_transcript(transcript)[0].text == " ".join(text for _, text in MEETING)


def test_named_turns_pack_to_the_target_and_long_turns_keep_their_name() -> None:
    long = " ".join(f"word{i}" for i in range(400))  # ~3,000 characters: over MAX_CHARS
    chunks = chunk_transcript(
        dialogue([(1, "Short opening line."), (2, long)]), {2: "Priya"}, target=300
    )
    assert all(len(c.text) <= chunking.MAX_CHARS for c in chunks)
    pieces = [line for c in chunks for line in c.text.splitlines()]
    assert pieces[0] == "Short opening line."
    assert len(pieces) > 2 and all(p.startswith("Priya: ") for p in pieces[1:])
    assert " ".join(p.removeprefix("Priya: ") for p in pieces[1:]) == long
