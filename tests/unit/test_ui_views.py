"""View data for the pages (no Streamlit): history, library and job rows, schemas, forms, uploads, page images."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from granit.store.db import NewExtraction, Store
from granit.ui import views
from tests.unit.test_ui_backend import library, make_backend, wait_for
from tests.unit.test_worker import FIXTURES

DOCS = FIXTURES / "documents"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return library(tmp_path / "data")


def test_history_rebuilds_turns_with_their_sources(tmp_path: Path) -> None:
    library(tmp_path / "data").close()
    backend = make_backend(tmp_path / "data").start()
    wait_for(lambda: backend.status().ready)
    answer = backend.ask("How much revenue was there in Q2?")
    live = views.turn_from_answer(answer)
    assert live.reasoning and live.cited and live.cited[0].citation.startswith("report.pdf")
    (stored,) = views.turns_from_history(backend.store)
    assert (stored.question, stored.answer, stored.thinking) == (live.question, live.answer, "low")
    assert [(s.number, s.citation, s.cited) for s in stored.sources] == [
        (s.number, s.citation, s.cited) for s in live.sources
    ]
    # A deleted source keeps its place in old answers.
    report = next(s for s in backend.store.sources() if s.name == "report.pdf")
    backend.store.delete_source(report)
    (after,) = views.turns_from_history(backend.store)
    assert (
        after.cited[0].citation == "(removed from the library)" and after.cited[0].source_id is None
    )
    backend.shutdown()


def test_library_rows(store: Store) -> None:
    rows = {r["Name"]: r for r in views.library_rows(store.sources())}
    assert rows["vad_pauses.wav"]["Type"] == "Recording"
    assert rows["vad_pauses.wav"]["Contents"] == "13 s · 50% speech"
    assert rows["report.pdf"]["Contents"] == "2 pages · 1 chart"
    assert rows["report.pdf"]["Status"] == "ready" and rows["report.pdf"]["Chunks"] == 5


@pytest.mark.parametrize(
    ("seconds", "text"), [(None, ""), (12.7, "13 s"), (754.2, "12:34"), (3754, "1:02:34")]
)
def test_duration(seconds: float | None, text: str) -> None:
    assert views.duration(seconds) == text


def test_size() -> None:
    assert [views.size(n) for n in (512, 2048, 5 * 1024**2, 3 * 1024**3)] == [
        "512 B",
        "2.0 KB",
        "5.0 MB",
        "3.0 GB",
    ]


def test_job_rows_name_the_task(store: Store) -> None:
    source = next(s for s in store.sources() if s.name == "report.pdf")
    store.queue_ingest(source, {"vision_tables": True})
    store.enqueue_extract(source, {"type": "object", "properties": {"x": {}}})
    rows = views.job_rows(store.job_rows())
    assert [(r["File"], r["Task"], r["Status"]) for r in rows[:2]] == [
        ("report.pdf", "Extract fields", "queued"),
        ("report.pdf", "Ingest (accurate tables)", "queued"),
    ]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("{", "Not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"type": "objekt"}', "Not a valid JSON Schema"),
        ('{"type": "string"}', "describe an object with fields"),
        ('{"type": "object", "properties": {}}', "describe an object with fields"),
    ],
)
def test_check_schema_explains_problems(text: str, problem: str) -> None:
    schema, error = views.check_schema(text)
    assert schema is None and error is not None and problem in error


def test_the_example_schema_is_the_fixture_schema() -> None:
    fixture = json.loads((DOCS / "invoice_schema.json").read_text())
    assert fixture == views.EXAMPLE_SCHEMA
    assert views.check_schema(json.dumps(fixture)) == (fixture, None)


def test_form_result_keeps_schema_order(store: Store) -> None:
    source = next(s for s in store.sources() if s.name == "report.pdf")
    schema = {"type": "object", "properties": {"b": {}, "a": {}, "c": {}}}
    store.replace_extraction(
        source,
        NewExtraction(
            kind="form",
            format="json",
            content=json.dumps({"a": "1", "b": "$2", "c": None, "extra": "x"}),
            valid=False,
            model="vision@rev",
            errors=["'c' is required"],
            missing=["c"],
            schema=schema,
        ),
    )
    result = views.form_result(views.latest(store.extractions(source.id), "form"))
    assert [(f["Field"], f["Value"], f["Found"]) for f in result.fields] == [
        ("b", "$2", True),
        ("a", "1", True),
        ("c", "", False),
        ("extra", "x", True),
    ]
    assert not result.valid and result.missing == ["c"] and result.errors == ["'c' is required"]


def test_uploads_queue_an_ingest_and_only_documents_take_accurate_tables(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    pdf = (DOCS / "memo.pdf").read_bytes()
    doc, created = views.add_upload(store, "Memo.PDF", pdf, accurate_tables=True)
    assert created and doc.name == "Memo.PDF" and doc.ext == ".pdf"
    wav = (FIXTURES / "audio" / "vad_pauses.wav").read_bytes()
    views.add_upload(store, "call.wav", wav, accurate_tables=True)
    assert [j.params for j in store.jobs()] == [{"vision_tables": True}, {}]
    again, created = views.add_upload(store, "copy.pdf", pdf)
    assert not created and again.id == doc.id
    with pytest.raises(ValueError, match="not an audio file or document"):
        views.add_upload(store, "notes.txt", b"hello")


def test_page_images() -> None:
    assert (
        views.page_count(DOCS / "report.pdf") == 2 and views.page_count(DOCS / "invoice.png") == 1
    )
    for path, page in ((DOCS / "report.pdf", 2), (DOCS / "invoice.png", 1)):
        image = Image.open(io.BytesIO(views.page_png(path, page)))
        assert image.format == "PNG" and image.width > 500


def test_document_and_transcript_text_are_escaped() -> None:
    md = "| Amount |\n|---|\n| $620.00 |\n| $3,720.00 |\n\n<!-- image -->"
    assert views.document_md(md) == "| Amount |\n|---|\n| \\$620.00 |\n| \\$3,720.00 |\n\n*(image)*"
    transcript = {"segments": [{"start": 754.2, "text": "it costs $5 or $6"}]}
    assert views.transcript_md(transcript) == "`12:34` it costs \\$5 or \\$6"


@pytest.mark.parametrize(
    ("value", "expected"), [("3", 3), (None, None), ("", None), ("x", None), ("-1", None)]
)
def test_source_param(value: str | None, expected: int | None) -> None:
    assert views.source_param(value) == expected


def test_model_label() -> None:
    assert (
        views.model_label("ibm-granite/granite-vision-4.1-4b@37d591f06319e8f1638b")
        == "granite-vision-4.1-4b @ 37d591f"
    )
    assert views.model_label("local-model") == "local-model"
