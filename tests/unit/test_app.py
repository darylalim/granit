"""The Streamlit pages, headless (``st.testing.v1.AppTest``) against a backend with fake processes and a scripted LLM (M6).

What a browser would need (layout, wrapping, colors) is checked by ``scripts/ui_screenshots.py``; this covers behavior:
each page renders, questions are answered with citations and remembered, a question asked while Q&A is paused waits,
and Library / Extract actions queue the right jobs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from granit.store.db import Store
from granit.ui import backend as backend_module
from granit.ui.backend import Backend
from tests.conftest import ROOT
from tests.unit.test_ui_backend import ANSWER, library, make_backend, wait_for

HOME = str(ROOT / "app" / "Home.py")
PAGES = ["app_pages/ingest.py", "app_pages/library.py", "app_pages/ask.py", "app_pages/extract.py"]


def use(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    st.cache_resource.clear()
    monkeypatch.setattr(backend_module, "create_backend", lambda _data: backend)


@pytest.fixture
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Backend]:
    library(tmp_path / "data").close()
    backend = make_backend(tmp_path / "data").start()
    use(backend, monkeypatch)
    yield backend
    st.cache_resource.clear()  # releases the backend: shuts it down


def app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(HOME, default_timeout=30).run()
    if page:
        at.switch_page(page).run()
    return at


def wait_ready(backend: Backend) -> None:
    wait_for(lambda: backend.status().ready)


def texts(at: AppTest) -> str:
    return "\n".join(
        str(e.value)
        for kind in ("markdown", "caption", "info", "success", "error")
        for e in getattr(at, kind)
    )


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(ready: Backend, page: str) -> None:
    wait_ready(ready)
    at = app(page)
    assert not at.exception, at.exception
    assert at.title[0].value == Path(page).stem.capitalize()
    assert at.success and "Ready" in at.success[0].value  # the phase banner


def test_pages_render_on_an_empty_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(tmp_path / "data")
    use(backend, monkeypatch)
    for page in PAGES:
        at = app(page)
        assert not at.exception, (page, at.exception)
    assert "Your library is empty" in texts(app())
    st.cache_resource.clear()


def test_the_banner_shows_start_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend(tmp_path / "data", other_phase=lambda: "granit.ingest.worker --data x")
    use(backend, monkeypatch)
    at = app()
    assert not at.exception
    assert at.info and "another granit process" in at.info[0].value
    st.cache_resource.clear()


def test_ask_answers_with_citations_and_remembers(ready: Backend) -> None:
    wait_ready(ready)
    at = app()
    at.chat_input[0].set_value("How much revenue was there in Q2?").run()
    assert not at.exception, at.exception
    assert "Revenue was 145 thousand USD in Q2 :gray-badge[1]." in texts(at)
    (turn,) = ready.store.turns()
    assert turn["answer"] == ANSWER
    # A new browser session starts with the recent history.
    again = app()
    assert "How much revenue was there in Q2?" in texts(again)


def test_a_question_asked_while_paused_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path / "data").close()
    # Q&A can't start while another granit process holds Phase A.
    backend = make_backend(tmp_path / "data", other_phase=lambda: "granit.ingest.worker --data x")
    use(backend, monkeypatch)
    at = app()
    at.chat_input[0].set_value("Who sends the draft?").run()
    assert not at.exception
    assert at.session_state["ask_pending"] == "Who sends the draft?"
    assert any("will be asked as soon as Q&A is back" in i.value for i in at.info)
    assert backend.store.turns() == []  # nothing was asked
    st.cache_resource.clear()


def test_library_shows_a_document_and_queues_accurate_tables(ready: Backend) -> None:
    wait_ready(ready)
    store = ready.store
    doc = next(s for s in store.sources() if s.name == "report.pdf")
    at = AppTest.from_file(HOME, default_timeout=30)
    at.session_state["library_selected"] = doc.id
    at.run()
    at.switch_page("app_pages/library.py").run()
    assert not at.exception, at.exception
    assert at.subheader[0].value == "report.pdf"
    reingest = next(b for b in at.button if b.label == "Re-ingest with accurate tables")
    reingest.click().run()
    job = store.jobs("queued")[-1]
    assert job.source_id == doc.id and job.params == {"vision_tables": True}


def test_library_shows_a_recording_transcript(ready: Backend) -> None:
    wait_ready(ready)
    audio = next(s for s in ready.store.sources() if s.kind == "audio")
    at = AppTest.from_file(HOME, default_timeout=30)
    at.session_state["library_selected"] = audio.id
    at.run()
    at.switch_page("app_pages/library.py").run()
    assert not at.exception, at.exception
    assert "`0:01` please send the revised invoice to finance" in texts(at)
    assert "No summary yet" in texts(at)


def test_library_names_a_recordings_speakers(ready: Backend) -> None:
    from granit.ingest.audio import Segment, Transcript, Word

    wait_ready(ready)
    audio = next(s for s in ready.store.sources() if s.kind == "audio")
    words = (Word("send", 1.0, 1.5, 1), Word("it", 1.5, 2.0, 1), Word("will", 3.0, 3.5, 2))
    spoken = Transcript(12.7, 6.4, 1, (Segment(1.0, 3.5, "send it will", words),), "speech@rev")
    (ready.store.derived_dir(audio) / "transcript.json").write_text(json.dumps(spoken.to_json()))
    at = AppTest.from_file(HOME, default_timeout=30)
    at.session_state["library_selected"] = audio.id
    at.run()
    at.switch_page("app_pages/library.py").run()
    assert not at.exception, at.exception
    assert [e.label for e in at.expander if e.label.startswith("Speakers")] == [
        "Speakers · 0 of 2 named"
    ]
    assert "**Speaker 2** will" in texts(at).replace("`0:03` ", "")
    at.text_input(key=f"speaker_name_{audio.id}_2").input("Priya")
    next(b for b in at.button if b.label == "Save names").click().run()
    assert not at.exception, at.exception
    assert ready.store.speaker_names(audio.id) == {2: "Priya"}
    assert "**Priya** will" in texts(at).replace("`0:03` ", "")


def test_extract_checks_the_schema_and_queues_a_job(ready: Backend) -> None:
    wait_ready(ready)
    at = app("app_pages/extract.py")
    assert not at.exception, at.exception
    at.text_area(key="extract_schema").set_value("{not json").run()
    assert at.error and "Not valid JSON" in at.error[0].value
    extract = next(b for b in at.button if b.label == "Extract fields")
    assert extract.disabled
    schema = {"type": "object", "properties": {"total": {"type": "string"}}}
    at.text_area(key="extract_schema").set_value(json.dumps(schema)).run()
    next(b for b in at.button if b.label == "Extract fields").click().run()
    assert not at.exception, at.exception
    (job,) = [j for j in ready.store.jobs("queued") if j.task == "extract"]
    assert job.params == {"schema": schema}
    assert any("Queued" in i.value for i in at.info)


def test_ingest_lists_jobs_and_retries_failures(ready: Backend) -> None:
    wait_ready(ready)
    store: Store = ready.store
    source, _ = store.add_file(ROOT / "tests" / "fixtures" / "documents" / "memo.pdf")
    store.conn.execute(
        "UPDATE jobs SET status = 'failed', attempts = 3, error = 'boom' WHERE source_id = ?",
        (source.id,),
    )
    at = app("app_pages/ingest.py")
    assert not at.exception, at.exception
    retry = next(b for b in at.button if b.label == "Retry")
    retry.click().run()
    assert [j.source_id for j in store.jobs("queued")] == [source.id]


def test_deep_links_open_a_source(ready: Backend) -> None:
    wait_ready(ready)
    doc = next(s for s in ready.store.sources() if s.name == "report.pdf")
    at = app()
    at.query_params["source"] = str(doc.id)
    at.switch_page("app_pages/library.py").run()
    assert not at.exception and at.subheader[0].value == "report.pdf"
    at.switch_page("app_pages/extract.py").run()
    assert not at.exception and at.selectbox(key="extract_source").value == doc.id
