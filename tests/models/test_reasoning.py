"""Reasoning + phases golden tests on the Mac (``uv run pytest -m model``), M5: real server, real answers, a real switch.

Thresholds: answers contain the expected facts and cite the expected sources; unanswerable questions are declined; the
meeting summary is valid JSON naming the right owners; Q&A comes back after a B → A → B switch and answers about the new file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from granit.models.phases import Phase, PhaseManager
from granit.reason.llm import LLMClient
from granit.reason.meetings import summarize, summarize_source
from granit.reason.prompts import SUMMARY_SCHEMA, summary_messages
from granit.reason.qa import QA, Answer
from granit.reason.tokens import GraniteTokens, messages_tokens
from granit.search.embed import Embedder
from granit.search.hybrid import Searcher
from granit.search.rerank import Reranker
from granit.search.vectors import VectorIndex
from granit.store.db import Store
from tests.conftest import ROOT

pytestmark = pytest.mark.model

MAX_ANSWER_S = 15.0  # M5 measured 2–5 s per answer on this library (thinking "low")
MAX_SWITCH_S = 30.0  # M1: B→A 7.6–8.9 s, A→B 6.5–6.7 s (+ the ingest itself)


@pytest.fixture(scope="module")
def phases(library: tuple[Store, float]) -> Iterator[PhaseManager]:
    store, _ = library
    manager = PhaseManager(store, idle_before_ingest_s=3600)
    manager.start()
    yield manager
    manager.stop()


@pytest.fixture(scope="module")
def qa(library: tuple[Store, float], phases: PhaseManager) -> QA:
    store, _ = library
    embedder = Embedder().load()
    searcher = Searcher(store, embedder, VectorIndex(store, embedder.revision), Reranker().load())
    searcher.warm_up()
    return QA(store, searcher, LLMClient(), GraniteTokens())


def ask(qa: QA, phases: PhaseManager, question: str) -> Answer:
    with phases.chat():
        return qa.ask(question)


def cited(answer: Answer) -> set[str]:
    return {h.source_name for h in answer.cited}


@pytest.mark.parametrize(
    ("question", "facts", "source"),
    [
        ("How much revenue was there in Q2?", ["145"], "report.pdf"),
        ("Who will send the draft to Legal, and by when?", ["marcus", "wednesday"], "meeting.flac"),
        ("What is the total due on invoice INV-2026-0042?", ["4,980"], "invoice.png"),
        ("Which region shipped the most units in Q3?", ["east", "1,560"], "report.pdf"),
    ],
)
def test_answers_contain_the_facts_and_cite_the_source(
    qa: QA, phases: PhaseManager, question: str, facts: list[str], source: str
) -> None:
    answer = ask(qa, phases, question)
    assert not answer.declined
    for fact in facts:
        assert fact in answer.text.lower(), answer.text
    assert source in cited(answer), (answer.text, cited(answer))
    assert answer.latency["total_s"] <= MAX_ANSWER_S


def test_unanswerable_questions_are_declined(qa: QA, phases: PhaseManager) -> None:
    for question in (
        "What is the warranty period for Widget A?",
        "Who won the 2026 football world cup?",
    ):
        answer = ask(qa, phases, question)
        assert answer.declined and not answer.cited, answer.text


def test_cross_source_question_cites_several_documents(qa: QA, phases: PhaseManager) -> None:
    answer = ask(qa, phases, "What did Finance approve, and what is still open?")
    assert len(cited(answer)) >= 2, (answer.text, cited(answer))


def test_turns_are_recorded_with_their_trace(qa: QA, phases: PhaseManager) -> None:
    answer = ask(qa, phases, "How much revenue was there in Q2?")
    turn = next(t for t in qa.store.turns() if t["id"] == answer.turn_id)
    trace = json.loads(turn["retrieval_trace"])
    assert set(trace["stages"]) == {"bm25", "ids", "vectors", "rrf", "rerank"}
    assert json.loads(turn["cited_chunk_ids"]) == [h.chunk_id for h in answer.cited]
    assert turn["prompt_tokens"] > 0 and turn["thinking"] == "low"


def owners(data: dict) -> dict[str, str]:
    return {(i["owner"] or "").lower(): i["task"].lower() for i in data["action_items"]}


def test_meeting_summary(library: tuple[Store, float], phases: PhaseManager) -> None:
    store, _ = library
    source = next(s for s in store.sources() if s.name == "meeting.flac")
    with phases.chat():
        summary = summarize_source(store, source, LLMClient(), GraniteTokens())
    assert summary.sections == 1
    items = owners(summary.data)
    assert "marcus" in items and "legal" in items["marcus"]
    assert "elena" in items and "forecast" in items["elena"]
    assert any("tuesday" in d.lower() for d in summary.data["decisions"])
    (stored,) = [e for e in store.extractions(source.id) if e["kind"] == "summary"]
    assert json.loads(stored["content"]) == summary.data


def test_long_meeting_path_sections_and_combines(
    library: tuple[Store, float], phases: PhaseManager
) -> None:
    """Force the sectioned path (used for meetings over ~1 hour) on the short meeting, and check nothing is lost."""
    from granit.ingest.audio import Transcript

    store, _ = library
    source = next(s for s in store.sources() if s.name == "meeting.flac")
    transcript = Transcript.from_json(
        json.loads((store.derived_dir(source) / "transcript.json").read_text())
    )
    counter = GraniteTokens()
    budget = (
        messages_tokens(summary_messages([], (99, 99)), counter) + 60
    )  # ~2–3 sentences per section
    with phases.chat():
        summary = summarize(transcript, LLMClient(), counter, budget=budget)
    assert summary.sections >= 2
    items = owners(summary.data)
    assert "marcus" in items and "elena" in items
    from jsonschema import validate

    validate(summary.data, SUMMARY_SCHEMA)


def test_phase_switch_ingests_and_qa_comes_back(qa: QA, phases: PhaseManager) -> None:
    store = qa.store
    source, _ = store.add_file(ROOT / "tests" / "fixtures" / "audio" / "formats" / "sample.m4a")
    try:
        result = phases.process_now()
        assert result is not None and len(result["done"]) == 1 and not result["failed"]
        status = phases.status()
        assert status.phase is Phase.QA and status.queued == 0
        timings = status.last_switch
        assert "error" not in timings
        assert timings["stop_s"] + timings["ingest_s"] + timings["restart_s"] <= MAX_SWITCH_S, (
            timings
        )
        answer = ask(qa, phases, "When is the quarterly invoice for the server racks due?")
        assert "friday" in answer.text.lower() and "sample.m4a" in cited(answer), answer.text
    finally:
        store.delete_source(source)  # the session library is shared: leave it as we found it
