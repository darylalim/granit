"""Reasoning without models (M5): prompt budget, citations, declining, JSON repair, Q&A turns, meeting summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from granit.ingest.audio import Segment, Transcript
from granit.models.server import Completion
from granit.reason import prompts
from granit.reason.llm import THINKING, InvalidJSON, LLMClient, parse_json
from granit.reason.meetings import summarize, summarize_source
from granit.reason.prompts import DECLINE, SUMMARY_SCHEMA, PromptTooLong
from granit.reason.qa import QA
from granit.reason.tokens import ApproxTokens, messages_tokens
from granit.search.hybrid import Hit, SearchResult
from granit.store.db import NewChunk, Store
from tests.conftest import ROOT

COUNTER = ApproxTokens()


def hit(i: int, text: str = "", source: str = "report.pdf", page: int | None = 1) -> Hit:
    return Hit(
        chunk_id=i,
        score=1.0 / i,
        source_id=1,
        source_name=source,
        source_kind="document",
        text=text or f"Fact number {i}.",
        context="Heading",
        element="text",
        page_start=page,
        page_end=page,
        start_s=None,
        end_s=None,
    )


def completion(content: str, reasoning: str = "") -> Completion:
    return Completion(content, reasoning, 1000, 20, 0, 0.5, 1.0, first_content_s=0.8)


# ── RAG prompt ──


def test_rag_prompt_numbers_sources_with_citations() -> None:
    prompt = prompts.rag_prompt("What is fact 2?", [hit(1), hit(2, page=2)], COUNTER)
    user = prompt.messages[1]["content"]
    assert prompt.messages[0]["content"] == prompts.RAG_SYSTEM
    assert "[1] report.pdf · p. 1 — Heading\nFact number 1." in user
    assert "[2] report.pdf · p. 2 — Heading\nFact number 2." in user
    assert user.endswith("Question: What is fact 2?")
    assert prompt.sources == [hit(1), hit(2, page=2)] and prompt.dropped == 0
    assert prompt.tokens == messages_tokens(prompt.messages, COUNTER)
    assert DECLINE in prompts.RAG_SYSTEM


def test_rag_prompt_keeps_to_the_budget() -> None:
    big, small = "x" * 3000, "short"
    hits = [hit(1, big), hit(2, big), hit(3, small)]
    base = messages_tokens(prompts.rag_prompt("q", [], COUNTER).messages, COUNTER)
    prompt = prompts.rag_prompt("q", hits, COUNTER, budget=base + 1200)
    # The first big source fits, the second doesn't, the small third one still does.
    assert [h.chunk_id for h in prompt.sources] == [1, 3] and prompt.dropped == 1
    assert prompt.tokens <= base + 1200
    with pytest.raises(PromptTooLong):
        prompts.rag_prompt("q" * 100_000, hits, COUNTER)


@pytest.mark.parametrize(
    ("answer", "numbers"),
    [
        ("Revenue was 145 [2].", [2]),
        ("A [1][3] and B [3].", [1, 3]),
        ("See [1, 3] and [2–3].", [1, 3, 2]),
        ("Out of range [9] and [0].", []),
        ("No citations.", []),
    ],
)
def test_parse_citations(answer: str, numbers: list[int]) -> None:
    assert prompts.parse_citations(answer, 3) == numbers


def test_is_decline() -> None:
    assert prompts.is_decline(DECLINE)
    assert prompts.is_decline("not found in your documents")
    assert not prompts.is_decline("Revenue was 145 [1].")


# ── meeting prompts ──


def segments(n: int, words: int = 8) -> list[Segment]:
    return [Segment(i * 5.0, i * 5.0 + 4, " ".join(["word"] * words)) for i in range(n)]


def test_transcript_lines_have_timestamps() -> None:
    assert prompts.transcript_lines([Segment(754.2, 760.0, "send the draft")]) == [
        "[12:34] send the draft"
    ]


def test_split_sections_packs_lines_under_the_budget() -> None:
    lines = prompts.transcript_lines(segments(200))
    base = messages_tokens(prompts.summary_messages([], (99, 99)), COUNTER)
    sections = prompts.split_sections(lines, COUNTER, budget=base + 500)
    assert len(sections) > 1
    assert [line for s in sections for line in s] == lines  # every line once, in order, never split
    for section in sections:
        assert (
            messages_tokens(prompts.summary_messages(section, (99, 99)), COUNTER) <= base + 500 + 64
        )
    assert prompts.split_sections(lines, COUNTER) == [lines]  # fits the real 14K budget in one
    with pytest.raises(PromptTooLong):
        prompts.split_sections(["x" * 10_000], COUNTER, budget=base + 100)


def test_summary_prompts_carry_the_schema() -> None:
    assert '"action_items"' in prompts.summary_messages(["[0:00] hi"])[0]["content"]
    combined = prompts.combine_messages([{"summary": "a"}, {"summary": "b"}])
    assert (
        "Section 2:" in combined[1]["content"]
        and "keep only the final version" in combined[0]["content"]
    )


# ── LLM client ──


class ScriptedLLM(LLMClient):
    """An LLMClient whose replies come from a script (no server)."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, Any]], str]] = []

    def chat(
        self, messages: list[dict[str, Any]], *, thinking: str | None = None, **_: Any
    ) -> Completion:
        self.calls.append((messages, thinking or self.thinking))
        return completion(self.replies.pop(0))


GOOD = json.dumps({"summary": "s", "decisions": [], "action_items": []})


def test_thinking_modes_map_to_template_arguments() -> None:
    assert THINKING == {
        "off": {"enable_thinking": False},
        "low": {"reasoning_effort": "low"},
        "on": {},
    }
    with pytest.raises(ValueError, match="thinking must be one of"):
        LLMClient(base_url="http://127.0.0.1:1").chat([], thinking="max")


def test_parse_json_validates_the_schema() -> None:
    assert parse_json(f"```json\n{GOOD}\n```", SUMMARY_SCHEMA)["summary"] == "s"
    with pytest.raises(InvalidJSON, match="decisions"):
        parse_json('{"summary": "s", "action_items": []}', SUMMARY_SCHEMA)
    with pytest.raises(InvalidJSON):
        parse_json("Sure! Here is the summary.", SUMMARY_SCHEMA)


def test_chat_json_repairs_once() -> None:
    llm = ScriptedLLM(["not json at all", GOOD])
    data, _ = llm.chat_json([{"role": "user", "content": "summarize"}], SUMMARY_SCHEMA)
    assert data["summary"] == "s" and len(llm.calls) == 2
    repair = llm.calls[1][0]
    assert repair[-2] == {"role": "assistant", "content": "not json at all"}
    assert "wasn't valid" in repair[-1]["content"] and llm.calls[1][1] == "off"
    with pytest.raises(InvalidJSON):
        ScriptedLLM(["bad", "still bad"]).chat_json([], SUMMARY_SCHEMA)


# ── Q&A ──


class FakeSearcher:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, mode: str, k: int) -> SearchResult:
        return SearchResult(
            query,
            mode,
            self.hits[:k],
            {"bm25": [(h.chunk_id, 1.0) for h in self.hits]},
            {"bm25": 0.001},
        )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "data")


def test_answer_maps_citations_to_chunks_and_records_the_turn(store: Store) -> None:
    hits = [hit(11, "Revenue in Q2 was 145."), hit(12, "Costs stayed flat.", page=2)]
    llm = ScriptedLLM(["Q2 revenue was 145 [1]; costs were flat [2]."])
    answer = QA(store, FakeSearcher(hits), llm, COUNTER).ask("How did Q2 go?")
    assert [h.chunk_id for h in answer.cited] == [11, 12] and answer.citation_numbers() == [1, 2]
    assert not answer.declined and answer.latency["first_token_s"] == 0.8
    assert llm.calls[0][1] == "low"  # the default thinking mode
    (turn,) = store.turns()
    assert turn["question"] == "How did Q2 go?" and json.loads(turn["cited_chunk_ids"]) == [11, 12]
    assert json.loads(turn["retrieval_trace"])["stages"]["bm25"] == [[11, 1.0], [12, 1.0]]
    assert json.loads(turn["retrieval"])["mode"] == "hybrid+rerank" and turn["thinking"] == "low"


def test_no_sources_means_decline_without_calling_the_model(store: Store) -> None:
    llm = ScriptedLLM([])
    answer = QA(store, FakeSearcher([]), llm, COUNTER).ask("Anything?", record=False)
    assert answer.text == DECLINE and answer.declined and answer.reply is None
    assert llm.calls == [] and store.turns() == []


def test_declined_answers_are_flagged(store: Store) -> None:
    answer = QA(store, FakeSearcher([hit(1)]), ScriptedLLM([DECLINE]), COUNTER).ask("Warranty?")
    assert answer.declined and answer.cited == [] and store.turns()[0]["declined"] == 1


# ── meetings ──


def transcript(n: int) -> Transcript:
    return Transcript(n * 5.0, n * 4.0, 1, tuple(segments(n)), "speech@rev")


def test_short_meetings_are_summarized_in_one_call() -> None:
    llm = ScriptedLLM([GOOD])
    summary = summarize(transcript(10), llm, COUNTER)
    assert summary.sections == 1 and summary.data["summary"] == "s" and len(llm.calls) == 1


def test_long_meetings_are_summarized_by_section_then_combined() -> None:
    llm = ScriptedLLM([GOOD] * 10)
    base = messages_tokens(prompts.summary_messages([], (99, 99)), COUNTER)
    summary = summarize(transcript(200), llm, COUNTER, budget=base + 600)
    assert summary.sections > 1 and len(llm.calls) == summary.sections + 1
    assert "(part 1 of" in llm.calls[0][0][1]["content"]
    assert llm.calls[-1][0][0]["content"] == prompts.COMBINE_SYSTEM


def test_placeholder_owners_become_null() -> None:
    items = [
        {"owner": "Someone", "task": "a", "due": None},
        {"owner": " the team ", "task": "b", "due": None},
        {"owner": "Sam", "task": "c", "due": None},
        {"owner": None, "task": "d", "due": None},
    ]
    reply = json.dumps({"summary": "s", "decisions": [], "action_items": items})
    summary = summarize(transcript(10), ScriptedLLM([reply]), COUNTER)
    assert [i["owner"] for i in summary.data["action_items"]] == [None, None, "Sam", None]


def test_empty_recordings_need_no_model() -> None:
    summary = summarize(Transcript(5.0, 0.0, 0, (), ""), ScriptedLLM([]), COUNTER)
    assert summary.sections == 0 and summary.data["action_items"] == []


def test_summarize_source_stores_a_summary_extraction(store: Store) -> None:
    source, _ = store.add_file(ROOT / "tests" / "fixtures" / "audio" / "vad_pauses.wav")
    job = store.claim_next()
    assert job is not None
    v = np.ones((1, 768), np.float16) / np.sqrt(768)
    store.complete_ingest(job, [NewChunk("hello", "speech", start_s=0.0, end_s=1.0)], v, "rev")
    out = store.derived_dir(source)
    out.mkdir(parents=True)
    (out / "transcript.json").write_text(json.dumps(transcript(3).to_json()))
    summarize_source(store, store.source(source.id), ScriptedLLM([GOOD, GOOD]), COUNTER)
    summarize_source(
        store, store.source(source.id), ScriptedLLM([GOOD]), COUNTER
    )  # replaces, not duplicates
    (summary,) = [e for e in store.extractions(source.id) if e["kind"] == "summary"]
    assert json.loads(summary["content"])["summary"] == "s"
    with pytest.raises(ValueError, match="isn't an ingested recording"):
        doc, _ = store.add_file(ROOT / "tests" / "fixtures" / "documents" / "invoice.png")
        summarize_source(store, doc, ScriptedLLM([]), COUNTER)


# ── named speakers (M9, PLAN.md §3.7) ──


def spoken(*turns: tuple[int | None, str]) -> Transcript:
    """One segment per turn, each word 1 s long, every word of a turn by that turn's speaker."""
    from granit.ingest.audio import Word

    segs, t = [], 0.0
    for speaker, text in turns:
        words = []
        for word in text.split():
            words.append(Word(word, t, t + 1, speaker))
            t += 1
        segs.append(Segment(words[0].start, words[-1].end, text, tuple(words)))
        t += 2
    return Transcript(t, t, 1, tuple(segs), "speech@rev")


MEETING = spoken(
    (1, "Priya can you send the renewal"),
    (2, "will do"),
    (3, "Sam can you book the review"),
    (4, "yes I will set that up"),
)


def test_without_names_lines_and_prompt_are_unchanged() -> None:
    plain = [f"[0:{s.start:02.0f}] {s.text}" for s in MEETING.segments]
    assert prompts.transcript_lines(MEETING.segments) == plain
    assert prompts.transcript_lines(MEETING.segments, {}) == plain
    assert prompts.summary_messages(plain)[0]["content"] == prompts.SUMMARY_SYSTEM
    llm = ScriptedLLM([GOOD])
    summary = summarize(MEETING, llm, COUNTER, names={9: "Nobody"})  # not a speaker here
    ((messages, _),) = llm.calls
    assert messages[0]["content"] == prompts.SUMMARY_SYSTEM
    assert messages[1]["content"].endswith("\n".join(plain))
    assert "speakers" not in summary.data


def test_named_speakers_label_their_turns_and_unnamed_stay_plain() -> None:
    lines = prompts.transcript_lines(MEETING.segments, {2: "Priya", 4: "Sam"})
    assert lines == [
        "[0:00] Priya can you send the renewal",
        "[0:08] Priya: will do",
        "[0:12] Sam can you book the review",
        "[0:20] Sam: yes I will set that up",
    ]


def test_two_speakers_with_one_name_are_one_person() -> None:
    t = spoken((1, "I will post"), (2, "the job ads"), (3, "great"))
    assert prompts.transcript_lines(t.segments, {1: "Marcus", 2: "Marcus"}) == [
        "[0:00] Marcus: I will post the job ads",
        "[0:10] great",
    ]


def test_summary_with_names_adds_the_rule_and_records_the_names() -> None:
    llm = ScriptedLLM([GOOD])
    summary = summarize(MEETING, llm, COUNTER, names={2: "Priya", 4: "Sam", 7: "Gone"})
    ((messages, _),) = llm.calls
    assert messages[0]["content"] == prompts.summary_system(named=True) != prompts.SUMMARY_SYSTEM
    assert prompts.NAMED_SPEAKERS_RULES in messages[0]["content"]
    assert "Priya: will do" in messages[1]["content"]
    assert summary.data["speakers"] == {"2": "Priya", "4": "Sam"}


def test_summarize_source_uses_the_librarys_speaker_names(store: Store) -> None:
    source, _ = store.add_file(ROOT / "tests" / "fixtures" / "audio" / "vad_pauses.wav")
    job = store.claim_next()
    assert job is not None
    v = np.ones((1, 768), np.float16) / np.sqrt(768)
    store.complete_ingest(job, [NewChunk("hello", "speech", start_s=0.0, end_s=1.0)], v, "rev")
    out = store.derived_dir(source)
    out.mkdir(parents=True)
    (out / "transcript.json").write_text(json.dumps(MEETING.to_json()))
    store.set_speaker_names(source, {2: "Priya"})
    llm = ScriptedLLM([GOOD])
    summarize_source(store, store.source(source.id), llm, COUNTER)
    assert "Priya: will do" in llm.calls[0][0][1]["content"]
    (summary,) = [e for e in store.extractions(source.id) if e["kind"] == "summary"]
    assert json.loads(summary["content"])["speakers"] == {"2": "Priya"}
