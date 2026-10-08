"""The M8 verify job (PLAN.md §2.4) with a fake Guardian: schema migration, criteria polarity, the Phase C worker, the
phase manager's verify batch, and the badges the UI shows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from granit.models.phases import Phase, PhaseManager
from granit.store.db import MIGRATIONS, NewChunk, NewExtraction, Store, migrate, now
from granit.ui import views
from granit.verify.criteria import (
    GROUNDED,
    RELEVANT,
    custom_criterion,
    summary_criteria,
    summary_text,
)
from granit.verify.worker import SOURCES_REMOVED, VerifyWorker
from tests.conftest import ROOT

DOCS = ROOT / "tests" / "fixtures" / "documents"
SUMMARY = {
    "summary": "Budget review.",
    "decisions": ["Ship in May"],
    "action_items": [{"owner": None, "task": "Book the venue", "due": "Friday"}],
}


class FakeGuardian:
    """Answers by rule and records every call: ``scores`` maps a criterion's first words to a reply."""

    def __init__(self, scores: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.scores = scores or {}

    def __call__(self, messages: list[dict[str, Any]], documents: list[str]) -> str:
        block = messages[-1]["content"]
        criterion = block.split("### Criteria: ", 1)[1].split("\n", 1)[0]
        self.calls.append((criterion, documents))
        for start, reply in self.scores.items():
            if criterion.startswith(start):
                return reply
        return "<think>\n</think><score>no</score>"


def ingested(
    store: Store, name: str = "report.pdf", texts: tuple[str, ...] = ("Q2 revenue was $4.2M.",)
) -> list[int]:
    store.add_file(DOCS / name)
    job = store.claim_next()
    assert job is not None
    vectors = np.ones((len(texts), 768), np.float16) / np.sqrt(768)
    chunks = [NewChunk(t, "text", context="Results") for t in texts]
    return store.complete_ingest(job, chunks, vectors, "rev")


def turn(
    store: Store, chunk_ids: list[int], declined: bool = False, question: str = "Q2 revenue?"
) -> int:
    return store.record_turn(
        question=question,
        answer="I don't know." if declined else "Q2 revenue was $4.2M [1].",
        declined=declined,
        source_chunk_ids=chunk_ids,
        cited_chunk_ids=chunk_ids[:1],
        model="llm",
        thinking="low",
        retrieval={},
        retrieval_trace={},
        prompt_tokens=None,
        completion_tokens=None,
        latency={"total_s": 1.0},
    )


def summary(store: Store, name: str = "memo.pdf") -> tuple[Any, int]:
    source, _ = store.add_file(DOCS / name)
    store.replace_extraction(
        source,
        NewExtraction(
            kind="summary", format="json", content=json.dumps(SUMMARY), valid=True, model="llm"
        ),
    )
    (row,) = [e for e in store.extractions(source.id) if e["kind"] == "summary"]
    return source, row["id"]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "data")


# ── schema ──


def test_migration_keeps_jobs_and_their_extractions(tmp_path: Path) -> None:
    """Rebuilding ``jobs`` must not null ``extractions.job_id`` (foreign keys are off while it runs)."""
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    for number, script in enumerate(MIGRATIONS[:2], start=1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {number}")
    conn.execute(
        "INSERT INTO sources (sha256, name, ext, kind, size_bytes, added_at) VALUES ('a', 'a.pdf', '.pdf', 'document', 1, ?)",
        (now(),),
    )
    conn.execute(
        "INSERT INTO jobs (task, source_id, status, created_at) VALUES ('extract', 1, 'done', ?)",
        (now(),),
    )
    conn.execute(
        "INSERT INTO extractions (source_id, job_id, kind, format, content, valid, model, created_at)"
        " VALUES (1, 1, 'form', 'json', '{}', 1, 'm', ?)",
        (now(),),
    )
    assert migrate(conn) == len(MIGRATIONS) >= 3  # v2 → latest (v3 rebuilt jobs)
    assert conn.execute("SELECT job_id FROM extractions").fetchone()[0] == 1
    assert conn.execute("SELECT task, status FROM jobs").fetchone() == ("extract", "done")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1  # back on
    with pytest.raises(sqlite3.IntegrityError):  # an ingest job still needs a source
        conn.execute("INSERT INTO jobs (task, created_at) VALUES ('ingest', ?)", (now(),))


def test_verify_jobs_have_no_source_and_only_their_worker_claims_them(store: Store) -> None:
    store.add_file(DOCS / "invoice.png")
    job = store.enqueue_verify()
    assert job.source_id is None and store.enqueue_verify().id == job.id  # one queued at a time
    assert store.queued_count() == 2 and store.queued_count(("verify",)) == 1
    ingest = store.claim_next()
    assert ingest is not None and ingest.task == "ingest"
    assert store.claim_next() is None  # the ingest worker never takes a verify job
    claimed = store.claim_next(tasks=("verify",))
    assert claimed is not None and claimed.id == job.id
    with pytest.raises(ValueError, match="has no source"):
        _ = claimed.source
    rows = views.job_rows(store.job_rows())
    assert {"File": "Answers and summaries", "Task": "Check with Guardian"}.items() <= next(
        r for r in rows if r["id"] == job.id
    ).items()


# ── criteria ──


@pytest.mark.parametrize(
    ("yes_means", "score", "passed"),
    [("risk", "yes", False), ("risk", "no", True), ("pass", "yes", True), ("pass", "no", False)]
    + [(m, s, None) for m in ("risk", "pass") for s in (None, "", "maybe")],
)
def test_polarity(yes_means: str, score: str | None, passed: bool | None) -> None:
    c = GROUNDED if yes_means == "risk" else custom_criterion("Every action item names an owner.")
    assert c.passed(score) is passed


def test_custom_criteria_ids_follow_their_text(store: Store) -> None:
    assert custom_criterion("Names  an owner.").id == custom_criterion("Names an owner.").id
    assert custom_criterion("Names an owner.").id != custom_criterion("Names a due date.").id
    assert [c.text for c in summary_criteria(store)] == [
        "Every action item names the person responsible for it."
    ]
    assert views.save_summary_criteria(store, "Names an owner.\n\n  Lists   decisions. \n") == [
        "Names an owner.",
        "Lists decisions.",
    ]
    assert views.summary_criteria_text(store) == "Names an owner.\nLists decisions."
    views.save_summary_criteria(store, "")
    assert summary_criteria(store) == []  # cleared on purpose, not back to the defaults


def test_summary_text_shows_missing_owners() -> None:
    text = summary_text(json.dumps(SUMMARY))
    assert (
        "Decisions:\n- Ship in May" in text and "- (no owner): Book the venue (due Friday)" in text
    )


# ── the worker ──


def test_answers_are_judged_once_with_their_passages(store: Store) -> None:
    chunk_ids = ingested(store, texts=("Q2 revenue was $4.2M.", "Q3 is next."))
    first = turn(store, chunk_ids)
    turn(store, chunk_ids, declined=True)
    guardian = FakeGuardian({"A text is considered ungrounded": "<score>yes</score>"})
    store.enqueue_verify()
    version = store.corpus_version()
    report = VerifyWorker(store, judge=guardian, mlx_cache_limit_gb=None).run(log=lambda _: None)
    assert report.done and not report.failed and report.verdicts == 2  # the decline isn't judged
    (grounded_docs, relevance_docs) = [docs for _, docs in guardian.calls]
    assert grounded_docs == ["Results\nQ2 revenue was $4.2M.", "Results\nQ3 is next."]
    assert relevance_docs == []  # relevance reads only the question and answer
    rows = {r["criterion_id"]: r for r in store.verdicts(turn_ids=[first])[("turn", first)]}
    assert (rows["groundedness"]["score"], rows["groundedness"]["passed"]) == ("yes", 0)
    assert (rows["answer_relevance"]["score"], rows["answer_relevance"]["passed"]) == ("no", 1)
    assert rows["groundedness"]["model"].startswith("ibm-granite/granite-guardian-4.1-8b@")
    assert store.corpus_version() == version  # verdicts don't make Q&A reload its vectors
    store.enqueue_verify()
    assert (
        VerifyWorker(store, judge=guardian, mlx_cache_limit_gb=None)
        .run(log=lambda _: None)
        .verdicts
        == 0
    )
    assert len(guardian.calls) == 2


def test_unparseable_scores_and_removed_sources_are_errors_not_passes(store: Store) -> None:
    chunk_ids = ingested(store)
    garbled = turn(store, chunk_ids)
    orphan = turn(store, [999_999], question="Gone?")
    guardian = FakeGuardian({"A text is considered inadequate": "I think it's fine"})
    VerifyWorker(store, judge=guardian, mlx_cache_limit_gb=None).verify(None, log=lambda _: None)
    found = store.verdicts(turn_ids=[garbled, orphan])
    by = {(k[1], r["criterion_id"]): r for k, rows in found.items() for r in rows}
    assert by[(garbled, "answer_relevance")]["passed"] is None
    assert by[(garbled, "answer_relevance")]["error"] == "unparseable score"
    assert by[(orphan, "groundedness")]["error"] == SOURCES_REMOVED
    assert len(guardian.calls) == 3  # both checks of the first answer; only relevance of the orphan


def test_summaries_are_judged_on_the_librarys_criteria(store: Store) -> None:
    source, extraction_id = summary(store)
    guardian = FakeGuardian({"Lists decisions": "<score>yes</score>"})  # anything else: "no"
    worker = VerifyWorker(store, judge=guardian, mlx_cache_limit_gb=None)
    assert worker.verify(None, log=lambda _: None) == 1
    assert views.summary_checks(store, extraction_id) == [
        {"Check": "Every action item names the person responsible for it.", "Result": "not met"}
    ]
    views.save_summary_criteria(
        store, "Every action item names the person responsible for it.\nLists decisions."
    )
    assert views.summary_checks(store, extraction_id)[1]["Result"] == "not checked yet"
    assert worker.verify(None, log=lambda _: None) == 1  # only the new criterion
    assert (
        views.summary_checks(store, extraction_id)[1]["Result"] == "met"
    )  # custom criteria: "yes" means met
    store.delete_source(source)
    assert store.verdicts(extraction_ids=[extraction_id]) == {}  # cascaded


def test_a_failing_judge_fails_the_job_and_is_retried(store: Store) -> None:
    turn(store, ingested(store))

    def broken(messages: list[dict[str, Any]], documents: list[str]) -> str:
        raise RuntimeError("GPU out of memory")

    store.enqueue_verify()
    lines: list[str] = []
    report = VerifyWorker(store, judge=broken, mlx_cache_limit_gb=None).run(log=lines.append)
    assert report.failed and "GPU out of memory" in lines[-1]
    assert store.queued_count(("verify",)) == 1  # back in the queue for the next batch


# ── Phase C in the phase manager ──


class World:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.running = False

    def server(self) -> Any:
        world = self

        class Server:
            def healthy(self) -> bool:
                return False

            def start(self) -> None:
                assert not world.running
                world.events.append("server.start")

            def wait_ready(self) -> None:
                pass

            def warm_up(self) -> None:
                pass

            def stop(self) -> None:
                world.events.append("server.stop")

        return Server()

    def worker(self, name: str, store: Store) -> Any:
        def run(data_dir: Path, log: Any) -> dict[str, Any]:
            self.events.append(name)
            tasks = ("verify",) if name == "verify" else ("ingest", "extract")
            while (job := store.claim_next(tasks=tasks)) is not None:
                store.complete_verify(job)  # marks it done; enough for the manager
            return {"done": [], "failed": []}

        return run


def test_verify_jobs_run_guardian_not_the_ingest_worker(store: Store) -> None:
    world = World()
    pm = PhaseManager(
        store,
        server_factory=world.server,
        run_worker=world.worker("ingest", store),
        run_verifier=world.worker("verify", store),
        idle_before_ingest_s=0,
    )
    pm.start()
    store.enqueue_verify()
    result = pm.process_now()
    assert result is not None and "verify" in result
    assert world.events == ["server.start", "server.stop", "verify", "server.start"]
    assert pm.status().phase is Phase.QA and "verify_s" in pm.status().last_switch


def test_ingest_and_verify_share_one_switch_ingest_first(store: Store) -> None:
    world = World()
    pm = PhaseManager(
        store,
        server_factory=world.server,
        run_worker=world.worker("ingest", store),
        run_verifier=world.worker("verify", store),
    )
    pm.start()
    store.enqueue_verify()
    store.add_file(DOCS / "invoice.png")
    pm.process_now()
    assert world.events == ["server.start", "server.stop", "ingest", "verify", "server.start"]


# ── badges ──


def verdict(criterion: str, passed: int | None, error: str | None = None) -> dict[str, Any]:
    return {
        "criterion_id": criterion,
        "passed": passed,
        "error": error,
        "created_at": "2026-10-07T10:00:00+00:00",
    }


def test_badges_show_groundedness_only() -> None:
    def labels(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(c.label, c.color) for c in views.turn_checks(rows)]

    assert labels([verdict("groundedness", 1), verdict("answer_relevance", 0)]) == [
        ("Grounded", "green")
    ]  # relevance is recorded, not shown (unvalidated)
    assert labels([verdict("groundedness", 0)]) == [("Unsupported claims", "orange")]
    (failed,) = views.turn_checks([verdict("groundedness", None, SOURCES_REMOVED)])
    assert (failed.color, failed.help) == ("gray", SOURCES_REMOVED)
    assert views.turn_checks([verdict("answer_relevance", 1)]) == []


def test_history_turns_get_their_badges(store: Store) -> None:
    first = turn(store, ingested(store))
    turns = views.turns_from_history(store)
    views.attach_checks(store, turns)
    assert turns[0].checks == [] and views.unchecked_answers(store) == 1
    VerifyWorker(store, judge=FakeGuardian(), mlx_cache_limit_gb=None).verify(
        None, log=lambda _: None
    )
    views.attach_checks(store, turns)
    assert [c.label for c in turns[0].checks] == ["Grounded"] and turns[0].turn_id == first
    assert views.unchecked_answers(store) == 0
    assert RELEVANT.id in {
        r["criterion_id"] for r in store.verdicts(turn_ids=[first])[("turn", first)]
    }
