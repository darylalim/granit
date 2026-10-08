"""SQLite store (PLAN.md §2.3): settings, schema, files by hash, the job queue and one transaction per job."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from granit.store import db
from granit.store.db import NewChunk, NewExtraction, Store
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures"
AUDIO = FIXTURES / "audio" / "vad_pauses.wav"
DOC = FIXTURES / "documents" / "invoice.png"
DOCS = FIXTURES / "documents"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "data")


def vectors(n: int) -> np.ndarray:
    v = np.random.default_rng(n).standard_normal((n, 768)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def chunks(n: int) -> list[NewChunk]:
    return [
        NewChunk(f"chunk {i} about invoice INV-2026-00{i}", "text", "Heading", 1, 1)
        for i in range(n)
    ]


def test_connection_settings(store: Store) -> None:
    pragma = lambda name: store.conn.execute(f"PRAGMA {name}").fetchone()[0]  # noqa: E731
    assert pragma("journal_mode") == "wal"
    assert pragma("foreign_keys") == 1
    assert pragma("busy_timeout") == 5000
    assert pragma("synchronous") == 1  # NORMAL
    assert pragma("user_version") == len(db.MIGRATIONS)


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    first = Store(tmp_path)
    first.close()
    again = Store(tmp_path)  # reopening must not re-run migration 1
    assert (
        again.conn.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()[0]
        == "0"
    )


def test_add_file_stores_by_hash_and_queues_one_ingest(store: Store) -> None:
    source, created = store.add_file(AUDIO)
    assert created and source.kind == "audio" and source.status == "queued"
    stored = store.file_path(source)
    assert stored.name == f"{source.sha256}.wav" and stored.read_bytes() == AUDIO.read_bytes()
    assert [(j.task, j.status, j.source_id) for j in store.jobs()] == [
        ("ingest", "queued", source.id)
    ]

    again, created = store.add_file(AUDIO, name="renamed.wav")
    assert not created and again.id == source.id
    assert len(store.jobs()) == 1  # duplicates aren't re-queued


def test_ingest_params_and_reingest(store: Store) -> None:
    source, _ = store.add_file(DOCS / "memo.pdf", params={"vision_tables": True})
    (job,) = store.jobs()
    assert job.params == {"vision_tables": True}
    # Asking again while that ingest is still queued doesn't add a second one.
    assert store.queue_ingest(source, {"vision_tables": False}).id == job.id
    claimed = store.claim_next()
    assert claimed is not None
    store.complete_ingest(claimed, [], np.zeros((0, 768), np.float16), "rev")
    again = store.queue_ingest(source, {"vision_tables": True})
    assert (
        again.id != job.id and again.params == {"vision_tables": True} and again.status == "queued"
    )
    assert (
        store.source(source.id).status == "ready"
    )  # still searchable until the new chunks replace the old ones


def test_job_rows_put_open_jobs_first(store: Store) -> None:
    first, _ = store.add_file(DOCS / "memo.pdf")
    store.add_file(DOCS / "invoice.png")
    job = store.claim_next()
    assert job is not None and job.source_id == first.id
    store.complete_ingest(job, [], np.zeros((0, 768), np.float16), "rev")
    store.queue_ingest(store.source(first.id))
    rows = store.job_rows()
    assert [(r["source_name"], r["status"]) for r in rows] == [
        ("memo.pdf", "queued"),
        ("invoice.png", "queued"),
        ("memo.pdf", "done"),
    ]


def test_a_shared_store_works_across_threads(tmp_path: Path) -> None:
    import threading

    shared = Store(tmp_path / "data", shared=True)
    counts: list[int] = []
    thread = threading.Thread(target=lambda: counts.append(shared.queued_count()))
    thread.start()
    thread.join()
    assert counts == [0]


def test_unknown_file_types_are_rejected(store: Store, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    with pytest.raises(ValueError, match="not an audio file or document"):
        store.add_file(path)
    assert db.kind_of(Path("a.M4A")) == "audio" and db.kind_of(Path("a.pdf")) == "document"


def test_claim_fail_retry_and_attempt_limit(store: Store) -> None:
    source, _ = store.add_file(DOC)
    for attempt in range(1, db.MAX_ATTEMPTS + 1):
        job = store.claim_next()
        assert job is not None and job.status == "running" and job.attempts == attempt
        assert store.source(source.id).status == "processing"
        store.fail(job, f"boom {attempt}")
    assert store.claim_next() is None  # failed for good
    assert (
        store.job(job.id).status == "failed"
        and store.job(job.id).error == f"boom {db.MAX_ATTEMPTS}"
    )
    assert store.source(source.id).status == "failed"
    store.retry(job.id)
    retried = store.claim_next()
    assert retried is not None and retried.attempts == 1


def test_claim_next_can_skip_jobs(store: Store) -> None:
    store.add_file(DOC)
    store.add_file(AUDIO)
    first = store.claim_next()
    assert first is not None
    store.fail(first, "boom")  # back in the queue
    second = store.claim_next(exclude=[first.id])
    assert second is not None and second.id != first.id
    assert store.claim_next(exclude=[first.id]) is None


def test_recover_interrupted_requeues_running_jobs(store: Store) -> None:
    store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    assert store.recover_interrupted() == 1  # a worker died mid-job
    assert store.job(job.id).status == "queued"
    assert store.claim_next() is not None


def test_complete_ingest_writes_everything_in_one_transaction(store: Store) -> None:
    source, _ = store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    extraction = NewExtraction("chart", "csv", "a,b\n1,2\n", True, "vision@rev", page=1)
    ids = store.complete_ingest(job, chunks(3), vectors(3), "rev1", [extraction], {"pages": 1})
    assert len(ids) == 3
    assert store.job(job.id).status == "done"
    assert store.source(source.id).status == "ready" and store.source(source.id).info == {
        "pages": 1
    }
    assert (
        store.conn.execute(
            "SELECT count(*) FROM chunk_vectors WHERE model_revision = 'rev1'"
        ).fetchone()[0]
        == 3
    )
    assert store.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 3
    assert [e["kind"] for e in store.extractions(source.id)] == ["chart"]
    assert store.corpus_version() == 1
    blob = store.conn.execute("SELECT embedding FROM chunk_vectors LIMIT 1").fetchone()[0]
    assert len(blob) == 768 * 2  # float16


def test_a_failure_mid_transaction_leaves_nothing_behind(store: Store) -> None:
    store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    bad = [*chunks(2), NewChunk("x", cast(str, None))]  # NOT NULL violation on the third chunk
    with pytest.raises(sqlite3.IntegrityError):
        store.complete_ingest(job, bad, vectors(3), "rev1")
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    assert store.job(job.id).status == "running"  # the worker then records the failure
    assert store.corpus_version() == 0


def test_mismatched_vectors_are_rejected_before_writing(store: Store) -> None:
    store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    with pytest.raises(ValueError, match="3 chunks but 2 vectors"):
        store.complete_ingest(job, chunks(3), vectors(2), "rev1")


def test_reingest_replaces_chunks(store: Store) -> None:
    source, _ = store.add_file(DOC)
    first = store.claim_next()
    assert first is not None
    store.complete_ingest(first, chunks(3), vectors(3), "rev1")
    store.conn.execute("UPDATE jobs SET status = 'queued' WHERE id = ?", (first.id,))
    second = store.claim_next()
    assert second is not None
    store.complete_ingest(second, chunks(1), vectors(1), "rev1")
    assert (
        store.conn.execute(
            "SELECT count(*) FROM chunks WHERE source_id = ?", (source.id,)
        ).fetchone()[0]
        == 1
    )
    assert store.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1


def test_delete_source_cascades_and_keeps_indexes_in_sync(store: Store) -> None:
    source, _ = store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    store.complete_ingest(job, chunks(2), vectors(2), "rev1")
    store.derived_dir(source).mkdir(parents=True)
    store.delete_source(source)
    for table in ("sources", "jobs", "chunks", "chunk_vectors", "extractions"):
        assert store.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table
    fts = store.conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'invoice'"
    ).fetchone()[0]
    assert fts == 0
    assert not store.file_path(source).exists() and not store.derived_dir(source).exists()
    assert store.corpus_version() == 2


def test_extract_jobs(store: Store) -> None:
    source, _ = store.add_file(DOC)
    ingest = store.claim_next()
    assert ingest is not None
    store.complete_ingest(ingest, chunks(1), vectors(1), "rev1")
    job = store.enqueue_extract(source, {"type": "object", "properties": {}})
    claimed = store.claim_next()
    assert (
        claimed is not None
        and claimed.id == job.id
        and claimed.params["schema"]["type"] == "object"
    )
    store.complete_extract(
        claimed, NewExtraction("form", "json", "{}", True, "vision@rev", schema={"type": "object"})
    )
    (form,) = [e for e in store.extractions(source.id) if e["kind"] == "form"]
    assert form["valid"] == 1 and form["schema"] == '{"type": "object"}'
    assert store.source(source.id).status == "ready"  # extract jobs don't touch the source status


def test_vector_blob_requires_one_dimension() -> None:
    assert len(db.vector_blob(np.zeros(4))) == 8
    with pytest.raises(ValueError):
        db.vector_blob(np.zeros((2, 2)))


def test_vocabulary_is_saved_for_the_library(store: Store) -> None:
    assert store.vocabulary() == []
    store.set_vocabulary(["Northbeam", "Priya Natarajan"])
    store.set_vocabulary(["Northbeam", "Priya", "Zoë"])  # replaced, not appended
    assert store.vocabulary() == ["Northbeam", "Priya", "Zoë"]
    assert Store(store.root).vocabulary() == ["Northbeam", "Priya", "Zoë"]  # persisted


# ── speaker names (M9, PLAN.md §3.7) ──


def test_speaker_names_replace_trim_and_unname(store: Store) -> None:
    source, _ = store.add_file(DOC)
    assert store.speaker_names(source.id) == {}
    stored = store.set_speaker_names(source, {1: "  Priya  Raman ", 2: "Sam", 3: "", 4: "   "})
    assert stored == {1: "Priya Raman", 2: "Sam"}
    assert store.speaker_names(source.id) == stored
    store.set_speaker_names(source, {2: "Sam", 3: "Sam"})  # same name: one person
    assert store.speaker_names(source.id) == {2: "Sam", 3: "Sam"}
    with pytest.raises(ValueError, match="numbered from 1"):
        store.set_speaker_names(source, {0: "Nobody"})


def test_reingest_clears_speaker_names_and_deleting_cascades(store: Store) -> None:
    source, _ = store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    store.complete_ingest(job, [], np.zeros((0, 768), np.float16), "rev")
    store.set_speaker_names(source, {1: "Elena"})
    again = store.queue_ingest(store.source(source.id))
    assert store.speaker_names(source.id) == {1: "Elena"}  # kept until the new transcript lands
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == again.id
    store.complete_ingest(claimed, [], np.zeros((0, 768), np.float16), "rev")
    assert store.speaker_names(source.id) == {}  # speakers were renumbered
    store.set_speaker_names(source, {1: "Elena"})
    store.delete_source(store.source(source.id))
    assert store.conn.execute("SELECT COUNT(*) FROM speaker_names").fetchone()[0] == 0


def test_recordings_named_before_m10_are_reindexed_once(tmp_path: Path) -> None:
    """Migration 5 (PLAN.md §3.8): names stored under M9 never reached search, and saving them again changes nothing."""
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    for number, script in enumerate(db.MIGRATIONS[:4], start=1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {number}")
    for sha in ("a", "b"):
        conn.execute(
            "INSERT INTO sources (sha256, name, ext, kind, size_bytes, added_at)"
            " VALUES (?, ?, '.wav', 'audio', 1, ?)",
            (sha, f"{sha}.wav", db.now()),
        )
    conn.executemany(
        "INSERT INTO speaker_names (source_id, speaker, name) VALUES (1, ?, ?)",
        [(1, "Elena"), (2, "Priya")],
    )
    assert db.migrate(conn) == len(db.MIGRATIONS) == 5
    assert conn.execute("SELECT task, source_id, status FROM jobs").fetchall() == [
        ("reindex", 1, "queued")
    ]
