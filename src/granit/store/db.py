"""SQLite store (PLAN.md §2.3): schema, connection settings, the job queue and one transaction per job.

- Every connection sets WAL mode, ``busy_timeout``, foreign keys and ``synchronous=NORMAL``, so the UI can read while the
  ingest worker writes, across processes.
- ``complete_ingest`` writes a source's chunks, FTS rows (via triggers), vectors, extractions and ``jobs.status='done'`` in
  **one transaction**: a crash leaves no half-ingested source, and the job is retried.
- Originals are stored once by content hash in ``files/<sha256><ext>``; derived outputs go to ``derived/<source_id>/``.
- The schema is versioned with ``PRAGMA user_version`` and upgraded by ``MIGRATIONS`` on open.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from granit.store.text import search_text

MAX_ATTEMPTS = 3  # a job that fails (or whose worker crashes) this many times stays failed

MIGRATIONS: list[str] = [
    # 1: M4 store
    """
    CREATE TABLE sources (
        id INTEGER PRIMARY KEY,
        sha256 TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        ext TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('audio', 'document')),
        size_bytes INTEGER NOT NULL,
        added_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued'
            CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
        info TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY,
        task TEXT NOT NULL CHECK (task IN ('ingest', 'extract')),
        source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        params TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE INDEX jobs_by_status ON jobs (status, id);
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        text TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT '',
        search_text TEXT NOT NULL,
        element TEXT NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        start_s REAL,
        end_s REAL,
        UNIQUE (source_id, seq)
    );
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        search_text, content='chunks', content_rowid='id', tokenize="unicode61 tokenchars '-_./'"
    );
    CREATE VIRTUAL TABLE chunks_trigram USING fts5(
        text, content='chunks', content_rowid='id', tokenize='trigram'
    );
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts (rowid, search_text) VALUES (new.id, new.search_text);
        INSERT INTO chunks_trigram (rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
        INSERT INTO chunks_trigram (chunks_trigram, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    CREATE TABLE chunk_vectors (
        chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        model_revision TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
    CREATE TABLE extractions (
        id INTEGER PRIMARY KEY,
        source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
        kind TEXT NOT NULL,
        format TEXT NOT NULL,
        content TEXT NOT NULL,
        schema TEXT,
        valid INTEGER NOT NULL,
        errors TEXT NOT NULL DEFAULT '[]',
        missing TEXT NOT NULL DEFAULT '[]',
        page INTEGER,
        crop TEXT,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO meta (key, value) VALUES ('corpus_version', '0');
    """,
    # 2: M5 Q&A history (PLAN.md §2.4, §4.9): every answer with its sources, citations and per-stage retrieval trace
    """
    CREATE TABLE qa_turns (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        declined INTEGER NOT NULL,
        source_chunk_ids TEXT NOT NULL,
        cited_chunk_ids TEXT NOT NULL,
        model TEXT NOT NULL,
        thinking TEXT NOT NULL,
        retrieval TEXT NOT NULL,
        retrieval_trace TEXT NOT NULL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        latency TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the database with granit's settings and bring the schema up to date."""
    conn = sqlite3.connect(
        path, timeout=5.0, isolation_level=None, check_same_thread=check_same_thread
    )  # autocommit; transactions are explicit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        with transaction(conn):
            for statement in _statements(script):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {number}")
    return len(MIGRATIONS)


def _statements(script: str) -> Iterator[str]:
    """Split a migration script into statements (``sqlite3.complete_statement`` handles trigger bodies)."""
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                yield buffer.strip()
            buffer = ""
    if buffer.strip():
        raise ValueError(f"incomplete SQL statement in migration: {buffer.strip()[:80]}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, or ``ROLLBACK`` on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# ── records ──


@dataclass(frozen=True)
class Source:
    id: int
    sha256: str
    name: str
    ext: str
    kind: str
    size_bytes: int
    added_at: str
    status: str
    info: dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Source:
        return cls(**{**dict(row), "info": json.loads(row["info"])})


@dataclass(frozen=True)
class Job:
    id: int
    task: str
    source_id: int
    params: dict[str, Any]
    status: str
    attempts: int
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(**{**dict(row), "params": json.loads(row["params"])})


@dataclass(frozen=True)
class NewChunk:
    """A chunk to store. ``text`` is what's shown and cited; ``context`` (headings) is prepended for search and embedding."""

    text: str
    element: str  # text | table | chart | speech | mixed
    context: str = ""
    page_start: int | None = None
    page_end: int | None = None
    start_s: float | None = None
    end_s: float | None = None

    @property
    def search_body(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass(frozen=True)
class NewExtraction:
    kind: str
    format: str
    content: str
    valid: bool
    model: str
    errors: Sequence[str] = field(default=())
    missing: Sequence[str] = field(default=())
    schema: dict[str, Any] | None = None
    page: int | None = None
    crop: str | None = None


# ── the store ──

AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".caf", ".mp4"}
)
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    raise ValueError(
        f"{path.name}: not an audio file or document granit can ingest ({ext or 'no extension'})"
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Store:
    """``data/`` on disk: ``granit.db``, ``files/`` and ``derived/`` (PLAN.md §2.3).

    ``shared=True`` lets one connection be used from several threads (the UI backend: SQLite here is serialized-threadsafe);
    the caller must still keep its write transactions from overlapping (the backend does them under one lock).
    """

    def __init__(self, root: Path | str, shared: bool = False) -> None:
        self.root = Path(root)
        (self.root / "files").mkdir(parents=True, exist_ok=True)
        (self.root / "derived").mkdir(exist_ok=True)
        self.conn = connect(self.root / "granit.db", check_same_thread=not shared)

    def close(self) -> None:
        self.conn.close()

    def file_path(self, source: Source) -> Path:
        return self.root / "files" / f"{source.sha256}{source.ext}"

    def derived_dir(self, source: Source) -> Path:
        return self.root / "derived" / str(source.id)

    # sources and jobs

    def add_file(
        self, path: Path | str, name: str | None = None, params: dict[str, Any] | None = None
    ) -> tuple[Source, bool]:
        """Store a file by content hash and queue an ingest job. Returns (source, created); duplicates aren't re-added.

        ``params`` go to the ingest job, e.g. ``{"vision_tables": True}`` ("Accurate tables", PLAN.md §3.6).
        """
        path = Path(path)
        kind = kind_of(path)
        sha = sha256_of(path)
        existing = self.conn.execute("SELECT * FROM sources WHERE sha256 = ?", (sha,)).fetchone()
        if existing:
            return Source.from_row(existing), False
        ext = path.suffix.lower()
        target = self.root / "files" / f"{sha}{ext}"
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(path, tmp)
            tmp.replace(target)  # atomic: a half-copied file never has the final name
        with transaction(self.conn):
            cursor = self.conn.execute(
                "INSERT INTO sources (sha256, name, ext, kind, size_bytes, added_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sha, name or path.name, ext, kind, target.stat().st_size, now()),
            )
            source_id = cursor.lastrowid
            if source_id is None:
                raise RuntimeError("INSERT INTO sources returned no row id")
            self._enqueue(source_id, "ingest", params or {})
        return self.source(source_id), True

    def queue_ingest(self, source: Source, params: dict[str, Any] | None = None) -> Job:
        """Ingest a stored file again (e.g. with accurate tables). Its current chunks stay searchable until the new ones
        replace them in one transaction. A queued or running ingest of the same source is returned instead of a second one."""
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT id FROM jobs WHERE source_id = ? AND task = 'ingest' AND status IN ('queued', 'running')",
                (source.id,),
            ).fetchone()
            job_id = row[0] if row else self._enqueue(source.id, "ingest", params or {})
        return self.job(job_id)

    def enqueue_extract(self, source: Source, schema: dict[str, Any]) -> Job:
        with transaction(self.conn):
            job_id = self._enqueue(source.id, "extract", {"schema": schema})
        return self.job(job_id)

    def _enqueue(self, source_id: int, task: str, params: dict[str, Any]) -> int:
        cursor = self.conn.execute(
            "INSERT INTO jobs (task, source_id, params, created_at) VALUES (?, ?, ?, ?)",
            (task, source_id, json.dumps(params), now()),
        )
        return int(cursor.lastrowid or 0)

    def source(self, source_id: int) -> Source:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"no source {source_id}")
        return Source.from_row(row)

    def sources(self) -> list[Source]:
        return [Source.from_row(r) for r in self.conn.execute("SELECT * FROM sources ORDER BY id")]

    def job(self, job_id: int) -> Job:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no job {job_id}")
        return Job.from_row(row)

    def jobs(self, status: str | None = None) -> list[Job]:
        if status:
            rows = self.conn.execute("SELECT * FROM jobs WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self.conn.execute("SELECT * FROM jobs ORDER BY id")
        return [Job.from_row(r) for r in rows]

    def job_rows(self, limit: int = 50) -> list[sqlite3.Row]:
        """Jobs for the UI, newest first: queued and running ones always, then the most recent finished ones."""
        return list(
            self.conn.execute(
                "SELECT j.*, s.name AS source_name, s.kind AS source_kind FROM jobs j JOIN sources s ON s.id = j.source_id"
                " ORDER BY j.status IN ('queued', 'running') DESC, j.id DESC LIMIT ?",
                (limit,),
            )
        )

    def queued_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM jobs WHERE status = 'queued'").fetchone()[0]

    def recover_interrupted(self) -> int:
        """Jobs left ``running`` by a worker that died go back to the queue (or fail after ``MAX_ATTEMPTS``)."""
        with transaction(self.conn):
            failed = self.conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'worker stopped while running (max attempts reached)',"
                " finished_at = ? WHERE status = 'running' AND attempts >= ?",
                (now(), MAX_ATTEMPTS),
            ).rowcount
            requeued = self.conn.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
            ).rowcount
            self.conn.execute(
                "UPDATE sources SET status = 'queued' WHERE status = 'processing'"
                " AND id IN (SELECT source_id FROM jobs WHERE status = 'queued')"
            )
        return failed + requeued

    def claim_next(self, exclude: Sequence[int] = ()) -> Job | None:
        """Atomically take the oldest queued job (``running``, attempts + 1), skipping ``exclude`` (failed this run)."""
        # Never "NOT IN (NULL)": a comparison with NULL is unknown, so it would match no job at all.
        skip = f" AND id NOT IN ({','.join(str(int(i)) for i in exclude)})" if exclude else ""
        with transaction(self.conn):
            row = self.conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, attempts = attempts + 1"
                " WHERE id = (SELECT id FROM jobs WHERE status = 'queued'"
                f"{skip} ORDER BY id LIMIT 1) RETURNING *",
                (now(),),
            ).fetchone()
            if row is None:
                return None
            job = Job.from_row(row)
            if job.task == "ingest":
                self.conn.execute(
                    "UPDATE sources SET status = 'processing' WHERE id = ?", (job.source_id,)
                )
        return job

    def fail(self, job: Job, error: str) -> None:
        """Record a failure; the job is re-queued until it has failed ``MAX_ATTEMPTS`` times."""
        final = job.attempts >= MAX_ATTEMPTS
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if final else "queued", error[:2000], now() if final else None, job.id),
            )
            if job.task == "ingest":
                self.conn.execute(
                    "UPDATE sources SET status = ? WHERE id = ?",
                    ("failed" if final else "queued", job.source_id),
                )

    def retry(self, job_id: int) -> None:
        """Re-queue a failed job by hand (resets its attempt count)."""
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE jobs SET status = 'queued', attempts = 0, error = NULL, finished_at = NULL"
                " WHERE id = ? AND status = 'failed'",
                (job_id,),
            )

    # the per-job transaction

    def complete_ingest(
        self,
        job: Job,
        chunks: Sequence[NewChunk],
        vectors: Any,  # (len(chunks), dim) float16 array, rows normalized
        model_revision: str,
        extractions: Sequence[NewExtraction] = (),
        info: dict[str, Any] | None = None,
    ) -> list[int]:
        """Everything an ingest produced, plus ``done``, in one transaction (replacing any earlier ingest)."""
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with transaction(self.conn):
            self.conn.execute("DELETE FROM chunks WHERE source_id = ?", (job.source_id,))
            self.conn.execute(
                "DELETE FROM extractions WHERE source_id = ? AND kind != 'form'", (job.source_id,)
            )
            ids = []
            for seq, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                cursor = self.conn.execute(
                    "INSERT INTO chunks (source_id, seq, text, context, search_text, element, page_start, page_end,"
                    " start_s, end_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job.source_id,
                        seq,
                        chunk.text,
                        chunk.context,
                        search_text(chunk.search_body),
                        chunk.element,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.start_s,
                        chunk.end_s,
                    ),
                )
                chunk_id = int(cursor.lastrowid or 0)
                ids.append(chunk_id)
                self.conn.execute(
                    "INSERT INTO chunk_vectors (chunk_id, model_revision, embedding) VALUES (?, ?, ?)",
                    (chunk_id, model_revision, vector_blob(vector)),
                )
            self._insert_extractions(job.id, job.source_id, extractions)
            self.conn.execute(
                "UPDATE sources SET status = 'ready', info = ? WHERE id = ?",
                (json.dumps(info or {}), job.source_id),
            )
            self._finish(job)
        return ids

    def complete_extract(self, job: Job, extraction: NewExtraction) -> None:
        with transaction(self.conn):
            self._insert_extractions(job.id, job.source_id, [extraction])
            self._finish(job)

    def _insert_extractions(
        self, job_id: int | None, source_id: int, extractions: Sequence[NewExtraction]
    ) -> None:
        for e in extractions:
            self.conn.execute(
                "INSERT INTO extractions (source_id, job_id, kind, format, content, schema, valid, errors, missing,"
                " page, crop, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    job_id,
                    e.kind,
                    e.format,
                    e.content,
                    json.dumps(e.schema) if e.schema is not None else None,
                    int(e.valid),
                    json.dumps(list(e.errors)),
                    json.dumps(list(e.missing)),
                    e.page,
                    e.crop,
                    e.model,
                    now(),
                ),
            )

    def _finish(self, job: Job) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = 'done', error = NULL, finished_at = ? WHERE id = ?",
            (now(), job.id),
        )
        # Phase B reloads its vector matrix when this changes (PLAN.md §2.3).
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'"
        )

    # Q&A history and Phase B outputs

    def record_turn(
        self,
        *,
        question: str,
        answer: str,
        declined: bool,
        source_chunk_ids: Sequence[int],
        cited_chunk_ids: Sequence[int],
        model: str,
        thinking: str,
        retrieval: dict[str, Any],
        retrieval_trace: dict[str, Any],
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency: dict[str, float],
    ) -> int:
        with transaction(self.conn):
            cursor = self.conn.execute(
                "INSERT INTO qa_turns (question, answer, declined, source_chunk_ids, cited_chunk_ids, model, thinking,"
                " retrieval, retrieval_trace, prompt_tokens, completion_tokens, latency, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question,
                    answer,
                    int(declined),
                    json.dumps(list(source_chunk_ids)),
                    json.dumps(list(cited_chunk_ids)),
                    model,
                    thinking,
                    json.dumps(retrieval),
                    json.dumps(retrieval_trace),
                    prompt_tokens,
                    completion_tokens,
                    json.dumps(latency),
                    now(),
                ),
            )
        return int(cursor.lastrowid or 0)

    def turns(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM qa_turns ORDER BY id DESC LIMIT ?", (limit,)))

    def replace_extraction(self, source: Source, extraction: NewExtraction) -> None:
        """Store a Phase B result (e.g. a meeting summary), replacing an earlier one of the same kind."""
        with transaction(self.conn):
            self.conn.execute(
                "DELETE FROM extractions WHERE source_id = ? AND kind = ?",
                (source.id, extraction.kind),
            )
            self._insert_extractions(None, source.id, [extraction])

    # reads used by search

    def corpus_version(self) -> int:
        return int(
            self.conn.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()[0]
        )

    def chunks_by_id(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT c.*, s.name AS source_name, s.kind AS source_kind FROM chunks c"
            f" JOIN sources s ON s.id = c.source_id WHERE c.id IN ({marks})",
            list(ids),
        )
        return {row["id"]: row for row in rows}

    def extractions(self, source_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM extractions WHERE source_id = ? ORDER BY id", (source_id,)
            )
        )

    def extractions_for_job(self, job_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM extractions WHERE job_id = ? ORDER BY id", (job_id,))
        )

    def delete_source(self, source: Source) -> None:
        """Remove a source, its chunks, vectors, extractions and jobs; then its files."""
        with transaction(self.conn):
            self.conn.execute("DELETE FROM sources WHERE id = ?", (source.id,))
            self.conn.execute(
                "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'"
            )
        self.file_path(source).unlink(missing_ok=True)
        shutil.rmtree(self.derived_dir(source), ignore_errors=True)


def vector_blob(vector: Any) -> bytes:
    """A normalized embedding as float16 bytes (768 × 2 = 1.5 KB)."""
    import numpy as np

    array = np.asarray(vector, dtype=np.float16)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {array.shape}")
    return array.tobytes()
