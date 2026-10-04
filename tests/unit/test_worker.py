"""The Phase A ingest worker with fake models (M4): one transaction per job, failures recorded, the queue drained."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from granit.ingest.audio import Segment, Transcript
from granit.ingest.vision import Extraction
from granit.ingest.worker import IngestWorker
from granit.store.db import Store
from tests.conftest import ROOT
from tests.unit.test_search import FakeEmbedder

FIXTURES = ROOT / "tests" / "fixtures"
REPORT_JSON = json.loads((FIXTURES / "documents" / "report.document.json").read_text())


class FakeTranscriber:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def transcribe(self, path: Path) -> Transcript:
        if self.fail:
            raise RuntimeError("decoder exploded")
        seg = Segment(1.0, 3.5, "please send the revised invoice to finance")
        return Transcript(12.7, 6.4, 1, (seg,), "speech@rev")


class FakeDocuments:
    def __init__(self) -> None:
        self.vision = SimpleNamespace(extract_fields=self.extract_fields)

    def ingest(self, path: Path, out: Path) -> Any:
        chart = Extraction(
            "chart",
            "csv",
            "Quarter,Revenue\nQ1,120\n",
            True,
            model="vision@rev",
            page=2,
            crop="crops/p2_picture1.png",
        )
        return SimpleNamespace(
            document=REPORT_JSON,
            extractions=[chart],
            summary=lambda: {"pages": 2, "charts": 1},
        )

    def extract_fields(self, pages: list[Any], schema: dict[str, Any]) -> Extraction:
        return Extraction(
            "form",
            "json",
            '{"total": "$4,980.00"}',
            True,
            data={"total": "$4,980.00"},
            missing=("po",),
            model="vision@rev",
        )


def worker(store: Store, **kw: Any) -> IngestWorker:
    return IngestWorker(
        store,
        transcriber=kw.get("transcriber", FakeTranscriber()),
        documents=FakeDocuments(),
        embedder=FakeEmbedder(),
        mlx_cache_limit_gb=None,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "data")


def test_worker_ingests_audio_and_documents(store: Store) -> None:
    audio, _ = store.add_file(FIXTURES / "audio" / "vad_pauses.wav")
    doc, _ = store.add_file(FIXTURES / "documents" / "report.pdf")
    logs: list[str] = []
    report = worker(store).run(log=logs.append)
    assert len(report.done) == 2 and not report.failed and store.queued_count() == 0

    audio, doc = store.source(audio.id), store.source(doc.id)
    assert audio.status == doc.status == "ready"
    assert audio.info == {
        "duration_s": 12.7,
        "speech_s": 6.4,
        "silence_s": 6.3,
        "segments": 1,
        "chunks": 1,
    }
    assert doc.info["chunks"] == 5 and doc.info["charts"] == 1
    assert (store.derived_dir(audio) / "transcript.json").is_file()

    rows = store.conn.execute(
        "SELECT element, start_s, page_start FROM chunks ORDER BY id"
    ).fetchall()
    assert [r["element"] for r in rows] == ["speech", "text", "table", "text", "chart", "text"]
    assert rows[0]["start_s"] == 1.0 and rows[4]["page_start"] == 2
    assert (
        store.conn.execute(
            "SELECT count(*) FROM chunk_vectors WHERE model_revision = 'fake-rev'"
        ).fetchone()[0]
        == 6
    )
    (chart,) = store.extractions(doc.id)
    assert chart["kind"] == "chart" and chart["page"] == 2 and chart["model"] == "vision@rev"
    assert any(line.startswith("✓ ingest report.pdf") for line in logs)


def test_a_failing_job_is_recorded_and_the_worker_moves_on(store: Store) -> None:
    store.add_file(FIXTURES / "audio" / "vad_pauses.wav")
    doc, _ = store.add_file(FIXTURES / "documents" / "report.pdf")
    report = worker(store, transcriber=FakeTranscriber(fail=True)).run(log=lambda _: None)
    assert len(report.done) == 1 and store.source(doc.id).status == "ready"
    # The audio job failed and was re-queued (attempt 1 of 3); a later run tries again.
    (failed_id, error) = report.failed[0]
    assert error == "RuntimeError: decoder exploded"
    assert store.job(failed_id).status == "queued" and store.job(failed_id).attempts == 1
    assert (
        store.conn.execute("SELECT count(*) FROM chunks WHERE element = 'speech'").fetchone()[0]
        == 0
    )


def test_extract_jobs_store_a_validated_form(store: Store) -> None:
    source, _ = store.add_file(FIXTURES / "documents" / "invoice.png")
    worker(store).run(log=lambda _: None)
    store.enqueue_extract(source, {"type": "object", "properties": {"total": {"type": "string"}}})
    report = worker(store).run(log=lambda _: None)
    assert len(report.done) == 1
    (form,) = [e for e in store.extractions(source.id) if e["kind"] == "form"]
    assert json.loads(form["content"]) == {"total": "$4,980.00"} and json.loads(
        form["missing"]
    ) == ["po"]


def test_jobs_left_running_are_recovered(store: Store) -> None:
    store.add_file(FIXTURES / "audio" / "vad_pauses.wav")
    assert store.claim_next() is not None  # a previous worker died here
    logs: list[str] = []
    report = worker(store).run(log=logs.append)
    assert len(report.done) == 1 and "re-queued 1 job" in logs[0]
