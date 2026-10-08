"""The Phase A ingest worker (PLAN.md §2.1, M4): load models once, drain the job queue, exit.

For each ``ingest`` job: audio → transcript (M2) or document → Docling + Vision (M3) → chunks with citations → embeddings →
**one transaction** with everything plus ``done`` (``Store.complete_ingest``). ``extract`` jobs run Granite Vision's
schema-based extraction on the document's pages. A failing job is recorded and re-queued up to ``MAX_ATTEMPTS``; the worker
moves on to the next one. Jobs left ``running`` by a worker that died are re-queued at start.

Models load lazily by kind (an audio-only batch never loads the 8 GB Vision model) and stay loaded for the batch. MLX keeps
freed buffers cached; the worker caps that cache (M1: Phase A peaked at 15.6 GB, mostly cached buffers).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granit.config import DOCUMENT_VISION_TABLES, INGEST_MLX_CACHE_LIMIT_GB
from granit.store.chunking import chunk_document, chunk_transcript
from granit.store.db import Job, NewChunk, NewExtraction, Store

MAX_FORM_PAGES = 4


@dataclass
class WorkerReport:
    done: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    seconds: float = 0.0
    peak_footprint_gb: float = 0.0


class IngestWorker:
    def __init__(
        self,
        store: Store,
        transcriber: Any = None,
        documents: Any = None,
        embedder: Any = None,
        mlx_cache_limit_gb: float | None = INGEST_MLX_CACHE_LIMIT_GB,
    ) -> None:
        self.store = store
        self._transcriber = transcriber
        self._documents = documents
        self._embedder = embedder
        self.mlx_cache_limit_gb = mlx_cache_limit_gb

    # lazily loaded models

    @property
    def transcriber(self) -> Any:
        if self._transcriber is None:
            from granit.ingest.audio import AudioTranscriber

            self._transcriber = AudioTranscriber().load()
        return self._transcriber

    @property
    def documents(self) -> Any:
        if self._documents is None:
            from granit.ingest.documents import DocumentIngestor

            self._documents = DocumentIngestor().load()
        return self._documents

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from granit.search.embed import Embedder

            self._embedder = Embedder().load()
        return self._embedder

    # the loop

    def run(self, log: Callable[[str], None] = print) -> WorkerReport:
        start = time.perf_counter()
        self._limit_mlx_cache()
        report = WorkerReport()
        recovered = self.store.recover_interrupted()
        if recovered:
            log(f"re-queued {recovered} job(s) left running by an earlier worker")
        # A job that fails waits for the next run (retries are for crashes and transient errors, not an
        # immediate replay of a deterministic failure: that could be minutes of Vision time while Q&A is paused).
        failed_this_run: list[int] = []
        while (job := self.store.claim_next(exclude=failed_this_run)) is not None:
            source = self.store.source(job.source)
            job_start = time.perf_counter()
            try:
                summary = self.process(job)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.fail(job, error)
                failed_this_run.append(job.id)
                report.failed.append((job.id, error))
                log(f"✗ {job.task} {source.name}: {error}")
            else:
                report.done.append(job.id)
                log(
                    f"✓ {job.task} {source.name} ({time.perf_counter() - job_start:.1f} s) {summary}"
                )
        report.seconds = round(time.perf_counter() - start, 2)
        report.peak_footprint_gb = _peak_footprint_gb()
        return report

    def process(self, job: Job) -> str:
        if job.task == "ingest":
            return self._ingest(job)
        if job.task == "extract":
            return self._extract(job)
        raise ValueError(f"unknown task {job.task!r}")

    def _ingest(self, job: Job) -> str:
        source = self.store.source(job.source)
        path, out = self.store.file_path(source), self.store.derived_dir(source)
        out.mkdir(parents=True, exist_ok=True)
        extractions: list[NewExtraction] = []
        if source.kind == "audio":
            vocabulary = self.store.vocabulary()
            transcript = self.transcriber.transcribe(path).with_vocabulary(vocabulary)
            (out / "transcript.json").write_text(
                json.dumps(transcript.to_json(), ensure_ascii=False)
            )
            chunks: list[NewChunk] = chunk_transcript(transcript)
            info: dict[str, Any] = {
                "duration_s": transcript.duration_s,
                "speech_s": transcript.speech_s,
                "silence_s": transcript.silence_s,
                "segments": len(transcript.segments),
                "vocabulary": vocabulary,
            }
        else:
            from docling_core.types.doc import DoclingDocument

            accurate = bool(job.params.get("vision_tables", DOCUMENT_VISION_TABLES))
            result = self.documents.ingest(path, out, vision_tables=accurate)
            chunks = chunk_document(DoclingDocument.model_validate(result.document))
            extractions = [
                NewExtraction(
                    kind=e.kind,
                    format=e.format,
                    content=e.content,
                    valid=e.valid,
                    model=e.model,
                    errors=e.errors,
                    page=e.page,
                    crop=e.crop,
                )
                for e in result.extractions
            ]
            info = {**result.summary(), "accurate_tables": accurate}
        vectors = self.embedder.encode([c.search_body for c in chunks])
        info["chunks"] = len(chunks)
        self.store.complete_ingest(job, chunks, vectors, self.embedder.revision, extractions, info)
        return f"{len(chunks)} chunks, {len(extractions)} extractions"

    def _extract(self, job: Job) -> str:
        from granit.ingest.documents import check_format, render_pages, text_layer_lines

        source = self.store.source(job.source)
        path = self.store.file_path(source)
        check_format(path)
        schema = job.params["schema"]
        pages = render_pages(path, max_pages=int(job.params.get("pages", MAX_FORM_PAGES)))
        text = "\n".join(
            line
            for _, lines in sorted(text_layer_lines(path).items())[: len(pages)]
            for line in lines
        )
        e = self.documents.vision.extract_fields(pages, schema, text)
        self.store.complete_extract(
            job,
            NewExtraction(
                kind="form",
                format="json",
                content=e.content,
                valid=e.valid,
                model=e.model,
                errors=e.errors,
                missing=e.missing,
                schema=schema,
            ),
        )
        return "valid" if e.valid else f"invalid: {'; '.join(e.errors)}"

    def _limit_mlx_cache(self) -> None:
        if self.mlx_cache_limit_gb is None:
            return
        try:
            import mlx.core as mx
        except ImportError:  # not on Apple Silicon (tests with fake models)
            return
        mx.set_cache_limit(int(self.mlx_cache_limit_gb * 1e9))


def _peak_footprint_gb() -> float:
    try:
        from granit.models.memory import footprint

        return round(footprint().peak_gb, 2)
    except OSError:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    """``python -m granit.ingest.worker --data DIR``: the Phase A process the phase manager starts (PLAN.md §2.1).

    Progress lines go to stdout as they happen; the last line is ``RESULT <json>`` for the phase manager.
    """
    import argparse
    import sys

    from granit.config import DATA_DIR

    parser = argparse.ArgumentParser(prog="python -m granit.ingest.worker")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    store = Store(args.data)
    try:
        report = IngestWorker(store).run(log=lambda line: print(line, flush=True))
    finally:
        store.close()
    result = {
        "done": report.done,
        "failed": report.failed,
        "seconds": report.seconds,
        "peak_footprint_gb": report.peak_footprint_gb,
    }
    print("RESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
