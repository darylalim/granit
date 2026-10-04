"""The UI backend with fake processes and a fake LLM (M6): start-up order, answers under a lease, "Process now", failures.

``ui_backend`` builds a real ``Backend`` over a real store: a fake LLM server and the ingest worker with fake models (in a
thread instead of a process), BM25-only search and a scripted LLM that streams its reply. ``tests/unit/test_app.py`` runs
the pages against the same thing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from granit.models.phases import Phase, PhaseBusy, PhaseManager
from granit.models.server import Completion
from granit.reason.qa import QA
from granit.reason.tokens import ApproxTokens
from granit.search.hybrid import Searcher
from granit.store.db import Store
from granit.ui.backend import Backend
from tests.unit.test_phases import World
from tests.unit.test_reason import ScriptedLLM, completion
from tests.unit.test_worker import FIXTURES, worker

ANSWER = "Revenue was 145 thousand USD in Q2 [1]."


class StreamingLLM(ScriptedLLM):
    """Replies from a script, streamed through ``on_delta`` like the real client (reasoning first, then the answer)."""

    def chat(
        self, messages: list[dict[str, Any]], *, thinking: str | None = None, **kw: Any
    ) -> Completion:
        reply = super().chat(messages, thinking=thinking)
        if on_delta := kw.get("on_delta"):
            on_delta(("reasoning", "The report gives Q2 revenue."))
            for word in reply.content.split(" "):
                on_delta(("content", word + " "))
        return completion(reply.content, reasoning="The report gives Q2 revenue.")


def fake_worker(data_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """The real worker loop with fake models, in-process (a stand-in for ``python -m granit.ingest.worker``)."""
    store = Store(data_dir)
    try:
        return asdict(worker(store).run(log=log))
    finally:
        store.close()


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.01)


def make_backend(
    data: Path,
    replies: list[str] | None = None,
    *,
    world: World | None = None,
    run_worker: Callable[[Path, Callable[[str], None]], dict[str, Any]] = fake_worker,
    qa_factory: Callable[[Store], Any] | None = None,
    other_phase: Callable[[], str | None] = lambda: None,
) -> Backend:
    world = world or World()
    return Backend(
        data,
        phases_factory=lambda store, log: PhaseManager(
            store,
            server_factory=world.server,
            run_worker=run_worker,
            idle_before_ingest_s=3600,
            log=log,
        ),
        qa_factory=qa_factory
        or (
            lambda store: QA(
                store,
                Searcher(store),
                StreamingLLM(replies or [ANSWER] * 20),
                ApproxTokens(),
                mode="bm25",
            )
        ),
        tick_s=0.02,
        other_phase=other_phase,
        poll_s=0.01,
    )


def library(data: Path) -> Store:
    """A small library ingested by the worker with fake models: a recording and the report."""
    store = Store(data)
    store.add_file(FIXTURES / "audio" / "vad_pauses.wav")
    store.add_file(FIXTURES / "documents" / "report.pdf")
    worker(store).run(log=lambda _: None)
    return store


@pytest.fixture
def ui_backend(tmp_path: Path) -> Iterator[Backend]:
    library(tmp_path / "data").close()
    backend = make_backend(tmp_path / "data").start()
    wait_for(lambda: backend.status().ready)
    yield backend
    backend.shutdown()


def test_start_up_loads_search_then_starts_qa(tmp_path: Path) -> None:
    world = World()
    backend = make_backend(tmp_path / "data", world=world)
    status = backend.status()
    assert status.phase is Phase.STOPPED and not status.ready and status.message == "Starting…"
    backend.start()
    wait_for(lambda: backend.status().ready)
    assert world.events == ["server.start", "server.ready", "server.warm_up"]
    assert any("Q&A ready" in line for _, line in backend.events)
    backend.shutdown()
    assert world.events[-1] == "server.stop" and backend.status().phase is Phase.STOPPED


def test_waits_for_an_ingest_worker_left_running(tmp_path: Path) -> None:
    calls = iter(["123 python -m granit.ingest.worker --data data", None])
    world = World()
    backend = make_backend(tmp_path / "data", world=world, other_phase=lambda: next(calls, None))
    backend.start()
    wait_for(lambda: backend.status().ready)
    assert world.events[0] == "server.start"  # only after the worker was gone
    backend.shutdown()


def test_answers_stream_and_are_recorded(ui_backend: Backend) -> None:
    deltas: list[tuple[str, str]] = []
    answer = ui_backend.ask("How much revenue in Q2?", thinking="low", on_delta=deltas.append)
    assert answer.text == ANSWER and [h.source_name for h in answer.cited] == ["report.pdf"]
    assert (
        deltas[0][0] == "reasoning"
        and "".join(t for k, t in deltas if k == "content").strip() == ANSWER
    )
    (turn,) = ui_backend.store.turns()
    assert turn["question"] == "How much revenue in Q2?" and turn["thinking"] == "low"


def test_process_now_runs_the_queue_and_qa_comes_back(ui_backend: Backend) -> None:
    store = ui_backend.store
    source, _ = store.add_file(FIXTURES / "documents" / "memo.pdf")
    assert ui_backend.status().queued == 1
    ui_backend.request_processing()
    wait_for(lambda: store.source(source.id).status == "ready" and ui_backend.status().ready)
    status = ui_backend.status()
    assert status.queued == 0 and not status.processing_requested
    assert {"stop_s", "ingest_s", "restart_s"} <= set(status.last_switch)
    assert any(line.startswith("✓ ingest memo.pdf") for _, line in ui_backend.events)


def test_questions_during_ingest_get_the_banner_message(tmp_path: Path) -> None:
    started, release = time.monotonic(), []

    def slow_worker(data_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
        while not release and time.monotonic() - started < 5:
            time.sleep(0.01)
        return {"done": [], "failed": []}

    library(tmp_path / "data").close()
    backend = make_backend(tmp_path / "data", run_worker=slow_worker).start()
    wait_for(lambda: backend.status().ready)
    backend.store.add_file(FIXTURES / "documents" / "memo.pdf")
    backend.request_processing()
    wait_for(lambda: backend.status().phase is Phase.INGEST)
    status = backend.status()
    assert not status.ready and status.message == "Q&A paused: ingesting 1 file"
    with pytest.raises(PhaseBusy, match="Q&A paused"):
        backend.ask("Anything?")
    release.append(True)
    wait_for(lambda: backend.status().ready)
    backend.shutdown()


def test_a_start_up_failure_is_shown_not_raised(tmp_path: Path) -> None:
    def broken(store: Store) -> Any:
        raise FileNotFoundError(
            "granite-embedding-english-r2 isn't downloaded: run `granit models download`"
        )

    backend = make_backend(tmp_path / "data", qa_factory=broken).start()
    wait_for(lambda: backend.status().error is not None)
    status = backend.status()
    assert not status.ready and "isn't downloaded" in (status.error or "")
    with pytest.raises(PhaseBusy):
        backend.ask("Anything?")
    backend.shutdown()


def test_a_second_server_on_the_port_is_reported(tmp_path: Path) -> None:
    world = World()
    world.already_running = True
    backend = make_backend(tmp_path / "data", world=world).start()
    wait_for(lambda: backend.status().error is not None)
    assert "already answering" in (backend.status().error or "")
    backend.shutdown()
