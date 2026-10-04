"""The phase manager as a state machine with fake processes (PLAN.md §2.1 test policy, M5).

Never two phases at once; queued jobs run as one batch; Q&A always comes back, even after a failed batch; no switch while an
answer is in progress; "Process now" waits for it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from granit.models.phases import Phase, PhaseBusy, PhaseManager, WorkerFailed
from granit.store.db import Store
from tests.conftest import ROOT

DOCS = ROOT / "tests" / "fixtures" / "documents"


class World:
    """Records what's running, and fails the test if the server and the worker ever overlap."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.server_running = False
        self.worker_running = False
        self.worker_result: Any = {"done": [], "failed": []}
        self.already_running = False

    def server(self) -> FakeServer:
        return FakeServer(self)

    def run_worker(self, data_dir: Path, log: Any) -> dict[str, Any]:
        assert not self.server_running, "worker started while the LLM server was still running"
        self.worker_running = True
        self.events.append("worker")
        try:
            if isinstance(self.worker_result, Exception):
                raise self.worker_result
            return self.worker_result
        finally:
            self.worker_running = False


class FakeServer:
    def __init__(self, world: World) -> None:
        self.world = world

    def healthy(self) -> bool:
        return self.world.already_running

    def start(self) -> None:
        assert not self.world.worker_running, "server started while the worker was running"
        self.world.server_running = True
        self.world.events.append("server.start")

    def wait_ready(self) -> float:
        self.world.events.append("server.ready")
        return 0.0

    def warm_up(self) -> None:
        self.world.events.append("server.warm_up")

    def stop(self) -> float:
        self.world.server_running = False
        self.world.events.append("server.stop")
        return 0.0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "data")


def manager(
    store: Store, world: World, clock: Clock | None = None, idle: float = 60.0
) -> PhaseManager:
    return PhaseManager(
        store,
        server_factory=world.server,
        run_worker=world.run_worker,
        idle_before_ingest_s=idle,
        clock=clock or Clock(),
    )


def queue(store: Store, *names: str) -> None:
    for name in names:
        store.add_file(DOCS / name)


def test_start_brings_up_qa_with_a_warm_up(store: Store, world: World) -> None:
    pm = manager(store, world)
    assert pm.status().phase is Phase.STOPPED
    pm.start()
    assert world.events == ["server.start", "server.ready", "server.warm_up"]
    assert pm.status().phase is Phase.QA and pm.status().message == "Ready"
    pm.start()  # idempotent
    assert world.events.count("server.start") == 1


def test_refuses_to_start_a_second_server(store: Store, world: World) -> None:
    world.already_running = True
    with pytest.raises(RuntimeError, match="already answering"):
        manager(store, world).start()


def test_queued_jobs_run_as_one_batch_with_no_overlap(store: Store, world: World) -> None:
    clock = Clock()
    pm = manager(store, world, clock)
    pm.start()
    queue(store, "invoice.png", "report.pdf", "memo.pdf")
    assert pm.tick() is None  # not idle long enough yet
    clock.now += 61
    pm.tick()
    assert world.events[3:] == [
        "server.stop",
        "worker",
        "server.start",
        "server.ready",
        "server.warm_up",
    ]
    assert world.events.count("worker") == 1  # three files, one batch
    status = pm.status()
    assert status.phase is Phase.QA and set(status.last_switch) >= {
        "stop_s",
        "ingest_s",
        "restart_s",
    }


def test_nothing_happens_without_queued_jobs(store: Store, world: World) -> None:
    clock = Clock()
    pm = manager(store, world, clock)
    pm.start()
    clock.now += 1000
    assert pm.tick() is None and pm.process_now() is None
    assert "worker" not in world.events


@pytest.mark.parametrize("failure", [WorkerFailed("exit code 1"), RuntimeError("boom")])
def test_qa_comes_back_after_a_failed_batch(store: Store, world: World, failure: Exception) -> None:
    pm = manager(store, world, idle=0)
    pm.start()
    queue(store, "invoice.png")
    world.worker_result = failure
    assert pm.process_now() is None
    assert world.events[-3:] == ["server.start", "server.ready", "server.warm_up"]
    assert pm.status().phase is Phase.QA
    assert type(failure).__name__ in pm.status().last_switch["error"]


def test_no_switch_while_an_answer_is_in_progress(store: Store, world: World) -> None:
    clock = Clock()
    pm = manager(store, world, clock, idle=0)
    pm.start()
    queue(store, "invoice.png")
    with pm.chat():
        assert not pm.should_ingest() and pm.tick() is None
    assert pm.status().active_chats == 0
    assert pm.tick() is not None


def test_idle_timer_restarts_with_every_answer(store: Store, world: World) -> None:
    clock = Clock()
    pm = manager(store, world, clock, idle=60)
    pm.start()
    queue(store, "invoice.png")
    clock.now += 50
    with pm.chat():
        pass
    clock.now += 50  # 100 s since start, but only 50 s since the last answer
    assert not pm.should_ingest()
    clock.now += 11
    assert pm.should_ingest()


def test_questions_during_a_switch_get_the_banner_message(store: Store, world: World) -> None:
    pm = manager(store, world, idle=0)
    pm.start()
    queue(store, "invoice.png", "memo.pdf")
    seen: list[str] = []

    def worker_that_asks(data_dir: Path, log: Any) -> dict[str, Any]:
        with pytest.raises(PhaseBusy) as busy, pm.chat():
            pass
        seen.append(str(busy.value))
        return {"done": [], "failed": []}

    pm._run_worker = worker_that_asks
    pm.process_now()
    assert seen == ["Q&A paused: ingesting 2 files"]


def test_process_now_waits_for_the_answer_in_progress(store: Store, world: World) -> None:
    pm = manager(store, world, idle=3600)  # the idle timer alone would never switch
    pm.start()
    queue(store, "invoice.png")
    order: list[str] = []
    answering = threading.Event()

    def answer() -> None:
        with pm.chat():
            answering.set()
            time.sleep(0.2)
            order.append("answer done")

    thread = threading.Thread(target=answer)
    thread.start()
    answering.wait(1)

    def ingest(data_dir: Path, log: Any) -> dict[str, Any]:
        order.append("ingest")
        return {"done": [], "failed": []}

    pm._run_worker = ingest
    pm.process_now(timeout=5)
    thread.join()
    assert order == ["answer done", "ingest"]


def test_stop(store: Store, world: World) -> None:
    pm = manager(store, world)
    pm.start()
    pm.stop()
    assert world.events[-1] == "server.stop" and pm.status().phase is Phase.STOPPED
    with pytest.raises(PhaseBusy, match="Stopped"), pm.chat():
        pass
