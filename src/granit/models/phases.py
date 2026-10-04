"""The phase manager (PLAN.md §2.1): one phase's models in memory at a time.

- **Q&A (Phase B)** is the default: ``mlx_lm.server`` runs (with granit's memory limits) and answers questions. Each answer
  holds a ``chat()`` lease; while any lease is open, no switch starts.
- **Ingest (Phase A):** when jobs are queued and Q&A has been idle for ``idle_before_ingest_s`` (``tick()``), or right away
  when the user asks (``process_now()``, which still waits for an in-flight answer), the manager stops the server and waits for
  it to exit (freeing its memory), runs the ingest worker **as its own process** to drain the whole queue in one batch, then
  **always** restarts the server, health-checks it and sends a warm-up request (``try/finally``).
- Questions during a switch get ``PhaseBusy`` with a message for the UI banner ("Q&A paused: ingesting 3 files").
- Phase C (Guardian verify, v1.1) will be another batch kind on the same mechanism.

M1 measured B→A 7.6–8.9 s and A→B 6.5–6.7 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

IDLE_BEFORE_INGEST_S = 60.0


class Phase(StrEnum):
    STOPPED = "stopped"
    QA = "qa"
    SWITCHING = "switching"
    INGEST = "ingest"


class PhaseBusy(RuntimeError):
    """Q&A isn't available right now (the UI keeps the question and shows the banner)."""


class WorkerFailed(RuntimeError):
    pass


@dataclass
class PhaseStatus:
    phase: Phase
    message: str
    queued: int
    active_chats: int
    last_switch: dict[str, Any] = field(default_factory=dict)


def run_worker_process(data_dir: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """Run ``python -m granit.ingest.worker`` and return its RESULT. Progress lines are passed to ``log`` as they arrive."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "granit.ingest.worker", "--data", str(data_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result: dict[str, Any] | None = None
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
        elif line.startswith(("✓", "✗", "re-queued")):
            log(line)
        else:
            tail = [*tail[-20:], line]
    code = proc.wait()
    if code != 0 or result is None:
        raise WorkerFailed(f"ingest worker exited with code {code}: " + " | ".join(tail[-3:]))
    return result


class PhaseManager:
    def __init__(
        self,
        store: Any,
        server_factory: Callable[[], Any] | None = None,
        run_worker: Callable[[Path, Callable[[str], None]], dict[str, Any]] = run_worker_process,
        idle_before_ingest_s: float = IDLE_BEFORE_INGEST_S,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = lambda _: None,
    ) -> None:
        self.store = store
        self._server_factory = server_factory or self._default_server
        self._run_worker = run_worker
        self.idle_before_ingest_s = idle_before_ingest_s
        self._clock = clock
        self._log = log
        self._cond = threading.Condition(threading.RLock())
        self._phase = Phase.STOPPED
        self._message = "Stopped"
        self._active = 0
        self._last_activity = clock()
        self._server: Any = None
        self.last_switch: dict[str, Any] = {}

    def _default_server(self) -> Any:
        from granit.config import HUB_MODELS
        from granit.models.download import local_snapshot
        from granit.models.server import LLMServer, ServerConfig

        logs = Path(self.store.root) / "logs"
        logs.mkdir(exist_ok=True)
        return LLMServer(
            ServerConfig(local_snapshot(HUB_MODELS["llm"])), log_path=logs / "llm-server.log"
        )

    # lifecycle

    def start(self) -> float:
        """Start Q&A (Phase B). Returns seconds until it answered a warm-up request."""
        with self._cond:
            if self._phase is Phase.QA:
                return 0.0
            self._set(Phase.SWITCHING, "Starting Q&A…")
        seconds = self._start_server()
        with self._cond:
            self._set(Phase.QA, "Ready")
        return seconds

    def stop(self) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._active == 0, timeout=60)
            if self._server is not None:
                self._server.stop()
                self._server = None
            self._set(Phase.STOPPED, "Stopped")

    def _start_server(self) -> float:
        start = time.perf_counter()
        server = self._server_factory()
        if server.healthy():
            raise RuntimeError(
                "an LLM server is already answering on this port: another granit (or a bench) is running Q&A"
            )
        server.start()
        server.wait_ready()
        self._warm_up(server)
        self._server = server
        return round(time.perf_counter() - start, 2)

    def _warm_up(self, server: Any) -> None:
        """mlx_lm.server loads weights on the first request: answer one token before reporting ready (M1)."""
        warm = getattr(server, "warm_up", None)
        if warm is not None:
            warm()
            return
        from granit.models.server import stream_chat

        stream_chat(
            server.config.base_url,
            [{"role": "user", "content": "Hi"}],
            max_tokens=1,
            enable_thinking=False,
        )

    # Q&A leases

    @contextmanager
    def chat(self) -> Iterator[None]:
        """Hold Q&A open for one answer. Raises ``PhaseBusy`` if a switch is in progress."""
        with self._cond:
            if self._phase is not Phase.QA:
                raise PhaseBusy(self._message)
            self._active += 1
            self._last_activity = self._clock()
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._last_activity = self._clock()
                self._cond.notify_all()

    # ingest batches

    def status(self) -> PhaseStatus:
        with self._cond:
            return PhaseStatus(
                self._phase,
                self._message,
                self.store.queued_count(),
                self._active,
                dict(self.last_switch),
            )

    def should_ingest(self) -> bool:
        with self._cond:
            return (
                self._phase is Phase.QA
                and self._active == 0
                and self._clock() - self._last_activity >= self.idle_before_ingest_s
                and self.store.queued_count() > 0
            )

    def tick(self) -> dict[str, Any] | None:
        """Called periodically (UI loop): run a batch if jobs are waiting and Q&A has been idle long enough."""
        return self._ingest(wait_for_chats=False) if self.should_ingest() else None

    def process_now(self, timeout: float = 300) -> dict[str, Any] | None:
        """ "Process now": run the queued jobs as soon as any answer in progress has finished."""
        return self._ingest(wait_for_chats=True, timeout=timeout)

    def _ingest(self, wait_for_chats: bool, timeout: float = 300) -> dict[str, Any] | None:
        with self._cond:
            if wait_for_chats:
                self._cond.wait_for(lambda: self._active == 0, timeout=timeout)
            if self._phase is not Phase.QA or self._active or not self.store.queued_count():
                return None
            queued = self.store.queued_count()
            self._set(Phase.SWITCHING, f"Q&A paused: ingesting {queued} file{'s' * (queued != 1)}")
        timings: dict[str, Any] = {"queued": queued}
        result: dict[str, Any] | None = None
        try:
            start = time.perf_counter()
            self._server.stop()  # waits for the process to exit: its memory is free before Phase A loads
            self._server = None
            timings["stop_s"] = round(time.perf_counter() - start, 2)
            with self._cond:
                self._set(Phase.INGEST, self._message)
            start = time.perf_counter()
            result = self._run_worker(Path(self.store.root), self._log)
            timings["ingest_s"] = round(time.perf_counter() - start, 2)
        except Exception as exc:  # the worker crashed: record it; Q&A comes back regardless
            timings["error"] = f"{type(exc).__name__}: {exc}"
            self._log(f"ingest batch failed: {timings['error']}")
        finally:
            with self._cond:
                self._set(Phase.SWITCHING, "Restarting Q&A…")
            timings["restart_s"] = self._start_server()
            with self._cond:
                self._set(Phase.QA, "Ready")
                self.last_switch = timings
        return result

    def _set(self, phase: Phase, message: str) -> None:
        self._phase, self._message = phase, message
        self._cond.notify_all()
