"""The UI's long-lived backend (PLAN.md §2.1, §2.2): one per Streamlit server process, shared by every browser session.

It owns the phase manager (and through it ``mlx_lm.server`` and the ingest worker process), the query embedder and reranker
(the only models in the UI process, ~0.6 GB) and a background thread that:

1. waits for an ingest worker left over from an earlier run to finish (never two phases at once),
2. loads the search models, then starts Q&A (Phase B) and warms it up,
3. every ``tick_s`` calls ``PhaseManager.tick()`` (switch to ingest after 60 s without questions), or ``process_now()`` as
   soon as the user presses "Process now" (which still waits for the answer in progress).

Pages never touch models or processes directly: they call ``ask``, ``summarize``, ``request_processing`` and ``status``.
No Streamlit import here, so it's unit-tested with fake phases (``tests/unit/test_ui_backend.py``).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granit.models.phases import Phase, PhaseManager
from granit.models.procs import running
from granit.store.db import Source, Store

TICK_S = 2.0
# Another granit process holding Phase A or C (or a bench): wait for it before starting Q&A.
OTHER_PHASE_PATTERN = (
    r"granit\.ingest\.worker|granit\.verify\.worker|granit\.bench\.workers"
    r"|granit (ingest|extract|transcribe|convert|bench|verify)"
)


def other_phase_running() -> str | None:
    return running(OTHER_PHASE_PATTERN)


@dataclass
class UiStatus:
    phase: Phase
    message: str
    queued: int
    ready: bool  # questions can be asked right now
    error: str | None = None  # Q&A couldn't start (or restart): shown in red until it does
    processing_requested: bool = False
    last_switch: dict[str, Any] = field(default_factory=dict)


class Backend:
    def __init__(
        self,
        data_dir: Path | str,
        *,
        phases_factory: Callable[[Store, Callable[[str], None]], PhaseManager] | None = None,
        qa_factory: Callable[[Store], Any] | None = None,
        tick_s: float = TICK_S,
        other_phase: Callable[[], str | None] = other_phase_running,
        poll_s: float = 2.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.store = Store(self.data_dir, shared=True)
        # One answer (or summary) at a time: the embedder and reranker run on MPS, and their writes share one connection.
        self.lock = threading.RLock()
        self.events: deque[tuple[float, str]] = deque(maxlen=200)
        make_phases = phases_factory or (lambda store, log: PhaseManager(store, log=log))
        self.phases = make_phases(self.store, self._log)
        self._qa_factory = qa_factory or default_qa
        self._qa: Any = None
        self._tick_s = tick_s
        self._other_phase = other_phase
        self._poll_s = poll_s
        self._error: str | None = None
        self._starting = "Starting…"
        self._wake = threading.Event()
        self._process = threading.Event()
        self._stopping = threading.Event()
        self._started = False
        self._thread = threading.Thread(target=self._run, name="granit-phases", daemon=True)

    # lifecycle

    def start(self) -> Backend:
        """Start the background thread once (every page calls this; later calls do nothing)."""
        with self.lock:
            if not self._started:
                self._started = True
                self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop the loop and the LLM server (called when Streamlit stops or the cached backend is released)."""
        self._stopping.set()
        self._wake.set()
        if self._started:
            self._thread.join(timeout=5)
        self.phases.stop()

    def _run(self) -> None:
        try:
            while (running := self._other_phase()) and not self._stopping.is_set():
                self._starting = f"Waiting for another granit process to finish: {running[:80]}"
                self._stopping.wait(self._poll_s)
            self._starting = "Loading the search models…"
            self._qa = self._qa_factory(self.store)
            self._starting = "Starting Q&A…"
            seconds = self.phases.start()
            self._log(f"Q&A ready in {seconds:.1f} s")
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._log(f"Q&A couldn't start: {self._error}")
            return
        while not self._stopping.is_set():
            self._wake.wait(self._tick_s)
            self._wake.clear()
            if self._stopping.is_set():
                break
            try:
                if self._process.is_set():
                    self._process.clear()
                    self.phases.process_now()
                else:
                    self.phases.tick()
                self._error = None
            except Exception as exc:  # the server didn't come back after a batch
                self._error = f"{type(exc).__name__}: {exc}"
                self._log(f"Q&A couldn't restart: {self._error}")

    def _log(self, line: str) -> None:
        self.events.append((time.time(), line))

    # what pages call

    def status(self) -> UiStatus:
        s = self.phases.status()
        message = self._starting if s.phase is Phase.STOPPED else s.message
        return UiStatus(
            phase=s.phase,
            message=message,
            queued=s.queued,
            ready=s.phase is Phase.QA and self._error is None,
            error=self._error,
            processing_requested=self._process.is_set(),
            last_switch=s.last_switch,
        )

    def request_processing(self) -> None:
        """ "Process now": the loop switches to ingest as soon as any answer in progress has finished."""
        self._process.set()
        self._wake.set()

    def ask(self, question: str, *, thinking: str | None = None, on_delta: Any = None) -> Any:
        """Answer with citations (``granit.reason.qa.Answer``). Raises ``PhaseBusy`` while Q&A is paused or starting."""
        with self.phases.chat(), self.lock:
            return self._qa.ask(question, thinking=thinking, on_delta=on_delta)

    def summarize(self, source: Source) -> Any:
        """Meeting summary JSON for an ingested recording (stored as its ``summary`` extraction)."""
        from granit.reason.llm import LLMClient
        from granit.reason.meetings import summarize_source

        with self.phases.chat(), self.lock:
            return summarize_source(self.store, source, LLMClient(), self._qa.counter)


def default_qa(store: Store) -> Any:
    """The real Q&A stack: Embedding R2 + Reranker R2 in this process (MPS), Granite 4.2 through ``mlx_lm.server``."""
    from granit.reason.llm import LLMClient
    from granit.reason.qa import QA
    from granit.reason.tokens import GraniteTokens
    from granit.search.embed import Embedder
    from granit.search.hybrid import Searcher
    from granit.search.rerank import Reranker
    from granit.search.vectors import VectorIndex

    embedder = Embedder().load()
    searcher = Searcher(store, embedder, VectorIndex(store, embedder.revision), Reranker().load())
    searcher.warm_up()
    return QA(store, searcher, LLMClient(), GraniteTokens())


def create_backend(data_dir: Path | str) -> Backend:
    """The factory the app calls (tests replace it with one that builds fakes)."""
    return Backend(data_dir)
