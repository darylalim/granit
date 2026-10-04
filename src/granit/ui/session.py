"""Streamlit glue shared by the pages: the one backend per server process, a store per script run, the phase banner.

- ``backend()`` is cached for the whole server (``st.cache_resource``): every browser tab shares one phase manager and one
  ``mlx_lm.server``. It's shut down when the cache entry is released and when the process exits, so the LLM server never
  outlives the app.
- ``open_store()`` opens a connection for the current script run. Streamlit runs each rerun in a new thread, and a SQLite
  connection belongs to the thread that opened it (opening one takes about a millisecond).
"""

from __future__ import annotations

import atexit

import streamlit as st

from granit import config
from granit.models.phases import Phase
from granit.store.db import Store
from granit.ui import backend as backend_module
from granit.ui.backend import Backend, UiStatus


def _release(backend: Backend) -> None:
    backend.shutdown()


@st.cache_resource(show_spinner=False, on_release=_release)
def backend() -> Backend:
    b = backend_module.create_backend(config.DATA_DIR)
    atexit.register(b.shutdown)
    return b.start()


def open_store() -> Store:
    return Store(backend().data_dir)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' * (n != 1)}"


@st.fragment(run_every=2)
def phase_banner() -> None:
    """Which phase is running, refreshed every 2 s. Status colors only (PLAN.md §4.7): green ready, blue paused, red failed.

    When the phase changes, the whole page reruns: lists pick up what the batch produced, and a question typed while Q&A
    was paused gets asked.
    """
    status = backend().status()
    if status.error:
        st.error(f"Q&A isn't available: {status.error}", icon=":material/error:")
    elif status.phase is Phase.QA:
        text = "**Ready**"
        if status.queued:
            text += (
                f" · {plural(status.queued, 'job')} waiting: they run after a minute without questions,"
                " or now with **Process now** on the Ingest page"
            )
        st.success(text, icon=":material/check_circle:")
        if failed := status.last_switch.get("error"):
            st.error(f"The last ingest batch failed: {failed}", icon=":material/error:")
    else:
        st.info(status.message, icon=":material/hourglass_top:")
    seen = st.session_state.get("_phase")
    st.session_state["_phase"] = status.phase
    if seen is not None and seen != status.phase:
        st.rerun(scope="app")


def process_now_button(status: UiStatus, key: str) -> None:
    """ "Process now": switch to ingest as soon as the answer in progress (if any) finishes."""
    if status.processing_requested:
        label, help_text = "Starting…", "Processing starts when the answer in progress finishes."
    else:
        label, help_text = "Process now", "Pause Q&A and process the queued jobs now."
    clicked = st.button(
        label,
        type="primary",
        icon=":material/play_arrow:",
        key=key,
        help=help_text,
        disabled=not status.queued or not status.ready or status.processing_requested,
    )
    if clicked:
        backend().request_processing()
        st.rerun()
