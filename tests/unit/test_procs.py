"""Finding another phase's process (PLAN.md §2.1): only Python processes count, not shells that mention one."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from granit.models import procs
from granit.models.procs import find, is_python
from granit.ui.backend import OTHER_PHASE_PATTERN

VENV = "/Users/me/granit/.venv/bin"
INGEST = r"granit\.ingest\.worker|granit ingest"


def test_a_shell_that_mentions_ingest_is_not_ingest() -> None:
    # The bug: `zsh -c "granit ingest && granit ask"` made ask refuse after ingest had finished.
    shell = (
        101,
        "/bin/zsh",
        "/bin/zsh -c uv run granit ingest --data lib && uv run granit ask 'who?' --data lib",
    )
    wrapper = (102, "uv", "uv run granit ask who? --data lib")
    assert find(INGEST, [shell, wrapper]) is None


def test_a_running_ingest_is_found() -> None:
    wrapper = (201, "uv", "uv run granit ingest --data lib")
    cli = (202, f"{VENV}/python3", f"{VENV}/python3 {VENV}/granit ingest --data lib")
    assert find(INGEST, [wrapper, cli]) == f"202 {VENV}/python3 {VENV}/granit ingest --data lib"


def test_a_worker_module_is_found() -> None:
    worker = (301, f"{VENV}/python", f"{VENV}/python -m granit.ingest.worker --data lib")
    assert find(INGEST, [worker]) == f"301 {VENV}/python -m granit.ingest.worker --data lib"


def test_other_tools_mentioning_a_phase_are_ignored() -> None:
    others = [
        (401, "/usr/bin/tail", "tail -f granit ingest.log"),
        (402, "/usr/bin/grep", "grep granit.verify.worker notes.md"),
        (403, "/Applications/Editor.app/Contents/MacOS/Editor", "Editor granit convert notes"),
    ]
    assert find(OTHER_PHASE_PATTERN, others) is None


@pytest.mark.parametrize(
    "exe",
    [
        "/usr/bin/python3",
        "python3.12",
        f"{VENV}/python",
        "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
        "/Users/me/My Projects/granit/.venv/bin/python",
    ],
)
def test_python_interpreters(exe: str) -> None:
    assert is_python(exe)


@pytest.mark.parametrize("exe", ["/bin/zsh", "bash", "uv", "/usr/bin/grep", "granit"])
def test_not_python(exe: str) -> None:
    assert not is_python(exe)


def test_processes_lists_this_one() -> None:
    mine = {pid: (exe, command) for pid, exe, command in procs.processes()}
    exe, command = mine[os.getpid()]
    assert is_python(exe)
    assert "pytest" in command or sys.argv[0] in command


def test_running_skips_this_process() -> None:
    # This pytest process's command line can't match; a pattern for its own pid's command would.
    mine = next(command for pid, _, command in procs.processes() if pid == os.getpid())
    assert find(re.escape(mine), procs.processes()) is not None
    assert procs.running(re.escape(mine)) is None


def test_ps_failure_means_nothing_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("no ps")

    monkeypatch.setattr(subprocess, "run", fail)
    assert procs.processes() == []
    assert procs.running(INGEST) is None
