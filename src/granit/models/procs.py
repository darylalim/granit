"""Find another phase's process (PLAN.md §2.1): ``granit ingest``, a worker or ``mlx_lm.server``.

Only Python processes count. ``pgrep -f`` matched any command line containing the pattern, so a shell running
``granit ingest && granit ask`` made ``ask`` refuse ("ingest is running") although the only ingest had finished. Matching the
interpreter rules out shells, ``uv`` wrappers (their Python child still counts), editors and ``grep``.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import PurePath

Process = tuple[int, str, str]  # pid, executable, command line


def processes() -> list[Process]:
    """Every process's pid, executable and command line (empty if ``ps`` fails)."""
    executables, commands = _ps("comm"), _ps("args")
    return [(pid, exe, commands[pid]) for pid, exe in executables.items() if pid in commands]


def _ps(column: str) -> dict[int, str]:
    try:
        out = subprocess.run(
            ["ps", "-axww", "-o", f"pid=,{column}="], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    rows: dict[int, str] = {}
    for line in out.stdout.splitlines():
        pid, _, value = line.strip().partition(" ")
        if pid.isdigit():
            rows[int(pid)] = value.strip()
    return rows


def is_python(executable: str) -> bool:
    return PurePath(executable).name.lower().startswith("python")


def find(pattern: str, procs: Iterable[Process]) -> str | None:
    """The first Python process whose command line matches ``pattern``, as ``"<pid> <command line>"``."""
    regex = re.compile(pattern)
    for pid, exe, command in procs:
        if is_python(exe) and regex.search(command):
            return f"{pid} {command}"
    return None


def running(pattern: str) -> str | None:
    """Another Python process matching ``pattern`` (never this one)."""
    return find(pattern, (p for p in processes() if p[0] != os.getpid()))
