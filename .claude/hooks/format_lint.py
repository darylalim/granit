"""H1 · PostToolUse Edit|Write: ruff format + ruff check --fix on the edited Python file (PLAN.md §4.1).

Lint errors that ruff can't fix are sent back to Claude (exit 2) so they're fixed right away.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _hooklib import (
    block,
    log,
    project_dir,
    read_input,
    relative_to_project,
    resolve,
    run,
    tool_input,
)

MAX_LINES = 30


def main() -> None:
    data = read_input()
    file_path = tool_input(data).get("file_path")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return
    project = project_dir(data)
    path = resolve(file_path, Path(data.get("cwd") or project))
    if not path.is_file() or relative_to_project(path, project) is None:
        return

    def ruff(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["uv", "run", "--quiet", "ruff", *args, str(path)],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=50,
        )

    try:
        ruff("format")
        result = ruff("check", "--fix", "--output-format", "concise")
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"ruff not run: {exc}")
        return
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).strip().splitlines()
        shown = "\n".join(lines[:MAX_LINES])
        more = f"\n… {len(lines) - MAX_LINES} more lines" if len(lines) > MAX_LINES else ""
        block(f"granit hook H1: ruff found problems it can't fix automatically:\n{shown}{more}")


if __name__ == "__main__":
    run(main)
