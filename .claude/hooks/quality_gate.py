"""H5 · Stop: if Python changed, run the same checks as CI before Claude finishes (PLAN.md §4.1, §4.2).

ruff format --check · ruff check · ty check src tests app · pytest -q -m "not model". Failures → exit 2 with a
short summary so Claude keeps working. ``stop_hook_active`` → exit 0, so it can never loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _hooklib import block, log, project_dir, read_input, run

CHECKS: tuple[tuple[str, list[str]], ...] = (
    ("ruff format --check", ["uv", "run", "--quiet", "ruff", "format", "--check"]),
    ("ruff check", ["uv", "run", "--quiet", "ruff", "check", "--output-format", "concise"]),
    ("ty check", ["uv", "run", "--quiet", "ty", "check", "src", "tests", "app"]),
    ("pytest", ["uv", "run", "--quiet", "pytest", "-q", "-m", "not model", "-x", "--no-header"]),
)
TAIL_LINES = 12


def changed_python_files(project: Path) -> list[str]:
    def git(*args: str) -> list[str]:
        out = subprocess.run(
            ["git", *args], cwd=project, capture_output=True, text=True, timeout=10
        )
        return out.stdout.splitlines() if out.returncode == 0 else []

    tracked = git("diff", "--name-only", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard")
    return sorted({f for f in tracked + untracked if f.endswith(".py")})


def main() -> None:
    data = read_input()
    if data.get("stop_hook_active"):
        return
    project = project_dir(data)
    if not changed_python_files(project):
        return

    failures = []
    for name, command in CHECKS:
        try:
            result = subprocess.run(
                command, cwd=project, capture_output=True, text=True, timeout=240
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"{name} not run: {exc}")
            continue
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().splitlines()[-TAIL_LINES:]
            failures.append(f"✗ {name}\n" + "\n".join(f"  {line}" for line in tail))
    if failures:
        block("granit hook H5: quality gate failed; fix before finishing:\n" + "\n".join(failures))


if __name__ == "__main__":
    run(main)
