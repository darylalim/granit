"""H6 · SessionStart: print a few lines of project state into Claude's context (PLAN.md §4.1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _hooklib import project_dir, read_input, run

MAX_FILES = 5


def git_state(project: Path) -> str:
    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=project, capture_output=True, text=True, timeout=5)
        return out.stdout

    branch = git("branch", "--show-current").strip() or "(detached)"
    # Porcelain lines are "XY path"; the leading status column can be a space, so never strip it.
    changed = [line[3:] for line in git("status", "--porcelain").splitlines() if line.strip()]
    if not changed:
        return f"branch {branch} · clean"
    shown = ", ".join(changed[:MAX_FILES]) + (" …" if len(changed) > MAX_FILES else "")
    return f"branch {branch} · {len(changed)} uncommitted: {shown}"


def model_state(project: Path) -> str:
    sys.path.insert(0, str(project / "src"))
    try:
        from granit.config import HUB_MODELS, LOCAL_MODELS, RUNTIME_HUB_MODELS
    except Exception as exc:  # config is mid-edit or missing: report, don't fail
        return f"models: unknown ({type(exc).__name__})"
    missing = [k for k in RUNTIME_HUB_MODELS if HUB_MODELS[k].cached_snapshot() is None]
    unbuilt = [k for k, m in LOCAL_MODELS.items() if not m.is_built()]
    have = len(RUNTIME_HUB_MODELS) - len(missing)
    line = f"models: {have}/{len(RUNTIME_HUB_MODELS)} downloaded"
    line += f", local builds missing: {', '.join(unbuilt)}" if unbuilt else ", local builds ok"
    if missing:
        line += f" · missing: {', '.join(missing)} (uv run granit models download)"
    return line


def main() -> None:
    project = project_dir(read_input())
    milestone_file = project / ".claude" / "milestone"
    milestone = milestone_file.read_text().strip() if milestone_file.is_file() else "unknown"
    print(f"granit · {git_state(project)}")
    print(f"milestone: {milestone} (PLAN.md §5)")
    print(model_state(project))
    print("plan: PLAN.md §7 Decisions; conventions: CLAUDE.md")


if __name__ == "__main__":
    run(main)
