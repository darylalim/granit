"""H2 · PreToolUse Edit|Write: never edit user data, model weights, the lockfile or secrets (PLAN.md §4.1)."""

from __future__ import annotations

from pathlib import Path

from _hooklib import block, project_dir, read_input, relative_to_project, resolve, run, tool_input

PROTECTED_DIRS = {
    "data": "data/ holds user data (granit.db, originals); never edit it by hand",
    "models": "models/ holds downloaded / converted weights; rebuild with `uv run granit models convert`",
}


def reason_for(path: Path, project: Path) -> str | None:
    rel = relative_to_project(path, project)
    if rel is None or not rel.parts:
        return None
    if rel.parts[0] in PROTECTED_DIRS:
        return PROTECTED_DIRS[rel.parts[0]]
    if rel == Path("uv.lock"):
        return "uv.lock is generated: change dependencies with `uv add` / `uv remove` / `uv lock`"
    if rel.name.startswith(".env") and rel.name != ".env.example":
        return f"{rel.name} may hold secrets; edit it yourself, outside Claude Code"
    return None


def main() -> None:
    data = read_input()
    args = tool_input(data)
    file_path = args.get("file_path") or args.get("notebook_path")
    if not isinstance(file_path, str) or not file_path:
        return
    project = project_dir(data)
    reason = reason_for(resolve(file_path, Path(data.get("cwd") or project)), project)
    if reason:
        block(f"Blocked by granit hook H2: {reason}.")


if __name__ == "__main__":
    run(main)
