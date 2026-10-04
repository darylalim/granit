"""H3 · PreToolUse Bash: enforce uv-only installs, model licensing, data safety and CI-only releases (PLAN.md §4.1)."""

from __future__ import annotations

from pathlib import Path

from _hooklib import (
    block,
    project_dir,
    python_module,
    read_input,
    relative_to_project,
    resolve,
    run,
    split_commands,
    tool_input,
)

# Reading or searching for the string is fine (e.g. `grep turboctc-nc PLAN.md`); using it is not.
READ_ONLY = {
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "wc",
    "echo",
}
READ_ONLY_GIT = {"log", "grep", "diff", "show", "blame", "status"}
GIT_TAG_VALUE_FLAGS = {
    "--contains",
    "--no-contains",
    "--sort",
    "--points-at",
    "--merged",
    "--no-merged",
}
GIT_TAG_LIST_FLAGS = {"-l", "--list", "-n", *GIT_TAG_VALUE_FLAGS}
PROTECTED_DIRS = {"data", "models"}


def check(argv: list[str], cwd: Path, project: Path) -> str | None:
    """Return a block reason for one simple command, or None to allow it."""
    if not argv:
        return None
    cmd = Path(argv[0]).name
    args = argv[1:]

    if cmd in {"pip", "pip3"} or python_module(argv) == "pip":
        return "use `uv add <pkg>` (or `uv add --dev`), not pip: the lockfile must stay in sync"
    if (
        cmd == "uv"
        and args[:1] == ["pip"]
        and args[1:2]
        and args[1] in {"install", "sync", "uninstall"}
    ):
        return "use `uv add` / `uv remove`, not `uv pip`: the lockfile must stay in sync"
    if cmd == "huggingface-cli":
        return "`huggingface-cli` is deprecated; use `hf` (or `uv run granit models download`)"
    if any("turboctc-nc" in a for a in argv) and not _read_only(cmd, args):
        return "turboctc-nc is CC-BY-NC-SA (non-commercial); granit uses granite-speech-5.0-470m-turboctc"
    if cmd == "rm" and _recursive(args) and _hits_protected(args, cwd, project):
        return (
            "refusing to recursively delete data/ or models/ (user data / weights); do it yourself"
        )
    if cmd == "sysctl" and any("iogpu" in a and ("=" in a or "-w" in args) for a in args):
        return "no GPU memory-limit tuning: every phase fits the default GPU memory limit (PLAN.md §3.3)"
    if cmd == "git":
        sub, rest = _git_subcommand(args)
        if sub == "tag" and not _lists_tags(rest):
            return "tags and releases come only from CI on main (PLAN.md §4.3); bump with `uv version --bump`"
        if sub == "push" and any(
            a in {"--tags", "--follow-tags"} or a.startswith("refs/tags/") for a in rest
        ):
            return "tags and releases come only from CI on main (PLAN.md §4.3); don't push tags"
    if cmd == "gh" and args[:2] == ["release", "create"]:
        return "releases come only from CI on main (PLAN.md §4.3); merge a version bump instead"
    return None


def _read_only(cmd: str, args: list[str]) -> bool:
    if cmd == "git":
        sub, _ = _git_subcommand(args)
        return sub in READ_ONLY_GIT
    return cmd in READ_ONLY


def _git_subcommand(args: list[str]) -> tuple[str | None, list[str]]:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in {"-C", "-c"} else 1
    return (args[i], args[i + 1 :]) if i < len(args) else (None, [])


def _lists_tags(rest: list[str]) -> bool:
    """True for read-only `git tag` forms: no args, `-l/--list [pattern]`, or only listing options."""
    if not rest or any(a in {"-l", "--list"} for a in rest):
        return True
    i = 0
    while i < len(rest):
        flag = rest[i].split("=", 1)[0]
        if flag not in GIT_TAG_LIST_FLAGS:
            return False
        # --contains X / --sort X style: skip the value too (unless given as --flag=value).
        i += 2 if flag in GIT_TAG_VALUE_FLAGS and "=" not in rest[i] else 1
    return True


def _recursive(args: list[str]) -> bool:
    for a in args:
        if a == "--recursive":
            return True
        if a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a):
            return True
    return False


def _hits_protected(args: list[str], cwd: Path, project: Path) -> bool:
    for a in args:
        if a.startswith("-"):
            continue
        rel = relative_to_project(resolve(a, cwd), project)
        if rel is not None and (not rel.parts or rel.parts[0] in PROTECTED_DIRS):
            return True
    return False


def main() -> None:
    data = read_input()
    command = tool_input(data).get("command")
    if not isinstance(command, str) or not command.strip():
        return
    project = project_dir(data)
    cwd = Path(data.get("cwd") or project)
    for argv in split_commands(command):
        reason = check(argv, cwd, project)
        if reason:
            block(f"Blocked by granit hook H3: {reason}.")


if __name__ == "__main__":
    run(main)
