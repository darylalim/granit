"""Shared helpers for granit's Claude Code hooks (PLAN.md §4.1).

Standard library only and Python 3.11-compatible: hooks run with the system ``python3`` before every
edit and command, so they must start in milliseconds and never depend on the project venv.
Contract: exit 0 = allow; exit 2 + one line on stderr = block / feedback. Never crash.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "|&"}
WRAPPERS = {"sudo", "env", "time", "nohup", "command", "exec", "nice", "caffeinate"}
SHELLS = {"bash", "sh", "zsh"}
# `uv run` / `uvx` options that take a value (so the value isn't mistaken for the command).
UV_VALUE_OPTIONS = {
    "--with",
    "--with-requirements",
    "--python",
    "-p",
    "--group",
    "--only-group",
    "--extra",
    "--env-file",
    "--from",
    "--directory",
    "--project",
    "--package",
    "--index",
    "--index-url",
}


def read_input() -> dict[str, Any]:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        log(f"ignoring malformed hook input: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def tool_input(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("tool_input")
    return value if isinstance(value, dict) else {}


def project_dir(data: dict[str, Any]) -> Path:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or Path.cwd()
    return Path(root).resolve()


def block(reason: str) -> NoReturn:
    print(reason, file=sys.stderr)
    sys.exit(2)


def log(message: str) -> None:
    print(f"[granit hook] {message}", file=sys.stderr)


def run(main: Callable[[], None]) -> None:
    """Run a hook's main(); unexpected errors are logged and allowed (a broken hook must not block work)."""
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log(f"hook error (allowing): {type(exc).__name__}: {exc}")
    sys.exit(0)


def split_commands(command: str) -> list[list[str]]:
    """Split a shell command line into simple commands (argv lists), recursing into ``bash -c '...'``."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes: fall back to plain whitespace splitting
        tokens = command.replace("\n", " ; ").split()

    commands: list[list[str]] = []
    current: list[str] = []
    for token in [*tokens, ";"]:
        if token in SEPARATORS:
            if current:
                commands.extend(_expand(current))
            current = []
        else:
            current.append(token)
    return commands


def _expand(argv: list[str]) -> list[list[str]]:
    argv = effective_argv(argv)
    if len(argv) >= 3 and Path(argv[0]).name in SHELLS and argv[1] in {"-c", "-lc", "-ec"}:
        return split_commands(argv[2])
    return [argv] if argv else []


def effective_argv(argv: list[str]) -> list[str]:
    """Drop env assignments and wrappers (``sudo``, ``env``, ``uv run …``, ``uvx …``) to find the real command."""
    i = 0
    while i < len(argv):
        word = argv[i]
        name = Path(word).name
        if "=" in word and not word.startswith("-") and word.split("=", 1)[0].isidentifier():
            i += 1
        elif name in WRAPPERS:
            i += 1
            while i < len(argv) and argv[i].startswith("-"):
                i += 1
        elif name == "uv" and i + 1 < len(argv) and argv[i + 1] == "run":
            i = _skip_options(argv, i + 2)
        elif name == "uvx":
            i = _skip_options(argv, i + 1)
        else:
            break
    return argv[i:]


def _skip_options(argv: list[str], i: int) -> int:
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in UV_VALUE_OPTIONS else 1
    return i


def python_module(argv: list[str]) -> str | None:
    """``python -m X …`` → ``X`` (also python3 / python3.12)."""
    if argv and Path(argv[0]).name.startswith("python") and len(argv) >= 3 and argv[1] == "-m":
        return argv[2]
    return None


def resolve(path: str, cwd: Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else cwd / p).resolve()


def relative_to_project(path: Path, project: Path) -> Path | None:
    try:
        return path.relative_to(project)
    except ValueError:
        return None
