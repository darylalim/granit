"""Claude Code hook scripts (PLAN.md §4.1): decisions as functions, plus end-to-end runs with real stdin JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import HOOKS_DIR, ROOT, load_hook

hooklib = load_hook("_hooklib")
guard_bash = load_hook("guard_bash")
guard_memory = load_hook("guard_memory")
protect_paths = load_hook("protect_paths")


def run_hook(
    name: str, payload: object, project: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / f"{name}.py")],
        input=stdin,
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project), **env},
        timeout=30,
    )


# ── command parsing ──


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls -la", [["ls", "-la"]]),
        ("cd x && pip install requests", [["cd", "x"], ["pip", "install", "requests"]]),
        ("a; b | c || d", [["a"], ["b"], ["c"], ["d"]]),
        ("FOO=1 sudo -E rm -rf data", [["rm", "-rf", "data"]]),
        ("uv run --with x pytest -m model", [["pytest", "-m", "model"]]),
        ("uvx --from huggingface_hub hf download a/b", [["hf", "download", "a/b"]]),
        ("bash -c 'git tag v1 && echo hi'", [["git", "tag", "v1"], ["echo", "hi"]]),
        ("echo one\necho two", [["echo", "one"], ["echo", "two"]]),
        ("echo 'unbalanced", [["echo", "'unbalanced"]]),
    ],
)
def test_split_commands(command: str, expected: list[list[str]]) -> None:
    assert hooklib.split_commands(command) == expected


# ── H3 guard_bash ──

BLOCKED = [
    "pip install requests",
    "pip3 install -U torch",
    "python -m pip install x",
    "python3.12 -m pip list",
    "uv pip install requests",
    "huggingface-cli download ibm-granite/x",
    "hf download ibm-granite/granite-speech-5.0-470m-turboctc-nc",
    "uv run python -c 'x' ibm-granite/granite-speech-5.0-470m-turboctc-nc",
    "rm -rf data",
    "rm -r ./models/granite-guardian-4.1-8b-q8-mlx",
    "rm -fr data/granit.db",
    "rm --recursive models",
    "rm -rf .",
    "sudo sysctl iogpu.wired_limit_mb=28000",
    "sysctl -w iogpu.wired_limit_mb 28000",
    "git tag v0.1.0",
    "git tag -a v1.0.0 -m release",
    "git tag -d v0.1.0",
    "git push --tags",
    "git push origin main --follow-tags",
    "git push origin refs/tags/v1.0.0",
    "gh release create v0.1.0",
    "ls && pip install x",
    "bash -c 'gh release create v1'",
]
ALLOWED = [
    "uv add requests",
    "uv sync --locked",
    "hf download ibm-granite/granite-speech-5.0-470m-turboctc",
    "grep turboctc-nc PLAN.md",
    "rg -n 'turboctc-nc' .",
    "git log --grep turboctc-nc",
    "rm -rf .venv",
    "rm -rf dist build",
    "rm data/granit.db.bak",  # not recursive: a single file is the user's call
    "sysctl hw.memsize",
    "sysctl iogpu.wired_limit_mb",  # reading the limit is fine
    "git tag",
    "git tag -l 'v0.*'",
    "git tag --list --sort=-v:refname",
    "git tag --contains HEAD",
    "git push -u origin m0/scaffold",
    "gh release list",
    "gh release view v0.1.0.dev0",
    "pytest -q",
]


@pytest.mark.parametrize("command", BLOCKED)
def test_guard_bash_blocks(command: str, project: Path) -> None:
    reasons = [guard_bash.check(argv, project, project) for argv in hooklib.split_commands(command)]
    assert any(reasons), command


@pytest.mark.parametrize("command", ALLOWED)
def test_guard_bash_allows(command: str, project: Path) -> None:
    reasons = [guard_bash.check(argv, project, project) for argv in hooklib.split_commands(command)]
    assert not any(reasons), (command, reasons)


def test_guard_bash_absolute_path_into_project(project: Path) -> None:
    argv = ["rm", "-rf", str(project / "data" / "derived")]
    assert guard_bash.check(argv, Path("/"), project)
    assert guard_bash.check(["rm", "-rf", "/tmp/elsewhere"], project, project) is None


def test_guard_bash_end_to_end(project: Path) -> None:
    blocked = run_hook("guard_bash", {"tool_input": {"command": "pip install requests"}}, project)
    assert blocked.returncode == 2
    assert "uv add" in blocked.stderr
    assert blocked.stderr.count("\n") == 1  # one-line reason
    allowed = run_hook("guard_bash", {"tool_input": {"command": "uv add requests"}}, project)
    assert allowed.returncode == 0


# ── H2 protect_paths ──


@pytest.mark.parametrize(
    "path",
    [
        "data/granit.db",
        "data/derived/x/document.md",
        "models/w/config.json",
        "uv.lock",
        ".env",
        ".env.local",
    ],
)
def test_protect_paths_blocks(path: str, project: Path) -> None:
    assert protect_paths.reason_for((project / path).resolve(), project)


@pytest.mark.parametrize(
    "path",
    [
        "src/granit/config.py",
        "pyproject.toml",
        ".env.example",
        "tests/fixtures/data/a.wav",
        "PLAN.md",
    ],
)
def test_protect_paths_allows(path: str, project: Path) -> None:
    assert protect_paths.reason_for((project / path).resolve(), project) is None


def test_protect_paths_ignores_files_outside_project(project: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("elsewhere") / "data" / "x.db"
    assert protect_paths.reason_for(outside, project) is None


def test_protect_paths_end_to_end(project: Path) -> None:
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(project / "data" / "granit.db")},
    }
    result = run_hook("protect_paths", payload, project)
    assert result.returncode == 2
    assert "H2" in result.stderr
    relative = {"tool_name": "Edit", "cwd": str(project), "tool_input": {"file_path": "uv.lock"}}
    assert run_hook("protect_paths", relative, project).returncode == 2
    ok = {"tool_name": "Edit", "tool_input": {"file_path": str(project / "src" / "a.py")}}
    assert run_hook("protect_paths", ok, project).returncode == 0


# ── H4 guard_memory ──


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -m model",
        "pytest -q -m 'model and slow'",
        "python -m pytest -mmodel",
        "uv run granit ingest",
        "granit eval run --set public",
        "uv run granit models smoke",
        "uv run granit models convert",
        "mlx_lm.server --model x",
        "uv run mlx_lm.generate --prompt hi",
        "python -m mlx_vlm.generate --model x",
        "python -m mlx_audio.stt.generate --audio a.wav",
        "python -m granit.models.smoke llm",
        "uv run granit bench",
        "uv run granit bench phase-b --quick",
        "python -m granit.bench.workers phase-a --hold",
        "python -m mlx_lm server --model x",
        "python -m granit.models.mlx_server --mlx-cache-limit-gb 1 --model x",
    ],
)
def test_guard_memory_detects_model_loading(command: str) -> None:
    assert any(guard_memory.loads_models(argv) for argv in hooklib.split_commands(command)), command


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest",
        "pytest -q -m 'not model'",
        "uv run granit models list",
        "uv run granit models download",
        "uv run ruff check",
        "pkill -f mlx_lm.server",
        "grep mlx_lm src -r",
    ],
)
def test_guard_memory_ignores_other_commands(command: str) -> None:
    assert not any(guard_memory.loads_models(argv) for argv in hooklib.split_commands(command)), (
        command
    )


def test_guard_memory_blocks_only_when_a_phase_runs(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        guard_memory, "read_input", lambda: {"tool_input": {"command": "pytest -m model"}}
    )
    monkeypatch.setattr(guard_memory, "running_phases", lambda: [])
    guard_memory.main()  # nothing running: allowed (returns normally)

    monkeypatch.setattr(guard_memory, "running_phases", lambda: ["123 python -m mlx_lm.server"])
    with pytest.raises(SystemExit) as exit_info:
        guard_memory.main()
    assert exit_info.value.code == 2
    assert "stop the running phase first" in capsys.readouterr().err


# ── robustness: malformed input never blocks ──


@pytest.mark.parametrize(
    "name", ["guard_bash", "guard_memory", "protect_paths", "format_lint", "quality_gate"]
)
@pytest.mark.parametrize("payload", ["not json", "[]", "{}", '{"tool_input": "x"}'])
def test_malformed_input_is_allowed(name: str, payload: str, project: Path) -> None:
    result = run_hook(name, payload, project)
    assert result.returncode == 0, result.stderr


# ── H1 format_lint / H5 quality_gate / H6 session_context (fast paths only; no uv runs) ──


def test_format_lint_skips_non_python(project: Path) -> None:
    (project / "notes.md").write_text("# hi\n")
    payload = {"tool_input": {"file_path": str(project / "notes.md")}}
    assert run_hook("format_lint", payload, project).returncode == 0


def test_quality_gate_respects_stop_hook_active(project: Path) -> None:
    result = run_hook("quality_gate", {"stop_hook_active": True}, project)
    assert result.returncode == 0
    assert result.stderr == ""


def test_quality_gate_skips_when_no_python_changed(project: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / "README.md").write_text("hi\n")
    result = run_hook("quality_gate", {"stop_hook_active": False}, project)
    assert result.returncode == 0
    assert result.stderr == ""


def test_session_context_reports_project_state() -> None:
    result = run_hook("session_context", {"hook_event_name": "SessionStart"}, ROOT)
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert lines[0].startswith("granit · branch ")
    assert lines[1].startswith("milestone: ")
    assert lines[2].startswith("models: ")
    assert "PLAN.md §7" in lines[3]


def test_session_context_keeps_full_paths_of_modified_files(project: Path) -> None:
    """Regression: stripping porcelain output turned ' M PLAN.md' into 'LAN.md'."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / "PLAN.md").write_text("v1\n")
    subprocess.run(["git", "add", "PLAN.md"], cwd=project, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=project, check=True)
    (project / "PLAN.md").write_text("v2\n")
    result = run_hook("session_context", {}, project)
    assert "1 uncommitted: PLAN.md" in result.stdout


def test_settings_json_wires_every_hook() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    commands = json.dumps(settings["hooks"])
    for script in (
        "session_context",
        "protect_paths",
        "guard_bash",
        "guard_memory",
        "format_lint",
        "quality_gate",
    ):
        assert f"/.claude/hooks/{script}.py" in commands
        assert (HOOKS_DIR / f"{script}.py").is_file()
