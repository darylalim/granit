"""Keep .gitignore honest (PLAN.md §4.5): user data, weights and secrets ignored; shared files committed."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.conftest import ROOT

IGNORED = [
    "data/granit.db",
    "data/granit.db-wal",
    "data/derived/abc/crops/p3_table1.png",
    "data/eval/questions.yaml",
    "models/granite-guardian-4.1-8b-q8-mlx/model.safetensors",
    "src/stray.gguf",
    "tests/weights.safetensors",
    "x.npy",
    ".venv/bin/python",
    "dist/granit-0.1.0.tar.gz",
    "src/granit/__pycache__/config.cpython-312.pyc",
    ".ruff_cache/x",
    ".pytest_cache/x",
    "THIRD_PARTY_NOTICES.md",
    ".env",
    ".env.local",
    ".streamlit/secrets.toml",
    ".claude/settings.local.json",
    ".DS_Store",
    ".vscode/settings.json",
]
COMMITTED = [
    "uv.lock",
    ".python-version",
    "pyproject.toml",
    ".claude/settings.json",
    ".claude/hooks/guard_bash.py",
    ".claude/milestone",
    ".github/workflows/ci.yml",
    ".github/rulesets/main.json",
    "LICENSE",
    "NOTICE",
    "licenses_overrides.toml",
    "CLAUDE.md",
    "PLAN.md",
    ".env.example",
    ".streamlit/config.toml",
    ".vscode/extensions.json",
    "tests/fixtures/a.wav",
    "eval/public/questions.yaml",
    "eval/results/2026-10-04-abc-public.json",
]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, capture_output=True
    )
    return result.returncode == 0


@pytest.mark.parametrize("path", IGNORED)
def test_ignored(path: str) -> None:
    assert ignored(path), f"{path} should be ignored"


@pytest.mark.parametrize("path", COMMITTED)
def test_not_ignored(path: str) -> None:
    assert not ignored(path), f"{path} must stay committable"


def test_no_trailing_comments() -> None:
    """Git treats `/data/   # note` as one literal pattern (PLAN.md §4.5)."""
    for line in (ROOT / ".gitignore").read_text().splitlines():
        assert line.startswith("#") or "#" not in line, f"trailing comment in: {line!r}"
