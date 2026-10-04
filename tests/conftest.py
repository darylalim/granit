from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = ROOT / ".claude" / "hooks"


def load_hook(name: str) -> ModuleType:
    """Import a hook script by path (hooks live outside the package and import their sibling ``_hooklib``)."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    spec = importlib.util.spec_from_file_location(f"granit_hook_{name}", HOOKS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An empty stand-in project directory with the protected folders."""
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path
