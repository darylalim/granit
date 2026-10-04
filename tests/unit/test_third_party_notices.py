"""scripts/third_party_notices.py builds the release's THIRD_PARTY_NOTICES.md (PLAN.md §4.4)."""

from __future__ import annotations

import importlib.util
import re
from types import ModuleType

import pytest

from tests.conftest import ROOT


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "third_party_notices", ROOT / "scripts" / "third_party_notices.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_lists_every_dependency_with_its_license() -> None:
    text = load_script().render()
    assert text.startswith("# Third-party notices")
    table = re.findall(r"^\| (\S+) \| (\S+) \| (.+) \|$", text, re.MULTILINE)
    names = {row[0] for row in table[1:]}  # skip the header row
    assert {"torch", "mlx", "transformers", "streamlit", "pypdfium2"} <= names
    assert "granit" not in names
    assert all(row[2] != "UNKNOWN" for row in table[1:])


def test_license_texts_are_fenced_verbatim() -> None:
    """Parse fences like a Markdown renderer: inside a block, only the exact opening fence closes it."""
    open_fence, blocks = None, 0
    for line in load_script().render().splitlines():
        if open_fence is None:
            assert not line.startswith("```") or line.endswith("text"), f"stray fence: {line!r}"
            if line.startswith("```"):
                open_fence, blocks = line.removesuffix("text"), blocks + 1
        elif line == open_fence:
            open_fence = None
    assert open_fence is None, "unclosed license block"
    assert blocks > 100


def test_fence_outgrows_backticks_inside_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script()
    fake = type("P", (), {"name": "x", "version": "1", "license": "MIT"})()
    monkeypatch.setattr(script, "installed", lambda _overrides: [fake])
    monkeypatch.setattr(script, "distribution", lambda _name: None)
    monkeypatch.setattr(
        script, "license_texts", lambda _dist: [("LICENSE", "see:\n```\ncode\n````\n")]
    )
    text = script.render()
    assert "`````text\nsee:\n```\ncode\n````\n`````" in text
