"""scripts/demo_video.py: the parts that run without a browser or models (docs/demo-plan.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))  # it imports the screenshot scripts' helpers
    try:
        spec = importlib.util.spec_from_file_location("demo_video", SCRIPTS / "demo_video.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["demo_video"] = module  # dataclasses look their module up
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


PS = """\
    1     0   1000
  100     1   2000
  200   100   3000
  201   100   4000
  300   200   5000
  400     1   9999
garbage line
"""


def test_tree_rss_counts_the_root_and_every_descendant(demo: ModuleType) -> None:
    assert demo.tree_rss_kb(PS, 100) == 2000 + 3000 + 4000 + 5000


def test_tree_rss_leaves_out_other_trees(demo: ModuleType) -> None:
    assert demo.tree_rss_kb(PS, 200) == 3000 + 5000  # not its parent (100) or sibling (201)


def test_tree_rss_of_a_gone_process_is_zero(demo: ModuleType) -> None:
    assert demo.tree_rss_kb(PS, 999) == 0


def test_shots_log_their_span_and_waits(demo: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter(
        [100.0, 101.5, 103.0, 107.25, 110.0]
    )  # started, shot start, wait start, wait end, shot end
    monkeypatch.setattr(demo.time, "monotonic", lambda: next(clock))
    take = demo.Take(page=None)
    with take.shot(6, "ask") as shot:
        take.wait(shot, lambda: None)
    assert take.shots == [
        {"id": 6, "name": "ask", "start": 1.5, "end": 10.0, "waits": [[3.0, 7.25]]}
    ]


def test_a_shot_can_start_after_its_setup(
    demo: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 1.0, 4.0, 9.0])  # started, shot start, after setup, shot end
    monkeypatch.setattr(demo.time, "monotonic", lambda: next(clock))
    take = demo.Take(page=None)
    with take.shot(4, "speakers") as shot:
        shot["start"] = take.now()  # setup the edit step cuts (shot 4 clears the name off camera)
    assert (take.shots[0]["start"], take.shots[0]["end"]) == (4.0, 9.0)


def test_playwright_is_imported_only_when_recording(demo: ModuleType) -> None:
    # CI has no `screenshots` group: importing the script (and testing it) mustn't need Playwright.
    assert "sync_playwright" not in vars(demo)
