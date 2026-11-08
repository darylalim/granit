"""scripts/demo_edit.py: the caption script, fitting footage to shots, caption timing, and what's checked before editing
(no ffmpeg or browser)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def edit() -> ModuleType:
    return load("demo_edit")


# ── the script ──

SCRIPT = """# Demo captions

Intro text isn't part of any shot.

## 1 · Title · 5 s

Private | On your Mac

## 4 · Speakers · 17 s

> a note

Name the people you recognize | granit only uses the names you give it.
@11 Who spoke when | The transcript shows who said what.
"""


def test_sections_have_captions_with_start_times(edit: ModuleType) -> None:
    title, speakers = edit.parse_script(SCRIPT)
    assert (title.shot, title.name, title.slot_s) == (1, "Title", 5.0)
    assert title.captions == (edit.Caption(0.0, "Private | On your Mac"),)
    assert [(c.at_s, c.text.split(" | ")[0]) for c in speakers.captions] == [
        (0.0, "Name the people you recognize"),
        (11.0, "Who spoke when"),
    ]


@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        ("## 1 · Title · 5 s\n\n> only a note\n", "no caption"),
        ("## 1 Title 5s\n\nHi\n", "heading"),
        ("## 2 · B · 5 s\n\nHi\n\n## 1 · A · 5 s\n\nHi\n", "in order"),
    ],
)
def test_bad_scripts_are_refused(edit: ModuleType, markdown: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        edit.parse_script(markdown)


def test_caption_words_ignore_the_separator(edit: ModuleType) -> None:
    assert edit.Caption(0.0, "Forms → JSON | Checked and ready").words == 6


# ── caption timing ──


def section(edit: ModuleType, slot: float, *captions: tuple[float, str]) -> object:
    return edit.Section(4, "Speakers", slot, tuple(edit.Caption(at, text) for at, text in captions))


def test_each_caption_lasts_until_the_next_and_the_last_until_the_shot_ends(
    edit: ModuleType,
) -> None:
    s = section(edit, 17.0, (0.0, "A | a"), (11.0, "B | b"))
    assert edit.caption_spans(s) == [
        (edit.CAPTION_IN_S, 11.0),
        (11.0, 17.0 - edit.CAPTION_END_S),
    ]


def test_a_caption_on_screen_too_briefly_to_read_is_reported(edit: ModuleType) -> None:
    words = " ".join(["word"] * 30)  # 10 s at 3 words a second, in a 6 s shot
    problems = edit.caption_problems(section(edit, 6.0, (0.0, f"Short | {words}")), card=False)
    assert len(problems) == 1 and "'Short'" in problems[0] and "31 words" in problems[0]


@pytest.mark.parametrize(
    "times",
    [(1.0,), (0.0, 5.0, 5.0), (0.0, 8.0, 4.0), (0.0, 17.0)],
)
def test_caption_times_start_at_zero_and_increase_inside_the_shot(
    edit: ModuleType, times: tuple[float, ...]
) -> None:
    s = section(edit, 17.0, *((t, "Hi") for t in times))
    assert any("must start at 0" in p for p in edit.caption_problems(s, card=False))


def test_a_card_has_one_caption_line(edit: ModuleType) -> None:
    two = section(edit, 5.0, (0.0, "A"), (2.0, "B"))
    assert edit.caption_problems(two, card=True) == ["shot 4 is a card: give it one caption line"]
    assert edit.caption_problems(section(edit, 5.0, (0.0, "A | B")), card=True) == []


# ── footage ──


def test_short_footage_is_kept_whole(edit: ModuleType) -> None:
    assert edit.keep_ranges(10.0, 17.5, [], 17.0) == [
        (10.0, 17.5)
    ]  # 7.5 s: the rest is a held frame


def test_long_footage_loses_the_middle_of_its_wait_first(edit: ModuleType) -> None:
    # 26 s of footage with a 10 s wait, into 20 s: 6 s go from the middle of the wait.
    assert edit.keep_ranges(0.0, 26.0, [[10.0, 20.0]], 20.0) == [(0.0, 12.0), (18.0, 26.0)]


def test_a_wait_keeps_its_last_second_then_the_tail_goes(edit: ModuleType) -> None:
    # 26 s with a 10 s wait, into 12 s: the wait shrinks to 1 s (9 s), then 5 s more come off the end.
    ranges = edit.keep_ranges(0.0, 26.0, [[10.0, 20.0]], 12.0)
    assert ranges == [(0.0, 10.5), (19.5, 21.0)]
    assert sum(b - a for a, b in ranges) == pytest.approx(12.0)


def test_footage_ends_before_the_next_page_loads(edit: ModuleType) -> None:
    footage = {"start": 56.4, "end": 84.9, "waits": []}
    assert edit.footage_ranges(footage, 28.0) == [(56.4, 84.9 - edit.END_MARGIN_S)]


# ── before editing ──


def sections(edit: ModuleType) -> list:
    return [
        edit.Section(1, "Title", 5.0, (edit.Caption(0.0, "Private | On your Mac"),)),
        edit.Section(2, "Ingest", 10.0, (edit.Caption(0.0, "Drop in files | Any kind."),)),
    ]


def test_a_complete_take_plans_one_shot_per_section(edit: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "demo-raw.webm").write_bytes(b"")
    (tmp_path / "beats.json").write_text(
        json.dumps({"shots": [{"id": 2, "start": 3, "end": 10, "waits": []}]})
    )
    shots, problems = edit.plan_shots(sections(edit), tmp_path)
    assert problems == []
    assert [(s.section.shot, s.start_s, s.footage is None) for s in shots] == [
        (1, 0.0, True),
        (2, 5.0, False),
    ]


def test_a_missing_take_and_unscripted_footage_are_reported(
    edit: ModuleType, tmp_path: Path
) -> None:
    _, problems = edit.plan_shots(sections(edit), tmp_path)
    assert any("no raw take" in p for p in problems)
    (tmp_path / "demo-raw.webm").write_bytes(b"")
    beats = [
        {"id": 2, "start": 0, "end": 5, "waits": []},
        {"id": 9, "start": 5, "end": 9, "waits": []},
    ]
    (tmp_path / "beats.json").write_text(json.dumps({"shots": beats}))
    _, problems = edit.plan_shots(sections(edit), tmp_path)
    assert problems == ["recorded shots with no section in the script: [9]"]


def test_the_repo_script_passes_its_own_checks(edit: ModuleType, tmp_path: Path) -> None:
    script = edit.parse_script((SCRIPTS.parent / "docs/demo/captions.md").read_text())
    (tmp_path / "demo-raw.webm").write_bytes(b"")
    cards = {1, 8}  # title and end: everything else is recorded
    beats = [
        {"id": s.shot, "start": 0, "end": 5, "waits": []} for s in script if s.shot not in cards
    ]
    (tmp_path / "beats.json").write_text(json.dumps({"shots": beats}))
    assert edit.plan_shots(script, tmp_path)[1] == []


# ── graphics and audio ──


def test_caption_and_card_text_is_escaped(edit: ModuleType) -> None:
    caption = edit.caption_html("Forms & <JSON> | Checked & ready", "")
    assert '<div class="headline">Forms &amp; &lt;JSON&gt;</div>' in caption
    assert '<div class="detail">Checked &amp; ready</div>' in caption
    assert 'class="detail"' not in edit.caption_html("Headline only", "")
    card = edit.card_html("github.com/x | Apache-2.0 & <MIT>", "")
    assert "<p>github.com/x</p><p>Apache-2.0 &amp; &lt;MIT&gt;</p>" in card


MEASURED = {
    "input_i": "-27.3",
    "input_tp": "-9.1",
    "input_lra": "6.2",
    "input_thresh": "-37.6",
    "target_offset": "0.4",
}


def test_music_is_cut_to_the_video_faded_and_normalized_in_two_passes(edit: ModuleType) -> None:
    first, second = edit.music_filter(88.0), edit.music_filter(88.0, MEASURED)
    for f in (first, second):
        assert f.startswith("atrim=duration=88.0,")
        assert f"afade=t=out:st={88.0 - edit.MUSIC_FADE_OUT_S}" in f
        assert f.index("afade") < f.index(f"loudnorm=I={edit.MUSIC_LUFS}")  # fade, then normalize
    assert first.endswith("print_format=json")  # measures only
    assert "measured_I=-27.3:" in second and "offset=0.4:linear=true" in second


def test_the_loudness_measurement_is_read_from_ffmpegs_log(edit: ModuleType) -> None:
    log = "[Parsed_loudnorm_2 @ 0x1] \n" + json.dumps(MEASURED, indent=1) + "\n[out#0/null] done\n"
    assert edit.loudness_json(log) == MEASURED


def test_playwright_is_imported_only_when_rendering(edit: ModuleType) -> None:
    assert "sync_playwright" not in vars(edit)  # CI has no `screenshots` group
