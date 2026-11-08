"""scripts/demo_edit.py: fitting footage to shots, subtitles, and what's checked before editing (no ffmpeg or browser)."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up; demo_edit imports demo_voice
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def edit() -> ModuleType:
    load("demo_voice")
    return load("demo_edit")


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


# ── subtitles ──


def test_spellings_for_the_ear_are_turned_back(edit: ModuleType) -> None:
    text = "This is granite, with IBM Granite models on Apple's M L X."
    assert edit.written(text) == "This is granit, with IBM Granite models on Apple's MLX."


def shot(edit: ModuleType, n: int, text: str, start: float, clip: float) -> object:
    section = sys.modules["demo_voice"].Section(n, f"S{n}", 20.0, text)
    return edit.Shot(section, start, clip, None)


def test_cues_follow_the_voice_and_group_short_sentences(edit: ModuleType) -> None:
    shots = [
        shot(edit, 1, "This is granite.\n\nPrivate by design.", 0.0, 4.0),
        shot(edit, 2, "One. Two.", 8.0, 2.0),
    ]
    cues = edit.subtitles(shots)
    assert [c[2] for c in cues] == ["This is granit. Private by design.", "One. Two."]
    assert cues[0][:2] == (edit.LEAD_S, edit.LEAD_S + 4.0)
    assert cues[1][:2] == (8.0 + edit.LEAD_S, 8.0 + edit.LEAD_S + 2.0)


def test_long_sentences_get_cues_of_their_own_timed_by_length(edit: ModuleType) -> None:
    first = "A sentence that is long enough to fill most of the two lines a subtitle has room for."
    cues = edit.subtitles([shot(edit, 1, f"{first} Short.", 0.0, 9.0)])
    assert [c[2] for c in cues] == [first, "Short."]
    (a0, b0, _), (a1, b1, _) = cues
    assert b0 == a1 and b1 == pytest.approx(edit.LEAD_S + 9.0)
    assert (b0 - a0) > 10 * (b1 - a1)


def test_srt_numbers_and_stamps_cues(edit: ModuleType) -> None:
    assert edit.srt([(0.5, 1.467, "Hi."), (61.0, 3725.25, "Bye.")]) == (
        "1\n00:00:00,500 --> 00:00:01,467\nHi.\n\n2\n00:01:01,000 --> 01:02:05,250\nBye.\n\n"
    )


# ── before editing ──


def write_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(b"\0\0" * round(seconds * 1000))


def sections(edit: ModuleType) -> list:
    voice = sys.modules["demo_voice"]
    return [
        voice.Section(1, "Title", 8.0, "This is granite.", caption="Private | On your Mac"),
        voice.Section(2, "Ingest", 17.0, "Drop in files."),
    ]


def test_a_complete_take_plans_one_shot_per_section(edit: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "demo-raw.webm").write_bytes(b"")
    (tmp_path / "beats.json").write_text(
        json.dumps({"shots": [{"id": 2, "start": 3, "end": 10, "waits": []}]})
    )
    write_wav(tmp_path / "voice/shot-1.wav", 5.6)
    write_wav(tmp_path / "voice/shot-2.wav", 15.2)
    shots, problems = edit.plan_shots(sections(edit), tmp_path)
    assert problems == []
    assert [(s.section.shot, s.start_s, s.footage is None) for s in shots] == [
        (1, 0.0, True),
        (2, 8.0, False),
    ]
    assert shots[1].clip_s == pytest.approx(15.2)


def test_missing_inputs_and_overlong_clips_are_reported(edit: ModuleType, tmp_path: Path) -> None:
    write_wav(tmp_path / "voice/shot-2.wav", 16.8)  # 0.5 s lead + 16.8 s > 17 s
    _, problems = edit.plan_shots(sections(edit), tmp_path)
    assert any("no raw take" in p for p in problems)
    assert any("shot-1.wav" in p for p in problems)
    assert any("shot 2" in p and "runs past" in p for p in problems)


def test_a_card_needs_a_caption_and_footage_needs_a_section(
    edit: ModuleType, tmp_path: Path
) -> None:
    voice = sys.modules["demo_voice"]
    (tmp_path / "demo-raw.webm").write_bytes(b"")
    beats = [
        {"id": 2, "start": 0, "end": 5, "waits": []},
        {"id": 9, "start": 5, "end": 9, "waits": []},
    ]
    (tmp_path / "beats.json").write_text(json.dumps({"shots": beats}))
    write_wav(tmp_path / "voice/shot-1.wav", 3.0)
    write_wav(tmp_path / "voice/shot-2.wav", 3.0)
    no_caption = [voice.Section(1, "Title", 8.0, "Hi."), voice.Section(2, "Ingest", 17.0, "Files.")]
    _, problems = edit.plan_shots(no_caption, tmp_path)
    assert any("shot 1 has no footage" in p for p in problems)
    assert any("[9]" in p for p in problems)


# ── graphics ──


def test_caption_and_card_text_is_escaped(edit: ModuleType) -> None:
    assert "Forms &amp; &lt;JSON&gt;" in edit.caption_html("Forms & <JSON>", "")
    card = edit.card_html("github.com/x | Apache-2.0 & <MIT>", "")
    assert "<p>github.com/x</p><p>Apache-2.0 &amp; &lt;MIT&gt;</p>" in card


def test_the_repo_voiceover_has_a_caption_for_every_shot(edit: ModuleType) -> None:
    script = (SCRIPTS.parent / "docs/demo/voiceover.md").read_text()
    assert all(s.caption for s in sys.modules["demo_voice"].parse_script(script))


def test_playwright_is_imported_only_when_rendering(edit: ModuleType) -> None:
    assert "sync_playwright" not in vars(edit)  # CI has no `screenshots` group
