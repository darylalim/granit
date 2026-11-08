"""scripts/demo_voice.py: parsing the voiceover script and `say`'s voice list, fitting clips to shots."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def voice() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_voice", ROOT / "scripts/demo_voice.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_voice"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


SAY_VOICES = """\
Albert              en_US    # Hello! My name is Albert.
Bad News            en_US    # Hello! My name is Bad News.
Eddy (English (UK)) en_GB    # Hello! My name is Eddy.
Zoe (Premium)       en_US    # Hello! My name is Zoe.
Amélie              fr_CA    # Bonjour, je m’appelle Amélie.
"""


def test_voice_names_keep_their_spaces_and_brackets(voice: ModuleType) -> None:
    names = voice.installed_voices(SAY_VOICES)
    assert names == {"Albert", "Bad News", "Eddy (English (UK))", "Zoe (Premium)", "Amélie"}


def test_a_voice_name_alone_isnt_a_match(voice: ModuleType) -> None:
    assert "Zoe" not in voice.installed_voices(SAY_VOICES)  # only "Zoe (Premium)" is installed


SCRIPT = """\
# Demo voiceover

Intro text that isn't read.

## 1 · Title · 10 s

This is granite.

## 2 · Ask · 45 s

> A note for the editor.

Now a question
that spans two lines.
"""


def test_sections_carry_shot_name_slot_and_joined_text(voice: ModuleType) -> None:
    sections = voice.parse_script(SCRIPT)
    assert [(s.shot, s.name, s.slot_s, s.text) for s in sections] == [
        (1, "Title", 10.0, "This is granite."),
        (2, "Ask", 45.0, "Now a question that spans two lines."),
    ]


@pytest.mark.parametrize(
    ("script", "problem"),
    [
        ("## 1 · Title\n\nText.\n", "heading"),
        ("## 1 · Title · 10 s\n\n> only a note\n", "no text"),
        ("## 2 · B · 5 s\n\nB.\n\n## 1 · A · 5 s\n\nA.\n", "in order"),
        ("## 1 · A · 5 s\n\nA.\n\n## 1 · B · 5 s\n\nB.\n", "in order"),
        ("# Just a title\n", "in order"),
    ],
)
def test_malformed_scripts_are_refused(voice: ModuleType, script: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        voice.parse_script(script)


def test_the_repo_script_parses_into_the_eight_shots(voice: ModuleType) -> None:
    sections = voice.parse_script((ROOT / "docs/demo/voiceover.md").read_text())
    assert [s.shot for s in sections] == list(range(1, 9))
    assert "granit " not in " ".join(s.text for s in sections)  # spelled for the ear: "granite"


def test_a_clip_fits_up_to_its_slot(voice: ModuleType) -> None:
    section = voice.Section(6, "Ask", 45.0, "one two three")
    assert voice.fit(section, 45.0).fits
    assert not voice.fit(section, 45.01).fits
    assert voice.fit(section, 12.345).words == 3


def test_blank_lines_split_paragraphs_and_notes_dont(voice: ModuleType) -> None:
    script = "## 4 · Speakers · 35 s\n\nFirst thought,\nstill first.\n\n> a note\n\n\nSecond thought.\n> note\nStill second.\n"
    (section,) = voice.parse_script(script)
    assert section.text == "First thought, still first.\n\nSecond thought. Still second."


def test_paragraphs_are_read_with_a_pause_between_them(voice: ModuleType) -> None:
    assert (
        voice.spoken("One.\n\nTwo.\n\nThree.", 450) == "One. [[slnc 450]] Two. [[slnc 450]] Three."
    )
    assert voice.spoken("One.\n\nTwo.", 0) == "One. Two."
    assert voice.spoken("Only one.", 450) == "Only one."


def test_pause_commands_arent_counted_as_words(voice: ModuleType) -> None:
    section = voice.Section(1, "Title", 10.0, "This is granite.\n\nPrivate by design.")
    assert voice.fit(section, 5.0).words == 6


def test_a_caption_note_is_kept_but_not_read(voice: ModuleType) -> None:
    script = "## 6 · Ask · 28 s\n\n> Caption: Cited answers, fully offline\n> just a note\n\nNow a question.\n"
    (section,) = voice.parse_script(script)
    assert (section.caption, section.text) == ("Cited answers, fully offline", "Now a question.")
