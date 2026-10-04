"""Guardian prompt format and score parsing (PLAN.md §2.4), checked against the model card's format."""

from __future__ import annotations

import pytest

from granit.verify import guardian


def test_block_matches_the_card_layout() -> None:
    block = guardian.guardian_block("Each line starts with a capital letter.")
    assert block.startswith("<no-think>As a judge agent")
    assert "\n\n### Criteria: Each line starts with a capital letter.\n\n" in block
    assert block.endswith(f"### Scoring Schema: {guardian.RESPONSE_SCHEMA}")


def test_think_mode_is_opt_in() -> None:
    assert guardian.guardian_block("x", think=True).startswith("<think>As a judge agent")


def test_groundedness_messages_end_with_the_guardian_block() -> None:
    messages = guardian.groundedness_messages("The total is $4,980.", "What is the total?")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert guardian.GROUNDEDNESS in messages[-1]["content"]
    assert [m["role"] for m in guardian.groundedness_messages("answer")] == ["assistant", "user"]


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ("<think>\n</think>\n<score> yes </score>", "yes"),
        ("<score>No</score>", "no"),
        ("<think>maybe <score>no</score></think><score>yes</score>", "yes"),
        ("I think yes", None),
        ("", None),
    ],
)
def test_parse_score(text: str, score: str | None) -> None:
    assert guardian.parse_score(text) == score
