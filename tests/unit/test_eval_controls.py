"""Control answers for the judge check (PLAN.md §4.9): deliberately wrong copies that Guardian should fail."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from granit.evaluate.controls import (
    EXPECTED_FAILURE,
    caught,
    corrupt_number,
    is_control,
    make_controls,
)
from granit.evaluate.dataset import load_set
from granit.evaluate.interactive import agreement, sample
from tests.unit.test_eval_run import build_set, runner


@pytest.mark.parametrize(
    ("answer", "wrong"),
    [
        ("The total due is $12,640.00 [1].", "The total due is $17,317.00 [1]."),
        ("Q2 revenue was 145 thousand USD [3].", "Q2 revenue was 199 thousand USD [3]."),
        ("It is 94.2% [1][2].", "It is 129.2% [1][2]."),
        (
            "INV-2026-0042 has 150 licenses [2, 3].",
            "INV-2026-0042 has 206 licenses [2, 3].",
        ),  # ids untouched
        (
            "Marcus sends it by Wednesday [1].",
            "Marcus sends it by Saturday [1].",
        ),  # no number: a weekday
        ("Renew for two years.", None),
    ],
)
def test_corrupt_number(answer: str, wrong: str | None) -> None:
    assert corrupt_number(answer) == wrong


def items(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"q{i}",
            "question": f"Q{i}?",
            "answer": f"Answer {i} is {10 + i} [1].",
            "documents": ["d"],
        }
        for i in range(n)
    ]


def test_controls_alternate_kinds_and_are_deterministic() -> None:
    made = make_controls(items(24), 12)
    assert len(made) == 12 and made == make_controls(items(24), 12)
    assert [c["control"] for c in made[:4]] == ["number", "swap", "number", "swap"]
    assert all(
        is_control(c["id"])
        and c["question"] == f"Q{c['id'].split('~')[0][1:]}?"  # keeps its own question
        for c in made
    )
    swap = next(c for c in made if c["control"] == "swap")
    original = next(i for i in items(24) if i["id"] == swap["id"].split("~")[0])
    assert swap["answer"] != original["answer"] and swap["question"] == original["question"]
    assert make_controls(items(1), 12) == [] and make_controls(items(24), 0) == []


def test_caught_counts_the_criterion_that_should_fail() -> None:
    controls = [{"id": "a~number", "control": "number"}, {"id": "b~swap", "control": "swap"}]
    verdicts = [
        {"id": "a~number", "criterion": "groundedness", "passed": False},  # caught
        {
            "id": "a~number",
            "criterion": "answer_relevance",
            "passed": True,
        },  # not the one that matters
        {"id": "b~swap", "criterion": "answer_relevance", "passed": True},  # missed
    ]
    assert caught(controls, verdicts) == {
        "made": 2,
        "caught": 0.5,
        "caught_number": 1.0,
        "caught_swap": 0.0,
    }


def strict_judge(work: list[dict[str, Any]], workdir: Path, log: Any) -> list[dict[str, Any]]:
    """Fails each control on the criterion it should fail; passes everything else."""
    return [
        {
            "id": i["id"],
            "criterion": c,
            "score": "x",
            "passed": not (i.get("control") and EXPECTED_FAILURE[i["control"]] == c),
            "raw": "",
        }
        for i in work
        for c in ("groundedness", "answer_relevance")
    ]


@pytest.fixture
def two_answers(tmp_path: Path) -> Path:
    root = build_set(tmp_path / "set")
    questions = yaml.safe_load((root / "questions.yaml").read_text())
    questions.append(
        {
            "id": "doc-2",
            "question": "How many units did North ship in Q1, again?",
            "category": "document",
            "expected_facts": ["1,240"],
        }
    )
    (root / "questions.yaml").write_text(yaml.safe_dump(questions))
    return root


def test_controls_are_judged_but_kept_out_of_real_pass_rates(
    two_answers: Path, tmp_path: Path
) -> None:
    r = runner(two_answers, tmp_path / "library")
    r.judge = strict_judge
    result = r.run()
    guardian = result["metrics"]["guardian"]
    assert (
        guardian["groundedness"] == 1.0
        and guardian["answer_relevance"] == 1.0
        and guardian["judged"] == 2
    )
    assert guardian["controls"]["made"] == 2 and guardian["controls"]["caught"] == 1.0
    saved = json.loads(next((tmp_path / "library").glob("verdicts-*.json")).read_text())
    assert sum(bool(i.get("control")) for i in saved["items"]) == 2


def test_the_labeling_sample_mixes_controls_in_blind(two_answers: Path, tmp_path: Path) -> None:
    r = runner(two_answers, tmp_path / "library")
    r.judge = strict_judge
    r.run()
    run = json.loads(next((tmp_path / "library").glob("verdicts-*.json")).read_text())
    picked = sample(run, 6)
    assert len(picked) == 6 and sum(is_control(v["id"]) for v in picked) == 2  # at most a third
    assert picked == sample(run, 6)  # deterministic
    shown: list[str] = []
    result = agreement(
        load_set(str(two_answers)), tmp_path / "library", ask=lambda _: "y", show=shown.append, n=6
    )
    assert not any("~" in line for line in shown)  # which answers are controls is never shown
    assert (
        result["controls"]["n"] == 2 and result["controls"]["agreement"] == 0.0
    )  # "y" on wrong answers disagrees
    assert result["real"]["n"] == 4 and result["real"]["agreement"] == 1.0


def test_sources_are_shown_whole(tmp_path: Path) -> None:
    """The 2026-10-04 check cut sources at 300 characters and hid the evidence for two correct answers."""
    root = build_set(tmp_path / "set")
    library = tmp_path / "library"
    library.mkdir()
    evidence = "Week 7: Boise handled 571 pallets."
    long_source = ("Week 1: 501 pallets. " * 50) + evidence
    run = {
        "items": [
            {
                "id": "q1",
                "question": "Boise, week 7?",
                "answer": "571 [1].",
                "documents": [long_source],
            }
        ],
        "verdicts": [
            {"id": "q1", "criterion": "groundedness", "score": "no", "passed": True, "raw": ""}
        ],
    }
    (library / "verdicts-20261004-120000.json").write_text(json.dumps(run))
    shown: list[str] = []
    agreement(load_set(str(root)), library, ask=lambda _: "y", show=shown.append)
    assert evidence in " ".join(" ".join(line.split()) for line in shown)
