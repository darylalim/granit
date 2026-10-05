"""Control answers for checking the judge (PLAN.md §4.9 *Trusting the judge*).

Real answers mostly pass (the first public run: 23 of 24), so a judge that always said "pass" would agree with a person on
~96 % of real verdicts. Controls are deliberately wrong copies of real answers, judged alongside them:

- **number**: one number in the answer changed (``$12,640.00`` → ``$17,329.80``): **groundedness** should fail.
- **swap**: another question's answer, under this question: **answer relevance** should fail.

They give an automatic sensitivity check (share of controls the judge catches) and a balanced sample for the hand-labeled
agreement check, where they're mixed in unmarked. Real-answer pass rates never include them.
"""

from __future__ import annotations

import re
from typing import Any

EXPECTED_FAILURE = {"number": "groundedness", "swap": "answer_relevance"}
SEPARATOR = "~"  # control ids: "<question id>~<kind>"

# A number that isn't a citation marker ([1], [2, 3]) and isn't part of an identifier (INV-2026-0042, PO-48213).
NUMBER = re.compile(r"(?<![\[\w,–-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w\]–-])(?!,\s?\d+\])")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def is_control(item_id: str) -> bool:
    return SEPARATOR in item_id


def _changed(whole: str, decimals: str | None) -> str:
    value = int(whole.replace(",", ""))
    new = value + max(3, round(value * 0.37))  # clearly different, same magnitude
    text = f"{new:,}" if "," in whole else str(new)
    return text + (decimals or "")


def corrupt_number(answer: str) -> str | None:
    """The answer with its first number changed (or, without numbers, its first weekday); None if neither is present."""
    match = NUMBER.search(answer)
    if match:
        return answer[: match.start()] + _changed(match[1], match[2]) + answer[match.end() :]
    for i, day in enumerate(WEEKDAYS):
        if re.search(rf"\b{day}\b", answer):
            return re.sub(rf"\b{day}\b", WEEKDAYS[(i + 3) % 7], answer, count=1)
    return None


def make_controls(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Up to ``count`` controls from the answered items, alternating number and swap, deterministic (same set → same
    controls). A swap borrows the answer of the item halfway around the list, so it's about something else."""
    controls: list[dict[str, Any]] = []
    if len(items) < 2 or count <= 0:
        return controls
    step = max(1, len(items) // max(1, count))
    for n, i in enumerate(range(0, len(items), step)):
        if len(controls) >= count:
            break
        item = items[i]
        if n % 2 == 0 and (wrong := corrupt_number(item["answer"])) is not None:
            controls.append(
                {
                    **item,
                    "id": f"{item['id']}{SEPARATOR}number",
                    "answer": wrong,
                    "control": "number",
                }
            )
        else:
            other = items[(i + len(items) // 2) % len(items)]
            controls.append(
                {
                    **item,
                    "id": f"{item['id']}{SEPARATOR}swap",
                    "answer": other["answer"],
                    "control": "swap",
                }
            )
    return controls


def caught(controls: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of controls the judge failed on the criterion that should fail, overall and by kind."""
    by_id = {(v["id"], v["criterion"]): v["passed"] for v in verdicts}
    results: dict[str, list[bool]] = {kind: [] for kind in EXPECTED_FAILURE}
    for c in controls:
        passed = by_id.get((c["id"], EXPECTED_FAILURE[c["control"]]))
        if passed is not None:
            results[c["control"]].append(not passed)
    every = [x for r in results.values() for x in r]
    return {
        "made": len(controls),
        "caught": _rate(every),
        **{f"caught_{k}": _rate(v) for k, v in results.items()},
    }


def _rate(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None
