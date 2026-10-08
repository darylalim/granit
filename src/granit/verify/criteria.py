"""The criterion registry (PLAN.md §2.4): what Guardian checks, and which way its yes/no points.

- **Built-in** criteria judge answers (``qa_turns``). They are IBM's *risk* definitions ("A text is considered ungrounded
  if…"), so ``yes`` means the problem is present: ``yes_means = "risk"``.
- **Custom** criteria judge meeting summaries. They are requirements the user writes ("Every action item names the person
  responsible"), so ``yes`` means the requirement is met: ``yes_means = "pass"``. The card says custom criteria "require
  testing": the Library shows each verdict next to the summary, so a criterion that judges badly is easy to spot and edit.

The UI and reports only ever show the derived ``passed`` flag. A custom criterion's id is a hash of its text, so editing
the text makes a new criterion and summaries get checked against it on the next verify job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from granit.verify.guardian import ANSWER_RELEVANCE, GROUNDEDNESS

YesMeans = Literal["risk", "pass"]

DEFAULT_SUMMARY_CRITERIA = ("Every action item names the person responsible for it.",)
SUMMARY_REQUEST = "Summarize this meeting: what was discussed, the decisions and the action items."


@dataclass(frozen=True)
class Criterion:
    id: str
    label: str  # short, for badges and tables
    text: str  # the ``### Criteria`` Guardian sees
    target: Literal["turn", "summary"]
    yes_means: YesMeans
    documents: bool = False  # judged against the chunks the answer was based on

    def passed(self, score: str | None) -> bool | None:
        """``yes`` / ``no`` → passed or not, by polarity; anything else is an error (None), never a pass."""
        if score not in ("yes", "no"):
            return None
        return (score == "yes") == (self.yes_means == "pass")


GROUNDED = Criterion("groundedness", "Grounded", GROUNDEDNESS, "turn", "risk", documents=True)
RELEVANT = Criterion("answer_relevance", "Answers the question", ANSWER_RELEVANCE, "turn", "risk")
TURN_CRITERIA: tuple[Criterion, ...] = (GROUNDED, RELEVANT)


def custom_criterion(text: str) -> Criterion:
    text = " ".join(text.split())
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return Criterion(f"summary:{digest}", text, text, "summary", "pass")


def summary_criteria(store: Any) -> list[Criterion]:
    """The library's summary checks: the user's list, or the defaults if they never set one."""
    texts = store.summary_criteria()
    return [custom_criterion(t) for t in (DEFAULT_SUMMARY_CRITERIA if texts is None else texts)]


def summary_text(content: str) -> str:
    """A stored summary (JSON) as the plain text Guardian judges."""
    data = json.loads(content)
    lines = [data.get("summary", "").strip()]
    if data.get("decisions"):
        lines += ["", "Decisions:", *(f"- {d}" for d in data["decisions"])]
    if data.get("action_items"):
        lines += ["", "Action items:"]
        for item in data["action_items"]:
            owner = item.get("owner") or "(no owner)"
            due = f" (due {item['due']})" if item.get("due") else ""
            lines.append(f"- {owner}: {item.get('task', '')}{due}")
    return "\n".join(lines).strip()
