"""Prompts (PLAN.md §1, §3.3, §4.9): grounded answers with citations, declining when unanswerable, meeting summaries as JSON.

Every prompt is built to fit ``LLM_PROMPT_BUDGET_TOKENS`` (16K context minus 2K for the answer). RAG sources are added in rank
order until the budget is reached; transcripts longer than one section are summarized section by section and then combined.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from granit.config import LLM_PROMPT_BUDGET_TOKENS
from granit.reason.tokens import TokenCounter, messages_tokens

DECLINE = "Not found in your documents."

RAG_SYSTEM = f"""You are granit, an assistant that answers questions using only the numbered sources provided.
Rules:
- Use only facts stated in the sources. Never use outside knowledge or guess.
- Cite every statement with the number of the source it comes from in square brackets, like [2]. Cite only sources you used.
- If the sources don't contain the answer, reply exactly: {DECLINE}
- Answer directly, in a few sentences or a short list. Don't describe the sources or explain your reasoning.
- Copy numbers, amounts, dates, names and IDs exactly as they appear in the sources.
- When sources disagree or come from different documents, say so and cite each one."""


class PromptTooLong(ValueError):
    pass


@dataclass(frozen=True)
class RagPrompt:
    messages: list[dict[str, Any]]
    sources: list[Any]  # the hits given to the model, in [1]..[n] order
    tokens: int
    dropped: int  # hits that didn't fit in the budget


def source_block(number: int, hit: Any) -> str:
    heading = f" — {hit.context}" if getattr(hit, "context", "") else ""
    return f"[{number}] {hit.citation}{heading}\n{hit.text}"


def rag_prompt(
    question: str,
    hits: Sequence[Any],
    counter: TokenCounter,
    budget: int = LLM_PROMPT_BUDGET_TOKENS,
) -> RagPrompt:
    """The system rules + numbered sources (as many as fit) + the question."""
    question = question.strip()

    def build(blocks: list[str]) -> list[dict[str, Any]]:
        sources = "\n\n".join(blocks) if blocks else "(no sources found)"
        return [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
        ]

    if messages_tokens(build([]), counter) > budget:
        raise PromptTooLong(f"the question alone is longer than the {budget}-token prompt budget")
    blocks: list[str] = []
    used: list[Any] = []
    for hit in hits:
        candidate = [*blocks, source_block(len(blocks) + 1, hit)]
        if messages_tokens(build(candidate), counter) > budget:
            continue  # a shorter, lower-ranked source may still fit
        blocks, used = candidate, [*used, hit]
    messages = build(blocks)
    return RagPrompt(messages, used, messages_tokens(messages, counter), len(hits) - len(used))


_CITATION = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


def parse_citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers cited in an answer (``[2]``, ``[1, 3]``, ``[2–4]``), in order of first appearance, valid only."""
    found: list[int] = []
    for group in _CITATION.findall(answer):
        for part in re.split(r"\s*,\s*", group):
            if m := re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part):
                numbers = range(int(m.group(1)), int(m.group(2)) + 1)
            else:
                numbers = range(int(part), int(part) + 1)
            found.extend(n for n in numbers if 1 <= n <= n_sources and n not in found)
    return found


def is_decline(answer: str) -> bool:
    return DECLINE.lower().rstrip(".") in answer.strip().lower()


# ── meeting summaries ──

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": ["string", "null"]},
                    "task": {"type": "string"},
                    "due": {"type": ["string", "null"]},
                },
                "required": ["owner", "task", "due"],
            },
        },
    },
    "required": ["summary", "decisions", "action_items"],
}

SUMMARY_SYSTEM = f"""You summarize meeting transcripts. Return ONLY a JSON object matching this schema, nothing else:

{json.dumps(SUMMARY_SCHEMA, indent=2)}

Rules:
- "summary": 2–4 sentences on what was discussed.
- "decisions": every conclusion the group reached: something agreed, chosen, approved or ruled out. A decision that also
  creates a task goes in both lists. Empty list only if nothing was settled.
- "action_items": one item per task per person. When tasks are handed out as a list ("Ana, you take the budget; Ben, the
  schedule"), each person gets their own item with their own task. Include tasks someone was asked to do or committed to
  do, and a next meeting that was arranged ("meet next Tuesday at two", owner null).
- "owner": who will do the task, as the transcript calls them: a name, or a role when people are addressed by role
  ("industrial designer, you have…"). If no one is named, owner is the JSON value null: never write "someone",
  "everyone", "all", "team" or a description of a person.
- "due": the deadline in the transcript's own words; null if none.
- Use only what the transcript says. Don't invent owners, dates or tasks."""

COMBINE_SYSTEM = f"""You merge partial summaries of consecutive sections of one meeting into a single summary.
Return ONLY a JSON object matching this schema, nothing else:

{json.dumps(SUMMARY_SCHEMA, indent=2)}

Rules:
- "summary": 2–4 sentences covering the whole meeting.
- Keep every decision and action item. Merge duplicates; if a later section changes or cancels something from an earlier one,
  keep only the final version.
- Use only what the partial summaries say."""


def transcript_lines(segments: Sequence[Any]) -> list[str]:
    from granit.ingest.audio import timestamp

    return [f"[{timestamp(s.start)}] {s.text}" for s in segments]


def summary_messages(
    lines: Sequence[str], part: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    label = f" (part {part[0]} of {part[1]})" if part else ""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"Transcript{label}:\n\n" + "\n".join(lines)},
    ]


def combine_messages(partials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    body = "\n\n".join(
        f"Section {i}:\n{json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(partials, 1)
    )
    return [{"role": "system", "content": COMBINE_SYSTEM}, {"role": "user", "content": body}]


def split_sections(
    lines: Sequence[str], counter: TokenCounter, budget: int = LLM_PROMPT_BUDGET_TOKENS
) -> list[list[str]]:
    """Pack transcript lines into sections whose summary prompt fits ``budget`` (lines are never split)."""
    base = messages_tokens(summary_messages([], (99, 99)), counter)
    sections: list[list[str]] = []
    current: list[str] = []
    size = base
    for line in lines:
        cost = counter.count(line) + 1
        if base + cost > budget:
            raise PromptTooLong("a single transcript line is longer than the prompt budget")
        if current and size + cost > budget:
            sections.append(current)
            current, size = [], base
        current.append(line)
        size += cost
    if current:
        sections.append(current)
    return sections
