"""Meeting summaries (PLAN.md §1, §3.3): transcript → summary, decisions and action items as validated JSON.

Transcripts that fit one prompt (about an hour of speech) are summarized in one call. Longer ones are split into sections at
segment boundaries, each section is summarized, and the partial summaries are combined in a final call (the 16K cap
decision, PLAN.md §3.3). The result is stored as a ``summary`` extraction for the source.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from granit.config import HUB_MODELS, LLM_PROMPT_BUDGET_TOKENS
from granit.reason.prompts import (
    SUMMARY_SCHEMA,
    combine_messages,
    split_sections,
    summary_messages,
    transcript_lines,
)
from granit.reason.tokens import TokenCounter


@dataclass
class MeetingSummary:
    data: dict[str, Any]
    sections: int
    seconds: float


def summarize(
    transcript: Any, llm: Any, counter: TokenCounter, budget: int = LLM_PROMPT_BUDGET_TOKENS
) -> MeetingSummary:
    start = time.perf_counter()
    lines = transcript_lines(transcript.segments)
    if not lines:
        empty = {
            "summary": "No speech was found in this recording.",
            "decisions": [],
            "action_items": [],
        }
        return MeetingSummary(empty, 0, 0.0)
    sections = split_sections(lines, counter, budget)
    if len(sections) == 1:
        data, _ = llm.chat_json(summary_messages(sections[0]), SUMMARY_SCHEMA)
    else:
        partials = [
            llm.chat_json(summary_messages(section, (i, len(sections))), SUMMARY_SCHEMA)[0]
            for i, section in enumerate(sections, 1)
        ]
        data, _ = llm.chat_json(combine_messages(partials), SUMMARY_SCHEMA)
    return MeetingSummary(data, len(sections), round(time.perf_counter() - start, 2))


def summarize_source(store: Any, source: Any, llm: Any, counter: TokenCounter) -> MeetingSummary:
    """Summarize an ingested recording and store the result as its ``summary`` extraction."""
    from granit.ingest.audio import Transcript
    from granit.store.db import NewExtraction

    if source.kind != "audio" or source.status != "ready":
        raise ValueError(
            f"{source.name} isn't an ingested recording (kind {source.kind}, status {source.status})"
        )
    path = store.derived_dir(source) / "transcript.json"
    transcript = Transcript.from_json(json.loads(path.read_text()))
    summary = summarize(transcript, llm, counter)
    spec = HUB_MODELS["llm"]
    store.replace_extraction(
        source,
        NewExtraction(
            kind="summary",
            format="json",
            content=json.dumps(summary.data, ensure_ascii=False),
            valid=True,
            model=f"{spec.repo_id}@{spec.revision}",
            schema=SUMMARY_SCHEMA,
        ),
    )
    return summary
