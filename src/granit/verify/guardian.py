"""Guardian prompt format and score parsing (PLAN.md §2.4), from the Granite Guardian 4.1 model card.

The last user message is the *guardian block*: a think / no-think instruction, ``### Criteria: …`` and
``### Scoring Schema: …``. Retrieved chunks go in through the chat template's ``documents=`` argument.
"""

from __future__ import annotations

import re
from typing import Any

NO_THINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given judging criteria using "
    "all available information, including conversations, documents, and tools. Provide your score "
    "immediately without explanation. Output empty <think>\n</think> tags followed by your score in "
    "<score></score> tags."
)
THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the given judging "
    "criteria. Use all available information, including conversations, documents, and tools. Reason "
    "through the evidence step by step before providing your score. Wrap your reasoning in "
    "<think></think> tags, then provide your score in <score></score> tags."
)
RESPONSE_SCHEMA = (
    "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
)

# Built-in criteria (risk definitions: "yes" means the problem is present).
GROUNDEDNESS = (
    "A text is considered ungrounded or unfaithful if it includes information lacking support from, or "
    "directly contradicting, the provided document(s). This risk arises when the text fabricates details, "
    "misinterprets the content, or makes unsupported extrapolations beyond what is explicitly stated in "
    "the document(s)."
)

ANSWER_RELEVANCE = (
    "A text is considered inadequate if it fails to address or adequately respond to the posed query. This includes "
    "providing off-topic information, misinterpreting the query, or omitting key details requested in the query. "
    "Information, even if factually sound, is irrelevant if it fails to directly answer or meet the specific intent of "
    "the query."
)
# Both are risk definitions: "yes" means the problem is present, so an answer passes on "no".
CRITERIA = {"groundedness": GROUNDEDNESS, "answer_relevance": ANSWER_RELEVANCE}


def guardian_block(criteria: str, *, think: bool = False, schema: str = RESPONSE_SCHEMA) -> str:
    instruction = THINK if think else NO_THINK
    return f"{instruction}\n\n### Criteria: {criteria}\n\n### Scoring Schema: {schema}"


def groundedness_messages(answer: str, question: str | None = None) -> list[dict[str, Any]]:
    """Chat messages judging whether ``answer`` is grounded; pass chunks as ``documents=``."""
    messages: list[dict[str, Any]] = []
    if question:
        messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": guardian_block(GROUNDEDNESS)})
    return messages


def judge_messages(criterion: str, question: str, answer: str) -> list[dict[str, Any]]:
    """A question and its answer, judged on one of ``CRITERIA`` (pass retrieved chunks as ``documents=``)."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": guardian_block(CRITERIA[criterion])},
    ]


def fit_prompt(
    tokenizer: Any, messages: list[dict[str, Any]], documents: list[str], budget: int
) -> str:
    """The chat-templated prompt, with ``documents`` dropped from the end until it fits ``budget`` tokens."""
    documents = list(documents)
    while True:
        kwargs: dict[str, Any] = {}
        if documents:
            kwargs["documents"] = [{"doc_id": str(i), "text": t} for i, t in enumerate(documents)]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        if not documents or len(tokenizer.encode(prompt)) <= budget:
            return prompt
        documents.pop()


def passed(score: str | None) -> bool | None:
    """Risk criteria: "no" (problem absent) passes, "yes" fails, anything else is an error (None)."""
    return {"no": True, "yes": False}.get(score or "")


def parse_score(text: str) -> str | None:
    """Strip any reasoning trace and read ``<score>``; None when unparseable (stored as an error)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"<score>\s*(.*?)\s*</score>", cleaned, flags=re.DOTALL)
    return match.group(1).strip().lower() if match else None
