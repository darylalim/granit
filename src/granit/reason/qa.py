"""Questions → answers with citations (PLAN.md §1, §2.4, §4.9).

search (hybrid + rerank) → a prompt with as many top sources as fit the 16K budget → Granite 4.2 (thinking "low") →
citations parsed back to chunks → the turn recorded in ``qa_turns`` with its per-stage ``retrieval_trace``. With no sources
at all the answer is the decline phrase, without an LLM call.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from granit.config import HUB_MODELS, SEARCH_TOP_K
from granit.models.server import Completion, Delta
from granit.reason.prompts import DECLINE, RagPrompt, is_decline, parse_citations, rag_prompt
from granit.reason.tokens import TokenCounter
from granit.search.hybrid import Hit, SearchResult


@dataclass
class Answer:
    question: str
    text: str
    declined: bool
    cited: list[Hit]  # sources the answer cites, in citation order
    sources: list[Hit]  # what the model was given ([1]..[n])
    search: SearchResult
    prompt: RagPrompt
    reply: Completion | None  # None when there was nothing to ask about
    latency: dict[str, float]
    turn_id: int | None = None

    def citation_numbers(self) -> list[int]:
        return [self.sources.index(h) + 1 for h in self.cited]


class QA:
    def __init__(
        self,
        store: Any,
        searcher: Any,
        llm: Any,
        counter: TokenCounter,
        mode: str = "hybrid+rerank",
        k: int = SEARCH_TOP_K,
    ) -> None:
        self.store, self.searcher, self.llm, self.counter = store, searcher, llm, counter
        self.mode, self.k = mode, k

    @property
    def model_id(self) -> str:
        spec = HUB_MODELS["llm"]
        return f"{spec.repo_id}@{spec.revision}"

    def ask(
        self,
        question: str,
        *,
        thinking: str | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        record: bool = True,
    ) -> Answer:
        start = time.perf_counter()
        result = self.searcher.search(question, self.mode, self.k)
        retrieval_s = time.perf_counter() - start
        prompt = rag_prompt(question, result.hits, self.counter)
        mode = thinking or getattr(self.llm, "thinking", "low")
        if not prompt.sources:
            text, reply = DECLINE, None
            if on_delta:
                on_delta(("content", DECLINE))
        else:
            reply = self.llm.chat(prompt.messages, thinking=mode, on_delta=on_delta)
            text = reply.content.strip()
        numbers = parse_citations(text, len(prompt.sources))
        answer = Answer(
            question=question,
            text=text,
            declined=is_decline(text),
            cited=[prompt.sources[n - 1] for n in numbers],
            sources=prompt.sources,
            search=result,
            prompt=prompt,
            reply=reply,
            latency={
                "retrieval_s": round(retrieval_s, 3),
                "first_token_s": round(reply.first_content_s or reply.ttft_s, 3) if reply else 0.0,
                "total_s": round(time.perf_counter() - start, 3),
            },
        )
        if record:
            answer.turn_id = self.store.record_turn(
                question=question,
                answer=text,
                declined=answer.declined,
                source_chunk_ids=[h.chunk_id for h in prompt.sources],
                cited_chunk_ids=[h.chunk_id for h in answer.cited],
                model=self.model_id,
                thinking=mode,
                retrieval={"mode": self.mode, "k": self.k, "dropped_for_budget": prompt.dropped},
                retrieval_trace=result.trace_json(),
                prompt_tokens=reply.prompt_tokens if reply else 0,
                completion_tokens=reply.completion_tokens if reply else 0,
                latency=answer.latency,
            )
        return answer
