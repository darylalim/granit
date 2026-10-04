"""OpenAI-compatible client for ``mlx_lm.server`` (Phase B) with Granite 4.2's thinking controls.

Thinking modes (M5 measurements on a ~3K-token RAG prompt):
- ``"low"`` (default, PLAN.md §6): ``reasoning_effort: "low"``: ~1.5 s of thinking, then a short answer (11 s total).
- ``"off"``: no thinking, but the model tended to reason out loud in the answer itself and ran to the token limit (32 s).
- ``"on"``: full thinking; used the whole 700-token budget without answering. For hard questions, opt-in per question.
JSON tasks use ``"off"`` with a strict output instruction and one repair attempt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from granit.config import LLM_HOST, LLM_MAX_OUTPUT_TOKENS, LLM_PORT
from granit.models.server import Completion, Delta, stream_chat

THINKING = {
    "off": {"enable_thinking": False},
    "low": {"reasoning_effort": "low"},
    "on": {},
}


class InvalidJSON(ValueError):
    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass
class LLMClient:
    base_url: str = f"http://{LLM_HOST}:{LLM_PORT}"
    thinking: str = "low"
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    timeout: float = 600

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: str | None = None,
        max_tokens: int | None = None,
        on_delta: Callable[[Delta], None] | None = None,
    ) -> Completion:
        mode = thinking or self.thinking
        if mode not in THINKING:
            raise ValueError(f"thinking must be one of {list(THINKING)}")
        return stream_chat(
            self.base_url,
            messages,
            max_tokens=max_tokens or self.max_tokens,
            template_kwargs=THINKING[mode],
            on_delta=on_delta,
            timeout=self.timeout,
        )

    def chat_json(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], *, thinking: str = "off"
    ) -> tuple[dict[str, Any], Completion]:
        """Ask for JSON matching ``schema``; on a parse or schema error, show the model its error once and retry."""
        reply = self.chat(messages, thinking=thinking)
        try:
            return parse_json(reply.content, schema), reply
        except InvalidJSON as error:
            repair = [
                *messages,
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": f"That wasn't valid: {error}. Return ONLY the corrected JSON object, nothing else.",
                },
            ]
            retry = self.chat(repair, thinking=thinking)
            return parse_json(retry.content, schema), retry


def parse_json(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Extract the JSON object from a reply and validate it against ``schema``."""
    from jsonschema.validators import validator_for

    from granit.ingest.vision import VisionOutputError, parse_json_object

    try:
        data = parse_json_object(text)
    except VisionOutputError as exc:
        raise InvalidJSON(str(exc), text) from exc
    errors = sorted(
        validator_for(schema)(schema).iter_errors(data), key=lambda e: list(e.absolute_path)
    )
    if errors:
        first = errors[0]
        where = "/".join(map(str, first.absolute_path)) or "(root)"
        raise InvalidJSON(f"{where}: {first.message}", text)
    return data
