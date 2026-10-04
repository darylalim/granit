"""Token counting for the 16K prompt budget (PLAN.md §3.3): ``mlx_lm.server`` doesn't limit prompts, so granit must.

``GraniteTokens`` uses the LLM's own ``tokenizer.json`` (loads in ~0.1 s; 10K tokens counted in ~10 ms). ``ApproxTokens`` is
for tests and has a deliberately pessimistic ratio (3.0 characters per token vs the measured 3.8), so it never under-counts.
"""

from __future__ import annotations

from typing import Any, Protocol

from granit.config import HUB_MODELS

MESSAGE_OVERHEAD_TOKENS = 8  # role markers and separators the chat template adds per message
TEMPLATE_RESERVE_TOKENS = (
    64  # generation prompt, thinking tags, the "{reasoning effort: low}" suffix
)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class GraniteTokens:
    def __init__(self) -> None:
        self._tokenizer: Any = None

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            from granit.models.download import local_snapshot

            path = local_snapshot(HUB_MODELS["llm"]) / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(path))
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class ApproxTokens:
    def __init__(self, chars_per_token: float = 3.0) -> None:
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 1


def messages_tokens(messages: list[dict[str, Any]], counter: TokenCounter) -> int:
    """Tokens a chat prompt will take (content + per-message overhead + a reserve for the template)."""
    return TEMPLATE_RESERVE_TOKENS + sum(
        counter.count(str(m["content"])) + MESSAGE_OVERHEAD_TOKENS for m in messages
    )
