"""Phase C for evaluation (PLAN.md §2.4, §4.9): Granite Guardian judges answers in its own process, then exits.

    python -m granit.evaluate.judge ITEMS.json VERDICTS.json

``ITEMS.json``: ``[{"id", "question", "answer", "documents": ["chunk text", …]}]``: one per answered question.
Each item is judged on **groundedness** (against the documents the model was given) and **answer relevance**
(against the question), in no-think mode. Both are risk definitions, so ``yes`` = problem present = fail.
``VERDICTS.json``: ``[{"id", "criterion", "score", "passed", "raw"}]``. Never loaded while the Q&A LLM runs.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from granit.config import GUARDIAN_CONTEXT_TOKENS, INGEST_MLX_CACHE_LIMIT_GB
from granit.verify.guardian import CRITERIA, judge_messages, parse_score, passed

MAX_TOKENS = 24  # no-think: empty <think></think> then <score>…</score>
PROMPT_BUDGET = GUARDIAN_CONTEXT_TOKENS - 512


Generate = Callable[[str], str]


def prompt_for(tokenizer: Any, criterion: str, item: dict[str, Any]) -> str:
    """The chat-templated prompt; documents only for groundedness, dropped from the end until the prompt fits."""
    messages = judge_messages(criterion, item["question"], item["answer"])
    documents = list(item.get("documents") or []) if criterion == "groundedness" else []
    while True:
        kwargs: dict[str, Any] = {}
        if documents:
            kwargs["documents"] = [{"doc_id": str(i), "text": t} for i, t in enumerate(documents)]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        if not documents or len(tokenizer.encode(prompt)) <= PROMPT_BUDGET:
            return prompt
        documents.pop()


def judge(
    items: list[dict[str, Any]],
    tokenizer: Any,
    generate: Generate,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    verdicts = []
    for item in items:
        for criterion in CRITERIA:
            raw = generate(prompt_for(tokenizer, criterion, item))
            score = parse_score(raw)
            verdicts.append(
                {
                    "id": item["id"],
                    "criterion": criterion,
                    "score": score,
                    "passed": passed(score),
                    "raw": raw.strip()[:200],
                }
            )
        log(
            f"judged {item['id']}: "
            + ", ".join(f"{v['criterion']}={v['score']}" for v in verdicts[-len(CRITERIA) :])
        )
    return verdicts


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m granit.evaluate.judge ITEMS.json VERDICTS.json", file=sys.stderr)
        return 2
    items = json.loads(Path(args[0]).read_text())
    start = time.perf_counter()
    import mlx.core as mx
    from mlx_lm import generate, load

    from granit.config import LOCAL_MODELS
    from granit.models.memory import footprint

    mx.set_cache_limit(int(INGEST_MLX_CACHE_LIMIT_GB * 1e9))
    guardian = LOCAL_MODELS["guardian"]
    if not guardian.is_built():
        raise SystemExit("Guardian q8 is not built: run `uv run granit models convert`")
    model, tokenizer = load(str(guardian.path))[:2]
    verdicts = judge(
        items, tokenizer, lambda prompt: generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS)
    )
    Path(args[1]).write_text(json.dumps(verdicts, indent=1))
    result = {
        "verdicts": len(verdicts),
        "seconds": round(time.perf_counter() - start, 1),
        "peak_footprint_gb": round(footprint().peak_gb, 2),
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
