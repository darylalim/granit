"""The Phase C verify worker (PLAN.md §2.4, M8): load Guardian once, judge what isn't verified yet, write verdicts, exit.

For each queued ``verify`` job: every answer (``qa_turns``, declines skipped) missing a verdict on a built-in criterion is
judged on **groundedness** (against the chunks the answer was based on) and **answer relevance**; every meeting summary
missing a verdict on one of the library's summary criteria is judged on those. Each target's verdicts are written in one
transaction, so a crash loses at most the target in progress. No-think mode, temperature 0; an unparseable score is stored as
an error, never a pass.

Guardian is loaded only if there is something to judge. The phase manager runs this as its own process, so its memory is
freed when it exits; it's never loaded alongside the Q&A LLM.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granit.config import (
    GUARDIAN_CONTEXT_TOKENS,
    HUB_MODELS,
    INGEST_MLX_CACHE_LIMIT_GB,
    LOCAL_MODELS,
)
from granit.store.db import Job, NewVerdict, Store
from granit.verify.criteria import (
    SUMMARY_REQUEST,
    TURN_CRITERIA,
    Criterion,
    summary_criteria,
    summary_text,
)
from granit.verify.guardian import fit_prompt, guardian_block, parse_score

MAX_TOKENS = 24  # no-think: empty <think></think> then <score>…</score>
PROMPT_BUDGET = GUARDIAN_CONTEXT_TOKENS - 512
SOURCES_REMOVED = "the passages this answer was based on were removed from the library"


@dataclass
class VerifyReport:
    done: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    verdicts: int = 0
    seconds: float = 0.0
    peak_footprint_gb: float = 0.0


def guardian_model_id() -> str:
    """The judge recorded with every verdict: the pinned source revision and the local build's quantization."""
    local = LOCAL_MODELS["guardian"]
    source = HUB_MODELS[local.source]
    return f"{source.repo_id}@{source.revision} q{local.q_bits}"


class GuardianJudge:
    """Granite Guardian 4.1 8B (local q8 build) through mlx-lm: messages + documents in, raw completion out."""

    def __init__(self) -> None:
        from mlx_lm import load

        local = LOCAL_MODELS["guardian"]
        if not local.is_built():
            raise RuntimeError("Guardian q8 is not built: run `uv run granit models convert`")
        self._model, self._tokenizer = load(str(local.path))[:2]

    def __call__(self, messages: list[dict[str, Any]], documents: list[str]) -> str:
        from mlx_lm import generate

        prompt = fit_prompt(self._tokenizer, messages, documents, PROMPT_BUDGET)
        return generate(self._model, self._tokenizer, prompt, max_tokens=MAX_TOKENS)


class VerifyWorker:
    def __init__(
        self,
        store: Store,
        judge: Any = None,
        mlx_cache_limit_gb: float | None = INGEST_MLX_CACHE_LIMIT_GB,
    ) -> None:
        self.store = store
        self._judge = judge
        self.mlx_cache_limit_gb = mlx_cache_limit_gb

    @property
    def judge(self) -> Any:
        if self._judge is None:
            self._judge = GuardianJudge()
        return self._judge

    def run(self, log: Callable[[str], None] = print) -> VerifyReport:
        start = time.perf_counter()
        self._limit_mlx_cache()
        report = VerifyReport()
        recovered = self.store.recover_interrupted()
        if recovered:
            log(f"re-queued {recovered} job(s) left running by an earlier worker")
        failed_this_run: list[int] = []
        while (
            job := self.store.claim_next(exclude=failed_this_run, tasks=("verify",))
        ) is not None:
            try:
                report.verdicts += self.verify(job, log)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.fail(job, error)
                failed_this_run.append(job.id)
                report.failed.append((job.id, error))
                log(f"✗ verify: {error}")
            else:
                self.store.complete_verify(job)
                report.done.append(job.id)
        report.seconds = round(time.perf_counter() - start, 2)
        report.peak_footprint_gb = _peak_footprint_gb()
        return report

    def verify(self, job: Job | None, log: Callable[[str], None] = print) -> int:
        """Judge every answer and summary that's missing a verdict. Returns the number of verdicts written."""
        written = 0
        for turn in self.store.turns_to_verify([c.id for c in TURN_CRITERIA]):
            verdicts = self._turn_verdicts(turn)
            self.store.record_verdicts(job, verdicts)
            written += len(verdicts)
            log(f"✓ verified answer #{turn['id']}: {_outcomes(verdicts)}")
        criteria = summary_criteria(self.store)
        for summary in self.store.summaries_to_verify([c.id for c in criteria]):
            verdicts = self._summary_verdicts(summary, criteria)
            self.store.record_verdicts(job, verdicts)
            written += len(verdicts)
            log(f"✓ verified summary of source #{summary['source_id']}: {_outcomes(verdicts)}")
        return written

    def _turn_verdicts(self, turn: Any) -> list[NewVerdict]:
        done = self._done("turn", turn["id"])
        chunk_ids = json.loads(turn["source_chunk_ids"])
        chunks = self.store.chunks_by_id(chunk_ids)
        documents = [_passage(chunks[i]) for i in chunk_ids if i in chunks]
        messages = [
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "content": turn["answer"]},
        ]
        verdicts = []
        for c in TURN_CRITERIA:
            if c.id in done:
                continue
            if c.documents and not documents:
                verdicts.append(self._error(c, SOURCES_REMOVED, turn_id=turn["id"]))
                continue
            verdicts.append(
                self._judged(c, messages, documents if c.documents else [], turn_id=turn["id"])
            )
        return verdicts

    def _summary_verdicts(self, summary: Any, criteria: list[Criterion]) -> list[NewVerdict]:
        done = self._done("extraction", summary["id"])
        messages = [
            {"role": "user", "content": SUMMARY_REQUEST},
            {"role": "assistant", "content": summary_text(summary["content"])},
        ]
        return [
            self._judged(c, messages, [], extraction_id=summary["id"])
            for c in criteria
            if c.id not in done
        ]

    def _done(self, kind: str, target_id: int) -> set[str]:
        """Criteria this answer (``turn``) or summary (``extraction``) already has a verdict on."""
        found = self.store.verdicts(**{f"{kind}_ids": [target_id]})
        return {r["criterion_id"] for r in found.get((kind, target_id), [])}

    def _judged(
        self,
        c: Criterion,
        messages: list[dict[str, Any]],
        documents: list[str],
        turn_id: int | None = None,
        extraction_id: int | None = None,
    ) -> NewVerdict:
        raw = self.judge(
            [*messages, {"role": "user", "content": guardian_block(c.text)}], documents
        )
        score = parse_score(raw)
        return NewVerdict(
            criterion_id=c.id,
            criterion=c.text,
            yes_means=c.yes_means,
            score=score if score in ("yes", "no") else None,
            passed=c.passed(score),
            error=None if score in ("yes", "no") else "unparseable score",
            model=guardian_model_id(),
            raw=raw.strip(),
            turn_id=turn_id,
            extraction_id=extraction_id,
        )

    def _error(
        self, c: Criterion, error: str, turn_id: int | None = None, extraction_id: int | None = None
    ) -> NewVerdict:
        return NewVerdict(
            criterion_id=c.id,
            criterion=c.text,
            yes_means=c.yes_means,
            score=None,
            passed=None,
            error=error,
            model=guardian_model_id(),
            turn_id=turn_id,
            extraction_id=extraction_id,
        )

    def _limit_mlx_cache(self) -> None:
        if self.mlx_cache_limit_gb is None:
            return
        try:
            import mlx.core as mx
        except ImportError:  # not on Apple Silicon (tests with a fake judge)
            return
        mx.set_cache_limit(int(self.mlx_cache_limit_gb * 1e9))


def _passage(row: Any) -> str:
    """What the answering model read: heading context, then the text (as ``Hit.passage``)."""
    return f"{row['context']}\n{row['text']}" if row["context"] else row["text"]


def _outcomes(verdicts: list[NewVerdict]) -> str:
    def word(v: NewVerdict) -> str:
        return "error" if v.passed is None else "pass" if v.passed else "fail"

    return ", ".join(f"{v.criterion_id}={word(v)}" for v in verdicts) or "nothing new"


def _peak_footprint_gb() -> float:
    try:
        from granit.models.memory import footprint

        return round(footprint().peak_gb, 2)
    except OSError:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    """``python -m granit.verify.worker --data DIR``: the Phase C process the phase manager starts.

    Progress lines go to stdout as they happen; the last line is ``RESULT <json>`` for the phase manager.
    """
    import argparse

    from granit.config import DATA_DIR

    parser = argparse.ArgumentParser(prog="python -m granit.verify.worker")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    store = Store(args.data)
    try:
        report = VerifyWorker(store).run(log=lambda line: print(line, flush=True))
    finally:
        store.close()
    result = {
        "done": report.done,
        "failed": report.failed,
        "verdicts": report.verdicts,
        "seconds": report.seconds,
        "peak_footprint_gb": report.peak_footprint_gb,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
