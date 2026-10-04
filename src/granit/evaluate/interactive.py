"""The two eval tasks that need a person (PLAN.md §4.9): labeling gold refs, and checking the Guardian judge.

- ``granit eval label``: for each question, the hybrid + rerank top 20 chunks; you type the numbers of the relevant ones,
  which become ``gold_refs`` (file hash + page or time range, so labels survive re-chunking). ~2 minutes per question.
- ``granit eval agreement``: Guardian's verdicts from the latest run, one at a time; you answer y / n. Agreement and
  Cohen's κ go to ``judge_agreement.json`` next to the set's questions. Guardian metrics count toward the pass criteria
  only at ≥ 85 % agreement (``report.JUDGE_AGREEMENT_MIN``).

Input and output are injectable (``ask``, ``show``), so both are unit-tested without a terminal.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from granit.evaluate import metrics as m
from granit.evaluate.dataset import EvalSet

Ask = Callable[[str], str]
Show = Callable[[str], None]
HEADER = "# granit eval set (PLAN.md §4.9). gold_refs were labeled with `granit eval label`.\n"


def label(
    eval_set: EvalSet,
    store: Any,
    searcher: Any,
    *,
    ask: Ask = input,
    show: Show = print,
    relabel: bool = False,
    top: int = 20,
) -> int:
    """Label questions without gold refs (or all with ``relabel``); returns how many were labeled. Saves after each one."""
    path = eval_set.root / "questions.yaml"
    entries = yaml.safe_load(path.read_text()) or []
    sha = {s.id: s.sha256 for s in store.sources()}
    labeled = 0
    for entry in entries:
        if entry.get("category") == "unanswerable" or (entry.get("gold_refs") and not relabel):
            continue
        hits = searcher.search(entry["question"], "hybrid+rerank", top).hits
        show(f"\n{entry['id']}: {entry['question']}")
        if entry.get("expected_facts"):
            show(f"  expected: {', '.join(map(str, entry['expected_facts']))}")
        for i, h in enumerate(hits, start=1):
            show(f"  {i:>2}. [{h.citation}] {' '.join(h.text.split())[:200]}")
        reply = (
            ask("relevant chunks (numbers; Enter = none relevant; s = skip; q = quit): ")
            .strip()
            .lower()
        )
        if reply == "q":
            break
        if reply == "s":
            continue
        refs = []
        for token in reply.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(hits):
                h = hits[int(token) - 1]
                ref: dict[str, Any] = {"source_sha256": sha[h.source_id], "file": h.source_name}
                if h.start_s is not None:
                    ref.update(start_s=round(h.start_s, 2), end_s=round(h.end_s or h.start_s, 2))
                elif h.page_start is not None:
                    ref["page"] = h.page_start
                if ref not in refs:
                    refs.append(ref)
        entry["gold_refs"] = refs
        labeled += 1
        path.write_text(
            HEADER + yaml.safe_dump(entries, sort_keys=False, width=120, allow_unicode=True)
        )
    return labeled


PLAIN = {
    "groundedness": "Is everything the answer says supported by the sources below?",
    "answer_relevance": "Does the answer actually address the question?",
}


def latest_verdicts(library: Path) -> dict[str, Any]:
    runs = sorted(library.glob("verdicts-*.json"))
    if not runs:
        raise FileNotFoundError(f"no Guardian verdicts in {library}: run `granit eval run` first")
    return json.loads(runs[-1].read_text())


def agreement(
    eval_set: EvalSet, library: Path, *, ask: Ask = input, show: Show = print, n: int = 50
) -> dict[str, Any]:
    """Label up to ``n`` verdicts by hand; save and return agreement and Cohen's κ against Guardian."""
    run = latest_verdicts(library)
    items = {i["id"]: i for i in run["items"]}
    labels = []
    for v in [v for v in run["verdicts"] if v["passed"] is not None][:n]:
        item = items[v["id"]]
        show(f"\n[{len(labels) + 1}] {v['id']} · {v['criterion']}")
        show(f"Question: {item['question']}\nAnswer: {item['answer']}")
        if v["criterion"] == "groundedness":
            for i, doc in enumerate(item["documents"], start=1):
                show(f"  source {i}: {' '.join(doc.split())[:300]}")
        reply = ask(f"{PLAIN[v['criterion']]} (y / n; s = skip; q = quit): ").strip().lower()
        if reply == "q":
            break
        if reply not in ("y", "n"):
            continue
        labels.append(
            {
                "id": v["id"],
                "criterion": v["criterion"],
                "human": reply == "y",
                "guardian": v["passed"],
            }
        )
    human, guardian = [x["human"] for x in labels], [x["guardian"] for x in labels]
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "labeled": len(labels),
        "agreement": round(
            sum(h == g for h, g in zip(human, guardian, strict=True)) / len(labels), 4
        )
        if labels
        else None,
        "kappa": None if (k := m.cohens_kappa(human, guardian)) is None else round(k, 4),
        "labels": labels,
    }
    (eval_set.root / "judge_agreement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
