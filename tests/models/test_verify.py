"""The M8 verify worker with the real Guardian q8 (PLAN.md §2.4), run as its own process the way the phase manager runs it.

A scratch library (not the shared session one) holds a grounded answer, the model card's contradicting answer, and two
meeting summaries: one whose action items all have owners, one with an ownerless item. Expected verdicts follow the card's
no-think examples and the custom criterion's "yes = met" polarity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from granit.models.smoke import GUARDIAN_ANSWER, GUARDIAN_DOCUMENT
from granit.store.db import NewChunk, NewExtraction, Store
from tests.conftest import ROOT

pytestmark = pytest.mark.model

DOCS = ROOT / "tests" / "fixtures" / "documents"
PEAK_GB = 13.0  # M1: 12.4 GB at an 8.2K-token check; these prompts are far shorter


def summary(store: Store, name: str, owner: str | None) -> int:
    source, _ = store.add_file(DOCS / name)
    content = {
        "summary": "The team reviewed the launch plan.",
        "decisions": ["Launch on May 4"],
        "action_items": [
            {"owner": "Priya", "task": "Send the press release", "due": "May 1"},
            {"owner": owner, "task": "Book the launch venue", "due": None},
        ],
    }
    store.replace_extraction(
        source,
        NewExtraction(
            kind="summary", format="json", content=json.dumps(content), valid=True, model="test"
        ),
    )
    return next(e["id"] for e in store.extractions(source.id) if e["kind"] == "summary")


def test_verify_worker_judges_answers_and_summaries(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    store.add_file(DOCS / "report.pdf")
    job = store.claim_next()
    assert job is not None
    (chunk_id,) = store.complete_ingest(
        job,
        [NewChunk(GUARDIAN_DOCUMENT, "text")],
        np.ones((1, 768), np.float16) / np.sqrt(768),
        "rev",
    )
    question = "When and where was Eat first shown?"

    def answer(text: str) -> int:
        return store.record_turn(
            question=question,
            answer=text,
            declined=False,
            source_chunk_ids=[chunk_id],
            cited_chunk_ids=[chunk_id],
            model="test",
            thinking="off",
            retrieval={},
            retrieval_trace={},
            prompt_tokens=None,
            completion_tokens=None,
            latency={},
        )

    good = answer(
        "Jonas Mekas first showed Eat on July 16, 1964, at the Washington Square Gallery [1]."
    )
    bad = answer(GUARDIAN_ANSWER)
    owned = summary(store, "memo.pdf", "Marcus")
    ownerless = summary(store, "invoice.png", None)
    store.enqueue_verify()
    store.close()

    proc = subprocess.run(
        [sys.executable, "-m", "granit.verify.worker", "--data", str(tmp_path / "data")],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1].removeprefix("RESULT "))
    assert result["verdicts"] == 6 and not result["failed"], proc.stdout
    assert result["peak_footprint_gb"] < PEAK_GB

    store = Store(tmp_path / "data")
    found = store.verdicts(turn_ids=[good, bad], extraction_ids=[owned, ownerless])
    passed = {
        (kind, target, r["criterion_id"].split(":")[0]): r["passed"]
        for (kind, target), rows in found.items()
        for r in rows
    }
    assert passed[("turn", good, "groundedness")] == 1, proc.stdout
    assert passed[("turn", bad, "groundedness")] == 0, proc.stdout
    assert passed[("turn", good, "answer_relevance")] == 1, proc.stdout
    assert passed[("extraction", owned, "summary")] == 1, proc.stdout
    assert passed[("extraction", ownerless, "summary")] == 0, proc.stdout
    store.close()
