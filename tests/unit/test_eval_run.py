"""The eval harness end to end with fake processes (M7): a tiny set, the real worker loop on fake models, a fake LLM
server, a scripted LLM and a fake judge. Plus the set format, pass criteria, comparisons, the table rule and the judge."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from granit import cli
from granit.evaluate import report
from granit.evaluate.dataset import EvalSetError, init_set, load_set
from granit.evaluate.interactive import agreement, label
from granit.evaluate.judge import judge
from granit.evaluate.run import EvalConfig, Runner, table_decision
from granit.models.phases import PhaseManager
from granit.models.server import Completion
from granit.reason.prompts import DECLINE, SUMMARY_SYSTEM
from granit.reason.qa import QA
from granit.reason.tokens import ApproxTokens
from granit.search.hybrid import Searcher
from granit.store.db import Store, sha256_of
from tests.unit.test_phases import World
from tests.unit.test_reason import completion
from tests.unit.test_worker import FIXTURES, REPORT_JSON, FakeDocuments, worker

ANSWER = "Shipments in the North reached 1,240 units in Q1 [1]."


class DocumentsWithJson(FakeDocuments):
    """Fake ingest that also writes document.json, as Docling does (the table metric reads it)."""

    def ingest(self, path: Path, out: Path, vision_tables: bool | None = None) -> Any:
        (out / "document.json").write_text(json.dumps(REPORT_JSON))
        return super().ingest(path, out, vision_tables)


class FakeLLM:
    """Answers by what's asked: summaries as JSON, the warranty question declined, anything else with a citation."""

    thinking = "low"
    counter = ApproxTokens()

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> Completion:
        if messages[0]["content"] == SUMMARY_SYSTEM:
            data = {
                "summary": "Invoice follow-up.",
                "decisions": [],
                "action_items": [
                    {"owner": "Marcus", "task": "send the revised invoice to Finance", "due": None}
                ],
            }
            return completion(json.dumps(data))
        if "warranty" in messages[-1]["content"].lower():
            return completion(DECLINE)
        return completion(ANSWER)

    def chat_json(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], **_: Any
    ) -> tuple[dict[str, Any], Any]:
        reply = self.chat(messages)
        return json.loads(reply.content), reply


def build_set(root: Path) -> Path:
    files = root / "files"
    (files / "docs").mkdir(parents=True)
    (files / "meetings").mkdir()
    shutil.copy(FIXTURES / "documents" / "report.pdf", files / "docs" / "report.pdf")
    shutil.copy(FIXTURES / "documents" / "invoice.png", files / "docs" / "invoice.png")
    shutil.copy(FIXTURES / "audio" / "vad_pauses.wav", files / "meetings" / "call.wav")
    questions = [
        {"id": "doc-1", "question": "How many units did North ship in Q1?", "category": "document",
         "expected_facts": ["1,240"], "gold_refs": [{"file": "files/docs/report.pdf", "page": 1}]},
        {"id": "none-1", "question": "What is the warranty period?", "category": "unanswerable"},
    ]  # fmt: skip
    (root / "questions.yaml").write_text(yaml.safe_dump(questions))
    for sub in ("extraction", "summaries", "transcripts", "tables"):
        (root / sub).mkdir()
    (root / "extraction" / "invoice.json").write_text(
        json.dumps(
            {
                "file": "files/docs/invoice.png",
                "schema": {
                    "type": "object",
                    "properties": {"total": {"type": "string"}, "po": {"type": "string"}},
                },
                "expected": {"total": "$4,980.00", "po": None},
            }
        )
    )
    (root / "summaries" / "call.yaml").write_text(
        yaml.safe_dump(
            {
                "file": "files/meetings/call.wav",
                "action_items": [
                    {"owner": "Marcus", "task": "send the revised invoice to Finance"}
                ],
            }
        )
    )
    (root / "transcripts" / "call.txt").write_text(
        "file: files/meetings/call.wav\nplease send the revised invoice to finance\n"
    )
    table = json.loads((FIXTURES / "documents" / "manifest.json").read_text())["report.pdf"][
        "table"
    ]
    grid = [table["header"], *table["rows"]]  # every cell the fixture's table really has
    (root / "tables" / "report.json").write_text(
        json.dumps({"file": "files/docs/report.pdf", "type": "control", "grid": grid})
    )
    return root


def fake_worker(data_dir: Path, log: Any) -> dict[str, Any]:
    store = Store(data_dir)
    try:
        w = worker(store)
        w._documents = DocumentsWithJson()
        return asdict(w.run(log=log))
    finally:
        store.close()


def fake_judge(items: list[dict[str, Any]], workdir: Path, log: Any) -> list[dict[str, Any]]:
    return [
        {"id": i["id"], "criterion": c, "score": "no", "passed": True, "raw": "<score>no</score>"}
        for i in items
        for c in ("groundedness", "answer_relevance")
    ]


def runner(root: Path, library: Path) -> Runner:
    world = World()
    return Runner(
        load_set(str(root)),
        EvalConfig(),
        library=library,
        log=lambda _: None,
        run_worker=fake_worker,
        phases_factory=lambda store: PhaseManager(
            store, server_factory=world.server, run_worker=fake_worker
        ),
        qa_factory=lambda store, config: (
            (searcher := Searcher(store)),
            QA(store, searcher, FakeLLM(), ApproxTokens(), mode="bm25"),
        ),
        judge=fake_judge,
    )


@pytest.fixture
def eval_root(tmp_path: Path) -> Path:
    return build_set(tmp_path / "set")


def test_a_full_run_scores_every_area(eval_root: Path, tmp_path: Path) -> None:
    result = runner(eval_root, tmp_path / "library").run()
    metrics = result["metrics"]
    assert metrics["retrieval"]["bm25"]["recall@8"] == 1.0
    assert metrics["answers"]["fact_coverage"] == 1.0
    assert (
        metrics["answers"]["unanswerable_declined"] == 1.0
        and metrics["answers"]["false_declines"] == 0.0
    )
    assert metrics["extraction"]["field_accuracy"] == 1.0 and metrics["extraction"]["invalid"] == 0
    assert (
        metrics["tables"]["docling"]["cell_f1"] == 1.0
        and metrics["tables"]["rule"]["default"] == "docling"
    )
    assert metrics["summaries"]["action_item_recall"] == 1.0
    assert metrics["asr"]["wer"] == 0.0
    assert metrics["guardian"]["groundedness"] == 1.0 and metrics["guardian"]["agreement"] is None
    assert result["failed_files"] == [] and set(result["timings"]) >= {
        "phase_a_s",
        "phase_b_s",
        "phase_c_s",
    }
    rows = {r["id"]: r for r in result["questions"]}
    assert rows["doc-1"]["guardian"] == {"groundedness": True, "answer_relevance": True}
    assert rows["none-1"]["declined"] and rows["none-1"]["guardian"] == {}
    # No document content in the result: no answer text, no question text.
    text = json.dumps(result)
    assert ANSWER not in text and "warranty" not in text
    # Guardian stays informational until the judge check passes.
    guardian = next(r for r in result["criteria"] if r["metric"] == "guardian.groundedness")
    assert guardian["informational"]


def test_a_second_run_reuses_the_library_but_extracts_again(
    eval_root: Path, tmp_path: Path
) -> None:
    library = tmp_path / "library"
    runner(eval_root, library).run()
    second = runner(eval_root, library).run()
    store = Store(library)
    assert len([j for j in store.jobs() if j.task == "ingest"]) == 3  # not re-ingested
    assert len([j for j in store.jobs() if j.task == "extract"]) == 2  # extracted again
    assert second["metrics"]["extraction"]["field_accuracy"] == 1.0


def test_results_are_written_and_compared(eval_root: Path, tmp_path: Path) -> None:
    result = runner(eval_root, tmp_path / "library").run()
    path = report.write(result, "public", tmp_path / "results")
    assert path.name == f"{result['date']}-{result['git_sha']}-public.json"
    worse = json.loads(path.read_text())
    worse["metrics"]["answers"]["fact_coverage"] = 0.95
    worse["metrics"]["asr"]["wer"] = 0.03
    rows = {r["metric"]: r for r in report.compare(result, worse)}
    assert rows["answers.fact_coverage"]["regression"] and rows["asr.wer"]["regression"]
    assert not rows["retrieval.bm25.recall@8"]["regression"]
    better = json.loads(path.read_text())
    better["metrics"]["answers"]["false_declines"] = (
        0.0  # fewer wrong declines is better, not a regression
    )
    better["metrics"]["guardian"]["judged"] = 99  # a count, not a quality metric
    result["metrics"]["answers"]["false_declines"] = 0.08
    rows = {r["metric"]: r for r in report.compare(result, better)}
    assert (
        not rows["answers.false_declines"]["regression"]
        and not rows["guardian.judged"]["regression"]
    )
    assert cli.main(["eval", "compare", str(path), str(path)]) == 0


def test_pass_criteria(eval_root: Path) -> None:
    metrics = {
        "retrieval": {"hybrid+rerank": {"recall@8": 0.95}},
        "asr": {"wer": 0.06},
        "guardian": {"groundedness": 0.5},
    }
    rows = {r["metric"]: r for r in report.check(metrics, "public")}
    assert rows["retrieval.hybrid+rerank.recall@8"]["pass"] is True
    assert rows["asr.wer"]["pass"] is False and rows["answers.fact_coverage"]["pass"] is None
    assert rows["guardian.groundedness"]["informational"]  # not trusted yet
    metrics["guardian"]["agreement"] = 0.9
    assert not {r["metric"]: r for r in report.check(metrics, "public")}["guardian.groundedness"][
        "informational"
    ]
    assert report.get({"a": {"b.c": {"d": 1}}}, "a.b.c.d") == 1 and report.get({}, "x.y") is None


def test_the_table_rule() -> None:
    by_type: dict[str, dict[str, float | None]] = {
        "control": {"docling": 1.0, "vision": 1.0},
        "skewed scan": {"docling": 0.6, "vision": 0.9},
    }
    close: dict[str, dict[str, float | None]] = {"control": {"docling": 1.0, "vision": 0.98}}
    assert table_decision(by_type, 0.80, 0.95) == {"default": "vision", "vision_types": []}
    assert table_decision(by_type, 0.80, 0.82) == {
        "default": "docling",
        "vision_types": ["skewed scan"],
    }
    assert table_decision(close, 1.0, 0.98)["vision_types"] == []


def test_set_validation(tmp_path: Path) -> None:
    root = build_set(tmp_path / "set")
    questions = yaml.safe_load((root / "questions.yaml").read_text())
    good = load_set(str(root))
    assert good.questions[0].gold_refs[0].source_sha256 == sha256_of(root / "files/docs/report.pdf")
    assert good.table_files() == {root / "files/docs/report.pdf"}
    for broken, message in [
        ([{**questions[0], "category": "trivia"}], "category"),
        ([{**questions[1], "expected_facts": ["x"]}], "no expected facts"),
        ([{**questions[0], "gold_refs": [{"file": "files/nope.pdf"}]}], "isn't in"),
        ([questions[0], questions[0]], "duplicate"),
    ]:
        (root / "questions.yaml").write_text(yaml.safe_dump(broken))
        with pytest.raises(EvalSetError, match=message):
            load_set(str(root))
    with pytest.raises(EvalSetError, match="eval init"):
        load_set(str(tmp_path))  # a directory without questions.yaml


def test_init_creates_a_template_without_overwriting(tmp_path: Path) -> None:
    root = init_set(str(tmp_path / "private"))
    assert (root / "files").is_dir() and "category: unanswerable" in (
        root / "questions.yaml"
    ).read_text()
    (root / "questions.yaml").write_text("[]")
    init_set(str(root))
    assert (root / "questions.yaml").read_text() == "[]"


def test_labeling_writes_gold_refs(eval_root: Path, tmp_path: Path) -> None:
    runner(eval_root, tmp_path / "library").run()
    store = Store(tmp_path / "library")
    questions = yaml.safe_load((eval_root / "questions.yaml").read_text())
    questions[0].pop("gold_refs")
    (eval_root / "questions.yaml").write_text(yaml.safe_dump(questions))
    shown: list[str] = []

    def pick_the_report(
        _: str,
    ) -> str:  # like a person: a chunk from report.pdf, plus a number out of range
        number = next(line.split(".")[0].strip() for line in shown if "[docs/report.pdf" in line)
        return f"{number}, 99"

    labeled = label(
        load_set(str(eval_root)), store, Searcher(store), ask=pick_the_report, show=shown.append
    )
    assert labeled == 1 and any("doc-1" in s for s in shown)
    (ref,) = load_set(str(eval_root)).questions[0].gold_refs
    assert (
        ref.source_sha256 == sha256_of(eval_root / "files/docs/report.pdf") and ref.page is not None
    )


def test_agreement_with_the_judge(eval_root: Path, tmp_path: Path) -> None:
    library = tmp_path / "library"
    runner(eval_root, library).run()
    replies = iter(["y", "n"])
    result = agreement(
        load_set(str(eval_root)), library, ask=lambda _: next(replies), show=lambda _: None
    )
    assert result["labeled"] == 2 and result["agreement"] == 0.5
    assert json.loads((eval_root / "judge_agreement.json").read_text())["agreement"] == 0.5


class FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        docs = "".join(d["text"] for d in kw.get("documents") or [])
        return docs + "|" + messages[-1]["content"][:20]

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text) // 4)


def test_the_judge_scores_both_criteria_and_trims_documents() -> None:
    item = {"id": "q1", "question": "Q?", "answer": "A [1].", "documents": ["x" * 40_000, "short"]}
    prompts: list[str] = []
    verdicts = judge(
        [item],
        FakeTokenizer(),
        lambda p: prompts.append(p) or "<think>\n</think><score>yes</score>",
        log=lambda _: None,
    )
    assert [(v["criterion"], v["score"], v["passed"]) for v in verdicts] == [
        ("groundedness", "yes", False),
        ("answer_relevance", "yes", False),
    ]
    assert len(prompts[0]) < 40_000  # documents dropped until the prompt fits Guardian's context
    assert prompts[1].startswith("|")  # answer relevance has no documents


def test_every_cli_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    """argparse formats help with %: a stray "%" in a help string crashes --help (found in M7)."""
    parser = cli.build_parser()
    stack = [parser]
    while stack:
        p = stack.pop()
        p.format_help()
        for action in p._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                stack.extend(action.choices.values())


def test_the_set_vocabulary_is_applied_and_retranscribes_on_change(
    eval_root: Path, tmp_path: Path
) -> None:
    (eval_root / "vocabulary.txt").write_text("Northbeam\nPriya, Elena\n")
    assert load_set(str(eval_root)).vocabulary == ["Northbeam", "Priya", "Elena"]
    library = tmp_path / "library"
    runner(eval_root, library).run()
    store = Store(library)
    assert store.vocabulary() == ["Northbeam", "Priya", "Elena"]
    audio = [s for s in store.sources() if s.kind == "audio"]
    assert audio and all(s.info["vocabulary"] == ["Northbeam", "Priya", "Elena"] for s in audio)
    ingests = len([j for j in store.jobs() if j.task == "ingest"])
    runner(eval_root, library).run()  # same list: nothing transcribed again
    assert len([j for j in Store(library).jobs() if j.task == "ingest"]) == ingests
    (eval_root / "vocabulary.txt").write_text("Northbeam\n")
    runner(eval_root, library).run()  # another list: the recordings are transcribed again
    assert len([j for j in Store(library).jobs() if j.task == "ingest"]) == ingests + len(audio)


def test_private_calibration_keeps_the_goals_in_view() -> None:
    # v38: today's real-world levels pass; the relaxed starting points show as goals (PLAN.md §4.9).
    metrics = {
        "answers": {"fact_coverage": 0.833},
        "extraction": {"field_accuracy": 0.857},
        "summaries": {"action_item_recall": 0.467},
        "asr": {"wer": 0.178},
    }
    rows = {r["metric"]: r for r in report.check(metrics, "private")}
    for path in report.PRIVATE_GOALS:
        assert rows[path]["pass"] is True and rows[path]["goal"] == report.PRIVATE_GOALS[path]
    assert rows["asr.wer"]["threshold"] == 0.18 and rows["asr.wer"]["goal"] == 0.10
    assert all(r["goal"] is None for r in report.check(metrics, "public"))
    lines = report.summary_lines(
        {
            "set": "private",
            "date": "d",
            "git_sha": "s",
            "criteria": list(rows.values()),
            "passed": False,
        }
    )
    assert any("asr.wer" in line and "(<= 0.18, goal <= 0.1)" in line for line in lines)


def test_an_area_a_set_does_not_cover_is_reported_not_failed() -> None:
    rows = [
        {"metric": "a", "pass": True, "informational": False},
        {"metric": "tables.cell_f1", "pass": None, "informational": False},  # n/a: no table cases
        {"metric": "guardian.groundedness", "pass": False, "informational": True},
    ]
    assert report.passed(rows)
    assert not report.passed([*rows, {"metric": "b", "pass": False, "informational": False}])
    assert not report.passed(
        [{"metric": "a", "pass": None, "informational": False}]
    )  # nothing measured
