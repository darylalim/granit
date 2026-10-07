"""``granit eval run`` (PLAN.md §4.9): one pass over an eval set, one phase at a time.

1. **Phase A** (ingest worker process): every file of the set is added to the set's own library
   (``data/eval/libraries/<set>/``, reused between runs unless ``fresh``); table files with "Accurate tables" so both
   table paths can be scored; then one ``extract`` job per extraction case (always re-run).
2. **Phase B** (``mlx_lm.server`` via the phase manager): retrieval for every question in all four setups, an answer
   with the chosen setup, and a summary of every recording.
3. **Phase C** (Guardian process): groundedness and answer relevance of every answered question.
4. **Metrics** → a result file with numbers, config and model revisions only (``report.py``).

Processes are injectable, so ``tests/unit/test_eval_run.py`` runs the whole flow with fakes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from granit.config import (
    DATA_DIR,
    DOCUMENT_VISION_TABLES,
    HUB_MODELS,
    LOCAL_MODELS,
    PROJECT_ROOT,
    SEARCH_TOP_K,
)
from granit.evaluate import metrics as m
from granit.evaluate import report
from granit.evaluate.controls import caught, is_control, make_controls
from granit.evaluate.dataset import EvalSet
from granit.models.phases import PhaseManager, run_worker_process
from granit.search.hybrid import MODES, Hit
from granit.store.db import Source, Store

LIBRARIES = DATA_DIR / "eval" / "libraries"
TABLE_WIN = 0.03  # §4.9: Vision must beat Docling by 3 points of cell F1


@dataclass
class EvalConfig:
    retrieval: str = "hybrid+rerank"
    k: int = SEARCH_TOP_K
    thinking: str = "low"
    judge: bool = True
    fresh: bool = False
    controls: int = (
        12  # deliberately wrong answers judged alongside the real ones (evaluate/controls.py)
    )


Log = Callable[[str], None]


def run_judge_process(items: list[dict[str, Any]], workdir: Path, log: Log) -> list[dict[str, Any]]:
    """Phase C: ``python -m granit.evaluate.judge`` in its own process (its memory is freed when it exits)."""
    items_path, out_path = workdir / "judge-items.json", workdir / "verdicts.json"
    items_path.write_text(json.dumps(items))
    proc = subprocess.Popen(
        [sys.executable, "-m", "granit.evaluate.judge", str(items_path), str(out_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith(("judged", "RESULT")):
            log(line)
        else:
            tail = [*tail[-20:], line]
    if proc.wait() != 0 or not out_path.is_file():
        raise RuntimeError("the Guardian judge failed: " + " | ".join(tail[-3:]))
    return json.loads(out_path.read_text())


def default_qa(store: Store, config: EvalConfig) -> tuple[Any, Any]:
    from granit.reason.llm import LLMClient
    from granit.reason.qa import QA
    from granit.reason.tokens import GraniteTokens
    from granit.search.embed import Embedder
    from granit.search.hybrid import Searcher
    from granit.search.rerank import Reranker
    from granit.search.vectors import VectorIndex

    embedder = Embedder().load()
    searcher = Searcher(store, embedder, VectorIndex(store, embedder.revision), Reranker().load())
    searcher.warm_up()
    qa = QA(
        store,
        searcher,
        LLMClient(thinking=config.thinking),
        GraniteTokens(),
        mode=config.retrieval,
        k=config.k,
    )
    return searcher, qa


def table_decision(
    by_type: dict[str, dict[str, float | None]], docling: float, vision: float
) -> dict[str, Any]:
    """The §4.9 rule, agreed before the run: Vision by ≥ 3 points overall → Vision; on some types only → those types
    (if they can be detected at ingest); otherwise Docling."""
    if vision - docling >= TABLE_WIN:
        return {"default": "vision", "vision_types": []}
    wins = sorted(
        t
        for t, s in by_type.items()
        if s["docling"] is not None
        and s["vision"] is not None
        and s["vision"] - s["docling"] >= TABLE_WIN
    )
    return {"default": "docling", "vision_types": wins}


@dataclass
class TableScore:
    name: str
    type: str
    docling: float
    vision: float
    default: float
    docling_s: float | None = None
    vision_s: float | None = None


class Runner:
    def __init__(
        self,
        eval_set: EvalSet,
        config: EvalConfig | None = None,
        *,
        library: Path | None = None,
        log: Log = print,
        run_worker: Callable[[Path, Log], dict[str, Any]] = run_worker_process,
        phases_factory: Callable[[Store], Any] | None = None,
        qa_factory: Callable[[Store, EvalConfig], tuple[Any, Any]] = default_qa,
        judge: Callable[
            [list[dict[str, Any]], Path, Log], list[dict[str, Any]]
        ] = run_judge_process,
    ) -> None:
        self.set = eval_set
        self.config = config or EvalConfig()
        self.library = library or LIBRARIES / eval_set.name
        self.log = log
        self.run_worker = run_worker
        self.phases_factory = phases_factory or (
            lambda store: PhaseManager(store, idle_before_ingest_s=1e9)
        )
        self.qa_factory = qa_factory
        self.judge = judge

    # the run

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.config.fresh:
            shutil.rmtree(self.library, ignore_errors=True)
        store = Store(self.library)
        try:
            timings: dict[str, float] = {}
            extract_jobs = self._queue(store)
            if store.queued_count():
                self.log(f"Phase A: ingesting and extracting ({store.queued_count()} jobs)…")
                start = time.perf_counter()
                self.run_worker(self.library, self.log)
                timings["phase_a_s"] = round(time.perf_counter() - start, 1)
            sources = {s.sha256: s for s in store.sources()}
            failed = [s.name for s in sources.values() if s.status != "ready"]

            self.log("Phase B: retrieval, answers and summaries…")
            start = time.perf_counter()
            phases = self.phases_factory(store)
            phases.start()
            try:
                searcher, qa = self.qa_factory(store, self.config)
                rows, items = self._questions(store, searcher, qa)
                summaries = self._summaries(store, sources, qa)
            finally:
                phases.stop()
            timings["phase_b_s"] = round(time.perf_counter() - start, 1)

            verdicts: list[dict[str, Any]] = []
            controls = make_controls(items, self.config.controls) if self.config.judge else []
            if self.config.judge and items:
                self.log(
                    f"Phase C: Guardian judges {len(items)} answers and {len(controls)} control answers…"
                )
                start = time.perf_counter()
                verdicts = self.judge(items + controls, self.library, self.log)
                timings["phase_c_s"] = round(time.perf_counter() - start, 1)
                (self.library / f"verdicts-{datetime.now():%Y%m%d-%H%M%S}.json").write_text(
                    json.dumps({"items": items + controls, "verdicts": verdicts}, indent=1)
                )

            metrics = {
                "retrieval": self._retrieval(rows),
                "answers": self._answers(rows),
                "guardian": self._guardian(verdicts, controls),
                "extraction": self._extraction(store, sources, extract_jobs),
                "tables": self._tables(store, sources),
                "summaries": summaries,
                "asr": self._asr(store, sources),
                "speed": self._speed(rows),
            }
            for row in rows:
                row["guardian"] = {
                    v["criterion"]: v["passed"] for v in verdicts if v["id"] == row["id"]
                }
            timings["total_s"] = round(time.perf_counter() - started, 1)
            criteria = report.check(metrics, self.set.name)
            return {
                "set": self.set.name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "git_sha": git_sha(),
                "config": {
                    "retrieval": self.config.retrieval,
                    "k": self.config.k,
                    "thinking": self.config.thinking,
                    "judge": self.config.judge,
                    "models": model_revisions(),
                },
                "metrics": metrics,
                "criteria": criteria,
                "passed": report.passed(criteria),
                "failed_files": failed,
                "timings": timings,
                "questions": rows,
            }
        finally:
            store.close()

    # Phase A

    def _queue(self, store: Store) -> dict[str, int]:
        tables = self.set.table_files()
        for path in self.set.files():
            params = {"vision_tables": True} if path in tables else None
            store.add_file(path, name=str(path.relative_to(self.set.files_dir)), params=params)
        sources = {s.sha256: s for s in store.sources()}
        from granit.store.db import sha256_of

        jobs = {}
        for case in self.set.extractions:
            jobs[case.name] = store.enqueue_extract(sources[sha256_of(case.file)], case.schema).id
        return jobs

    # Phase B

    def _questions(
        self, store: Store, searcher: Any, qa: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sha = {s.id: s.sha256 for s in store.sources()}

        def located(h: Hit) -> m.Located:
            return m.Located(sha[h.source_id], h.page_start, h.page_end, h.start_s, h.end_s)

        rows, items = [], []
        for q in self.set.questions:
            retrieval = {}
            for mode in MODES:
                ranked = [located(h) for h in searcher.search(q.question, mode, self.config.k).hits]
                retrieval[mode] = {
                    "recall": m.recall_at_k(ranked, q.gold_refs, self.config.k),
                    "rr": m.reciprocal_rank(ranked, q.gold_refs, self.config.k),
                }
            answer = qa.ask(q.question, thinking=self.config.thinking)
            answerable = q.category != "unanswerable"
            rows.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "retrieval": retrieval,
                    "facts": m.fact_coverage(q.expected_facts, answer.text) if answerable else None,
                    "facts_missing": sum(
                        not m.fact_found(f, answer.text) for f in q.expected_facts
                    ),
                    "declined": answer.declined,
                    "cited": len(answer.cited),
                    "citation_precision": m.citation_precision(
                        [located(h) for h in answer.cited], q.gold_refs
                    ),
                    "latency": answer.latency,
                    "turn_id": answer.turn_id,
                }
            )
            if not answer.declined:
                items.append(
                    {
                        "id": q.id,
                        "question": q.question,
                        "answer": answer.text,
                        "documents": [h.passage for h in answer.sources],
                    }
                )
            mark = "declined" if answer.declined else f"facts {rows[-1]['facts']}"
            self.log(f"  {q.id}: recall {retrieval[self.config.retrieval]['recall']} · {mark}")
        return rows, items

    def _summaries(self, store: Store, sources: dict[str, Source], qa: Any) -> dict[str, Any]:
        from granit.ingest.audio import Transcript
        from granit.reason.meetings import summarize_source
        from granit.store.db import sha256_of

        recalls, precisions, decisions = [], [], []
        for case in self.set.summaries:
            source = sources.get(sha256_of(case.file))
            if source is None or source.status != "ready":
                recalls.append(0.0)
                continue
            summary = summarize_source(store, source, qa.llm, qa.counter)
            transcript = Transcript.from_json(
                json.loads((store.derived_dir(source) / "transcript.json").read_text())
            )
            recall, precision = m.action_items(
                case.action_items, summary.data["action_items"], transcript.text
            )
            recalls.append(recall)
            precisions.append(precision)
            for d in case.decisions:
                decisions.append(
                    any(
                        m.item_matches({"owner": None, "task": d}, {"owner": None, "task": got})
                        for got in summary.data["decisions"]
                    )
                )
            self.log(f"  summary {case.name}: action-item recall {recall}, precision {precision}")
        return {
            "action_item_recall": m.mean(recalls),
            "action_item_precision": m.mean(precisions),
            "decision_recall": m.mean([float(d) for d in decisions]),
        }

    # metrics

    def _retrieval(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        k = self.config.k
        return {
            mode: {
                f"recall@{k}": m.mean([r["retrieval"][mode]["recall"] for r in rows]),
                f"mrr@{k}": m.mean([r["retrieval"][mode]["rr"] for r in rows]),
            }
            for mode in MODES
        }

    def _answers(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [r for r in rows if r["category"] != "unanswerable"]
        unanswerable = [r for r in rows if r["category"] == "unanswerable"]
        return {
            "fact_coverage": m.mean([r["facts"] for r in answerable]),
            "citation_precision": m.mean([r["citation_precision"] for r in answerable]),
            "unanswerable_declined": m.mean([float(r["declined"]) for r in unanswerable]),
            "false_declines": m.mean([float(r["declined"]) for r in answerable]),
            "by_category": {
                c: {"fact_coverage": m.mean([r["facts"] for r in answerable if r["category"] == c])}
                for c in ("document", "meeting", "cross-source")
            },
        }

    def _guardian(
        self, verdicts: list[dict[str, Any]], controls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"controls": caught(controls, verdicts)}
        verdicts = [
            v for v in verdicts if not is_control(v["id"])
        ]  # pass rates are about real answers only
        for criterion in ("groundedness", "answer_relevance"):
            scored = [
                v["passed"]
                for v in verdicts
                if v["criterion"] == criterion and v["passed"] is not None
            ]
            out[criterion] = m.mean([float(p) for p in scored])
        out["errors"] = sum(v["passed"] is None for v in verdicts)
        out["judged"] = len({v["id"] for v in verdicts})
        agreement = self.set.root / "judge_agreement.json"
        out["agreement"] = (
            json.loads(agreement.read_text()).get("agreement") if agreement.is_file() else None
        )
        return out

    def _extraction(
        self, store: Store, sources: dict[str, Source], jobs: dict[str, int]
    ) -> dict[str, Any]:
        correct = total = all_correct = invalid = 0
        per_doc = {}
        for case in self.set.extractions:
            row = next((e for e in store.extractions_for_job(jobs[case.name])), None)
            got = json.loads(row["content"]) if row is not None and row["content"] else {}
            if row is None or not row["valid"]:
                invalid += 1
            fields = m.field_matches(case.expected, got)
            correct += sum(fields.values())
            total += len(fields)
            all_correct += all(fields.values())
            per_doc[case.name] = sorted(f for f, ok in fields.items() if not ok)
        n = len(self.set.extractions)
        return {
            "field_accuracy": round(correct / total, 4) if total else None,
            "documents_all_correct": round(all_correct / n, 4) if n else None,
            "invalid": invalid,
            "wrong_fields": {k: v for k, v in per_doc.items() if v},
        }

    def _tables(self, store: Store, sources: dict[str, Source]) -> dict[str, Any]:
        from granit.store.db import sha256_of

        if not self.set.tables:
            return {}
        scores = []
        for case in self.set.tables:
            source = sources.get(sha256_of(case.file))
            grids = table_grids(store, source) if source else TableGrids([], [], [], {})
            scores.append(
                TableScore(
                    case.name,
                    case.type,
                    round(m.cell_f1(case.grid, grids.docling), 4),
                    round(m.cell_f1(case.grid, grids.vision), 4),
                    round(m.cell_f1(case.grid, grids.default), 4),
                    grids.seconds.get("docling_s"),
                    grids.seconds.get("vision_s"),
                )
            )
            last = scores[-1]
            self.log(
                f"  table {last.name}: docling {last.docling}, vision {last.vision}, default {last.default}"
            )
        by_type: dict[str, dict[str, float | None]] = {}
        for t in sorted({s.type for s in scores}):
            of_type = [s for s in scores if s.type == t]
            by_type[t] = {
                "docling": m.mean([s.docling for s in of_type]),
                "vision": m.mean([s.vision for s in of_type]),
                "default": m.mean([s.default for s in of_type]),
            }
        docling = m.mean([s.docling for s in scores]) or 0.0
        vision = m.mean([s.vision for s in scores]) or 0.0
        default = m.mean([s.default for s in scores]) or 0.0
        return {
            "docling": {"cell_f1": docling, "s_per_table": m.mean([s.docling_s for s in scores])},
            "vision": {"cell_f1": vision, "s_per_table": m.mean([s.vision_s for s in scores])},
            "default": {"cell_f1": default},
            "cell_f1": vision if DOCUMENT_VISION_TABLES else default,  # what ingest does today
            "by_type": by_type,
            "rule": table_decision(by_type, docling, vision),  # §4.9, Docling alone vs Vision
            "per_table": {
                s.name: {"docling": s.docling, "vision": s.vision, "default": s.default}
                for s in scores
            },
        }

    def _asr(self, store: Store, sources: dict[str, Source]) -> dict[str, Any]:
        from granit.evaluate.asr import wer
        from granit.store.db import sha256_of

        rates = []
        for case in self.set.transcripts:
            source = sources.get(sha256_of(case.file))
            path = store.derived_dir(source) / "transcript.json" if source else None
            if path is None or not path.is_file():
                rates.append(1.0)
                continue
            hypothesis = " ".join(s["text"] for s in json.loads(path.read_text())["segments"])
            rates.append(wer(case.text, hypothesis))
            self.log(f"  transcript {case.name}: WER {rates[-1]:.3f}")
        return {"wer": m.mean(rates), "recordings": len(rates)}

    def _speed(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        answered = [r for r in rows if r["latency"].get("first_token_s")]
        first = [r["latency"]["first_token_s"] for r in answered]
        total = [r["latency"]["total_s"] for r in rows]
        return {
            "first_token_p50_s": m.percentile(first, 0.5),
            "first_token_p90_s": m.percentile(first, 0.9),
            "answer_p50_s": m.percentile(total, 0.5),
            "answer_p90_s": m.percentile(total, 0.9),
        }


@dataclass
class TableGrids:
    """One document's tables three ways (all rows concatenated in order), for the §4.9 comparison."""

    docling: list[
        list[str]
    ]  # Granite-Docling alone (empty where it found a table but transcribed nothing)
    default: list[
        list[str]
    ]  # what ingest keeps by default: Docling, with Vision for the tables Docling left empty
    vision: list[list[str]]  # Vision on every table ("Accurate tables")
    seconds: dict[str, float]


def table_grids(store: Store, source: Source) -> TableGrids:
    out = store.derived_dir(source)
    own: list[list[list[str]]] = []
    if (out / "docling_tables.json").is_file():
        own = json.loads((out / "docling_tables.json").read_text())
    elif (out / "document.json").is_file():  # ingested before docling_tables.json existed
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.model_validate_json((out / "document.json").read_text())
        own = [[[cell.text for cell in row] for row in t.data.grid] for t in doc.tables]
    extractions = [
        e for e in store.extractions(source.id) if e["kind"] == "table" and e["format"] == "html"
    ]
    vision_tables = [m.html_grid(e["content"]) for e in extractions]
    docling = [row for grid in own for row in grid]
    vision = [row for grid in vision_tables for row in grid]
    default: list[list[str]] = []
    for i, grid in enumerate(own):
        empty = not any(cell.strip() for row in grid for cell in row)
        default += vision_tables[i] if empty and i < len(vision_tables) else grid
    seconds = source.info.get("seconds", {})
    timing: dict[str, float] = {}
    if "docling" in seconds:
        timing["docling_s"] = round(seconds["docling"] / max(len(own), 1), 2)
    if "tables" in seconds and extractions:
        timing["vision_s"] = round(seconds["tables"] / len(extractions), 2)
    return TableGrids(docling, default, vision, timing)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_revisions() -> dict[str, str]:
    revisions = {key: f"{spec.repo_id}@{spec.revision[:7]}" for key, spec in HUB_MODELS.items()}
    for key, local in LOCAL_MODELS.items():
        revisions[key] = f"{local.path.name} (built from {local.source})"
    return revisions
