"""Pass criteria, result files and comparisons (PLAN.md §4.9).

A result file holds metrics, config and model revisions only, **never document content**: per-question rows carry ids,
categories and numbers. Public results go to ``eval/results/`` (committed); private ones to ``data/eval/results/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from granit.config import DATA_DIR, PROJECT_ROOT

# (metric path, direction, public threshold, private threshold). Starting points from §4.9; the first full run
# calibrates them, and any later change needs a one-line reason in the PR.
CRITERIA: list[tuple[str, str, float, float]] = [
    ("retrieval.hybrid+rerank.recall@8", ">=", 0.90, 0.85),
    ("answers.fact_coverage", ">=", 0.90, 0.85),
    ("guardian.groundedness", ">=", 0.90, 0.90),
    ("answers.unanswerable_declined", ">=", 0.90, 0.80),
    ("extraction.field_accuracy", ">=", 0.98, 0.95),
    ("tables.cell_f1", ">=", 0.95, 0.90),  # the path the §4.9 decision picks
    ("summaries.action_item_recall", ">=", 0.90, 0.80),
    ("asr.wer", "<=", 0.05, 0.10),
]
JUDGE_AGREEMENT_MIN = 0.85
REGRESSION = 0.02  # 2 points
LOWER_IS_BETTER = ("asr.wer", "speed.")


def results_dir(set_name: str) -> Path:
    return (
        PROJECT_ROOT / "eval" / "results" if set_name == "public" else DATA_DIR / "eval" / "results"
    )


def get(metrics: dict[str, Any], path: str) -> Any:
    """``"retrieval.hybrid+rerank.recall@8"`` → the nested value (keys may contain dots after the first level)."""
    node: Any = metrics
    parts = path.split(".")
    while parts and isinstance(node, dict):
        for n in range(len(parts), 0, -1):
            key = ".".join(parts[:n])
            if key in node:
                node, parts = node[key], parts[n:]
                break
        else:
            return None
    return None if parts else node


def judge_trusted(metrics: dict[str, Any]) -> bool:
    agreement = (metrics.get("guardian") or {}).get("agreement")
    return agreement is not None and agreement >= JUDGE_AGREEMENT_MIN


def check(metrics: dict[str, Any], set_name: str) -> list[dict[str, Any]]:
    """Each pass criterion with its value; Guardian ones are informational until the judge check passes."""
    rows = []
    for path, op, public, private in CRITERIA:
        threshold = public if set_name == "public" else private
        value = get(metrics, path)
        informational = path.startswith("guardian.") and not judge_trusted(metrics)
        ok = None if value is None else (value >= threshold if op == ">=" else value <= threshold)
        rows.append(
            {
                "metric": path,
                "value": value,
                "op": op,
                "threshold": threshold,
                "pass": ok,
                "informational": informational,
            }
        )
    return rows


def passed(rows: list[dict[str, Any]]) -> bool:
    return all(r["pass"] for r in rows if not r["informational"])


def flatten(node: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(node, bool):
        return {}
    if isinstance(node, (int, float)):
        return {prefix: float(node)}
    if isinstance(node, dict):
        out: dict[str, float] = {}
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {}


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-metric change from run ``a`` to run ``b``; a drop of 2 points or more (a rise, for WER and times) regresses."""
    fa, fb = flatten(a["metrics"]), flatten(b["metrics"])
    rows = []
    for key in sorted(set(fa) | set(fb)):
        before, after = fa.get(key), fb.get(key)
        delta = None if before is None or after is None else round(after - before, 4)
        worse = delta is not None and (
            delta >= REGRESSION if key.startswith(LOWER_IS_BETTER) else delta <= -REGRESSION
        )
        if key.startswith("speed."):
            worse = False  # recorded, not a pass criterion in v1
        rows.append(
            {"metric": key, "before": before, "after": after, "delta": delta, "regression": worse}
        )
    return rows


def write(result: dict[str, Any], set_name: str, directory: Path | None = None) -> Path:
    out = directory or results_dir(set_name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['date']}-{result['git_sha']}-{set_name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_lines(result: dict[str, Any]) -> list[str]:
    lines = [f"eval set {result['set']} · {result['date']} · {result['git_sha']}"]
    for row in result["criteria"]:
        mark = (
            "info"
            if row["informational"]
            else ("PASS" if row["pass"] else "FAIL" if row["pass"] is False else "n/a")
        )
        lines.append(
            f"  {mark:<4}  {row['metric']:<36} {fmt(row['value']):>7}  ({row['op']} {row['threshold']})"
        )
    lines.append(f"  → {'PASSED' if result['passed'] else 'NOT PASSED'}")
    return lines
