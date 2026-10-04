"""Pure helpers for benchmark results: percentiles, timing summaries and the §3.3 memory budget check."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# PLAN.md §3.3 estimates for each phase's peak; M1 confirms or corrects them.
PLAN_PEAK_GB = {"A": 12.0, "B": 15.8, "C": 10.5}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100), matching numpy's default method."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(seconds: Sequence[float]) -> dict[str, float]:
    """p50 / p90 / min / max in milliseconds, rounded for the report."""
    ms = [s * 1000 for s in seconds]
    return {
        "n": len(ms),
        "p50_ms": round(percentile(ms, 50), 2),
        "p90_ms": round(percentile(ms, 90), 2),
        "min_ms": round(min(ms), 2),
        "max_ms": round(max(ms), 2),
    }


def timed(fn: Callable[[], Any], repeat: int, warmup: int = 1) -> list[float]:
    """Wall-clock seconds for ``repeat`` calls of ``fn`` after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return out


@dataclass(frozen=True)
class BudgetCheck:
    phase: str
    measured_gb: float
    plan_gb: float
    limit_gb: float

    @property
    def within_limit(self) -> bool:
        return self.measured_gb < self.limit_gb

    @property
    def headroom_gb(self) -> float:
        return self.limit_gb - self.measured_gb

    def line(self) -> str:
        mark = "✅" if self.within_limit else "❌"
        delta = self.measured_gb - self.plan_gb
        return (
            f"{mark} Phase {self.phase}: peak {self.measured_gb:.1f} GB "
            f"(plan ~{self.plan_gb:.1f} GB, {delta:+.1f}); "
            f"{self.headroom_gb:.1f} GB below the {self.limit_gb:.1f} GB GPU limit"
        )


def budget(peaks_gb: dict[str, float], limit_gb: float) -> list[BudgetCheck]:
    return [
        BudgetCheck(phase, round(peak, 2), PLAN_PEAK_GB[phase], round(limit_gb, 2))
        for phase, peak in sorted(peaks_gb.items())
    ]
