"""Benchmark harness logic without models: statistics, budget check, prompt sizing, report plumbing."""

from __future__ import annotations

import sys

import pytest

from granit import fixtures
from granit.bench import run, stats, workers
from granit.models import memory


def test_percentile_matches_numpy_linear() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert stats.percentile(values, 50) == 3.0
    assert stats.percentile(values, 90) == pytest.approx(4.6)
    assert stats.percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        stats.percentile([], 50)


def test_summarize_reports_milliseconds() -> None:
    summary = stats.summarize([0.010, 0.020, 0.030])
    assert summary == {"n": 3, "p50_ms": 20.0, "p90_ms": 28.0, "min_ms": 10.0, "max_ms": 30.0}


def test_timed_runs_warmup_then_repeats() -> None:
    calls = []
    durations = stats.timed(lambda: calls.append(1), repeat=4, warmup=2)
    assert len(calls) == 6
    assert len(durations) == 4


def test_budget_check_against_the_gpu_limit() -> None:
    checks = stats.budget({"B": 22.0, "A": 11.0}, limit_gb=21.3)
    assert [c.phase for c in checks] == ["A", "B"]
    assert checks[0].within_limit
    assert not checks[1].within_limit
    assert "plan ~12.0 GB, -1.0" in checks[0].line()
    assert checks[1].line().startswith("❌ Phase B")


def test_phase_peaks_reads_each_phase() -> None:
    results = {
        "phase-a": {"memory": {"peak_footprint_gb": 11.0}},
        "phase-b": {"peak_footprint_gb": 15.0},
        "phase-c": {"memory": {"peak_footprint_gb": 10.0}},
        "switches": {},
    }
    assert run.phase_peaks(results) == {"A": 11.0, "B": 15.0, "C": 10.0}
    assert run.phase_peaks({"switches": {}}) == {}


def test_rag_messages_are_distinct_and_sized() -> None:
    a, b = run.rag_messages(1000, seed=1), run.rag_messages(1000, seed=2)
    assert a != b  # distinct prompts, so the server's prompt cache can't hit
    chars = len(a[0]["content"])
    assert 1000 * run.CHARS_PER_TOKEN <= chars <= 1000 * run.CHARS_PER_TOKEN + 800


def test_scenarios_and_workers_are_registered() -> None:
    assert list(run.SCENARIOS) == [
        "phase-a",
        "phase-b",
        "prompt-cache",
        "long-context",
        "phase-c",
        "switches",
    ]
    assert set(workers.WORKERS) == {"phase-a", "retrieval", "guardian"}


def test_worker_rejects_unknown_names(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        workers.main(["phase-z"])
    assert "invalid choice" in capsys.readouterr().err


def test_synthetic_text_is_deterministic_business_prose() -> None:
    text = fixtures.synthetic_text(2000, seed=3)
    assert text == fixtures.synthetic_text(2000, seed=3)
    assert text != fixtures.synthetic_text(2000, seed=4)
    assert len(text) >= 2000
    assert "INV-2026-" in text
    chunks = fixtures.synthetic_chunks(5, chars=300)
    assert len(set(chunks)) == 5
    assert all(len(c) == 300 for c in chunks)


@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint is macOS-only")
def test_footprint_of_this_process() -> None:
    fp = memory.footprint()
    assert 0 < fp.current_gb <= fp.peak_gb
    block = bytearray(64 * 1024 * 1024)  # touch 64 MB and see the footprint grow
    for i in range(0, len(block), 4096):
        block[i] = 1
    assert memory.footprint().current_gb >= fp.current_gb + 0.05


@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint is macOS-only")
def test_footprint_of_a_missing_process() -> None:
    with pytest.raises(OSError):
        memory.footprint(2**22 + 12345)
