"""Shared fixtures for model tests: one fixture library, ingested once per session by `granit ingest` in its own process."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from granit.store.db import Store
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures"
LIBRARY_FILES = [
    FIXTURES / "audio" / "meeting.flac",
    FIXTURES / "audio" / "vad_pauses.wav",
    FIXTURES / "documents" / "report.pdf",
    FIXTURES / "documents" / "memo.pdf",
    FIXTURES / "documents" / "invoice.png",
]


def run_ingest(data: Path, files: list[Path]) -> float:
    """`granit ingest` as its own process (as Phase A runs in production); returns its peak memory in GB."""
    proc = subprocess.run(
        [sys.executable, "-m", "granit.cli", "ingest", "--data", str(data), *map(str, files)],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"peak memory ([\d.]+) GB", proc.stdout)
    assert match, proc.stdout
    return float(match.group(1))


@pytest.fixture(scope="session")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Store, float]]:
    """(store, Phase A peak GB) for the fixture library. Measuring in the pytest process would report the whole session."""
    data = tmp_path_factory.mktemp("library")
    peak = run_ingest(data, LIBRARY_FILES)
    store = Store(data)
    yield store, peak
    store.close()
