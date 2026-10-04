"""Store + search golden tests on the Mac (``uv run pytest -m model``), M4: real worker, real retrieval.

The ingest worker processes every fixture (audio and documents) into a fresh library; then real hybrid search (BM25 +
trigram IDs + Granite Embedding on MPS → RRF → Granite Reranker) must find the right chunk for each kind of question.
Thresholds: the expected source/element within the top 3; memory and latency ceilings below.
"""

from __future__ import annotations

import statistics
import time

import pytest

from granit.search.embed import Embedder
from granit.search.hybrid import Hit, Searcher
from granit.search.rerank import Reranker
from granit.search.vectors import VectorIndex
from granit.store.db import Store
from tests.models.conftest import LIBRARY_FILES

pytestmark = pytest.mark.model

FILES = LIBRARY_FILES
TOP = 3
MAX_PHASE_A_GB = 16.0  # M1 measured 15.6 GB without an MLX cache limit; the worker sets one
MAX_SEARCH_P50_MS = 300  # M4 measured ~90 ms for hybrid + rerank on this library


@pytest.fixture(scope="module")
def searcher(library: tuple[Store, float]) -> Searcher:
    store, _ = library
    embedder = Embedder().load()
    searcher = Searcher(store, embedder, VectorIndex(store, embedder.revision), Reranker().load())
    searcher.warm_up()
    return searcher


def where(hits: list[Hit]) -> list[tuple[str, str]]:
    return [(h.source_name, h.element) for h in hits]


def test_every_file_is_ingested_within_the_memory_budget(library: tuple[Store, float]) -> None:
    store, peak_gb = library
    names = {f.name for f in FILES}
    assert sorted(s.name for s in store.sources() if s.name in names) == sorted(names)
    assert all(s.status == "ready" for s in store.sources() if s.name in names)
    assert peak_gb <= MAX_PHASE_A_GB
    counts = dict(
        store.conn.execute("SELECT element, count(*) FROM chunks GROUP BY element").fetchall()
    )
    assert counts["speech"] >= 2 and counts["table"] >= 2 and counts["chart"] == 1
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == sum(
        counts.values()
    )


@pytest.mark.parametrize(
    ("question", "source", "element"),
    [
        ("How much revenue was there in Q2?", "report.pdf", "chart"),  # chart data from M3, page 2
        ("How many units did the North region ship in Q1?", "report.pdf", "table"),
        ("Who will send the draft to Legal, and by when?", "meeting.flac", "speech"),
        ("Who has to approve the revised invoice?", "memo.pdf", "table"),
        ("When is invoice INV-2026-0042 due?", "invoice.png", "text"),
    ],
)
def test_questions_find_the_right_chunk(
    searcher: Searcher, question: str, source: str, element: str
) -> None:
    hits = searcher.search(question, "hybrid+rerank", k=TOP).hits
    assert (source, element) in where(hits), where(hits)


def test_chart_answer_is_cited_with_its_page(searcher: Searcher) -> None:
    hits = searcher.search("How much revenue was there in Q2?", "hybrid+rerank", k=TOP).hits
    chart = next(h for h in hits if h.element == "chart")
    assert chart.citation == "report.pdf · p. 2" and "145" in chart.text
    # Known (M4): the reranker scores this Markdown table just below the report's intro paragraph (0.881 vs 0.893),
    # although RRF ranked it first. M7 measures whether linearizing tables/charts for the reranker helps.


def test_meeting_answer_is_cited_with_its_time(searcher: Searcher) -> None:
    hits = searcher.search(
        "Who will send the draft to Legal, and by when?", "hybrid+rerank", k=TOP
    ).hits
    meeting = next(h for h in hits if h.source_name == "meeting.flac")
    assert meeting.citation.startswith("meeting.flac · 0:") and "wednesday" in meeting.text


def test_partial_id_is_found_through_the_trigram_stage(searcher: Searcher) -> None:
    result = searcher.search("2026-004", "hybrid+rerank", k=5)
    assert not result.trace["bm25"]  # not a whole token anywhere
    id_sources = {
        row["source_name"]
        for row in searcher.store.chunks_by_id([c for c, _ in result.trace["ids"]]).values()
    }
    assert {"report.pdf", "memo.pdf", "invoice.png"} <= id_sources


def test_hybrid_search_latency(searcher: Searcher) -> None:
    questions = [
        "How much revenue was there in Q2?",
        "Who will send the draft to Legal, and by when?",
        "What is the total due on the invoice?",
        "Which region shipped the most units?",
        "What happens at the next review?",
    ]
    times = []
    for q in questions:
        start = time.perf_counter()
        searcher.search(q, "hybrid+rerank")
        times.append((time.perf_counter() - start) * 1000)
    assert statistics.median(times) <= MAX_SEARCH_P50_MS, times
