"""Search without models (PLAN.md §2.3, §3.2): text normalization, BM25, trigram IDs, vectors, RRF, rerank, trace."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from granit.search.hybrid import MODES, Searcher, rrf
from granit.search.vectors import VectorIndex
from granit.store.db import NewChunk, Store
from granit.store.text import fts_or_query, id_like_terms, query_terms, search_text
from tests.conftest import ROOT

DOC = ROOT / "tests" / "fixtures" / "documents" / "invoice.png"


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words point the same way. Deterministic, no model."""

    revision = "fake-rev"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 768] += 1.0
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out.astype(np.float16)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class FakeReranker:
    """Scores a passage by how many query words it contains (reverses BM25's order on purpose in one test)."""

    def __init__(self, prefer: str | None = None) -> None:
        self.prefer = prefer
        self.calls: list[int] = []

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(len(passages))
        words = set(re.findall(r"\w+", query.lower()))
        return [
            (10.0 if self.prefer and self.prefer in p else 0.0)
            + len(words & set(re.findall(r"\w+", p.lower())))
            for p in passages
        ]


TEXTS = [
    ("Invoice INV-2026-0042 was approved by Finance.", "Invoices"),
    ("Revenue peaked in Q4 after the new warehouse opened in October.", "Revenue"),
    ("| Quarter | Revenue |\n|---|---|\n| Q2 | 145 |", "Revenue"),
    ("Marcus will send the contract to Legal/Procurement by Wednesday.", "Open items"),
    ("Release v1.2.3 ships with part AB_12 and order 2026-0042-B.", "Release notes"),
]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "data")
    store.add_file(DOC)
    job = store.claim_next()
    assert job is not None
    chunks = [NewChunk(text, "text", context, 1, 1) for text, context in TEXTS]
    vectors = FakeEmbedder().encode([c.search_body for c in chunks])
    store.complete_ingest(job, chunks, vectors, FakeEmbedder.revision)
    return store


def ids_for(store: Store, ranked: list[tuple[int, float]]) -> list[str]:
    rows = store.chunks_by_id([c for c, _ in ranked])
    return [rows[c]["text"][:12] for c, _ in ranked]


# ── text normalization ──


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("approved by Finance.", "approved by Finance"),  # M4 finding: no "finance." token
        ("Legal/Procurement", "Legal Procurement"),
        ("e-mail", "e mail"),
        ("INV-2026-0042", "INV-2026-0042"),  # tokens with digits keep their separators
        ("v1.2.3 and 2026-09-14.", "v1.2.3 and 2026-09-14"),
        ("AB_12", "AB_12"),
    ],
)
def test_search_text(text: str, expected: str) -> None:
    assert search_text(text) == expected


def test_query_terms_drop_stop_words_but_never_everything() -> None:
    assert query_terms("What did Finance approve for INV-2026-0042?") == [
        "finance",
        "approve",
        "inv-2026-0042",
    ]
    assert query_terms("what is it") == ["what", "is", "it"]  # only stop words: keep them


def test_id_like_terms() -> None:
    assert id_like_terms("INV-2026-0042, 2026-004, v1.2 and AB_12 in 2026 by e-mail") == [
        "INV-2026-0042",
        "2026-004",
        "v1.2",
        "AB_12",
    ]


def test_fts_query_quotes_every_term() -> None:
    assert fts_or_query(['say "hi"', "inv-2026"]) == '"say ""hi""" OR "inv-2026"'


# ── lexical stages ──


def test_bm25_matches_plain_words_at_sentence_ends(store: Store) -> None:
    searcher = Searcher(store)
    assert ids_for(store, searcher.bm25("finance"))[0].startswith("Invoice")
    assert ids_for(store, searcher.bm25("procurement"))[0].startswith("Marcus")
    assert searcher.bm25("zebra") == []


def test_bm25_matches_exact_ids_as_single_tokens(store: Store) -> None:
    searcher = Searcher(store)
    assert ids_for(store, searcher.bm25("INV-2026-0042")) == ["Invoice INV-"]
    assert ids_for(store, searcher.bm25("v1.2.3")) == ["Release v1.2"]


def test_trigram_finds_partial_ids(store: Store) -> None:
    searcher = Searcher(store)
    assert searcher.bm25("2026-004") == []  # not a whole token anywhere
    found = ids_for(store, searcher.ids("2026-004"))
    assert set(found) == {"Invoice INV-", "Release v1.2"}  # INV-2026-0042 and 2026-0042-B
    assert ids_for(store, searcher.ids("ab_12")) == ["Release v1.2"]  # case-insensitive
    assert searcher.ids("plain words only") == []


def test_headings_are_searchable_context(store: Store) -> None:
    assert ids_for(store, Searcher(store).bm25("release notes"))[0].startswith("Release")


# ── vectors ──


def test_vector_index_loads_current_revision_and_reloads_on_change(store: Store) -> None:
    index = VectorIndex(store, FakeEmbedder.revision, device="cpu")
    assert index.refresh() is True and len(index) == len(TEXTS)
    assert index.refresh() is False  # nothing changed
    hits = index.search(FakeEmbedder().encode_query("revenue peaked warehouse"), 2)
    assert ids_for(store, hits)[0].startswith("Revenue peak")
    assert hits[0][1] >= hits[1][1]
    assert (
        len(VectorIndex(store, "other-revision", device="cpu").search(np.zeros(768, np.float16), 5))
        == 0
    )
    store.delete_source(store.sources()[0])
    assert (
        index.search(FakeEmbedder().encode_query("revenue"), 5) == []
    )  # corpus_version changed → reloaded


# ── fusion and the pipeline ──


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [c for c, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([]) == []


def searcher(store: Store, reranker: FakeReranker | None = None) -> Searcher:
    return Searcher(
        store, FakeEmbedder(), VectorIndex(store, FakeEmbedder.revision, device="cpu"), reranker
    )


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_cited_hits_and_a_trace(store: Store, mode: str) -> None:
    result = searcher(store, FakeReranker()).search("How much revenue in Q2?", mode, k=3)
    assert 1 <= len(result.hits) <= 3
    assert all(h.citation == "invoice.png · p. 1" for h in result.hits)
    stages = set(result.trace)
    expected = {
        "bm25": {"bm25", "ids"},
        "vectors": {"vectors"},
        "hybrid": {"bm25", "ids", "vectors", "rrf"},
        "hybrid+rerank": {"bm25", "ids", "vectors", "rrf", "rerank"},
    }[mode]
    assert stages == expected
    assert set(result.seconds) == expected
    json_trace = result.trace_json()
    assert json_trace["mode"] == mode and all(
        isinstance(p, list) for p in json_trace["stages"].values()
    )


def test_rerank_reorders_the_fused_candidates(store: Store) -> None:
    reranker = FakeReranker(prefer="Marcus")
    result = searcher(store, reranker).search("revenue", "hybrid+rerank", k=2)
    assert result.hits[0].text.startswith("Marcus")  # the reranker has the last word
    assert reranker.calls == [len(result.trace["rrf"])]  # it saw the RRF candidates (≤ 30)


def test_partial_id_reaches_hybrid_results_through_the_trigram_stage(store: Store) -> None:
    result = searcher(store).search("order 2026-004", "hybrid", k=5)
    assert result.trace["ids"] and result.trace["bm25"]
    assert "Invoice INV-" in [h.text[:12] for h in result.hits]


def test_unknown_mode_is_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        Searcher(store).search("x", "magic")


def test_bm25_mode_needs_no_models(store: Store) -> None:
    result = Searcher(store).search("finance", "bm25")
    assert result.hits[0].text.startswith("Invoice")
    assert result.to_json()["hits"][0]["citation"] == "invoice.png · p. 1"
