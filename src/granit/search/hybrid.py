"""Hybrid retrieval (PLAN.md §3.2, M4): BM25 + trigram ID matches + vectors → Reciprocal Rank Fusion → reranker.

- **BM25** (``chunks_fts``): question terms (normalized like the index, stop words dropped) OR-ed together; FTS5 ranks by BM25.
- **IDs** (``chunks_trigram``): ID-like tokens (``INV-2026-0042``, ``2026-004``) as exact substrings, so partial IDs match.
- **Vectors** (``VectorIndex``): cosine similarity to the question's embedding.
- **RRF:** score = Σ 1 / (60 + rank) over the lists a chunk appears in; robust to the stages' different score scales.
- **Rerank:** the cross-encoder re-scores the RRF top 30; the top 8 go to the LLM.

Each result carries a **trace** (chunk ids + scores per stage), stored with every answer as ``retrieval_trace`` (PLAN.md §4.9)
so a bad answer shows which stage lost the right chunk. ``mode`` switches stages off for M7's comparison.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from granit.config import RERANK_CANDIDATES, RRF_K, SEARCH_CANDIDATES, SEARCH_TOP_K
from granit.store.chunking import citation
from granit.store.db import Store
from granit.store.text import fts_or_query, id_like_terms, query_terms

MODES = ("bm25", "vectors", "hybrid", "hybrid+rerank")
Ranked = list[tuple[int, float]]  # (chunk_id, score), best first


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> Ranked:
    """Reciprocal Rank Fusion of several ranked id lists (rank 1 = best). Ties keep first-seen order."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float
    source_id: int
    source_name: str
    source_kind: str
    text: str
    context: str
    element: str
    page_start: int | None
    page_end: int | None
    start_s: float | None
    end_s: float | None

    @property
    def citation(self) -> str:
        return citation(self.source_name, self.page_start, self.page_end, self.start_s)

    @property
    def passage(self) -> str:
        """What the reranker and the LLM read: heading context, then the text."""
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    trace: dict[str, Ranked] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def trace_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stages": {
                stage: [[cid, round(score, 5)] for cid, score in ranked]
                for stage, ranked in self.trace.items()
            },
            "seconds": self.seconds,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [{**asdict(h), "citation": h.citation} for h in self.hits],
            "trace": self.trace_json(),
        }


class Searcher:
    """Phase B retrieval, in the UI backend process (PLAN.md §2.2). Embedder / index / reranker are optional for BM25-only use."""

    def __init__(
        self,
        store: Store,
        embedder: Any = None,
        index: Any = None,
        reranker: Any = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.index = index
        self.reranker = reranker

    def warm_up(self) -> None:
        """Run one throwaway query so the first real question isn't slowed by model / GPU warm-up (~100 ms, M4)."""
        if self.index is not None:
            self.index.refresh()
        self.search("warm up", "hybrid+rerank" if self.reranker else "hybrid", k=1)

    # stages

    def bm25(self, query: str, k: int = SEARCH_CANDIDATES) -> Ranked:
        terms = query_terms(query)
        if not terms:
            return []
        rows = self.store.conn.execute(
            "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ?"
            " ORDER BY score LIMIT ?",
            (fts_or_query(terms), k),
        )
        return [(int(r[0]), -float(r[1])) for r in rows]  # bm25(): lower is better

    def ids(self, query: str, k: int = SEARCH_CANDIDATES) -> Ranked:
        best: dict[int, float] = {}
        for term in id_like_terms(query):
            rows = self.store.conn.execute(
                "SELECT rowid, bm25(chunks_trigram) AS score FROM chunks_trigram WHERE chunks_trigram MATCH ?"
                " ORDER BY score LIMIT ?",
                ('"' + term.replace('"', '""') + '"', k),
            )
            for chunk_id, score in rows:
                best[int(chunk_id)] = max(best.get(int(chunk_id), float("-inf")), -float(score))
        return sorted(best.items(), key=lambda item: -item[1])[:k]

    def vectors(self, query: str, k: int = SEARCH_CANDIDATES) -> Ranked:
        if self.embedder is None or self.index is None:
            return []
        self.index.refresh()  # one SELECT; reloads only after an ingest or deletion (long-lived UI backend)
        return self.index.search(self.embedder.encode_query(query), k)

    # the pipeline

    def search(
        self, query: str, mode: str = "hybrid+rerank", k: int = SEARCH_TOP_K
    ) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        trace: dict[str, Ranked] = {}
        seconds: dict[str, float] = {}

        def stage(name: str, fn: Any, *args: Any) -> Ranked:
            start = time.perf_counter()
            ranked = fn(*args)
            seconds[name] = round(time.perf_counter() - start, 4)
            trace[name] = ranked
            return ranked

        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            stage("bm25", self.bm25, query)
            stage("ids", self.ids, query)
        if mode in ("vectors", "hybrid", "hybrid+rerank"):
            stage("vectors", self.vectors, query)

        if mode == "vectors":
            ranked = trace["vectors"]
        elif mode == "bm25":
            ranked = (
                rrf([[c for c, _ in trace["ids"]], [c for c, _ in trace["bm25"]]])
                if trace["ids"]
                else trace["bm25"]
            )
        else:
            ranked = stage("rrf", rrf, [[c for c, _ in r] for r in trace.values() if r])

        if mode == "hybrid+rerank" and self.reranker is not None and ranked:
            ranked = stage("rerank", self._rerank, query, ranked[:RERANK_CANDIDATES])

        top = ranked[:k]
        rows = self.store.chunks_by_id([c for c, _ in top])
        hits = [hit_from_row(rows[c], score) for c, score in top if c in rows]
        return SearchResult(query, mode, hits, trace, seconds)

    def _rerank(self, query: str, candidates: Ranked) -> Ranked:
        rows = self.store.chunks_by_id([c for c, _ in candidates])
        ids = [c for c, _ in candidates if c in rows]
        passages = [hit_from_row(rows[c], 0.0).passage for c in ids]
        scores = self.reranker.scores(query, passages)
        return sorted(zip(ids, scores, strict=True), key=lambda item: -item[1])


def hit_from_row(row: Any, score: float) -> Hit:
    return Hit(
        chunk_id=row["id"],
        score=score,
        source_id=row["source_id"],
        source_name=row["source_name"],
        source_kind=row["source_kind"],
        text=row["text"],
        context=row["context"],
        element=row["element"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        start_s=row["start_s"],
        end_s=row["end_s"],
    )
