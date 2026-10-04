"""Chunking with citations (PLAN.md §2.3, M4): transcripts and Docling documents → ~300-token chunks.

- **Documents:** Docling's ``HierarchicalChunker`` yields one piece per element with its heading path and page provenance.
  Tables are serialized as Markdown (the default "North, 1 = 1,240" triplets drop the column names a question would use);
  chart data (``meta.tabular_chart``, M3) comes through as a Markdown table. Consecutive text under the same heading is
  merged up to ``TARGET_CHARS``; tables and charts stay separate chunks so a citation can point at "the table on p. 1".
  Oversized text is split at sentence ends, oversized tables by rows with the header repeated.
- **Transcripts:** consecutive segments are grouped up to ``TARGET_CHARS``, keeping start / end seconds for citations.

Sizes are in characters: ~3.75 characters per token for the Granite tokenizer (measured in M1), so 1,100 ≈ 300 tokens. No
tokenizer is needed, which keeps chunking testable in CI without model files.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from granit.ingest.audio import Transcript
from granit.store.db import NewChunk

TARGET_CHARS = 1_100  # ≈ 300 tokens
MAX_CHARS = 2_000  # hard ceiling for any chunk (≈ 530 tokens)
CONTEXT_SEPARATOR = " > "

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


# ── transcripts ──


def chunk_transcript(transcript: Transcript, target: int = TARGET_CHARS) -> list[NewChunk]:
    chunks: list[NewChunk] = []
    texts: list[str] = []
    start = end = 0.0
    for segment in transcript.segments:
        if texts and len(" ".join(texts)) + 1 + len(segment.text) > target:
            chunks.append(NewChunk(" ".join(texts), "speech", start_s=start, end_s=end))
            texts = []
        if not texts:
            start = segment.start
        texts.append(segment.text)
        end = segment.end
    if texts:
        chunks.append(NewChunk(" ".join(texts), "speech", start_s=start, end_s=end))
    return chunks


# ── documents ──


@dataclass
class _Piece:
    text: str
    element: str
    context: str
    pages: list[int]


def _element(doc_items: list[Any]) -> str:
    labels = {str(item.label) for item in doc_items}
    if labels == {"table"}:
        return "table"
    if labels == {"picture"} or labels == {"chart"}:
        return "chart"  # pictures only produce text when they carry chart data
    return "text"


def _pieces(doc: Any) -> Iterator[_Piece]:
    from docling_core.transforms.chunker import HierarchicalChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
        DocChunk,
    )
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class MarkdownTables(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
            return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())

    for base in HierarchicalChunker(serializer_provider=MarkdownTables()).chunk(doc):
        chunk = DocChunk.model_validate(base)  # typed access to doc_items / headings
        text = chunk.text.strip()
        if not text:
            continue
        items = chunk.meta.doc_items
        pages = sorted({prov.page_no for item in items for prov in item.prov})
        yield _Piece(
            text, _element(items), CONTEXT_SEPARATOR.join(chunk.meta.headings or []), pages
        )


def chunk_document(doc: Any, target: int = TARGET_CHARS, limit: int = MAX_CHARS) -> list[NewChunk]:
    """A ``DoclingDocument`` → chunks with heading context and page range."""
    chunks: list[NewChunk] = []
    pending: _Piece | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            chunks.extend(_split(pending, limit))
            pending = None

    for piece in _pieces(doc):
        mergeable = (
            pending is not None
            and piece.element == "text" == pending.element
            and piece.context == pending.context
            and len(pending.text) + 2 + len(piece.text) <= target
        )
        if mergeable and pending is not None:
            pending = _Piece(
                f"{pending.text}\n\n{piece.text}",
                "text",
                pending.context,
                sorted(set(pending.pages) | set(piece.pages)),
            )
        else:
            flush()
            pending = piece
    flush()
    return chunks


def _split(piece: _Piece, limit: int) -> list[NewChunk]:
    if len(piece.text) <= limit:
        parts = [piece.text]
    elif piece.element in ("table", "chart"):
        parts = _split_table(piece.text, limit)
    else:
        parts = _split_text(piece.text, limit)
    first, last = (piece.pages[0], piece.pages[-1]) if piece.pages else (None, None)
    return [NewChunk(p, piece.element, piece.context, first, last) for p in parts]


def _split_text(text: str, limit: int) -> list[str]:
    """Pack sentences into parts of at most ``limit`` characters (a single huge sentence is cut hard)."""
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _split_table(text: str, limit: int) -> list[str]:
    """Split a Markdown table by rows, repeating the header (and its separator line) in every part."""
    lines = text.splitlines()
    header = lines[:2] if len(lines) > 2 and set(lines[1]) <= set("|-: ") else lines[:1]
    body = lines[len(header) :]
    head = "\n".join(header)
    parts: list[str] = []
    rows: list[str] = []
    for line in body:
        if rows and len(head) + sum(len(r) + 1 for r in rows) + len(line) + 1 > limit:
            parts.append("\n".join([head, *rows]))
            rows = []
        rows.append(line)
    if rows or not parts:
        parts.append("\n".join([head, *rows]))
    return parts


def citation(
    source_name: str, page_start: int | None, page_end: int | None, start_s: float | None
) -> str:
    """How a chunk is cited: ``report.pdf · p. 2``, ``report.pdf · pp. 3–4`` or ``call.m4a · 12:40``."""
    from granit.ingest.audio import timestamp

    if start_s is not None:
        return f"{source_name} · {timestamp(start_s)}"
    if page_start is None:
        return source_name
    if page_end is not None and page_end != page_start:
        return f"{source_name} · pp. {page_start}–{page_end}"
    return f"{source_name} · p. {page_start}"
