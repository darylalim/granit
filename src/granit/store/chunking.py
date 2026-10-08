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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from granit.ingest.audio import Transcript
from granit.store.db import NewChunk

TARGET_CHARS = 1_100  # ≈ 300 tokens
MAX_CHARS = 2_000  # hard ceiling for any chunk (≈ 530 tokens)
CONTEXT_SEPARATOR = " > "

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


# ── transcripts ──


def chunk_transcript(
    transcript: Transcript, names: Mapping[int, str] | None = None, target: int = TARGET_CHARS
) -> list[NewChunk]:
    """Segments packed to ``target`` characters. With speaker names (PLAN.md §3.8), turns instead: a named speaker's turn
    is a ``Priya: …`` line, everyone else's a plain line, so search finds the name and Ask sees who spoke. No names:
    today's chunks exactly."""
    if names:
        return _chunk_turns(transcript, names, target)
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


def _turn_lines(
    transcript: Transcript, names: Mapping[int, str], limit: int
) -> list[tuple[str, float, float]]:
    """``(line, start, end)`` per turn; a turn longer than ``limit`` is split at words, each piece keeping its name."""
    from granit.ingest.speakers import labelled_turns

    out = []
    for turn in labelled_turns(transcript.segments, names):
        prefix = f"{names[turn.speaker]}: " if turn.speaker is not None else ""
        piece: list[str] = []
        for word in turn.text.split():
            if piece and len(prefix) + len(" ".join([*piece, word])) > limit:
                out.append((prefix + " ".join(piece), turn.start, turn.end))
                piece = []
            piece.append(word)
        if piece:
            out.append((prefix + " ".join(piece), turn.start, turn.end))
    return out


def _chunk_turns(transcript: Transcript, names: Mapping[int, str], target: int) -> list[NewChunk]:
    chunks: list[NewChunk] = []
    lines: list[str] = []
    start = end = 0.0
    for line, line_start, line_end in _turn_lines(transcript, names, MAX_CHARS):
        if lines and len("\n".join(lines)) + 1 + len(line) > target:
            chunks.append(NewChunk("\n".join(lines), "speech", start_s=start, end_s=end))
            lines = []
        if not lines:
            start = line_start
        lines.append(line)
        end = line_end
    if lines:
        chunks.append(NewChunk("\n".join(lines), "speech", start_s=start, end_s=end))
    return chunks


# ── documents ──


@dataclass
class _Piece:
    text: str
    element: str
    context: str
    pages: list[int]


def _element(doc_items: list[Any], text: str) -> str:
    # a caption comes with its table or picture (v30: captioned tables were chunked as text, split without a header)
    labels = {str(item.label) for item in doc_items} - {"caption"}
    has_table = any(line.startswith("|") for line in text.splitlines())
    if labels == {"table"}:
        return "table"
    if (labels == {"picture"} or labels == {"chart"}) and has_table:
        return "chart"  # a picture's text is its chart data (and caption); a bare caption is text
    return "text"


def document_title(doc: Any) -> str:
    """The first heading in reading order: the vendor on an invoice, the title of a report.

    Granite-Docling marks every heading as level 1, so a chunk's own headings stop at the nearest one ("Invoice",
    "Delivery Note") and lose which document it is. M7 found the model then declined questions that named the vendor.
    """
    from docling_core.types.doc import DocItemLabel

    for item, _ in doc.iterate_items():
        if (
            getattr(item, "label", None) in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER)
            and item.text.strip()
        ):
            return item.text.strip()
    return ""


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

    title = document_title(doc)
    for base in HierarchicalChunker(serializer_provider=MarkdownTables()).chunk(doc):
        chunk = DocChunk.model_validate(base)  # typed access to doc_items / headings
        text = chunk.text.strip()
        if not text:
            continue
        items = chunk.meta.doc_items
        pages = sorted({prov.page_no for item in items for prov in item.prov})
        headings = list(chunk.meta.headings or [])
        if title and headings[:1] != [title]:
            headings.insert(0, title)  # every chunk says which document it's from
        yield _Piece(text, _element(items, text), CONTEXT_SEPARATOR.join(headings), pages)


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


def section_row(line: str) -> bool:
    """A table row that labels the rows under it: one spanning cell, repeated in every column by the Markdown export
    (``| Not Adjusted | Not Adjusted | … |``), and not a number."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return (
        len(cells) > 1
        and cells[0] != ""
        and len(set(cells)) == 1
        and not re.fullmatch(r"[-+()$%.,\d\s]+", cells[0])
    )


def _split_table(text: str, limit: int) -> list[str]:
    """Split a Markdown table by rows, repeating the caption above it, the header and its separator line in every
    part, and the latest section row ("Not Adjusted") in the parts that continue its section."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), 0)
    end = (
        start + 2 if len(lines) > start + 2 and set(lines[start + 1]) <= set("|-: ") else start + 1
    )
    header, body = lines[:end], lines[end:]
    head = "\n".join(header)
    parts: list[str] = []
    rows: list[str] = []
    section: str | None = None
    for line in body:
        if rows and len(head) + sum(len(r) + 1 for r in rows) + len(line) + 1 > limit:
            parts.append("\n".join([head, *rows]))
            rows = [section] if section and not section_row(line) else []
        if section_row(line):
            section = line
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
