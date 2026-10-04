"""Text normalization shared by the full-text index and queries (PLAN.md §2.3, M4).

The plan's FTS5 tokenizer keeps ``- _ . /`` inside tokens so IDs like ``INV-2026-0042`` and ``v1.2.3`` stay whole. On its own,
that also glued punctuation to ordinary words: "approved by Finance." was indexed as ``finance.`` and "Legal/Procurement" as
one token, so searching for "finance" or "legal" found nothing (M4 finding). Both the indexed text and every query therefore
go through ``search_text``: a token **with a digit** keeps its separators (an ID, version, date or amount); any other token is
split at them.
"""

from __future__ import annotations

import re

_RUN = re.compile(r"[\w\-./]+")
_SEPARATORS = re.compile(r"[\-./_]+")
_EDGES = "-./_"
_ID_LIKE = re.compile(r"^(?=[\w\-./]*\d)(?=[\w\-./]*[A-Za-z\-./_])[\w][\w\-./]*[\w]$")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def _normalize_run(run: str) -> str:
    token = run.strip(_EDGES)
    if not token:
        return " "
    if any(ch.isdigit() for ch in token):
        return token  # ID-like: INV-2026-0042, v1.2.3, 2026-09-14, 4.6
    return _SEPARATORS.sub(" ", token)


def search_text(text: str) -> str:
    """The string the full-text index sees (and queries are matched against)."""
    return _RUN.sub(lambda m: _normalize_run(m.group(0)), text)


def query_terms(query: str) -> list[str]:
    """Lowercased, de-duplicated terms of a question for BM25 (stop words dropped unless nothing else is left)."""
    terms = [t.lower() for t in _RUN.findall(search_text(query)) if t.strip(_EDGES)]
    unique = list(dict.fromkeys(terms))
    content = [t for t in unique if t not in STOPWORDS]
    return content or unique


def fts_or_query(terms: list[str]) -> str:
    """An FTS5 MATCH expression: every term quoted (so punctuation is literal), joined with OR (BM25 ranks)."""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def id_like_terms(query: str) -> list[str]:
    """Query tokens that look like IDs or codes (letters/digits with separators, ≥ 3 chars) for trigram search."""
    found = []
    for run in _RUN.findall(query):
        token = run.strip(_EDGES)
        if len(token) >= 3 and _ID_LIKE.match(token) and not token.isdigit():
            found.append(token)
    return list(dict.fromkeys(found))
