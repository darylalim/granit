"""Text normalization shared by the full-text index and queries (PLAN.md §2.3, M4).

The plan's FTS5 tokenizer keeps ``- _ . /`` inside tokens so IDs like ``INV-2026-0042`` and ``v1.2.3`` stay whole. On its own,
that also glued punctuation to ordinary words: "approved by Finance." was indexed as ``finance.`` and "Legal/Procurement" as
one token, so searching for "finance" or "legal" found nothing (M4 finding). Both the indexed text and every query therefore
go through ``search_text``: a token **with a digit** keeps its separators (an ID, version, date or amount); any other token is
split at them.
"""

from __future__ import annotations

import re
import unicodedata

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


# ── canonical form for comparing texts (ingest's text-layer check, eval metrics) ──

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_MDY_NAME = re.compile(rf"\b({MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
DATE_DMY_NAME = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH})\.?,?\s+(\d{{4}})\b")
DATE_MDY_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


def _iso(year: str, month: int, day: str) -> str:
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def canonical(text: str) -> str:
    """Lowercase text with numbers, dates and punctuation in one canonical form, for comparing two texts that say the same
    thing in different formats (``$4,980.00`` = ``4980``, ``October 14, 2026`` = ``2026-10-14``): used by ingest's text-layer
    check and by the eval's fact and field matching."""
    t = unicodedata.normalize("NFKC", str(text)).translate(DASHES).translate(QUOTES).lower()
    t = DATE_MDY_NAME.sub(lambda m: _iso(m[3], MONTHS[m[1]], m[2]), t)
    t = DATE_DMY_NAME.sub(lambda m: _iso(m[3], MONTHS[m[2]], m[1]), t)
    t = DATE_MDY_SLASH.sub(lambda m: _iso(m[3], int(m[1]), m[2]), t)
    t = DATE_ISO.sub(lambda m: _iso(m[1], int(m[2]), m[3]), t)
    t = re.sub(r"[$€£¥]", " ", t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)  # 4,980 → 4980
    t = re.sub(r"(?<=\d)\.0+(?!\d)", "", t)  # 4980.00 → 4980
    t = re.sub(
        r"(?<!\d)[.:](?!\d)|(?<=\d)[.:](?!\d)|(?<!\d)[.:](?=\d)", " ", t
    )  # keep 4.5 and 10:30, drop the rest
    t = re.sub(r"[^\w\s.:%/@#+-]", " ", t)
    t = re.sub(
        r"(?<![\w])-|-(?![\w])", " ", t
    )  # a dash between words or digits stays (INV-2026-0042)
    return " ".join(t.split())
