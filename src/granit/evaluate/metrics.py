"""Evaluation metrics (PLAN.md §4.9): pure functions, unit-tested in ``tests/unit/test_eval_metrics.py``.

- **Normalization** for matching facts and field values: case, whitespace, dashes and quotes, currency symbols, thousands
  separators, trailing ``.00``, and dates in any common form → ISO 8601.
- **Retrieval:** recall@k and MRR@k against gold refs (a chunk matches a ref if it's from the same file and overlaps the
  ref's page or time range), and citation precision.
- **Extraction:** per-field exact match after normalization.
- **Tables:** cell F1 between grids, after expanding merged cells and aligning rows and columns in order (a 2D longest
  common subsequence, GriTS-style), so an extra header row or a shifted column costs only the cells that are really wrong.
- **Summaries:** action-item recall / precision (owner matches, at least half of the reference task's keywords present). An
  owner the transcript never says (a role the annotators knew from metadata) can't be heard, so no owner meets it.
- **Agreement:** Cohen's κ for the Guardian check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from granit.ingest.vision import html_grid as html_grid
from granit.store.text import canonical as normalize  # tables are scored on the grid ingest uses

# ── normalization ──


def contains(haystack: str, needle: str) -> bool:
    """``needle`` appears in ``haystack`` as whole words, after normalizing both."""
    n, h = normalize(needle), normalize(haystack)
    return bool(n) and re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", h) is not None


def fact_found(fact: str, answer: str) -> bool:
    """An expected fact is in the answer; ``"two years | 2 years"`` accepts any of the alternatives."""
    return any(contains(answer, alt) for alt in fact.split("|") if alt.strip())


def fact_coverage(facts: Sequence[str], answer: str) -> float | None:
    """Share of expected facts found in the answer; None when the question lists none."""
    if not facts:
        return None
    return sum(fact_found(f, answer) for f in facts) / len(facts)


# ── retrieval ──


@dataclass(frozen=True)
class Ref:
    """Where an answer's evidence is: a file (by content hash), optionally a page or a time range."""

    source_sha256: str
    page: int | None = None
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class Located:
    """Where a retrieved chunk is."""

    source_sha256: str
    page_start: int | None = None
    page_end: int | None = None
    start_s: float | None = None
    end_s: float | None = None


def matches(chunk: Located, ref: Ref) -> bool:
    if chunk.source_sha256 != ref.source_sha256:
        return False
    if ref.page is not None:
        if chunk.page_start is None:
            return False
        return chunk.page_start <= ref.page <= (chunk.page_end or chunk.page_start)
    if ref.start_s is not None:
        if chunk.start_s is None:
            return False
        end = ref.end_s if ref.end_s is not None else ref.start_s
        chunk_end = chunk.end_s if chunk.end_s is not None else chunk.start_s
        return chunk.start_s <= end and ref.start_s <= chunk_end
    return True


def recall_at_k(ranked: Sequence[Located], refs: Sequence[Ref], k: int) -> float | None:
    """Share of gold refs matched by at least one of the top ``k`` chunks; None without refs."""
    if not refs:
        return None
    top = ranked[:k]
    return sum(any(matches(c, r) for c in top) for r in refs) / len(refs)


def reciprocal_rank(ranked: Sequence[Located], refs: Sequence[Ref], k: int) -> float | None:
    """1 / rank of the first chunk matching any ref within the top ``k`` (0 if none); None without refs."""
    if not refs:
        return None
    for rank, chunk in enumerate(ranked[:k], start=1):
        if any(matches(chunk, r) for r in refs):
            return 1.0 / rank
    return 0.0


def citation_precision(cited: Sequence[Located], refs: Sequence[Ref]) -> float | None:
    """Share of cited chunks that match a gold ref; None when nothing was cited or there are no refs."""
    if not cited or not refs:
        return None
    return sum(any(matches(c, r) for r in refs) for c in cited) / len(cited)


# ── extraction ──


def value_matches(expected: Any, got: Any) -> bool:
    """Exact match after normalization; an expected ``None`` means the field must be missing (None or empty)."""
    if expected is None:
        return got is None or str(got).strip() == ""
    return got is not None and normalize(str(expected)) == normalize(str(got))


def field_matches(expected: dict[str, Any], got: dict[str, Any]) -> dict[str, bool]:
    return {field: value_matches(value, got.get(field)) for field, value in expected.items()}


# ── tables ──


def _clean(grid: Sequence[Sequence[str]]) -> list[list[str]]:
    """Normalized cells; rows and columns that are entirely empty are dropped."""
    rows = [[normalize(c) for c in row] for row in grid]
    rows = [r for r in rows if any(r)]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [c for c in range(width) if any(r[c] for r in rows)]
    return [[r[c] for c in keep] for r in rows]


def _row_match(a: list[str], b: list[str]) -> int:
    """Non-empty cells that match in order (longest common subsequence over the two rows)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x and x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def cell_f1(truth: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]) -> float:
    """F1 over non-empty cells, with rows aligned in order (LCS weighted by matching cells per row pair)."""
    t, p = _clean(truth), _clean(predicted)
    n_true = sum(bool(c) for r in t for c in r)
    n_pred = sum(bool(c) for r in p for c in r)
    if not n_true or not n_pred:
        return 1.0 if n_true == n_pred else 0.0
    prev = [0] * (len(p) + 1)
    for row_t in t:
        cur = [0]
        for j, row_p in enumerate(p, start=1):
            cur.append(max(prev[j - 1] + _row_match(row_t, row_p), prev[j], cur[j - 1]))
        prev = cur
    matched = prev[-1]
    precision, recall = matched / n_pred, matched / n_true
    return 0.0 if not matched else 2 * precision * recall / (precision + recall)


# ── summaries ──

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "to",
        "of",
        "for",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "about",
        "into",
        "over",
        "before",
        "after",
        "up",
        "out",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "be",
        "will",
        "should",
        "would",
        "can",
        "could",
        "must",
        "need",
        "needs",
        "needed",
        "it",
        "its",
        "their",
        "our",
        "his",
        "her",
        "them",
        "we",
        "they",
        "he",
        "she",
        "i",
        "you",
        "me",
        "us",
        "next",
        "new",
        "all",
        "any",
        "some",
    ]
)


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def _stem(word: str) -> str:
    """A light stemmer: "arranging", "arrangement" and "arrange" → "arrang"; "splitting" → "split"; "offices" → "offic"."""
    for suffix in ("ments", "ment", "ings", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    return word


def _same_name(a: str, b: str) -> bool:
    """Equal, or a one-letter-ish spelling slip ("Priya" heard as "pria"): the name's error already counts in WER."""
    from difflib import SequenceMatcher

    return a == b or (min(len(a), len(b)) >= 3 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def owner_matches(expected: str | None, got: str | None) -> bool:
    """Same person: every name part of the shorter name matches one of the other's ("Elena" = "Elena Ruiz")."""
    if not expected:
        return True
    a, b = normalize(expected).split(), normalize(got or "").split()
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return all(any(_same_name(x, y) for y in long) for x in short)


def mentions(text: str, name: str) -> bool:
    """``name``'s words appear in a row in ``text``, allowing the slips ``owner_matches`` allows ("Priya" said, "pria" heard)."""
    want, words = normalize(name).split(), normalize(text).split()
    return bool(want) and any(
        all(_same_name(w, words[i + j]) for j, w in enumerate(want))
        for i in range(len(words) - len(want) + 1)
    )


def item_matches(
    expected: dict[str, Any],
    got: dict[str, Any],
    min_overlap: float = 0.5,
    transcript: str | None = None,
) -> bool:
    """Same owner, and at least ``min_overlap`` of the reference task's keywords in the summary's task.

    With the ``transcript``, an owner it never says is met by no owner: there's nothing to hear it from.
    """
    owner = expected.get("owner")
    unheard = transcript is not None and owner and not mentions(transcript, owner)
    if not (owner_matches(owner, got.get("owner")) or (unheard and not got.get("owner"))):
        return False
    want = keywords(expected["task"])
    return not want or len(want & keywords(got.get("task") or "")) / len(want) >= min_overlap


def action_items(
    expected: Sequence[dict[str, Any]], got: Sequence[dict[str, Any]], transcript: str | None = None
) -> tuple[float | None, float | None]:
    """(recall, precision) with one-to-one greedy matching; None where there's nothing to divide by."""
    unused = list(range(len(got)))
    matched = 0
    for e in expected:
        hit = next((i for i in unused if item_matches(e, got[i], transcript=transcript)), None)
        if hit is not None:
            unused.remove(hit)
            matched += 1
    recall = matched / len(expected) if expected else None
    precision = matched / len(got) if got else None
    return recall, precision


# ── agreement and aggregation ──


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Agreement beyond chance between two raters on yes / no labels; None for an empty or one-sided sample."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo), 3)
