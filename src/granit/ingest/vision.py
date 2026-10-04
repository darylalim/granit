"""Granite Vision 4.1 tasks (PLAN.md §3.1): chart → CSV, table → HTML, form → JSON Schema-validated fields.

Task tags (``<chart2csv>``, ``<tables_html>``) are expanded by the model's own chat template, so only the tag is sent.
Key-value extraction uses the model card's VAREX prompt with the user's JSON Schema. Every call uses temperature 0.
Output parsing and validation are pure functions (unit-tested); the model is imported lazily (PLAN.md §4.2).
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from granit.config import HUB_MODELS

CHART_QUESTION = (
    "Is this image a chart or graph that shows data values (bar, line, pie, scatter or area)? "
    "Answer only yes or no."
)
MAX_TOKENS = {"is_chart": 4, "chart": 1024, "table": 4096, "form": 2048}


class VisionOutputError(ValueError):
    """The model's output can't be parsed into the requested format."""


@dataclass(frozen=True)
class Extraction:
    """One Vision result, as stored in the ``extractions`` table (PLAN.md §2.3)."""

    kind: str  # chart | table | form
    format: str  # csv | html | json
    content: str  # the parsed, normalized output ("" if it couldn't be parsed)
    valid: bool
    errors: tuple[str, ...] = ()
    raw: str = ""  # the model's text, kept for debugging when parsing fails
    data: Any = None  # parsed form: CSV rows, HTML table strings, or the JSON object
    missing: tuple[str, ...] = ()  # form fields the model returned as null
    page: int | None = None
    crop: str | None = None  # path of the image region sent to the model
    model: str = ""
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "data"}
        out["errors"], out["missing"] = list(self.errors), list(self.missing)
        return out


# ── output parsing (pure) ──

_FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)


def strip_fences(text: str) -> str:
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def parse_csv(text: str) -> list[list[str]]:
    """``<chart2csv>`` output (usually fenced as ```csv) → rows. Needs a header and at least one data row."""
    body = strip_fences(text)
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(body)) if any(row)]
    if len(rows) < 2:
        raise VisionOutputError("chart CSV needs a header and at least one data row")
    width = len(rows[0])
    if width < 2 or any(len(r) != width for r in rows):
        raise VisionOutputError(
            f"chart CSV rows have inconsistent widths: {[len(r) for r in rows]}"
        )
    return rows


def parse_tables_html(text: str) -> list[str]:
    """``<tables_html>`` output → one HTML string per table. The model wraps several tables in a list: ``[<html>…]``."""
    tables = re.findall(r"<table\b.*?</table>", strip_fences(text), flags=re.DOTALL | re.IGNORECASE)
    if not tables or not all(re.search(r"<t[dh]\b", t, re.IGNORECASE) for t in tables):
        raise VisionOutputError("no HTML table with cells in the output")
    return tables


class _Grid(HTMLParser):
    """An HTML table → a grid of cell texts, with rowspan / colspan expanded (the text repeated in every covered cell)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.spans: dict[tuple[int, int], str] = {}  # (row, col) filled by a rowspan from above
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._span = (_int(a.get("rowspan")), _int(a.get("colspan")))
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            text = " ".join("".join(self._cell).split())
            r = len(self.rows) - 1
            row = self.rows[r]
            rowspan, colspan = self._span
            for _ in range(colspan):
                while (r, len(row)) in self.spans:
                    row.append(self.spans.pop((r, len(row))))
                c = len(row)
                row.append(text)
                for dr in range(1, rowspan):
                    self.spans[(r + dr, c)] = text
            self._cell = None
        elif tag == "tr" and self.rows:
            r, row = len(self.rows) - 1, self.rows[-1]
            while (r, len(row)) in self.spans:
                row.append(self.spans.pop((r, len(row))))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _int(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def html_grid(html: str) -> list[list[str]]:
    parser = _Grid()
    parser.feed(html)
    rows = parser.rows
    # rows that only exist because of rowspans from above
    while parser.spans:
        r = min(k[0] for k in parser.spans)
        while len(rows) <= r:
            rows.append([])
        row = rows[r]
        while (r, len(row)) in parser.spans:
            row.append(parser.spans.pop((r, len(row))))
        if any(k[0] == r for k in parser.spans):  # a gap: fill and continue
            row.append("")
    return [row for row in rows if row]


def parse_json_object(text: str) -> dict[str, Any]:
    body = strip_fences(text)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise VisionOutputError("no JSON object in the output")
    try:
        value = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VisionOutputError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise VisionOutputError("expected a JSON object")
    return value


# ── form extraction: prompt + JSON Schema validation (pure) ──


def kvp_prompt(schema: dict[str, Any]) -> str:
    """The model card's VAREX prompt for schema-based key-value extraction."""
    return (
        "Extract structured data from this document.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Return null for fields you cannot find.\n"
        "Return ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself."
    )


def check_schema(schema: dict[str, Any]) -> None:
    """Raise ``jsonschema.SchemaError`` for a malformed schema (before spending model time on it)."""
    from jsonschema.validators import validator_for

    validator_for(schema).check_schema(schema)
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError(
            'the schema must describe an object: {"type": "object", "properties": {...}}'
        )


_MONTHS = {
    name: i
    for i, names in enumerate(
        [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
         ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
         ("oct", "october"), ("nov", "november"), ("dec", "december")],
        start=1,
    )
    for name in names
}  # fmt: skip
_NUMERIC_DATE = re.compile(r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})$")
_TEXT_DATE = re.compile(
    r"^(?:(\d{1,2})[\s-]+([a-z]+)\.?[\s-]+(\d{4})|([a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4}))$"
)


def to_iso_date(text: str) -> str | None:
    """A date in a common written form → ``YYYY-MM-DD``, or None if it's unparseable or ambiguous.

    Granite Vision rewrites dates in its own formats (M3: ``2026-09-14`` came back as ``14/09/2026``). Day/month
    order is decided only when one part is above 12; ``03/04/2026`` stays ambiguous rather than guessed.
    """
    import datetime as dt

    value = text.strip().lower()
    year = month = day = None
    if match := _NUMERIC_DATE.match(value):
        a, b, c = (int(g) for g in match.groups())
        if len(match.group(1)) == 4:
            year, month, day = a, b, c
        elif len(match.group(3)) == 4:
            year = c
            if a > 12 >= b:
                day, month = a, b
            elif b > 12 >= a:
                month, day = a, b
            elif a == b:
                day = month = a
            else:
                return None  # 03/04/2026: day-month or month-day?
    elif match := _TEXT_DATE.match(value):
        d1, m1, y1, m2, d2, y2 = match.groups()
        day, name, year = (int(d1), m1, int(y1)) if d1 else (int(d2), m2, int(y2))
        month = _MONTHS.get(name)
    if year is None or month is None or day is None:
        return None
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_dates(value: Any, schema: dict[str, Any]) -> Any:
    """Rewrite string fields whose schema says ``"format": "date"`` to ISO 8601 when that's unambiguous."""
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        return {
            key: normalize_dates(item, schema["properties"].get(key, {}))
            for key, item in value.items()
        }
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [normalize_dates(item, schema["items"]) for item in value]
    if isinstance(value, str) and schema.get("format") == "date":
        return to_iso_date(value) or value
    return value


def drop_nulls(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """Remove null fields (the model's "not found"), returning the cleaned value and the dotted paths removed."""
    if isinstance(value, dict):
        cleaned, missing = {}, []
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if item is None:
                missing.append(child)
            else:
                cleaned[key], sub = drop_nulls(item, child)
                missing.extend(sub)
        return cleaned, missing
    if isinstance(value, list):
        items, missing = [], []
        for i, item in enumerate(value):
            if item is not None:
                cleaned_item, sub = drop_nulls(item, f"{path}[{i}]")
                items.append(cleaned_item)
                missing.extend(sub)
        return items, missing
    return value, []


def validate(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate extracted fields against the schema.

    Dates in ``"format": "date"`` fields are normalized to ISO 8601 first, and formats are checked. Nulls are treated
    as "absent" (the prompt asks for null when a field isn't found), so a missing optional
    field is fine and a missing ``required`` one is an error. Returns (cleaned data, errors, missing paths).
    """
    from jsonschema.validators import validator_for

    cleaned, missing = drop_nulls(data)
    cleaned = normalize_dates(cleaned, schema)
    cls = validator_for(schema)
    validator = cls(schema, format_checker=cls.FORMAT_CHECKER)  # enforces "format": "date" etc.
    errors = [
        f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
        for e in sorted(
            validator.iter_errors(cleaned), key=lambda e: list(map(str, e.absolute_path))
        )
    ]
    return cleaned, errors, missing


def merge_pages(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-page form results: the first page with a non-null value wins, field by field."""
    merged: dict[str, Any] = {}
    for result in results:
        for key, value in result.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
            merged.setdefault(key, value)
    return merged


# ── the model ──


@dataclass
class VisionModel:
    """Granite Vision 4.1 4B (bf16) on mlx-vlm, loaded once by the Phase A worker."""

    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

    @property
    def model_id(self) -> str:
        spec = HUB_MODELS["vision"]
        return f"{spec.repo_id}@{spec.revision}"

    def load(self) -> VisionModel:
        if self._model is None:
            from mlx_vlm import load

            from granit.models.download import local_snapshot

            self._model, self._processor = load(str(local_snapshot(HUB_MODELS["vision"])))
        return self

    def generate(self, image: Any, prompt: str, max_tokens: int) -> str:
        """One image + prompt → text, at temperature 0. ``image`` is a PIL image or a file path."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self.load()
        chat = apply_chat_template(self._processor, self._model.config, prompt, num_images=1)
        assert isinstance(chat, str)
        result = generate(
            self._model,
            self._processor,
            chat,
            image=[image if isinstance(image, str) else _as_rgb(image)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return result.text.strip()

    def is_chart(self, image: Any) -> bool:
        answer = self.generate(image, CHART_QUESTION, MAX_TOKENS["is_chart"])
        return answer.lower().lstrip(" *\"'").startswith("yes")

    def chart_to_csv(self, image: Any, **where: Any) -> Extraction:
        return self._task(image, "<chart2csv>", "chart", "csv", where)

    def table_to_html(self, image: Any, **where: Any) -> Extraction:
        return self._task(image, "<tables_html>", "table", "html", where)

    def extract_fields(self, images: list[Any], schema: dict[str, Any], **where: Any) -> Extraction:
        """Schema-based key-value extraction over one or more page images (merged page by page)."""
        check_schema(schema)
        start = time.perf_counter()
        raws, pages, errors = [], [], []
        for image in images:
            raw = self.generate(image, kvp_prompt(schema), MAX_TOKENS["form"])
            raws.append(raw)
            try:
                pages.append(parse_json_object(raw))
            except VisionOutputError as exc:
                errors.append(str(exc))
        seconds = round(time.perf_counter() - start, 2)
        raw = "\n\n".join(raws)
        if not pages:
            return Extraction(
                "form",
                "json",
                "",
                False,
                tuple(errors),
                raw,
                None,
                (),
                model=self.model_id,
                seconds=seconds,
                **where,
            )
        cleaned, problems, missing = validate(merge_pages(pages), schema)
        return Extraction(
            "form",
            "json",
            json.dumps(cleaned, ensure_ascii=False),
            not problems,
            tuple(problems),
            raw,
            cleaned,
            tuple(missing),
            model=self.model_id,
            seconds=seconds,
            **where,
        )

    def _task(self, image: Any, tag: str, kind: str, fmt: str, where: dict[str, Any]) -> Extraction:
        start = time.perf_counter()
        raw = self.generate(image, tag, MAX_TOKENS[kind])
        seconds = round(time.perf_counter() - start, 2)
        try:
            if fmt == "csv":
                data: Any = parse_csv(raw)
                buffer = io.StringIO()
                csv.writer(buffer, lineterminator="\n").writerows(data)
                content = buffer.getvalue()
            else:
                data = parse_tables_html(raw)
                content = "\n".join(data)
        except VisionOutputError as exc:
            return Extraction(
                kind,
                fmt,
                "",
                False,
                (str(exc),),
                raw,
                model=self.model_id,
                seconds=seconds,
                **where,
            )
        return Extraction(
            kind, fmt, content, True, (), raw, data, model=self.model_id, seconds=seconds, **where
        )


def _as_rgb(image: Any) -> Any:
    return image if getattr(image, "mode", "RGB") == "RGB" else image.convert("RGB")
