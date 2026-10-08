"""What the pages show, as plain data (no Streamlit): conversation turns, library and job tables, transcripts, forms.

Kept apart from the page scripts so it's unit-tested directly; pages only lay these out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from granit.ingest.audio import parse_terms, timestamp
from granit.search.hybrid import Hit, hit_from_row
from granit.store.db import AUDIO_EXTENSIONS, DOCUMENT_EXTENSIONS, Source, Store, kind_of
from granit.ui.layout import safe_md
from granit.verify.criteria import TURN_CRITERIA, summary_criteria

# ── verdicts (Phase C, PLAN.md §2.4) ──


CheckColor = Literal["green", "orange", "gray"]


@dataclass
class Check:
    """A Guardian verdict as a badge: status colors only (PLAN.md §4.7): green pass, orange fail, gray error."""

    label: str
    color: CheckColor
    icon: str
    help: str


CHECK_ICONS: dict[CheckColor, str] = {
    "green": ":material/check_circle:",
    "orange": ":material/warning:",
    "gray": ":material/help:",
}
# Shown on answers: groundedness only. It passed the M7 agreement check (0.92 vs hand labels, PLAN.md §4.9); answer
# relevance is recorded but not shown: unvalidated, it fails short correct answers ("24% [3]") as "omitting key details".
SHOWN_TURN_CHECK = ("groundedness", "Grounded", "Unsupported claims")


def check(label: str, passed: bool | None, help_text: str) -> Check:
    color: CheckColor = "gray" if passed is None else "green" if passed else "orange"
    return Check(label, color, CHECK_ICONS[color], help_text)


def turn_checks(rows: list[Any]) -> list[Check]:
    """An answer's badge: ✅ grounded / ⚠️ unsupported claims / ❓ check failed; none until it's been checked."""
    criterion, ok, bad = SHOWN_TURN_CHECK
    row = next((r for r in rows if r["criterion_id"] == criterion), None)
    if row is None:
        return []
    if row["passed"] is None:
        return [
            check(f"{ok}: check failed", None, row["error"] or "Guardian's reply had no score.")
        ]
    passed = bool(row["passed"])
    return [
        check(ok if passed else bad, passed, f"Granite Guardian, {stamp(row['created_at'])} UTC")
    ]


def attach_checks(store: Store, turns: list[ChatTurn]) -> None:
    """Refresh each turn's badges from ``verdicts`` (a verify job may have run since the page loaded them)."""
    ids = [t.turn_id for t in turns if t.turn_id is not None]
    found = store.verdicts(turn_ids=ids)
    for t in turns:
        if t.turn_id is not None:
            t.checks = turn_checks(found.get(("turn", t.turn_id), []))


def unchecked_answers(store: Store) -> int:
    return len(store.turns_to_verify([c.id for c in TURN_CRITERIA]))


def summary_checks(store: Store, extraction_id: int) -> list[dict[str, Any]]:
    """The summary's checks against the library's current criteria: one row per criterion, unchecked ones included."""
    rows = {
        r["criterion_id"]: r
        for r in store.verdicts(extraction_ids=[extraction_id]).get(
            ("extraction", extraction_id), []
        )
    }
    out = []
    for c in summary_criteria(store):
        row = rows.get(c.id)
        if row is None:
            result = "not checked yet"
        elif row["passed"] is None:
            result = f"check failed: {row['error'] or 'no score'}"
        else:
            result = "met" if row["passed"] else "not met"
        out.append({"Check": c.text, "Result": result})
    return out


def summary_criteria_text(store: Store) -> str:
    return "\n".join(c.text for c in summary_criteria(store))


def save_summary_criteria(store: Store, text: str) -> list[str]:
    """The Library's checks box → the library's summary criteria (one per line; blank lines dropped)."""
    texts = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    store.set_summary_criteria(texts)
    return texts


# ── conversation ──


@dataclass
class SourceRef:
    number: int  # [n] in the answer
    citation: str  # "report.pdf · p. 2" / "meeting.flac · 0:12"
    text: str
    context: str = ""
    cited: bool = False
    source_id: int | None = None  # None when the source has since been deleted


@dataclass
class ChatTurn:
    question: str
    answer: str
    declined: bool = False
    sources: list[SourceRef] = field(default_factory=list)
    seconds: float | None = None
    thinking: str | None = None
    reasoning: str = ""
    turn_id: int | None = None
    checks: list[Check] = field(
        default_factory=list
    )  # Guardian verdicts (empty until a verify job ran)

    @property
    def cited(self) -> list[SourceRef]:
        return [s for s in self.sources if s.cited]


def ref(number: int, hit: Hit, cited: bool) -> SourceRef:
    return SourceRef(number, hit.citation, hit.text, hit.context, cited, hit.source_id)


def turn_from_answer(answer: Any) -> ChatTurn:
    """A ``granit.reason.qa.Answer`` as a conversation turn."""
    cited = {h.chunk_id for h in answer.cited}
    return ChatTurn(
        question=answer.question,
        answer=answer.text,
        declined=answer.declined,
        sources=[ref(i, h, h.chunk_id in cited) for i, h in enumerate(answer.sources, start=1)],
        seconds=answer.latency.get("total_s"),
        reasoning=answer.reply.reasoning if answer.reply else "",
        turn_id=answer.turn_id,
    )


def turns_from_history(store: Store, limit: int = 20) -> list[ChatTurn]:
    """The last ``limit`` answers from ``qa_turns``, oldest first (the Ask page's persistent history)."""
    rows = list(reversed(store.turns(limit)))
    ids = {i for r in rows for i in json.loads(r["source_chunk_ids"])}
    chunks = store.chunks_by_id(sorted(ids))
    turns = []
    for r in rows:
        cited = set(json.loads(r["cited_chunk_ids"]))
        sources = []
        for n, chunk_id in enumerate(json.loads(r["source_chunk_ids"]), start=1):
            row = chunks.get(chunk_id)
            if row is None:
                sources.append(
                    SourceRef(n, "(removed from the library)", "", cited=chunk_id in cited)
                )
            else:
                sources.append(ref(n, hit_from_row(row, 0.0), chunk_id in cited))
        turns.append(
            ChatTurn(
                question=r["question"],
                answer=r["answer"],
                declined=bool(r["declined"]),
                sources=sources,
                seconds=json.loads(r["latency"]).get("total_s"),
                thinking=r["thinking"],
                turn_id=r["id"],
            )
        )
    return turns


# ── library ──

KIND_LABEL = {"audio": "Recording", "document": "Document"}


def duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f} s"
    return timestamp(seconds)


def size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def stamp(iso: str) -> str:
    """``2026-10-04T21:10:38+00:00`` → ``2026-10-04 21:10`` (stored times are UTC)."""
    return iso[:16].replace("T", " ")


def model_label(model: str) -> str:
    """``ibm-granite/granite-vision-4.1-4b@37d591f…`` → ``granite-vision-4.1-4b @ 37d591f``."""
    name, _, revision = model.partition("@")
    name = name.rsplit("/", 1)[-1]
    return f"{name} @ {revision[:7]}" if revision else name


def added(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def details(source: Source) -> str:
    """One line about what's in a source: "3 pages · 2 tables · 1 chart" or "31 s · 64% speech"."""
    info = source.info
    if source.kind == "audio":
        if "duration_s" not in info:
            return ""
        parts = [duration(info["duration_s"])]
        if info["duration_s"]:
            parts.append(f"{100 * info.get('speech_s', 0) / info['duration_s']:.0f}% speech")
        return " · ".join(parts)
    parts = []
    for key, word in (("pages", "page"), ("tables", "table"), ("charts", "chart")):
        if info.get(key):
            parts.append(f"{info[key]} {word}{'s' * (info[key] != 1)}")
    if info.get("accurate_tables"):
        parts.append("accurate tables")
    return " · ".join(parts)


def library_rows(sources: list[Source]) -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "Name": s.name,
            "Type": KIND_LABEL[s.kind],
            "Status": s.status,
            "Contents": details(s),
            "Chunks": s.info.get("chunks"),
            "Size": size(s.size_bytes),
            "Added": added(s.added_at),
        }
        for s in sources
    ]


TASK_LABEL = {"ingest": "Ingest", "extract": "Extract fields", "verify": "Check with Guardian"}


def job_rows(rows: list[Any]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        params = json.loads(r["params"])
        task = TASK_LABEL.get(r["task"], r["task"])
        if params.get("vision_tables"):
            task += " (accurate tables)"
        out.append(
            {
                "id": r["id"],
                "File": r["source_name"] or "Answers and summaries",
                "Task": task,
                "Status": r["status"],
                "Attempts": r["attempts"],
                "Error": r["error"] or "",
                "Queued": added(r["created_at"]),
            }
        )
    return out


# ── source contents ──

IMAGE_COMMENT = re.compile(r"<!--\s*image\s*-->")


def document_md(markdown: str) -> str:
    """Docling's Markdown for ``st.markdown``: image placeholders as text, every ``$`` escaped (amounts, never LaTeX)."""
    return safe_md(IMAGE_COMMENT.sub("*(image)*", markdown))


def transcript_md(transcript: dict[str, Any]) -> str:
    """One line per segment with its time, ready for ``st.markdown``."""
    return "\n\n".join(
        f"`{timestamp(s['start'])}` {safe_md(s['text'])}" for s in transcript["segments"]
    )


def vocabulary_text(terms: list[str]) -> str:
    return "\n".join(terms)


def save_vocabulary(store: Store, text: str) -> list[str]:
    """The Ingest page's names-and-terms box → the library's vocabulary (one per line or comma-separated)."""
    terms = parse_terms(text)
    store.set_vocabulary(terms)
    return terms


def other_vocabulary(source: Source, terms: list[str]) -> bool:
    """A recording transcribed with a different names list than the library's current one (re-transcribe to apply it)."""
    return source.kind == "audio" and source.info.get("vocabulary", []) != terms


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def source_param(value: str | None) -> int | None:
    """``?source=3`` in a page URL (deep links into Library and Extract) → 3; anything else → None."""
    return int(value) if value and value.isdigit() else None


def latest(extractions: list[Any], kind: str) -> Any:
    rows = [e for e in extractions if e["kind"] == kind]
    return rows[-1] if rows else None


# ── forms ──


@dataclass
class FormResult:
    fields: list[dict[str, Any]]
    valid: bool
    missing: list[str]
    errors: list[str]
    model: str
    created_at: str


def form_result(row: Any) -> FormResult:
    """A ``form`` extraction as rows of (field, value), in the schema's order, missing fields last and empty."""
    data = json.loads(row["content"]) if row["content"] else {}
    schema = json.loads(row["schema"]) if row["schema"] else {}
    order = list(schema.get("properties", {})) or list(data)
    order += [k for k in data if k not in order]
    fields = [
        {
            "Field": k,
            "Value": "" if data.get(k) is None else str(data[k]),
            "Found": data.get(k) is not None,
        }
        for k in order
    ]
    return FormResult(
        fields=fields,
        valid=bool(row["valid"]),
        missing=json.loads(row["missing"]),
        errors=json.loads(row["errors"]),
        model=row["model"],
        created_at=row["created_at"],
    )


def check_schema(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """(schema, None) if ``text`` is a JSON Schema for an object with fields; (None, what's wrong) otherwise."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    if not isinstance(schema, dict):
        return None, "The schema must be a JSON object."
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return None, f"Not a valid JSON Schema: {exc.message}"
    if schema.get("type") != "object" or not schema.get("properties"):
        return (
            None,
            'The schema must describe an object with fields: {"type": "object", "properties": {…}}.',
        )
    return schema, None


EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "description": "The invoice number"},
        "invoice_date": {
            "type": "string",
            "format": "date",
            "description": "The date the invoice was issued",
        },
        "due_date": {"type": "string", "format": "date", "description": "The date payment is due"},
        "vendor_name": {"type": "string", "description": "The company that issued the invoice"},
        "customer_name": {"type": "string", "description": "The company being billed"},
        "total_amount": {"type": "string", "description": "The total amount due, with currency"},
        "purchase_order": {
            "type": "string",
            "description": "The customer's purchase order number",
        },
    },
    "required": ["invoice_number", "total_amount"],
}


def page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def page_png(path: Path, page: int = 1, scale: float = 2.0) -> bytes:
    """A document page as PNG bytes for display (PDFs rendered with pdfium; images re-encoded as they are)."""
    import io

    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            image = pdf[page - 1].render(scale=scale).to_pil()
        finally:
            pdf.close()
    else:
        image = Image.open(path)
        image.load()
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# ── uploads ──

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in AUDIO_EXTENSIONS | DOCUMENT_EXTENSIONS)
DOCUMENT_TYPES = sorted(ext.lstrip(".") for ext in DOCUMENT_EXTENSIONS)


def add_upload(
    store: Store, name: str, data: bytes | memoryview, accurate_tables: bool = False
) -> tuple[Source, bool]:
    """Store an uploaded file and queue its ingest; "Accurate tables" applies to documents only."""
    import tempfile

    suffix = Path(name).suffix.lower()
    params = (
        {"vision_tables": True} if accurate_tables and kind_of(Path(name)) == "document" else None
    )
    with tempfile.TemporaryDirectory(prefix="granit-upload-") as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        return store.add_file(path, name=name, params=params)
