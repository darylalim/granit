"""Document golden tests on the Mac (``uv run pytest -m model``), M3. Every test has an explicit threshold.

Ground truth: the model card's ``chart.jpg`` / ``table.png`` / ``invoice.png`` (values read from the images, in
``fixtures/documents/card_table_truth.json`` and below) and the generated fixtures' ``manifest.json``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from granit.config import HUB_MODELS
from granit.ingest.documents import DocumentIngestor, render_pages
from granit.ingest.vision import VisionModel
from granit.models.download import local_snapshot
from tests.conftest import ROOT

pytestmark = pytest.mark.model

FIXTURES = ROOT / "tests" / "fixtures" / "documents"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
TABLE_TRUTH = json.loads((FIXTURES / "card_table_truth.json").read_text())["rows"]
CARD_CHART = [
    ["NJ", "4.6", "4.1"],
    ["CT", "4.7", "4.1"],
    ["DE", "4.5", "3.8"],
    ["NY", "4.7", "4.1"],
    ["PA", "4.9", "4.3"],
]
CARD_INVOICE = {
    "invoice_date": "17-MAY-2020",
    "order_number": "655283253",
    "seller_tax_id": "DE194149069",
}
MIN_TABLE_ACCURACY = (
    0.98  # share of the card table's 96 values extracted exactly (measured: 96/96 for both)
)


@pytest.fixture(scope="module")
def card() -> Path:
    return local_snapshot(HUB_MODELS["vision"])


@pytest.fixture(scope="module")
def ingestor() -> Iterator[DocumentIngestor]:
    yield DocumentIngestor(vision_tables=False).load()


@pytest.fixture(scope="module")
def vision(ingestor: DocumentIngestor) -> VisionModel:
    return ingestor.vision


def number(text: str) -> str:
    return re.sub(r"[*\s,]", "", text.replace("−", "-").replace("–", "-"))


def table_accuracy(rows: dict[str, list[str]]) -> float:
    """Share of the card table's values found exactly in the same row and column."""
    total = sum(len(v) for v in TABLE_TRUTH.values())
    hits = sum(
        number(a) == number(b)
        for label, values in TABLE_TRUTH.items()
        for a, b in zip(rows.get(label, []), values, strict=False)
    )
    return hits / total


def markdown_rows(markdown: str) -> list[list[str]]:
    """Cells of every Markdown table row (separator lines skipped)."""
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("|") and not set(line.strip()) <= set("|-: ")
    ]


class _HtmlRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: str | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        del attrs
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(self._cell.strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell += data


def html_rows(html: str) -> list[list[str]]:
    parser = _HtmlRows()
    parser.feed(html)
    return [r for r in parser.rows if r]


# ── the model card's examples ──


def test_card_chart_is_extracted_exactly(
    ingestor: DocumentIngestor, card: Path, tmp_path: Path
) -> None:
    result = ingestor.ingest(card / "chart.jpg", tmp_path)
    assert (
        result.pictures == 2 and result.charts == 1
    )  # the chart, and the logo (too small for Vision)
    (chart,) = result.extractions
    assert chart.valid and chart.data[0] == ["State", "2017", "2018"]
    assert chart.data[1:] == CARD_CHART
    assert ["NJ", "4.6", "4.1"] in markdown_rows(
        result.markdown
    )  # under the chart, in the Markdown


def test_card_table_with_docling(ingestor: DocumentIngestor, card: Path, tmp_path: Path) -> None:
    result = ingestor.ingest(card / "table.png", tmp_path)
    assert result.tables == 1
    rows = markdown_rows(result.markdown)
    assert table_accuracy({r[0]: r[1:] for r in rows}) >= MIN_TABLE_ACCURACY


def test_card_table_with_vision(vision: VisionModel, card: Path) -> None:
    """The optional Vision table path (DOCUMENT_VISION_TABLES): slower (~30 s), as accurate on this table."""
    extraction = vision.table_to_html(str(card / "table.png"))
    assert extraction.valid
    rows = html_rows(extraction.data[0])
    assert table_accuracy({r[0]: r[1:] for r in rows}) >= MIN_TABLE_ACCURACY


def test_card_invoice_fields(vision: VisionModel, card: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "invoice_date": {"type": "string", "description": "The date the invoice was issued"},
            "order_number": {
                "type": "string",
                "description": "The unique identifier for the order",
            },
            "seller_tax_id": {
                "type": "string",
                "description": "The tax identification number of the seller",
            },
        },
    }
    extraction = vision.extract_fields(render_pages(card / "invoice.png", 1), schema)
    assert extraction.valid and extraction.data == CARD_INVOICE


# ── generated fixtures ──


def test_scanned_report(ingestor: DocumentIngestor, tmp_path: Path) -> None:
    expected = MANIFEST["report.pdf"]
    result = ingestor.ingest(FIXTURES / "report.pdf", tmp_path)
    assert result.pages == expected["pages"] and result.charts == expected["charts"]
    for text in expected["texts"]:
        assert text in result.markdown
    rows = markdown_rows(result.markdown)
    for row in [expected["table"]["header"], *expected["table"]["rows"]]:
        assert row in rows, f"table row {row} not in {rows}"
    (chart,) = result.extractions
    assert chart.page == 2
    assert [r[0] for r in chart.data[1:]] == expected["chart"]["categories"]
    assert [float(r[1]) for r in chart.data[1:]] == expected["chart"]["values"]
    # Only the chart was sent to Vision: the logo on page 1 (46 pt) is below MIN_PICTURE_SIZE.
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == ["p2_picture1.png"]
    assert (tmp_path / "document.md").read_text() == result.markdown
    assert json.loads((tmp_path / "document.json").read_text())["name"]


@pytest.mark.skipif("memo.pdf" not in MANIFEST, reason="memo.pdf needs Chrome to generate")
def test_digital_pdf(ingestor: DocumentIngestor, tmp_path: Path) -> None:
    expected = MANIFEST["memo.pdf"]
    result = ingestor.ingest(FIXTURES / "memo.pdf", tmp_path)
    for text in expected["texts"]:
        assert text in result.markdown
    rows = markdown_rows(result.markdown)
    for row in [expected["table"]["header"], *expected["table"]["rows"]]:
        assert row in rows, f"table row {row} not in {rows}"
    assert result.charts == 0 and not result.extractions


@pytest.mark.parametrize("name", ["invoice.png", "invoice.pdf"])
def test_invoice_fields_with_iso_dates(vision: VisionModel, name: str) -> None:
    schema = json.loads((FIXTURES / "invoice_schema.json").read_text())
    extraction = vision.extract_fields(render_pages(FIXTURES / name, 4), schema)
    assert extraction.valid, extraction.errors
    assert extraction.data == MANIFEST["invoice.png"]["expected"]  # dates normalized to ISO 8601
    assert list(extraction.missing) == MANIFEST["invoice.png"]["absent"]
