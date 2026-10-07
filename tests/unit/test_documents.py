"""Document ingest plumbing without models (M3): formats, crop geometry, page rendering, chart annotations."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from granit.ingest import documents
from granit.ingest.vision import VisionModel
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "documents"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.mark.parametrize("name", ["a.pdf", "a.PNG", "a.jpg", "a.jpeg", "a.tiff", "a.webp"])
def test_supported_formats(name: str) -> None:
    documents.check_format(Path(name))


@pytest.mark.parametrize("name", ["a.docx", "a.txt", "noext"])
def test_unsupported_formats(name: str) -> None:
    with pytest.raises(documents.UnsupportedDocument, match="use PDF or an image"):
        documents.check_format(Path(name))


def test_crop_name_matches_the_plan() -> None:
    assert documents.crop_name(3, "table", 1) == "p3_table1.png"
    assert documents.crop_name(None, "picture", 2) == "p0_picture2.png"


def test_crop_box_scales_pads_and_clamps() -> None:
    # A 100×200-pt page rendered at 3× → 300×600 px; 1 % padding = 1 pt.
    assert documents.crop_box((10, 20, 50, 80), (100, 200), (300, 600), padding=0.01) == (
        27,
        57,
        153,
        243,
    )
    # At the page edge the box is clamped to the image.
    assert documents.crop_box((0, 0, 100, 200), (100, 200), (300, 600), padding=0.05) == (
        0,
        0,
        300,
        600,
    )


def fake_item(page: int, bbox: tuple[float, float, float, float]) -> types.SimpleNamespace:
    from docling_core.types.doc import BoundingBox, CoordOrigin

    left, top, right, bottom = bbox
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    return types.SimpleNamespace(prov=[types.SimpleNamespace(page_no=page, bbox=box)])


def fake_doc(width: float, height: float, pages: int = 2) -> types.SimpleNamespace:
    size = types.SimpleNamespace(width=width, height=height)
    return types.SimpleNamespace(
        pages={n: types.SimpleNamespace(size=size) for n in range(1, pages + 1)}
    )


def test_small_pictures_skip_vision() -> None:
    assert documents.is_large_enough(
        fake_item(1, (71.4, 122.9, 523.6, 461.4))
    )  # the report's chart
    assert not documents.is_large_enough(fake_item(1, (497.4, 32.0, 543.8, 77.5)))  # its 46 pt logo
    assert not documents.is_large_enough(
        fake_item(1, (41.0, 689.7, 180.2, 752.6))
    )  # card logo, 63 px tall
    assert not documents.is_large_enough(types.SimpleNamespace(prov=[]))


def test_page_images_crop_a_pdf_region_at_crop_scale() -> None:
    pages = documents.PageImages(FIXTURES / "report.pdf")
    try:
        cropped = pages.crop(fake_item(2, (71.4, 122.9, 523.6, 461.4)), fake_doc(595.0, 842.0))
        assert cropped is not None
        image, page = cropped
        assert page == 2
        # (523.6 − 71.4 + 2 × padding) pt × 3 ≈ 1410 px wide
        assert 1390 <= image.width <= 1430
        width, height = pages.page(2).size  # A4 is 595.28 × 841.89 pt
        assert abs(width - 595.28 * documents.CROP_SCALE) <= 1
        assert abs(height - 841.89 * documents.CROP_SCALE) <= 1
    finally:
        pages.close()


def test_page_images_use_images_as_they_are() -> None:
    pages = documents.PageImages(FIXTURES / "invoice.png")
    image = pages.page(1)
    assert image.size == (1240, 1754)
    assert pages.crop(types.SimpleNamespace(prov=[]), fake_doc(1240, 1754)) is None
    pages.close()


def test_render_pages_for_form_extraction() -> None:
    assert len(documents.render_pages(FIXTURES / "report.pdf", max_pages=4)) == 2
    assert len(documents.render_pages(FIXTURES / "report.pdf", max_pages=1)) == 1
    assert len(documents.render_pages(FIXTURES / "invoice.png", max_pages=4)) == 1


def test_chart_data_renders_under_the_chart_in_markdown() -> None:
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.TEXT, text="Before.")
    picture = doc.add_picture()
    documents.attach_chart(
        picture, [["Quarter", "Revenue"], ["Q1", "120"], ["Q2", "145"]], "vision@rev"
    )
    doc.add_text(label=DocItemLabel.TEXT, text="After.")
    markdown = doc.export_to_markdown()
    before, chart, after = (
        markdown.index("Before."),
        markdown.index("| Q2 "),
        markdown.index("After."),
    )
    assert before < markdown.index("<!-- image -->") < chart < after
    assert picture.meta.tabular_chart.created_by == "vision@rev"
    assert "tabular_chart" in str(doc.export_to_dict())  # kept in document.json too


def test_an_empty_docling_table_is_filled_from_vision() -> None:
    """M7: Granite-Docling found dense / long tables but emitted an empty <otsl>; Vision's cells go into the document."""
    from docling_core.types.doc import DoclingDocument, TableData

    from granit.ingest.vision import html_grid

    doc = DoclingDocument(name="t")
    table = doc.add_table(data=TableData(num_rows=0, num_cols=0))
    assert documents.is_empty_table(table)
    html = "<table><tr><th>Week</th><th>Boise</th></tr><tr><td>7</td><td>571</td></tr><tr><td>8</td></tr></table>"
    documents.fill_table(table, html_grid(html))
    assert not documents.is_empty_table(table)
    assert [[c.text for c in row] for row in table.data.grid] == [
        ["Week", "Boise"],
        ["7", "571"],
        ["8", ""],
    ]
    lines = doc.export_to_markdown().splitlines()
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
    assert ["7", "571"] in rows  # in the Markdown, so searchable and cited like any table


def test_fixture_manifest_is_consistent() -> None:
    report = MANIFEST["report.pdf"]
    assert report["pages"] == 2 and report["charts"] == 1
    assert len(report["chart"]["categories"]) == len(report["chart"]["values"])
    schema = json.loads((FIXTURES / "invoice_schema.json").read_text())
    expected = MANIFEST["invoice.png"]["expected"]
    assert set(expected) | set(MANIFEST["invoice.png"]["absent"]) == set(schema["properties"])


# ── the text-layer safety net (M7: Granite-Docling dropped field values next to their labels) ──

EVAL_FILES = ROOT / "eval" / "public" / "files"


def as_docling_saw_inv01() -> Any:
    """inv-01 as Granite-Docling transcribed it: labels without their values, and the line items (Vision filled them)."""
    from docling_core.types.doc import (
        BoundingBox,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
        TableData,
    )

    def on(text: str) -> ProvenanceItem:
        return ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))
        )

    doc = DoclingDocument(name="inv-01")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    for label, text in [
        (DocItemLabel.SECTION_HEADER, "Granite Supply Co."),
        (DocItemLabel.TEXT, "12 Quarry Lane, Barre, VT 05641"),
        (DocItemLabel.SECTION_HEADER, "INVOICE"),
        (DocItemLabel.TEXT, "Invoice number:"),
        (DocItemLabel.TEXT, "Invoice date:"),
        (DocItemLabel.TEXT, "Due date:"),
        (DocItemLabel.TEXT, "Purchase order:"),
        (DocItemLabel.TEXT, "Bill to:"),
        (DocItemLabel.TEXT, "Northwind Logistics Ltd. 400 Harbor Road, Portland, ME 04101"),
    ]:
        doc.add_text(label=label, text=text, prov=on(text))
    table = doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=on("table"))
    documents.fill_table(
        table,
        [
            ["Description", "Qty", "Unit price", "Amount"],
            ["Granite countertop slabs", "12", "$780.00", "$9,360.00"],
            ["Edge polishing", "12", "$95.00", "$1,140.00"],
            ["Delivery and installation", "1", "$2,140.00", "$2,140.00"],
            ["Total due", "", "", "$12,640.00"],
        ],
    )
    return doc


def test_the_text_layer_gives_back_exactly_what_docling_dropped() -> None:
    lines = documents.text_layer_lines(EVAL_FILES / "invoices" / "inv-01.pdf")
    assert "Invoice number: GS-2026-0117" in lines[1]
    doc = as_docling_saw_inv01()
    missing = documents.missing_lines(doc, lines)
    assert missing == {
        1: [
            "Invoice number: GS-2026-0117",
            "Invoice date: August 3, 2026",
            "Due date: September 2, 2026",
            "Purchase order: PO-48213",
        ]
    }
    assert documents.add_text_layer_lines(doc, missing) == 4
    assert documents.missing_lines(doc, lines) == {}  # nothing left to add


def test_recovered_lines_are_chunked_with_their_page_and_heading() -> None:
    from granit.store.chunking import chunk_document

    doc = as_docling_saw_inv01()
    lines = documents.text_layer_lines(EVAL_FILES / "invoices" / "inv-01.pdf")
    documents.add_text_layer_lines(doc, documents.missing_lines(doc, lines))
    recovered = [c for c in chunk_document(doc) if "GS-2026-0117" in c.text]
    assert len(recovered) == 1 and recovered[0].page_start == 1
    assert recovered[0].context == "Granite Supply Co. > " + documents.TEXT_LAYER_HEADING.format(
        page=1
    )


def test_text_layer_hyphens_are_real_hyphens() -> None:
    """pdfium returns this PDF's hyphen as U+FFFE; the recovered ID must read PO-48502 (Docling had misread PO48502)."""
    (line,) = [
        x
        for x in documents.text_layer_lines(EVAL_FILES / "invoices" / "inv-06.pdf")[1]
        if "Purchase order" in x
    ]
    assert line.endswith("Purchase order: PO-48502") and "\ufffe" not in line


def test_images_and_scans_have_no_text_layer() -> None:
    assert documents.text_layer_lines(FIXTURES / "invoice.png") == {}
    assert (
        documents.text_layer_lines(EVAL_FILES / "invoices" / "inv-05.pdf") == {}
    )  # a scan: pictures only


# ── the Vision page pass ──


def receipt_as_docling_saw_it() -> Any:
    """cord-094 as Granite-Docling read it: the whole photo as one picture, nothing else."""
    from docling_core.types.doc import DocItemLabel, DoclingDocument, Size

    doc = DoclingDocument(name="cord-094")
    doc.add_page(page_no=1, size=Size(width=864, height=1296))
    doc.add_picture(prov=documents._on(1, ""))
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="1 / 1", prov=documents._on(1, "1 / 1"))
    return doc


class PageReader(VisionModel):
    """Vision that reads every page as ``lines`` and counts the pages it was asked to read."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.lines, self.pages = lines, 0

    def read_page(self, image: Any) -> list[str]:
        self.pages += 1
        return self.lines


RECEIPT_LINES = ["1 PHO TAI CHIN (R) 63,000", "Sub Total 119,000", "Grand Total 140,063"]


def test_a_page_with_little_text_needs_the_page_pass() -> None:
    assert documents.needs_page_pass(documents.page_markdown(receipt_as_docling_saw_it(), 1))
    words = " ".join(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60))
    assert not documents.needs_page_pass(words.replace("word", "w"))


def test_replace_page_swaps_docling_for_vision_and_cites_the_page() -> None:
    from granit.store.chunking import chunk_document

    doc = receipt_as_docling_saw_it()
    documents.replace_page(doc, 1, RECEIPT_LINES)
    assert not doc.pictures and "1 / 1" not in documents.page_markdown(doc, 1)
    (chunk,) = [c for c in chunk_document(doc) if "140,063" in c.text]
    assert chunk.page_start == 1
    assert chunk.context.endswith(documents.VISION_PAGE_HEADING.format(page=1))


def test_read_pages_replaces_an_unreadable_image_page() -> None:
    ingestor = documents.DocumentIngestor(vision=PageReader(RECEIPT_LINES))
    doc = receipt_as_docling_saw_it()
    assert ingestor._read_pages(doc, FIXTURES / "invoice.png", {}) == 1
    assert "Grand Total 140,063" in doc.export_to_markdown()


def test_read_pages_keeps_docling_when_vision_reads_less() -> None:
    ingestor = documents.DocumentIngestor(vision=PageReader([]))
    doc = receipt_as_docling_saw_it()
    assert ingestor._read_pages(doc, FIXTURES / "invoice.png", {}) == 0
    assert len(doc.pictures) == 1


def test_a_looping_page_is_cleared_even_if_vision_reads_nothing() -> None:
    from docling_core.types.doc import DocItemLabel

    doc = receipt_as_docling_saw_it()
    for _ in range(30):
        doc.add_text(label=DocItemLabel.TEXT, text="loc>loc>loc>201", prov=documents._on(1, "x"))
    ingestor = documents.DocumentIngestor(vision=PageReader([]))
    assert ingestor._read_pages(doc, FIXTURES / "invoice.png", {}) == 1
    assert "loc>" not in doc.export_to_markdown()


def test_digital_pages_skip_vision_but_lose_a_loop() -> None:
    """A page with a text layer is never read by Vision; if Docling looped on it, it's cleared for the safety net."""
    from docling_core.types.doc import DocItemLabel

    reader = PageReader(RECEIPT_LINES)
    ingestor = documents.DocumentIngestor(vision=reader)
    doc = receipt_as_docling_saw_it()
    assert ingestor._read_pages(doc, FIXTURES / "invoice.png", {1: ["text layer"]}) == 0
    for _ in range(30):
        doc.add_text(label=DocItemLabel.TEXT, text="loc>loc>loc>201", prov=documents._on(1, "x"))
    assert ingestor._read_pages(doc, FIXTURES / "invoice.png", {1: ["text layer"]}) == 1
    assert reader.pages == 0 and "loc>" not in doc.export_to_markdown()


def test_items_without_provenance_go_with_the_page_before_them() -> None:
    """cord-020: Granite-Docling's looping output ("239" ×100) had no provenance and survived a per-page delete."""
    from docling_core.types.doc import DocItemLabel

    doc = receipt_as_docling_saw_it()
    for _ in range(30):
        doc.add_text(label=DocItemLabel.TEXT, text="239")
    assert documents.needs_page_pass(documents.docling_page_text(doc, 1))
    documents.replace_page(doc, 1, RECEIPT_LINES)
    assert not [t for t in doc.texts if t.text == "239"]
    assert "Grand Total 140,063" in doc.export_to_markdown()


def test_a_page_with_a_table_or_chart_keeps_doclings_reading() -> None:
    """M7 public set: a table or chart page has few words, but Docling's structure is what the eval scores."""
    from docling_core.types.doc import TableData

    doc = receipt_as_docling_saw_it()
    table = doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=documents._on(1, "t"))
    assert not documents.has_structure(doc, 1)  # an empty table doesn't count
    documents.fill_table(table, [["Week", "Boise"], ["7", "571"]])
    assert documents.has_structure(doc, 1)
    reader = PageReader(RECEIPT_LINES)
    assert (
        documents.DocumentIngestor(vision=reader)._read_pages(doc, FIXTURES / "invoice.png", {})
        == 0
    )
    assert reader.pages == 0

    chart = receipt_as_docling_saw_it()
    documents.attach_chart(chart.pictures[0], [["Quarter", "Revenue"], ["Q1", "120"]])
    assert documents.has_structure(chart, 1)


def projections_table(doc: Any) -> Any:
    """The Fed projections table as Granite-Docling transcribed it (v29 private set): a spanning group row over the years."""
    from docling_core.types.doc import TableCell, TableData

    groups = ["Median 1", "Central Tendency 2", "Range 3"]
    years = ["2024", "2025", "2026", "Longer run"]
    cells = [
        TableCell(
            text="",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        )
    ]
    cells += [
        TableCell(
            text=g,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1 + 4 * i,
            end_col_offset_idx=5 + 4 * i,
        )
        for i, g in enumerate(groups)
    ]
    body = [["Variable", *years * 3]] + [
        [f"Variable {r}", *[f"{r}.{c}" for c in range(12)]] for r in range(30)
    ]
    cells += [
        TableCell(
            text=text,
            start_row_offset_idx=r + 1,
            end_row_offset_idx=r + 2,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
        )
        for r, row in enumerate(body)
        for c, text in enumerate(row)
    ]
    return doc.add_table(data=TableData(num_rows=len(body) + 1, num_cols=13, table_cells=cells))


def test_a_multi_level_header_is_merged_into_one_row() -> None:
    """v29 private set: Markdown kept only "Median 1 | Median 1 | …" as the header; the years were a body row."""
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument(name="t")
    table = projections_table(doc)
    assert documents.table_header_depth(table) == 2
    assert documents.merge_headers(doc) == 1
    grid = [[c.text for c in row] for row in table.data.grid]
    assert grid[0][:2] == ["Variable", "Median 1 / 2024"]
    assert grid[0][5:7] == ["Central Tendency 2 / 2024", "Central Tendency 2 / 2025"]
    assert grid[0][-1] == "Range 3 / Longer run"
    assert grid[1][:2] == ["Variable 0", "0.0"] and len(grid) == 31
    assert documents.merge_headers(doc) == 0  # already one row


def test_every_chunk_of_a_split_table_names_its_columns() -> None:
    from docling_core.types.doc import DoclingDocument

    from granit.store.chunking import chunk_document

    doc = DoclingDocument(name="t")
    doc.add_heading("Summary of Economic Projections")
    projections_table(doc)
    documents.merge_headers(doc)
    tables = [c for c in chunk_document(doc) if c.element == "table"]
    assert len(tables) > 1  # split by rows, the header repeated
    assert all("Central Tendency 2 / 2026" in c.text.splitlines()[0] for c in tables)


@pytest.mark.parametrize(
    ("rows", "depth", "header"),
    [
        # a rowspan repeats the stub's name: kept once
        (
            [["Region", "2025", "2025"], ["Region", "H1", "H2"], ["North", "1", "2"]],
            2,
            ["Region", "2025 / H1", "2025 / H2"],
        ),
        # three levels (Census: year / Number / Estimate), ragged Vision rows padded
        (
            [
                ["", "2022", "2022"],
                ["State", "Number", "Number"],
                ["", "Estimate", "Margin"],
                ["Ohio", "1"],
            ],
            3,
            ["State", "2022 / Number / Estimate", "2022 / Number / Margin"],
        ),
    ],
)
def test_merge_header(rows: list[list[str]], depth: int, header: list[str]) -> None:
    merged = documents.merge_header(rows, depth)
    assert merged[0] == header and merged[1:] == rows[depth:]


def test_one_row_headers_and_spanning_totals_are_left_alone() -> None:
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    rows = [["Description", "Amount"], ["Paper", "$478.40"]]
    assert documents.merge_header(rows, 0) == rows
    doc = DoclingDocument(name="t")
    cells = [
        TableCell(
            text=t,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
        )
        for r, row in enumerate([["Item", "Qty", "Amount"], ["Paper", "40", "$478.40"]])
        for c, t in enumerate(row)
    ]
    cells.append(
        TableCell(
            text="Total due",
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=0,
            end_col_offset_idx=2,
        )
    )
    doc.add_table(data=TableData(num_rows=3, num_cols=3, table_cells=cells))
    assert documents.merge_headers(doc) == 0  # a spanning totals row isn't a header
