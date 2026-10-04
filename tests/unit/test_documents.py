"""Document ingest plumbing without models (M3): formats, crop geometry, page rendering, chart annotations."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from granit.ingest import documents
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
