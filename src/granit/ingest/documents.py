"""Document ingest (PLAN.md §2.3, M3): Granite-Docling → DoclingDocument + Markdown; charts (and optionally tables) → Vision.

1. **Docling's VLM pipeline** (Granite-Docling 258M on MLX) turns each page into DocTags → a ``DoclingDocument`` with
   text, tables (with cell structure) and pictures, each with page provenance for citations. PDF pages are rendered by
   **pypdfium2**: Docling's default backend (docling-parse) rendered a scanned page slightly darker and softer, and
   Granite-Docling then read nothing but the logo (M3 finding).
2. **Pictures** are cropped from our own page render (``CROP_SCALE``; Docling's own crops were offset at
   ``images_scale`` > 1). Pictures at least ``MIN_PICTURE_SIZE`` on both sides go to Granite Vision, which is asked whether
   each is a chart. Charts go through ``<chart2csv>``; the data is attached to the picture as a
   ``meta.tabular_chart``, so it appears as a table right under the chart in the Markdown (searchable
   and citable) and in ``document.json``.
3. **Tables** stay as Docling extracted them unless ``DOCUMENT_VISION_TABLES`` is on; then each table crop also goes
   through ``<tables_html>`` and is stored as an extraction. A **multi-level column header** ("Median" spanning four
   years) is merged into one row ("Median 1 / 2024"): Markdown has one header row, and a table split into chunks
   repeats only that row, so the years would be lost from every part but the first (v30, private set).
4. **Page pass:** a page without a text layer (an image, a scan) where Granite-Docling looped or found fewer than
   ``MIN_PAGE_WORDS`` distinct words is read again by Vision, line by line; Vision's reading replaces Docling's if it
   found more. A looping page is cleared either way, so a loop never reaches the index. Digital pages are covered by
   the **text-layer safety net** instead, which adds back lines Docling dropped.

Outputs land in ``<out_dir>/``: ``document.json``, ``document.md`` and ``crops/p<page>_<kind><n>.png`` (the regions sent
to Vision), matching ``data/derived/<source_id>/`` in PLAN.md §2.3.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granit.config import DOCUMENT_VISION_TABLES, HUB_MODELS
from granit.ingest.vision import Extraction, VisionModel

IMAGE_FORMATS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
DOCUMENT_FORMATS = frozenset({".pdf"}) | IMAGE_FORMATS
CROP_SCALE = (
    3.0  # PDF pages are rendered at 3× (216 dpi) for crops; images are used at their own resolution
)
MIN_PICTURE_SIZE = 72  # page units (1 inch on a PDF, 72 px on an image): smaller pictures (logos, icons) skip Vision
CROP_PADDING = 0.015  # of the page width, so axis labels and titles at a box's edge aren't clipped


class UnsupportedDocument(ValueError):
    pass


def check_format(path: Path) -> None:
    if path.suffix.lower() not in DOCUMENT_FORMATS:
        raise UnsupportedDocument(
            f"{path.name}: unsupported document type {path.suffix or '(no extension)'}; "
            f"use PDF or an image ({', '.join(sorted(IMAGE_FORMATS))})"
        )


def crop_name(page: int | None, kind: str, index: int) -> str:
    """``crops/p3_table1.png`` style names (PLAN.md §2.3); 1-based index within the page."""
    return f"p{page or 0}_{kind}{index}.png"


def crop_box(
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    image_size: tuple[int, int],
    padding: float = CROP_PADDING,
) -> tuple[int, int, int, int]:
    """A top-left-origin box in page units → a pixel box in a render of that page, padded and clamped."""
    (left, top, right, bottom), (page_w, page_h), (img_w, img_h) = bbox, page_size, image_size
    sx, sy = img_w / page_w, img_h / page_h
    pad = padding * page_w
    return (
        max(0, round((left - pad) * sx)),
        max(0, round((top - pad) * sy)),
        min(img_w, round((right + pad) * sx)),
        min(img_h, round((bottom + pad) * sy)),
    )


def attach_chart(picture: Any, rows: list[list[str]], created_by: str = "") -> None:
    """Attach CSV rows (header first) to a Docling picture as chart data (``meta.tabular_chart``).

    The Markdown export renders it as a table right under the chart's placeholder, so chart values are searchable
    and cited with the picture's page.
    """
    from docling_core.types.doc import TableCell, TableData
    from docling_core.types.doc.items.picture.meta import PictureMeta, TabularChartMetaField

    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    ]
    chart = TabularChartMetaField(
        chart_data=TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells),
        created_by=created_by or None,
    )
    if picture.meta is None:
        picture.meta = PictureMeta(tabular_chart=chart)
    else:
        picture.meta.tabular_chart = chart


def fill_table(table: Any, rows: list[list[str]]) -> None:
    """Replace a Docling table's cells with ``rows`` (header first), so the Markdown export and the chunks use them."""
    from docling_core.types.doc import TableCell, TableData

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    ]
    table.data = TableData(num_rows=len(rows), num_cols=width, table_cells=cells)


HEADER_JOIN = " / "


def merge_header(rows: list[list[str]], depth: int) -> list[list[str]]:
    """The first ``depth`` rows as one header row: each column's distinct texts, top down ("Median 1 / 2024")."""
    if depth < 2:
        return rows
    width = max(len(r) for r in rows)
    levels = [r + [""] * (width - len(r)) for r in rows[:depth]]
    header = []
    for column in zip(*levels, strict=True):
        parts: list[str] = []
        for text in column:
            if text.strip() and text not in parts:
                parts.append(text)
        header.append(HEADER_JOIN.join(parts))
    return [header, *rows[depth:]]


def table_header_depth(table: Any) -> int:
    from granit.ingest.vision import header_depth

    spanning = {
        c.start_row_offset_idx
        for c in table.data.table_cells
        if c.end_col_offset_idx - c.start_col_offset_idx > 1
    }
    return header_depth(spanning, table.data.num_rows)


def merge_headers(doc: Any) -> int:
    """Merge every multi-level column header Docling transcribed; returns the tables changed. Vision's tables are merged
    as they're filled (``fill_table`` keeps no spans, so they're skipped here)."""
    merged = 0
    for table in doc.tables:
        depth = table_header_depth(table)
        if depth:
            grid = [[c.text for c in row] for row in table.data.grid]
            fill_table(table, merge_header(grid, depth))
            merged += 1
    return merged


def is_empty_table(table: Any) -> bool:
    """Granite-Docling sometimes finds a table but transcribes none of it (M7: dense and long tables)."""
    return not table.data.table_cells or not any(c.text.strip() for c in table.data.table_cells)


# ── the text-layer safety net (M7 finding: Granite-Docling drops some text, e.g. field values next to their labels) ──

TEXT_LAYER_HEADING = "Text found only in the PDF's text layer (page {page})"
MISSING_SHARE = 0.5  # a line is missing if half its words are, or any token with a digit (an ID, amount or date) is


def text_layer_lines(path: Path) -> dict[int, list[str]]:
    """A digital PDF's own text, line by line, per page (1-based). Empty for images and for scans without a text layer."""
    if path.suffix.lower() != ".pdf":
        return {}
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        out = {}
        for number, page in enumerate(pdf, start=1):
            text = page.get_textpage().get_text_range()
            # pdfium reports some hyphens as U+FFFE (its soft-hyphen marker): PO-48502 came back as "PO\ufffe48502"
            lines = [" ".join(line.replace("\ufffe", "-").split()) for line in text.splitlines()]
            if lines := [line for line in lines if line]:
                out[number] = lines
        return out
    finally:
        pdf.close()


def _tokens(text: str) -> list[str]:
    from granit.store.text import canonical

    return [t for t in canonical(text).split() if len(t) > 1 or t.isdigit()]


def missing_lines(doc: Any, lines_by_page: dict[int, list[str]]) -> dict[int, list[str]]:
    """Text-layer lines whose content Docling didn't capture on that page, compared as canonical tokens (``$9,360.00`` =
    ``9360``, any date format = ISO). Headers and footers count as captured, so they aren't recovered as noise."""
    out = {}
    for page, lines in lines_by_page.items():
        captured = set(_tokens(page_markdown(doc, page)))
        missing = []
        for line in lines:
            tokens = _tokens(line)
            gone = [t for t in tokens if t not in captured]
            has_digit = any(c.isdigit() for t in gone for c in t)
            if gone and (has_digit or len(gone) / len(tokens) >= MISSING_SHARE):
                missing.append(line)
        if missing:
            out[page] = missing
    return out


def _on(page: int, text: str) -> Any:
    """Provenance for text we add: the whole page (no box), so chunks cite the right page."""
    from docling_core.types.doc import BoundingBox, ProvenanceItem

    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))
    )


def add_lines(doc: Any, page: int, heading: str, lines: list[str]) -> int:
    """Append lines under a top-level heading (not under the last heading), with that page's provenance."""
    from docling_core.types.doc import DocItemLabel

    doc.add_heading(heading, level=1, prov=_on(page, heading))
    for line in lines:
        doc.add_text(label=DocItemLabel.TEXT, text=line, prov=_on(page, line))
    return len(lines)


MIN_ROW_VALUES = 2  # a row is matched to a text-layer line by at least this many value tokens


def _line_label(line: str, values: list[str]) -> str | None:
    """The words before ``values`` (canonical tokens) when ``line`` is a label followed by exactly those values."""
    words = line.split()
    for k in range(1, len(words)):
        if _tokens(" ".join(words[k:])) == values:
            return " ".join(words[:k])
    return None


def relabel_rows(doc: Any, lines_by_page: dict[int, list[str]]) -> int:
    """Correct row labels from a digital PDF's text layer; returns the rows changed. Granite-Docling shifted one block's
    labels down a row (v30 private set: the March federal funds rate under *Memo: Projected appropriate policy path*,
    December's under *Federal funds rate*), so the values were right and the label wrong. A body row whose values appear on
    exactly one text-layer line, after a label, takes that line's label; ambiguous or unmatched rows are left alone."""
    changed = 0
    for table in doc.tables:
        pages = {p.page_no for p in table.prov}
        lines = [line for page in sorted(pages) for line in lines_by_page.get(page, [])]
        if not lines:
            continue
        for row in table.data.grid[1:]:
            if not row or any(c.column_header for c in row):
                continue
            label, values = row[0], _tokens(" ".join(c.text for c in row[1:]))
            if len(values) < MIN_ROW_VALUES:
                continue
            found = {found for line in lines if (found := _line_label(line, values))}
            if len(found) != 1:
                continue
            new = found.pop()
            if "".join(_tokens(new)) != "".join(_tokens(label.text)):
                label.text = new
                changed += 1
    return changed


def add_text_layer_lines(doc: Any, missing: dict[int, list[str]]) -> int:
    """Append the missing lines under one heading per page."""
    return sum(
        add_lines(doc, page, TEXT_LAYER_HEADING.format(page=page), lines)
        for page, lines in sorted(missing.items())
    )


# ── the Vision page pass (M7 private set: Granite-Docling looped or read nothing on receipt photos and a scan) ──

VISION_PAGE_HEADING = "Text read from the page image by Granite Vision (page {page})"
MIN_PAGE_WORDS = (
    40  # distinct words: below this, a page without a text layer is read again by Vision
)
PAGE_SCALE = 2.0  # PDF render scale for page reading (tested at 2×; images are used as they are)


def page_markdown(doc: Any, page: int) -> str:
    from docling_core.types.doc import ContentLayer

    return doc.export_to_markdown(page_no=page, included_content_layers=set(ContentLayer))


def page_items(doc: Any, page: int) -> list[Any]:
    """Items on ``page`` only. An item without provenance (Granite-Docling's looping output has none) belongs to the
    page of the item before it in reading order, so it's found and removed with its page."""
    from docling_core.types.doc import ContentLayer

    current, out = min(doc.pages, default=1), []
    for item, _ in doc.iterate_items(
        included_content_layers=set(ContentLayer), traverse_pictures=True
    ):
        pages = {p.page_no for p in getattr(item, "prov", None) or []}
        if len(pages) > 1:  # spans pages: never removed with one of them
            current = max(pages)
            continue
        current = pages.pop() if pages else current
        if current == page:
            out.append(item)
    return out


def docling_page_text(doc: Any, page: int) -> str:
    """The page's Markdown plus the text of its items without provenance (which a per-page export leaves out)."""
    unplaced = [getattr(i, "text", "") for i in page_items(doc, page) if not i.prov]
    return "\n".join([page_markdown(doc, page), *unplaced])


def needs_page_pass(text: str) -> bool:
    """Docling's reading of a page looped, or found too little text to be the whole page."""
    from granit.ingest.vision import distinct_words, is_looping

    return is_looping(text) or distinct_words(text) < MIN_PAGE_WORDS


def has_structure(doc: Any, page: int) -> bool:
    """The page has a table with cells or a chart with data: a page of numbers with few words that Docling read well
    (M7 public set: the table fixtures and the chart report)."""
    from docling_core.types.doc import PictureItem, TableItem

    for item in page_items(doc, page):
        if isinstance(item, TableItem) and not is_empty_table(item):
            return True
        if isinstance(item, PictureItem) and item.meta and item.meta.tabular_chart:
            return True
    return False


def replace_page(doc: Any, page: int, lines: list[str]) -> None:
    """Delete everything Docling found on ``page`` (text, tables, pictures, headers) and add ``lines`` in its place."""
    items = page_items(doc, page)
    refs = {item.self_ref for item in items}

    def inside_another(item: Any) -> bool:  # deleting a parent deletes its children
        parent = item.parent
        while parent is not None:
            if parent.cref in refs:
                return True
            parent = parent.resolve(doc).parent
        return False

    doc.delete_items(node_items=[item for item in items if not inside_another(item)])
    if lines:
        add_lines(doc, page, VISION_PAGE_HEADING.format(page=page), lines)


def is_large_enough(item: Any, minimum: float = MIN_PICTURE_SIZE) -> bool:
    """Judged on the element's own box in page units (not the padded, scaled crop)."""
    if not item.prov:
        return False
    box = item.prov[0].bbox
    return min(abs(box.width), abs(box.height)) >= minimum


def drop_images(doc: Any) -> None:
    """Remove page and picture images Docling embeds as base64 (~80 KB per scanned page)."""
    for page in doc.pages.values():
        page.image = None
    for picture in doc.pictures:
        picture.image = None


class PageImages:
    """Renders pages on demand for cropping: PDFs via pypdfium2 at ``scale``, images as they are. Keeps one page."""

    def __init__(self, path: Path, scale: float = CROP_SCALE) -> None:
        self.path, self.scale = path, scale
        self._pdf: Any = None
        self._cached: tuple[int, Any] | None = None

    def page(self, page_no: int) -> Any:
        if self._cached and self._cached[0] == page_no:
            return self._cached[1]
        from PIL import Image

        if self.path.suffix.lower() in IMAGE_FORMATS:
            image = Image.open(self.path).convert("RGB")
        else:
            import pypdfium2 as pdfium

            if self._pdf is None:
                self._pdf = pdfium.PdfDocument(str(self.path))
            image = self._pdf[page_no - 1].render(scale=self.scale).to_pil().convert("RGB")
        self._cached = (page_no, image)
        return image

    def crop(self, item: Any, doc: Any) -> tuple[Any, int] | None:
        """The region of a Docling item (picture / table), or None if it has no position."""
        if not item.prov:
            return None
        prov = item.prov[0]
        size = doc.pages[prov.page_no].size
        box = prov.bbox.to_top_left_origin(page_height=size.height)
        image = self.page(prov.page_no)
        region = crop_box((box.l, box.t, box.r, box.b), (size.width, size.height), image.size)
        if region[2] <= region[0] or region[3] <= region[1]:
            return None
        return image.crop(region), prov.page_no

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None
        self._cached = None


def render_pages(path: Path, max_pages: int, scale: float = 2.0) -> list[Any]:
    """Whole-page images for form extraction (PDF pages via pypdfium2, or the image itself)."""
    pages = PageImages(path, scale)
    try:
        if path.suffix.lower() in IMAGE_FORMATS:
            return [pages.page(1)]
        import pypdfium2 as pdfium

        count = len(pdfium.PdfDocument(str(path)))
        return [pages.page(n) for n in range(1, min(count, max_pages) + 1)]
    finally:
        pages.close()


@dataclass
class DocumentResult:
    markdown: str
    document: dict[str, Any]  # DoclingDocument.export_to_dict()
    pages: int
    extractions: list[Extraction] = field(default_factory=list)
    pictures: int = 0  # pictures found by Docling
    charts: int = 0
    tables: int = 0
    # tables whose cells Vision supplied (accurate tables, or Docling left them empty)
    tables_from_vision: int = 0
    # lines Docling missed, added back from the PDF's own text layer
    text_layer_lines: int = 0
    # pages without a text layer that Docling couldn't read, read again by Vision (or cleared, if Docling looped)
    pages_from_vision: int = 0
    # tables whose multi-level column header was merged into one row
    merged_headers: int = 0
    # table rows whose label Docling shifted, corrected from the PDF's text layer
    relabeled_rows: int = 0
    seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "tables": self.tables,
            "tables_from_vision": self.tables_from_vision,
            "text_layer_lines": self.text_layer_lines,
            "pages_from_vision": self.pages_from_vision,
            "merged_headers": self.merged_headers,
            "relabeled_rows": self.relabeled_rows,
            "pictures": self.pictures,
            "charts": self.charts,
            "extractions": len(self.extractions),
            "invalid_extractions": sum(not e.valid for e in self.extractions),
            "seconds": self.seconds,
        }


class DocumentIngestor:
    """Docling converter + Granite Vision, loaded once (by the Phase A worker) and reused for every document."""

    def __init__(
        self, vision: VisionModel | None = None, vision_tables: bool = DOCUMENT_VISION_TABLES
    ) -> None:
        self.vision = vision or VisionModel()
        self.vision_tables = vision_tables
        self._converter: Any = None

    def load(self) -> DocumentIngestor:
        if self._converter is None:
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )
            from docling.pipeline.vlm_pipeline import VlmPipeline

            from granit.models.download import local_snapshot

            spec = HUB_MODELS["docling"]
            local_snapshot(spec)  # fail early with a clear message if it isn't downloaded
            options = VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX.model_copy(
                    update={"repo_id": spec.repo_id, "revision": spec.revision}
                )
            )
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=options,
                        backend=PyPdfiumDocumentBackend,
                    ),
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_cls=VlmPipeline, pipeline_options=options
                    ),
                }
            )
            for fmt in (InputFormat.PDF, InputFormat.IMAGE):
                self._converter.initialize_pipeline(fmt)
        self.vision.load()
        return self

    def ingest(
        self, path: str | Path, out_dir: str | Path, vision_tables: bool | None = None
    ) -> DocumentResult:
        """``vision_tables`` overrides the ingestor's default for this document ("Accurate tables" per upload)."""
        path, out_dir = Path(path), Path(out_dir)
        tables_by_vision = self.vision_tables if vision_tables is None else vision_tables
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        check_format(path)
        self.load()
        crops = out_dir / "crops"
        crops.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        doc = self._converter.convert(path).document
        seconds = {"docling": round(time.perf_counter() - start, 2)}

        pages = PageImages(path)
        tables_from_vision = merged_headers = 0
        try:
            start = time.perf_counter()
            extractions, charts = self._charts(doc, pages, crops, out_dir)
            seconds["charts"] = round(time.perf_counter() - start, 2)
            # Accurate tables: Vision reads every table. Otherwise only the ones Docling found but left empty, which
            # would vanish from search (M7: Granite-Docling emits an empty <otsl> for dense and long tables).
            # Docling's own tables, before Vision fills any: kept so the eval can score each path (§4.9).
            own = [[[c.text for c in row] for row in t.data.grid] for t in doc.tables]
            (out_dir / "docling_tables.json").write_text(json.dumps(own, ensure_ascii=False))
            tables = [t for t in doc.tables if tables_by_vision or is_empty_table(t)]
            if tables:
                start = time.perf_counter()
                found, replaced, merged_headers = self._tables(tables, doc, pages, crops, out_dir)
                extractions += found
                seconds["tables"] = round(time.perf_counter() - start, 2)
                tables_from_vision = replaced
            merged_headers += merge_headers(doc)
        finally:
            pages.close()

        lines_by_page = text_layer_lines(path)
        relabeled_rows = relabel_rows(doc, lines_by_page)
        # after charts and tables, so a page whose few words sit in a table or chart keeps Docling's structure
        start = time.perf_counter()
        pages_from_vision = self._read_pages(doc, path, lines_by_page)
        seconds["pages"] = round(time.perf_counter() - start, 2)

        start = time.perf_counter()
        text_layer = add_text_layer_lines(doc, missing_lines(doc, lines_by_page))
        seconds["text_layer"] = round(time.perf_counter() - start, 3)

        drop_images(
            doc
        )  # crops are saved separately; embedded base64 pages would bloat document.json
        markdown = doc.export_to_markdown()
        document = doc.export_to_dict()
        (out_dir / "document.md").write_text(markdown)
        (out_dir / "document.json").write_text(json.dumps(document, ensure_ascii=False))
        return DocumentResult(
            markdown=markdown,
            document=document,
            pages=len(doc.pages),
            extractions=extractions,
            pictures=len(doc.pictures),
            charts=charts,
            tables=len(doc.tables),
            tables_from_vision=tables_from_vision,
            text_layer_lines=text_layer,
            pages_from_vision=pages_from_vision,
            merged_headers=merged_headers,
            relabeled_rows=relabeled_rows,
            seconds=seconds,
        )

    def _read_pages(self, doc: Any, path: Path, lines_by_page: dict[int, list[str]]) -> int:
        """The Vision page pass; returns the pages replaced. A digital page Docling looped on is only cleared: the
        text-layer safety net then adds back all of its text."""
        from granit.ingest.vision import distinct_words, is_looping

        replaced = 0
        images = PageImages(path, PAGE_SCALE)
        try:
            for page in sorted(doc.pages):
                markdown = docling_page_text(doc, page)
                if page in lines_by_page:
                    if is_looping(markdown):
                        replace_page(doc, page, [])
                        replaced += 1
                    continue
                if not is_looping(markdown) and (
                    has_structure(doc, page) or not needs_page_pass(markdown)
                ):
                    continue
                lines = self.vision.read_page(images.page(page))
                # Vision's reading wins if it found more; a looping Docling page goes either way
                if is_looping(markdown) or distinct_words("\n".join(lines)) > distinct_words(
                    markdown
                ):
                    replace_page(doc, page, lines)
                    replaced += 1
        finally:
            images.close()
        return replaced

    def _charts(
        self, doc: Any, pages: PageImages, crops: Path, out_dir: Path
    ) -> tuple[list[Extraction], int]:
        extractions: list[Extraction] = []
        charts, per_page = 0, {}
        for picture in doc.pictures:
            if not is_large_enough(picture):
                continue
            cropped = pages.crop(picture, doc)
            if cropped is None:
                continue
            image, page = cropped
            per_page[page] = per_page.get(page, 0) + 1
            crop = crops / crop_name(page, "picture", per_page[page])
            image.save(crop)
            if not self.vision.is_chart(image):
                continue
            charts += 1
            extraction = self.vision.chart_to_csv(
                image, page=page, crop=str(crop.relative_to(out_dir))
            )
            extractions.append(extraction)
            if extraction.valid:
                attach_chart(picture, extraction.data, created_by=extraction.model)
        return extractions, charts

    def _tables(
        self, tables: list[Any], doc: Any, pages: PageImages, crops: Path, out_dir: Path
    ) -> tuple[list[Extraction], int, int]:
        """Vision's ``<tables_html>`` for each table; a valid result replaces the table's cells, its multi-level header
        merged. Returns (extractions, tables replaced, headers merged)."""
        from granit.ingest.vision import html_grid, html_header_depth

        extractions, per_page, replaced, merged = [], {}, 0, 0
        for table in tables:
            cropped = pages.crop(table, doc)
            if cropped is None:
                continue
            image, page = cropped
            per_page[page] = per_page.get(page, 0) + 1
            crop = crops / crop_name(page, "table", per_page[page])
            image.save(crop)
            extraction = self.vision.table_to_html(
                image, page=page, crop=str(crop.relative_to(out_dir))
            )
            extractions.append(extraction)
            htmls = (extraction.data or []) if extraction.valid else []
            rows = [row for html in htmls for row in html_grid(html)]
            if rows:
                depth = html_header_depth(htmls[0])  # the header comes first
                fill_table(table, merge_header(rows, depth))
                replaced += 1
                merged += depth > 0
        return extractions, replaced, merged
