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
   through ``<tables_html>`` and is stored as an extraction.

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


def is_empty_table(table: Any) -> bool:
    """Granite-Docling sometimes finds a table but transcribes none of it (M7: dense and long tables)."""
    return not table.data.table_cells or not any(c.text.strip() for c in table.data.table_cells)


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
    tables_from_vision: int = (
        0  # tables whose cells Vision supplied (accurate tables, or Docling left them empty)
    )
    seconds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "tables": self.tables,
            "tables_from_vision": self.tables_from_vision,
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
        tables_from_vision = 0
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
                found, replaced = self._tables(tables, doc, pages, crops, out_dir)
                extractions += found
                seconds["tables"] = round(time.perf_counter() - start, 2)
                tables_from_vision = replaced
        finally:
            pages.close()

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
            seconds=seconds,
        )

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
    ) -> tuple[list[Extraction], int]:
        """Vision's ``<tables_html>`` for each table; a valid result replaces the table's cells. Returns (extractions,
        tables replaced)."""
        from granit.ingest.vision import html_grid

        extractions, per_page, replaced = [], {}, 0
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
            rows = (
                [row for html in (extraction.data or []) for row in html_grid(html)]
                if extraction.valid
                else []
            )
            if rows:
                fill_table(table, rows)
                replaced += 1
        return extractions, replaced
