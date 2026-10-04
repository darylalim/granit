"""Generate the document test fixtures in tests/fixtures/documents/ (M3).

Pages are drawn with Pillow (macOS Helvetica) and saved as images and raster PDFs, like a scan: no third-party documents.
Every expected value is written to ``manifest.json`` from the same data that drew the page. Re-run after changing a
fixture, then commit the output:

    uv run python scripts/make_document_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "documents"
PAGE = (1240, 1754)  # A4 at 150 dpi
FONT = "/System/Library/Fonts/Helvetica.ttc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MARGIN = 110

TABLE = {
    "title": "Shipments by region (units)",
    "header": ["Region", "Q1", "Q2", "Q3"],
    "rows": [
        ["North", "1,240", "1,310", "1,425"],
        ["South", "980", "1,045", "1,102"],
        ["East", "1,515", "1,488", "1,560"],
        ["West", "760", "842", "905"],
    ],
}
CHART_TITLE = "Revenue by quarter (thousand USD)"
CHART_CATEGORIES = ["Q1", "Q2", "Q3", "Q4"]
CHART_VALUES = [120, 145, 98, 160]
INVOICE = {
    "invoice_number": "INV-2026-0042",
    "invoice_date": "2026-09-14",
    "due_date": "2026-10-14",
    "vendor_name": "Granite Supply Co.",
    "customer_name": "Northwind Logistics Ltd.",
    "total_amount": "$4,980.00",
}
INVOICE_SCHEMA = {
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
        "purchase_order": {"type": "string", "description": "The customer's purchase order number"},
    },
    "required": ["invoice_number", "total_amount"],
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Helvetica.ttc: index 0 regular, 1 bold
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", PAGE, "white")
    return image, ImageDraw.Draw(image)


def paragraph(draw: ImageDraw.ImageDraw, y: int, lines: list[str], size: int = 26) -> int:
    for line in lines:
        draw.text((MARGIN, y), line, fill="black", font=font(size))
        y += int(size * 1.5)
    return y + 20


def table(draw: ImageDraw.ImageDraw, y: int) -> int:
    draw.text((MARGIN, y), TABLE["title"], fill="black", font=font(28, bold=True))
    y += 55
    widths = [360, 220, 220, 220]
    rows = [TABLE["header"], *TABLE["rows"]]
    for r, row in enumerate(rows):
        x = MARGIN
        for c, value in enumerate(row):
            draw.rectangle([x, y, x + widths[c], y + 56], outline="black", width=2)
            draw.text((x + 16, y + 13), value, fill="black", font=font(26, bold=r == 0))
            x += widths[c]
        y += 56
    return y + 50


def bar_chart(image: Image.Image, top: int) -> tuple[int, int, int, int]:
    """A labelled bar chart drawn on its own white panel; returns its box."""
    draw = ImageDraw.Draw(image)
    left, right, bottom = MARGIN + 40, PAGE[0] - MARGIN - 40, top + 700
    draw.rectangle([left, top, right, bottom], outline="#888888", width=2)
    draw.text((left + 30, top + 25), CHART_TITLE, fill="black", font=font(30, bold=True))
    plot_left, plot_bottom, plot_top = left + 120, bottom - 90, top + 120
    scale = (plot_bottom - plot_top) / 200
    for tick in range(0, 201, 50):
        ty = plot_bottom - tick * scale
        draw.line([plot_left, ty, right - 40, ty], fill="#dddddd", width=1)
        draw.text((left + 40, ty - 14), str(tick), fill="black", font=font(22))
    bar_w, gap = 130, (right - 40 - plot_left - 4 * 130) // 5
    for i, (label, value) in enumerate(zip(CHART_CATEGORIES, CHART_VALUES, strict=True)):
        bx = plot_left + gap + i * (bar_w + gap)
        draw.rectangle([bx, plot_bottom - value * scale, bx + bar_w, plot_bottom], fill="#4F46E5")
        draw.text(
            (bx + 38, plot_bottom - value * scale - 38),
            str(value),
            fill="black",
            font=font(26, bold=True),
        )
        draw.text((bx + 45, plot_bottom + 18), label, fill="black", font=font(26))
    draw.line([plot_left, plot_bottom, right - 40, plot_bottom], fill="black", width=2)
    return left, top, right, bottom


def logo(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """A small mark (90 px ≈ 43 pt in the PDF): below MIN_PICTURE_SIZE, so it must never be sent to Vision."""
    draw.ellipse([x, y, x + 90, y + 90], fill="#1C1917")
    draw.text((x + 22, y + 22), "G", fill="white", font=font(44, bold=True))


def report() -> dict[str, Any]:
    p1, d1 = page()
    logo(d1, PAGE[0] - MARGIN - 90, 70)
    d1.text((MARGIN, 90), "Quarterly Operations Report", fill="black", font=font(44, bold=True))
    y = paragraph(
        d1,
        190,
        [
            "This report summarizes shipments and revenue for the first three quarters.",
            "Shipments grew in every region except East, which dipped slightly in Q2.",
            "Invoice INV-2026-0042 from Granite Supply Co. was approved by Finance.",
        ],
    )
    table(d1, y + 20)

    p2, d2 = page()
    d2.text((MARGIN, 90), "Revenue", fill="black", font=font(40, bold=True))
    y = paragraph(d2, 170, ["Revenue peaked in Q4 after the new warehouse opened in October."])
    bar_chart(p2, y + 30)
    paragraph(d2, y + 30 + 700 + 60, ["Next review: Tuesday, with Legal and Procurement."])

    p1.save(OUT / "report_page1.png")
    p1.save(OUT / "report.pdf", save_all=True, append_images=[p2], resolution=150)
    return {
        "pages": 2,
        "texts": [
            "Quarterly Operations Report",
            "INV-2026-0042",
            "Revenue peaked in Q4",
            "Next review: Tuesday",
        ],
        "table": {"header": TABLE["header"], "rows": TABLE["rows"]},
        "chart": {"categories": CHART_CATEGORIES, "values": CHART_VALUES},
        "charts": 1,
    }


def invoice() -> dict[str, Any]:
    image, draw = page()
    draw.text((MARGIN, 90), INVOICE["vendor_name"], fill="black", font=font(40, bold=True))
    draw.text((MARGIN, 145), "12 Quarry Lane, Barre, VT 05641", fill="black", font=font(24))
    draw.text((PAGE[0] - MARGIN - 300, 90), "INVOICE", fill="black", font=font(48, bold=True))
    y = 260
    for label, key in (
        ("Invoice number:", "invoice_number"),
        ("Invoice date:", "invoice_date"),
        ("Due date:", "due_date"),
    ):
        draw.text((MARGIN, y), label, fill="black", font=font(28, bold=True))
        draw.text((MARGIN + 290, y), INVOICE[key], fill="black", font=font(28))
        y += 48
    draw.text((MARGIN, y + 40), "Bill to:", fill="black", font=font(28, bold=True))
    draw.text((MARGIN, y + 85), INVOICE["customer_name"], fill="black", font=font(28))
    draw.text((MARGIN, y + 125), "400 Harbor Road, Portland, ME 04101", fill="black", font=font(24))
    y += 230
    items = [
        ("Granite countertop slabs", "6", "$620.00", "$3,720.00"),
        ("Delivery and installation", "1", "$1,260.00", "$1,260.00"),
    ]
    x_cols = [MARGIN, MARGIN + 520, MARGIN + 640, MARGIN + 840]
    for x, head in zip(x_cols, ("Description", "Qty", "Unit price", "Amount"), strict=True):
        draw.text((x, y), head, fill="black", font=font(26, bold=True))
    draw.line([MARGIN, y + 40, PAGE[0] - MARGIN, y + 40], fill="black", width=2)
    y += 60
    for row in items:
        for x, value in zip(x_cols, row, strict=True):
            draw.text((x, y), value, fill="black", font=font(26))
        y += 46
    draw.line([MARGIN + 600, y + 10, PAGE[0] - MARGIN, y + 10], fill="black", width=2)
    draw.text((MARGIN + 640, y + 30), "Total due:", fill="black", font=font(30, bold=True))
    draw.text(
        (MARGIN + 840, y + 30), INVOICE["total_amount"], fill="black", font=font(30, bold=True)
    )
    image.save(OUT / "invoice.png")
    image.save(OUT / "invoice.pdf", resolution=150)
    (OUT / "invoice_schema.json").write_text(json.dumps(INVOICE_SCHEMA, indent=2) + "\n")
    # purchase_order isn't on the page: the model must return null and it's reported as missing (not an error).
    return {"expected": INVOICE, "absent": ["purchase_order"]}


MEMO_HTML = """<html><body style="font-family: Helvetica; font-size: 12pt; margin: 2cm">
<h1>Procurement Memo</h1>
<p>Granite Supply Co. confirmed delivery of the countertop slabs on Thursday. Invoice INV-2026-0042 totals $4,980.00.</p>
<h2>Open items</h2>
<table border="1" cellpadding="6" style="border-collapse: collapse">
<tr><th>Owner</th><th>Task</th><th>Due</th></tr>
<tr><td>Priya</td><td>Approve the revised invoice</td><td>Friday</td></tr>
<tr><td>Marcus</td><td>Send the contract to Legal</td><td>Wednesday</td></tr>
</table>
</body></html>"""


def memo() -> dict[str, Any] | None:
    """A digital PDF with a real text layer (the report is a scan), printed by headless Chrome."""
    if not Path(CHROME).exists() and not shutil.which("google-chrome"):
        print("  skipped memo.pdf: Google Chrome not installed")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "memo.html"
        html.write_text(MEMO_HTML)
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={OUT / 'memo.pdf'}", html.as_uri()],
            check=True, capture_output=True,
        )  # fmt: skip
    return {
        "texts": ["Procurement Memo", "INV-2026-0042", "$4,980.00"],
        "table": {
            "header": ["Owner", "Task", "Due"],
            "rows": [
                ["Priya", "Approve the revised invoice", "Friday"],
                ["Marcus", "Send the contract to Legal", "Wednesday"],
            ],
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"report.pdf": report(), "invoice.png": invoice()}
    if memo_info := memo():
        manifest["memo.pdf"] = memo_info
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for path in sorted(OUT.iterdir()):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1000:.0f} KB")


if __name__ == "__main__":
    main()
