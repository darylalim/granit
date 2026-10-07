"""Generate the public eval set in eval/public/ (PLAN.md §4.9): synthetic, so it can be committed and run by anyone.

    uv run python scripts/make_eval_fixtures.py

Everything comes from the data in this file, so every expected value, page and time range is known exactly:

- **files/invoices, files/forms:** 6 invoices in different layouts (one a skewed, noisy scan) and 4 forms → ``extraction/``.
- **files/reports:** 3 multi-page reports with facts on known pages.
- **files/tables:** 8 hard tables (merged headers, borderless, skewed scan, split across pages, dense numbers, line breaks,
  two controls) → ``tables/`` with every cell, for Docling vs Vision.
- **files/meetings:** 3 meetings spoken by several macOS ``say`` voices with known action items → ``summaries/``,
  ``transcripts/`` (for WER), and the time range of every line.
- **questions.yaml:** 31 questions (document, meeting, cross-source, unanswerable) whose ``gold_refs`` point at the page or
  time range holding the answer, by the file's sha256.

Needs Google Chrome (HTML → PDF) and macOS ``say`` / ``afconvert``. Generated files are regenerated, never hand-edited.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageFilter

from granit.evaluate.dataset import CATEGORIES
from granit.ingest.audio import SAMPLE_RATE, decode
from granit.store.db import sha256_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "public"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
COMPANY = "Northwind Logistics Ltd."
COMPANY_ADDRESS = "400 Harbor Road, Portland, ME 04101"
PAUSE_S = 0.7  # between speakers
REF_PAD_S = (
    1.0  # gold time ranges are padded: chunk boundaries and AAC priming shift times slightly
)

BASE_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 12.5px; color: #111; margin: 48px; }
h1 { font-size: 24px; margin: 0 0 6px; } h2 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; } td, th { padding: 5px 8px; text-align: left; vertical-align: top; }
.ruled td, .ruled th { border: 1px solid #333; } .ruled th { background: #eee; }
.num { text-align: right; } .muted { color: #555; } .page { page-break-before: always; }
"""


def money(value: float) -> str:
    return f"${value:,.2f}"


# ── HTML → PDF, and scans ──


def chrome_pdf(html: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(
            f"<html><head><meta charset='utf-8'><style>{BASE_CSS}</style></head><body>{html}</body></html>"
        )
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={out}", page.as_uri()],
            check=True, capture_output=True,
        )  # fmt: skip


def scanned(pdf: Path, out: Path, angle: float, seed: int, dpi: int = 150) -> None:
    """A PDF as a scan: rasterized, rotated, blurred and speckled (deterministic), saved as an image PDF or PNG."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    rng = np.random.default_rng(seed)
    pages = []
    for page in doc:
        image = page.render(scale=dpi / 72).to_pil().convert("L")
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
        image = image.filter(ImageFilter.GaussianBlur(0.6))
        pixels = np.asarray(image, dtype=np.int16) + rng.normal(
            0, 9, (image.height, image.width)
        ).astype(np.int16)
        pages.append(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))
    doc.close()
    if out.suffix == ".png":  # a photographed slip: cropped to what's printed, with a margin
        image = pages[0]
        box = Image.eval(image, lambda v: 255 if v < 200 else 0).getbbox() or (
            0,
            0,
            image.width,
            image.height,
        )
        margin = 40
        image.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                    min(image.width, box[2] + margin), min(image.height, box[3] + margin))).save(out, optimize=True)  # fmt: skip
    else:
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=dpi)


# ── invoices and forms ──


@dataclass
class Invoice:
    name: str
    vendor: str
    vendor_address: str
    number: str
    date: tuple[str, str]  # (ISO, as printed)
    due: tuple[str, str]
    po: str | None
    lines: list[tuple[str, float, float]]  # description, quantity, unit price
    layout: str = "left"
    scan: bool = False

    @property
    def total(self) -> float:
        return round(sum(q * u for _, q, u in self.lines), 2)


INVOICES = [
    Invoice("inv-01", "Granite Supply Co.", "12 Quarry Lane, Barre, VT 05641", "GS-2026-0117",
            ("2026-08-03", "August 3, 2026"), ("2026-09-02", "September 2, 2026"), "PO-48213",
            [("Granite countertop slabs", 12, 780.0), ("Edge polishing", 12, 95.0), ("Delivery and installation", 1, 2140.0)]),
    Invoice("inv-02", "Bluefin Office Supplies", "88 Market Street, Boston, MA 02110", "BOS-88412",
            ("2026-08-11", "11 Aug 2026"), ("2026-08-25", "25 Aug 2026"), None,
            [("Printer paper, case of 10 reams", 40, 11.96), ("Toner cartridge, black", 8, 105.0)], layout="right"),
    Invoice("inv-03", "Cedar & Pine Catering", "5 Orchard Way, Portland, ME 04102", "CPC-3307",
            ("2026-09-05", "2026-09-05"), ("2026-10-05", "2026-10-05"), "PO-48390",
            [("Lunch buffet, 50 guests", 50, 45.0), ("Coffee service", 1, 525.0), ("Serving staff", 1, 500.0)], layout="compact"),
    Invoice("inv-04", "Harbor Freight Partners", "1 Pier Road, Newark, NJ 07114", "HFP-2026-551",
            ("2026-09-18", "Sep 18, 2026"), ("2026-10-18", "Oct 18, 2026"), "PO-48455",
            [("Container shipping Rotterdam to Portland", 2, 9800.0), ("Customs handling", 2, 1250.0), ("Cargo insurance", 1, 650.0)]),
    Invoice("inv-05", "Summit Electric Services", "77 Ridge Avenue, Burlington, VT 05401", "SES-10492",
            ("2026-09-22", "22 September 2026"), ("2026-10-22", "22 October 2026"), "PO-48471",
            [("Warehouse lighting retrofit", 1, 5800.0), ("Electrician labor (hours)", 12.5, 89.0)], layout="right", scan=True),
    Invoice("inv-06", "Northbeam Software Inc.", "300 Pine Street, Seattle, WA 98101", "NB-INV-7781",
            ("2026-10-01", "October 1, 2026"), ("2026-10-31", "October 31, 2026"), "PO-48502",
            [("Fleet tracking licenses, annual", 150, 110.0), ("Onboarding and training", 1, 1500.0)], layout="compact"),
]  # fmt: skip

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


def invoice_html(inv: Invoice) -> str:
    fields = [
        ("Invoice number", inv.number),
        ("Invoice date", inv.date[1]),
        ("Due date", inv.due[1]),
    ]
    if inv.po:
        fields.append(("Purchase order", inv.po))
    rows = "".join(f"<tr><td><b>{k}:</b></td><td>{v}</td></tr>" for k, v in fields)
    lines = "".join(
        f"<tr><td>{d}</td><td class='num'>{q:g}</td><td class='num'>{money(u)}</td><td class='num'>{money(q * u)}</td></tr>"
        for d, q, u in inv.lines
    )
    items = (
        "<table class='ruled' style='width:100%; margin-top:28px'>"
        "<tr><th>Description</th><th class='num'>Qty</th><th class='num'>Unit price</th><th class='num'>Amount</th></tr>"
        f"{lines}<tr><td colspan='3' class='num'><b>Total due</b></td><td class='num'><b>{money(inv.total)}</b></td></tr></table>"
    )
    vendor = f"<h1>{inv.vendor}</h1><div class='muted'>{inv.vendor_address}</div>"
    bill_to = (
        f"<div style='margin-top:24px'><b>Bill to:</b><br>{COMPANY}<br>{COMPANY_ADDRESS}</div>"
    )
    fields_table = f"<table style='margin-top:24px'>{rows}</table>"
    if inv.layout == "right":
        return (
            f"<table style='width:100%'><tr><td>{vendor}{bill_to}</td>"
            f"<td style='text-align:right'><h1 style='font-size:30px'>INVOICE</h1>{fields_table}</td></tr></table>{items}"
            "<p class='muted' style='margin-top:30px'>Payment by bank transfer. Thank you for your business.</p>"
        )
    if inv.layout == "compact":
        info = " · ".join(f"{k}: {v}" for k, v in fields)
        return f"{vendor}<h2>Invoice</h2><p>{info}</p>{bill_to}{items}"
    return f"{vendor}<h2 style='font-size:26px'>INVOICE</h2>{fields_table}{bill_to}{items}"


@dataclass
class Form:
    name: str
    html: str
    schema: dict[str, Any]
    expected: dict[str, Any]
    png: bool = False


def text_field(description: str, date: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "string", "description": description}
    if date:
        out["format"] = "date"
    return out


def kv(rows: list[tuple[str, str]]) -> str:
    return "<table class='ruled' style='margin-top:16px'>" + "".join(
        f"<tr><th style='width:200px'>{k}</th><td>{v}</td></tr>" for k, v in rows
    ) + "</table>"  # fmt: skip


FORMS = [
    Form(
        "po-01",
        f"<h1>{COMPANY}</h1><div class='muted'>{COMPANY_ADDRESS}</div><h2 style='font-size:22px'>Purchase Order</h2>"
        + kv([("PO number", "PO-48455"), ("Order date", "September 10, 2026"), ("Supplier", "Harbor Freight Partners"),
              ("Requested delivery", "September 30, 2026"), ("Total", "$22,750.00"), ("Approved by", "Priya Raman")])
        + "<p style='margin-top:20px'>Two containers, Rotterdam to Portland, including customs handling and insurance.</p>",
        {"type": "object", "properties": {
            "po_number": text_field("The purchase order number"),
            "order_date": text_field("The date the order was placed", date=True),
            "supplier_name": text_field("The supplier the order is placed with"),
            "delivery_date": text_field("The requested delivery date", date=True),
            "total_amount": text_field("The order total, with currency"),
            "approved_by": text_field("Who approved the order")}},
        {"po_number": "PO-48455", "order_date": "2026-09-10", "supplier_name": "Harbor Freight Partners",
         "delivery_date": "2026-09-30", "total_amount": "$22,750.00", "approved_by": "Priya Raman"},
    ),
    Form(
        "exp-01",
        f"<h1>Expense Report</h1><div class='muted'>{COMPANY}</div>"
        + kv([("Employee", "Marcus Lee"), ("Employee ID", "E-2291"), ("Trip", "Chicago customer visit"),
              ("Travel dates", "14–16 September 2026"), ("Total claimed", "$1,486.20"), ("Manager approval", "Elena Ruiz")])
        + "<h2>Items</h2><table class='ruled'><tr><th>Item</th><th class='num'>Amount</th></tr>"
          "<tr><td>Flight</td><td class='num'>$612.40</td></tr><tr><td>Hotel, 2 nights</td><td class='num'>$438.00</td></tr>"
          "<tr><td>Meals</td><td class='num'>$221.80</td></tr><tr><td>Taxis</td><td class='num'>$214.00</td></tr></table>",
        {"type": "object", "properties": {
            "employee_name": text_field("The employee claiming expenses"),
            "employee_id": text_field("The employee ID"),
            "trip_purpose": text_field("What the trip was for"),
            "total_claimed": text_field("The total amount claimed, with currency"),
            "approved_by": text_field("The manager who approved the report")}},
        {"employee_name": "Marcus Lee", "employee_id": "E-2291", "trip_purpose": "Chicago customer visit",
         "total_claimed": "$1,486.20", "approved_by": "Elena Ruiz"},
    ),
    Form(
        "dn-01",
        "<h1>Harbor Freight Partners</h1><div class='muted'>1 Pier Road, Newark, NJ 07114</div><h2 style='font-size:22px'>Delivery Note</h2>"
        + kv([("Delivery note", "DN-55120"), ("Ship date", "24 Sep 2026"), ("Consignee", COMPANY),
              ("Tracking number", "HFP-TRK-9917342"), ("Packages", "14"), ("Received by", "Sam Okafor")]),
        {"type": "object", "properties": {
            "delivery_note_number": text_field("The delivery note number"),
            "ship_date": text_field("The date the goods were shipped", date=True),
            "tracking_number": text_field("The carrier's tracking number"),
            "packages": {"type": "integer", "description": "How many packages were delivered"},
            "received_by": text_field("Who signed for the delivery")}},
        {"delivery_note_number": "DN-55120", "ship_date": "2026-09-24", "tracking_number": "HFP-TRK-9917342",
         "packages": 14, "received_by": "Sam Okafor"},
    ),
    Form(
        "rcpt-01",
        "<div style='width:340px; font-family: Courier, monospace'><h1 style='font-size:20px'>CORNER BISTRO</h1>"
        "<div>210 Wabash Ave, Chicago, IL</div><div>15/09/2026 19:42</div><hr>"
        "<table style='width:100%'><tr><td>2 x Pasta</td><td class='num'>$38.00</td></tr>"
        "<tr><td>1 x Salad</td><td class='num'>$12.50</td></tr><tr><td>2 x Iced tea</td><td class='num'>$9.00</td></tr>"
        "<tr><td>Tax</td><td class='num'>$5.10</td></tr><tr><td>Tip</td><td class='num'>$20.00</td></tr>"
        "<tr><td><b>TOTAL</b></td><td class='num'><b>$84.60</b></td></tr></table><hr><div>VISA **** 4421</div></div>",
        {"type": "object", "properties": {
            "merchant": text_field("The business that issued the receipt"),
            "total": text_field("The total paid, with currency"),
            "card_last_digits": text_field("The last four digits of the card used"),
            "tip": text_field("The tip amount, with currency")}},
        {"merchant": "Corner Bistro", "total": "$84.60", "card_last_digits": "4421", "tip": "$20.00"},
        png=True,
    ),
]  # fmt: skip


# ── reports (facts on known pages) ──

REPORTS = {
    "ops-review-q3": [
        "<h1>Q3 Operations Review</h1><div class='muted'>Northwind Logistics · prepared by Sam Okafor</div>"
        "<h2>Summary</h2><p>The on-time delivery rate was 94.2% in Q3, up from 91.5% in Q2. Damage claims fell to 0.8% of "
        "shipments. The Tacoma warehouse reached 87% of its capacity in September, the highest level this year.</p>"
        "<p>Fuel costs per shipment dropped after the switch to optimized routes in July.</p>",
        "<h2>Shipments by month</h2><table class='ruled'><tr><th>Month</th><th class='num'>Shipments</th>"
        "<th class='num'>On time</th><th class='num'>Damage claims</th></tr>"
        "<tr><td>July</td><td class='num'>3,120</td><td class='num'>93.1%</td><td class='num'>29</td></tr>"
        "<tr><td>August</td><td class='num'>3,410</td><td class='num'>94.0%</td><td class='num'>27</td></tr>"
        "<tr><td>September</td><td class='num'>3,655</td><td class='num'>95.3%</td><td class='num'>25</td></tr></table>",
        "<h2>Risks for Q4</h2><p>Peak season staffing: we need 25 seasonal workers in place by November 15. Without them, "
        "the on-time rate is expected to fall below 90% in December.</p><p>Overflow storage: Tacoma will run out of space "
        "in November unless additional storage is leased.</p>",
    ],
    "travel-policy": [
        "<h1>Travel and Expense Policy</h1><div class='muted'>Northwind Logistics · effective July 1, 2026</div>"
        "<h2>Meals</h2><p>The meal per diem is $75 per day for domestic travel and $95 per day for international travel. "
        "Alcohol is not reimbursed.</p><h2>Hotels</h2><p>Hotels are reimbursed up to $220 per night in tier-1 cities and "
        "$160 per night elsewhere.</p>",
        "<h2>Flights</h2><p>Book economy class for flights under 6 hours; premium economy is allowed for longer flights.</p>"
        "<h2>Submitting expenses</h2><p>Expense reports must be submitted within 30 days of the end of the trip, with "
        "itemized receipts for every expense over $25. Reports are approved by the employee's manager.</p>",
    ],
    "atlas-status": [
        "<h1>Project Atlas: Status Update</h1><div class='muted'>Warehouse management system migration · October 2026</div>"
        "<h2>Schedule</h2><p>Go-live has moved from October 20 to November 3, 2026, because the data migration from the "
        "old system took longer than planned.</p><h2>Budget</h2><p>The approved budget is $410,000. To date, $268,500 has "
        "been spent.</p>",
        "<h2>Open risks</h2><table class='ruled'><tr><th>Risk</th><th>Owner</th><th>Status</th></tr>"
        "<tr><td>Barcode scanner firmware update</td><td>Sam Okafor</td><td>In progress</td></tr>"
        "<tr><td>Customer notification of the new date</td><td>Marcus Lee</td><td>Not started</td></tr>"
        "<tr><td>Training for night shift</td><td>Elena Ruiz</td><td>Scheduled</td></tr></table>",
    ],
}


# ── hard tables (Docling vs Vision) ──


def ruled(grid: list[list[str]], header_rows: int = 1, cls: str = "ruled", style: str = "") -> str:
    out = []
    for i, row in enumerate(grid):
        tag = "th" if i < header_rows else "td"
        out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
    return f"<table class='{cls}' style='{style}'>{''.join(out)}</table>"


@dataclass
class HardTable:
    name: str
    type: str
    title: str
    html: str
    grid: list[list[str]]
    scan: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def hard_tables() -> list[HardTable]:
    control_a = [
        ["Carrier", "Shipments", "On time", "Claims"],
        ["Harbor Freight Partners", "1,204", "96.1%", "7"],
        ["Bay Line Trucking", "882", "93.4%", "11"],
        ["Atlas Rail", "415", "90.2%", "3"],
        ["Coastal Air Cargo", "236", "98.7%", "1"],
    ]
    control_b = [
        ["Item", "SKU", "On hand", "Reorder at", "Supplier"],
        ["Pallet wrap", "PW-100", "340", "120", "Bluefin"],
        ["Box, large", "BX-L", "1,250", "400", "Bluefin"],
        ["Box, small", "BX-S", "2,980", "800", "Bluefin"],
        ["Label rolls", "LB-4", "96", "40", "Pacific Print"],
        ["Tape", "TP-2", "512", "150", "Bluefin"],
    ]
    merged_html = (
        "<table class='ruled'><tr><th rowspan='2'>Region</th><th colspan='2'>2025</th><th colspan='2'>2026</th></tr>"
        "<tr><th>H1</th><th>H2</th><th>H1</th><th>H2</th></tr>"
        "<tr><td>North</td><td>4,210</td><td>4,580</td><td>4,890</td><td>5,120</td></tr>"
        "<tr><td>South</td><td>3,050</td><td>3,310</td><td>3,470</td><td>3,600</td></tr>"
        "<tr><td>West</td><td>2,640</td><td>2,900</td><td>3,150</td><td>3,385</td></tr></table>"
    )
    merged = [
        ["Region", "2025", "2025", "2026", "2026"],
        ["Region", "H1", "H2", "H1", "H2"],
        ["North", "4,210", "4,580", "4,890", "5,120"],
        ["South", "3,050", "3,310", "3,470", "3,600"],
        ["West", "2,640", "2,900", "3,150", "3,385"],
    ]
    borderless = [
        ["Depot", "Manager", "Trucks", "Opened"],
        ["Tacoma", "Sam Okafor", "42", "2015"],
        ["Portland", "Elena Ruiz", "35", "2011"],
        ["Boise", "Marcus Lee", "18", "2019"],
        ["Reno", "Priya Raman", "12", "2022"],
        ["Fresno", "Dana Kim", "21", "2017"],
    ]
    scan = [
        ["Week", "Inbound", "Outbound", "Returns"],
        ["36", "1,840", "1,792", "64"],
        ["37", "1,905", "1,866", "58"],
        ["38", "1,977", "1,931", "71"],
        ["39", "2,041", "2,010", "66"],
        ["40", "2,118", "2,075", "69"],
    ]
    split = [["Order", "Customer", "Pallets", "Status"]] + [
        [
            f"SO-{70100 + i}",
            ["Acme Foods", "Brightline Retail", "Cobalt Tools", "Delta Pharma"][i % 4],
            str(4 + (i * 7) % 19),
            ["Shipped", "Packed", "Picking", "Delivered"][(i * 3) % 4],
        ]
        for i in range(56)
    ]
    warehouses = ["Tacoma", "Portland", "Boise", "Reno", "Fresno", "Spokane"]
    dense = [["Week", *warehouses, "Total"]]
    for w in range(1, 17):
        values = [400 + 13 * w + 37 * i + (w * i * 7) % 23 for i in range(len(warehouses))]
        dense.append([str(w), *[f"{v:,}" for v in values], f"{sum(values):,}"])
    breaks = [
        ["Site", "Address", "Contact"],
        ["Tacoma", "1800 Port Way<br>Tacoma, WA 98421", "Sam Okafor<br>+1 253 555 0110"],
        ["Portland", "400 Harbor Road<br>Portland, ME 04101", "Elena Ruiz<br>+1 207 555 0142"],
        ["Boise", "75 Airport Blvd<br>Boise, ID 83705", "Marcus Lee<br>+1 208 555 0177"],
    ]
    breaks_truth = [[c.replace("<br>", " ") for c in r] for r in breaks]
    return [
        HardTable("control-a", "control", "Carrier performance, September", ruled(control_a), control_a),
        HardTable("control-b", "control", "Packaging inventory", ruled(control_b), control_b),
        HardTable("merged-headers", "merged headers", "Shipments by region and half-year", merged_html, merged),
        HardTable("borderless", "borderless", "Depots",
                  ruled(borderless, cls="plain", style="width:70%; border-spacing: 0 6px"), borderless),
        HardTable("scan-skewed", "skewed scan", "Dock activity by week", ruled(scan), scan, scan=True),
        HardTable("split-pages", "split across pages", "Open sales orders", ruled(split), split,
                  extra={"intro": "<p>" + "Orders open at the end of September, by order number. " * 18 + "</p>"}),
        HardTable("dense-numeric", "dense numeric", "Weekly throughput by warehouse (pallets)", ruled(dense), dense),
        HardTable("line-breaks", "line breaks in cells", "Site contacts", ruled(breaks), breaks_truth),
    ]  # fmt: skip


# ── meetings (spoken by `say`, every line's time recorded) ──

VOICES = {"Elena": "Samantha", "Priya": "Karen", "Sam": "Daniel", "Marcus": "Rishi"}


# What a user would list on the Ingest page ("Names and terms in recordings"): the people and the names that come up.
VOCABULARY = ["Elena", "Priya", "Sam", "Marcus", "Northbeam", "Tacoma", "Atlas"]

MEETINGS: dict[str, dict[str, Any]] = {
    "vendor-review": {
        "lines": [
            ("Elena", "Thanks for joining. Today we decide on the Northbeam fleet tracking contract."),
            ("Priya", "The current contract ends on October thirty first. Renewal is sixteen thousand five hundred dollars a year for one hundred fifty licenses."),
            ("Sam", "Drivers rely on it every day, and the new route reports saved us about nine percent on fuel."),
            ("Elena", "Then let's renew with Northbeam for two years."),
            ("Elena", "Priya, please send the signed renewal to procurement by Thursday."),
            ("Priya", "Will do."),
            ("Elena", "Sam, can you schedule the security review with Northbeam before the end of the month?"),
            ("Sam", "Yes, I'll set that up."),
        ],
        "action_items": [
            {"owner": "Priya", "task": "send the signed renewal to procurement", "due": "Thursday"},
            {"owner": "Sam", "task": "schedule the security review with Northbeam", "due": "end of the month"},
        ],
        "decisions": ["Renew the Northbeam fleet tracking contract for two years"],
    },
    "peak-season": {
        "lines": [
            ("Sam", "September volumes were the highest this year, with three thousand six hundred fifty five shipments."),
            ("Elena", "We need the twenty five seasonal workers in place by November fifteenth. Marcus, can you start the hiring?"),
            ("Marcus", "Sure, I'll post the seasonal job ads on Monday."),
            ("Sam", "The Tacoma warehouse is at eighty seven percent capacity, so we should lease overflow space."),
            ("Elena", "Agreed. Sam, please get three quotes for overflow storage by next Friday."),
            ("Sam", "Okay."),
        ],
        "action_items": [
            {"owner": "Marcus", "task": "post the seasonal job ads", "due": "Monday"},
            {"owner": "Sam", "task": "get three quotes for overflow storage", "due": "next Friday"},
        ],
        "decisions": ["Lease overflow storage space for Tacoma"],
    },
    "atlas-checkin": {
        "lines": [
            ("Elena", "Where are we on Project Atlas?"),
            ("Sam", "The data migration slipped, so go-live moves to November third."),
            ("Priya", "We have spent two hundred sixty eight thousand five hundred dollars of the four hundred ten thousand dollar budget."),
            ("Elena", "Sam, please update the barcode scanner firmware before go-live."),
            ("Elena", "Marcus, tell the key customers about the new date this week."),
            ("Marcus", "Sure, I'll email them tomorrow."),
        ],
        "action_items": [
            {"owner": "Sam", "task": "update the barcode scanner firmware", "due": "before go-live"},
            {"owner": "Marcus", "task": "tell the key customers about the new go-live date", "due": "this week"},
        ],
        "decisions": ["Go-live moves to November 3"],
    },
}  # fmt: skip


def say_line(text: str, voice: str, tmp: Path, name: str) -> np.ndarray:
    aiff = tmp / f"{name}.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    audio = decode(aiff)
    loud = np.flatnonzero(np.abs(audio) > 1e-3)
    return audio[loud[0] : loud[-1] + 1] if loud.size else audio


def make_meeting(
    name: str, spec: dict[str, Any], out: Path, tmp: Path
) -> list[tuple[float, float]]:
    """Speak every line with its speaker's voice, PAUSE_S apart; write an AAC .m4a; return each line's (start, end)."""
    pause = np.zeros(int(PAUSE_S * SAMPLE_RATE), dtype=np.float32)
    parts, times, t = [pause], [], PAUSE_S
    for i, (speaker, text) in enumerate(spec["lines"]):
        audio = say_line(text, VOICES[speaker], tmp, f"{name}-{i}")
        times.append((round(t, 2), round(t + len(audio) / SAMPLE_RATE, 2)))
        parts += [audio, pause]
        t += len(audio) / SAMPLE_RATE + PAUSE_S
    wav = tmp / f"{name}.wav"
    import wave

    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000", str(wav), str(out)], check=True
    )
    return times


# ── questions (refs as file + page, or meeting + line numbers; turned into sha256 + range at the end) ──


def q(
    qid: str, question: str, category: str, facts: list[str], *refs: tuple[str, Any]
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "category": category,
        "expected_facts": facts,
        "refs": list(refs),
    }


QUESTIONS = [
    q("doc-01", "What is the total on Granite Supply invoice GS-2026-0117?", "document", ["$12,640.00"], ("invoices/inv-01.pdf", 1)),
    q("doc-02", "When is Bluefin Office Supplies invoice BOS-88412 due?", "document", ["2026-08-25"], ("invoices/inv-02.pdf", 1)),
    q("doc-03", "Which purchase order does the Cedar & Pine Catering invoice reference?", "document", ["PO-48390"], ("invoices/inv-03.pdf", 1)),
    q("doc-04", "How much did Harbor Freight Partners charge for customs handling?", "document", ["$2,500.00"], ("invoices/inv-04.pdf", 1)),
    q("doc-05", "What is the total on the Summit Electric Services invoice?", "document", ["$6,912.50"], ("invoices/inv-05.pdf", 1)),
    q("doc-06", "How many fleet tracking licenses are on the Northbeam Software invoice?", "document", ["150"], ("invoices/inv-06.pdf", 1)),
    q("doc-07", "What was the on-time delivery rate in Q3?", "document", ["94.2%"], ("reports/ops-review-q3.pdf", 1)),
    q("doc-08", "How many shipments did Northwind make in August?", "document", ["3,410"], ("reports/ops-review-q3.pdf", 2)),
    q("doc-09", "What is the meal per diem for international travel?", "document", ["$95"], ("reports/travel-policy.pdf", 1)),
    q("doc-10", "Within how many days must expense reports be submitted?", "document", ["30 days"], ("reports/travel-policy.pdf", 2)),
    q("doc-11", "What is the approved budget for Project Atlas?", "document", ["$410,000"], ("reports/atlas-status.pdf", 1)),
    q("doc-12", "Who approved purchase order PO-48455?", "document", ["Priya Raman"], ("forms/po-01.pdf", 1)),
    q("doc-13", "What is the tracking number on delivery note DN-55120?", "document", ["HFP-TRK-9917342"], ("forms/dn-01.pdf", 1)),
    q("doc-14", "How much did Marcus Lee claim for his Chicago customer visit?", "document", ["$1,486.20"], ("forms/exp-01.pdf", 1)),
    q("doc-15", "In the weekly throughput table, how many pallets did Boise handle in week 7?", "document", ["DENSE_BOISE_W7"], ("tables/dense-numeric.pdf", 1)),
    q("mtg-01", "Who will send the signed Northbeam renewal to procurement, and by when?", "meeting", ["Priya", "Thursday"], ("vendor-review", [4, 5])),
    q("mtg-02", "What did the team decide about the fleet tracking contract?", "meeting", ["two years | 2 years | 2-year | two-year"], ("vendor-review", [3])),
    q("mtg-03", "How much did the new route reports save on fuel?", "meeting", ["9% | nine percent | 9 percent"], ("vendor-review", [2])),
    q("mtg-04", "Who will post the seasonal job ads, and when?", "meeting", ["Marcus", "Monday"], ("peak-season", [1, 2])),
    q("mtg-05", "How full is the Tacoma warehouse, according to the peak season meeting?", "meeting", ["87% | eighty seven percent | eighty-seven percent | 87 percent"], ("peak-season", [3])),
    q("mtg-06", "What is the new go-live date for Project Atlas mentioned in the check-in?", "meeting", ["November 3 | November third | 2026-11-03 | Nov 3"], ("atlas-checkin", [1])),
    q("mtg-07", "Who is responsible for the barcode scanner firmware update?", "meeting", ["Sam"], ("atlas-checkin", [3])),
    q("x-01", "How much does the Northbeam renewal cost per year, and what was the total on Northbeam's invoice?", "cross-source",
      ["16,500 | sixteen thousand five hundred", "$18,000.00"], ("vendor-review", [1]), ("invoices/inv-06.pdf", 1)),
    q("x-02", "Do the Atlas status report and the check-in meeting agree on the go-live date?", "cross-source",
      ["November 3 | November third | 2026-11-03 | Nov 3"], ("reports/atlas-status.pdf", 1), ("atlas-checkin", [1])),
    q("x-03", "How many seasonal workers are needed, and by when?", "cross-source",
      ["25 | twenty five | twenty-five", "November 15 | November fifteenth | 2026-11-15 | Nov 15"],
      ("reports/ops-review-q3.pdf", 3), ("peak-season", [1])),
    q("x-04", "Which carrier delivered under delivery note DN-55120, and what was its invoice total?", "cross-source",
      ["Harbor Freight Partners", "$22,750.00"], ("forms/dn-01.pdf", 1), ("invoices/inv-04.pdf", 1)),
    q("none-01", "What is the warranty period on the warehouse lighting retrofit?", "unanswerable", []),
    q("none-02", "Who won the employee of the month award in September?", "unanswerable", []),
    q("none-03", "What is Northwind's parental leave policy?", "unanswerable", []),
    q("none-04", "What was Northwind's net profit in Q3?", "unanswerable", []),
    q("none-05", "How many parking spaces does the Tacoma warehouse have?", "unanswerable", []),
]  # fmt: skip


# ── build ──


def main() -> None:
    if not Path(CHROME).exists():
        raise SystemExit("Google Chrome is needed to print the PDFs")
    for sub in ("files", "extraction", "summaries", "transcripts", "tables"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    files = OUT / "files"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for inv in INVOICES:
            target = files / "invoices" / f"{inv.name}.pdf"
            if inv.scan:
                chrome_pdf(invoice_html(inv), tmp / f"{inv.name}.pdf")
                scanned(tmp / f"{inv.name}.pdf", target, angle=1.4, seed=5)
            else:
                chrome_pdf(invoice_html(inv), target)
            expected = {
                "invoice_number": inv.number, "invoice_date": inv.date[0], "due_date": inv.due[0],
                "vendor_name": inv.vendor, "customer_name": COMPANY, "total_amount": money(inv.total),
                "purchase_order": inv.po,
            }  # fmt: skip
            write_json(OUT / "extraction" / f"{inv.name}.json",
                       {"file": f"files/invoices/{inv.name}.pdf", "schema": INVOICE_SCHEMA, "expected": expected})  # fmt: skip

        for form in FORMS:
            ext = ".png" if form.png else ".pdf"
            target = files / "forms" / f"{form.name}{ext}"
            if form.png:
                chrome_pdf(form.html, tmp / f"{form.name}.pdf")
                target.parent.mkdir(parents=True, exist_ok=True)
                scanned(tmp / f"{form.name}.pdf", target, angle=-0.8, seed=11)
            else:
                chrome_pdf(form.html, target)
            write_json(OUT / "extraction" / f"{form.name}.json",
                       {"file": f"files/forms/{form.name}{ext}", "schema": form.schema, "expected": form.expected})  # fmt: skip

        for name, pages in REPORTS.items():
            html = "".join(
                p if i == 0 else f"<div class='page'>{p}</div>" for i, p in enumerate(pages)
            )
            chrome_pdf(html, files / "reports" / f"{name}.pdf")

        dense_boise_w7 = ""
        for table in hard_tables():
            target = files / "tables" / f"{table.name}.pdf"
            html = f"<h2>{table.title}</h2>{table.extra.get('intro', '')}{table.html}"
            if table.scan:
                chrome_pdf(html, tmp / f"{table.name}.pdf")
                scanned(tmp / f"{table.name}.pdf", target, angle=2.2, seed=23)
            else:
                chrome_pdf(html, target)
            write_json(OUT / "tables" / f"{table.name}.json",
                       {"file": f"files/tables/{table.name}.pdf", "type": table.type, "grid": table.grid})  # fmt: skip
            if table.name == "dense-numeric":
                dense_boise_w7 = table.grid[7][3]  # row "7", column "Boise"

        times: dict[str, list[tuple[float, float]]] = {}
        for name, spec in MEETINGS.items():
            times[name] = make_meeting(name, spec, files / "meetings" / f"{name}.m4a", tmp)
            text = " ".join(line for _, line in spec["lines"])
            (OUT / "transcripts").mkdir(parents=True, exist_ok=True)
            (OUT / "transcripts" / f"{name}.txt").write_text(
                f"file: files/meetings/{name}.m4a\n{text}\n"
            )
            (OUT / "summaries").mkdir(parents=True, exist_ok=True)
            summary = {"file": f"files/meetings/{name}.m4a", "action_items": spec["action_items"],
                       "decisions": spec["decisions"]}  # fmt: skip
            (OUT / "summaries" / f"{name}.yaml").write_text(
                yaml.safe_dump(summary, sort_keys=False, width=120)
            )
        (OUT / "vocabulary.txt").write_text("\n".join(VOCABULARY) + "\n")

    questions = []
    for item in QUESTIONS:
        assert item["category"] in CATEGORIES
        facts = [dense_boise_w7 if f == "DENSE_BOISE_W7" else f for f in item["expected_facts"]]
        refs = []
        for where, at in item["refs"]:
            if where in MEETINGS:
                path = files / "meetings" / f"{where}.m4a"
                start = min(times[where][i][0] for i in at)
                end = max(times[where][i][1] for i in at)
                refs.append({"source_sha256": sha256_of(path), "file": f"files/meetings/{where}.m4a",
                             "start_s": round(max(0.0, start - REF_PAD_S), 2), "end_s": round(end + REF_PAD_S, 2)})  # fmt: skip
            else:
                path = files / where
                refs.append(
                    {"source_sha256": sha256_of(path), "file": f"files/{where}", "page": at}
                )
        entry: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
        }
        if facts:
            entry["expected_facts"] = facts
        if refs:
            entry["gold_refs"] = refs
        questions.append(entry)
    header = (
        "# Generated by scripts/make_eval_fixtures.py: regenerate, don't edit (PLAN.md §4.9).\n"
    )
    (OUT / "questions.yaml").write_text(
        header + yaml.safe_dump(questions, sort_keys=False, width=120, allow_unicode=True)
    )

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(
        f"{len(questions)} questions, {len(list(files.rglob('*.*')))} files, {total / 1e6:.1f} MB in {OUT.relative_to(ROOT)}"
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
