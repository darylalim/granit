"""Granite Vision output handling without the model (M3): parsing, the KVP prompt, schema validation, dates."""

from __future__ import annotations

import json

import pytest
from jsonschema import SchemaError

from granit.ingest import vision
from granit.ingest.vision import Extraction, VisionOutputError

SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string", "format": "date"},
        "total": {"type": "number"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "shipped": {"type": "string", "format": "date"},
                },
            },
        },
    },
    "required": ["invoice_number", "total"],
}


# ── output parsing ──


def test_parse_csv_strips_the_fence() -> None:
    text = "```csv\nState,2017,2018\nNJ,4.6,4.1\nCT,4.7,4.1\n```"
    assert vision.parse_csv(text) == [
        ["State", "2017", "2018"],
        ["NJ", "4.6", "4.1"],
        ["CT", "4.7", "4.1"],
    ]
    assert vision.parse_csv("A,B\n1,2") == [["A", "B"], ["1", "2"]]  # no fence is fine too


@pytest.mark.parametrize("bad", ["", "just prose", "```csv\nA,B\n```", "A,B\n1,2,3", "A\n1"])
def test_parse_csv_rejects_non_tables(bad: str) -> None:
    with pytest.raises(VisionOutputError):
        vision.parse_csv(bad)


def test_parse_tables_html_handles_the_list_wrapper() -> None:
    text = "[<html><table><tr><td>a</td></tr></table></html>, <html><table><tr><th>b</th></tr></table></html>]"
    assert vision.parse_tables_html(text) == [
        "<table><tr><td>a</td></tr></table>",
        "<table><tr><th>b</th></tr></table>",
    ]
    with pytest.raises(VisionOutputError):
        vision.parse_tables_html("<table></table>")  # no cells
    with pytest.raises(VisionOutputError):
        vision.parse_tables_html("I could not find a table.")


def test_parse_json_object() -> None:
    assert vision.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert vision.parse_json_object('Here you go: {"a": {"b": [1]}} done') == {"a": {"b": [1]}}
    for bad in ("[1, 2]", "{not json}", "nothing"):
        with pytest.raises(VisionOutputError):
            vision.parse_json_object(bad)


# ── KVP prompt and schema ──


def test_kvp_prompt_is_the_model_cards_varex_format() -> None:
    prompt = vision.kvp_prompt({"type": "object", "properties": {"a": {"type": "string"}}})
    assert prompt.startswith(
        "Extract structured data from this document.\nReturn a JSON object matching this schema:\n\n{"
    )
    assert prompt.endswith(
        "Return null for fields you cannot find.\nReturn ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself."
    )


def test_check_schema() -> None:
    vision.check_schema(SCHEMA)
    with pytest.raises(SchemaError):
        vision.check_schema({"type": "object", "properties": {"a": {"type": "nonsense"}}})
    with pytest.raises(ValueError, match="describe an object"):
        vision.check_schema({"type": "string"})


# ── validation ──


def test_nulls_are_missing_not_errors_unless_required() -> None:
    data = {
        "invoice_number": "INV-1",
        "invoice_date": None,
        "total": 12.5,
        "lines": [None, {"sku": None}],
    }
    cleaned, errors, missing = vision.validate(data, SCHEMA)
    assert errors == []
    assert cleaned == {"invoice_number": "INV-1", "total": 12.5, "lines": [{}]}
    assert missing == ["invoice_date", "lines[1].sku"]


def test_missing_required_and_wrong_types_are_errors() -> None:
    _, errors, missing = vision.validate({"invoice_number": None, "total": "twelve"}, SCHEMA)
    assert missing == ["invoice_number"]
    assert any("invoice_number" in e and "required" in e for e in errors)
    assert any(e.startswith("total:") for e in errors)


def test_dates_are_normalized_then_format_checked() -> None:
    data = {
        "invoice_number": "INV-1",
        "total": 1,
        "invoice_date": "14/09/2026",
        "lines": [{"shipped": "Sept. 3 2026"}, {"shipped": "03/04/2026"}],
    }
    cleaned, errors, _ = vision.validate(data, SCHEMA)
    assert cleaned["invoice_date"] == "2026-09-14"
    assert cleaned["lines"][0]["shipped"] == "2026-09-03"
    assert cleaned["lines"][1]["shipped"] == "03/04/2026"  # ambiguous: kept, and flagged
    assert errors == ["lines/1/shipped: '03/04/2026' is not a 'date'"]


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("2026-09-14", "2026-09-14"),
        ("14/09/2026", "2026-09-14"),  # 14 can't be a month: day first
        ("10-14-2026", "2026-10-14"),  # 14 can't be a month: month first
        ("05.05.2026", "2026-05-05"),
        ("17-MAY-2020", "2020-05-17"),
        ("May 17, 2020", "2020-05-17"),
        ("17 May 2020", "2020-05-17"),
        ("03/04/2026", None),  # day-month or month-day? never guessed
        ("31/02/2026", None),
        ("next friday", None),
    ],
)
def test_to_iso_date(text: str, iso: str | None) -> None:
    assert vision.to_iso_date(text) == iso


def test_merge_pages_first_non_null_wins() -> None:
    pages = [{"a": None, "b": "page 1", "c": None}, {"a": "page 2", "b": "page 2", "c": None}]
    assert vision.merge_pages(pages) == {"a": "page 2", "b": "page 1", "c": None}


def test_extraction_json_is_storable() -> None:
    e = Extraction("form", "json", '{"a": 1}', True, data={"a": 1}, missing=("b",), page=1)
    out = json.loads(json.dumps(e.to_json()))
    assert out["kind"] == "form" and out["missing"] == ["b"] and "data" not in out


# ── page reading (M7 private set: receipts and a scan Granite-Docling couldn't read) ──

DOCLING_LOOP = "\n\n".join(["1", "2", "3"] + ["loc>loc>loc>201"] * 30)  # as stored for cord-004
VISION_LOOP = (
    "Here is the text from the image, transcribed line by line:\n\n```\n" + "$0.21\n" * 400
)
RECEIPT = "1 TAHU GORENG 28,000\n1 CAKWE 17,000\n1 PHO TAI CHIN (R) 63,000\n1 TEA 11,000\nSub Total 119,000\nGrand Total 140,063"


@pytest.mark.parametrize("text", [DOCLING_LOOP, VISION_LOOP, " . \n" * 50, "the " * 40])
def test_loops_are_detected(text: str) -> None:
    assert vision.is_looping(text)


@pytest.mark.parametrize(
    "text",
    [RECEIPT, "short", "1 TEA\n" * 3, "| Week | Boise |\n|---|---|\n| 7 | 571 |\n| 8 | 602 |"],
)
def test_real_text_is_not_a_loop(text: str) -> None:
    assert not vision.is_looping(text)


def test_distinct_words_ignore_numbers_and_markup() -> None:
    assert vision.distinct_words("\n".join(str(n) for n in range(200))) == 0
    assert vision.distinct_words(DOCLING_LOOP) == 1  # "loc"
    assert vision.distinct_words("Sub Total 119,000 Grand Total") == 3


def test_clean_page_text_drops_the_preamble_fences_and_tags() -> None:
    fenced = "Here is the text from the image, transcribed line by line:\n\n```\nICE BLACK COFFEE 2 82,000\n\nTOTAL  174,600\n```"
    assert vision.clean_page_text(fenced) == ["ICE BLACK COFFEE 2 82,000", "TOTAL 174,600"]
    assert vision.clean_page_text(
        "<doc> 104-10003-10041 RELEASE \n . \n 7. \n Room 2593 </doc>"
    ) == [
        "104-10003-10041 RELEASE",
        "7.",
        "Room 2593",
    ]


class ScriptedVision(vision.VisionModel):
    """Returns the scripted outputs in order and records each call's options."""

    def __init__(self, *outputs: str) -> None:
        super().__init__()
        self.outputs, self.calls = list(outputs), []

    def generate(self, image, prompt, max_tokens, **options):  # type: ignore[override]
        self.calls.append(options)
        return self.outputs.pop(0)


def test_read_page_retries_a_loop_with_a_repetition_penalty() -> None:
    model = ScriptedVision(VISION_LOOP, RECEIPT)
    assert model.read_page(None)[-1] == "Grand Total 140,063"
    assert model.calls == [{}, {"repetition_penalty": 1.1, "repetition_context_size": 64}]


def test_read_page_gives_nothing_rather_than_a_loop() -> None:
    assert ScriptedVision(VISION_LOOP, VISION_LOOP).read_page(None) == []


def test_read_page_keeps_the_first_good_reading() -> None:
    model = ScriptedVision(RECEIPT)
    assert len(model.read_page(None)) == 6 and model.calls == [{}]


@pytest.mark.parametrize(
    ("html", "depth"),
    [
        ("<table><tr><th>Region</th><th>Q1</th></tr><tr><td>North</td><td>1</td></tr></table>", 0),
        (
            "<table><tr><th rowspan=2>Region</th><th colspan=2>2025</th></tr><tr><th>H1</th><th>H2</th></tr>"
            "<tr><td>North</td><td>1</td><td>2</td></tr></table>",
            2,
        ),
        (
            "<table><tr><td></td><td colspan=2>2022</td></tr><tr><td>State</td><td colspan=2>Number</td></tr>"
            "<tr><td></td><td>Estimate</td><td>Margin</td></tr><tr><td>Ohio</td><td>1</td><td>2</td></tr></table>",
            3,
        ),
        # a spanning totals row at the bottom isn't a header
        (
            "<table><tr><th>Item</th><th>Qty</th><th>Amount</th></tr><tr><td colspan=2>Total</td><td>9</td></tr></table>",
            0,
        ),
        # all header, no body: nothing to merge into
        ("<table><tr><th colspan=2>2025</th></tr><tr><th>H1</th><th>H2</th></tr></table>", 0),
    ],
)
def test_html_header_depth(html: str, depth: int) -> None:
    assert vision.html_header_depth(html) == depth


# ── x-sums: arithmetic checks on extracted amounts ──

SUMS_SCHEMA = {
    "type": "object",
    "properties": {
        k: {"type": "string"}
        for k in ("subtotal", "discount", "service_charge", "tax", "total", "cash_paid", "change")
    },
    "x-sums": [
        {"total": "total", "parts": ["subtotal", "-discount", "service_charge", "tax"]},
        {"total": "change", "parts": ["cash_paid", "-total"]},
    ],
}


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("377,859", 377859.0),
        ("377.859", 377859.0),  # Indonesian grouping
        ("Rp 1.234.567", 1234567.0),
        ("$1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("12.5", 12.5),
        ("-19,400", -19400.0),
        ("n/a", None),
    ],
)
def test_parse_amount(text: str, value: float | None) -> None:
    assert vision.parse_amount(text) == value


def test_sums_flag_a_misread_digit() -> None:
    # CORD receipt cord-004: Vision read the subtotal 194,000 as 174,000.
    read = {
        "subtotal": "174,000",
        "discount": "19,400",
        "total": "174,600",
        "cash_paid": "200,000",
        "change": "25,400",
    }
    assert vision.check_sums(read, SUMS_SCHEMA) == [
        "total: subtotal - discount = 154600, but total is 174,600"
    ]
    assert vision.check_sums({**read, "subtotal": "194,000"}, SUMS_SCHEMA) == []
    _, errors, _ = vision.validate({**read, "tax": None}, SUMS_SCHEMA)
    assert errors == ["total: subtotal - discount = 154600, but total is 174,600"]


def test_sums_skip_rules_without_their_fields() -> None:
    assert (
        vision.check_sums(
            {"total": "46,000", "cash_paid": "50,000", "change": "4,000"}, SUMS_SCHEMA
        )
        == []
    )
    assert vision.check_sums({"subtotal": "10"}, SUMS_SCHEMA) == []  # no total
    assert (
        vision.check_sums({"total": "10", "subtotal": "ten"}, SUMS_SCHEMA) == []
    )  # unparseable: not judged
    assert (
        vision.check_sums({"total": "10"}, {"type": "object", "properties": {}}) == []
    )  # no rules


def test_sums_stay_out_of_the_prompt() -> None:
    prompt = vision.kvp_prompt(SUMS_SCHEMA)
    assert "x-sums" not in prompt and '"subtotal"' in prompt
    vision.check_schema(SUMS_SCHEMA)  # an extra keyword is still a valid JSON Schema
