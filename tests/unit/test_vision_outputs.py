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
