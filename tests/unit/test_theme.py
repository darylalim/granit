"""The "granite" theme stays readable and private (PLAN.md §4.7): a tweak that breaks contrast or the light/dark pairing fails CI.

Contrast is WCAG 2.x: 4.5:1 for text, 3:1 for UI parts and chart marks. 32 pairs per mode: text on page, panel and sidebar;
white on the primary color; primary vs page and sidebar; links; code; dataframe headers; each status color's text on its tint
and the color on the page; each categorical chart color on the page.
"""

from __future__ import annotations

import colorsys
import tomllib
from typing import Any

import pytest

from tests.conftest import ROOT

CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
MODES = ("light", "dark")
STATUS = ("green", "yellow", "orange", "red", "blue", "violet", "gray")
TEXT, UI = 4.5, 3.0


def luminance(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def hue(hex_color: str) -> float:
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def pairs(mode: str) -> list[tuple[str, str, str, float]]:
    """(what, foreground, background, minimum) for one mode."""
    t: dict[str, Any] = CONFIG["theme"][mode]
    side: dict[str, Any] = t["sidebar"]
    page = t["backgroundColor"]
    out = [
        ("text on page", t["textColor"], page, TEXT),
        ("text on panel", t["textColor"], t["secondaryBackgroundColor"], TEXT),
        ("text on sidebar", t["textColor"], side["backgroundColor"], TEXT),
        ("text on sidebar panel", t["textColor"], side["secondaryBackgroundColor"], TEXT),
        ("white on primary", "#FFFFFF", t["primaryColor"], TEXT),
        ("primary vs page", t["primaryColor"], page, UI),
        ("primary vs sidebar", t["primaryColor"], side["backgroundColor"], UI),
        ("link on page", t["linkColor"], page, TEXT),
        ("code", t["codeTextColor"], t["codeBackgroundColor"], TEXT),
        (
            "dataframe header",
            t["dataframeHeaderTextColor"],
            t["dataframeHeaderBackgroundColor"],
            TEXT,
        ),
    ]
    for s in STATUS:
        out.append((f"{s} text on tint", t[f"{s}TextColor"], t[f"{s}BackgroundColor"], TEXT))
        out.append((f"{s} vs page", t[f"{s}Color"], page, UI))
    for i, color in enumerate(t["chartCategoricalColors"]):
        out.append((f"chart color {i + 1}", color, page, UI))
    return out


@pytest.mark.parametrize("mode", MODES)
def test_every_pair_meets_wcag_aa(mode: str) -> None:
    checked = pairs(mode)
    assert len(checked) == 32
    failing = [
        (what, round(contrast(fg, bg), 2))
        for what, fg, bg, minimum in checked
        if contrast(fg, bg) < minimum
    ]
    assert failing == []


def test_light_and_dark_are_a_pair() -> None:
    light, dark = CONFIG["theme"]["light"], CONFIG["theme"]["dark"]
    assert set(light) == set(dark)
    assert set(light["sidebar"]) == set(dark["sidebar"])
    assert abs(hue(light["primaryColor"]) - hue(dark["primaryColor"])) <= 2  # one indigo hue
    assert round(hue(light["primaryColor"])) in range(240, 247)
    # Sequential ramps are reversed in dark mode, so low values fade into the page in both.
    assert dark["chartSequentialColors"] == light["chartSequentialColors"][::-1]
    assert len(light["chartCategoricalColors"]) == len(dark["chartCategoricalColors"]) == 8


def test_privacy_and_offline_settings() -> None:
    assert CONFIG["browser"]["gatherUsageStats"] is False
    assert CONFIG["server"]["address"] == "localhost"
    assert CONFIG["server"]["showEmailPrompt"] is False
    assert CONFIG["client"]["toolbarMode"] == "viewer"
    # Bundled fonts only: a Google Fonts URL would make network requests from a "fully local" app.
    assert CONFIG["theme"]["font"] == "sans-serif" and CONFIG["theme"]["codeFont"] == "monospace"
    assert "http" not in (ROOT / ".streamlit" / "config.toml").read_text()
