"""Adaptive layout for 760–3440 px windows (PLAN.md §4.8): the width constants live here and nowhere else.

Python can't see the window size, so pages adapt through **wrapping rows of fixed-width panels**: in
``st.container(horizontal=True, wrap=True)``, fixed-width children sit side by side when there's room and wrap below when
there isn't. The panel widths decide where that switch happens. Changing a width moves a switch point, so re-run
``scripts/ui_screenshots.py`` (760 / 1512 / 2560 px, light and dark) after any change here.

Every container gets a ``key``: Streamlit renders it as the CSS class ``st-key-<key>``, which the screenshot check uses to
measure where panels ended up.
"""

from __future__ import annotations

import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

READING_WIDTH = 720  # ~96 characters per line at 15 px Source Sans
SIDE_PANEL_WIDTH = (
    380  # Ask: sources beside the answer from ~1470 px (13" MacBook Air), below on smaller windows
)
EXTRACT_PANEL_WIDTH = (
    520  # Extract: document and fields side by side from ~1512 px (14" MacBook Pro)
)


def reading_column(key: str) -> DeltaGenerator:
    """One left-aligned column at reading width (Ingest)."""
    return st.container(width=READING_WIDTH, key=key)


def reading_with_side_panel(key: str) -> tuple[DeltaGenerator, DeltaGenerator]:
    """(reading column, side panel): side by side on wide windows, the panel below on narrow ones (Ask)."""
    row = st.container(horizontal=True, wrap=True, gap="large", key=f"{key}_row")
    main = row.container(width=READING_WIDTH, key=f"{key}_main")
    side = row.container(width=SIDE_PANEL_WIDTH, border=True, key=f"{key}_side")
    return main, side


def two_panels(key: str) -> tuple[DeltaGenerator, DeltaGenerator]:
    """Two equal centered panels that stack on narrow windows (Extract: document image │ fields)."""
    row = st.container(
        horizontal=True, wrap=True, gap="large", horizontal_alignment="center", key=f"{key}_row"
    )
    left = row.container(width=EXTRACT_PANEL_WIDTH, key=f"{key}_left")
    right = row.container(width=EXTRACT_PANEL_WIDTH, key=f"{key}_right")
    return left, right


ROW_PX = 33  # a dataframe row at baseFontSize 15 (measured in the M6 screenshots; Streamlit's default 35 is at 16 px)


def table_height(rows: int, max_rows: int = 10) -> int:
    """A dataframe height that fits ``rows`` exactly (no empty row below), scrolling past ``max_rows``."""
    return (min(rows, max_rows) + 1) * ROW_PX + 3


def safe_md(text: str) -> str:
    """Model output and document text are prose: a dollar sign is currency, never LaTeX.

    Found in the M6 screenshots: two amounts in one paragraph (``$4,980 … $1,200``) rendered the text between them as math.
    """
    return text.replace("$", r"\$")


CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def answer_md(text: str) -> str:
    """An answer for ``st.markdown``: ``safe_md``, then citations like ``[2]`` or ``[1, 3]`` as small gray badges."""
    return CITATION.sub(lambda m: f":gray-badge[{m.group(1)}]", safe_md(text))
