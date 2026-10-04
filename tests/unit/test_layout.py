"""Layout rules (PLAN.md §4.8): widths live in one place, and nothing reaches Markdown without ``safe_md``.

Two dollar amounts in one paragraph (``$4,980 … $1,200``) render the text between them as LaTeX. The grep-style check
below walks every page's syntax tree: each call that renders Markdown (``markdown``, ``caption``, ``info``, ``badge``, …)
may only interpolate values that went through an escaping helper, so new code can't skip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from granit.ui import layout
from granit.ui.layout import answer_md, safe_md
from tests.conftest import ROOT

PAGES = sorted((ROOT / "app").rglob("*.py"))
RENDERS_MARKDOWN = {
    "markdown",
    "caption",
    "info",
    "success",
    "error",
    "warning",
    "title",
    "header",
    "subheader",
    "badge",
}
ESCAPES = {"safe_md", "answer_md", "document_md", "transcript_md"}
NO_MARKDOWN = {"plural", "details", "len", "int"}  # numbers and fixed words only


def test_safe_md_escapes_every_dollar() -> None:
    text = "Total $4,980.00, deposit $1,200 and $$ (no math)"
    assert safe_md(text) == r"Total \$4,980.00, deposit \$1,200 and \$\$ (no math)"
    assert "$" not in safe_md(text).replace(r"\$", "")


def test_answers_escape_dollars_and_show_citations_as_badges() -> None:
    assert answer_md("Total $4,980 [2]; deposit $1,200 [1, 3].") == (
        r"Total \$4,980 :gray-badge[2]; deposit \$1,200 :gray-badge[1, 3]."
    )
    assert answer_md("See [2–4].") == "See :gray-badge[2–4]."
    assert answer_md("A [link](x) stays.") == "A [link](x) stays."


def test_widths_match_the_plan() -> None:
    assert (layout.READING_WIDTH, layout.SIDE_PANEL_WIDTH, layout.EXTRACT_PANEL_WIDTH) == (
        720,
        380,
        520,
    )


def is_safe(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(is_safe(v) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return node.format_spec is not None or is_safe(node.value)  # {x:.1f} is a number
    if isinstance(node, ast.BinOp):
        return is_safe(node.left) and is_safe(node.right)
    if isinstance(node, ast.IfExp):
        return is_safe(node.body) and is_safe(node.orelse)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in ESCAPES | NO_MARKDOWN
        if (
            isinstance(func, ast.Attribute) and func.attr == "join"
        ):  # "\n".join(f"- {safe_md(d)}" for d in …)
            arg = node.args[0] if node.args else None
            return (
                isinstance(func.value, ast.Constant)
                and isinstance(arg, (ast.GeneratorExp, ast.ListComp))
                and is_safe(arg.elt)
            )
    return False


def unescaped_calls(path: Path) -> list[str]:
    problems = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in RENDERS_MARKDOWN
            and node.args
            and not is_safe(node.args[0])
        ):
            problems.append(f"{path.name}:{node.lineno} {ast.unparse(node)[:90]}")
    return problems


def test_the_checker_catches_raw_values(tmp_path: Path) -> None:
    page = tmp_path / "page.py"
    page.write_text(
        "st.markdown(answer.text)\n"
        "st.caption(f'Total: {total}')\n"
        "slot.markdown(safe_md(text) + ' ▌')\n"
        "st.caption(f'{seconds:.1f} s')\n"
    )
    assert [p.split()[0].rsplit(":", 1)[1] for p in unescaped_calls(page)] == ["1", "2"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT / "app")))
def test_every_page_escapes_what_it_renders(page: Path) -> None:
    assert unescaped_calls(page) == []


def test_all_pages_are_checked() -> None:
    names = {p.name for p in PAGES}
    assert {"Home.py", "ask.py", "ingest.py", "library.py", "extract.py"} <= names
