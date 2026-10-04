"""M6 layout check (PLAN.md §4.7, §4.8): every page × 760 / 1512 / 2560 px × light / dark → 24 screenshots and a report.

    uv run --group screenshots python scripts/ui_screenshots.py --data DIR [--out DIR] [--url http://localhost:8501]

Starts the app on the given library (or uses ``--url`` if it's already running), waits until Q&A is ready, then for each
page, width and theme takes a screenshot and measures:

- **placement:** is the side panel (Ask: sources; Extract: fields) beside or below the main column?
- **chars/line:** characters per line of wrapped prose in an Ask answer (target ~96, §4.8);
- **input aligned:** does the chat input start where the conversation does, at the same width?
- **math:** elements rendered as LaTeX (``.katex``). Must be 0: dollar amounts are text (``safe_md``);
- **h-scroll:** does the page scroll sideways? (it never should)

A manual pre-merge check for UI changes, not part of CI (it needs the app and its models). Playwright drives the installed
Google Chrome, so no browser is downloaded. Uses ``?source=ID`` deep links to show a document in Library and Extract.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (760, 1512, 2560)
THEMES = ("light", "dark")
HEIGHT = 1000
READY_TIMEOUT_S = 180

MEASURE = """
() => {
  const box = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
  };
  const main = box('.st-key-ask_main') || box('.st-key-extract_left');
  const side = box('.st-key-ask_side') || box('.st-key-extract_right');
  let placement = null;
  if (main && side) placement = side.x >= main.x + main.w - 1 ? 'beside' : 'below';

  // Characters per line of real wrapped prose: a paragraph added where answers render (inside a chat message, so it has
  // body text's font and the avatar's indent), its rendered lines counted.
  let cpl = null;
  let font = null;
  const host = document.querySelector('.st-key-ask_main [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]');
  if (host) {
    const p = document.createElement('p');
    p.textContent = ('This report summarizes shipments and revenue for the first three quarters. Shipments grew in every '
      + 'region except East, which dipped slightly in Q2. Marcus will send the revised draft to Legal by Wednesday, and '
      + 'Elena will update the customer forecast before the next review. ').repeat(3);
    host.appendChild(p);
    const range = document.createRange();
    range.selectNodeContents(p);
    const lines = new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size;
    cpl = Math.round(p.textContent.length / Math.max(lines - 0.5, 1));  // the last line is about half full
    font = getComputedStyle(p).fontSize;
    p.remove();
  }
  const input = box('[data-testid="stChatInput"]');
  let inputAligned = null;
  if (input && main) inputAligned = Math.abs(input.x - main.x) <= 2 && Math.abs(input.w - main.w) <= 4;
  return {
    placement, main, side, cpl, font, inputAligned,
    math: document.querySelectorAll('.katex, code.language-math').length,
    hscroll: document.documentElement.scrollWidth > window.innerWidth + 1,
    sidebar: !!document.querySelector('[data-testid="stSidebar"][aria-expanded="true"]'),
  };
}
"""


def wait_for_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"the app didn't answer at {url} within {timeout:.0f} s")


def settle(page: Any) -> None:
    """Wait for the script run to finish (title rendered, no "Running…" indicator for a moment)."""
    page.wait_for_selector("h1", timeout=60_000)
    quiet_since = time.monotonic()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        running = page.query_selector('[data-testid="stStatusWidget"]')
        if running and running.is_visible():
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > 1.5:
            return
        time.sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, help="library to show (started with GRANIT_DATA_DIR=DATA)"
    )
    parser.add_argument("--url", help="use an app that's already running instead of starting one")
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "screenshots")
    parser.add_argument(
        "--source", type=int, help="source id to show in Library (default: the first document)"
    )
    parser.add_argument("--form", type=int, help="source id to show in Extract (default: --source)")
    args = parser.parse_args(argv)
    if not args.url and not args.data:
        parser.error("give --data (start the app on a library) or --url (use a running app)")

    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    app = None
    url = args.url or f"http://localhost:{args.port}"
    if not args.url:
        env = {**os.environ, "GRANIT_DATA_DIR": str(args.data.resolve()), "HF_HUB_OFFLINE": "1"}
        command = [sys.executable, "-m", "streamlit", "run", "app/Home.py"]
        command += ["--server.port", str(args.port), "--server.headless", "true"]
        log = (args.out / "app.log").open("w")
        app = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    report: list[dict[str, Any]] = []
    try:
        wait_for_http(url, 60)
        source = args.source or first_document(args.data)
        form = args.form or source
        pages = {
            "ask": "/",
            "ingest": "/ingest",
            "library": f"/library?source={source}" if source else "/library",
            "extract": f"/extract?source={form}" if form else "/extract",
        }
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome")
            wait_until_ready(browser, url)
            for theme in THEMES:
                for width in WIDTHS:
                    context = browser.new_context(
                        viewport={"width": width, "height": HEIGHT}, color_scheme=theme
                    )
                    page = context.new_page()
                    for name, path in pages.items():
                        page.goto(url + path)
                        settle(page)
                        shot = args.out / f"{name}-{width}-{theme}.png"
                        page.screenshot(path=str(shot))
                        row = {
                            "page": name,
                            "width": width,
                            "theme": theme,
                            **page.evaluate(MEASURE),
                        }
                        report.append(row)
                        print(line(row), flush=True)
                    context.close()
            browser.close()
    finally:
        if app is not None:
            app.send_signal(signal.SIGINT)  # Streamlit shuts down; the backend stops the LLM server
            try:
                app.wait(timeout=60)
            except subprocess.TimeoutExpired:
                app.kill()
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    problems = [r for r in report if r["math"] or r["hscroll"] or r["inputAligned"] is False]
    print(f"\n{len(report)} screenshots in {args.out}; {len(problems)} with problems")
    return 1 if problems else 0


def first_document(data: Path | None) -> int | None:
    if data is None:
        return None
    from granit.store.db import Store

    store = Store(data)
    try:
        return next((s.id for s in store.sources() if s.kind == "document"), None)
    finally:
        store.close()


def wait_until_ready(browser: Any, url: str) -> None:
    """The first visit starts the backend; wait for the green "Ready" banner (search models + Q&A warm)."""
    page = browser.new_page()
    page.goto(url)
    start = time.monotonic()
    page.wait_for_selector('[data-testid="stAlertContentSuccess"]', timeout=READY_TIMEOUT_S * 1000)
    print(f"Q&A ready after {time.monotonic() - start:.0f} s", flush=True)
    page.close()


def line(row: dict[str, Any]) -> str:
    bits = [f"{row['page']:<8} {row['width']:>5} {row['theme']:<5}"]
    if row["placement"]:
        bits.append(f"panel {row['placement']}")
    if row["cpl"]:
        bits.append(f"~{row['cpl']} chars/line at {row['font']}")
    if row["inputAligned"] is not None:
        bits.append("input aligned" if row["inputAligned"] else "INPUT MISALIGNED")
    bits.append("sidebar open" if row["sidebar"] else "sidebar collapsed")
    if row["math"]:
        bits.append(f"MATH x{row['math']}")
    if row["hscroll"]:
        bits.append("H-SCROLL")
    return " · ".join(bits)


if __name__ == "__main__":
    raise SystemExit(main())
