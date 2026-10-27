"""README screenshots: Ask, Library (recording, document) and Extract × light / dark → docs/screenshots/.

    uv run --group screenshots python scripts/readme_screenshots.py [--library DIR] [--out DIR]

Shows the public eval library (synthetic files only, never your own) from a temporary copy, so the library itself is never
changed. The copy's Ask history is cleared before each theme (the app saves every answer in ``qa_turns``), so Ask shows one
fresh cited answer. Library pages scroll to the source's detail; the source table would otherwise fill the screenshot.
Needs Q&A, so no other phase may run (hook H4). Re-run after UI changes and look at the images before committing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ui_screenshots import ROOT, settle, wait_for_http, wait_until_ready

WIDTH, HEIGHT = 1512, 1000
THEMES = ("light", "dark")
QUESTION = "Where is Project Atlas on budget and schedule?"
RECORDING = "meetings/atlas-checkin.m4a"
DOCUMENT = "reports/atlas-status.pdf"
FORM = "invoices/inv-01.pdf"
ANSWER_TIMEOUT_S = 300


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", type=Path, default=ROOT / "data/eval/libraries/public")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/screenshots")
    parser.add_argument("--port", type=int, default=8598)
    args = parser.parse_args(argv)
    if not (args.library / "granit.db").exists():
        parser.error(f"no library at {args.library}: run `granit eval run --set public` first")

    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "library"
        shutil.copytree(args.library, data)
        ids = source_ids(data)
        pages = {  # name: (path, heading scrolled to the top, or None)
            "library-recording": (f"/library?source={ids[RECORDING]}", RECORDING),
            "library-document": (f"/library?source={ids[DOCUMENT]}", DOCUMENT),
            "extract": (f"/extract?source={ids[FORM]}", None),
        }
        url = f"http://localhost:{args.port}"
        env = {**os.environ, "GRANIT_DATA_DIR": str(data), "HF_HUB_OFFLINE": "1"}
        command = [sys.executable, "-m", "streamlit", "run", "app/Home.py"]
        command += ["--server.port", str(args.port), "--server.headless", "true"]
        log = (Path(tmp) / "app.log").open("w")
        app = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_http(url, 60)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel="chrome")
                wait_until_ready(browser, url)
                for theme in THEMES:
                    clear_history(data)
                    context = browser.new_context(
                        viewport={"width": WIDTH, "height": HEIGHT}, color_scheme=theme
                    )
                    page = context.new_page()
                    ask(page, url)
                    shoot(page, args.out / f"ask-{theme}.png")
                    for name, (path, heading) in pages.items():
                        page.goto(url + path)
                        settle(page)
                        if heading:
                            scroll_to(page, heading)
                        shoot(page, args.out / f"{name}-{theme}.png")
                    context.close()
                browser.close()
        finally:
            app.send_signal(signal.SIGINT)  # Streamlit shuts down; the backend stops the LLM server
            try:
                app.wait(timeout=60)
            except subprocess.TimeoutExpired:
                app.kill()
            log.close()
    print(f"{len(THEMES) * (len(pages) + 1)} screenshots in {args.out}")
    return 0


def source_ids(data: Path) -> dict[str, int]:
    from granit.store.db import Store

    store = Store(data)
    try:
        ids = {s.name: s.id for s in store.sources()}
    finally:
        store.close()
    missing = [n for n in (RECORDING, DOCUMENT, FORM) if n not in ids]
    if missing:
        raise SystemExit(f"not in the library: {', '.join(missing)}")
    return ids


def clear_history(data: Path) -> None:
    """Empty the copy's Ask history (and its verdicts, by cascade) so Ask shows one answer."""
    with sqlite3.connect(data / "granit.db") as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("DELETE FROM qa_turns")


def ask(page: Any, url: str) -> None:
    page.goto(url + "/")
    settle(page)
    box = page.locator('[data-testid="stChatInputTextArea"]')
    box.fill(QUESTION)
    box.press("Enter")
    time.sleep(2)
    idle(page, 4, ANSWER_TIMEOUT_S)


def idle(page: Any, quiet_s: float, timeout: float) -> None:
    """Wait until Streamlit's "Running…" indicator has been hidden for ``quiet_s``."""
    quiet_since = time.monotonic()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = page.query_selector('[data-testid="stStatusWidget"]')
        if running and running.is_visible():
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > quiet_s:
            return
        time.sleep(0.3)
    raise SystemExit(f"no answer within {timeout:.0f} s")


def scroll_to(page: Any, heading: str) -> None:
    """Scroll the source's detail to the top (the app scrolls an inner container, so use the DOM)."""
    element = page.locator("h2, h3").filter(has_text=heading).first
    element.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.move(5, HEIGHT // 2)  # no hover highlight on a table row


def shoot(page: Any, path: Path) -> None:
    time.sleep(1)
    page.screenshot(path=str(path))
    print(path.name, flush=True)


if __name__ == "__main__":
    sys.exit(main())
