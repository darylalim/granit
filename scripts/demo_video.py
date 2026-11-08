"""Demo video, raw take (docs/demo-plan.md): prepare a library copy, then record shots 2–7 with Playwright.

    uv run --group screenshots python scripts/demo_video.py prepare [--library DIR] [--out DIR]
    uv run --group screenshots python scripts/demo_video.py record [--out DIR] [--theme light|dark]

``prepare`` copies the public eval library (synthetic files only, never your own) to ``OUT/library`` and gets it ready to
film: speaker 1 of the Atlas check-in is named (its ``reindex`` runs now, in phase A), the meeting is summarized again with
that name (phase B), and the Ask history is cleared. ``record`` films a fresh copy of that library each take, so a retake
starts from the same state, and writes ``OUT/demo-raw.webm`` (1536×864, upscaled by the edit step), ``OUT/beats.json``
(each shot's start and end in video seconds) and ``OUT/record-report.json`` (timings, peak memory).

Shot 4 types speaker 1's name on camera: the copy already has it, so saving changes nothing and queues no ``reindex``
(``Store.set_speaker_names``). A queued job would make the app pause Q&A for a batch in the middle of the take; ``record``
fails if the copy has one afterwards. Both steps load models, so no other phase may run (hook H4).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from readme_screenshots import clear_history, idle
from ui_screenshots import ROOT, settle, wait_for_http, wait_until_ready

WIDTH, HEIGHT = (
    1536,
    864,
)  # recorded at CSS size; the edit step upscales to 1920×1080 (docs/demo-plan.md)
RECORDING = "meetings/atlas-checkin.m4a"
DOCUMENT = "reports/atlas-status.pdf"
FORM = "invoices/inv-01.pdf"
SPEAKER, NAME = 1, "Priya"  # the check-in's unnamed speaker (2 and 3 are named by the eval run)
WARMUP = "What is the approved budget for Project Atlas?"
# Chosen by its answer (docs/demo-plan.md): the eval set's "Do … agree on the go-live date?" writes "[3/4/5]" into it.
QUESTION = (
    "When does Project Atlas go live, according to the status report and the check-in meeting?"
)
ANSWER_TIMEOUT_S = 300
TYPE_DELAY_MS = 60

# A visible cursor and click ring: headless Chrome draws neither, so a recording would show clicks out of nowhere.
CURSOR_JS = """
(() => {
  const install = () => {
    if (document.getElementById('demo-cursor')) return;
    const style = document.createElement('style');
    style.textContent = `
      #demo-cursor { position: fixed; z-index: 2147483647; pointer-events: none; width: 22px; height: 22px;
        margin: -3px 0 0 -3px; left: -100px; top: -100px; }
      .demo-ring { position: fixed; z-index: 2147483646; pointer-events: none; width: 36px; height: 36px;
        margin: -18px 0 0 -18px; border-radius: 50%; border: 3px solid rgba(99, 102, 241, 0.85);
        animation: demo-ring 600ms ease-out forwards; }
      @keyframes demo-ring { from { transform: scale(0.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }`;
    document.head.appendChild(style);
    const cursor = document.createElement('div');
    cursor.id = 'demo-cursor';
    cursor.innerHTML = '<svg viewBox="0 0 22 22" width="22" height="22"><path d="M3 2 L3 18 L7.5 13.8 L10.6 20.4 '
      + 'L13.3 19.2 L10.2 12.7 L16.2 12.7 Z" fill="#111" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    document.body.appendChild(cursor);
    document.addEventListener('mousemove', (e) => {
      cursor.style.left = e.clientX + 'px';
      cursor.style.top = e.clientY + 'px';
    }, true);
    document.addEventListener('mousedown', (e) => {
      const ring = document.createElement('div');
      ring.className = 'demo-ring';
      ring.style.left = e.clientX + 'px';
      ring.style.top = e.clientY + 'px';
      document.body.appendChild(ring);
      setTimeout(() => ring.remove(), 650);
    }, true);
  };
  if (document.body) install(); else document.addEventListener('DOMContentLoaded', install);
})();
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    prep = sub.add_parser("prepare", help="copy the public eval library and get it ready to film")
    prep.add_argument("--library", type=Path, default=ROOT / "data/eval/libraries/public")
    rec = sub.add_parser(
        "record", help="record shots 2–7 from a fresh copy of the prepared library"
    )
    rec.add_argument("--theme", choices=("light", "dark"), default="light")
    rec.add_argument("--port", type=int, default=8596)
    for p in (prep, rec):
        p.add_argument("--out", type=Path, default=ROOT / "build/demo")
    args = parser.parse_args(argv)
    return (
        prepare(args.library, args.out)
        if args.step == "prepare"
        else record(args.out, args.theme, args.port)
    )


# ── prepare ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def prepare(library: Path, out: Path) -> int:
    if not (library / "granit.db").exists():
        raise SystemExit(f"no library at {library}: run `granit eval run --set public` first")
    data = out / "library"
    if data.exists():
        shutil.rmtree(data)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(library, data)
    ids = source_ids(data)
    granit(
        "speakers", RECORDING, f"{SPEAKER}={NAME}", data=data
    )  # queues a reindex if the name is new
    granit("ingest", data=data)  # runs it (phase A): search and Ask see the name
    granit(
        "meeting", RECORDING, data=data
    )  # summary with the name (phase B), so shot 5 isn't marked out of date
    clear_history(data)
    problems = check_prepared(data, ids[RECORDING])
    if problems:
        raise SystemExit(
            "the prepared library isn't ready to film:\n" + "\n".join(f"- {p}" for p in problems)
        )
    print(f"prepared {data}")
    return 0


def granit(*args: str, data: Path) -> None:
    command = [sys.executable, "-m", "granit.cli", *args, "--data", str(data)]
    print("$ granit " + " ".join(args), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        check=True,
        stdout=subprocess.DEVNULL,
    )


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


def check_prepared(data: Path, recording_id: int) -> list[str]:
    """What would show up wrong on camera: a missing name, a stale summary, queued work, old Ask turns."""
    from granit.store.db import Store
    from granit.ui.views import names_changed

    store = Store(data)
    try:
        names = store.speaker_names(recording_id)
        summaries = [e for e in store.extractions(recording_id) if e["kind"] == "summary"]
        problems = []
        if names.get(SPEAKER) != NAME:
            problems.append(f"speaker {SPEAKER} isn't named {NAME!r}: {names}")
        if not summaries:
            problems.append(f"{RECORDING} has no summary")
        elif names_changed(summaries[-1]["content"], names):
            problems.append(f"{RECORDING}'s summary was written before the names changed")
        problems += queued_jobs(data)
    finally:
        store.close()
    with sqlite3.connect(data / "granit.db") as db:
        if db.execute("SELECT count(*) FROM qa_turns").fetchone()[0]:
            problems.append("the Ask history isn't empty")
    return problems


def queued_jobs(data: Path) -> list[str]:
    with sqlite3.connect(data / "granit.db") as db:
        rows = db.execute(
            "SELECT task, source_id FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchall()
    return [f"{task} job queued for source {source}" for task, source in rows]


# ── record ───────────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Take:
    """The page being recorded and the shot log, in seconds since the video started."""

    page: Any
    started: float = field(default_factory=lambda: time.monotonic())
    shots: list[dict[str, Any]] = field(default_factory=list)

    def now(self) -> float:
        return round(time.monotonic() - self.started, 2)

    @contextmanager
    def shot(
        self, number: int, name: str, url: str | None = None, top: str | None = None
    ) -> Generator[dict[str, Any]]:
        """Log a shot. With ``url``, it starts once that page has loaded, scrolled to the heading ``top`` (if given):
        the loading page and the jump stay out of the cut."""
        print(f"shot {number}: {name}", flush=True)
        if url is not None:
            open_page(self.page, url)
            if top is not None:
                heading = self.page.locator("h2, h3").filter(has_text=top).first
                heading.evaluate("e => e.scrollIntoView({block: 'start'})")
                pause(0.5)
        entry: dict[str, Any] = {"id": number, "name": name, "start": self.now(), "waits": []}
        yield entry
        entry["end"] = self.now()
        self.shots.append(entry)

    def wait(self, entry: dict[str, Any], until: Callable[[], None]) -> None:
        """A stretch the edit step may cut or speed up (e.g. waiting for an answer)."""
        start = self.now()
        until()
        entry["waits"].append([start, self.now()])


def record(out: Path, theme: Literal["light", "dark"], port: int) -> int:
    prepared = out / "library"
    if not (prepared / "granit.db").exists():
        raise SystemExit(f"no prepared library at {prepared}: run `demo_video.py prepare` first")
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "library"
        shutil.copytree(prepared, data)
        ids = source_ids(data)
        url = f"http://localhost:{port}"
        env = {**os.environ, "GRANIT_DATA_DIR": str(data), "HF_HUB_OFFLINE": "1"}
        command = [sys.executable, "-m", "streamlit", "run", "app/Home.py"]
        command += ["--server.port", str(port), "--server.headless", "true"]
        log = (Path(tmp) / "app.log").open("w")
        app = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        memory = MemorySampler(
            os.getpid()
        )  # this process's tree: the app, its LLM server and Playwright's Chrome
        memory.start()
        report: dict[str, Any] = {}
        try:
            wait_for_http(url, 60)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel="chrome")
                start = time.monotonic()
                wait_until_ready(browser, url)
                warm_up(browser, url)
                report["warmup_s"] = round(time.monotonic() - start, 1)
                clear_history(data)
                context = browser.new_context(
                    viewport={"width": WIDTH, "height": HEIGHT},
                    color_scheme=theme,
                    record_video_dir=tmp,
                    record_video_size={"width": WIDTH, "height": HEIGHT},
                )
                context.add_init_script(CURSOR_JS)
                take = Take(context.new_page())
                film(take, url, ids)
                video = take.page.video
                context.close()  # writes the video
                raw = Path(video.path())  # only while Playwright runs
                browser.close()
            out.mkdir(parents=True, exist_ok=True)
            shutil.move(raw, out / "demo-raw.webm")
        finally:
            app.send_signal(signal.SIGINT)  # Streamlit shuts down; the backend stops the LLM server
            try:
                app.wait(timeout=60)
            except subprocess.TimeoutExpired:
                app.kill()
            log.close()
            memory.stop.set()
        leftovers = queued_jobs(data)
    (out / "beats.json").write_text(
        json.dumps({"size": [WIDTH, HEIGHT], "shots": take.shots}, indent=2) + "\n"
    )
    report |= {
        "peak_rss_gb": memory.peak_gb,
        "shots_s": {s["name"]: round(s["end"] - s["start"], 1) for s in take.shots},
    }
    (out / "record-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"{out / 'demo-raw.webm'}, beats.json, record-report.json")
    if leftovers:
        print(
            "the take queued work, so Q&A may have paused on camera:\n" + "\n".join(leftovers),
            file=sys.stderr,
        )
        return 1
    return 0


def warm_up(browser: Any, url: str) -> None:
    """One answer off camera: the LLM server, embedder and reranker are loaded and the "Loading…" banner is gone."""
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
    page.goto(url + "/")
    settle(page)
    box = page.locator('[data-testid="stChatInputTextArea"]')
    box.fill(WARMUP)
    box.press("Enter")
    time.sleep(2)
    idle(page, 4, ANSWER_TIMEOUT_S)
    page.close()


def film(take: Take, url: str, ids: dict[str, int]) -> None:
    page = take.page

    with take.shot(2, "ingest", url + "/ingest"):
        point_at(page, page.locator('[data-testid="stFileUploaderDropzone"]'))
        pause(2.5)
        reveal(page.locator("h3").filter(has_text="Queue").first)
        pause(3)

    with take.shot(3, "document", f"{url}/library?source={ids[DOCUMENT]}", top=DOCUMENT):
        pause(4)
        # The tab is shorter than the document, so the page would scroll up to the Library table. Room below the
        # content (on camera only) lets the heading stay at the top.
        page.evaluate(
            "document.querySelector('[data-testid=\"stMainBlockContainer\"]').style.paddingBottom = '100vh'"
        )
        click(page, page.get_by_role("tab", name="Tables and charts"))
        reveal(page.locator("h2, h3").filter(has_text=DOCUMENT).first)
        pause(4)

    with take.shot(4, "speakers", f"{url}/library?source={ids[RECORDING]}", top=RECORDING) as shot:
        click(
            page,
            page.locator('[data-testid="stExpander"] summary').filter(has_text="Speakers").first,
        )
        field_ = page.get_by_label(f"Name for speaker {SPEAKER}")
        field_.fill("")  # the copy has the name already: show it typed (the shot starts after this)
        shot["start"] = take.now()
        pause(2)
        click(page, field_)
        page.keyboard.type(NAME, delay=TYPE_DELAY_MS * 2)
        pause(0.8)
        click(page, page.get_by_role("button", name="Save names"))
        settle(page)
        pause(2.5)
        reveal(page.get_by_role("tab", name="Transcript"))
        click(page, page.get_by_role("tab", name="Transcript"))
        pause(4)

    with take.shot(5, "summary"):
        click(page, page.get_by_role("tab", name="Summary"))
        pause(1.5)
        reveal(page.get_by_text("Action items", exact=True).first)
        pause(5)

    with take.shot(6, "ask", url + "/") as shot:
        box = page.locator('[data-testid="stChatInputTextArea"]')
        click(page, box)
        page.keyboard.type(QUESTION, delay=TYPE_DELAY_MS)
        pause(0.6)
        page.keyboard.press("Enter")
        take.wait(shot, lambda: wait_for_answer(page))
        pause(2)
        reveal(page.locator("h3").filter(has_text="Sources").first)
        pause(5)

    with take.shot(7, "extract", f"{url}/extract?source={ids[FORM]}"):
        reveal(page.locator("h3").filter(has_text="Fields").first)
        pause(5)


def open_page(page: Any, url: str) -> None:
    page.goto(url)
    settle(page)
    page.mouse.move(WIDTH * 0.6, HEIGHT * 0.45)
    pause(1)


def point_at(page: Any, locator: Any) -> tuple[float, float]:
    """Glide the cursor to the element's left part (where a label or text starts), not its exact centre."""
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    if box is None:
        raise SystemExit(f"not on the page: {locator}")
    x, y = box["x"] + min(box["width"] / 2, 60), box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=25)
    return x, y


def click(page: Any, locator: Any) -> None:
    x, y = point_at(page, locator)
    pause(0.3)
    page.mouse.click(x, y)
    pause(0.5)


def reveal(locator: Any) -> None:
    """Scroll the element to the top smoothly (the app scrolls an inner container, so use the DOM)."""
    locator.evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'start'})")
    pause(1.2)


def wait_for_answer(page: Any) -> None:
    time.sleep(2)  # the "Running…" indicator appears after a moment
    idle(page, 4, ANSWER_TIMEOUT_S)


def pause(seconds: float) -> None:
    time.sleep(seconds)


class MemorySampler(threading.Thread):
    """Peak resident memory (GB) of a process tree, sampled every second. RSS undercounts the LLM's GPU buffers."""

    def __init__(self, root: int) -> None:
        super().__init__(daemon=True)
        self.root = root
        self.peak_gb = 0.0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            ps = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True
            ).stdout
            self.peak_gb = max(self.peak_gb, round(tree_rss_kb(ps, self.root) / 2**20, 2))
            time.sleep(1)


def tree_rss_kb(ps_output: str, root: int) -> int:
    """Total RSS (KiB) of ``root`` and all its descendants, from ``ps -axo pid=,ppid=,rss=`` output."""
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in ps_output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, kb = (int(p) for p in parts)
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    total, stack, seen = 0, [root], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack += children.get(pid, [])
    return total


if __name__ == "__main__":
    sys.exit(main())
