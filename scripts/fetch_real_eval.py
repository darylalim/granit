"""Fetch a real-world eval set into data/eval/ (PLAN.md §4.9): published documents and recordings, nothing synthetic.

    uv run python scripts/fetch_real_eval.py [--out data/eval]

The private set is meant to be your own documents. Until you have them, this fills it with real content nobody designed
for granit, whose references were written by someone else (meeting summaries by AMI annotators, receipt fields by CORD
labelers, document facts read from the published PDFs):

- **files/reports:** 5 US federal PDFs (public domain, 17 U.S.C. §105): Fed projections (tables + charts), Census
  foreign-born brief (52-row table, map), FOMC minutes (text-heavy), Census e-commerce release (dense tables), IRS W-9.
- **files/scans:** a 1964 routing sheet from the National Archives JFK release: image only, typewriter + stamps.
- **files/receipts:** 5 phone photos of Indonesian receipts from CORD v2 (CC BY 4.0) → ``extraction/``.
- **files/meetings:** 2 AMI meetings (CC BY 4.0), ~33 and ~37 min, four real speakers on a room mix → ``summaries/``
  (the annotators' action items and decisions) and ``transcripts/`` (their word-level transcript, for WER).
- **questions.yaml:** 31 questions with ``gold_refs`` (page or time range), each answer checked against the source.

Every download is pinned by sha256 (CORD by dataset revision) and refused on mismatch. Existing files are kept, so the
script can be rerun; ``questions.yaml`` is written only if it's missing or still ``granit eval init``'s template,
otherwise next to it as ``questions.fetched.yaml``. Attribution for the CC BY sources goes to ``SOURCES.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from granit.config import DATA_DIR
from granit.evaluate.dataset import TEMPLATE, init_set

FED = "https://www.federalreserve.gov/monetarypolicy/files"
AMI = "https://groups.inf.ed.ac.uk/ami"

# (published URL, sha256, path under files/)
DOWNLOADS = [
    (f"{FED}/fomcprojtabl20240320.pdf",
     "9e6705e28948d10b54b37bbda2c94fa4eb553864aed82be5bf985f5cfe735a5a", "reports/fed-projections-2024-03.pdf"),
    ("https://www2.census.gov/library/publications/2024/demo/acsbr-019.pdf",
     "d1a868b7ca76587e9acc51dfec676b279cd2348628d34cb0d7a35551af499ab0", "reports/census-foreign-born-2022.pdf"),
    (f"{FED}/fomcminutes20240131.pdf",
     "6ce451a5bcf43bf90ac2172f6872cd245e04689c80d4b01e28a01519bb3f0dcc", "reports/fomc-minutes-2024-01.pdf"),
    ("https://www2.census.gov/retail/releases/historical/ecomm/24q2.pdf",
     "2fa56301ce5817410b2e529d56c2ae4815c4472875f457bb5ee99045d2ef7cb3", "reports/census-ecommerce-2024q2.pdf"),
    ("https://www.irs.gov/pub/irs-prior/fw9--2024.pdf",
     "2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53", "reports/irs-w9-2024.pdf"),
    ("https://www.archives.gov/files/research/jfk/releases/2025/0318/104-10003-10041.pdf",
     "c3c7e7b44e6f65e08d4e73950b74428f40d26e7ead49e945f003f6b8f0903e48", "scans/nara-104-10003-10041.pdf"),
    (f"{AMI}/AMICorpusMirror/amicorpus/IB4003/audio/IB4003.Mix-Headset.wav",
     "a28e876a8dc3ed076ce38a167d8de4cc1d3ff8c0b5142c57d4c194ece7752fc2", "meetings/ami-IB4003.wav"),
    (f"{AMI}/AMICorpusMirror/amicorpus/ES2008b/audio/ES2008b.Mix-Headset.wav",
     "8aa976aa242e895775857b028db2ca5c5f56a2a2d8a096adc81cc82a9c4b6ba4", "meetings/ami-ES2008b.wav"),
]  # fmt: skip
AMI_ANNOTATIONS = (
    f"{AMI}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
    "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d",
)
CORD = ("naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98",
        "data/test-00000-of-00001-9c204eb3f4e11791.parquet")  # fmt: skip
CORD_ROWS = (4, 6, 20, 94, 98)  # test-split rows: discount, cash + change, service + tax

RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtotal": {"type": "string", "description": "Subtotal before service charge and tax"},
        "discount": {"type": "string", "description": "Discount amount, if any"},
        "service_charge": {"type": "string", "description": "Service charge amount, if any"},
        "tax": {"type": "string", "description": "Tax amount, if any"},
        "total": {"type": "string", "description": "The total / grand total"},
        "cash_paid": {"type": "string", "description": "Cash handed over, if paid in cash"},
        "change": {"type": "string", "description": "Change given back, if any"},
    },
}
# schema field → CORD gt_parse (section, key); a field CORD doesn't label is expected to be missing (null)
CORD_FIELDS = {
    "subtotal": ("sub_total", "subtotal_price"),
    "discount": ("sub_total", "discount_price"),
    "service_charge": ("sub_total", "service_price"),
    "tax": ("sub_total", "tax_price"),
    "total": ("total", "total_price"),
    "cash_paid": ("total", "cashprice"),
    "change": ("total", "changeprice"),
}

# Action items from the AMI abstractive summaries, with the owner made explicit (None = the whole group). Owners are
# meeting roles: the participants mostly address each other by role, not name.
MEETINGS: dict[str, dict[str, Any]] = {
    "IB4003": {
        "action_items": [
            {"owner": "Project Manager", "task": "find out whether offices can be split up with walls and doors"},
            {"owner": "Project Manager", "task": "find out whether Gisella can be placed with the other admin staff of the school"},
            {"owner": None, "task": "come up with their own preferences for arranging people in the space"},
            {"owner": None, "task": "poll other people about their preferences for the arrangement"},
            {"owner": None, "task": "ask Maggie and Pierrette about their preferences"},
            {"owner": None, "task": "meet next Tuesday at two o'clock", "due": "next Tuesday"},
        ],
        "decisions": [
            "People of the same project will not be grouped together in offices.",
            "Gisella gets a two-person office whose second place is used by visitors.",
            "There will not be a cafeteria.",
        ],
    },
    "ES2008b": {
        "action_items": [
            {"owner": "Project Manager", "task": "post the minutes and project documentation"},
            {"owner": "Industrial Designer", "task": "work on the components concept"},
            {"owner": "User Interface Designer", "task": "work on the user interface concept"},
            {"owner": "Marketing Expert", "task": "work on trend watching"},
            {"owner": None, "task": "complete a questionnaire and a summary"},
        ],
        "decisions": [
            "The team will not work with teletext.",
            "The remote will be used only with televisions.",
            "The corporate image must be recognizable on the remote.",
            "The design will focus on simplicity and fashion.",
            "The remote will have a function to help find it when lost.",
            "The remote will have large buttons for the essential functions.",
            "Extra, rarely used functions will be hidden in the design.",
            "The remote will have a charging station.",
        ],
    },
}  # fmt: skip
FILLERS = {"uh", "um", "mm", "hmm", "mm-hmm", "uh-huh", "mm-mm", "uh-uh", "huh", "ah", "oh", "eh"}

SEP, CENSUS, FOMC = (
    "files/reports/fed-projections-2024-03.pdf",
    "files/reports/census-foreign-born-2022.pdf",
    "files/reports/fomc-minutes-2024-01.pdf",
)
ECOMM, W9, SCAN = (
    "files/reports/census-ecommerce-2024q2.pdf",
    "files/reports/irs-w9-2024.pdf",
    "files/scans/nara-104-10003-10041.pdf",
)
IB, ES = "files/meetings/ami-IB4003.wav", "files/meetings/ami-ES2008b.wav"


def q(
    qid: str, question: str, category: str, facts: list[str], *refs: dict[str, Any]
) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": qid, "question": question, "category": category}
    if facts:
        entry["expected_facts"] = facts
    if refs:
        entry["gold_refs"] = list(refs)
    return entry


def page(file: str, n: int) -> dict[str, Any]:
    return {"file": file, "page": n}


def span(file: str, start: float, end: float) -> dict[str, Any]:
    return {"file": file, "start_s": start, "end_s": end}


# Every answer was read from the source: PDF text by page (pypdfium2), the scan by eye, meetings from AMI's transcript.
QUESTIONS = [
    q("doc-01", "In the Fed's March 2024 projections, what is the median projected change in real GDP for 2024?", "document", ["2.1"], page(SEP, 2)),
    q("doc-02", "What is the median projected federal funds rate for the end of 2025 in the March 2024 projections?", "document", ["3.9"], page(SEP, 2)),
    q("doc-03", "What is the central tendency of the unemployment rate projection for 2026 in the March 2024 projections?", "document", ["3.9–4.3 | 3.9-4.3 | 3.9 to 4.3"], page(SEP, 2)),
    q("doc-04", "According to Table 2 of the March 2024 projections, what is the average historical projection error for the unemployment rate in 2026?", "document", ["1.9"], page(SEP, 16)),
    q("doc-05", "How large was Nevada's foreign-born population in 2022, and what share of the state was it?", "document", ["601", "18.9"], page(CENSUS, 3)),
    q("doc-06", "Which state had the smallest foreign-born share in 2022, and what was it?", "document", ["West Virginia", "1.8"], page(CENSUS, 3)),
    q("doc-07", "How big was the US foreign-born population in 1970?", "document", ["9.6 million"], page(CENSUS, 1)),
    q("doc-08", "Who was selected as manager of the System Open Market Account at the January 2024 FOMC meeting?", "document", ["Roberto Perli"], page(FOMC, 1)),
    q("doc-09", "What interest rate on reserve balances took effect on February 1, 2024?", "document", ["5.4 percent | 5.4%"], page(FOMC, 11)),
    q("doc-10", "When did the January 2024 FOMC meeting adjourn?", "document", ["10:25"], page(FOMC, 11)),
    q("doc-11", "What were the not seasonally adjusted e-commerce sales in the 4th quarter of 2023?", "document", ["322,862"], page(ECOMM, 2)),
    q("doc-12", "About how many retail firms does the Census e-commerce survey sample?", "document", ["10,800"], page(ECOMM, 3)),
    q("doc-13", "What backup withholding rate does Form W-9 mention?", "document", ["24%"], page(W9, 2)),
    q("doc-14", "On Form W-9, what is the exempt payee code for a real estate investment trust?", "document", ["8"], page(W9, 4)),
    q("doc-15", "What is the penalty for failing to furnish a correct TIN to a requester on Form W-9?", "document", ["$50"], page(W9, 3)),
    q("scan-01", "What date is on the CIA routing and record sheet about the Rinascita article?", "document", ["1964-03-26 | 26 March 1964 | March 26, 1964"], page(SCAN, 1)),
    q("scan-02", "Who wrote the article that the routing sheet forwards, and in which weekly did it appear?", "document", ["Corsini", "Rinascita"], page(SCAN, 1)),
    q("rcpt-01", "What was the grand total on the receipt with the Pho Tai Chin?", "document", ["140,063 | 140.063 | 140063"], page("files/receipts/cord-094.png", 1)),
    q("rcpt-02", "How much was the discount on the receipt with the avocado coffee?", "document", ["19,400 | 19.400 | 19400"], page("files/receipts/cord-004.png", 1)),
    q("mtg-01", "When is the next office-move meeting?", "meeting", ["Tuesday", "two o'clock | 2 o'clock | 2:00 | 2 pm | 14:00"], span(IB, 1920, 1982)),
    q("mtg-02", "Where can people smoke in the new building?", "meeting", ["balcony"], span(IB, 1355, 1380)),
    q("mtg-03", "In the office-move meeting, which two people are mentioned as not present whose office preferences matter?", "meeting", ["Maggie", "Pierrette"], span(IB, 1585, 1600)),
    q("mtg-04", "What option did the project manager agree to look into for Gisella?", "meeting", ["admin"], span(IB, 1475, 1530)),
    q("mtg-05", "Why won't the remote control team work with teletext?", "meeting", ["internet"], span(ES, 1275, 1292)),
    q("mtg-06", "What is the remote control's selling price, and what is the limit on its production cost?", "meeting", ["25 | twenty five | twenty-five", "12.5 | 12.50 | twelve and a half"], span(ES, 1470, 1570)),
    q("mtg-07", "What did the team suggest to keep the remote's battery charged?", "meeting", ["charging station | cradle"], span(ES, 2050, 2075)),
    q("cross-01", "What federal funds target range did the FOMC keep in January 2024, and what was the median projected rate for the end of 2024 in the March projections?", "cross-source", ["5¼ | 5.25 | 5-1/4 | 5 1/4", "4.6"], page(FOMC, 11), page(SEP, 2)),
    q("none-01", "What is the median projection for real GDP growth in 2027 in the March 2024 projections?", "unanswerable", []),
    q("none-02", "What percentage of Puerto Rico's population was foreign-born in 2022?", "unanswerable", []),
    q("none-03", "How long is the warranty on the remote control the design team discussed?", "unanswerable", []),
    q("none-04", "What is the street address of the building the office-move team is moving to?", "unanswerable", []),
]  # fmt: skip

SOURCES_MD = """\
# Sources of this eval set

Fetched by `scripts/fetch_real_eval.py`. Kept in data/ (never committed).

- **AMI Meeting Corpus** (meetings IB4003, ES2008b; manual annotations 1.6.2), University of Edinburgh / AMI Consortium,
  CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/ . Transcripts and summaries here are derived from its annotations.
- **CORD v2** (test rows {cord_rows}), NAVER Clova, CC BY 4.0, https://huggingface.co/datasets/naver-clova-ix/cord-v2
- US federal government works (public domain, 17 U.S.C. §105): Federal Reserve Board (Summary of Economic Projections,
  March 2024; FOMC minutes, January 2024), US Census Bureau (ACSBR-019; Quarterly Retail E-Commerce Sales 2024 Q2),
  IRS (Form W-9, Rev. March 2024), National Archives (JFK Assassination Records, 104-10003-10041).
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, expected: str) -> bytes:
    print(f"  ↓ {url}")
    # Some hosts (federalreserve.gov) refuse Python's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "granit-eval/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if sha256(data) != expected:
        raise SystemExit(f"{url} changed upstream (sha256 {sha256(data)[:12]}…); not using it")
    return data


def save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def transcript(annotations: zipfile.ZipFile, meeting: str) -> str:
    """AMI's word-level transcript, all speakers merged by start time; punctuation tokens and fillers dropped."""
    words: list[tuple[float, str]] = []
    for name in annotations.namelist():
        if re.fullmatch(rf"words/{meeting}\.[A-Z]\.words\.xml", name):
            xml = annotations.read(name).decode("latin-1")
            for attrs, word in re.findall(r"<w ([^>]*)>(.*?)</w>", xml):
                start = re.search(r'starttime="([\d.]+)"', attrs)
                if start and 'punc="true"' not in attrs:
                    words.append((float(start.group(1)), html.unescape(word)))
    words.sort()
    return " ".join(w for _, w in words if w.lower() not in FILLERS)


def receipts(files: Path, extraction: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision, filename = CORD
    parquet = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    rows = pq.read_table(parquet).to_pylist()
    for i in CORD_ROWS:
        name = f"cord-{i:03d}"
        image = files / "receipts" / f"{name}.png"
        if not image.exists():
            save(image, rows[i]["image"]["bytes"])
        gt = json.loads(rows[i]["ground_truth"])["gt_parse"]
        expected = {f: (gt.get(sec) or {}).get(key) for f, (sec, key) in CORD_FIELDS.items()}
        case = {
            "file": f"files/receipts/{name}.png",
            "schema": RECEIPT_SCHEMA,
            "expected": expected,
        }
        (extraction / f"{name}.json").write_text(json.dumps(case, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DATA_DIR / "eval")
    root = init_set(str(parser.parse_args().out))
    files = root / "files"

    print("documents and recordings")
    for url, digest, relative in DOWNLOADS:
        target = files / relative
        if not target.exists():
            save(target, fetch(url, digest))

    print("AMI annotations")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ami.zip"
        archive.write_bytes(fetch(*AMI_ANNOTATIONS))
        with zipfile.ZipFile(archive) as annotations:
            for meeting, refs in MEETINGS.items():
                file = f"files/meetings/ami-{meeting}.wav"
                (root / "transcripts" / f"ami-{meeting}.txt").write_text(
                    f"file: {file}\n{transcript(annotations, meeting)}\n"
                )
                summary = {"file": file, **refs}
                (root / "summaries" / f"ami-{meeting}.yaml").write_text(
                    yaml.safe_dump(summary, sort_keys=False, width=120, allow_unicode=True)
                )
            license_text = annotations.read("LICENCE.txt")
    save(root / "licenses" / "AMI-LICENCE.txt", license_text)

    print("CORD receipts")
    receipts(files, root / "extraction")

    (root / "SOURCES.md").write_text(SOURCES_MD.format(cord_rows=", ".join(map(str, CORD_ROWS))))
    header = "# Real-world eval set from scripts/fetch_real_eval.py (PLAN.md §4.9). Answers checked against the sources.\n"
    content = header + yaml.safe_dump(QUESTIONS, sort_keys=False, width=120, allow_unicode=True)
    questions = root / "questions.yaml"
    untouched = not questions.exists() or questions.read_text() in (TEMPLATE, content)
    target = questions if untouched else root / "questions.fetched.yaml"
    target.write_text(content)
    print(f"\n{root}: {len(QUESTIONS)} questions → {target.name}")
    if target != questions:
        print(
            "  questions.yaml already has your own questions: merge questions.fetched.yaml into it by hand"
        )


if __name__ == "__main__":
    main()
