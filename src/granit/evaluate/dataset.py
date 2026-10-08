"""Eval sets (PLAN.md §4.9): the public set in ``eval/public/`` (committed, synthetic) and the private set in ``data/eval/``.

Both use one format, in one directory:

- ``questions.yaml``: ``id``, ``question``, ``category`` (document | meeting | cross-source | unanswerable),
  ``expected_facts`` (each matched after normalization; ``"two years | 2 years"`` accepts either), ``gold_refs``
  (``{source_sha256 | file, page | start_s + end_s}``; ``granit eval label`` fills them in for a private set).
- ``files/``: the documents and recordings, ingested into the set's own library.
- ``extraction/<name>.json``: ``{file, schema, expected}``: expected field values (``null`` = must be missing).
- ``summaries/<name>.yaml``: ``{file, action_items: [{owner, task, due}], decisions: [...]}``, optionally
  ``speakers: [{name, start, end}]`` (who spoke when, for simulated speaker naming, PLAN.md §3.7).
- ``transcripts/<name>.txt``: what was said, for WER (``file:`` on the first line names the recording).
- ``tables/<name>.json``: ``{file, type, grid}``: the known cells of the file's table(s), for Docling vs Vision.
- ``vocabulary.txt`` (optional): the library's names and terms, one per line or comma-separated (``audio.parse_terms``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from granit.config import DATA_DIR, PROJECT_ROOT
from granit.evaluate.metrics import Ref
from granit.ingest.audio import parse_terms
from granit.store.db import sha256_of

CATEGORIES = ("document", "meeting", "cross-source", "unanswerable")
SETS = {"public": PROJECT_ROOT / "eval" / "public", "private": DATA_DIR / "eval"}


class EvalSetError(ValueError):
    pass


@dataclass
class Question:
    id: str
    question: str
    category: str
    expected_facts: list[str] = field(default_factory=list)
    gold_refs: list[Ref] = field(default_factory=list)


@dataclass
class ExtractionCase:
    name: str
    file: Path
    schema: dict[str, Any]
    expected: dict[str, Any]


@dataclass
class SummaryCase:
    name: str
    file: Path
    action_items: list[dict[str, Any]]
    decisions: list[str]
    speakers: list[dict[str, Any]] = field(default_factory=list)  # [{name, start, end}]


@dataclass
class TranscriptCase:
    name: str
    file: Path
    text: str


@dataclass
class TableCase:
    name: str
    file: Path
    type: str
    grid: list[list[str]]


@dataclass
class EvalSet:
    name: str
    root: Path
    questions: list[Question]
    extractions: list[ExtractionCase]
    summaries: list[SummaryCase]
    transcripts: list[TranscriptCase]
    tables: list[TableCase]
    vocabulary: list[str] = field(default_factory=list)

    @property
    def files_dir(self) -> Path:
        return self.root / "files"

    def files(self) -> list[Path]:
        return sorted(
            p for p in self.files_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
        )

    def table_files(self) -> set[Path]:
        """Files ingested with "Accurate tables", so both table paths can be scored."""
        return {t.file for t in self.tables}


def set_dir(name: str) -> Path:
    if name in SETS:
        return SETS[name]
    path = Path(name)
    if path.is_dir():
        return path
    raise EvalSetError(f"no eval set {name!r}: use public, private or a directory")


def load_set(name: str) -> EvalSet:
    root = set_dir(name)
    questions_path = root / "questions.yaml"
    if not questions_path.is_file():
        raise EvalSetError(
            f"{questions_path} doesn't exist. Start one with `granit eval init --set {name}` (PLAN.md §4.9)."
        )
    hashes = _hashes(root / "files")
    questions = [
        _question(q, root, hashes) for q in yaml.safe_load(questions_path.read_text()) or []
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise EvalSetError(f"duplicate question ids in {questions_path}")
    return EvalSet(
        name=name,
        root=root,
        questions=questions,
        extractions=[
            ExtractionCase(p.stem, _file(root, d["file"]), d["schema"], d["expected"])
            for p, d in _each(root / "extraction", "*.json", json.loads)
        ],
        summaries=[
            SummaryCase(
                p.stem,
                _file(root, d["file"]),
                d.get("action_items") or [],
                d.get("decisions") or [],
                d.get("speakers") or [],
            )
            for p, d in _each(root / "summaries", "*.yaml", yaml.safe_load)
        ],
        transcripts=[_transcript(p, root) for p in sorted((root / "transcripts").glob("*.txt"))],
        tables=[
            TableCase(p.stem, _file(root, d["file"]), d["type"], d["grid"])
            for p, d in _each(root / "tables", "*.json", json.loads)
        ],
        vocabulary=parse_terms(vocab.read_text())
        if (vocab := root / "vocabulary.txt").is_file()
        else [],
    )


def _each(folder: Path, pattern: str, parse: Any) -> list[tuple[Path, dict[str, Any]]]:
    return (
        [(p, parse(p.read_text())) for p in sorted(folder.glob(pattern))] if folder.is_dir() else []
    )


def _file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise EvalSetError(f"{relative} is referenced but missing from {root}")
    return path


def _hashes(files: Path) -> dict[str, str]:
    """relative path → sha256, so refs can name a file instead of its hash."""
    if not files.is_dir():
        return {}
    root = files.parent
    return {str(p.relative_to(root)): sha256_of(p) for p in files.rglob("*") if p.is_file()}


def _question(raw: dict[str, Any], root: Path, hashes: dict[str, str]) -> Question:
    qid = raw.get("id") or "?"
    category = raw.get("category")
    if category not in CATEGORIES:
        raise EvalSetError(f"{qid}: category must be one of {', '.join(CATEGORIES)}")
    facts = [str(f) for f in raw.get("expected_facts") or []]
    if category == "unanswerable" and facts:
        raise EvalSetError(f"{qid}: an unanswerable question has no expected facts")
    refs = []
    for ref in raw.get("gold_refs") or []:
        sha = ref.get("source_sha256")
        if sha is None:
            if ref.get("file") not in hashes:
                raise EvalSetError(
                    f"{qid}: gold ref file {ref.get('file')!r} isn't in {root / 'files'}"
                )
            sha = hashes[ref["file"]]
        elif hashes and sha not in hashes.values() and root == SETS["public"]:
            raise EvalSetError(
                f"{qid}: gold ref {sha[:12]}… matches no file (regenerate the public set?)"
            )
        refs.append(Ref(sha, ref.get("page"), ref.get("start_s"), ref.get("end_s")))
    return Question(str(qid), raw["question"], category, facts, refs)


def _transcript(path: Path, root: Path) -> TranscriptCase:
    first, _, text = path.read_text().partition("\n")
    if not first.startswith("file:"):
        raise EvalSetError(f"{path.name}: the first line must be `file: files/…`")
    return TranscriptCase(path.stem, _file(root, first[len("file:") :].strip()), text.strip())


TEMPLATE = """\
# granit private eval set (PLAN.md §4.9). Never committed: data/ is ignored.
# Put your documents and recordings in files/, then list ~30 questions here, then run
#   uv run granit eval label --set private     # tick the relevant chunks → gold_refs
#   uv run granit eval run --set private
- id: doc-001
  question: "What is the total on invoice …?"
  category: document            # document | meeting | cross-source | unanswerable
  expected_facts: ["$1,234.00"]  # "a | b" accepts either
- id: none-001
  question: "A question your documents can't answer"
  category: unanswerable
"""


def init_set(name: str) -> Path:
    """Create an empty set's folders and a commented questions.yaml (never overwrites)."""
    root = set_dir(name) if name in SETS else Path(name)
    for sub in ("files", "extraction", "summaries", "transcripts", "tables"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    questions = root / "questions.yaml"
    if not questions.exists():
        questions.write_text(TEMPLATE)
    return root
