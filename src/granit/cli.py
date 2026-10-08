"""``granit`` command line. M0: ``granit models list|download|convert|smoke``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from granit.config import HUB_MODELS, LOCAL_MODELS, RUNTIME_HUB_MODELS


def _models_list(_: argparse.Namespace) -> int:
    print(f"{'key':<16} {'status':<12} {'GB':>6}  {'phase':<7} model @ revision")
    for m in HUB_MODELS.values():
        status = "downloaded" if m.cached_snapshot() else "missing"
        print(
            f"{m.key:<16} {status:<12} {m.size_gb:>6.2f}  {m.phase:<7} {m.repo_id} @ {m.revision[:10]}"
        )
    for m in LOCAL_MODELS.values():
        status = "built" if m.is_built() else "not built"
        print(
            f"{m.key:<16} {status:<12} {m.size_gb:>6.2f}  {m.phase:<7} {m.path} (from {m.source})"
        )
    return 0


def _models_download(args: argparse.Namespace) -> int:
    from granit.models.download import download

    keys = args.keys or list(RUNTIME_HUB_MODELS)
    for key in keys:
        model = HUB_MODELS[key]
        print(f"→ {model.repo_id} @ {model.revision[:10]} (~{model.size_gb:.2f} GB)", flush=True)
        print(f"  {download(model)}")
    return 0


def _models_convert(args: argparse.Namespace) -> int:
    from granit.models.convert import build
    from granit.models.download import delete_cached

    for key in args.keys or list(LOCAL_MODELS):
        model = LOCAL_MODELS[key]
        precision = f"{model.q_bits}-bit" if model.q_bits else "unquantized"
        print(f"→ building {model.path.name} ({precision}) from {model.source}", flush=True)
        print(f"  {build(model)}")
        if args.delete_source:
            freed = delete_cached(HUB_MODELS[model.source])
            print(f"  deleted the source from the HF cache ({freed / 1e9:.1f} GB freed)")
    return 0


def _models_smoke(args: argparse.Namespace) -> int:
    from granit.models.smoke import CHECKS, run_isolated

    failed = 0
    for key in args.keys or list(CHECKS):
        result = run_isolated(key)
        failed += not result.ok
        print(result.summary(), flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="granit", description="Local document & meeting intelligence."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="download, convert and check model weights")
    actions = models.add_subparsers(dest="action", required=True)

    actions.add_parser("list", help="show pinned models and what's on disk").set_defaults(
        func=_models_list
    )

    download = actions.add_parser("download", help="download pinned runtime models (default: all)")
    download.add_argument("keys", nargs="*", choices=[[], *HUB_MODELS], metavar="KEY")
    download.set_defaults(func=_models_download)

    convert = actions.add_parser("convert", help="build local models (Guardian q8, diarization)")
    convert.add_argument("keys", nargs="*", choices=[[], *LOCAL_MODELS], metavar="KEY")
    convert.add_argument(
        "--delete-source",
        action="store_true",
        help="remove the bf16 source from the HF cache after a successful build",
    )
    convert.set_defaults(func=_models_convert)

    from granit.models.smoke import CHECKS

    smoke = actions.add_parser(
        "smoke", help="load each model in its own process and run a tiny check"
    )
    smoke.add_argument("keys", nargs="*", choices=[[], *CHECKS], metavar="KEY")
    smoke.set_defaults(func=_models_smoke)

    transcribe = commands.add_parser(
        "transcribe", help="transcribe audio files (VAD + TurboCTC) and print timestamped segments"
    )
    transcribe.add_argument("files", nargs="+", type=Path, metavar="FILE")
    transcribe.add_argument(
        "--json", type=Path, metavar="DIR", help="also write <DIR>/<file stem>.transcript.json"
    )
    transcribe.set_defaults(func=_transcribe)

    convert = commands.add_parser(
        "convert", help="convert PDFs / images to Markdown + chart data (Docling + Granite Vision)"
    )
    convert.add_argument("files", nargs="+", type=Path, metavar="FILE")
    convert.add_argument(
        "--out", type=Path, default=Path("converted"), metavar="DIR", help="writes DIR/<file stem>/"
    )
    convert.add_argument(
        "--vision-tables",
        action="store_true",
        help="also re-extract every table with Granite Vision (slow)",
    )
    convert.set_defaults(func=_convert)

    extract = commands.add_parser(
        "extract", help="extract fields from a document with a JSON Schema (Granite Vision)"
    )
    extract.add_argument("file", type=Path, metavar="FILE")
    extract.add_argument("--schema", type=Path, required=True, metavar="SCHEMA.json")
    extract.add_argument(
        "--pages", type=int, default=4, help="at most this many PDF pages (default 4)"
    )
    extract.set_defaults(func=_extract)

    data_help = "data directory (default: GRANIT_DATA_DIR or ./data)"
    ingest = commands.add_parser(
        "ingest", help="add files to the library and process the queue (Phase A)"
    )
    ingest.add_argument("files", nargs="*", type=Path, metavar="FILE")
    ingest.add_argument("--data", type=Path, help=data_help)
    ingest.set_defaults(func=_ingest)

    search = commands.add_parser("search", help="hybrid search over the library, with citations")
    search.add_argument("query")
    search.add_argument(
        "--mode", default="hybrid+rerank", choices=["bm25", "vectors", "hybrid", "hybrid+rerank"]
    )
    search.add_argument("-k", type=int, default=8)
    search.add_argument(
        "--json", action="store_true", help="print the hits and per-stage trace as JSON"
    )
    search.add_argument("--data", type=Path, help=data_help)
    search.set_defaults(func=_search)

    sources = commands.add_parser("sources", help="list the library: files, status, chunks")
    sources.add_argument("--data", type=Path, help=data_help)
    sources.set_defaults(func=_sources)

    ask = commands.add_parser(
        "ask", help="answer a question from the library, with citations (Phase B)"
    )
    ask.add_argument("question")
    ask.add_argument("--thinking", default="low", choices=["off", "low", "on"])
    ask.add_argument(
        "--mode", default="hybrid+rerank", choices=["bm25", "vectors", "hybrid", "hybrid+rerank"]
    )
    ask.add_argument("--data", type=Path, help=data_help)
    ask.set_defaults(func=_ask)

    meeting = commands.add_parser(
        "meeting", help="summarize an ingested recording: decisions + action items (JSON)"
    )
    meeting.add_argument("source", help="source id (see `granit sources`) or file name")
    meeting.add_argument("--data", type=Path, help=data_help)
    meeting.set_defaults(func=_meeting)

    speakers = commands.add_parser(
        "speakers",
        help="list a recording's speakers, or name them: granit speakers 3 1=Priya 2='Project Manager'",
    )
    speakers.add_argument("source", help="source id (see `granit sources`) or file name")
    speakers.add_argument(
        "names", nargs="*", metavar="SPEAKER=NAME", help="name speakers (blank name: unname)"
    )
    speakers.add_argument("--data", type=Path, help=data_help)
    speakers.set_defaults(func=_speakers)

    verify = commands.add_parser(
        "verify",
        help="check answers and meeting summaries with Granite Guardian (Phase C)",
    )
    verify.add_argument("--data", type=Path, help=data_help)
    verify.set_defaults(func=_verify)

    ui = commands.add_parser(
        "ui", help="open the app in your browser (Ingest, Library, Ask, Extract)"
    )
    ui.add_argument("--data", type=Path, help=data_help)
    ui.add_argument("--port", type=int, default=8501, help="local port for the app (default: 8501)")
    ui.set_defaults(func=_ui)

    evaluate = commands.add_parser("eval", help="quality evaluation on an eval set (PLAN.md §4.9)")
    eval_actions = evaluate.add_subparsers(dest="eval_action", required=True)
    run = eval_actions.add_parser(
        "run", help="ingest, answer, judge and score a set (~15 min for public)"
    )
    run.add_argument(
        "--set", default="public", help="public, private or a directory (default: public)"
    )
    run.add_argument(
        "--retrieval",
        default="hybrid+rerank",
        choices=["bm25", "vectors", "hybrid", "hybrid+rerank"],
    )
    run.add_argument("--thinking", default="low", choices=["off", "low", "on"])
    run.add_argument("--no-judge", action="store_true", help="skip Phase C (Guardian)")
    run.add_argument(
        "--fresh",
        action="store_true",
        help="re-ingest everything (after model or chunking changes)",
    )
    run.add_argument(
        "--controls",
        type=int,
        default=12,
        help="deliberately wrong answers judged alongside the real ones, to check the judge (default 12)",
    )
    run.set_defaults(func=_eval_run)
    compare = eval_actions.add_parser("compare", help="per-metric changes between two result files")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.set_defaults(func=_eval_compare)
    init = eval_actions.add_parser(
        "init", help="start a private set: folders + a questions.yaml template"
    )
    init.add_argument("--set", default="private")
    init.set_defaults(func=_eval_init)
    label = eval_actions.add_parser(
        "label", help="tick the relevant chunks for each question → gold_refs"
    )
    label.add_argument("--set", default="private")
    label.add_argument(
        "--relabel", action="store_true", help="also questions that already have gold refs"
    )
    label.set_defaults(func=_eval_label)
    agree = eval_actions.add_parser(
        "agreement", help="hand-check Guardian's verdicts (50; ≥ 85 %% agreement to trust it)"
    )
    agree.add_argument("--set", default="public")
    agree.add_argument("-n", type=int, default=50)
    agree.set_defaults(func=_eval_agreement)

    from granit.bench.run import SCENARIOS

    bench = commands.add_parser(
        "bench", help="M1 benchmarks: speed, memory per phase, phase switch time (~15 min)"
    )
    bench.add_argument("scenarios", nargs="*", choices=[[], *SCENARIOS], metavar="SCENARIO")
    bench.add_argument(
        "--quick", action="store_true", help="smaller workloads to check the harness"
    )
    bench.set_defaults(func=_bench)
    return parser


def _transcribe(args: argparse.Namespace) -> int:
    from granit.ingest.audio import AudioError, AudioTranscriber, decode, timestamp
    from granit.ingest.speakers import Diarizer, speaker_count, speaker_label, turns

    transcriber = AudioTranscriber().load()  # models load once for every file
    diarizer = Diarizer().load() if Diarizer.available() else None
    failed = 0
    for path in args.files:
        try:
            audio = decode(path)
            transcript = transcriber.transcribe_audio(audio)
            if diarizer is not None:
                transcript = diarizer.label(transcript, audio)
        except (AudioError, FileNotFoundError) as exc:
            print(f"✗ {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(
            f"── {path.name}: {timestamp(transcript.duration_s)} long, "
            f"{transcript.speech_s:.0f} s speech / {transcript.silence_s:.0f} s silence, "
            f"{len(transcript.segments)} segments"
            + (f", {speaker_count(transcript)} speakers" if diarizer is not None else "")
        )
        if diarizer is not None:
            for turn in turns(transcript.segments):
                print(f"[{timestamp(turn.start)}] {speaker_label(turn.speaker, {})}: {turn.text}")
        else:
            for segment in transcript.segments:
                print(f"[{timestamp(segment.start)}] {segment.text}")
        if args.json:
            args.json.mkdir(parents=True, exist_ok=True)
            out = args.json / f"{path.stem}.transcript.json"
            out.write_text(json.dumps(transcript.to_json(), indent=2) + "\n")
            print(f"   → {out}")
    return 1 if failed else 0


def _convert(args: argparse.Namespace) -> int:
    from granit.ingest.documents import DocumentIngestor, UnsupportedDocument

    ingestor = DocumentIngestor(vision_tables=args.vision_tables).load()
    failed = 0
    for path in args.files:
        out = args.out / path.stem
        try:
            result = ingestor.ingest(path, out)
        except (UnsupportedDocument, FileNotFoundError) as exc:
            print(f"✗ {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        summary = result.summary()
        print(
            f"── {path.name}: {summary['pages']} pages, {summary['tables']} tables, "
            f"{summary['charts']} charts of {summary['pictures']} pictures → {out}/document.md"
        )
        for e in result.extractions:
            mark = "✓" if e.valid else "✗"
            print(
                f"   {mark} p{e.page} {e.kind} ({e.format}, {e.seconds:.1f} s) {'; '.join(e.errors)}"
            )
        (out / "extractions.json").write_text(
            json.dumps([e.to_json() for e in result.extractions], indent=2, ensure_ascii=False)
            + "\n"
        )
    return 1 if failed else 0


def _extract(args: argparse.Namespace) -> int:
    from granit.ingest.documents import check_format, render_pages, text_layer_lines
    from granit.ingest.vision import VisionModel, check_schema

    schema = json.loads(args.schema.read_text())
    check_schema(schema)  # a bad schema fails before any model loads
    check_format(args.file)
    pages = render_pages(args.file, args.pages)
    text = "\n".join(
        line
        for _, lines in sorted(text_layer_lines(args.file).items())[: len(pages)]
        for line in lines
    )
    extraction = VisionModel().load().extract_fields(pages, schema, text)
    print(json.dumps(extraction.data, indent=2, ensure_ascii=False))
    if extraction.missing:
        print(f"not found: {', '.join(extraction.missing)}", file=sys.stderr)
    if not extraction.valid:
        print("invalid: " + "; ".join(extraction.errors), file=sys.stderr)
    return 0 if extraction.valid else 1


def _ui(args: argparse.Namespace) -> int:
    """``streamlit run app/Home.py`` from the project root, so ``.streamlit/config.toml`` (theme, privacy) applies."""
    import os
    import subprocess

    from granit.config import PROJECT_ROOT

    env = dict(os.environ)
    if args.data:
        env["GRANIT_DATA_DIR"] = str(args.data.resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/Home.py",
        "--server.port",
        str(args.port),
    ]
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
    except KeyboardInterrupt:
        return 0


def _store(args: argparse.Namespace) -> Any:
    from granit.config import DATA_DIR
    from granit.store.db import Store

    return Store(args.data or DATA_DIR)


def _ingest(args: argparse.Namespace) -> int:
    from granit.ingest.worker import IngestWorker
    from granit.store.db import INGEST_TASKS

    store = _store(args)
    for path in args.files:
        try:
            source, created = store.add_file(path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"✗ {path}: {exc}", file=sys.stderr)
            continue
        print(
            f"{'+ queued' if created else '= already in the library'}: {source.name} (#{source.id})"
        )
    if not store.queued_count(INGEST_TASKS):
        print("nothing to process")
        return 0
    report = IngestWorker(store).run()
    print(
        f"── {len(report.done)} done, {len(report.failed)} failed in {report.seconds:.1f} s "
        f"(peak memory {report.peak_footprint_gb:.1f} GB)"
    )
    return 1 if report.failed else 0


def _verify(args: argparse.Namespace) -> int:
    import subprocess

    from granit.verify.worker import VerifyWorker

    busy = subprocess.run(
        [
            "pgrep",
            "-fl",
            r"mlx_lm[. ]server|granit\.models\.mlx_server|granit\.ingest\.worker|granit (ingest|ui)",
        ],
        capture_output=True,
        text=True,
    )
    if busy.returncode == 0:
        raise SystemExit(
            "another phase is running (Q&A or ingest); stop it first: Guardian and the Q&A model don't fit together"
        )
    store = _store(args)
    store.enqueue_verify()
    report = VerifyWorker(store).run(log=lambda line: print(line, flush=True))
    print(
        f"── {report.verdicts} verdicts in {report.seconds:.1f} s "
        f"(peak memory {report.peak_footprint_gb:.1f} GB)"
    )
    return 1 if report.failed else 0


def _search(args: argparse.Namespace) -> int:
    from granit.search.embed import Embedder
    from granit.search.hybrid import Searcher
    from granit.search.rerank import Reranker
    from granit.search.vectors import VectorIndex

    store = _store(args)
    embedder = Embedder() if args.mode != "bm25" else None
    index = VectorIndex(store, embedder.revision) if embedder else None
    reranker = Reranker() if args.mode == "hybrid+rerank" else None
    result = Searcher(store, embedder, index, reranker).search(args.query, args.mode, args.k)
    if args.json:
        print(json.dumps(result.to_json(), indent=2, ensure_ascii=False))
        return 0
    for rank, hit in enumerate(result.hits, start=1):
        preview = " ".join(hit.text.split())[:160]
        print(f"{rank}. [{hit.score:.3f}] {hit.citation} · {hit.element}\n   {preview}")
    if not result.hits:
        print("no results")
    return 0


def _sources(args: argparse.Namespace) -> int:
    store = _store(args)
    for source in store.sources():
        chunks = source.info.get("chunks", "-")
        print(
            f"#{source.id:<4} {source.status:<10} {source.kind:<8} {chunks!s:>5} chunks  {source.name}"
        )
    for job in store.jobs("failed"):
        print(f"   failed job #{job.id} ({job.task}, source #{job.source_id}): {job.error}")
    return 0


@contextmanager
def _llm_server() -> Iterator[None]:
    """Use the running Q&A server, or start one for this command (never while an ingest worker holds Phase A)."""
    import subprocess

    from granit.config import HUB_MODELS
    from granit.models.download import local_snapshot
    from granit.models.server import LLMServer, ServerConfig, stream_chat

    server = LLMServer(ServerConfig(local_snapshot(HUB_MODELS["llm"])))
    if server.healthy():
        yield
        return
    busy = subprocess.run(
        ["pgrep", "-f", r"granit\.ingest\.worker|granit ingest"], capture_output=True
    )
    if busy.returncode == 0:
        raise SystemExit("ingest is running (Phase A); ask again when it has finished")
    print("starting the Q&A model…", file=sys.stderr)
    with server:
        server.wait_ready()
        stream_chat(
            server.config.base_url,
            [{"role": "user", "content": "Hi"}],
            max_tokens=1,
            enable_thinking=False,
        )
        yield


def _ask(args: argparse.Namespace) -> int:
    from granit.reason.llm import LLMClient
    from granit.reason.qa import QA
    from granit.reason.tokens import GraniteTokens
    from granit.search.embed import Embedder
    from granit.search.hybrid import Searcher
    from granit.search.rerank import Reranker
    from granit.search.vectors import VectorIndex

    store = _store(args)
    embedder = Embedder()
    searcher = Searcher(store, embedder, VectorIndex(store, embedder.revision), Reranker())
    with _llm_server():
        qa = QA(store, searcher, LLMClient(thinking=args.thinking), GraniteTokens(), mode=args.mode)
        answer = qa.ask(
            args.question,
            on_delta=lambda d: print(d[1], end="", flush=True) if d[0] == "content" else None,
        )
    print()
    for number, hit in zip(answer.citation_numbers(), answer.cited, strict=True):
        print(f"  [{number}] {hit.citation}")
    print(f"  ({answer.latency['total_s']:.1f} s, turn #{answer.turn_id})", file=sys.stderr)
    return 0


def _speakers(args: argparse.Namespace) -> int:
    from granit.ingest.audio import Transcript, timestamp
    from granit.ingest.speakers import parse_names, speaker_infos

    store = _store(args)
    matches = [s for s in store.sources() if str(s.id) == args.source or s.name == args.source]
    if not matches or matches[0].kind != "audio" or matches[0].status != "ready":
        print(f"no ingested recording {args.source!r} (see `granit sources`)", file=sys.stderr)
        return 1
    source = matches[0]
    path = store.derived_dir(source) / "transcript.json"
    infos = speaker_infos(Transcript.from_json(json.loads(path.read_text())).segments)
    if not infos:
        print(
            f"{source.name} has no speakers: build the model (`granit models convert diarization`),"
            " then re-transcribe it",
            file=sys.stderr,
        )
        return 1
    names = store.speaker_names(source.id)
    if args.names:
        try:
            given = parse_names(args.names)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        unknown = sorted(set(given) - {i.speaker for i in infos})
        if unknown:
            print(f"{source.name} has no speaker {unknown[0]}", file=sys.stderr)
            return 1
        names = store.set_speaker_names(source, {**names, **given})
        if store.queued_count(["reindex"]):
            print(
                "search and Ask use the names after the next processing run (`granit ingest` runs it now)"
            )
    for info in infos:
        name = names.get(info.speaker, "(unnamed)")
        print(f"Speaker {info.speaker}: {name} · {info.talk_s:.0f} s in {info.turns} turns")
        for turn in info.samples:
            print(f"   [{timestamp(turn.start)}] {turn.text[:120]}")
    return 0


def _meeting(args: argparse.Namespace) -> int:
    from granit.reason.llm import LLMClient
    from granit.reason.meetings import summarize_source
    from granit.reason.tokens import GraniteTokens

    store = _store(args)
    matches = [s for s in store.sources() if str(s.id) == args.source or s.name == args.source]
    if not matches:
        print(f"no source {args.source!r} (see `granit sources`)", file=sys.stderr)
        return 1
    with _llm_server():
        summary = summarize_source(store, matches[0], LLMClient(), GraniteTokens())
    print(json.dumps(summary.data, indent=2, ensure_ascii=False))
    print(f"({summary.sections} section(s), {summary.seconds:.1f} s)", file=sys.stderr)
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    from granit.evaluate import report
    from granit.evaluate.dataset import EvalSetError, load_set
    from granit.evaluate.run import EvalConfig, Runner

    try:
        eval_set = load_set(args.set)
    except EvalSetError as exc:
        print(exc, file=sys.stderr)
        return 1
    config = EvalConfig(
        retrieval=args.retrieval,
        thinking=args.thinking,
        judge=not args.no_judge,
        fresh=args.fresh,
        controls=args.controls,
    )
    result = Runner(
        eval_set, config, log=lambda line: print(line, flush=True)
    ).run()  # progress shows live in a log
    path = report.write(result, eval_set.name if args.set in ("public", "private") else "custom")
    print("\n".join(report.summary_lines(result)))
    print(f"results: {path}")
    return 0 if result["passed"] else 1


def _eval_compare(args: argparse.Namespace) -> int:
    from granit.evaluate import report

    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))
    rows = report.compare(before, after)
    for r in rows:
        if r["delta"]:
            flag = "  REGRESSION" if r["regression"] else ""
            print(
                f"{r['metric']:<52} {report.fmt(r['before']):>7} → {report.fmt(r['after']):>7} ({r['delta']:+.3f}){flag}"
            )
    regressions = [r for r in rows if r["regression"]]
    print(f"{len(regressions)} regression(s) of 2 points or more")
    return 1 if regressions else 0


def _eval_init(args: argparse.Namespace) -> int:
    from granit.evaluate.dataset import init_set

    root = init_set(args.set)
    print(
        f"eval set in {root}: add files to files/, questions to questions.yaml, then `granit eval label --set {args.set}`"
    )
    return 0


def _eval_label(args: argparse.Namespace) -> int:
    from granit.evaluate.dataset import load_set
    from granit.evaluate.interactive import label
    from granit.evaluate.run import LIBRARIES, EvalConfig, default_qa
    from granit.store.db import Store

    eval_set = load_set(args.set)
    library = LIBRARIES / eval_set.name
    if not (library / "granit.db").is_file():
        print(
            f"the set isn't ingested yet: run `granit eval run --set {args.set} --no-judge` first",
            file=sys.stderr,
        )
        return 1
    store = Store(library)
    searcher, _ = default_qa(store, EvalConfig())
    print(f"labeled {label(eval_set, store, searcher, relabel=args.relabel)} question(s)")
    return 0


def _eval_agreement(args: argparse.Namespace) -> int:
    from granit.evaluate.dataset import load_set
    from granit.evaluate.interactive import agreement
    from granit.evaluate.report import JUDGE_AGREEMENT_MIN
    from granit.evaluate.run import LIBRARIES

    eval_set = load_set(args.set)
    result = agreement(eval_set, LIBRARIES / eval_set.name, n=args.n)
    print(
        f"{result['labeled']} labeled · agreement {result['agreement']} · Cohen's κ {result['kappa']}"
    )
    for subset in ("real", "controls"):
        part = result[subset]
        print(
            f"  {subset}: {part['n']} labeled · agreement {part['agreement']} · κ {part['kappa']}"
        )
    trusted = result["agreement"] is not None and result["agreement"] >= JUDGE_AGREEMENT_MIN
    print(
        "Guardian metrics now count toward the pass criteria."
        if trusted
        else "Guardian metrics stay informational."
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    from granit.bench.run import SCENARIOS, run, summary_lines

    report, path = run(args.scenarios or list(SCENARIOS), quick=args.quick)
    print("\n".join(summary_lines(report)))
    print(f"results: {path}")
    return 0 if all(c["within_limit"] for c in report["budget"]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
