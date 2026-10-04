"""``granit`` command line. M0: ``granit models list|download|convert|smoke``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

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
        print(f"→ building {model.path.name} ({model.q_bits}-bit) from {model.source}", flush=True)
        print(f"  {build(model)}")
        if args.delete_source:
            freed = delete_cached(HUB_MODELS[model.source])
            print(f"  deleted bf16 source from the HF cache ({freed / 1e9:.1f} GB freed)")
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

    convert = actions.add_parser("convert", help="build local quantized models (Guardian q8)")
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
    from granit.ingest.audio import AudioError, AudioTranscriber, timestamp

    transcriber = AudioTranscriber().load()  # models load once for every file
    failed = 0
    for path in args.files:
        try:
            transcript = transcriber.transcribe(path)
        except (AudioError, FileNotFoundError) as exc:
            print(f"✗ {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(
            f"── {path.name}: {timestamp(transcript.duration_s)} long, "
            f"{transcript.speech_s:.0f} s speech / {transcript.silence_s:.0f} s silence, "
            f"{len(transcript.segments)} segments"
        )
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
    from granit.ingest.documents import check_format, render_pages
    from granit.ingest.vision import VisionModel, check_schema

    schema = json.loads(args.schema.read_text())
    check_schema(schema)  # a bad schema fails before any model loads
    check_format(args.file)
    extraction = VisionModel().load().extract_fields(render_pages(args.file, args.pages), schema)
    print(json.dumps(extraction.data, indent=2, ensure_ascii=False))
    if extraction.missing:
        print(f"not found: {', '.join(extraction.missing)}", file=sys.stderr)
    if not extraction.valid:
        print("invalid: " + "; ".join(extraction.errors), file=sys.stderr)
    return 0 if extraction.valid else 1


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
