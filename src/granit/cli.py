"""``granit`` command line. M0: ``granit models list|download|convert|smoke``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
