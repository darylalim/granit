"""PLAN.md §4.2: model runtimes are imported lazily, so unit tests and CI never need weights or a GPU."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import granit

HEAVY = (
    "mlx",
    "mlx_lm",
    "mlx_vlm",
    "mlx_audio",
    "torch",
    "sentence_transformers",
    "transformers",
    "docling",
)


def test_importing_granit_loads_no_model_runtime() -> None:
    modules = [m.name for m in pkgutil.walk_packages(granit.__path__, "granit.")]
    assert "granit.models.smoke" in modules
    code = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    heavy = sorted(m for m in loaded if m.split(".")[0] in HEAVY)
    assert not heavy, f"imported at module level: {heavy}"
