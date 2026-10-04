"""H4 · PreToolUse Bash: one phase at a time (PLAN.md §2.1, §4.1).

If a command would load models while an ``mlx_lm.server`` or granit worker is already running, deny it:
two phases at once would exceed the ~21 GB the GPU may use on a 32 GB Mac.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _hooklib import block, python_module, read_input, run, split_commands, tool_input

MODEL_CLI_COMMANDS = {"ingest", "eval", "verify", "bench", "transcribe", "convert", "extract"}
MODEL_CLI_MODEL_ACTIONS = {"smoke", "convert"}
MLX_PREFIXES = ("mlx_lm", "mlx_vlm", "mlx_audio")
# Extended regex for `pgrep -f`: processes that hold a phase's models in memory.
RUNNING_PHASE_PATTERN = (
    r"mlx_lm[. ]server|granit\.ingest\.worker|granit\.verify\.worker|granit\.models\.smoke|granit\.bench\.workers|granit\.models\.mlx_server"
    r"|granit (ingest|eval|verify|bench|transcribe|convert|extract)|granit models (smoke|convert)"
)


def loads_models(argv: list[str]) -> bool:
    if not argv:
        return False
    cmd = Path(argv[0]).name
    args = argv[1:]
    module = python_module(argv)
    if module:
        args = argv[3:]  # what follows `python -m <module>`
    if cmd == "pytest" or module == "pytest":
        return _selects_model_marker(args)
    if cmd == "granit" or module == "granit.cli":
        rest = args
        if rest[:1] and rest[0] in MODEL_CLI_COMMANDS:
            return True
        return rest[:1] == ["models"] and rest[1:2] != [] and rest[1] in MODEL_CLI_MODEL_ACTIONS
    if module and (
        module.split(".")[0] in MLX_PREFIXES
        or module.startswith(("granit.models.smoke", "granit.models.mlx_server", "granit.bench"))
    ):
        return True
    return cmd.split(".")[0] in MLX_PREFIXES


def _selects_model_marker(args: list[str]) -> bool:
    for i, a in enumerate(args):
        expr = None
        if a == "-m" and i + 1 < len(args):
            expr = args[i + 1]
        elif a.startswith("-m") and len(a) > 2:
            expr = a[2:]
        if expr is not None and "model" in expr and not expr.strip().startswith("not "):
            return True
    return False


def running_phases() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-fl", RUNNING_PHASE_PATTERN], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def main() -> None:
    command = tool_input(read_input()).get("command")
    if not isinstance(command, str):
        return
    if not any(loads_models(argv) for argv in split_commands(command)):
        return
    running = running_phases()
    if running:
        first = running[0][:160]
        block(
            "Blocked by granit hook H4: stop the running phase first; two phases at once would exceed "
            f"the ~21 GB GPU limit. Running: {first}"
        )


if __name__ == "__main__":
    run(main)
