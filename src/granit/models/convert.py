"""Build local MLX quantized copies of Hub models (Guardian 4.1 8B → 8-bit), PLAN.md §2.4 / §3."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from granit.config import HUB_MODELS, SOURCE_RECORD, LocalModel
from granit.models.download import local_snapshot

# Quantized output is about half of a bf16 source; keep headroom for macOS on a nearly full disk.
MIN_FREE_BYTES_MARGIN = 2 * 1024**3


def source_record(model: LocalModel) -> dict[str, object]:
    source = HUB_MODELS[model.source]
    return {
        "source_repo": source.repo_id,
        "source_revision": source.revision,
        "q_bits": model.q_bits,
        "tool": f"mlx-lm {version('mlx-lm')}",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def check_disk_space(target: Path, needed_bytes: int) -> None:
    probe = target if target.exists() else target.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed_bytes + MIN_FREE_BYTES_MARGIN:
        raise OSError(
            f"Not enough disk space to build {target.name}: {free / 1e9:.1f} GB free, "
            f"needs about {(needed_bytes + MIN_FREE_BYTES_MARGIN) / 1e9:.1f} GB."
        )


def build(model: LocalModel) -> Path:
    """Quantize the pinned bf16 source into ``model.path`` and record where it came from."""
    if model.is_built():
        return model.path
    if model.path.exists():
        raise FileExistsError(
            f"{model.path} exists but is incomplete (no {SOURCE_RECORD}); delete it and rebuild."
        )
    source_path = local_snapshot(HUB_MODELS[model.source])
    check_disk_space(model.path, int(model.size_gb * 1e9))
    model.path.parent.mkdir(parents=True, exist_ok=True)

    from mlx_lm import convert

    convert(
        hf_path=str(source_path),
        mlx_path=str(model.path),
        quantize=True,
        q_bits=model.q_bits,
    )
    # Written last: its presence marks a complete build.
    (model.path / SOURCE_RECORD).write_text(json.dumps(source_record(model), indent=2) + "\n")
    return model.path
