"""Download Hub models at their pinned revisions into the standard HF cache (PLAN.md §3)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from granit.config import HUB_MODELS, HubModel


def download(model: HubModel) -> Path:
    """Fetch ``model`` at its pinned revision (a no-op for files already cached) and return the snapshot."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model.repo_id,
        revision=model.revision,
        ignore_patterns=list(model.ignore_patterns) or None,
    )
    return Path(path)


def download_all(keys: Iterable[str]) -> dict[str, Path]:
    return {key: download(HUB_MODELS[key]) for key in keys}


def local_snapshot(model: HubModel) -> Path:
    """The pinned snapshot path, or a clear error telling the user how to download it. Never hits the network."""
    path = model.cached_snapshot()
    if path is None:
        raise FileNotFoundError(
            f"{model.repo_id}@{model.revision[:10]} is not downloaded. "
            f"Run: uv run granit models download {model.key}"
        )
    return path


def delete_cached(model: HubModel) -> int:
    """Remove the pinned revision from the HF cache. Returns the bytes freed (0 if it wasn't cached)."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    if not any(
        repo.repo_id == model.repo_id
        and any(r.commit_hash == model.revision for r in repo.revisions)
        for repo in cache.repos
    ):
        return 0
    strategy = cache.delete_revisions(model.revision)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed
