"""Model IDs + pinned revisions, paths and context sizes (PLAN.md §3).

Standard library only: the Claude Code hooks (run with the system ``python3``) import this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRANIT_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("GRANIT_MODELS_DIR", PROJECT_ROOT / "models"))
DB_PATH = DATA_DIR / "granit.db"

# One LLM request (prompt + output) is capped at 16K tokens (PLAN.md §3.3, decided after M1): it keeps Phase B
# around 20 GB instead of 23.5 GB at 30K, and a 30K prompt alone takes ~2.3 min to read. Longer inputs (meetings over
# about an hour) are summarized in sections. mlx_lm.server doesn't enforce this; the prompt builder (M5) must.
LLM_CONTEXT_TOKENS = 16_384
LLM_MAX_OUTPUT_TOKENS = 2_048
LLM_PROMPT_BUDGET_TOKENS = LLM_CONTEXT_TOKENS - LLM_MAX_OUTPUT_TOKENS
# Phase B: mlx_lm.server on localhost only (never reachable from the network).
LLM_HOST = "127.0.0.1"
LLM_PORT = int(os.environ.get("GRANIT_LLM_PORT", "8765"))
# Phase B memory limits (measured in M1, PLAN.md §3.3). mlx-lm's defaults keep up to 10 prompt KV caches with no
# byte limit, and MLX keeps freed KV-growth buffers cached; either can push Phase B past the GPU limit.
LLM_PROMPT_CACHE_BYTES = "2GB"
LLM_MLX_CACHE_LIMIT_GB = 1.0
# Documents (M3): Granite-Docling's own tables matched Vision's exactly on the card's table (96/96 values) in 3.4 s for
# the page vs 29 s for Vision on one table, so Vision re-extracts tables only when this is on. M7 (table cell F1) decides.
DOCUMENT_VISION_TABLES = False
GUARDIAN_CONTEXT_TOKENS = 8_192
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class HubModel:
    """A model downloaded from the Hugging Face Hub at a pinned commit."""

    key: str
    repo_id: str
    revision: str
    license: str
    phase: str
    runtime: str
    size_gb: float
    ignore_patterns: tuple[str, ...] = field(default=())

    def cached_snapshot(self) -> Path | None:
        """The pinned snapshot in the local HF cache, or None if it isn't downloaded (no network)."""
        snapshot = hf_hub_cache() / f"models--{self.repo_id.replace('/', '--')}" / "snapshots"
        path = snapshot / self.revision
        return path if path.is_dir() and any(path.iterdir()) else None


def hf_hub_cache() -> Path:
    """Resolve the HF hub cache the way huggingface_hub does (HF_HUB_CACHE > HF_HOME > default)."""
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache).expanduser()
    home = (
        os.environ.get("HF_HOME")
        or Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"
    )
    return Path(home).expanduser() / "hub"


@dataclass(frozen=True)
class LocalModel:
    """A model converted locally from a pinned Hub source (lives in ``MODELS_DIR``, never committed)."""

    key: str
    source: str
    path: Path
    q_bits: int
    phase: str
    size_gb: float

    def is_built(self) -> bool:
        return (self.path / "config.json").is_file() and (self.path / SOURCE_RECORD).is_file()


# Written next to a locally converted model: which Hub repo + revision it was built from, and how.
SOURCE_RECORD = "granit_source.json"


HUB_MODELS: dict[str, HubModel] = {
    m.key: m
    for m in (
        HubModel(
            key="speech",
            repo_id="ibm-granite/granite-speech-5.0-470m-turboctc",
            revision="947f59af40db9791170a0628cf0f3f4812d720f1",
            license="apache-2.0",
            phase="A",
            runtime="mlx-audio",
            size_gb=0.95,
        ),
        HubModel(
            key="vad",
            repo_id="mlx-community/silero-vad-v6",
            revision="2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
            license="mit",
            phase="A",
            runtime="mlx-audio",
            size_gb=0.001,
        ),
        HubModel(
            key="docling",
            repo_id="ibm-granite/granite-docling-258M-mlx",
            revision="e9939db25d2f296c8678d0491c4609a8c596c50a",
            license="apache-2.0",
            phase="A",
            runtime="docling → mlx-vlm",
            size_gb=0.64,
        ),
        HubModel(
            key="vision",
            repo_id="ibm-granite/granite-vision-4.1-4b",
            revision="37d591f06319e8f1638b5adcf58bdf50e0f84f7a",
            license="apache-2.0",
            phase="A",
            runtime="mlx-vlm",
            size_gb=8.01,
            # Benchmark charts from the card; chart.jpg / table.png / invoice.png are kept for golden tests.
            ignore_patterns=("bench_*.png",),
        ),
        HubModel(
            key="llm",
            repo_id="ibm-granite/granite-4.2-8b-q8-mlx",
            revision="a60999a162a4799c52a7bd6ec65c780d6138cfcf",
            license="apache-2.0",
            phase="B",
            runtime="mlx-lm server",
            size_gb=9.35,
        ),
        HubModel(
            key="embedding",
            repo_id="ibm-granite/granite-embedding-english-r2",
            revision="47ea694b257b703fee9253d75c2b1f2985180498",
            license="apache-2.0",
            phase="A+B",
            runtime="sentence-transformers (mps)",
            size_gb=0.30,
            # Same weights as model.safetensors.
            ignore_patterns=("pytorch_model.bin",),
        ),
        HubModel(
            key="reranker",
            repo_id="ibm-granite/granite-embedding-reranker-english-r2",
            revision="d09d3d6971b689bf9c23839e45a470874d46e13a",
            license="apache-2.0",
            phase="B",
            runtime="sentence-transformers CrossEncoder (mps, fp16)",
            size_gb=0.60,
        ),
        HubModel(
            key="guardian-source",
            repo_id="ibm-granite/granite-guardian-4.1-8b",
            revision="ab01ccca5dcfb80246369a086a4a87a29198f5af",
            license="apache-2.0",
            phase="convert",
            runtime="bf16 source for the local q8 build",
            size_gb=16.77,
        ),
    )
}

LOCAL_MODELS: dict[str, LocalModel] = {
    m.key: m
    for m in (
        LocalModel(
            key="guardian",
            source="guardian-source",
            path=MODELS_DIR / "granite-guardian-4.1-8b-q8-mlx",
            q_bits=8,
            phase="C",
            size_gb=8.9,
        ),
    )
}

# Downloaded for runtime use. The Guardian source is only needed while building its q8 copy.
RUNTIME_HUB_MODELS: tuple[str, ...] = tuple(
    k for k, m in HUB_MODELS.items() if m.phase != "convert"
)
