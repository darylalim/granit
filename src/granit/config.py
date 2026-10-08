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
# Phase A worker: the same limit (M1 measured Phase A peaking at 15.6 GB, mostly freed buffers MLX kept cached).
INGEST_MLX_CACHE_LIMIT_GB = 1.0
# Documents: tables come from Granite-Docling, and Vision reads only the tables Docling leaves empty (it finds some tables
# but emits no cells). Decided in M7 (PLAN.md §4.9 "Tables: Docling vs Vision"): this scored 0.997 cell F1 vs 0.925 for
# Vision on every table, at ~1/8 of the time. True = Vision reads every table (the per-upload "Accurate tables").
DOCUMENT_VISION_TABLES = False
# Retrieval (PLAN.md §3.2): BM25 top-50 + trigram ID hits + vector top-50 → RRF (k = 60) → rerank top 30 → top 8 to the LLM.
SEARCH_CANDIDATES = 50
RRF_K = 60
RERANK_CANDIDATES = 30
SEARCH_TOP_K = 8
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
    q_bits: int | None  # None: converted without quantization
    phase: str
    size_gb: float
    # How it's built (models/convert.py): "mlx-lm" (convert + quantize) or "nemotron-diarization" (mlx-audio's NeMo converter)
    builder: str = "mlx-lm"

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
        HubModel(
            key="diarization-source",
            repo_id="nvidia/Nemotron-3-Diarization",
            revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
            license="openmdw-1.1",
            phase="convert",
            runtime="NeMo source for the local fp32 mlx-audio build (speakers, PLAN.md §3.7)",
            size_gb=0.20,
            # mlx-audio converts from the .nemo archive; the rest are other runtimes' copies and demo media.
            ignore_patterns=("model.safetensors", "*.gguf", "*.mp4", "*.gif"),
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
        LocalModel(
            key="diarization",
            source="diarization-source",
            path=MODELS_DIR / "nemotron-3-diarization-mlx",
            q_bits=None,  # fp32, as measured in the spike (PLAN.md §3.7)
            phase="A",
            size_gb=0.40,
            builder="nemotron-diarization",
        ),
    )
}

# Downloaded for runtime use. Convert sources (Guardian, diarization) are only needed while building their local copies.
RUNTIME_HUB_MODELS: tuple[str, ...] = tuple(
    k for k, m in HUB_MODELS.items() if m.phase != "convert"
)
