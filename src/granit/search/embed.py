"""Granite Embedding English R2 (PLAN.md §3.2): normalized, float16, revision-tagged vectors.

The model's output isn't normalized (no Normalize module), so every call passes ``normalize_embeddings=True``. No query or
passage prefixes. Runs on MPS in fp16 (sentence-transformers, the reference implementation); CPU in fp32 when MPS isn't
available. Torch is imported lazily (PLAN.md §4.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from granit.config import EMBEDDING_DIM, HUB_MODELS


def best_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Embedder:
    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model: Any = None

    @property
    def revision(self) -> str:
        return HUB_MODELS["embedding"].revision

    def load(self) -> Embedder:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            from granit.models.download import local_snapshot

            self.device = self.device or best_device()
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self._model = SentenceTransformer(
                str(local_snapshot(HUB_MODELS["embedding"])),
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
            )
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        """``(len(texts), 768)`` float16 NumPy array, each row of length 1."""
        import numpy as np

        self.load()
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float16)
        vectors = self._model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype(np.float16)

    def encode_query(self, text: str) -> Any:
        return self.encode([text])[0]
