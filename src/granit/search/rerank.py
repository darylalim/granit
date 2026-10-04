"""Granite Embedding Reranker English R2 (PLAN.md §3.2): a cross-encoder over (question, chunk) pairs, fp16 on MPS.

Scores (0–1 after the sigmoid) are only for ordering within one question: there's no fixed "nothing relevant" threshold.
M1: ~555 ms for 30 pairs of ~300 tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from granit.config import HUB_MODELS


class Reranker:
    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model: Any = None

    def load(self) -> Reranker:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            from granit.models.download import local_snapshot
            from granit.search.embed import best_device

            self.device = self.device or best_device()
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self._model = CrossEncoder(
                str(local_snapshot(HUB_MODELS["reranker"])),
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
            )
        return self

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        self.load()
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]
