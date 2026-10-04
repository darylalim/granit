"""The vector matrix for Phase B (PLAN.md §2.3): fp16 on MPS, loaded when Phase B starts and reloaded when the corpus changes.

M1 measured top-50 over 150K vectors at 1.0 ms on MPS (fp16) vs 182 ms with NumPy fp16 (no BLAS path). Without MPS (CI,
tests) the matrix is fp32 on CPU. Only vectors from the current embedding revision are loaded, so after a model upgrade the
outdated rows are simply ignored until they're re-embedded.
"""

from __future__ import annotations

from typing import Any

from granit.config import EMBEDDING_DIM
from granit.store.db import Store


class VectorIndex:
    def __init__(self, store: Store, revision: str, device: str | None = None) -> None:
        self.store = store
        self.revision = revision
        self.device = device
        self._version: int | None = None
        self._ids: Any = None
        self._matrix: Any = None

    def __len__(self) -> int:
        return 0 if self._ids is None else len(self._ids)

    def refresh(self) -> bool:
        """Reload if ingest or deletion changed the corpus since the last load. Returns True if it reloaded."""
        version = self.store.corpus_version()
        if version == self._version:
            return False
        self._load()
        self._version = version
        return True

    def _load(self) -> None:
        import numpy as np
        import torch

        from granit.search.embed import best_device

        rows = self.store.conn.execute(
            "SELECT chunk_id, embedding FROM chunk_vectors WHERE model_revision = ? ORDER BY chunk_id",
            (self.revision,),
        ).fetchall()
        self.device = self.device or best_device()
        self._ids = np.array([r[0] for r in rows], dtype=np.int64)
        if not rows:
            self._matrix = None
            return
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(
            -1, EMBEDDING_DIM
        )
        if self.device == "mps":
            self._matrix = torch.from_numpy(matrix.copy()).to("mps")
        else:
            self._matrix = torch.from_numpy(matrix.astype(np.float32))

    def search(self, query: Any, k: int) -> list[tuple[int, float]]:
        """Top-``k`` (chunk_id, cosine similarity) for a normalized query vector."""
        import torch

        self.refresh()
        if self._matrix is None or k <= 0:
            return []
        q = torch.from_numpy(query.astype("float16" if self.device == "mps" else "float32")).to(
            self._matrix.device
        )
        scores = self._matrix @ q
        top = torch.topk(scores, min(k, scores.shape[0]))
        return [
            (int(self._ids[i]), float(s))
            for s, i in zip(
                top.values.float().cpu().tolist(), top.indices.cpu().tolist(), strict=True
            )
        ]
