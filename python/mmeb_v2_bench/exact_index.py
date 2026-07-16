from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .index_base import VectorIndex


@dataclass
class ExactIndexConfig:
    normalize_inputs: bool = False


class ExactCosineIndex(VectorIndex):
    def __init__(self, cfg: ExactIndexConfig | None = None):
        self.cfg = cfg or ExactIndexConfig()
        self.xb: np.ndarray | None = None

    def fit(self, xb: np.ndarray) -> "ExactCosineIndex":
        self.xb = np.asarray(xb, dtype=np.float32)
        return self

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "ExactCosineIndex":
        del quantizer_cache_dir, quantizer_cache_prefix
        del train_xb
        return self.fit(xb)

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.xb is None:
            raise RuntimeError("ExactCosineIndex is not fitted.")
        scores = np.asarray(xq, dtype=np.float32) @ self.xb.T
        top_k = min(int(top_k), scores.shape[1])
        idx = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        score_top = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-score_top, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        score_top = np.take_along_axis(score_top, order, axis=1)
        return score_top, idx
