from __future__ import annotations

from typing import Protocol
from pathlib import Path

import numpy as np


class VectorIndex(Protocol):
    def fit(self, xb: np.ndarray) -> "VectorIndex":
        ...

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "VectorIndex":
        ...

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        ...
