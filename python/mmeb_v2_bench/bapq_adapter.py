from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .index_base import VectorIndex
from .quantizer_cache import quantizer_cache_path


@dataclass
class BAPQAdapterConfig:
    total_bits: int = 96
    subspace_dim: int = 4
    bmax: int = 12
    max_train_rows: int = 200000
    pca_max_train_rows: int = 200000
    km_niter: int = 20
    km_nredo: int = 1
    seed: int = 123
    verbose: bool = False


class BAPQAdapterIndex(VectorIndex):
    def __init__(self, cfg: BAPQAdapterConfig):
        self.cfg = cfg
        self._index = None

    def fit(self, xb: np.ndarray) -> "BAPQAdapterIndex":
        return self.fit_database(xb, xb)

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "BAPQAdapterIndex":
        try:
            from bapq_index import BAPQIndex, BAPQIndexConfig
        except ImportError as exc:
            raise RuntimeError(
                "BAPQ backend depends on the repository root BAPQ stack and faiss. "
                "Install faiss and keep the repository root importable before using --index-backend bapq."
            ) from exc

        train_xb = np.asarray(train_xb, dtype=np.float32)
        xb = np.asarray(xb, dtype=np.float32)
        dim = int(xb.shape[1])
        cache_path = None
        if quantizer_cache_dir is not None and quantizer_cache_prefix is not None:
            cache_path = quantizer_cache_path(
                quantizer_cache_dir,
                backend="bapq",
                prefix_payload=quantizer_cache_prefix,
                train_xb=train_xb,
            )
        if cache_path is not None and (cache_path / "metadata.json").exists() and (cache_path / "state.npz").exists() and (cache_path / "pca.faiss").exists():
            print(f"[quantizer cache] backend=bapq hit={cache_path}")
            self._index = BAPQIndex.load(cache_path)
        else:
            self._index = BAPQIndex(
                BAPQIndexConfig(
                    d=dim,
                    B=int(self.cfg.total_bits),
                    q=int(self.cfg.subspace_dim),
                    bmax=int(self.cfg.bmax),
                    seed=int(self.cfg.seed),
                    max_train_rows=int(self.cfg.max_train_rows),
                    pca_max_train_rows=int(self.cfg.pca_max_train_rows),
                    km_niter=int(self.cfg.km_niter),
                    km_nredo=int(self.cfg.km_nredo),
                )
            )
            self._index.train(train_xb)
            if cache_path is not None:
                self._index.save(cache_path)
                print(f"[quantizer cache] backend=bapq saved={cache_path}")
        self._index.add(xb)
        return self

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("BAPQAdapterIndex is not fitted.")
        distances, indices = self._index.search(np.asarray(xq, dtype=np.float32), int(top_k), mode="adc")
        return -np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)
