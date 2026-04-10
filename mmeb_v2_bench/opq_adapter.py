from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .index_base import VectorIndex
from .quantizer_cache import quantizer_cache_path


@dataclass
class OPQAdapterConfig:
    total_bits: int = 96
    nbits: int = 8
    n_subquantizers: int = 0
    seed: int = 123
    verbose: bool = False


class OPQAdapterIndex(VectorIndex):
    def __init__(self, cfg: OPQAdapterConfig):
        self.cfg = cfg
        self._index = None

    def fit(self, xb: np.ndarray) -> "OPQAdapterIndex":
        return self.fit_database(xb, xb)

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "OPQAdapterIndex":
        try:
            from opq_index import OPQIndex, OPQIndexConfig
        except ImportError as exc:
            raise RuntimeError(
                "OPQ backend depends on the repository root OPQ stack and faiss. "
                "Install faiss and keep the repository root importable before using --index-backend opq."
            ) from exc

        train_xb = np.asarray(train_xb, dtype=np.float32)
        xb = np.asarray(xb, dtype=np.float32)
        opq_cfg = OPQIndexConfig(
            d=int(xb.shape[1]),
            B=int(self.cfg.total_bits),
            nbits=int(self.cfg.nbits),
            M=int(self.cfg.n_subquantizers),
            seed=int(self.cfg.seed),
            verbose=bool(self.cfg.verbose),
        )
        cache_path = None
        if quantizer_cache_dir is not None and quantizer_cache_prefix is not None:
            cache_path = quantizer_cache_path(
                quantizer_cache_dir,
                backend="opq",
                prefix_payload=quantizer_cache_prefix,
                train_xb=train_xb,
            )
        if cache_path is not None and (cache_path / "metadata.json").exists() and (cache_path / "state.npz").exists():
            print(f"[quantizer cache] backend=opq hit={cache_path}")
            opq = OPQIndex.load(cache_path)
        else:
            opq = OPQIndex(opq_cfg)
            opq.train(train_xb)
            if cache_path is not None:
                opq.save(cache_path)
                print(f"[quantizer cache] backend=opq saved={cache_path}")
        opq.add(xb)
        self._index = opq
        return self

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("OPQAdapterIndex is not fitted.")
        distances, indices = self._index.search(np.asarray(xq, dtype=np.float32), int(top_k), mode="adc")
        return -np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)
