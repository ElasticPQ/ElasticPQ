from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .index_base import VectorIndex
from .quantizer_cache import quantizer_cache_path


@dataclass
class EPQAdapterConfig:
    total_bits: int = 96
    max_bits: int = 12
    enable_uneven_opq: bool = False
    train_rows: int = 65536
    seed: int = 123
    verbose: bool = False


class EPQAdapterIndex(VectorIndex):
    def __init__(self, cfg: EPQAdapterConfig):
        self.cfg = cfg
        self._index = None

    def fit(self, xb: np.ndarray) -> "EPQAdapterIndex":
        return self.fit_database(xb, xb)

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "EPQAdapterIndex":
        try:
            from epq_index import EPQIndex, EPQIndexConfig
        except ImportError as exc:
            raise RuntimeError(
                "EPQ backend depends on the repository root EPQ stack and faiss. "
                "Install faiss and keep the repository root importable before using --index-backend epq."
            ) from exc

        train_xb = np.asarray(train_xb, dtype=np.float32)
        xb = np.asarray(xb, dtype=np.float32)
        epq_cfg = EPQIndexConfig(
            d=int(xb.shape[1]),
            B=int(self.cfg.total_bits),
            max_bits=int(self.cfg.max_bits),
            enable_uneven_opq=bool(self.cfg.enable_uneven_opq),
            uneven_opq_niter=0,
            seed=int(self.cfg.seed),
            verbose=bool(self.cfg.verbose),
        )
        cache_path = None
        if quantizer_cache_dir is not None and quantizer_cache_prefix is not None:
            cache_path = quantizer_cache_path(
                quantizer_cache_dir,
                backend="epq",
                prefix_payload=quantizer_cache_prefix,
                train_xb=train_xb,
            )
        if cache_path is not None and (cache_path / "metadata.json").exists() and (cache_path / "state.npz").exists():
            print(f"[quantizer cache] backend=epq hit={cache_path}")
            epq = EPQIndex.load(cache_path)
        else:
            epq = EPQIndex(epq_cfg)
            epq.train(train_xb)
            if cache_path is not None:
                epq.save(cache_path)
                print(f"[quantizer cache] backend=epq saved={cache_path}")
        epq.add(xb)
        self._index = epq
        return self

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("EPQAdapterIndex is not fitted.")
        distances, indices = self._index.search(np.asarray(xq, dtype=np.float32), int(top_k), mode="adc")
        return -np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)
