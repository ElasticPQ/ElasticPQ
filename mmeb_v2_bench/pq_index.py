from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .index_base import VectorIndex
from .quantizer_cache import quantizer_cache_path


@dataclass
class ProductQuantizerConfig:
    n_subquantizers: int = 32
    bits_per_subquantizer: int = 8
    train_size: int = 4096
    kmeans_iters: int = 20
    seed: int = 123


def _kmeans_numpy(x: np.ndarray, k: int, *, n_iter: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot train kmeans with zero rows")
    if k >= n:
        return np.ascontiguousarray(x.copy(), dtype=np.float32)

    rng = np.random.default_rng(seed)
    centroids = np.ascontiguousarray(x[rng.choice(n, size=k, replace=False)], dtype=np.float32)
    for _ in range(n_iter):
        dist = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(centroids * centroids, axis=1)[None, :]
            - 2.0 * (x @ centroids.T)
        )
        labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                new_centroids[idx] = x[mask].mean(axis=0)
            else:
                new_centroids[idx] = x[rng.integers(0, n)]
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.ascontiguousarray(centroids, dtype=np.float32)


class ProductQuantizerIndex(VectorIndex):
    def __init__(self, cfg: ProductQuantizerConfig):
        self.cfg = cfg
        self.codebooks: list[np.ndarray] = []
        self.codes: np.ndarray | None = None
        self.ksub: int = 0
        self.subdim: int = 0
        self.padded_dim: int = 0

    def _pad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[1] == self.padded_dim:
            return x
        out = np.zeros((x.shape[0], self.padded_dim), dtype=np.float32)
        out[:, : x.shape[1]] = x
        return out

    def fit(self, xb: np.ndarray) -> "ProductQuantizerIndex":
        return self.fit_database(xb, xb)

    def _train_codebooks(self, train_xb: np.ndarray) -> None:
        train_xb = np.asarray(train_xb, dtype=np.float32)
        dim = int(train_xb.shape[1])
        self.subdim = (dim + self.cfg.n_subquantizers - 1) // self.cfg.n_subquantizers
        self.padded_dim = self.subdim * self.cfg.n_subquantizers
        xt_pad = self._pad(train_xb)

        rng = np.random.default_rng(self.cfg.seed)
        n_train = min(int(self.cfg.train_size), xt_pad.shape[0])
        train_idx = rng.choice(xt_pad.shape[0], size=n_train, replace=False)
        xt = xt_pad[train_idx]

        self.ksub = min(1 << int(self.cfg.bits_per_subquantizer), max(1, n_train))
        self.codebooks = []
        for m in range(self.cfg.n_subquantizers):
            start = m * self.subdim
            stop = start + self.subdim
            subv = xt[:, start:stop]
            codebook = _kmeans_numpy(
                subv,
                self.ksub,
                n_iter=int(self.cfg.kmeans_iters),
                seed=int(self.cfg.seed + m),
            )
            self.codebooks.append(codebook)
        self.codes = None

    def _encode_database(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb, dtype=np.float32)
        xb_pad = self._pad(xb)

        codes = np.empty((xb_pad.shape[0], self.cfg.n_subquantizers), dtype=np.int32)
        for m, codebook in enumerate(self.codebooks):
            start = m * self.subdim
            stop = start + self.subdim
            subv = xb_pad[:, start:stop]
            dist = (
                np.sum(subv * subv, axis=1, keepdims=True)
                + np.sum(codebook * codebook, axis=1)[None, :]
                - 2.0 * (subv @ codebook.T)
            )
            codes[:, m] = np.argmin(dist, axis=1)
        self.codes = np.ascontiguousarray(codes, dtype=np.int32)

    def save(self, path: str | Path) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": {
                "n_subquantizers": int(self.cfg.n_subquantizers),
                "bits_per_subquantizer": int(self.cfg.bits_per_subquantizer),
                "train_size": int(self.cfg.train_size),
                "kmeans_iters": int(self.cfg.kmeans_iters),
                "seed": int(self.cfg.seed),
            },
            "state": {
                "ksub": int(self.ksub),
                "subdim": int(self.subdim),
                "padded_dim": int(self.padded_dim),
                "n_codebooks": int(len(self.codebooks)),
            },
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        arrays = {
            "codebooks": np.asarray(self.codebooks, dtype=np.float32),
        }
        np.savez_compressed(out_dir / "state.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ProductQuantizerIndex":
        in_dir = Path(path)
        with (in_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cfg = ProductQuantizerConfig(**meta["config"])
        obj = cls(cfg)
        state = meta["state"]
        obj.ksub = int(state["ksub"])
        obj.subdim = int(state["subdim"])
        obj.padded_dim = int(state["padded_dim"])
        state_npz = np.load(in_dir / "state.npz", allow_pickle=False)
        obj.codebooks = [np.ascontiguousarray(cb, dtype=np.float32) for cb in state_npz["codebooks"]]
        obj.codes = None
        return obj

    def fit_database(
        self,
        train_xb: np.ndarray,
        xb: np.ndarray,
        *,
        quantizer_cache_dir: str | Path | None = None,
        quantizer_cache_prefix: object | None = None,
    ) -> "ProductQuantizerIndex":
        train_xb = np.asarray(train_xb, dtype=np.float32)
        xb = np.asarray(xb, dtype=np.float32)
        cache_path = None
        if quantizer_cache_dir is not None and quantizer_cache_prefix is not None:
            cache_path = quantizer_cache_path(
                quantizer_cache_dir,
                backend="pq",
                prefix_payload=quantizer_cache_prefix,
                train_xb=train_xb,
            )
        if cache_path is not None and (cache_path / "metadata.json").exists() and (cache_path / "state.npz").exists():
            print(f"[quantizer cache] backend=pq hit={cache_path}")
            loaded = self.load(cache_path)
            self.codebooks = loaded.codebooks
            self.codes = loaded.codes
            self.ksub = loaded.ksub
            self.subdim = loaded.subdim
            self.padded_dim = loaded.padded_dim
        else:
            self._train_codebooks(train_xb)
            if cache_path is not None:
                self.save(cache_path)
                print(f"[quantizer cache] backend=pq saved={cache_path}")
        self._encode_database(xb)
        return self

    def search(self, xq: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.codes is None:
            raise RuntimeError("ProductQuantizerIndex is not fitted.")
        xq_pad = self._pad(np.asarray(xq, dtype=np.float32))
        nb = self.codes.shape[0]
        top_k = min(int(top_k), nb)
        all_scores = np.empty((xq_pad.shape[0], top_k), dtype=np.float32)
        all_indices = np.empty((xq_pad.shape[0], top_k), dtype=np.int64)

        for qi, query in enumerate(xq_pad):
            dist = np.zeros((nb,), dtype=np.float32)
            for m, codebook in enumerate(self.codebooks):
                start = m * self.subdim
                stop = start + self.subdim
                qsub = query[start:stop]
                lut = (
                    np.sum(codebook * codebook, axis=1)
                    + np.sum(qsub * qsub)
                    - 2.0 * (codebook @ qsub)
                )
                dist += lut[self.codes[:, m]]
            idx = np.argpartition(dist, kth=top_k - 1)[:top_k]
            score = -dist[idx]
            order = np.argsort(-score)
            all_scores[qi] = score[order]
            all_indices[qi] = idx[order]
        return all_scores, all_indices
