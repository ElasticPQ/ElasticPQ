from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import faiss
import numpy as np

from bapq import BAPQ, BAPQIndexADC

QueryMode = Literal["adc", "sdc"]


@dataclass
class BAPQIndexConfig:
    d: int
    B: int

    q: int = 4
    bmax: int = 12

    max_train_rows: int = 200000
    pca_max_train_rows: int = 200000

    km_niter: int = 20
    km_nredo: int = 1

    seed: int = 123
    verbose: bool = False

    lut_chunk: int = 4096
    query_batch: int = 32
    db_chunk: int = 200000

    name: str = "bapq"


class BAPQIndex:
    def __init__(self, cfg: BAPQIndexConfig):
        self.cfg = cfg
        self.name = str(cfg.name)
        self._index_adc: BAPQIndexADC | None = None
        self.is_trained: bool = False
        self.nb: int = 0

    @staticmethod
    def _resolve_num_subspaces(d: int, q: int) -> int:
        d = int(d)
        q = int(q)
        if d <= 0 or q <= 0:
            raise ValueError(f"d and q must be positive, got d={d}, q={q}")
        return (d + q - 1) // q

    @staticmethod
    def _activate_index(index_adc: BAPQIndexADC) -> None:
        bapq = index_adc.bapq
        if bapq.groups is None or bapq.codebooks is None or bapq.nbits_per_group is None:
            raise RuntimeError("BAPQ model is incomplete")
        index_adc._groups_all = [list(map(int, g)) for g in bapq.groups]
        index_adc._codebooks_all = [np.ascontiguousarray(c, dtype=np.float32) for c in bapq.codebooks]
        index_adc._nbits_all = [int(b) for b in bapq.nbits_per_group]
        active_gi = [gi for gi, b in enumerate(index_adc._nbits_all) if int(b) > 0]
        if not active_gi:
            active_gi = [0]
        index_adc._active_gi = active_gi
        index_adc._groups = [index_adc._groups_all[gi] for gi in active_gi]
        index_adc._codebooks = [index_adc._codebooks_all[gi] for gi in active_gi]
        index_adc._M = int(len(active_gi))
        index_adc.codes_db = None
        index_adc.nb = 0

    @staticmethod
    def _apply_transform(vt, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if hasattr(vt, "apply"):
            return np.ascontiguousarray(vt.apply(x), dtype=np.float32)
        return np.ascontiguousarray(vt.apply_py(x), dtype=np.float32)

    def _build_bapq(self) -> BAPQ:
        return BAPQ(
            d=int(self.cfg.d),
            M=self._resolve_num_subspaces(int(self.cfg.d), int(self.cfg.q)),
            B=int(self.cfg.B),
            bmax=int(self.cfg.bmax),
            seed=int(self.cfg.seed),
            max_train_rows=int(self.cfg.max_train_rows),
            pca_max_train_rows=int(self.cfg.pca_max_train_rows),
            km_niter=int(self.cfg.km_niter),
            km_nredo=int(self.cfg.km_nredo),
        )

    def _require_index(self) -> BAPQIndexADC:
        if self._index_adc is None:
            raise RuntimeError("BAPQIndex is not trained.")
        return self._index_adc

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        if xt.ndim != 2 or xt.shape[1] != int(self.cfg.d):
            raise ValueError(f"xt must have shape (n, {self.cfg.d}), got {xt.shape}")
        index_adc = BAPQIndexADC(
            self._build_bapq(),
            lut_chunk=int(self.cfg.lut_chunk),
            query_batch=int(self.cfg.query_batch),
            db_chunk=int(self.cfg.db_chunk),
        )
        index_adc.train(xt)
        self._index_adc = index_adc
        self.is_trained = True
        self.nb = 0

    def fit(self, xt: np.ndarray) -> "BAPQIndex":
        self.train(xt)
        return self

    def add(self, xb: np.ndarray) -> None:
        index_adc = self._require_index()
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != int(self.cfg.d):
            raise ValueError(f"xb must have shape (n, {self.cfg.d}), got {xb.shape}")
        index_adc.add(xb)
        self.nb = int(index_adc.nb)

    def fit_add(self, xt: np.ndarray, xb: np.ndarray) -> "BAPQIndex":
        self.train(xt)
        self.add(xb)
        return self

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        if str(mode).lower() != "adc":
            raise NotImplementedError("BAPQIndex supports mode='adc' only.")
        index_adc = self._require_index()
        distances, indices = index_adc.search(np.ascontiguousarray(xq, dtype=np.float32), int(k))
        return np.ascontiguousarray(distances, dtype=np.float32), np.ascontiguousarray(indices, dtype=np.int64)

    def _meta_dict(self) -> Dict[str, Any]:
        index_adc = self._require_index()
        bapq = index_adc.bapq
        if bapq.groups is None or bapq.nbits_per_group is None or bapq.codebooks is None:
            raise RuntimeError("BAPQ model is incomplete")
        return {
            "config": {
                "d": int(self.cfg.d),
                "B": int(self.cfg.B),
                "q": int(self.cfg.q),
                "bmax": int(self.cfg.bmax),
                "max_train_rows": int(self.cfg.max_train_rows),
                "pca_max_train_rows": int(self.cfg.pca_max_train_rows),
                "km_niter": int(self.cfg.km_niter),
                "km_nredo": int(self.cfg.km_nredo),
                "seed": int(self.cfg.seed),
                "verbose": bool(self.cfg.verbose),
                "lut_chunk": int(self.cfg.lut_chunk),
                "query_batch": int(self.cfg.query_batch),
                "db_chunk": int(self.cfg.db_chunk),
                "name": str(self.cfg.name),
            },
            "state": {
                "is_trained": bool(self.is_trained),
                "nb": int(self.nb),
                "M": int(bapq.M),
                "groups": [list(map(int, g)) for g in bapq.groups],
                "nbits_per_group": [int(b) for b in bapq.nbits_per_group],
                "n_codebooks": int(len(bapq.codebooks)),
            },
        }

    def save(self, path: str | Path) -> None:
        index_adc = self._require_index()
        bapq = index_adc.bapq
        if bapq.pca is None or bapq.codebooks is None:
            raise RuntimeError("BAPQ model is incomplete and cannot be saved.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self._meta_dict(), handle, ensure_ascii=False, indent=2)

        arrays = {
            f"codebook_{idx:04d}": np.ascontiguousarray(codebook, dtype=np.float32)
            for idx, codebook in enumerate(bapq.codebooks)
        }
        np.savez_compressed(out_dir / "state.npz", **arrays)
        faiss.write_VectorTransform(bapq.pca, str(out_dir / "pca.faiss"))

    @classmethod
    def load(cls, path: str | Path) -> "BAPQIndex":
        in_dir = Path(path)
        with (in_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        cfg = BAPQIndexConfig(**meta["config"])
        obj = cls(cfg)
        state = meta["state"]

        bapq = obj._build_bapq()
        bapq.pca = faiss.read_VectorTransform(str(in_dir / "pca.faiss"))
        bapq.transform = lambda x: cls._apply_transform(bapq.pca, x)  # type: ignore[method-assign]
        bapq.groups = [list(map(int, g)) for g in state["groups"]]
        bapq.nbits_per_group = [int(b) for b in state["nbits_per_group"]]
        state_npz = np.load(in_dir / "state.npz", allow_pickle=False)
        bapq.codebooks = [
            np.ascontiguousarray(state_npz[f"codebook_{idx:04d}"], dtype=np.float32)
            for idx in range(int(state["n_codebooks"]))
        ]

        index_adc = BAPQIndexADC(
            bapq,
            lut_chunk=int(cfg.lut_chunk),
            query_batch=int(cfg.query_batch),
            db_chunk=int(cfg.db_chunk),
        )
        cls._activate_index(index_adc)
        obj._index_adc = index_adc
        obj.is_trained = bool(state["is_trained"])
        obj.nb = 0
        return obj
