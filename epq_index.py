from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import faiss
import numpy as np

from elastic_pq import ElasticPQ, ElasticPQConfig
from forwarder_cryst import CrystallizationForwarder, CrystallizationForwarderConfig
from forwarder_mc import MarginalBeamForwarder, MarginalBeamForwarderConfig
from grouper_grow import ClusterGrowGrouper, ClusterGrowGrouperConfig

QueryMode = Literal["adc", "sdc"]


@dataclass
class EPQIndexConfig:
    d: int
    B: int

    min_bits: int = 0
    max_bits: int = 12

    kmeans_niter: int = 25
    kmeans_nredo: int = 1

    enable_uneven_opq: bool = True
    uneven_opq_niter: int = 0
    uneven_opq_kmeans_niter: int = 15
    uneven_opq_kmeans_nredo: int = 1
    uneven_opq_max_train: int = 65536
    uneven_opq_max_eval: int = 16384
    uneven_opq_eval_frac: float = 0.2
    uneven_opq_seed: int = 0

    seed: int = 123
    verbose: bool = False

    lut_chunk: int = 4096
    query_batch: int = 32

    structure_save_path: str = ""
    name: str = "epq"


class EPQIndex:
    """A trainable/searchable EPQ index with built-in save/load support.

    This class wraps the current ElasticPQ training pipeline and exposes a
    self-contained index object that manages:

    - config
    - training
    - database encoding
    - ADC/SDC search
    - persistence
    """

    def __init__(self, cfg: EPQIndexConfig):
        self.cfg = cfg
        self.name = str(cfg.name)

        self.is_trained: bool = False
        self.codes_db: Optional[np.ndarray] = None
        self.nb: int = 0

        self.M: int = 0
        self.groups_orig: List[List[int]] = []
        self.groups_contig: List[List[int]] = []
        self.nbits_per_group: List[int] = []
        self.ksub_per_group: List[int] = []
        self.codebooks: List[np.ndarray] = []
        self.global_A: Optional[np.ndarray] = None

    # ------------------------
    # Build / Train
    # ------------------------

    def _build_default_grouper(self):
        grouper = ClusterGrowGrouper(ClusterGrowGrouperConfig())
        grouper = grouper.then(
            CrystallizationForwarder(
                CrystallizationForwarderConfig(verbose=bool(self.cfg.verbose))
            )
        )
        grouper = grouper.then(
            MarginalBeamForwarder(
                MarginalBeamForwarderConfig(
                    verbose=bool(self.cfg.verbose),
                    seed=int(self.cfg.seed),
                )
            )
        )
        return grouper

    def _make_elastic_cfg(self) -> ElasticPQConfig:
        return ElasticPQConfig(
            d=int(self.cfg.d),
            B=int(self.cfg.B),
            min_bits=int(self.cfg.min_bits),
            max_bits=int(self.cfg.max_bits),
            kmeans_niter=int(self.cfg.kmeans_niter),
            kmeans_nredo=int(self.cfg.kmeans_nredo),
            enable_uneven_opq=bool(self.cfg.enable_uneven_opq),
            uneven_opq_niter=int(self.cfg.uneven_opq_niter),
            uneven_opq_kmeans_niter=int(self.cfg.uneven_opq_kmeans_niter),
            uneven_opq_kmeans_nredo=int(self.cfg.uneven_opq_kmeans_nredo),
            uneven_opq_max_train=int(self.cfg.uneven_opq_max_train),
            uneven_opq_max_eval=int(self.cfg.uneven_opq_max_eval),
            uneven_opq_eval_frac=float(self.cfg.uneven_opq_eval_frac),
            uneven_opq_seed=int(self.cfg.uneven_opq_seed),
            structure_save_path=str(self.cfg.structure_save_path or ""),
            seed=int(self.cfg.seed),
            verbose=bool(self.cfg.verbose),
        )

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        if xt.ndim != 2 or xt.shape[1] != int(self.cfg.d):
            raise ValueError(f"xt must have shape (n, {self.cfg.d}), got {xt.shape}")

        epq = ElasticPQ(self._make_elastic_cfg(), grouper=self._build_default_grouper())
        epq.train(xt)

        self._load_from_elastic(epq)
        self.codes_db = None
        self.nb = 0
        self.is_trained = True

    def fit(self, xt: np.ndarray) -> "EPQIndex":
        self.train(xt)
        return self

    def _load_from_elastic(self, epq: ElasticPQ) -> None:
        self.M = int(epq.M)
        self.groups_orig = [list(map(int, g)) for g in epq.groups_orig]
        self.groups_contig = [list(map(int, g)) for g in epq.groups_contig]
        self.nbits_per_group = [int(b) for b in epq.nbits_per_group]
        self.ksub_per_group = [int(k) for k in epq.ksub_per_group]
        self.codebooks = [np.ascontiguousarray(c, dtype=np.float32) for c in epq.codebooks]
        self.global_A = None if epq.global_A is None else np.ascontiguousarray(epq.global_A, dtype=np.float32)

    # ------------------------
    # Encoding / Decoding
    # ------------------------

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("EPQIndex is not trained.")

    def _groups_for_codes(self) -> List[List[int]]:
        self._require_trained()
        if self.global_A is not None:
            return self.groups_contig
        return self.groups_orig

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.global_A is None:
            return x
        return np.ascontiguousarray(x @ self.global_A, dtype=np.float32)

    @staticmethod
    def _codes_dtype_for_bits(bits_per_group: List[int]) -> np.dtype:
        mb = int(max(bits_per_group)) if bits_per_group else 0
        if mb <= 8:
            return np.uint8
        if mb <= 16:
            return np.uint16
        return np.uint32

    def compute_codes(self, xb: np.ndarray) -> np.ndarray:
        self._require_trained()

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != int(self.cfg.d):
            raise ValueError(f"xb must have shape (n, {self.cfg.d}), got {xb.shape}")

        groups = self._groups_for_codes()
        xwork = self._apply_A(xb)
        codes = np.empty((xwork.shape[0], self.M), dtype=self._codes_dtype_for_bits(self.nbits_per_group))

        for gi, dims in enumerate(groups):
            if self.nbits_per_group[gi] == 0:
                codes[:, gi] = 0
                continue
            sub = np.ascontiguousarray(xwork[:, dims], dtype=np.float32)
            _, I = faiss.knn(sub, self.codebooks[gi], 1)
            codes[:, gi] = I[:, 0].astype(codes.dtype, copy=False)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        self._require_trained()

        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.M:
            raise ValueError(f"codes shape must be (n, {self.M}), got {codes.shape}")

        groups = self._groups_for_codes()
        y_hat = np.zeros((codes.shape[0], int(self.cfg.d)), dtype=np.float32)
        for gi, dims in enumerate(groups):
            idx = codes[:, gi].astype(np.int64, copy=False)
            y_hat[:, dims] = self.codebooks[gi][idx]

        if self.global_A is not None:
            return np.ascontiguousarray(y_hat @ self.global_A.T, dtype=np.float32)
        return y_hat

    # ------------------------
    # Database ops
    # ------------------------

    def add(self, xb: np.ndarray) -> None:
        self._require_trained()
        codes = np.ascontiguousarray(self.compute_codes(xb))
        if codes.ndim != 2 or codes.shape[1] != self.M:
            raise RuntimeError(f"codes shape mismatch: got {codes.shape}, expected (*, {self.M})")
        self.codes_db = codes
        self.nb = int(codes.shape[0])

    def fit_add(self, xt: np.ndarray, xb: np.ndarray) -> "EPQIndex":
        self.train(xt)
        self.add(xb)
        return self

    # ------------------------
    # Search
    # ------------------------

    def _build_lut_q_to_C(self, qg: np.ndarray, C: np.ndarray) -> np.ndarray:
        qg = np.ascontiguousarray(qg, dtype=np.float32)
        C = np.ascontiguousarray(C, dtype=np.float32)
        b = int(qg.shape[0])
        k = int(C.shape[0])

        qn = np.sum(qg * qg, axis=1, keepdims=True)
        cn = np.sum(C * C, axis=1, keepdims=True).T
        out = np.empty((b, k), dtype=np.float32)

        step = int(max(128, self.cfg.lut_chunk))
        for a in range(0, k, step):
            z = min(k, a + step)
            dot = qg @ C[a:z].T
            out[:, a:z] = (qn + cn[:, a:z] - 2.0 * dot).astype(np.float32, copy=False)
        return out

    @staticmethod
    def _build_lut_code_to_C(C: np.ndarray) -> np.ndarray:
        C = np.ascontiguousarray(C, dtype=np.float32)
        cn = np.sum(C * C, axis=1, keepdims=True)
        dist = cn + cn.T - 2.0 * (C @ C.T)
        return np.ascontiguousarray(dist.astype(np.float32, copy=False))

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        self._require_trained()
        if self.codes_db is None:
            raise RuntimeError("EPQIndex.add(xb) must be called before search().")

        mode = str(mode).lower()
        if mode not in ("adc", "sdc"):
            raise ValueError(f"mode must be 'adc' or 'sdc', got {mode!r}")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != int(self.cfg.d):
            raise ValueError(f"xq must have shape (n, {self.cfg.d}), got {xq.shape}")

        k = int(k)
        nq = int(xq.shape[0])
        nb = int(self.nb)
        M = int(self.M)
        codes_db = self.codes_db
        groups = self._groups_for_codes()
        xq2 = self._apply_A(xq)

        I_all = np.empty((nq, k), dtype=np.int64)
        D_all = np.empty((nq, k), dtype=np.float32)
        qb = int(max(1, self.cfg.query_batch))

        if mode == "adc":
            for q0 in range(0, nq, qb):
                q1 = min(nq, q0 + qb)
                batch_luts: List[np.ndarray] = []
                for gi, dims in enumerate(groups):
                    qg = np.ascontiguousarray(xq2[q0:q1, dims], dtype=np.float32)
                    batch_luts.append(self._build_lut_q_to_C(qg, self.codebooks[gi]))

                for bi in range(q1 - q0):
                    dist = np.zeros((nb,), dtype=np.float32)
                    for gi in range(M):
                        idx = codes_db[:, gi].astype(np.int64, copy=False)
                        dist += batch_luts[gi][bi, idx]

                    if k >= nb:
                        sel = np.argsort(dist, kind="stable")[:k]
                    else:
                        sel = np.argpartition(dist, k)[:k]
                        sel = sel[np.argsort(dist[sel], kind="stable")]

                    I_all[q0 + bi] = sel.astype(np.int64, copy=False)
                    D_all[q0 + bi] = dist[sel].astype(np.float32, copy=False)
            return D_all, I_all

        dist_tables = [self._build_lut_code_to_C(C) for C in self.codebooks]
        codes_q = np.ascontiguousarray(self.compute_codes(xq))

        for qi in range(nq):
            dist = np.zeros((nb,), dtype=np.float32)
            for gi in range(M):
                cq = int(codes_q[qi, gi])
                idx_db = codes_db[:, gi].astype(np.int64, copy=False)
                dist += dist_tables[gi][cq, idx_db]

            if k >= nb:
                sel = np.argsort(dist, kind="stable")[:k]
            else:
                sel = np.argpartition(dist, k)[:k]
                sel = sel[np.argsort(dist[sel], kind="stable")]

            I_all[qi] = sel.astype(np.int64, copy=False)
            D_all[qi] = dist[sel].astype(np.float32, copy=False)

        return D_all, I_all

    # ------------------------
    # Persistence
    # ------------------------

    def _meta_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "state": {
                "is_trained": bool(self.is_trained),
                "nb": int(self.nb),
                "M": int(self.M),
                "groups_orig": self.groups_orig,
                "groups_contig": self.groups_contig,
                "nbits_per_group": self.nbits_per_group,
                "ksub_per_group": self.ksub_per_group,
                "has_global_A": self.global_A is not None,
                "has_codes_db": self.codes_db is not None,
            },
        }

    def save(self, path: str | Path) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self._meta_dict(), f, ensure_ascii=False, indent=2)

        arrays: Dict[str, np.ndarray] = {}
        if self.global_A is not None:
            arrays["global_A"] = np.ascontiguousarray(self.global_A, dtype=np.float32)
        if self.codes_db is not None:
            arrays["codes_db"] = np.ascontiguousarray(self.codes_db)
        for gi, cb in enumerate(self.codebooks):
            arrays[f"codebook_{gi:04d}"] = np.ascontiguousarray(cb, dtype=np.float32)

        np.savez_compressed(out_dir / "state.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "EPQIndex":
        in_dir = Path(path)
        with (in_dir / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = EPQIndexConfig(**meta["config"])
        obj = cls(cfg)

        state = meta["state"]
        obj.is_trained = bool(state["is_trained"])
        obj.nb = int(state["nb"])
        obj.M = int(state["M"])
        obj.groups_orig = [list(map(int, g)) for g in state["groups_orig"]]
        obj.groups_contig = [list(map(int, g)) for g in state["groups_contig"]]
        obj.nbits_per_group = [int(b) for b in state["nbits_per_group"]]
        obj.ksub_per_group = [int(k) for k in state["ksub_per_group"]]

        state_npz = np.load(in_dir / "state.npz", allow_pickle=False)
        obj.global_A = None
        if "global_A" in state_npz.files:
            obj.global_A = np.ascontiguousarray(state_npz["global_A"], dtype=np.float32)

        obj.codebooks = []
        for gi in range(obj.M):
            key = f"codebook_{gi:04d}"
            if key not in state_npz.files:
                raise RuntimeError(f"Missing codebook array: {key}")
            obj.codebooks.append(np.ascontiguousarray(state_npz[key], dtype=np.float32))

        obj.codes_db = None
        if "codes_db" in state_npz.files:
            obj.codes_db = np.ascontiguousarray(state_npz["codes_db"])
            obj.nb = int(obj.codes_db.shape[0])

        return obj
