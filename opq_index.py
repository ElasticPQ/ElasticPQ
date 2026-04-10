from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import faiss
import numpy as np

QueryMode = Literal["adc", "sdc"]


@dataclass
class OPQIndexConfig:
    d: int
    B: int

    nbits: int = 8
    M: int = 0

    opq_niter: int = 25
    opq_niter_pq: int = 4
    opq_max_train_points: int = 65536

    seed: int = 123
    verbose: bool = False

    lut_chunk: int = 4096
    query_batch: int = 32

    name: str = "opq"


class OPQIndex:
    """FAISS-backed OPQ training with self-contained numpy persistence/search.

    Training uses:
      - faiss.OPQMatrix for rotation learning
      - faiss.ProductQuantizer for codebook learning on rotated space

    After training, this class stores only numpy parameters so save/load/search do
    not rely on serialized FAISS objects.
    """

    def __init__(self, cfg: OPQIndexConfig):
        self.cfg = cfg
        self.name = str(cfg.name)

        self.M: int = 0
        self.nbits: int = int(cfg.nbits)
        self.ksub: int = 0
        self.dsub: int = 0
        self.d2: int = 0

        self.A: Optional[np.ndarray] = None  # shape (d2, d)
        self.codebooks: Optional[np.ndarray] = None  # shape (M, ksub, dsub)
        self.codes_db: Optional[np.ndarray] = None
        self.nb: int = 0
        self.is_trained: bool = False

        self._resolve_layout()

    def _resolve_layout(self) -> None:
        nbits = int(self.cfg.nbits)
        if nbits <= 0:
            raise ValueError(f"nbits must be positive, got {nbits}")

        if int(self.cfg.M) > 0:
            M = int(self.cfg.M)
            if M * nbits != int(self.cfg.B):
                raise ValueError(f"M * nbits must equal B, got M={M}, nbits={nbits}, B={self.cfg.B}")
        else:
            if int(self.cfg.B) % nbits != 0:
                raise ValueError(f"B must be divisible by nbits, got B={self.cfg.B}, nbits={nbits}")
            M = int(self.cfg.B) // nbits

        if M <= 0:
            raise ValueError(f"M must be positive, got {M}")

        self.M = M
        self.nbits = nbits
        self.ksub = 1 << self.nbits
        self.d2 = ((int(self.cfg.d) + self.M - 1) // self.M) * self.M
        self.dsub = self.d2 // self.M

    def _require_trained(self) -> None:
        if not self.is_trained or self.A is None or self.codebooks is None:
            raise RuntimeError("OPQIndex is not trained.")

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._require_trained()
        return np.ascontiguousarray(x @ self.A.T, dtype=np.float32)

    def _invert_A(self, z: np.ndarray) -> np.ndarray:
        z = np.ascontiguousarray(z, dtype=np.float32)
        self._require_trained()
        return np.ascontiguousarray(z @ self.A, dtype=np.float32)

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        if xt.ndim != 2 or xt.shape[1] != int(self.cfg.d):
            raise ValueError(f"xt must have shape (n, {self.cfg.d}), got {xt.shape}")

        opq = faiss.OPQMatrix(int(self.cfg.d), self.M, self.d2)
        opq.niter = int(self.cfg.opq_niter)
        opq.niter_pq = int(self.cfg.opq_niter_pq)
        opq.max_train_points = int(self.cfg.opq_max_train_points)
        opq.verbose = bool(self.cfg.verbose)
        opq.train(xt)

        A = faiss.vector_to_array(opq.A).astype(np.float32, copy=False).reshape(self.d2, int(self.cfg.d))
        x_rot = np.ascontiguousarray(opq.apply_py(xt), dtype=np.float32)

        pq = faiss.ProductQuantizer(self.d2, self.M, self.nbits)
        pq.train(x_rot)
        centroids = faiss.vector_to_array(pq.centroids).astype(np.float32, copy=False).reshape(self.M, self.ksub, self.dsub)

        self.A = np.ascontiguousarray(A, dtype=np.float32)
        self.codebooks = np.ascontiguousarray(centroids, dtype=np.float32)
        self.codes_db = None
        self.nb = 0
        self.is_trained = True

    def fit(self, xt: np.ndarray) -> "OPQIndex":
        self.train(xt)
        return self

    @staticmethod
    def _codes_dtype_for_nbits(nbits: int) -> np.dtype:
        if nbits <= 8:
            return np.uint8
        if nbits <= 16:
            return np.uint16
        return np.uint32

    def compute_codes(self, xb: np.ndarray) -> np.ndarray:
        self._require_trained()
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != int(self.cfg.d):
            raise ValueError(f"xb must have shape (n, {self.cfg.d}), got {xb.shape}")

        x_rot = self._apply_A(xb).reshape(xb.shape[0], self.M, self.dsub)
        dtype = self._codes_dtype_for_nbits(self.nbits)
        codes = np.empty((xb.shape[0], self.M), dtype=dtype)

        for m in range(self.M):
            sub = np.ascontiguousarray(x_rot[:, m, :], dtype=np.float32)
            _, I = faiss.knn(sub, self.codebooks[m], 1)
            codes[:, m] = I[:, 0].astype(dtype, copy=False)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        self._require_trained()
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.M:
            raise ValueError(f"codes shape must be (n, {self.M}), got {codes.shape}")

        z = np.zeros((codes.shape[0], self.d2), dtype=np.float32)
        for m in range(self.M):
            idx = codes[:, m].astype(np.int64, copy=False)
            z[:, m * self.dsub : (m + 1) * self.dsub] = self.codebooks[m][idx]

        x_hat = self._invert_A(z)
        return np.ascontiguousarray(x_hat[:, : int(self.cfg.d)], dtype=np.float32)

    def add(self, xb: np.ndarray) -> None:
        codes = np.ascontiguousarray(self.compute_codes(xb))
        self.codes_db = codes
        self.nb = int(codes.shape[0])

    def fit_add(self, xt: np.ndarray, xb: np.ndarray) -> "OPQIndex":
        self.train(xt)
        self.add(xb)
        return self

    def _build_lut_q_to_C(self, qsub: np.ndarray, C: np.ndarray) -> np.ndarray:
        qsub = np.ascontiguousarray(qsub, dtype=np.float32)
        C = np.ascontiguousarray(C, dtype=np.float32)
        b = int(qsub.shape[0])
        k = int(C.shape[0])

        qn = np.sum(qsub * qsub, axis=1, keepdims=True)
        cn = np.sum(C * C, axis=1, keepdims=True).T
        out = np.empty((b, k), dtype=np.float32)

        step = int(max(128, self.cfg.lut_chunk))
        for a in range(0, k, step):
            z = min(k, a + step)
            dot = qsub @ C[a:z].T
            out[:, a:z] = (qn + cn[:, a:z] - 2.0 * dot).astype(np.float32, copy=False)
        return out

    @staticmethod
    def _build_lut_code_to_C(C: np.ndarray) -> np.ndarray:
        C = np.ascontiguousarray(C, dtype=np.float32)
        cn = np.sum(C * C, axis=1, keepdims=True)
        return np.ascontiguousarray((cn + cn.T - 2.0 * (C @ C.T)).astype(np.float32, copy=False))

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        self._require_trained()
        if self.codes_db is None:
            raise RuntimeError("OPQIndex.add(xb) must be called before search().")

        mode = str(mode).lower()
        if mode not in ("adc", "sdc"):
            raise ValueError(f"mode must be 'adc' or 'sdc', got {mode!r}")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != int(self.cfg.d):
            raise ValueError(f"xq must have shape (n, {self.cfg.d}), got {xq.shape}")

        k = int(k)
        nq = int(xq.shape[0])
        nb = int(self.nb)
        qb = int(max(1, self.cfg.query_batch))
        xq_rot = self._apply_A(xq).reshape(nq, self.M, self.dsub)

        I_all = np.empty((nq, k), dtype=np.int64)
        D_all = np.empty((nq, k), dtype=np.float32)

        if mode == "adc":
            for q0 in range(0, nq, qb):
                q1 = min(nq, q0 + qb)
                luts: List[np.ndarray] = []
                for m in range(self.M):
                    luts.append(self._build_lut_q_to_C(xq_rot[q0:q1, m, :], self.codebooks[m]))

                for bi in range(q1 - q0):
                    dist = np.zeros((nb,), dtype=np.float32)
                    for m in range(self.M):
                        idx = self.codes_db[:, m].astype(np.int64, copy=False)
                        dist += luts[m][bi, idx]

                    if k >= nb:
                        sel = np.argsort(dist, kind="stable")[:k]
                    else:
                        sel = np.argpartition(dist, k)[:k]
                        sel = sel[np.argsort(dist[sel], kind="stable")]

                    I_all[q0 + bi] = sel.astype(np.int64, copy=False)
                    D_all[q0 + bi] = dist[sel].astype(np.float32, copy=False)
            return D_all, I_all

        dist_tables = [self._build_lut_code_to_C(self.codebooks[m]) for m in range(self.M)]
        codes_q = self.compute_codes(xq)
        for qi in range(nq):
            dist = np.zeros((nb,), dtype=np.float32)
            for m in range(self.M):
                cq = int(codes_q[qi, m])
                idx = self.codes_db[:, m].astype(np.int64, copy=False)
                dist += dist_tables[m][cq, idx]

            if k >= nb:
                sel = np.argsort(dist, kind="stable")[:k]
            else:
                sel = np.argpartition(dist, k)[:k]
                sel = sel[np.argsort(dist[sel], kind="stable")]

            I_all[qi] = sel.astype(np.int64, copy=False)
            D_all[qi] = dist[sel].astype(np.float32, copy=False)
        return D_all, I_all

    def _meta_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "state": {
                "is_trained": bool(self.is_trained),
                "nb": int(self.nb),
                "M": int(self.M),
                "nbits": int(self.nbits),
                "ksub": int(self.ksub),
                "dsub": int(self.dsub),
                "d2": int(self.d2),
                "has_codes_db": self.codes_db is not None,
            },
        }

    def save(self, path: str | Path) -> None:
        self._require_trained()
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self._meta_dict(), f, ensure_ascii=False, indent=2)

        arrays = {
            "A": np.ascontiguousarray(self.A, dtype=np.float32),
            "codebooks": np.ascontiguousarray(self.codebooks, dtype=np.float32),
        }
        if self.codes_db is not None:
            arrays["codes_db"] = np.ascontiguousarray(self.codes_db)
        np.savez_compressed(out_dir / "state.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "OPQIndex":
        in_dir = Path(path)
        with (in_dir / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = OPQIndexConfig(**meta["config"])
        obj = cls(cfg)

        state = meta["state"]
        obj.is_trained = bool(state["is_trained"])
        obj.nb = int(state["nb"])
        obj.M = int(state["M"])
        obj.nbits = int(state["nbits"])
        obj.ksub = int(state["ksub"])
        obj.dsub = int(state["dsub"])
        obj.d2 = int(state["d2"])

        state_npz = np.load(in_dir / "state.npz", allow_pickle=False)
        obj.A = np.ascontiguousarray(state_npz["A"], dtype=np.float32)
        obj.codebooks = np.ascontiguousarray(state_npz["codebooks"], dtype=np.float32)
        obj.codes_db = None
        if "codes_db" in state_npz.files:
            obj.codes_db = np.ascontiguousarray(state_npz["codes_db"])
            obj.nb = int(obj.codes_db.shape[0])
        return obj
