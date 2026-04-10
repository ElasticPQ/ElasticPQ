# bapq.py
# -*- coding: utf-8 -*-
"""
BAPQ (Adaptive Bit Allocation Product Quantization)
Paper: Qin-Zhen Guo et al., "Adaptive bit allocation product quantization", Neurocomputing 2016.

Implements:
  - PCA rotation (no dimensionality reduction)
  - fixed partition of PCA components into M subspaces
  - greedy bit allocation (Algorithm 1): allocate total B bits across M subspaces
  - per-subspace k-means codebooks with k=2^b centroids
  - ADC (LUT) index for evaluation (compatible with your benchmark style)

Dependencies:
  - faiss
  - numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
import faiss


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _sample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    x = _as_f32(x)
    n = int(x.shape[0])
    mr = int(max_rows)
    if mr <= 0 or mr >= n:
        return x
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    idx = rng.choice(n, size=mr, replace=False)
    return x[idx]


def _make_groups(d: int, M: int) -> List[List[int]]:
    d = int(d)
    M = int(M)
    if d <= 0 or M <= 0:
        raise ValueError("d and M must be positive")
    base = d // M
    extra = d % M
    groups: List[List[int]] = []
    cur = 0
    for i in range(M):
        sz = base + (1 if i < extra else 0)
        if sz <= 0:
            raise ValueError("Invalid grouping: too many subspaces for d")
        groups.append(list(range(cur, cur + sz)))
        cur += sz
    assert cur == d
    return groups


def _kmeans_mse(x_train: np.ndarray, k: int, *, niter: int, nredo: int, seed: int) -> Tuple[float, np.ndarray]:
    """
    Train kmeans on x_train and return (avg squared L2 to nearest centroid, centroids).
    """
    x_train = _as_f32(x_train)
    n, dsub = x_train.shape
    if n <= 0:
        raise ValueError("empty training data")

    kk = int(k)
    if kk <= 1:
        c = x_train.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        diff = x_train - c
        mse = float(np.mean(np.sum(diff * diff, axis=1)))
        return mse, c

    # Faiss Kmeans
    km = faiss.Kmeans(
        dsub,
        kk,
        niter=int(niter),
        nredo=int(nredo),
        seed=int(seed),
        verbose=False,
    )
    km.train(x_train)
    C = _as_f32(km.centroids)  # (k, dsub)

    # compute mse on train (paper uses quantization distortion; train-dist is standard proxy here)
    index = faiss.IndexFlatL2(dsub)
    index.add(C)
    D, _ = index.search(x_train, 1)
    mse = float(np.mean(D.reshape(-1)))
    return mse, C


@dataclass
class BAPQ:
    """
    BAPQ trainer + encoder.

    After train():
      - groups: List[List[int]] (PCA component indices)
      - nbits_per_group: List[int]
      - codebooks: List[np.ndarray] (ksub_g, dim_g)
      - pca: faiss.PCAMatrix (d->d rotation)
    """

    d: int
    M: int
    B: int
    bmax: int = 12

    # training control
    seed: int = 123
    max_train_rows: int = 200000  # cap for kmeans training (speed)
    pca_max_train_rows: int = 200000

    # kmeans params
    km_niter: int = 20
    km_nredo: int = 1

    # learned
    pca: Optional[faiss.PCAMatrix] = None
    groups: Optional[List[List[int]]] = None
    nbits_per_group: Optional[List[int]] = None
    codebooks: Optional[List[np.ndarray]] = None

    def _check(self) -> None:
        if self.d <= 0 or self.M <= 0 or self.B < 0:
            raise ValueError("invalid (d,M,B)")
        if self.B > self.M * self.bmax:
            raise ValueError(f"Infeasible: B={self.B} > M*bmax={self.M*self.bmax}")

    def _fit_pca(self, xt: np.ndarray) -> None:
        xt = _as_f32(xt)
        xs = _sample_rows(xt, int(self.pca_max_train_rows), seed=self.seed + 17)
        pca = faiss.PCAMatrix(int(self.d), int(self.d), 0, False)  # no reduction;
        # Note: FAISS PCAMatrix signature may vary; in common builds (d_in, d_out, eigen_power, random_rotation).
        # Here we follow "rotate but not reduce" intent; if your build differs, adjust accordingly.
        pca.train(xs)
        self.pca = pca

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("BAPQ not trained: PCA missing")
        return _as_f32(self.pca.apply(_as_f32(x)))

    def train(self, xt: np.ndarray) -> None:
        """
        Train BAPQ:
          1) PCA rotation (no reduction)
          2) fixed groups on PCA components
          3) greedy bit allocation (Algorithm 1)
          4) finalize codebooks for allocated bits
        """
        self._check()
        xt = _as_f32(xt)

        # 1) PCA
        self._fit_pca(xt)
        xtp = self.transform(xt)

        # 2) groups
        groups = _make_groups(self.d, self.M)
        self.groups = groups

        # Prepare training slices (sample to control runtime)
        xtp_s = _sample_rows(xtp, int(self.max_train_rows), seed=self.seed + 23)

        # Cache: (gi, bits) -> (mse, centroids)
        cache_mse: dict[Tuple[int, int], float] = {}
        cache_C: dict[Tuple[int, int], np.ndarray] = {}

        def get_mse_and_C(gi: int, b: int) -> Tuple[float, np.ndarray]:
            key = (int(gi), int(b))
            if key in cache_mse:
                return float(cache_mse[key]), cache_C[key]
            dims = groups[gi]
            xg = _as_f32(xtp_s[:, dims])
            k = 1 << int(b)
            mse, C = _kmeans_mse(
                xg,
                k,
                niter=int(self.km_niter),
                nredo=int(self.km_nredo),
                seed=int(self.seed + 10007 * gi + 7919 * b),
            )
            cache_mse[key] = float(mse)
            cache_C[key] = _as_f32(C)
            return float(mse), cache_C[key]

        # 3) Greedy bit allocation (Algorithm 1)
        bits = [0] * self.M
        Ej = [0.0] * self.M
        for gi in range(self.M):
            mse0, _ = get_mse_and_C(gi, 0)
            Ej[gi] = float(mse0)
        total = float(sum(Ej))

        for _step in range(int(self.B)):
            best_g = -1
            best_total = float("inf")

            for gi in range(self.M):
                if bits[gi] >= int(self.bmax):
                    continue
                b_new = bits[gi] + 1
                mse_new, _ = get_mse_and_C(gi, b_new)
                cand = total - Ej[gi] + float(mse_new)
                if cand < best_total:
                    best_total = float(cand)
                    best_g = int(gi)

            if best_g < 0:
                raise RuntimeError("Greedy allocation failed (no feasible group).")
            # commit
            bits[best_g] += 1
            mse_new, _ = get_mse_and_C(best_g, bits[best_g])
            total = total - Ej[best_g] + float(mse_new)
            Ej[best_g] = float(mse_new)

        self.nbits_per_group = [int(b) for b in bits]

        # 4) Finalize codebooks
        codebooks: List[np.ndarray] = []
        for gi in range(self.M):
            b = bits[gi]
            _, C = get_mse_and_C(gi, b)
            codebooks.append(_as_f32(C))
        self.codebooks = codebooks

    def compute_codes(self, x: np.ndarray) -> np.ndarray:
        if self.groups is None or self.codebooks is None or self.nbits_per_group is None:
            raise RuntimeError("BAPQ not trained")
        x = _as_f32(x)
        xp = self.transform(x)

        n = int(xp.shape[0])
        M = int(self.M)
        codes = np.empty((n, M), dtype=np.uint16)

        for gi, dims in enumerate(self.groups):
            C = self.codebooks[gi]
            dsub = int(C.shape[1])
            xg = _as_f32(xp[:, dims])
            if xg.shape[1] != dsub:
                raise RuntimeError("subspace dim mismatch")
            # nearest centroid id
            index = faiss.IndexFlatL2(dsub)
            index.add(C)
            _, I = index.search(xg, 1)
            codes[:, gi] = I.reshape(-1).astype(np.uint16, copy=False)

        return codes

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode to original space (best effort):
          code -> PCA space reconstruction -> inverse PCA
        """
        if self.groups is None or self.codebooks is None or self.pca is None:
            raise RuntimeError("BAPQ not trained")
        codes = np.ascontiguousarray(codes)
        n = int(codes.shape[0])

        # reconstruct in PCA space
        xp = np.zeros((n, int(self.d)), dtype=np.float32)
        for gi, dims in enumerate(self.groups):
            C = self.codebooks[gi]
            idx = codes[:, gi].astype(np.int64, copy=False)
            xp[:, dims] = C[idx]

        # inverse PCA (FAISS supports reverse_transform in many builds)
        try:
            x = self.pca.reverse_transform(_as_f32(xp))
            return _as_f32(x)
        except Exception:
            # fallback: return PCA-space reconstruction if inverse unavailable
            return _as_f32(xp)


class BAPQIndexADC:
    """
    Fast ADC (LUT) index for BAPQ.

    Key optimizations for large M (e.g., M=d/4=240 on GIST):
      1) Ignore groups with nbits==0 (ksub=1): their contribution is constant across DB items.
      2) Vectorize over DB chunks and maintain per-query top-k incrementally.
    """

    def __init__(
        self,
        bapq: BAPQ,
        *,
        name: str = "bapq_adc",
        lut_chunk: int = 4096,
        query_batch: int = 32,
        db_chunk: int = 200_000,   # tune: 50k~300k depending on RAM
    ):
        self.bapq = bapq
        self.name = str(name)
        self.lut_chunk = int(max(128, lut_chunk))
        self.query_batch = int(max(1, query_batch))
        self.db_chunk = int(max(10_000, db_chunk))

        # store only active (bits>0) codes to reduce work
        self.codes_db: Optional[np.ndarray] = None  # (nb, M_active) uint16/uint32
        self.nb: int = 0

        # cached after train
        self._groups_all: Optional[List[List[int]]] = None
        self._codebooks_all: Optional[List[np.ndarray]] = None
        self._nbits_all: Optional[List[int]] = None

        # active subset
        self._active_gi: Optional[List[int]] = None
        self._groups: Optional[List[List[int]]] = None
        self._codebooks: Optional[List[np.ndarray]] = None
        self._M: int = 0  # M_active

    def train(self, xt: np.ndarray) -> None:
        self.bapq.train(_as_f32(xt))
        if self.bapq.groups is None or self.bapq.codebooks is None or self.bapq.nbits_per_group is None:
            raise RuntimeError("BAPQ training failed")

        self._groups_all = [list(map(int, g)) for g in self.bapq.groups]
        self._codebooks_all = [_as_f32(C) for C in self.bapq.codebooks]
        self._nbits_all = [int(b) for b in self.bapq.nbits_per_group]

        # active groups: bits>0 => ksub>1 influences ranking
        active_gi = [gi for gi, b in enumerate(self._nbits_all) if int(b) > 0]
        if len(active_gi) == 0:
            # Degenerate: all bits are 0. Distances are constant -> arbitrary ranking.
            # Still keep one dummy group to avoid empty loops.
            active_gi = [0]

        self._active_gi = active_gi
        self._groups = [self._groups_all[gi] for gi in active_gi]
        self._codebooks = [self._codebooks_all[gi] for gi in active_gi]
        self._M = int(len(active_gi))

        # optional: sanity
        for C in self._codebooks:
            if C.shape[0] <= 1:
                # Should not happen for bits>0, but keep safe
                pass

    def add(self, xb: np.ndarray) -> None:
        if self._active_gi is None:
            raise RuntimeError("train() before add()")

        xb = _as_f32(xb)
        codes_full = self.bapq.compute_codes(xb)  # (nb, M_all)
        codes_full = np.ascontiguousarray(codes_full)

        # keep only active columns
        act = np.asarray(self._active_gi, dtype=np.int64)
        codes = np.ascontiguousarray(codes_full[:, act])

        # dtype compact
        if codes.dtype != np.uint16 and codes.max() < 65535:
            codes = codes.astype(np.uint16, copy=False)

        self.codes_db = codes
        self.nb = int(codes.shape[0])

    def _build_lut_for_group(self, qg: np.ndarray, C: np.ndarray) -> np.ndarray:
        qg = _as_f32(qg)
        C = _as_f32(C)
        b, _dim = qg.shape
        k = int(C.shape[0])

        qn = np.sum(qg * qg, axis=1, keepdims=True)          # (b,1)
        cn = np.sum(C * C, axis=1, keepdims=True).T          # (1,k)
        out = np.empty((b, k), dtype=np.float32)

        step = self.lut_chunk
        for a in range(0, k, step):
            z = min(k, a + step)
            Cc = C[a:z]
            dot = qg @ Cc.T
            out[:, a:z] = (qn + cn[:, a:z] - 2.0 * dot).astype(np.float32, copy=False)
        return out

    def _decode_from_codes(self, codes_active: np.ndarray) -> np.ndarray:
        # For recon diagnostics only: we need full codes.
        # Recompute full codes on demand in benchmark via bapq.compute_codes(xb_s),
        # so here just decode with BAPQ helper (expects full codes); this method is kept for API parity.
        return self.bapq.decode_codes(codes_active)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized ADC search for BAPQ.

        Patch vs naive:
          - For each DB chunk, compute distances, then keep only chunk top-k per query.
          - Merge with global best by selecting top-k from 2k candidates.
        This avoids O(chunk_size) merges and makes large M (even with active groups) tractable.
        """
        if self.codes_db is None:
            raise RuntimeError("add() before search()")
        if self._groups is None or self._codebooks is None:
            raise RuntimeError("train() before search()")

        xq = _as_f32(xq)
        k = int(k)
        nq = int(xq.shape[0])
        nb = int(self.nb)
        M = int(self._M)

        # PCA-transform queries once
        xq2 = self.bapq.transform(xq)

        codes_db = self.codes_db
        assert codes_db is not None

        I_all = np.empty((nq, k), dtype=np.int64)
        D_all = np.empty((nq, k), dtype=np.float32)

        qb = int(self.query_batch)
        db_chunk = int(self.db_chunk)

        for q0 in range(0, nq, qb):
            q1 = min(nq, q0 + qb)
            bsz = q1 - q0

            # Build LUTs for active groups only: list of (bsz, ksub_g)
            luts: List[np.ndarray] = []
            for gi in range(M):
                dims = self._groups[gi]
                C = self._codebooks[gi]
                qg = _as_f32(xq2[q0:q1, dims])
                luts.append(self._build_lut_for_group(qg, C))

            # Global best per query (bsz, k)
            best_D = np.full((bsz, k), np.inf, dtype=np.float32)
            best_I = np.full((bsz, k), -1, dtype=np.int64)

            # Iterate DB chunks
            for b0 in range(0, nb, db_chunk):
                b1 = min(nb, b0 + db_chunk)
                csz = b1 - b0
                codes_chunk = codes_db[b0:b1]  # (csz, M)

                # dist_chunk: (bsz, csz)
                dist_chunk = np.zeros((bsz, csz), dtype=np.float32)

                # Vectorized gather-add per group
                for gi in range(M):
                    idx = codes_chunk[:, gi].astype(np.int64, copy=False)  # (csz,)
                    dist_chunk += luts[gi][:, idx]

                # --- NEW: keep only chunk top-k per query ---
                if k >= csz:
                    selc = np.argsort(dist_chunk, axis=1, kind="stable")[:, :k]
                else:
                    selc = np.argpartition(dist_chunk, k, axis=1)[:, :k]
                    r = np.arange(bsz)[:, None]
                    selc = selc[r, np.argsort(dist_chunk[r, selc], axis=1, kind="stable")]

                r = np.arange(bsz)[:, None]
                cand_D = dist_chunk[r, selc].astype(np.float32, copy=False)  # (bsz, k)
                cand_I = (b0 + selc).astype(np.int64, copy=False)  # (bsz, k)

                # Merge best (k) with cand (k) -> select best k from 2k
                merged_D = np.concatenate([best_D, cand_D], axis=1)  # (bsz, 2k)
                merged_I = np.concatenate([best_I, cand_I], axis=1)  # (bsz, 2k)

                sel = np.argpartition(merged_D, k, axis=1)[:, :k]
                sel = sel[r, np.argsort(merged_D[r, sel], axis=1, kind="stable")]

                best_D = merged_D[r, sel].astype(np.float32, copy=False)
                best_I = merged_I[r, sel].astype(np.int64, copy=False)

            I_all[q0:q1] = best_I
            D_all[q0:q1] = best_D

        return D_all, I_all

