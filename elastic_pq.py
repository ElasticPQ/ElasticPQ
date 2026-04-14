#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from grouper import (
    Grouper,
    EPQStructure,
    FixedStructureGrouper,
    make_default_context_with_proxy,
    Groups,
    Bits,
)


# ============================================================
# Config
# ============================================================

@dataclass
class ElasticPQConfig:
    d: int
    B: int

    min_bits: int = 0
    max_bits: int = 12

    # k-means training params (per-group)
    kmeans_niter: int = 25
    kmeans_nredo: int = 1

    # UnevenOPQ (global rotation learned for uneven groups/bits)
    enable_uneven_opq: bool = True

    # if >0: fixed iterations; if <=0: auto early stop
    uneven_opq_niter: int = 0

    uneven_opq_kmeans_niter: int = 15
    uneven_opq_kmeans_nredo: int = 1
    uneven_opq_max_train: int = 65536
    uneven_opq_max_eval: int = 16384
    uneven_opq_eval_frac: float = 0.2
    uneven_opq_seed: int = 0

    # Structure save path. If non-empty and grouper is NOT fixed, will:
    #   - save EPQStructure to this path
    #   - replace self.grouper with FixedStructureGrouper(struct)
    structure_save_path: str = "data/temp_epq_structure.json"

    seed: int = 123
    verbose: bool = False


# ============================================================
# Helpers
# ============================================================

def _codes_dtype_for_bits(bits_per_group: Sequence[int]) -> np.dtype:
    mb = int(max(bits_per_group)) if bits_per_group else 0
    if mb <= 8:
        return np.uint8
    if mb <= 16:
        return np.uint16
    return np.uint32


def _stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _is_fixed_grouper(grouper: Any) -> bool:
    if isinstance(grouper, FixedStructureGrouper):
        return True
    try:
        return bool(getattr(grouper, "is_fixed_structure"))
    except Exception:
        return False


def _perm_from_groups(groups: List[List[int]], d: int) -> np.ndarray:
    perm = np.array([int(i) for g in groups for i in g], dtype=np.int64)
    if perm.size != int(d):
        raise ValueError(f"perm size mismatch: expected {d}, got {perm.size}")
    if len(set(perm.tolist())) != int(d):
        raise ValueError("perm has duplicates (invalid groups)")
    if set(perm.tolist()) != set(range(int(d))):
        raise ValueError("perm is not a permutation of [0..d-1]")
    return perm


def _perm_matrix_from_perm(perm: np.ndarray) -> np.ndarray:
    """Permutation matrix P s.t. X @ P == X[:, perm]."""
    perm = np.asarray(perm, dtype=np.int64)
    d = int(perm.size)
    P = np.zeros((d, d), dtype=np.float32)
    P[perm, np.arange(d, dtype=np.int64)] = 1.0
    return P


def _groups_as_contiguous_blocks(sizes: List[int]) -> Groups:
    out: Groups = []
    off = 0
    for s in sizes:
        s = int(s)
        if s <= 0:
            raise ValueError(f"invalid block size: {s}")
        out.append(list(range(off, off + s)))
        off += s
    return out


def _orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """R = argmin_{R orthogonal} ||X R - Y||_F^2."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)

    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("Non-finite values in Procrustes inputs (NaN/Inf).")

    A = (X.T.astype(np.float64, copy=False) @ Y.astype(np.float64, copy=False))

    def _svd_to_R(A64: np.ndarray) -> np.ndarray:
        U, _S, Vt = np.linalg.svd(A64, full_matrices=False)
        R = (U @ Vt).astype(np.float32, copy=False)
        # enforce det(R)=+1
        try:
            if np.linalg.det(R.astype(np.float64)) < 0:
                U[:, -1] *= -1.0
                R = (U @ Vt).astype(np.float32, copy=False)
        except Exception:
            pass
        return np.ascontiguousarray(R, dtype=np.float32)

    try:
        return _svd_to_R(A)
    except np.linalg.LinAlgError:
        eps = 1e-8 * (np.linalg.norm(A) + 1.0)
        A2 = A + eps * np.eye(A.shape[0], dtype=np.float64)
        try:
            return _svd_to_R(A2)
        except np.linalg.LinAlgError:
            d = A.shape[0]
            return np.eye(d, dtype=np.float32)


# ============================================================
# UnevenOPQ
# ============================================================

class UnevenOPQ:
    """Global orthogonal rotation for a fixed uneven block structure (sizes,bits).
    Assumes X has been pre-permuted so blocks are contiguous.
    """

    def __init__(
        self,
        *,
        niter: int,
        kmeans_niter: int,
        kmeans_nredo: int,
        max_train: int,
        max_eval: int,
        eval_frac: float,
        seed: int,
        verbose: bool = False,
    ):
        self.niter = int(niter)
        self.kmeans_niter = int(kmeans_niter)
        self.kmeans_nredo = int(kmeans_nredo)
        self.max_train = int(max_train)
        self.max_eval = int(max_eval)
        self.eval_frac = float(eval_frac)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.R: Optional[np.ndarray] = None

    @staticmethod
    def _blocks_from_sizes(d: int, sizes: List[int]) -> List[Tuple[int, int]]:
        blocks: List[Tuple[int, int]] = []
        off = 0
        for s in sizes:
            s = int(s)
            if s <= 0:
                raise ValueError(f"invalid block size: {s}")
            blocks.append((off, off + s))
            off += s
        if off != int(d):
            raise ValueError(f"sizes must sum to d={d}, got {off}")
        return blocks

    def _split_fit_eval(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = int(X.shape[0])
        if n <= 0:
            return X, None

        rng = _stable_rng(self.seed)

        if self.max_train > 0 and n > self.max_train:
            idx = rng.permutation(n)
            idx_fit = idx[: self.max_train]
            idx_rem = idx[self.max_train :]
            Xfit = np.ascontiguousarray(X[idx_fit], dtype=np.float32)
            rem = int(idx_rem.shape[0])
            if rem <= 0:
                return Xfit, None

            if self.max_eval > 0:
                nev = min(int(self.max_eval), rem)
            else:
                nev = int(round(float(Xfit.shape[0]) * max(0.0, self.eval_frac)))

            if nev <= 0:
                return Xfit, None

            Xev = np.ascontiguousarray(X[idx_rem[:nev]], dtype=np.float32)
            return Xfit, Xev

        if self.max_eval <= 0 and self.eval_frac <= 0:
            return X, None

        target_ev = int(self.max_eval) if self.max_eval > 0 else int(round(n * max(0.0, self.eval_frac)))
        if target_ev <= 0 or target_ev >= n:
            return X, None

        idx = rng.permutation(n)
        Xev = np.ascontiguousarray(X[idx[:target_ev]], dtype=np.float32)
        Xfit = np.ascontiguousarray(X[idx[target_ev:]], dtype=np.float32)
        return Xfit, Xev

    def fit(
            self,
            X: np.ndarray,
            sizes: List[int],
            nbits: List[int],
            *,
            # auto stop params used when niter <= 0
            auto_patience: int = 5,
            auto_min_delta: float = 1e-7,
            auto_max_iter: int = 256,
    ) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        if d <= 0:
            raise ValueError("X has invalid dimension")
        if len(sizes) != len(nbits):
            raise ValueError("sizes and nbits length mismatch")
        if sum(map(int, sizes)) != int(d):
            raise ValueError(f"sizes must sum to d={d}, got {sum(map(int, sizes))}")

        blocks = self._blocks_from_sizes(d, sizes)
        nbits_i = [int(b) for b in nbits]

        Xfit, Xev = self._split_fit_eval(X)
        nfit = int(Xfit.shape[0])
        nev = int(Xev.shape[0]) if Xev is not None else 0

        R = np.eye(d, dtype=np.float32)

        def _train_codebooks(Yfit: np.ndarray) -> List[np.ndarray]:
            """Train block-wise codebooks on Yfit (contiguous blocks)."""
            codebooks: List[np.ndarray] = []
            for gi, (a, b) in enumerate(blocks):
                sub = np.ascontiguousarray(Yfit[:, a:b], dtype=np.float32)
                dim_g = int(sub.shape[1])
                bb = int(nbits_i[gi])
                k = (1 << bb) if bb > 0 else 1

                if bb == 0 or k == 1 or dim_g == 0:
                    c = sub.mean(axis=0, keepdims=True)
                    codebooks.append(np.ascontiguousarray(c, dtype=np.float32))
                    continue

                km = faiss.Kmeans(
                    dim_g,
                    k,
                    niter=self.kmeans_niter,
                    nredo=self.kmeans_nredo,
                    verbose=False,
                )
                try:
                    km.cp.min_points_per_centroid = 1
                except Exception:
                    pass
                km.train(sub)
                C = np.ascontiguousarray(km.centroids, dtype=np.float32)
                codebooks.append(C)
            return codebooks

        def _quantize_with_codebooks(Y: np.ndarray, codebooks: List[np.ndarray]) -> np.ndarray:
            """Quantize Y with pre-trained codebooks (one per block)."""
            Yhat = np.empty_like(Y)
            for gi, (a, b) in enumerate(blocks):
                sub = np.ascontiguousarray(Y[:, a:b], dtype=np.float32)
                C = codebooks[gi]

                # C is either (k, dim) for bb>0, or (1, dim) for bb==0
                if C.shape[0] == 1 or sub.shape[1] == 0:
                    Yhat[:, a:b] = C[0]
                    continue

                _, I = faiss.knn(sub, C, 1)
                Yhat[:, a:b] = C[I[:, 0]]
            return Yhat

        def _eval_mse_for_R(Rcand: np.ndarray) -> float:
            """Scheme A: codebooks trained on Yfit (Xfit@R), evaluate on Yev (Xev@R)."""
            Yfit = np.ascontiguousarray(Xfit @ Rcand, dtype=np.float32)
            codebooks = _train_codebooks(Yfit)

            if Xev is None or nev <= 0:
                Yhat_fit = _quantize_with_codebooks(Yfit, codebooks)
                return float(np.mean((Yfit - Yhat_fit) ** 2))

            Yev = np.ascontiguousarray(Xev @ Rcand, dtype=np.float32)
            Yhat_ev = _quantize_with_codebooks(Yev, codebooks)
            return float(np.mean((Yev - Yhat_ev) ** 2))

        # fixed-iter mode
        if self.niter > 0:
            total_iters = self.niter
            t0 = time.time()
            for it in range(total_iters):
                # Step ①: build reconstruction targets Yhat_fit using codebooks trained on Yfit
                Yfit = np.ascontiguousarray(Xfit @ R, dtype=np.float32)
                codebooks = _train_codebooks(Yfit)
                Yhat_fit = _quantize_with_codebooks(Yfit, codebooks)

                # Step ②: Procrustes update
                R_new = _orthogonal_procrustes(Xfit, Yhat_fit)

                if self.verbose:
                    mse = _eval_mse_for_R(R_new)  # report under consistent R_new
                    tag = "eval_mse" if (Xev is not None and nev > 0) else "train_mse"
                    print(f"[UnevenOPQ] iter {it + 1}/{total_iters} nfit={nfit} nev={nev} {tag}={mse:.6g}")

                R = R_new

            if self.verbose:
                print(f"[UnevenOPQ] done in {time.time() - t0:.3f}s (nfit={nfit}, nev={nev}, d={d})")

            self.R = np.ascontiguousarray(R, dtype=np.float32)
            return self.R

        # auto early stop
        patience = max(1, int(auto_patience))
        min_delta = float(auto_min_delta)
        max_iter = max(1, int(auto_max_iter))

        best = float("inf")
        bad = 0
        t0 = time.time()

        for it in range(max_iter):
            Yfit = np.ascontiguousarray(Xfit @ R, dtype=np.float32)
            codebooks = _train_codebooks(Yfit)
            Yhat_fit = _quantize_with_codebooks(Yfit, codebooks)
            R_new = _orthogonal_procrustes(Xfit, Yhat_fit)

            mse = _eval_mse_for_R(R_new)
            improved = (best - mse) > min_delta

            if improved:
                best = mse
                bad = 0
            else:
                bad += 1

            if self.verbose:
                tag = "eval_mse" if (Xev is not None and nev > 0) else "train_mse"
                print(
                    f"[UnevenOPQ] iter {it + 1}/auto nfit={nfit} nev={nev} {tag}={mse:.6g} "
                    f"best={best:.6g} bad={bad}/{patience}"
                )

            R = R_new
            if bad >= patience:
                break

        if self.verbose:
            print(f"[UnevenOPQ] auto-stop in {time.time() - t0:.3f}s (iters={it + 1}, d={d})")

        self.R = np.ascontiguousarray(R, dtype=np.float32)
        return self.R


# ============================================================
# ElasticPQ
# ============================================================

class ElasticPQ:
    """Grouped PQ with injected dimension grouper.
    Optional global UnevenOPQ rotation (encode map A).
    """

    def __init__(self, cfg: ElasticPQConfig, *, grouper: Grouper):
        self.cfg = cfg
        self.d = int(cfg.d)
        if self.d <= 0:
            raise ValueError("d must be positive")

        self.B = int(cfg.B)
        if self.B < 0:
            raise ValueError("B (total bits) must be non-negative")

        self.min_bits = int(cfg.min_bits)
        self.max_bits = int(cfg.max_bits)
        if self.min_bits < 0 or self.max_bits < 0 or self.min_bits > self.max_bits:
            raise ValueError("invalid min_bits/max_bits")

        self.kmeans_niter = int(cfg.kmeans_niter)
        self.kmeans_nredo = int(cfg.kmeans_nredo)

        self.enable_uneven_opq = bool(cfg.enable_uneven_opq)
        self.uneven_opq_niter = int(cfg.uneven_opq_niter)
        self.uneven_opq_kmeans_niter = int(cfg.uneven_opq_kmeans_niter)
        self.uneven_opq_kmeans_nredo = int(cfg.uneven_opq_kmeans_nredo)
        self.uneven_opq_max_train = int(cfg.uneven_opq_max_train)
        self.uneven_opq_max_eval = int(cfg.uneven_opq_max_eval)
        self.uneven_opq_eval_frac = float(cfg.uneven_opq_eval_frac)
        self.uneven_opq_seed = int(cfg.uneven_opq_seed)

        self.structure_save_path = str(cfg.structure_save_path or "")
        self.seed = int(cfg.seed)
        self.verbose = bool(cfg.verbose)

        self.grouper = grouper

        # learned state
        self.M: int = 0
        self.groups_orig: Groups = []
        self.nbits_per_group: Bits = []
        self.groups_contig: Groups = []

        self.perm: Optional[np.ndarray] = None
        self.ksub_per_group: List[int] = []
        self.codebooks: List[np.ndarray] = []

        self.global_A: Optional[np.ndarray] = None
        self.is_trained: bool = False
        self.last_structure_time: float = 0.0
        self.last_preparation_time: float = 0.0
        self.last_codebook_time: float = 0.0
        self.last_train_total_time: float = 0.0

    def _save_and_fix_structure(self, groups: Groups, nbits: Bits) -> None:
        if _is_fixed_grouper(self.grouper):
            return
        if not self.structure_save_path:
            return

        meta: Dict[str, Any] = {
            "created_at_unix": time.time(),
            "grouper_repr": repr(self.grouper),
        }
        try:
            meta["numpy"] = getattr(np, "__version__", "")
        except Exception:
            pass
        try:
            meta["faiss"] = getattr(faiss, "__version__", "")
        except Exception:
            pass

        struct = EPQStructure(
            d=self.d,
            B=self.B,
            groups=[list(g) for g in groups],
            nbits=[int(b) for b in nbits],
            meta=meta,
        )
        struct.save_json(self.structure_save_path)
        self.grouper = FixedStructureGrouper(struct)

        if self.verbose:
            print(f"[ElasticPQ] structure saved to {self.structure_save_path} and grouper fixed.")

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        n, d = xt.shape
        if d != self.d:
            raise ValueError(f"dimension mismatch: expected d={self.d}, got {d}")

        t0 = time.time()
        fixed_structure = _is_fixed_grouper(self.grouper)
        if self.verbose:
            print(f"[ElasticPQ] training n={n} d={d} B={self.B}")

        # 1) groups + bits (original dim ids)
        ctx = make_default_context_with_proxy(
            x=xt,
            d=self.d,
            B=self.B,
            bmax=self.max_bits,
            seed=self.seed,
        )
        groups, nbits = self.grouper.build_groups(ctx)
        self._validate_groups(groups)
        self._validate_bits(nbits, M=len(groups))

        self.groups_orig = [list(map(int, g)) for g in groups]
        self.nbits_per_group = [int(b) for b in nbits]
        self.M = len(self.groups_orig)
        self.ksub_per_group = [(1 << b) if b > 0 else 1 for b in self.nbits_per_group]

        sizes = [len(g) for g in self.groups_orig]
        self.groups_contig = _groups_as_contiguous_blocks(sizes)

        # Structure is fully determined here; if requested, persist it as part of the
        # structure phase so the reported phase sum matches the observed total better.
        if (not _is_fixed_grouper(self.grouper)) and self.structure_save_path:
            self._save_and_fix_structure(self.groups_orig, self.nbits_per_group)

        t_after_structure = time.time()
        self.last_structure_time = 0.0 if fixed_structure else (t_after_structure - t0)

        if self.verbose:
            print(f"[ElasticPQ] groups built: M={self.M} sizes={sizes} bits={self.nbits_per_group}")

        # 2) optional UnevenOPQ
        self.global_A = None
        xt_train = xt
        groups_for_training = self.groups_orig
        self.last_preparation_time = 0.0

        if self.enable_uneven_opq:
            t_rot0 = time.time()
            perm = _perm_from_groups(self.groups_orig, self.d)
            self.perm = perm

            if self.verbose:
                mode = "auto-stop" if self.uneven_opq_niter <= 0 else f"niter={self.uneven_opq_niter}"
                print(
                    "[ElasticPQ] fitting UnevenOPQ "
                    f"({mode}) "
                    f"kmeans_niter={self.uneven_opq_kmeans_niter} "
                    f"max_train={self.uneven_opq_max_train} "
                    f"max_eval={self.uneven_opq_max_eval} "
                    f"eval_frac={self.uneven_opq_eval_frac}"
                )

            xt_perm = np.ascontiguousarray(xt[:, perm], dtype=np.float32)

            uopq = UnevenOPQ(
                niter=self.uneven_opq_niter,
                kmeans_niter=self.uneven_opq_kmeans_niter,
                kmeans_nredo=self.uneven_opq_kmeans_nredo,
                max_train=self.uneven_opq_max_train,
                max_eval=self.uneven_opq_max_eval,
                eval_frac=self.uneven_opq_eval_frac,
                seed=self.uneven_opq_seed,
                verbose=self.verbose,
            )

            R_prime = uopq.fit(
                xt_perm,
                sizes,
                self.nbits_per_group,
            )

            Pmat = _perm_matrix_from_perm(perm)
            self.global_A = np.ascontiguousarray(Pmat @ R_prime, dtype=np.float32)

            xt_train = np.ascontiguousarray(xt @ self.global_A, dtype=np.float32)
            groups_for_training = self.groups_contig
            self.last_preparation_time = time.time() - t_rot0

        # 3) train per-group codebooks
        t_cb0 = time.time()
        self.codebooks = []
        for gi, dims in enumerate(groups_for_training):
            sub = np.ascontiguousarray(xt_train[:, dims], dtype=np.float32)
            dim_g = int(sub.shape[1])
            b = int(self.nbits_per_group[gi])
            ksub = int(self.ksub_per_group[gi])

            if b == 0 or ksub == 1 or dim_g == 0:
                c0 = sub.mean(axis=0, keepdims=True)
                self.codebooks.append(np.ascontiguousarray(c0, dtype=np.float32))
                continue

            km = faiss.Kmeans(
                dim_g,
                ksub,
                niter=self.kmeans_niter,
                nredo=self.kmeans_nredo,
                verbose=False,
            )
            try:
                km.cp.min_points_per_centroid = 1
            except Exception:
                pass
            km.train(sub)
            self.codebooks.append(np.ascontiguousarray(km.centroids, dtype=np.float32))

        self.is_trained = True
        self.last_train_total_time = time.time() - t0
        self.last_codebook_time = time.time() - t_cb0

        if self.verbose:
            print(
                f"[ElasticPQ] trained in {time.time() - t0:.3f}s "
                f"M={self.M} d={self.d} B={self.B} uneven_opq={self.enable_uneven_opq}"
            )

    def compute_codes(self, xb: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("ElasticPQ not trained. Call train(xt) first.")

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        n, d = xb.shape
        if d != self.d:
            raise ValueError(f"dimension mismatch: expected d={self.d}, got {d}")

        if self.global_A is not None:
            xb = np.ascontiguousarray(xb @ self.global_A, dtype=np.float32)
            groups = self.groups_contig
        else:
            groups = self.groups_orig

        codes = np.empty((n, self.M), dtype=_codes_dtype_for_bits(self.nbits_per_group))

        for gi, dims in enumerate(groups):
            if self.nbits_per_group[gi] == 0:
                codes[:, gi] = 0
                continue

            sub = np.ascontiguousarray(xb[:, dims], dtype=np.float32)
            centroids = self.codebooks[gi]
            _, I = faiss.knn(sub, centroids, 1)
            codes[:, gi] = I[:, 0].astype(codes.dtype, copy=False)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("ElasticPQ not trained. Call train(xt) first.")

        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.M:
            raise ValueError(f"codes shape must be (n, M={self.M}), got {codes.shape}")

        n = int(codes.shape[0])

        if self.global_A is not None:
            groups = self.groups_contig
            y_hat = np.zeros((n, self.d), dtype=np.float32)
        else:
            groups = self.groups_orig
            y_hat = np.zeros((n, self.d), dtype=np.float32)

        for gi, dims in enumerate(groups):
            centroids = self.codebooks[gi]
            idx = codes[:, gi].astype(np.int64, copy=False)
            y_hat[:, dims] = centroids[idx]

        if self.global_A is not None:
            return np.ascontiguousarray(y_hat @ self.global_A.T, dtype=np.float32)
        return y_hat

    # ------------------------
    # validation
    # ------------------------

    def _validate_groups(self, groups: Groups) -> None:
        if not groups:
            raise ValueError("grouper produced no groups")
        if any(len(g) == 0 for g in groups):
            raise ValueError("grouper produced an empty group")

        flat = [int(i) for g in groups for i in g]
        if len(flat) != self.d:
            raise ValueError(f"grouper must cover exactly d={self.d} dims, got {len(flat)}")
        if len(set(flat)) != len(flat):
            raise ValueError("grouper produced duplicated dims")

        want = set(range(self.d))
        got = set(flat)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(f"grouper produced invalid dim ids: missing={missing} extra={extra}")

    def _validate_bits(self, nbits: Bits, *, M: int) -> None:
        if len(nbits) != int(M):
            raise ValueError(f"bits must have length M={M}, got {len(nbits)}")

        nbits_i = [int(b) for b in nbits]
        if any(b < self.min_bits or b > self.max_bits for b in nbits_i):
            raise ValueError(f"bits outside bounds [{self.min_bits},{self.max_bits}]: {nbits_i}")

        s = int(sum(nbits_i))
        if s != self.B:
            raise ValueError(f"sum(bits) must equal B={self.B}, got {s}")
