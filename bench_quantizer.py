#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_codec_faiss_with_epq_adc.py

FAISS Index benchmark script + EPQ (ElasticPQ) with ADC (LUT) evaluation.

This version changes the CLI budget syntax:
- Instead of "Mxnbits" (e.g., 16x8), you pass ONLY total bit budget B (e.g., 128).
- For PQ/OPQ/RQ/LSQ/PRQ baselines: we fix nbits=8 and set M = B / 8.
  (FAISS IndexPQ requires uniform nbits per subquantizer, so B must be divisible by 8.)
- For BAPQ: follow the paper setting:
    group q dimensions per subspace, q=4 by default,
    so M_bapq = d / q (must be divisible; for SIFT(128) and GIST(960) it is).
  Then allocate total bits B unevenly across these M_bapq subspaces.

NEW in this revision
-------------------
1) Rename bapq_recall@k => overlap@k (overlap/coverage style metric).
2) Extend BenchIndex protocol to support query mode:
      - mode="adc" (asymmetric distance computation)
      - mode="sdc" (symmetric distance computation)
   EPQIndexADC implements both. FAISS baselines generally implement ADC; SDC is
   not uniformly exposed in Python, so we raise NotImplementedError for SDC there
   (explicit, loud, no silent behavior changes).

Examples
--------
  python bench_codec_faiss_with_epq_adc.py sift1M 128 pq opq epq repq bapq
  python bench_codec_faiss_with_epq_adc.py gist1M 64  pq opq epq repq bapq
  python bench_codec_faiss_with_epq_adc.py sift1M 128 epq --mode=sdc

"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any, Callable, Literal


_THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _get_flag_value(argv: Sequence[str], name: str) -> Optional[str]:
    for i in range(1, len(argv)):
        arg = argv[i]
        if arg == name:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for {name}")
            return argv[i + 1]
        prefix = f"{name}="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _parse_positive_int(raw: str, *, flag: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid {flag}={raw!r}; expected a positive integer") from exc
    if value <= 0:
        raise ValueError(f"Invalid {flag}={raw!r}; expected a positive integer")
    return value


def _parse_cpu_affinity(spec: str) -> Tuple[int, ...]:
    cpus = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                raise ValueError(f"Invalid CPU range {token!r}: start > end")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    ordered = tuple(sorted(cpus))
    if not ordered:
        raise ValueError("CPU affinity cannot be empty")
    if ordered[0] < 0:
        raise ValueError(f"Invalid CPU affinity {spec!r}; CPU ids must be >= 0")
    return ordered


def _set_thread_env(num_threads: int) -> None:
    value = str(int(num_threads))
    for key in _THREAD_ENV_VARS:
        os.environ[key] = value


def _set_process_affinity(cpus: Sequence[int]) -> None:
    cpu_set = tuple(sorted(set(int(cpu) for cpu in cpus)))
    if not cpu_set:
        raise ValueError("CPU affinity cannot be empty")

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_set))
        return

    if os.name == "nt":
        import ctypes

        if cpu_set[-1] >= 64:
            raise ValueError("Windows affinity mask in this script supports CPU ids 0-63 only")

        mask = 0
        for cpu in cpu_set:
            mask |= 1 << cpu

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise OSError("SetProcessAffinityMask failed")
        return

    raise NotImplementedError("CPU affinity control is not supported on this platform")


def _apply_early_runtime_from_argv(argv: Sequence[str]) -> None:
    threads_raw = _get_flag_value(argv, "--threads")
    affinity_raw = _get_flag_value(argv, "--cpu-affinity") or _get_flag_value(argv, "--affinity")

    if threads_raw is not None:
        _set_thread_env(_parse_positive_int(threads_raw, flag="--threads"))
    if affinity_raw is not None:
        _set_process_affinity(_parse_cpu_affinity(affinity_raw))


if __name__ == "__main__":
    _apply_early_runtime_from_argv(sys.argv)

import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M, DatasetGIST1M, DatasetDeep1B, DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M, DatasetGIST1M, DatasetDeep1B, DatasetBigANN, DatasetGlove


# ============================================================
# Metrics
# ============================================================

QueryMode = Literal["adc", "sdc"]
EPQStage = Literal["grow", "crystallize", "mbeam"]
EPQ_STAGE_ORDER: Tuple[EPQStage, ...] = ("grow", "crystallize", "mbeam")


def _parse_epq_stages(spec: str) -> Tuple[EPQStage, ...]:
    raw = str(spec).strip().lower()
    if raw in ("", "full", "default", "all"):
        return EPQ_STAGE_ORDER
    if raw in ("none", "off"):
        return ()

    aliases = {
        "grow": "grow",
        "g": "grow",
        "crystallize": "crystallize",
        "crystal": "crystallize",
        "cryst": "crystallize",
        "c": "crystallize",
        "mbeam": "mbeam",
        "beam": "mbeam",
        "mb": "mbeam",
        "marginal-beam": "mbeam",
        "marginal_beam": "mbeam",
        "marginalbeam": "mbeam",
    }

    seen = set()
    out: List[EPQStage] = []
    for token in raw.replace("+", ",").split(","):
        key = token.strip()
        if not key:
            continue
        canonical = aliases.get(key)
        if canonical is None:
            raise ValueError(
                f"Invalid EPQ stage {token!r}; expected a subset of "
                "{grow, crystallize, mbeam} or one of full/none"
            )
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)  # type: ignore[arg-type]

    ordered = [stage for stage in EPQ_STAGE_ORDER if stage in seen]
    return tuple(ordered)


def _epq_stages_label(stages: Sequence[EPQStage]) -> str:
    stages_list = [str(stage) for stage in stages]
    if stages_list == list(EPQ_STAGE_ORDER):
        return "grow -> crystallize -> marginal beam search"
    if not stages_list:
        return "singleton-init -> solve_bits"

    parts: List[str] = []
    if "grow" in stages_list:
        parts.append("grow")
    else:
        parts.append("singleton-init")
    if "crystallize" in stages_list:
        parts.append("crystallize")
    if "mbeam" in stages_list:
        parts.append("marginal beam search")
    return " -> ".join(parts)


def _epq_structure_cache_path(d: int, B: int, stages: Sequence[EPQStage]) -> str:
    default = f"data/{d}d_{B}B_epq_structure.json"
    stages_list = [str(stage) for stage in stages]
    if stages_list == list(EPQ_STAGE_ORDER):
        return default
    if not stages_list:
        suffix = "singleton"
    elif "grow" not in stages_list:
        suffix = "singleton_" + "_".join(stages_list)
    else:
        suffix = "_".join(stages_list)
    return f"data/{d}d_{B}B_epq_structure__{suffix}.json"


def _recall_at_k(I: np.ndarray, gt0: np.ndarray, k: int) -> float:
    """1-NN hit-rate recall@k: P(gt[0] in top-k)."""
    k = int(k)
    if k <= 0:
        return 0.0
    k = min(k, int(I.shape[1]))
    hit = (I[:, :k] == gt0[:, None]).any(axis=1)
    return float(hit.mean())


def _report_recalls(I: np.ndarray, gt: np.ndarray, *, Ks=(1, 10, 100, 1000)) -> str:
    gt0 = gt[:, 0].astype(np.int64, copy=False)
    parts = []
    for k in Ks:
        parts.append(f"recall@{k}: {_recall_at_k(I, gt0, k):.4f}")
    return " ".join(parts)


def _overlap_at_k(I: np.ndarray, gt: np.ndarray, k: int, *, gt_k: int = 100) -> float:
    """Overlap@k (coverage-style):
        overlap@k = avg_i |TopK_i ∩ GT_i(gt_use)| / gt_use
    where gt_use = min(gt_k, gt.shape[1]).

    This matches BAPQ-style evaluation (set overlap / coverage), NOT 1-NN hit-rate.
    """
    k = int(k)
    if k <= 0:
        return 0.0
    k = min(k, int(I.shape[1]))

    if gt.ndim != 2:
        raise ValueError(f"gt must be 2D, got shape={gt.shape}")

    gt_use = int(min(int(gt_k), int(gt.shape[1])))
    if gt_use <= 0:
        return 0.0

    gt_use_mat = np.ascontiguousarray(gt[:, :gt_use])
    I_use = np.ascontiguousarray(I[:, :k])

    nq = int(I_use.shape[0])
    denom = float(gt_use)

    hits = np.empty((nq,), dtype=np.float32)
    for i in range(nq):
        gti = set(int(x) for x in gt_use_mat[i].tolist())
        cnt = 0
        for x in I_use[i].tolist():
            if int(x) in gti:
                cnt += 1
        hits[i] = cnt / denom

    return float(hits.mean())


def _report_overlaps(I: np.ndarray, gt: np.ndarray, *, Ks=(1000,), gt_k: int = 1000) -> str:
    gt_cols = int(gt.shape[1]) if gt.ndim == 2 else -1
    gt_use = min(int(gt_k), gt_cols) if gt_cols > 0 else int(gt_k)

    parts = []
    for k in Ks:
        parts.append(f"overlap@{k}(gt={gt_use}): {_overlap_at_k(I, gt, k, gt_k=gt_k):.4f}")
    return " ".join(parts)


def _sample_indices(n: int, ns: int, seed: int = 123) -> np.ndarray:
    ns = int(ns)
    if ns <= 0 or ns >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    return rng.choice(n, size=ns, replace=False).astype(np.int64, copy=False)


# ============================================================
# BenchIndex protocol (duck-typed) with mode
# ============================================================

@dataclass
class TrainStats:
    structure_time: float = 0.0
    preparation_time: float = 0.0
    codebook_time: float = 0.0
    total_training_time: float = 0.0


class BenchIndex:
    def train(self, xt: np.ndarray) -> None: ...
    def add(self, xb: np.ndarray) -> None: ...
    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]: ...
    def get_train_stats(self) -> Optional[TrainStats]: ...


def _normalize_train_stats(stats: Optional[TrainStats], fallback_total: float) -> TrainStats:
    total = float(max(0.0, fallback_total))
    if stats is None:
        return TrainStats(
            structure_time=0.0,
            preparation_time=0.0,
            codebook_time=total,
            total_training_time=total,
        )

    structure = float(max(0.0, getattr(stats, "structure_time", 0.0)))
    preparation = float(max(0.0, getattr(stats, "preparation_time", 0.0)))
    codebook = float(max(0.0, getattr(stats, "codebook_time", 0.0)))
    stats_total = float(max(0.0, getattr(stats, "total_training_time", 0.0)))
    total_final = stats_total if stats_total > 0.0 else total

    if structure + preparation + codebook <= 0.0:
        codebook = total_final
    elif abs((structure + preparation + codebook) - total_final) > 1e-6:
        total_final = structure + preparation + codebook

    return TrainStats(
        structure_time=structure,
        preparation_time=preparation,
        codebook_time=codebook,
        total_training_time=total_final,
    )


# ============================================================
# FAISS wrappers
# ============================================================

class FaissIndexWrapper(BenchIndex):
    """Wrap a faiss.Index to present a unified interface and report timings.

    Notes on mode:
      - Most FAISS PQ-style indices expose ADC via Index.search in Python.
      - SDC is not consistently exposed via a simple flag across indices in Python bindings.
        Therefore, mode="sdc" raises NotImplementedError here (explicit by design).
    """

    def __init__(self, index: faiss.Index, *, name: str = "faiss"):
        self.index = index
        self.name = str(name)
        self._last_train_stats: Optional[TrainStats] = None

    def train(self, xt: np.ndarray) -> None:
        t0 = time.time()
        self.index.train(np.ascontiguousarray(xt, dtype=np.float32))
        total = time.time() - t0
        self._last_train_stats = TrainStats(
            structure_time=0.0,
            preparation_time=0.0,
            codebook_time=total,
            total_training_time=total,
        )

    def add(self, xb: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(xb, dtype=np.float32))

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        mode = str(mode).lower()
        if mode != "adc":
            raise NotImplementedError(
                f"{self.name}.search(mode={mode!r}) not supported via this wrapper. "
                "FAISS Python bindings do not provide a uniform SDC switch for these indices."
            )
        D, I = self.index.search(np.ascontiguousarray(xq, dtype=np.float32), int(k))
        return D, I

    def get_train_stats(self) -> Optional[TrainStats]:
        return self._last_train_stats


class ProductQuantizerADCIndex(BenchIndex):
    """Pure-Python ADC benchmark path for PQ / OPQ using extracted codebooks."""

    def __init__(
        self,
        d: int,
        M: int,
        nbits: int,
        *,
        name: str,
        use_opq: bool = False,
        lut_chunk: int = 4096,
        query_batch: int = 32,
    ):
        self.d = int(d)
        self.M = int(M)
        self.nbits = int(nbits)
        self.name = str(name)
        self.use_opq = bool(use_opq)
        self.lut_chunk = int(max(128, lut_chunk))
        self.query_batch = int(max(1, query_batch))

        self.d2 = ((self.d + self.M - 1) // self.M) * self.M if self.use_opq else self.d
        if self.d2 % self.M != 0:
            raise ValueError(f"invalid d2/M for PQ ADC index: d2={self.d2}, M={self.M}")
        if self.d % self.M != 0 and not self.use_opq:
            raise ValueError(f"PQ requires d % M == 0, got d={self.d}, M={self.M}")

        self.dsub = self.d2 // self.M
        self.ksub = 1 << self.nbits

        self.A: Optional[np.ndarray] = None
        self.codebooks: Optional[np.ndarray] = None  # (M, ksub, dsub)
        self.codes_db: Optional[np.ndarray] = None
        self.nb: int = 0
        self._last_train_stats: Optional[TrainStats] = None

    @staticmethod
    def _codes_dtype_for_nbits(nbits: int) -> np.dtype:
        if nbits <= 8:
            return np.uint8
        if nbits <= 16:
            return np.uint16
        return np.uint32

    def _require_trained(self) -> None:
        if self.codebooks is None:
            raise RuntimeError(f"{self.name} is not trained.")

    def _apply_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.A is None:
            return x
        return np.ascontiguousarray(x @ self.A.T, dtype=np.float32)

    def project(self, x: np.ndarray) -> np.ndarray:
        return self._apply_transform(x)

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        if xt.ndim != 2 or xt.shape[1] != self.d:
            raise ValueError(f"xt must have shape (n, {self.d}), got {xt.shape}")

        t0 = time.time()
        x_train = xt
        self.A = None
        rotation_time = 0.0

        if self.use_opq:
            t_rot0 = time.time()
            opq = faiss.OPQMatrix(self.d, self.M, self.d2)
            opq.train(xt)
            self.A = faiss.vector_to_array(opq.A).astype(np.float32, copy=False).reshape(self.d2, self.d)
            if hasattr(opq, "apply_py"):
                x_train = np.ascontiguousarray(opq.apply_py(xt), dtype=np.float32)
            else:
                x_train = np.ascontiguousarray(opq.apply(xt), dtype=np.float32)
            rotation_time = time.time() - t_rot0

        t_cb0 = time.time()
        pq = faiss.ProductQuantizer(self.d2, self.M, self.nbits)
        pq.train(x_train)
        centroids = faiss.vector_to_array(pq.centroids).astype(np.float32, copy=False)
        self.codebooks = np.ascontiguousarray(centroids.reshape(self.M, self.ksub, self.dsub), dtype=np.float32)
        self.codes_db = None
        self.nb = 0
        codebook_time = time.time() - t_cb0

        total = time.time() - t0
        self._last_train_stats = TrainStats(
            structure_time=0.0,
            preparation_time=rotation_time,
            codebook_time=codebook_time,
            total_training_time=total,
        )

    def compute_codes(self, xb: np.ndarray) -> np.ndarray:
        self._require_trained()
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.d:
            raise ValueError(f"xb must have shape (n, {self.d}), got {xb.shape}")

        x_work = self._apply_transform(xb).reshape(xb.shape[0], self.M, self.dsub)
        dtype = self._codes_dtype_for_nbits(self.nbits)
        codes = np.empty((xb.shape[0], self.M), dtype=dtype)

        assert self.codebooks is not None
        for m in range(self.M):
            sub = np.ascontiguousarray(x_work[:, m, :], dtype=np.float32)
            _, I = faiss.knn(sub, self.codebooks[m], 1)
            codes[:, m] = I[:, 0].astype(dtype, copy=False)
        return codes

    def add(self, xb: np.ndarray) -> None:
        codes = np.ascontiguousarray(self.compute_codes(xb))
        self.codes_db = codes
        self.nb = int(codes.shape[0])

    def _build_lut_q_to_C(self, qsub: np.ndarray, C: np.ndarray) -> np.ndarray:
        qsub = np.ascontiguousarray(qsub, dtype=np.float32)
        C = np.ascontiguousarray(C, dtype=np.float32)
        qn = np.sum(qsub * qsub, axis=1, keepdims=True)
        cn = np.sum(C * C, axis=1, keepdims=True).T
        out = np.empty((qsub.shape[0], C.shape[0]), dtype=np.float32)

        for a in range(0, C.shape[0], self.lut_chunk):
            z = min(int(C.shape[0]), a + self.lut_chunk)
            out[:, a:z] = (qn + cn[:, a:z] - 2.0 * (qsub @ C[a:z].T)).astype(np.float32, copy=False)
        return out

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        self._require_trained()
        if self.codes_db is None:
            raise RuntimeError(f"{self.name}.add(xb) must be called before search().")
        if str(mode).lower() != "adc":
            raise NotImplementedError(f"{self.name} supports mode='adc' only in this benchmark.")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.d:
            raise ValueError(f"xq must have shape (n, {self.d}), got {xq.shape}")

        xq_work = self._apply_transform(xq).reshape(xq.shape[0], self.M, self.dsub)
        nq = int(xq.shape[0])
        nb = int(self.nb)
        k = int(k)

        I_all = np.empty((nq, k), dtype=np.int64)
        D_all = np.empty((nq, k), dtype=np.float32)

        assert self.codebooks is not None
        assert self.codes_db is not None

        for q0 in range(0, nq, self.query_batch):
            q1 = min(nq, q0 + self.query_batch)
            luts: List[np.ndarray] = []
            for m in range(self.M):
                luts.append(self._build_lut_q_to_C(xq_work[q0:q1, m, :], self.codebooks[m]))

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

    def get_train_stats(self) -> Optional[TrainStats]:
        return self._last_train_stats


# ============================================================
# EPQ ADC/SDC Index
# ============================================================

class EPQIndexADC(BenchIndex):
    """ADC/SDC Index for ElasticPQ.

    Stores only database codes.

    mode="adc":
        dist(q, code_i) = sum_g ||q_g - c_{g, code_i[g]}||^2
        (classic LUT / ADC)

    mode="sdc":
        dist(code_q, code_i) = sum_g ||c_{g, code_q[g]} - c_{g, code_i[g]}||^2
        (symmetric: quantize the query too)
    """

    def __init__(
        self,
        epq: Any,
        *,
        name: str = "epq",
        lut_chunk: int = 4096,
        query_batch: int = 32,
        seed: int = 123,
    ):
        self.epq = epq
        self.name = str(name)
        self.lut_chunk = int(max(128, lut_chunk))
        self.query_batch = int(max(1, query_batch))
        self.seed = int(seed)

        self.codes_db: Optional[np.ndarray] = None  # (nb, M) uint*
        self.nb: int = 0

        self._groups_for_codes: Optional[List[List[int]]] = None
        self._codebooks: Optional[List[np.ndarray]] = None
        self._M: int = 0
        self._last_train_stats: Optional[TrainStats] = None

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        A = getattr(self.epq, "global_A", None)
        if A is None:
            return x
        return np.ascontiguousarray(x @ np.ascontiguousarray(A, dtype=np.float32), dtype=np.float32)

    def train(self, xt: np.ndarray) -> None:
        xt = np.ascontiguousarray(xt, dtype=np.float32)
        t0 = time.time()
        self.epq.train(xt)
        total = time.time() - t0

        if getattr(self.epq, "global_A", None) is not None:
            groups = self.epq.groups_contig
        else:
            groups = self.epq.groups_orig

        if groups is None:
            raise RuntimeError("ElasticPQ missing groups/groups_orig attributes needed for index.")

        codebooks = getattr(self.epq, "codebooks", None)
        if codebooks is None or not isinstance(codebooks, list) or len(codebooks) == 0:
            raise RuntimeError("ElasticPQ missing codebooks after training.")

        self._groups_for_codes = [list(map(int, g)) for g in groups]
        self._codebooks = [np.ascontiguousarray(C, dtype=np.float32) for C in codebooks]
        self._M = int(len(self._groups_for_codes))
        self._last_train_stats = _normalize_train_stats(
            TrainStats(
                structure_time=float(getattr(self.epq, "last_structure_time", 0.0)),
                preparation_time=float(getattr(self.epq, "last_preparation_time", 0.0)),
                codebook_time=float(getattr(self.epq, "last_codebook_time", 0.0)),
                total_training_time=float(getattr(self.epq, "last_train_total_time", 0.0)),
            ),
            total,
        )

    def add(self, xb: np.ndarray) -> None:
        if self._groups_for_codes is None or self._codebooks is None:
            raise RuntimeError("EPQIndexADC must be trained before add().")

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        codes = self.epq.compute_codes(xb)
        codes = np.ascontiguousarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self._M:
            raise RuntimeError(f"codes shape mismatch: got {codes.shape}, expected (*,{self._M})")
        self.codes_db = codes
        self.nb = int(codes.shape[0])

    def _build_lut_q_to_C(self, qg: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return LUT[b, k] = ||qg[b] - C[k]||^2."""
        qg = np.ascontiguousarray(qg, dtype=np.float32)
        C = np.ascontiguousarray(C, dtype=np.float32)
        b, _dim = qg.shape
        k = int(C.shape[0])

        qn = np.sum(qg * qg, axis=1, keepdims=True)       # (b,1)
        cn = np.sum(C * C, axis=1, keepdims=True).T       # (1,k)

        out = np.empty((b, k), dtype=np.float32)

        step = self.lut_chunk
        for a in range(0, k, step):
            z = min(k, a + step)
            Cc = C[a:z]
            dot = qg @ Cc.T
            out[:, a:z] = (qn + cn[:, a:z] - 2.0 * dot).astype(np.float32, copy=False)
        return out

    def _build_lut_code_to_C(self, C: np.ndarray) -> np.ndarray:
        """Return LUT[k, k] = ||C[i] - C[j]||^2 (symmetric distance table)."""
        C = np.ascontiguousarray(C, dtype=np.float32)
        k = int(C.shape[0])

        cn = np.sum(C * C, axis=1, keepdims=True)  # (k,1)
        # dist(i,j) = ||Ci||^2 + ||Cj||^2 - 2 Ci·Cj
        dist = cn + cn.T - 2.0 * (C @ C.T)
        return np.ascontiguousarray(dist.astype(np.float32, copy=False))

    def _decode_from_codes(self, codes: np.ndarray) -> np.ndarray:
        if self._groups_for_codes is None or self._codebooks is None:
            raise RuntimeError("train() first")
        codes = np.ascontiguousarray(codes)
        n = int(codes.shape[0])

        d_code = 0
        for g in self._groups_for_codes:
            d_code = max(d_code, max(g) + 1 if g else d_code)
        x = np.zeros((n, d_code), dtype=np.float32)
        for gi, dims in enumerate(self._groups_for_codes):
            C = self._codebooks[gi]
            idx = codes[:, gi].astype(np.int64, copy=False)
            x[:, dims] = C[idx]

        A = getattr(self.epq, "global_A", None)
        if A is not None:
            x = np.ascontiguousarray(x @ np.ascontiguousarray(A, dtype=np.float32).T, dtype=np.float32)
        return x

    def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
        if self.codes_db is None:
            raise RuntimeError("EPQIndexADC.add(xb) must be called before search().")
        if self._groups_for_codes is None or self._codebooks is None:
            raise RuntimeError("EPQIndexADC.train(xt) must be called before search().")

        mode = str(mode).lower()
        if mode not in ("adc", "sdc"):
            raise ValueError(f"mode must be 'adc' or 'sdc', got {mode!r}")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        k = int(k)
        nq = int(xq.shape[0])
        M = self._M
        nb = self.nb

        # apply global rotation if present
        xq2 = self._apply_A(xq)

        codes_db = self.codes_db
        assert codes_db is not None

        I_all = np.empty((nq, k), dtype=np.int64)
        D_all = np.empty((nq, k), dtype=np.float32)

        qb = self.query_batch

        if mode == "adc":
            # ADC: LUT per query batch per group (q -> centroids)
            for q0 in range(0, nq, qb):
                q1 = min(nq, q0 + qb)
                bsz = q1 - q0

                luts_q: List[np.ndarray] = []
                for gi, dims in enumerate(self._groups_for_codes):
                    C = self._codebooks[gi]
                    qg = np.ascontiguousarray(xq2[q0:q1, dims], dtype=np.float32)
                    lut = self._build_lut_q_to_C(qg, C)  # (bsz, k_g)
                    luts_q.append(lut)

                for bi in range(bsz):
                    dist = np.zeros((nb,), dtype=np.float32)
                    for gi in range(M):
                        idx = codes_db[:, gi].astype(np.int64, copy=False)
                        dist += luts_q[gi][bi, idx]

                    if k >= nb:
                        order = np.argsort(dist, kind="stable")
                        sel = order[:k]
                    else:
                        sel = np.argpartition(dist, k)[:k]
                        sel = sel[np.argsort(dist[sel], kind="stable")]

                    I_all[q0 + bi] = sel.astype(np.int64, copy=False)
                    D_all[q0 + bi] = dist[sel].astype(np.float32, copy=False)

            return D_all, I_all

        # SDC: quantize query to codes first, then use centroid-to-centroid LUTs
        # Precompute per-group centroid distance tables (k_g x k_g)
        dist_tables: List[np.ndarray] = []
        for gi in range(M):
            C = self._codebooks[gi]
            dist_tables.append(self._build_lut_code_to_C(C))

        # compute query codes on rotated space
        # (compute_codes expects original x; EPQ may internally handle global_A.
        #  To stay consistent with database code generation, feed the original xq here.
        #  This mirrors add(xb), which also passes xb directly to compute_codes rather than xb@A.
        try:
            codes_q = self.epq.compute_codes(xq)
        except Exception:
            # fallback: try rotated queries
            codes_q = self.epq.compute_codes(xq2)
        codes_q = np.ascontiguousarray(codes_q)

        if codes_q.ndim != 2 or codes_q.shape[1] != M:
            raise RuntimeError(f"query codes shape mismatch: got {codes_q.shape}, expected (*,{M})")

        for q0 in range(0, nq, qb):
            q1 = min(nq, q0 + qb)
            for qi in range(q0, q1):
                dist = np.zeros((nb,), dtype=np.float32)
                for gi in range(M):
                    cq = int(codes_q[qi, gi])
                    idx_db = codes_db[:, gi].astype(np.int64, copy=False)
                    dist += dist_tables[gi][cq, idx_db]

                if k >= nb:
                    order = np.argsort(dist, kind="stable")
                    sel = order[:k]
                else:
                    sel = np.argpartition(dist, k)[:k]
                    sel = sel[np.argsort(dist[sel], kind="stable")]

                I_all[qi] = sel.astype(np.int64, copy=False)
                D_all[qi] = dist[sel].astype(np.float32, copy=False)

        return D_all, I_all

    def get_train_stats(self) -> Optional[TrainStats]:
        return self._last_train_stats


# ============================================================
# EPQ builders
# ============================================================

def _resolve_structure_path(structure: str) -> Path:
    raw = Path(structure)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(Path("result") / "structure" / raw)
        if raw.suffix != ".json":
            candidates.append(raw.with_suffix(".json"))
            candidates.append(Path("result") / "structure" / raw.with_suffix(".json"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"structure file not found; tried: {tried}")


def build_epq(
    d: int,
    *,
    B: int,
    seed: int = 123,
    verbose: bool = True,
    structure: Optional[str] = None,
    enable_uneven_opq: bool = True,
    stages: Sequence[EPQStage] = EPQ_STAGE_ORDER,
):
    """Construct an ElasticPQ instance."""
    path = _epq_structure_cache_path(d=int(d), B=int(B), stages=stages)
    from elastic_pq import ElasticPQ, ElasticPQConfig

    if structure:
        from grouper import FixedStructureGrouper, EPQStructure

        structure_path = _resolve_structure_path(structure)
        if verbose:
            print(f"[EPQ] loading fixed structure from {structure_path}")
        grouper = FixedStructureGrouper(EPQStructure.load_json(str(structure_path)))
    else:
        from grouper import SingletonDimGrouper
        from grouper_grow import ClusterGrowGrouper, ClusterGrowGrouperConfig
        from forwarder_cryst import CrystallizationForwarder, CrystallizationForwarderConfig
        from forwarder_mc import MarginalBeamForwarder, MarginalBeamForwarderConfig

        stage_set = set(str(stage) for stage in stages)

        if "grow" in stage_set:
            grouper = ClusterGrowGrouper(ClusterGrowGrouperConfig())
        else:
            grouper = SingletonDimGrouper()

        if "crystallize" in stage_set:
            grouper = grouper.then(
                CrystallizationForwarder(
                    CrystallizationForwarderConfig(verbose=bool(verbose))
                )
            )
        if verbose:
            print(f"[EPQ] grouper pipeline: {_epq_stages_label(stages)}")
        if "mbeam" in stage_set:
            grouper = grouper.then(
                MarginalBeamForwarder(
                    MarginalBeamForwarderConfig(
                        verbose=bool(verbose),
                        seed=int(seed),
                    )
                )
            )

    cfg = ElasticPQConfig(
        d=int(d),
        B=int(B),
        enable_uneven_opq=bool(enable_uneven_opq),
        structure_save_path=path,
        verbose=bool(verbose),
        seed=int(seed),
    )
    return ElasticPQ(cfg, grouper=grouper)


def _balanced_contiguous_groups(d: int, M: int) -> List[List[int]]:
    d = int(d)
    M = int(M)
    if d <= 0 or M <= 0:
        raise ValueError(f"invalid d/M for groups: d={d}, M={M}")
    base = d // M
    extra = d % M
    groups: List[List[int]] = []
    cur = 0
    for i in range(M):
        sz = base + (1 if i < extra else 0)
        if sz <= 0:
            raise ValueError(f"too many groups for d={d}, M={M}")
        groups.append(list(range(cur, cur + sz)))
        cur += sz
    return groups


def _make_proxy_ctx(x: np.ndarray, *, d: int, B: int, bmax: int, seed: int = 123):
    from grouper import make_default_context_with_proxy

    return make_default_context_with_proxy(
        x=np.ascontiguousarray(x, dtype=np.float32),
        d=int(d),
        B=int(B),
        bmax=int(max(0, bmax)),
        seed=int(seed),
    )


def _epq_groups_for_proxy(epq: Any) -> Sequence[Sequence[int]]:
    return epq.groups_contig if getattr(epq, "global_A", None) is not None else epq.groups_orig


def _epq_space_for_proxy(epq: Any, x: np.ndarray) -> np.ndarray:
    x2 = np.ascontiguousarray(x, dtype=np.float32)
    A = getattr(epq, "global_A", None)
    if A is None:
        return x2
    return np.ascontiguousarray(x2 @ np.ascontiguousarray(A, dtype=np.float32), dtype=np.float32)


def _print_group_proxy_stats(
    *,
    quantizer_name: str,
    groups: Sequence[Sequence[int]],
    bits: Sequence[int],
    proxy,
    entry_label: str = "group",
    space_label: str = "original",
) -> None:
    if len(groups) != len(bits):
        raise ValueError(
            f"group/bits length mismatch for {quantizer_name}: "
            f"len(groups)={len(groups)} len(bits)={len(bits)}"
        )

    total_dims = int(sum(len(g) for g in groups))
    total_bits = int(sum(int(b) for b in bits))
    print(
        f"\t[group-stats] quantizer={quantizer_name} space={space_label} "
        f"entries={len(groups)} total_dims={total_dims} total_bits={total_bits}"
    )

    J_proxy = 0.0
    for i, (g, b) in enumerate(zip(groups, bits)):
        ndims = int(len(g))
        bits_i = int(b)
        D_proxy = float(proxy.D(list(map(int, g)), bits_i))
        J_proxy += D_proxy
        print(
            f"\t[group-stats] {entry_label}[{i:03d}] "
            f"ndims={ndims} bits={bits_i} D_proxy={D_proxy:.6f}"
        )

    print(f"\t[group-stats] J_proxy={J_proxy:.6f}")


def _make_group_stats_task(
    *,
    quantizer_name: str,
    groups: Sequence[Sequence[int]] | Callable[[], Sequence[Sequence[int]]],
    bits: Sequence[int] | Callable[[], Sequence[int]],
    proxy_input_fn: Callable[[], np.ndarray],
    proxy_d: int,
    seed: int = 123,
    entry_label: str = "group",
    space_label: str = "original",
) -> Callable[[], None]:
    proxy_d_i = int(proxy_d)
    seed_i = int(seed)
    entry_label_s = str(entry_label)
    space_label_s = str(space_label)
    quantizer_name_s = str(quantizer_name)

    def _run() -> None:
        groups_val = groups() if callable(groups) else groups
        bits_val = bits() if callable(bits) else bits
        groups_copy = [list(map(int, g)) for g in groups_val]
        bits_copy = [int(b) for b in bits_val]
        proxy_ctx = _make_proxy_ctx(
            proxy_input_fn(),
            d=proxy_d_i,
            B=int(sum(bits_copy)),
            bmax=int(max(bits_copy) if bits_copy else 0),
            seed=seed_i,
        )
        _print_group_proxy_stats(
            quantizer_name=quantizer_name_s,
            groups=groups_copy,
            bits=bits_copy,
            proxy=proxy_ctx.require_proxy(),
            entry_label=entry_label_s,
            space_label=space_label_s,
        )

    return _run


# ============================================================
# Eval (Index style)
# ============================================================

def eval_index(
    index: BenchIndex,
    *,
    name: str,
    xq: np.ndarray,
    xb: np.ndarray,
    gt: np.ndarray,
    xt: np.ndarray,
    k: int = 1000,
    query_mode: QueryMode = "adc",
    recon_sample: int = 200000,
    recon_seed: int = 123,
    recon_fn: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
    pre_test_fn: Optional[Callable[[], None]] = None,
) -> None:
    print(f"===== {name} (mode={query_mode})")

    t0 = time.time()
    index.train(xt)
    t1 = time.time()
    stats_time = 0.0
    if pre_test_fn is not None:
        ts = time.time()
        pre_test_fn()
        stats_time = time.time() - ts

    index.add(xb)
    t2 = time.time()

    D, I = index.search(xq, k, mode=query_mode)
    t3 = time.time()
    train_stats = _normalize_train_stats(
        getattr(index, "get_train_stats", lambda: None)(),
        t1 - t0,
    )
    add_time = float(t2 - t1 - stats_time)
    search_time = float(t3 - t2)
    nb = int(xb.shape[0])
    nq = int(xq.shape[0])
    encode_per_vector = add_time / nb if nb > 0 else float("nan")
    search_per_query = search_time / nq if nq > 0 else float("nan")
    qps = nq / search_time if search_time > 0 else float("inf")

    print(f"\tstructure time: {train_stats.structure_time:.3f} s")
    print(f"\tpreparation time: {train_stats.preparation_time:.3f} s")
    print(f"\tcodebook time: {train_stats.codebook_time:.3f} s")
    print(f"\ttraining total: {train_stats.total_training_time:.3f} s")
    if pre_test_fn is not None:
        print(f"\tgroup-stats time: {stats_time:.3f} s")
    print(f"\tadd/encode time: {add_time:.3f} s")
    print(f"\tencode per vector: {encode_per_vector:.9f} s/vector")
    print(f"\tsearch time: {search_time:.3f} s")
    print(f"\tsearch per query: {search_per_query:.9f} s/query")
    print(f"\tQPS: {qps:.3f} queries/s")
    print(f"\t{_report_recalls(I, gt, Ks=(1, 10, 100, 1000))}")
    print(f"\t{_report_overlaps(I, gt, Ks=(1000,), gt_k=1000)}")

    if recon_fn is not None:
        xb_s, xb_dec_s = recon_fn()
        recons_err = float(((xb_s - xb_dec_s) ** 2).sum() / xb_s.shape[0])
        err_compat = float(np.linalg.norm(xb_s - xb_dec_s, axis=1).mean())
        print(f"\treconstruction error (sample): {recons_err:.3f} recons_err_compat(sample): {err_compat:.3f}")


# ============================================================
# CLI / dataset
# ============================================================

@dataclass
class Args:
    dataset: str
    B: int
    targets: List[str]
    # fixed for PQ/OPQ family in this benchmark
    pq_nbits: int = 8
    # BAPQ paper setting
    bapq_q: int = 4
    # query mode
    mode: QueryMode = "adc"
    print_group_stats: bool = False
    epq_structure: Optional[str] = None
    repq_structure: Optional[str] = None
    epq_stages: Tuple[EPQStage, ...] = EPQ_STAGE_ORDER
    threads: Optional[int] = None
    cpu_affinity: Optional[str] = None


def _parse_args(argv: Sequence[str]) -> Args:
    todo = list(argv[1:])
    if len(todo) < 2:
        print(
            "usage: bench_codec_faiss_with_epq_adc.py dataset B target1 target2 ... "
            "[--mode=adc|sdc] [--print-group-stats] "
            "[--epq-structure=name-or-path] [--repq-structure=name-or-path] "
            "[--epq-stages=full|none|grow,crystallize,mbeam] "
            "[--threads=N] [--cpu-affinity=0,1,2-5]"
        )
        print("example: bench_codec_faiss_with_epq_adc.py sift1M 128 pq opq epq repq bapq")
        sys.exit(1)

    dataset = todo.pop(0)
    try:
        B = int(todo.pop(0))
    except Exception:
        print("Second argument must be integer total bit budget B (e.g., 64, 128, ...)")
        sys.exit(1)

    if not todo:
        print("no targets specified")
        sys.exit(1)

    mode: QueryMode = "adc"
    print_group_stats = False
    epq_structure: Optional[str] = None
    repq_structure: Optional[str] = None
    epq_stages = EPQ_STAGE_ORDER
    threads: Optional[int] = None
    cpu_affinity: Optional[str] = None
    targets: List[str] = []
    for t in todo:
        if t.startswith("--mode="):
            v = t.split("=", 1)[1].strip().lower()
            if v not in ("adc", "sdc"):
                raise ValueError(f"Invalid --mode={v!r}, expected adc or sdc")
            mode = v  # type: ignore[assignment]
        elif t.startswith("--structure="):
            epq_structure = t.split("=", 1)[1].strip() or None
        elif t.startswith("--epq-structure="):
            epq_structure = t.split("=", 1)[1].strip() or None
        elif t.startswith("--repq-structure="):
            repq_structure = t.split("=", 1)[1].strip() or None
        elif t.startswith("--epq-stages="):
            epq_stages = _parse_epq_stages(t.split("=", 1)[1].strip())
        elif t.startswith("--threads="):
            threads = _parse_positive_int(t.split("=", 1)[1].strip(), flag="--threads")
        elif t.startswith("--cpu-affinity="):
            cpu_affinity = t.split("=", 1)[1].strip() or None
            if cpu_affinity is not None:
                _parse_cpu_affinity(cpu_affinity)
        elif t.startswith("--affinity="):
            cpu_affinity = t.split("=", 1)[1].strip() or None
            if cpu_affinity is not None:
                _parse_cpu_affinity(cpu_affinity)
        elif t in ("--print-group-stats", "--print-groups"):
            print_group_stats = True
        else:
            targets.append(t)

    if not targets:
        print("no targets specified (after parsing flags)")
        sys.exit(1)

    return Args(
        dataset=dataset,
        B=B,
        targets=targets,
        mode=mode,
        print_group_stats=print_group_stats,
        epq_structure=epq_structure,
        repq_structure=repq_structure,
        epq_stages=epq_stages,
        threads=threads,
        cpu_affinity=cpu_affinity,
    )


def _load_dataset(name: str):
    s = name.lower()
    if "deep1m" in s:
        return DatasetDeep1B(10**6)
    if "deep10m" in s:
        return DatasetDeep1B(10**7)
    if "bigann1m" in s:
        return DatasetBigANN(nb_M=1)
    if "gist1m" in s:
        return DatasetGIST1M()
    if "glove" in s:
        return DatasetGlove()
    if "sift1m" in s:
        return DatasetSIFT1M()
    print("unknown dataset, defaulting to SIFT1M")
    return DatasetSIFT1M()


# ============================================================
# Main
# ============================================================

def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    effective_threads: Optional[int] = None
    if args.threads is not None:
        faiss.omp_set_num_threads(int(args.threads))
        omp_get_max_threads = getattr(faiss, "omp_get_max_threads", None)
        if callable(omp_get_max_threads):
            try:
                effective_threads = int(omp_get_max_threads())
            except Exception:
                effective_threads = int(args.threads)
        else:
            effective_threads = int(args.threads)

    effective_affinity: Optional[str] = None
    if args.cpu_affinity is not None:
        cpus = _parse_cpu_affinity(args.cpu_affinity)
        _set_process_affinity(cpus)
        effective_affinity = ",".join(str(cpu) for cpu in cpus)

    ds = _load_dataset(args.dataset)

    xq = np.ascontiguousarray(ds.get_queries(), dtype=np.float32)
    xb = np.ascontiguousarray(ds.get_database(), dtype=np.float32)
    gt = np.ascontiguousarray(ds.get_groundtruth())

    nb, d = xb.shape
    nq = xq.shape[0]
    _ = nb, nq

    # ---- budgets ----
    B = int(args.B)
    pq_nbits = int(args.pq_nbits)

    # PQ/OPQ family: nbits fixed to 8, M derived from B
    if B % pq_nbits != 0:
        raise ValueError(f"For PQ/OPQ baselines with nbits={pq_nbits}, require B % {pq_nbits} == 0, got B={B}")
    M_pq = B // pq_nbits
    if M_pq <= 0:
        raise ValueError("B too small: M_pq<=0")

    # BAPQ: paper setting q=4 => M = d/q
    q = int(args.bapq_q)
    if d % q != 0:
        raise ValueError(f"BAPQ paper setting requires d % q == 0. Got d={d}, q={q}.")
    M_bapq = d // q
    if M_bapq <= 0:
        raise ValueError("Invalid M_bapq")

    # Training set size used by ds.get_train():
    maxtrain = max(100 << pq_nbits, 10**5)
    xt = np.ascontiguousarray(ds.get_train(maxtrain=maxtrain), dtype=np.float32)

    print(
        f"eval on {args.dataset} targets={args.targets} "
        f"| nb={nb} nq={nq} "
        f"B={B} | PQ/OPQ: nbits={pq_nbits}, M={M_pq} | BAPQ: q={q}, M={M_bapq} | "
        f"maxtrain={maxtrain} | mode={args.mode} | print_group_stats={args.print_group_stats} | "
        f"epq_structure={args.epq_structure or '-'} | repq_structure={args.repq_structure or '-'} | "
        f"epq_stages={','.join(args.epq_stages) if args.epq_stages else 'none'} | "
        f"threads={effective_threads if effective_threads is not None else '-'} | "
        f"cpu_affinity={effective_affinity or '-'}"
    )

    todo = set(args.targets)
    mode: QueryMode = args.mode

    # ---------- helpers for recon sampling ----------
    recon_sample = min(200000, int(nb))
    recon_idx = _sample_indices(int(nb), recon_sample, seed=123)

    # ============================================================
    # EPQ targets
    # ============================================================

    if "epq" in todo:
        epq = build_epq(
            d,
            B=B,
            seed=123,
            verbose=True,
            structure=args.epq_structure,
            enable_uneven_opq=True,
            stages=args.epq_stages,
        )
        epq_index = EPQIndexADC(epq, name="epq", lut_chunk=4096, query_batch=16, seed=123)

        def _recon_epq():
            xb_s = xb[recon_idx]
            codes_s = epq.compute_codes(xb_s)
            xb_dec_s = epq_index._decode_from_codes(codes_s)
            return xb_s, xb_dec_s

        eval_index(
            epq_index,
            name=f"EPQ B={B}",
            xq=xq,
            xb=xb,
            gt=gt,
            xt=xt,
            k=1000,
            query_mode=mode,
            recon_fn=_recon_epq,
            pre_test_fn=(
                _make_group_stats_task(
                    quantizer_name="EPQ",
                    groups=lambda epq=epq: _epq_groups_for_proxy(epq),
                    bits=lambda epq=epq: epq.nbits_per_group,
                    proxy_input_fn=lambda xt=xt, epq=epq: _epq_space_for_proxy(epq, xt),
                    proxy_d=d,
                    seed=123,
                    entry_label="group",
                    space_label="epq-train-space",
                )
                if args.print_group_stats else None
            ),
        )

    if "repq" in todo:
        repq = build_epq(
            d,
            B=B,
            seed=123,
            verbose=True,
            structure=args.repq_structure or args.epq_structure,
            enable_uneven_opq=False,
            stages=args.epq_stages,
        )
        repq_index = EPQIndexADC(repq, name="repq", lut_chunk=4096, query_batch=16, seed=123)

        def _recon_repq():
            xb_s = xb[recon_idx]
            codes_s = repq.compute_codes(xb_s)
            xb_dec_s = repq_index._decode_from_codes(codes_s)
            return xb_s, xb_dec_s

        eval_index(
            repq_index,
            name=f"REPQ B={B}",
            xq=xq,
            xb=xb,
            gt=gt,
            xt=xt,
            k=1000,
            query_mode=mode,
            recon_fn=_recon_repq,
            pre_test_fn=(
                _make_group_stats_task(
                    quantizer_name="REPQ",
                    groups=lambda repq=repq: _epq_groups_for_proxy(repq),
                    bits=lambda repq=repq: repq.nbits_per_group,
                    proxy_input_fn=lambda xt=xt, repq=repq: _epq_space_for_proxy(repq, xt),
                    proxy_d=d,
                    seed=123,
                    entry_label="group",
                    space_label="repq-train-space",
                )
                if args.print_group_stats else None
            ),
        )

    # ============================================================
    # BAPQ (paper setting)
    # ============================================================

    if "bapq" in todo:
        from bapq_index import BAPQIndex, BAPQIndexConfig

        bapq_index = BAPQIndex(
            BAPQIndexConfig(
                d=int(d),
                B=int(B),
                q=int(q),
                bmax=12,
                seed=123,
                max_train_rows=maxtrain,
                pca_max_train_rows=maxtrain,
                km_niter=50,
                km_nredo=3,
                lut_chunk=4096,
                query_batch=32,
            )
        )

        class _BAPQAdapter(BenchIndex):
            def __init__(self, inner):
                self.inner = inner

            def train(self, xt: np.ndarray) -> None:
                self.inner.train(xt)

            def add(self, xb: np.ndarray) -> None:
                self.inner.add(xb)

            def search(self, xq: np.ndarray, k: int, *, mode: QueryMode = "adc") -> Tuple[np.ndarray, np.ndarray]:
                return self.inner.search(xq, k, mode=mode)

            def get_train_stats(self) -> Optional[TrainStats]:
                bapq = self.inner._require_index().bapq
                return TrainStats(
                    structure_time=float(getattr(bapq, "last_structure_time", 0.0)),
                    preparation_time=float(getattr(bapq, "last_preparation_time", 0.0)),
                    codebook_time=float(getattr(bapq, "last_codebook_time", 0.0)),
                    total_training_time=float(getattr(bapq, "last_train_total_time", 0.0)),
                )

        eval_index(
            _BAPQAdapter(bapq_index),
            name=f"BAPQ paper-setting q={q} => M={M_bapq}, total_bits B={B}",
            xq=xq,
            xb=xb,
            gt=gt,
            xt=xt,
            k=1000,
            query_mode=mode if mode == "adc" else "adc",
            recon_fn=None,
            pre_test_fn=(
                _make_group_stats_task(
                    quantizer_name="BAPQ",
                    groups=lambda bapq_index=bapq_index: bapq_index._require_index().bapq.groups or [],
                    bits=lambda bapq_index=bapq_index: bapq_index._require_index().bapq.nbits_per_group or [],
                    proxy_input_fn=lambda xt=xt, bapq_index=bapq_index: bapq_index._require_index().bapq.transform(xt),
                    proxy_d=int(d),
                    seed=123,
                    entry_label="group",
                    space_label="pca-space",
                )
                if args.print_group_stats else None
            ),
        )
        if mode == "sdc":
            print("\tNOTE: BAPQIndex is ADC-only in this benchmark; ran ADC instead.")

    # ============================================================
    # PQ / OPQ baselines with Python-side ADC
    # ============================================================

    if "pq" in todo:
        idx = ProductQuantizerADCIndex(
            d=d,
            M=M_pq,
            nbits=pq_nbits,
            name="PQ-ADC(py)",
            use_opq=False,
            lut_chunk=4096,
            query_batch=16,
        )
        pq_groups = _balanced_contiguous_groups(d, M_pq)
        pq_bits = [pq_nbits] * M_pq
        eval_index(
            idx,
            name=f"PQ(Python-ADC) M={M_pq} nbits={pq_nbits} (B={B})",
            xq=xq,
            xb=xb,
            gt=gt,
            xt=xt,
            k=1000,
            query_mode=mode,
            recon_fn=None,
            pre_test_fn=(
                _make_group_stats_task(
                    quantizer_name="PQ",
                    groups=pq_groups,
                    bits=pq_bits,
                    proxy_input_fn=lambda xt=xt: xt,
                    proxy_d=d,
                    seed=123,
                    entry_label="group",
                    space_label="original",
                )
                if args.print_group_stats else None
            ),
        )

    if "opq" in todo:
        idx = ProductQuantizerADCIndex(
            d=d,
            M=M_pq,
            nbits=pq_nbits,
            name="OPQ-ADC(py)",
            use_opq=True,
            lut_chunk=4096,
            query_batch=16,
        )
        d2 = int(idx.d2)
        opq_groups = _balanced_contiguous_groups(d2, M_pq)
        opq_bits = [pq_nbits] * M_pq
        eval_index(
            idx,
            name=f"OPQ+PQ(Python-ADC) M={M_pq} nbits={pq_nbits} (B={B}) d2={d2}",
            xq=xq,
            xb=xb,
            gt=gt,
            xt=xt,
            k=1000,
            query_mode=mode,
            recon_fn=None,
            pre_test_fn=(
                _make_group_stats_task(
                    quantizer_name="OPQ",
                    groups=opq_groups,
                    bits=opq_bits,
                    proxy_input_fn=lambda xt=xt, idx=idx: idx.project(xt),
                    proxy_d=d2,
                    seed=123,
                    entry_label="group",
                    space_label=f"opq-space(d2={d2})",
                )
                if args.print_group_stats else None
            ),
        )

    if "prq" in todo:
        print("===== PRQ requested; availability depends on FAISS build.")
        try:
            prq = faiss.ProductResidualQuantizer(d, 1, int(M_pq), pq_nbits)
            idx = faiss.IndexResidualQuantizer(d, prq)  # may not exist in some builds
            stage_bits = [pq_nbits] * M_pq
            stage_groups = [list(range(d)) for _ in range(M_pq)]
            eval_index(
                FaissIndexWrapper(idx, name="IndexResidualQuantizer(PRQ)"),
                name=f"PRQ(Index) stages={M_pq} nbits={pq_nbits} (B={B})",
                xq=xq,
                xb=xb,
                gt=gt,
                xt=xt,
                k=1000,
                query_mode=mode,
                pre_test_fn=(
                    _make_group_stats_task(
                        quantizer_name="PRQ",
                        groups=stage_groups,
                        bits=stage_bits,
                        proxy_input_fn=lambda xt=xt: xt,
                        proxy_d=d,
                        seed=123,
                        entry_label="stage",
                        space_label="original-full-dim-surrogate",
                    )
                    if args.print_group_stats else None
                ),
            )
        except Exception as e:
            print(f"PRQ(Index) not available: {e}")

    if "rq" in todo:
        print("===== RQ")
        try:
            rq = faiss.ResidualQuantizer(d, int(M_pq), pq_nbits)
            idx = faiss.IndexResidualQuantizer(d, rq)
            stage_bits = [pq_nbits] * M_pq
            stage_groups = [list(range(d)) for _ in range(M_pq)]
            try:
                idx.rq.max_beam_size = 30
            except Exception:
                pass
            eval_index(
                FaissIndexWrapper(idx, name="IndexResidualQuantizer(RQ)"),
                name=f"RQ(Index) M={M_pq} nbits={pq_nbits} (B={B})",
                xq=xq,
                xb=xb,
                gt=gt,
                xt=xt,
                k=1000,
                query_mode=mode,
                pre_test_fn=(
                    _make_group_stats_task(
                        quantizer_name="RQ",
                        groups=stage_groups,
                        bits=stage_bits,
                        proxy_input_fn=lambda xt=xt: xt,
                        proxy_d=d,
                        seed=123,
                        entry_label="stage",
                        space_label="original-full-dim-surrogate",
                    )
                    if args.print_group_stats else None
                ),
            )
        except Exception as e:
            print(f"RQ(Index) not available: {e}")

    if "lsq" in todo:
        print("===== LSQ")
        try:
            lsq = faiss.LocalSearchQuantizer(d, int(M_pq), pq_nbits)
            idx = faiss.IndexLSQ(d, lsq)  # may not exist in some builds
            stage_bits = [pq_nbits] * M_pq
            stage_groups = [list(range(d)) for _ in range(M_pq)]
            eval_index(
                FaissIndexWrapper(idx, name="IndexLSQ"),
                name=f"LSQ(Index) M={M_pq} nbits={pq_nbits} (B={B})",
                xq=xq,
                xb=xb,
                gt=gt,
                xt=xt,
                k=1000,
                query_mode=mode,
                pre_test_fn=(
                    _make_group_stats_task(
                        quantizer_name="LSQ",
                        groups=stage_groups,
                        bits=stage_bits,
                        proxy_input_fn=lambda xt=xt: xt,
                        proxy_d=d,
                        seed=123,
                        entry_label="stage",
                        space_label="original-full-dim-surrogate",
                    )
                    if args.print_group_stats else None
                ),
            )
        except Exception as e:
            print(f"LSQ(Index) not available: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
