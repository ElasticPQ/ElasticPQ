#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_gt1k.py

Compute exact GT@k (default k=1000) using FAISS IndexFlatL2 and save as .ivecs.

Supports multiple datasets with the common FAISS TexMex naming convention:

  ./data/sift1M/sift_base.fvecs
  ./data/sift1M/sift_query.fvecs
  ./data/gist1M/gist_base.fvecs
  ./data/gist1M/gist_query.fvecs

Outputs (by default):
  ./data/<dataset>/<prefix>_groundtruth_<k>.ivecs

Examples
--------
  # Compute SIFT1M groundtruth@1000 (GPU if available)
  python make_gt1k.py --dataset sift1M

  # Compute GIST1M groundtruth@1000 on CPU
  python make_gt1k.py --dataset gist1M --no-gpu

  # Custom root and k
  python make_gt1k.py --root ./data --dataset sift1M --k 1000

  # Override basedir directly
  python make_gt1k.py --basedir ./data/gist1M --prefix gist --k 1000

Notes
-----
- Groundtruth must be computed in ORIGINAL L2 space of (xb, xq).
- This script does not touch OPQ/PQ; it's purely exact NN.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple, Optional

import numpy as np
import faiss


# -----------------------------
# fvecs / ivecs IO
# -----------------------------

def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs -> float32 array (n, d)."""
    x = np.fromfile(path, dtype=np.int32)
    if x.size == 0:
        raise RuntimeError(f"Empty file: {path}")
    d = int(x[0])
    x = x.reshape(-1, d + 1)
    if not np.all(x[:, 0] == d):
        raise RuntimeError(f"Invalid fvecs header in {path}")
    return x[:, 1:].view(np.float32)


def write_ivecs(path: str, I: np.ndarray) -> None:
    """Write (n, k) int matrix to ivecs format: each row [k, i0, ..., i_{k-1}] int32."""
    I = np.ascontiguousarray(I, dtype=np.int32)
    n, k = I.shape
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = I
    out.tofile(path)


# -----------------------------
# dataset path logic
# -----------------------------

def infer_prefix_from_dataset(dataset: str) -> str:
    """Common mapping: sift1M -> sift, gist1M -> gist, deep1M -> deep (best effort)."""
    s = dataset.strip().lower()
    # common cases
    if s.startswith("sift"):
        return "sift"
    if s.startswith("gist"):
        return "gist"
    if s.startswith("deep"):
        return "deep"
    if s.startswith("glove"):
        return "glove"
    # fallback: take leading alpha chunk
    out = []
    for ch in s:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out) if out else s


def resolve_paths(*, basedir: str, prefix: str) -> Tuple[str, str]:
    """Return (base_fvecs, query_fvecs) paths."""
    base_path = os.path.join(basedir, f"{prefix}_base.fvecs")
    query_path = os.path.join(basedir, f"{prefix}_query.fvecs")
    if not os.path.exists(base_path):
        raise FileNotFoundError(base_path)
    if not os.path.exists(query_path):
        raise FileNotFoundError(query_path)
    return base_path, query_path


def load_xb_xq(*, basedir: str, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    base_path, query_path = resolve_paths(basedir=basedir, prefix=prefix)
    xb = np.ascontiguousarray(read_fvecs(base_path), dtype=np.float32)
    xq = np.ascontiguousarray(read_fvecs(query_path), dtype=np.float32)
    if xb.ndim != 2 or xq.ndim != 2:
        raise RuntimeError(f"Bad shapes: xb={xb.shape}, xq={xq.shape}")
    if xb.shape[1] != xq.shape[1]:
        raise RuntimeError(f"Dim mismatch: xb d={xb.shape[1]} vs xq d={xq.shape[1]}")
    return xb, xq


# -----------------------------
# GT computation
# -----------------------------

def build_flat_index(d: int, *, use_gpu: bool, gpu_device: int) -> faiss.Index:
    idx = faiss.IndexFlatL2(int(d))
    if use_gpu:
        res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(res, int(gpu_device), idx)
    return idx


def compute_gt_exact(
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int,
    use_gpu: bool,
    gpu_device: int,
    query_bs: int,
) -> np.ndarray:
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    nb, d = xb.shape
    nq = xq.shape[0]

    k = int(k)
    if k <= 0 or k > nb:
        raise ValueError(f"Invalid k={k} (nb={nb})")

    idx = build_flat_index(d, use_gpu=use_gpu, gpu_device=gpu_device)

    t0 = time.time()
    idx.add(xb)
    t1 = time.time()

    I_all = np.empty((nq, k), dtype=np.int64)

    qbs = int(max(1, query_bs))
    for i0 in range(0, nq, qbs):
        i1 = min(nq, i0 + qbs)
        _D, I = idx.search(xq[i0:i1], k)
        I_all[i0:i1] = I

        # light progress
        if i0 == 0 or (i0 // qbs) % 10 == 0 or i1 == nq:
            print(f"[search] {i1}/{nq} queries done")

    t2 = time.time()
    print(f"[timing] add: {t1 - t0:.3f}s  search: {t2 - t1:.3f}s  (use_gpu={use_gpu})")

    if I_all.min() < 0 or I_all.max() >= nb:
        raise RuntimeError("GT indices out of range, something went wrong")

    return I_all


# -----------------------------
# main
# -----------------------------

def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sift1M",
                    help="Dataset name (e.g., sift1M, gist1M). Used with --root to form basedir.")
    ap.add_argument("--root", type=str, default="./data",
                    help="Data root directory (default: ./data). basedir = <root>/<dataset>/")
    ap.add_argument("--basedir", type=str, default="",
                    help="Override dataset directory directly (contains <prefix>_base.fvecs).")
    ap.add_argument("--prefix", type=str, default="",
                    help="File prefix (default inferred from dataset: sift1M->sift, gist1M->gist).")
    ap.add_argument("--k", type=int, default=1000, help="GT neighbors to compute (default: 1000).")
    ap.add_argument("--out", type=str, default="",
                    help="Output ivecs path. Default: <basedir>/<prefix>_groundtruth_<k>.ivecs")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU IndexFlatL2.")
    ap.add_argument("--gpu", type=int, default=0, help="GPU device id if using GPU (default: 0).")
    ap.add_argument("--query-bs", type=int, default=256, help="Query batch size (default: 256).")
    ap.add_argument("--force", action="store_true", help="Overwrite output if exists.")
    args = ap.parse_args(argv[1:])

    dataset = str(args.dataset)
    prefix = str(args.prefix).strip() or infer_prefix_from_dataset(dataset)

    if args.basedir.strip():
        basedir = os.path.abspath(args.basedir.strip())
    else:
        basedir = os.path.abspath(os.path.join(str(args.root), dataset))

    use_gpu = not bool(args.no_gpu)

    print(f"[cfg] dataset={dataset} prefix={prefix} basedir={basedir} k={int(args.k)} use_gpu={use_gpu}")

    xb, xq = load_xb_xq(basedir=basedir, prefix=prefix)
    print(f"[data] xb={xb.shape} xq={xq.shape}")

    out_path = str(args.out).strip()
    if not out_path:
        out_path = os.path.join(basedir, f"{prefix}_groundtruth_{int(args.k)}.ivecs")
    out_path = os.path.abspath(out_path)

    if os.path.exists(out_path) and not args.force:
        print(f"[skip] {out_path} exists. Use --force to overwrite.")
        return 0

    I = compute_gt_exact(
        xb=xb,
        xq=xq,
        k=int(args.k),
        use_gpu=use_gpu,
        gpu_device=int(args.gpu),
        query_bs=int(args.query_bs),
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_ivecs(out_path, I.astype(np.int32, copy=False))
    print(f"[save] wrote {out_path} shape={I.shape} dtype=int32(ivecs)")

    # size sanity check
    n, k = I.shape
    expected = n * (k + 1) * 4
    actual = os.path.getsize(out_path)
    print(f"[check] bytes={actual} expected={expected}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
