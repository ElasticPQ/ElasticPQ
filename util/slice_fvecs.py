#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
slice_fvecs.py

Extract a prefix (or a window) of vectors from a FAISS .fvecs file (or shard)
and write to a new .fvecs.

fvecs record format:
  [int32 d][float32 x d]

Example:
  python slice_fvecs.py --in learn_00 --out deep_learn_200k.fvecs --n 200000
  python slice_fvecs.py --in learn_00 --out deep_learn_200k.fvecs --n 200000 --skip 100000
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from typing import Optional


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"


def slice_fvecs(
    in_path: str,
    out_path: str,
    n_take: int,
    skip: int = 0,
    *,
    progress_every: int = 20000,
    strict_dim: bool = True,
) -> None:
    if n_take <= 0:
        raise ValueError("--n must be > 0")
    if skip < 0:
        raise ValueError("--skip must be >= 0")
    if progress_every <= 0:
        progress_every = 20000

    file_size = os.path.getsize(in_path)
    print(f"[info] input: {in_path} ({human_bytes(file_size)})")
    print(f"[info] output: {out_path}")
    print(f"[info] skip={skip}, take={n_take}")

    # We'll stream-read record by record to avoid any huge memory use.
    d_ref: Optional[int] = None
    written = 0
    skipped = 0

    t0 = time.time()

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        while True:
            # Read dimension header
            head = fin.read(4)
            if not head:
                break  # EOF
            (d,) = struct.unpack("<i", head)

            if d <= 0 or d > 100000:
                raise RuntimeError(f"invalid dimension d={d} at record #{skipped+written}")

            vec_bytes = fin.read(4 * d)
            if len(vec_bytes) != 4 * d:
                raise RuntimeError(
                    f"truncated record: expected {4*d} bytes, got {len(vec_bytes)} "
                    f"at record #{skipped+written}"
                )

            if d_ref is None:
                d_ref = d
                est_out = n_take * (4 + 4 * d_ref)
                print(f"[info] detected d={d_ref}, estimated output size ~ {human_bytes(est_out)}")
            else:
                if strict_dim and d != d_ref:
                    raise RuntimeError(
                        f"dimension mismatch: expected d={d_ref}, got d={d} "
                        f"at record #{skipped+written}"
                    )

            # Skip phase
            if skipped < skip:
                skipped += 1
                continue

            # Write phase
            if written < n_take:
                fout.write(head)
                fout.write(vec_bytes)
                written += 1

                if written % progress_every == 0:
                    dt = time.time() - t0
                    rate = written / dt if dt > 0 else 0.0
                    print(f"[prog] written={written}/{n_take} ({rate:.1f} vec/s)")
            else:
                break  # done

    dt = time.time() - t0
    print(f"[done] wrote {written} vectors to {out_path} in {dt:.2f}s")

    if written != n_take:
        raise RuntimeError(f"EOF before enough vectors: wrote {written}, expected {n_take}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input fvecs shard/file (e.g. learn_00)")
    ap.add_argument("--out", dest="out_path", required=True, help="output fvecs file (e.g. deep_learn_200k.fvecs)")
    ap.add_argument("--n", dest="n_take", type=int, default=200000, help="number of vectors to take (default: 200000)")
    ap.add_argument("--skip", dest="skip", type=int, default=0, help="number of vectors to skip first (default: 0)")
    ap.add_argument("--progress-every", dest="progress_every", type=int, default=20000, help="progress print interval")
    ap.add_argument("--no-strict-dim", dest="strict_dim", action="store_false",
                    help="do not fail on dimension changes (rare; default is strict)")
    args = ap.parse_args()

    slice_fvecs(
        in_path=args.in_path,
        out_path=args.out_path,
        n_take=args.n_take,
        skip=args.skip,
        progress_every=args.progress_every,
        strict_dim=args.strict_dim,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        sys.exit(1)
