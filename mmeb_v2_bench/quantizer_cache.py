from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def train_matrix_digest(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x, dtype=np.float32)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(v) for v in x.shape)).encode("ascii"))
    hasher.update(x.tobytes())
    return hasher.hexdigest()


def _slug(text: object, *, max_len: int = 32) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    if not value:
        return "unnamed"
    return value[:max_len].rstrip("_")


def _human_cache_stem(backend: str, prefix_payload: object) -> str:
    if not isinstance(prefix_payload, dict):
        return _slug(backend, max_len=16)

    embedder = prefix_payload.get("embedder")
    embed_dim = None
    if isinstance(embedder, dict):
        embed_dim = embedder.get("output_dim")

    index = prefix_payload.get("index")
    task_name = prefix_payload.get("task_name")
    train_pool_tasks = prefix_payload.get("train_pool_tasks")

    parts = [_slug(backend, max_len=16)]
    if embed_dim is not None:
        parts.append(f"{int(embed_dim)}d")

    if isinstance(index, dict):
        if backend == "epq":
            total_bits = index.get("total_bits")
            max_bits = index.get("max_bits")
            if total_bits is not None:
                parts.append(f"{int(total_bits)}b")
            if max_bits is not None:
                parts.append(f"{int(max_bits)}bmax")
        elif backend == "bapq":
            total_bits = index.get("total_bits")
            bmax = index.get("bmax")
            subspace_dim = index.get("subspace_dim")
            if total_bits is not None:
                parts.append(f"{int(total_bits)}b")
            if bmax is not None:
                parts.append(f"{int(bmax)}bmax")
            if subspace_dim is not None:
                parts.append(f"q{int(subspace_dim)}")
        elif backend in {"pq", "opq"}:
            total_bits = index.get("total_bits")
            n_subquantizers = index.get("n_subquantizers")
            bits_per_subquantizer = index.get("bits_per_subquantizer")
            if total_bits is not None:
                parts.append(f"{int(total_bits)}b")
            if n_subquantizers is not None:
                parts.append(f"m{int(n_subquantizers)}")
            if bits_per_subquantizer is not None:
                parts.append(f"b{int(bits_per_subquantizer)}")

    if isinstance(train_pool_tasks, list) and train_pool_tasks:
        parts.append(f"pool{len(train_pool_tasks)}")
    elif task_name:
        parts.append(_slug(task_name, max_len=24))

    return "_".join(parts)


def quantizer_cache_path(
    cache_dir: str | Path,
    *,
    backend: str,
    prefix_payload: object,
    train_xb: np.ndarray,
) -> Path:
    root = Path(cache_dir)
    prefix_json = stable_json_dumps(prefix_payload)
    train_digest = train_matrix_digest(train_xb)
    prefix_digest = hashlib.sha256(prefix_json.encode("utf-8")).hexdigest()[:12]
    readable_stem = _human_cache_stem(backend, prefix_payload)
    return root / backend / f"{readable_stem}_{prefix_digest}_{train_digest[:20]}"
