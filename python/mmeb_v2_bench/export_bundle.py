from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .benchmark import _filter_queries_by_available_candidates, _select_by_indices
from .embedder import Embedder
from .types import QueryExample, TaskDataset


@dataclass
class ExportedTask:
    task_name: str
    task_dir: str
    dim: int
    n_raw_queries: int
    n_raw_candidates: int
    n_queries: int
    n_candidates: int
    n_train_vectors: int
    n_skipped_queries: int
    n_skipped_candidates: int
    query_keep_rate: float
    candidate_keep_rate: float


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return text or "task"


def _write_f32_matrix(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    x.tofile(path)


def _label_indices(
    queries: Iterable[QueryExample],
    candidate_to_index: dict[str, int],
) -> list[list[int]]:
    rows: list[list[int]] = []
    for query in queries:
        labels = [candidate_to_index[name] for name in query.labels if name in candidate_to_index]
        if not labels:
            raise RuntimeError(f"query has no exported labels: {query.query_id}")
        rows.append(labels)
    return rows


def export_task_bundle(
    dataset: TaskDataset,
    *,
    embedder: Embedder,
    output_dir: str | Path,
    train_xb: np.ndarray | None = None,
) -> ExportedTask:
    output_dir = Path(output_dir)
    task_dir = output_dir / _slug(dataset.spec.name)
    task_dir.mkdir(parents=True, exist_ok=True)

    corpus_parts = [candidate.parts for candidate in dataset.corpus]
    corpus_result = embedder.embed(corpus_parts, is_query=False)
    corpus = _select_by_indices(dataset.corpus, corpus_result.kept_indices)
    xb = np.ascontiguousarray(corpus_result.vectors, dtype=np.float32)
    n_skipped_candidates = len(corpus_result.skipped_indices)
    if xb.shape[0] == 0:
        raise RuntimeError(f"all candidates became unavailable for task={dataset.spec.name}")

    available_candidate_names = {candidate.name for candidate in corpus}
    filtered_queries, skipped_by_candidate = _filter_queries_by_available_candidates(
        dataset.queries,
        available_candidate_names,
    )
    query_parts = [query.parts for query in filtered_queries]
    query_result = embedder.embed(query_parts, is_query=True)
    queries = _select_by_indices(filtered_queries, query_result.kept_indices)
    xq = np.ascontiguousarray(query_result.vectors, dtype=np.float32)
    n_skipped_queries = skipped_by_candidate + len(query_result.skipped_indices)
    if xq.shape[0] == 0:
        raise RuntimeError(f"all queries became unavailable for task={dataset.spec.name}")
    if xb.shape[1] != xq.shape[1]:
        raise RuntimeError(
            f"embedding dim mismatch for task={dataset.spec.name}: corpus={xb.shape[1]} query={xq.shape[1]}"
        )

    candidate_names = [candidate.name for candidate in corpus]
    candidate_to_index = {name: idx for idx, name in enumerate(candidate_names)}
    labels = _label_indices(queries, candidate_to_index)
    query_ids = [query.query_id for query in queries]

    _write_f32_matrix(task_dir / "corpus.f32", xb)
    _write_f32_matrix(task_dir / "queries.f32", xq)

    manifest = {
        "format": "mmeb-vector-bundle-task-v1",
        "task_name": dataset.spec.name,
        "dim": int(xb.shape[1]),
        "corpus": {"path": "corpus.f32", "rows": int(xb.shape[0]), "dim": int(xb.shape[1])},
        "queries": {"path": "queries.f32", "rows": int(xq.shape[0]), "dim": int(xq.shape[1])},
        "candidate_names": candidate_names,
        "query_ids": query_ids,
        "labels": labels,
        "n_raw_queries": len(dataset.queries),
        "n_raw_candidates": len(dataset.corpus),
        "n_skipped_queries": n_skipped_queries,
        "n_skipped_candidates": n_skipped_candidates,
    }
    if train_xb is not None:
        manifest["train"] = {
            "path": "../train.f32",
            "rows": int(train_xb.shape[0]),
            "dim": int(train_xb.shape[1]),
        }
    else:
        manifest["train"] = {
            "path": "corpus.f32",
            "rows": int(xb.shape[0]),
            "dim": int(xb.shape[1]),
        }
    with (task_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return ExportedTask(
        task_name=dataset.spec.name,
        task_dir=task_dir.name,
        dim=int(xb.shape[1]),
        n_raw_queries=len(dataset.queries),
        n_raw_candidates=len(dataset.corpus),
        n_queries=int(xq.shape[0]),
        n_candidates=int(xb.shape[0]),
        n_train_vectors=int(train_xb.shape[0]) if train_xb is not None else int(xb.shape[0]),
        n_skipped_queries=n_skipped_queries,
        n_skipped_candidates=n_skipped_candidates,
        query_keep_rate=float(len(queries) / len(dataset.queries) if dataset.queries else 0.0),
        candidate_keep_rate=float(len(corpus) / len(dataset.corpus) if dataset.corpus else 0.0),
    )


def write_bundle_metadata(
    output_dir: str | Path,
    *,
    tasks: list[ExportedTask],
    train_xb: np.ndarray | None,
    payload: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_xb is not None:
        _write_f32_matrix(output_dir / "train.f32", train_xb)
    metadata = {
        "format": "mmeb-vector-bundle-v1",
        **payload,
        "train": (
            {"path": "train.f32", "rows": int(train_xb.shape[0]), "dim": int(train_xb.shape[1])}
            if train_xb is not None
            else None
        ),
        "tasks": [asdict(task) for task in tasks],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
