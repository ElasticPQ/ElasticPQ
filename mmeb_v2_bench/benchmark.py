from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from .embedder import Embedder
from .index_base import VectorIndex
from .metrics import MetricsConfig, evaluate_rankings
from .types import QueryExample, TaskDataset


@dataclass
class BenchmarkResult:
    task_name: str
    n_raw_queries: int
    n_raw_candidates: int
    n_queries: int
    n_candidates: int
    n_train_vectors: int
    n_skipped_queries: int
    n_skipped_candidates: int
    query_keep_rate: float
    candidate_keep_rate: float
    metrics: dict[str, float]


def _select_by_indices(items: list, indices: list[int]) -> list:
    return [items[idx] for idx in indices]


def _filter_queries_by_available_candidates(
    queries: list[QueryExample],
    available_candidate_names: set[str],
) -> tuple[list[QueryExample], int]:
    filtered: list[QueryExample] = []
    skipped = 0
    for query in queries:
        candidate_names = tuple(name for name in query.candidate_names if name in available_candidate_names)
        labels = tuple(name for name in query.labels if name in available_candidate_names)
        if not candidate_names or not labels:
            skipped += 1
            continue
        filtered.append(
            QueryExample(
                query_id=query.query_id,
                parts=query.parts,
                labels=labels,
                candidate_names=candidate_names,
            )
        )
    return filtered, skipped


def run_benchmark(
    dataset: TaskDataset,
    *,
    embedder: Embedder,
    index: VectorIndex,
    top_k: int,
    output_dir: str | Path,
    metrics_cfg: MetricsConfig | None = None,
    save_rankings: bool = False,
    train_xb: object | None = None,
    quantizer_cache_dir: str | Path | None = None,
    quantizer_cache_prefix: object | None = None,
) -> BenchmarkResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_parts = [candidate.parts for candidate in dataset.corpus]
    corpus_result = embedder.embed(corpus_parts, is_query=False)
    corpus = _select_by_indices(dataset.corpus, corpus_result.kept_indices)
    xb = corpus_result.vectors
    n_skipped_candidates = len(corpus_result.skipped_indices)
    if xb.shape[0] == 0:
        raise RuntimeError(f"all candidates became unavailable for task={dataset.spec.name}")

    if train_xb is None:
        index.fit_database(
            xb,
            xb,
            quantizer_cache_dir=quantizer_cache_dir,
            quantizer_cache_prefix=quantizer_cache_prefix,
        )
        n_train_vectors = int(xb.shape[0])
    else:
        train_xb_np = train_xb
        index.fit_database(
            train_xb_np,
            xb,
            quantizer_cache_dir=quantizer_cache_dir,
            quantizer_cache_prefix=quantizer_cache_prefix,
        )
        n_train_vectors = int(train_xb_np.shape[0])

    available_candidate_names = {candidate.name for candidate in corpus}
    filtered_queries, skipped_by_candidate = _filter_queries_by_available_candidates(
        dataset.queries,
        available_candidate_names,
    )
    query_parts = [query.parts for query in filtered_queries]
    query_result = embedder.embed(query_parts, is_query=True)
    queries = _select_by_indices(filtered_queries, query_result.kept_indices)
    xq = query_result.vectors
    n_skipped_queries = skipped_by_candidate + len(query_result.skipped_indices)
    if xq.shape[0] == 0:
        raise RuntimeError(f"all queries became unavailable for task={dataset.spec.name}")

    scores, indices = index.search(xq, top_k=top_k)
    candidate_names = [candidate.name for candidate in corpus]
    predictions: list[list[str]] = []
    ranking_rows: list[dict[str, object]] = []
    for query, query_scores, query_indices in tqdm(
        list(zip(queries, scores, indices)),
        desc=f"rank:{dataset.spec.name}",
        leave=False,
    ):
        ranking = [candidate_names[int(idx)] for idx in query_indices]
        predictions.append(ranking)
        if save_rankings:
            ranking_rows.append(
                {
                    "query_id": query.query_id,
                    "labels": list(query.labels),
                    "prediction": ranking,
                    "scores": [float(score) for score in query_scores],
                }
            )

    metrics = evaluate_rankings(
        predictions=predictions,
        labels=[query.labels for query in queries],
        cfg=metrics_cfg,
    )
    result = BenchmarkResult(
        task_name=dataset.spec.name,
        n_raw_queries=len(dataset.queries),
        n_raw_candidates=len(dataset.corpus),
        n_queries=len(queries),
        n_candidates=len(corpus),
        n_train_vectors=n_train_vectors,
        n_skipped_queries=n_skipped_queries,
        n_skipped_candidates=n_skipped_candidates,
        query_keep_rate=float(len(queries) / len(dataset.queries) if dataset.queries else 0.0),
        candidate_keep_rate=float(len(corpus) / len(dataset.corpus) if dataset.corpus else 0.0),
        metrics=metrics,
    )

    summary_path = output_dir / f"{dataset.spec.name}.summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, ensure_ascii=False)

    if save_rankings:
        ranking_path = output_dir / f"{dataset.spec.name}.rankings.jsonl"
        with ranking_path.open("w", encoding="utf-8") as handle:
            for row in ranking_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return result
