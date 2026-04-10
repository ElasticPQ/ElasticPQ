from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bapq_adapter import BAPQAdapterConfig, BAPQAdapterIndex
from .benchmark import run_benchmark
from .dataset import load_hf_mmeb_task, load_manifest_task
from .embed_cache import EmbeddingCache
from .embedder import (
    GeminiEmbedding2Embedder,
    GeminiEmbedderConfig,
    MockEmbedder,
    MockEmbedderConfig,
)
from .epq_adapter import EPQAdapterConfig, EPQAdapterIndex
from .exact_index import ExactCosineIndex
from .metrics import MetricsConfig
from .opq_adapter import OPQAdapterConfig, OPQAdapterIndex
from .pq_index import ProductQuantizerConfig, ProductQuantizerIndex
from .types import TaskDataset
from .utils import media_signature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMEB-V2 -> Gemini -> PQ/EPQ -> benchmark sandbox")
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="Local MMEB-V2 media root")
    parser.add_argument("--catalog", type=Path, default=None, help="Task catalog YAML")
    parser.add_argument("--task", action="append", default=[], help="Task name to run, repeatable")
    parser.add_argument("--task-group", default="image", help="Task group filter when --task is omitted")
    parser.add_argument("--train-pool-task", action="append", default=[], help="Auxiliary task used only for quantizer training, repeatable")
    parser.add_argument("--train-pool-group", action="append", default=[], help="Auxiliary task group used only for quantizer training, repeatable")
    parser.add_argument("--train-pool-num-samples", type=int, default=None, help="Optional subset size per train-pool task")
    parser.add_argument("--annotation-source", default="ziyjiang/MMEB_Test_Instruct", help="HF dataset id or local dataset path")
    parser.add_argument("--annotation-backend", choices=("auto", "hf", "modelscope", "local"), default="auto")
    parser.add_argument("--annotation-fallback-source", default="TIGER-Lab/MMEB-V2", help="ModelScope dataset id used when HF annotation loading fails in auto mode")
    parser.add_argument("--annotation-cache-dir", type=Path, default=Path("mmeb_v2_bench/cache/annotations"), help="Local persistent cache for HF parquet annotations")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Local manifest jsonl for smoke tests")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional subset size per task")
    parser.add_argument("--embedder", choices=("gemini", "mock"), default="mock")
    parser.add_argument("--gemini-model", default="gemini-embedding-2-preview")
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--cache-db", type=Path, default=Path("mmeb_v2_bench/cache/embeddings.sqlite"))
    parser.add_argument("--index-backend", choices=("pq", "exact", "epq", "opq", "bapq"), default="pq")
    parser.add_argument("--quantizer-cache-dir", type=Path, default=Path("mmeb_v2_bench/cache/quantizers"))
    parser.add_argument("--pq-subquantizers", type=int, default=32)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--pq-train-size", type=int, default=4096)
    parser.add_argument("--epq-total-bits", type=int, default=None)
    parser.add_argument("--epq-max-bits", type=int, default=12)
    parser.add_argument("--epq-verbose", action="store_true")
    parser.add_argument("--opq-verbose", action="store_true")
    parser.add_argument("--bapq-total-bits", type=int, default=None)
    parser.add_argument("--bapq-subspace-dim", type=int, default=4)
    parser.add_argument("--bapq-bmax", type=int, default=12)
    parser.add_argument("--bapq-max-train-rows", type=int, default=200000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output-dir", type=Path, default=Path("mmeb_v2_bench/runs/latest"))
    parser.add_argument("--save-rankings", action="store_true")
    return parser.parse_args()


def _build_embedder(args: argparse.Namespace, cache: EmbeddingCache | None):
    if args.embedder == "mock":
        return MockEmbedder(MockEmbedderConfig(output_dimensionality=int(args.output_dim)))
    return GeminiEmbedding2Embedder(
        GeminiEmbedderConfig(
            model=str(args.gemini_model),
            output_dimensionality=int(args.output_dim),
            batch_size=int(args.embed_batch_size),
            normalize=True,
        ),
        cache=cache,
    )


def _build_index(args: argparse.Namespace):
    if args.index_backend == "exact":
        return ExactCosineIndex()
    if args.index_backend == "epq":
        total_bits = (
            int(args.epq_total_bits)
            if args.epq_total_bits is not None
            else int(args.pq_subquantizers * args.pq_bits)
        )
        return EPQAdapterIndex(
            EPQAdapterConfig(
                total_bits=total_bits,
                max_bits=int(args.epq_max_bits),
                enable_uneven_opq=False,
                verbose=bool(args.epq_verbose),
            )
        )
    if args.index_backend == "opq":
        total_bits = int(args.pq_subquantizers * args.pq_bits)
        return OPQAdapterIndex(
            OPQAdapterConfig(
                total_bits=total_bits,
                nbits=int(args.pq_bits),
                n_subquantizers=int(args.pq_subquantizers),
                verbose=bool(args.opq_verbose),
            )
        )
    if args.index_backend == "bapq":
        total_bits = (
            int(args.bapq_total_bits)
            if args.bapq_total_bits is not None
            else int(args.pq_subquantizers * args.pq_bits)
        )
        return BAPQAdapterIndex(
            BAPQAdapterConfig(
                total_bits=total_bits,
                subspace_dim=int(args.bapq_subspace_dim),
                bmax=int(args.bapq_bmax),
                max_train_rows=int(args.bapq_max_train_rows),
            )
        )
    return ProductQuantizerIndex(
        ProductQuantizerConfig(
            n_subquantizers=int(args.pq_subquantizers),
            bits_per_subquantizer=int(args.pq_bits),
            train_size=int(args.pq_train_size),
        )
    )


def _collect_unique_corpus_parts(datasets: list[TaskDataset]) -> list[tuple]:
    unique_parts: list[tuple] = []
    seen: set[str] = set()
    for dataset in datasets:
        for candidate in dataset.corpus:
            signature = media_signature(candidate.parts)
            if signature in seen:
                continue
            seen.add(signature)
            unique_parts.append(candidate.parts)
    return unique_parts


def _count_raw_corpus_candidates(datasets: list[TaskDataset]) -> int:
    return sum(len(dataset.corpus) for dataset in datasets)


def _unique_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _task_in_group(spec, group: str) -> bool:
    if spec.group == group:
        return True
    return group in getattr(spec, "aliases", ())


def _resolve_top_k(args: argparse.Namespace) -> int:
    requested_top_k = int(args.top_k)
    max_metric_k = max(int(k) for k in args.k_values) if args.k_values else requested_top_k
    effective_top_k = max(requested_top_k, max_metric_k)
    if effective_top_k != requested_top_k:
        print(
            f"[metrics] promote top_k from {requested_top_k} to {effective_top_k} "
            f"because --k-values requests up to @{max_metric_k}"
        )
    return effective_top_k


def _build_quantizer_cache_prefix(
    args: argparse.Namespace,
    *,
    task_name: str | None,
    train_pool_tasks: list[str],
) -> dict[str, object] | None:
    if args.index_backend == "exact":
        return None
    embedder_payload: dict[str, object] = {
        "embedder": str(args.embedder),
        "output_dim": int(args.output_dim),
    }
    if args.embedder == "gemini":
        embedder_payload["model"] = str(args.gemini_model)
        embedder_payload["batch_size"] = int(args.embed_batch_size)

    if args.index_backend == "pq":
        index_payload = {
            "backend": "pq",
            "n_subquantizers": int(args.pq_subquantizers),
            "bits_per_subquantizer": int(args.pq_bits),
            "train_size": int(args.pq_train_size),
        }
    elif args.index_backend == "epq":
        index_payload = {
            "backend": "epq",
            "total_bits": int(args.epq_total_bits) if args.epq_total_bits is not None else int(args.pq_subquantizers * args.pq_bits),
            "max_bits": int(args.epq_max_bits),
            "enable_uneven_opq": False,
            "verbose": bool(args.epq_verbose),
        }
    elif args.index_backend == "opq":
        index_payload = {
            "backend": "opq",
            "total_bits": int(args.pq_subquantizers * args.pq_bits),
            "n_subquantizers": int(args.pq_subquantizers),
            "bits_per_subquantizer": int(args.pq_bits),
            "verbose": bool(args.opq_verbose),
        }
    elif args.index_backend == "bapq":
        index_payload = {
            "backend": "bapq",
            "total_bits": int(args.bapq_total_bits) if args.bapq_total_bits is not None else int(args.pq_subquantizers * args.pq_bits),
            "subspace_dim": int(args.bapq_subspace_dim),
            "bmax": int(args.bapq_bmax),
            "max_train_rows": int(args.bapq_max_train_rows),
        }
    else:
        return None

    return {
        "embedder": embedder_payload,
        "index": index_payload,
        "train_pool_tasks": list(train_pool_tasks),
        "task_name": task_name,
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_top_k = _resolve_top_k(args)

    cache = None if args.embedder == "mock" else EmbeddingCache(args.cache_db)
    try:
        embedder = _build_embedder(args, cache)
        metrics_cfg = MetricsConfig(k_values=tuple(int(k) for k in args.k_values))

        if args.manifest_path is not None:
            dataset = load_manifest_task(args.manifest_path, task_name=args.manifest_path.stem)
            result = run_benchmark(
                dataset,
                embedder=embedder,
                index=_build_index(args),
                top_k=effective_top_k,
                output_dir=args.output_dir,
                metrics_cfg=metrics_cfg,
                save_rankings=bool(args.save_rankings),
                quantizer_cache_dir=args.quantizer_cache_dir,
                quantizer_cache_prefix=_build_quantizer_cache_prefix(
                    args,
                    task_name=dataset.spec.name,
                    train_pool_tasks=[],
                ),
            )
            print(json.dumps({"task": result.task_name, **result.metrics}, ensure_ascii=False, indent=2))
            return

        from .catalog import load_catalog

        catalog = load_catalog(args.catalog)
        selected = args.task or [name for name, spec in catalog.items() if _task_in_group(spec, args.task_group)]
        train_pool_xb = None
        train_pool_groups = [str(group) for group in args.train_pool_group]
        train_pool_tasks = [str(task) for task in args.train_pool_task]
        for group in train_pool_groups:
            group_tasks = [name for name, spec in catalog.items() if _task_in_group(spec, group)]
            if not group_tasks:
                print(f"[train_pool] warning: no tasks found for group={group!r} in current catalog")
            train_pool_tasks.extend(group_tasks)
        train_pool_tasks = _unique_preserve_order(train_pool_tasks)

        if train_pool_tasks:
            if args.index_backend == "exact":
                print("[train_pool] ignored for exact backend")
            else:
                train_pool_datasets: list[TaskDataset] = []
                for task_name in train_pool_tasks:
                    if task_name not in catalog:
                        raise KeyError(f"unknown train-pool task: {task_name}")
                    train_pool_datasets.append(
                        load_hf_mmeb_task(
                            catalog[task_name],
                            annotation_source=str(args.annotation_source),
                            annotation_backend=str(args.annotation_backend),
                            annotation_fallback_source=str(args.annotation_fallback_source) if args.annotation_fallback_source else None,
                            annotation_cache_dir=args.annotation_cache_dir,
                            dataset_root=args.dataset_root,
                            num_samples=args.train_pool_num_samples,
                        )
                    )
                raw_candidates = _count_raw_corpus_candidates(train_pool_datasets)
                train_pool_parts = _collect_unique_corpus_parts(train_pool_datasets)
                dedup_candidates = len(train_pool_parts)
                print(
                    f"[train_pool] groups={train_pool_groups} tasks={train_pool_tasks} "
                    f"raw_candidates={raw_candidates} dedup_candidates={dedup_candidates}"
                )
                train_pool_result = embedder.embed(train_pool_parts, is_query=False)
                train_pool_xb = train_pool_result.vectors
                skipped_unavailable = len(train_pool_result.skipped_indices)
                if train_pool_xb.shape[0] == 0:
                    raise RuntimeError("all train-pool items became unavailable after embedding")
                print(
                    f"[train_pool] unavailable={skipped_unavailable} effective={int(train_pool_xb.shape[0])} "
                    f"dim={int(train_pool_xb.shape[1])}"
                )

        summaries: list[dict[str, object]] = []
        for task_name in selected:
            dataset = load_hf_mmeb_task(
                catalog[task_name],
                annotation_source=str(args.annotation_source),
                annotation_backend=str(args.annotation_backend),
                annotation_fallback_source=str(args.annotation_fallback_source) if args.annotation_fallback_source else None,
                annotation_cache_dir=args.annotation_cache_dir,
                dataset_root=args.dataset_root,
                num_samples=args.num_samples,
            )
            result = run_benchmark(
                dataset,
                embedder=embedder,
                index=_build_index(args),
                top_k=effective_top_k,
                output_dir=args.output_dir,
                metrics_cfg=metrics_cfg,
                save_rankings=bool(args.save_rankings),
                train_xb=train_pool_xb,
                quantizer_cache_dir=args.quantizer_cache_dir,
                quantizer_cache_prefix=_build_quantizer_cache_prefix(
                    args,
                    task_name=dataset.spec.name if train_pool_xb is None else None,
                    train_pool_tasks=train_pool_tasks,
                ),
            )
            row = {
                "task": result.task_name,
                "n_raw_queries": result.n_raw_queries,
                "n_raw_candidates": result.n_raw_candidates,
                "n_queries": result.n_queries,
                "n_candidates": result.n_candidates,
                "n_train_vectors": result.n_train_vectors,
                "n_skipped_queries": result.n_skipped_queries,
                "n_skipped_candidates": result.n_skipped_candidates,
                "query_keep_rate": result.query_keep_rate,
                "candidate_keep_rate": result.candidate_keep_rate,
                **result.metrics,
            }
            summaries.append(row)
            print(json.dumps(row, ensure_ascii=False))

        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, ensure_ascii=False)
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
