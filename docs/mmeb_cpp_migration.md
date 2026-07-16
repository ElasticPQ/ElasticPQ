# MMEB Vector Benchmark Migration

MMEB data loading, media materialization, and Gemini embedding calls remain in
Python. The benchmark execution path is C++: Python exports embedded vectors and
labels into a stable vector bundle, then `mmeb_vector_benchmark` trains/adds/searches
the selected index backend.

## Download Media

Run from the repository root:

```bash
PYTHONPATH=./python python -m mmeb_v2_bench.download_media \
  --output-dir=./mmeb_v2_bench/data
```

Use `--no-extract` to download only, or `--no-download` to extract archives that
already exist locally.

## Export a vector bundle

Run from the repository root:

```bash
PYTHONPATH=./python python -m mmeb_v2_bench.cli \
  --dataset-root=./mmeb_v2_bench/data \
  --task-group=mm_core12 \
  --embedder=gemini \
  --gemini-model=gemini-embedding-2-preview \
  --output-dim=768 \
  --cache-db=./mmeb_v2_bench/cache/embeddings.sqlite \
  --export-bundle=./mmeb_v2_bench/bundles/mm_core12_gemini2_768
```

For a smoke test, use `--embedder=mock --num-samples=16`.

Optional shared training pool:

```bash
PYTHONPATH=./python python -m mmeb_v2_bench.cli \
  --dataset-root=./mmeb_v2_bench/data \
  --task-group=mm_core12 \
  --train-pool-group=mm_core12 \
  --embedder=gemini \
  --output-dim=768 \
  --export-bundle=./mmeb_v2_bench/bundles/mm_core12_gemini2_768_trainpool
```

## Run C++ benchmark

Build:

```bash
cmake --build ./build-avx2 -j 4 --target mmeb_vector_benchmark
```

Run one backend:

```bash
./build-avx2/mmeb_vector_benchmark \
  --bundle=./mmeb_v2_bench/bundles/mm_core12_gemini2_768_trainpool \
  --target=epq \
  --bits=128 \
  --config=./configs/epq_train_standard.json \
  --threads=12 \
  --topk=10 \
  --k-values=1,5,10 \
  --output-dir=./mmeb_v2_bench/cpp_runs/mm_core12_epq_128b
```

Supported C++ targets: `exact`, `pq`, `opq`, `rq`, `lsq`, `epq`, `repq`,
`bapq`, `arepq`.

`mmeb_vector_benchmark` trains once per process and reuses the trained index
configuration across all selected tasks. For smoke tests, add
`--max-train-rows=256` or another small cap.

Paper-style serial multimodal run on the even CPU set:

```bash
screen -dmS mmeb_mm_core12_cpp_even bash -lc \
  './scripts/run_mmeb_mm_core12_cpp_serial_even.sh'
```

Use `--task=ImageNet-1K` to filter a bundle to one task.

## Bundle Layout

`metadata.json` lists tasks and optional shared `train.f32`. Each task directory
contains:

- `manifest.json`: shape metadata, query IDs, candidate names, and label indices.
- `corpus.f32`: row-major `float32` candidate embeddings.
- `queries.f32`: row-major `float32` query embeddings.

If no shared train pool is exported, the task corpus is used for quantizer
training, matching the old Python benchmark behavior.
