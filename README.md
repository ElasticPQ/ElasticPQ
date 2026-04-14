# Elastic PQ

This repository has two main entry points:

- `bench_quantizer.py`: low-level ANN benchmarks on classic FAISS datasets.
- `mmeb_v2_bench/`: multimodal retrieval benchmarks on MMEB-V2-style tasks with pluggable embedders and quantizers.

## Repository Layout

```text
.
|-- bench_quantizer.py         # standalone quantizer benchmark entry
|-- epq_index.py               # trainable/searchable EPQ index with save/load
|-- elastic_pq.py              # ElasticPQ training pipeline
|-- opq_index.py               # OPQ wrapper used by the MMEB benchmark
|-- bapq_index.py              # BAPQ wrapper used by the MMEB benchmark
|-- mmeb_v2_bench/             # multimodal benchmark package
`-- util/                      # helper scripts for slicing vectors / making GT
```

## `bench_quantizer.py`

`bench_quantizer.py` is the direct benchmarking entry for vector quantizers on FAISS benchmark datasets. It loads a dataset from `faiss.contrib.datasets`, trains one or more indices, encodes the database, runs search, and prints timing plus quality metrics.

### What it benchmarks

The script can compare:

- `pq`: FAISS `IndexPQ`
- `opq`: FAISS `OPQMatrix + IndexPQ`
- `epq`: repository-native Elastic PQ with `unevenOPQ` enabled
- `repq`: EPQ with `unevenOPQ` disabled
- `bapq`: repository-native BAPQ
- `prq`, `rq`, `lsq`: FAISS residual/local-search variants when available in the local FAISS build

### Supported datasets

Dataset names are matched by substring and currently map to:

- `sift1M`
- `gist1M`
- `deep1M`
- `deep10M`
- `bigann1M`
- `glove`

Unknown names fall back to `SIFT1M`.

### Requirements

The `bench_quantizer.py` path and its EPQ/BAPQ/OPQ dependencies only require:

- `numpy`
- a Python FAISS build that includes `faiss.contrib.datasets`

For a basic CPU environment, install from [requirements-bench-quantizer.txt](D:\Projects\Python\PythonProject\epq_ano\requirements-bench-quantizer.txt):

```bash
pip install -r requirements-bench-quantizer.txt
```

Notes:

- The requirements file uses `faiss-cpu` by default.
- If you use a GPU FAISS build in your environment, replace `faiss-cpu` with the corresponding FAISS package provided by your package manager.
- Everything else used by `bench_quantizer.py` and the root quantizer modules comes from the Python standard library.

### Downloading classic ANN datasets

`bench_quantizer.py` follows the on-disk conventions used by `faiss.contrib.datasets`, so the easiest approach is to prepare a local `data/` directory with the exact filenames FAISS expects.

Reference links:

- FAISS benchmark docs: <https://github.com/facebookresearch/faiss/tree/main/benchs>
- FAISS dataset loader implementation: <https://github.com/facebookresearch/faiss/blob/main/contrib/datasets.py>
- TexMex dataset host: <http://corpus-texmex.irisa.fr/>

Expected layout:

```text
data/
|-- sift1M/
|   |-- sift_base.fvecs
|   |-- sift_learn.fvecs
|   |-- sift_query.fvecs
|   `-- sift_groundtruth.ivecs
|-- gist1M/
|   |-- gist_base.fvecs
|   |-- gist_learn.fvecs
|   |-- gist_query.fvecs
|   `-- gist_groundtruth.ivecs
`-- deep1b/
    |-- base.fvecs
    |-- learn.fvecs
    |-- deep1B_queries.fvecs
    `-- deep10M_groundtruth.ivecs
```

#### SIFT1M

Download `ANN_SIFT1M` from TexMex and place the four files below under `data/sift1M/`:

- `sift_base.fvecs`
- `sift_learn.fvecs`
- `sift_query.fvecs`
- `sift_groundtruth.ivecs`

#### GIST1M

Download `ANN_GIST1M` from TexMex and place the four files below under `data/gist1M/`:

- `gist_base.fvecs`
- `gist_learn.fvecs`
- `gist_query.fvecs`
- `gist_groundtruth.ivecs`

#### Deep1M / Deep10M

For Deep1B-derived subsets, follow FAISS's own `Getting Deep1B` note in `benchs/README.md`.

- Queries and ground truth: <https://yadi.sk/d/11eDCm7Dsn9GA>
- Learning vectors and database shards: use the upstream helper mentioned by FAISS:
  <https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py>

Prepare `data/deep1b/` so that FAISS can read:

- `deep1B_queries.fvecs`
- `deep10M_groundtruth.ivecs`
- `learn.fvecs`
- `base.fvecs`

Notes:

- `bench_quantizer.py deep10M ...` uses `DatasetDeep1B(10**7)`, which reads the first 10M vectors from `data/deep1b/base.fvecs`.
- `bench_quantizer.py deep1M ...` uses the same `deep1b/` source and reads the first 1M vectors.
- The script only asks FAISS for about `100000` training vectors, so `learn.fvecs` does not need to contain the full Deep1B training set if you are only doing local experiments.

### Slicing large `.fvecs` files with `util/`

If the original Deep1B files are too large for your machine, use `util/slice_fvecs.py` to materialize a smaller prefix or window from any `.fvecs` file or shard.

Examples:

```bash
python util/slice_fvecs.py --in data/deep1b/learn_00 --out data/deep1b/learn.fvecs --n 200000
python util/slice_fvecs.py --in data/deep1b/base.fvecs --out data/deep1b/base_10M.fvecs --n 10000000
python util/slice_fvecs.py --in data/deep1b/base.fvecs --out data/deep1b/base_window.fvecs --skip 10000000 --n 1000000
```

For `bench_quantizer.py deep1M/deep10M`, keep the final filenames aligned with FAISS expectations:

- database file: `data/deep1b/base.fvecs`
- training file: `data/deep1b/learn.fvecs`

So a common workflow is:

1. Download the original shards or merged file.
2. Use `util/slice_fvecs.py` to cut out the subset you want locally.
3. Rename or place the sliced output as `data/deep1b/base.fvecs` or `data/deep1b/learn.fvecs`.

### CLI shape

```bash
python bench_quantizer.py <dataset> <total_bits> <target...> [--mode=adc|sdc] [--print-group-stats] [--epq-structure=name-or-path] [--repq-structure=name-or-path] [--epq-stages=full|none|grow,crystallize,mbeam] [--threads=N] [--cpu-affinity=0,1,2-5]
```

Examples:

```bash
python bench_quantizer.py sift1M 128 pq opq epq repq bapq
python bench_quantizer.py gist1M 64 pq opq epq repq bapq
python bench_quantizer.py deep10M 64 pq opq epq
python bench_quantizer.py sift1M 128 epq --mode=sdc
python bench_quantizer.py sift1M 128 epq --epq-stages=grow
python bench_quantizer.py sift1M 128 epq --epq-stages=grow,crystallize
python bench_quantizer.py sift1M 128 epq --epq-stages=crystallize,mbeam
python bench_quantizer.py sift1M 128 epq --epq-stages=none
python bench_quantizer.py gist1M 128 epq --epq-structure=gist_128B_epq_structure.json
python bench_quantizer.py gist1M 128 repq --repq-structure=gist_128B_epq_structure.json
python bench_quantizer.py sift1M 128 pq opq epq --threads=8
python bench_quantizer.py sift1M 128 pq opq epq --threads=8 --cpu-affinity=0-7
```

### Important behavior

- The second positional argument is the total bit budget `B`, not `M x nbits`.
- For PQ/OPQ-style baselines, the script fixes `nbits=8` by default and derives `M = B / 8`.
- For BAPQ, the script follows the paper-style setup with `q=4` dimensions per subspace by default, so `M = d / q`.
- `--mode=sdc` is only implemented for the EPQ path. FAISS wrappers explicitly reject SDC, and BAPQ is forced back to ADC.
- `--print-group-stats` prints proxy statistics for learned groups and bit allocation before the main search evaluation.
- `--epq-structure` loads a precomputed EPQ structure, usually from `result/structure`, and skips the grow/crystallize/marginal-beam structure search step.
- `--repq-structure` does the same for `repq`. If omitted, `repq` also accepts the value passed through `--epq-structure`.
- `--epq-stages` controls the EPQ structure-search pipeline for ablations. The default is `full`, which means `grow -> crystallize -> marginal beam search`.
- `--threads=N` caps FAISS OpenMP threads and also exports common BLAS/OpenMP thread env vars before `numpy` / `faiss` import, so the overall benchmark process is much less likely to fan out across all CPUs.
- `--cpu-affinity=...` optionally pins the benchmark process to specific logical CPUs such as `0-7` or `0,2,4,6`. This is the stricter option when you want the process and FAISS to stay on a fixed CPU set.
- `--epq-stages=grow` keeps only the initial grow stage.
- `--epq-stages=grow,crystallize` disables the marginal beam stage.
- `--epq-stages=crystallize,mbeam` or `--epq-stages=none` disables grow. In that case the script replaces the missing grow initializer with a singleton partition where each dimension starts in its own group and the bit allocation is solved by `ctx.solve_bits(...)`.
- The `--epq-stages` setting is shared by both `epq` and `repq` runs when they build a fresh structure instead of loading one from `--epq-structure` / `--repq-structure`.

### What gets reported

For each target index, the script reports:

- `structure time`
- `preparation time`
- `codebook time`
- `training total`
- add/encode time
- search time
- `recall@1/10/100/1000`
- `overlap@1000`, a coverage-style metric used for BAPQ-style evaluation
- sampled reconstruction error when the backend exposes a reconstruction path

Benchmark note:

- `PQ` and `OPQ` search are evaluated through the repository's Python-side LUT/ADC path instead of `faiss.IndexPQ.search`, so their search-time comparison is closer to the EPQ/BAPQ implementation style and less affected by FAISS's specialized PQ scan optimizations.

### Existing classic ANN results

The repository already includes summarized `bench_quantizer.py` results under [result/record.md](D:\Projects\Python\PythonProject\epq_ano\result\record.md).

This file currently records comparison tables for:

- `SIFT1M` at `64b` and `128b`
- `GIST1M` at `64b` and `128b`
- `DEEP10M` at `64b` and `128b`
- `PQ`, `OPQ`, `BAPQ`, `EPQ(raw)`, and `EPQ`

The tables are intended as a compact experiment log of the current checked-in benchmark outputs, rather than a machine-readable export format.

### Precomputed EPQ structures

The repository also includes precomputed EPQ grouping/bit-allocation structures under [result/structure](D:\Projects\Python\PythonProject\epq_ano\result\structure).

Current files include:

- `sift_64B_epq_structure.json`
- `sift_128B_epq_structure.json`
- `gist_64B_epq_structure.json`
- `gist_128B_epq_structure.json`
- `deep_64B_epq_structure.json`
- `deep_128B_epq_structure.json`

These files are shortcut artifacts for rerunning evaluation without manually retraining the EPQ structure search pipeline. In other words, they let `bench_quantizer.py` skip the grow/crystallize/marginal-beam grouping stage and go straight to training/evaluation with a fixed structure.

Typical usage:

```bash
python bench_quantizer.py sift1M 64 epq --epq-structure=result/structure/sift_64B_epq_structure.json
python bench_quantizer.py deep10M 128 epq --epq-structure=result/structure/deep_128B_epq_structure.json
python bench_quantizer.py gist1M 128 repq --repq-structure=result/structure/gist_128B_epq_structure.json
```

## `mmeb_v2_bench/` package

`mmeb_v2_bench` is a small benchmark framework for multimodal retrieval experiments on MMEB-V2. It connects three layers:

- task loading from MMEB-V2 annotations and local media files
- embedding generation through Gemini or a deterministic mock embedder
- retrieval through exact search, PQ, EPQ, OPQ, or BAPQ backends

### Primary entry

The package entry point is:

```bash
python -m mmeb_v2_bench.cli [options]
```

### Downloading MMEB-V2 media

This repository does not automatically fetch MMEB-V2 media. Prepare the media locally first, then point `--dataset-root` at the extracted root directory.

Reference:

- Upstream MMEB-V2 README: <https://huggingface.co/datasets/TIGER-Lab/MMEB-V2>

Per the upstream MMEB-V2 README, the media archives are organized as:

- `image-tasks/mmeb_v1.tar.gz`
- `image-tasks/visdoc.tar.gz`
- `video-tasks/frames/video_cls.tar.gz`
- `video-tasks/frames/video_qa.tar.gz`
- `video-tasks/frames/video_ret.tar.gz`
- `video-tasks/frames/video_mret.tar.gz`

In this repository, the actual extracted layout used by `mmeb_v2_bench` is:

```text
/path/to/MMEB/
|-- image-tasks/
|   |-- A-OKVQA/
|   |-- ChartQA/
|   |-- CIRR/
|   |-- Country211/
|   |-- DocVQA/
|   |-- ...
|   |-- VisualNews_t2i/
|   |-- VizWiz/
|   |-- VOC2007/
|   |-- WebQA/
|   `-- Wiki-SS-NQ/
|-- video-tasks/
|   |-- data/
|   `-- frames/
|       |-- video_cls/
|       |-- video_qa/
|       |-- video_ret/
|       `-- video_mret/
`-- visdoc-tasks/
    |-- data/
    `-- images/
```

Notes:

- `--dataset-root` should point to the directory that directly contains `image-tasks/`, `video-tasks/`, and `visdoc-tasks/`.
- Under `image-tasks/`, the image tasks are expanded directly as task directories such as `ImageNet-1K/`, `MSCOCO_t2i/`, `OK-VQA/`, `VisualNews_t2i/`, not nested under a single `mmeb_v1/` directory.
- Under `video-tasks/`, this sandbox expects both `data/` and `frames/`.
- Annotations are loaded from Hugging Face or ModelScope and cached locally by this repo under `mmeb_v2_bench/cache/annotations` unless overridden.
- The visual-document tasks used by this sandbox (`ViDoRe_docvqa`, `MMLongBench-doc`, `VisRAG_MP-DocVQA`, etc.) are read from a separate local `visdoc-tasks/` tree in BEIR-like form:

```text
visdoc-tasks/
|-- data/<task_name>/queries/*.parquet
|-- data/<task_name>/qrels/*.parquet
|-- data/<task_name>/corpus/*.parquet
`-- images/<task_name>/
```

### Typical MMEB-V2 run

```bash
python -m mmeb_v2_bench.cli \
  --annotation-backend hf \
  --dataset-root /path/to/MMEB \
  --task MSCOCO_t2i \
  --task ImageNet-1K \
  --task Kinetics-700 \
  --task QVHighlight \
  --task ViDoRe_docvqa \
  --task MMLongBench-doc \
  --train-pool-group mm_core12 \
  --embedder gemini \
  --gemini-model gemini-embedding-2-preview \
  --output-dim 768 \
  --k-values 1 10 50 100 \
  --index-backend epq \
  --epq-total-bits 64 \
  --epq-max-bits 12 \
  --epq-verbose \
  --output-dir mmeb_v2_bench/runs/mm_core12_epq_64b
```

PQ baseline:

```bash
python -m mmeb_v2_bench.cli \
  --annotation-backend hf \
  --dataset-root /path/to/MMEB \
  --task MSCOCO_t2i \
  --task ImageNet-1K \
  --task Kinetics-700 \
  --task QVHighlight \
  --task ViDoRe_docvqa \
  --task MMLongBench-doc \
  --train-pool-group mm_core12 \
  --embedder gemini \
  --gemini-model gemini-embedding-2-preview \
  --output-dim 768 \
  --k-values 1 10 50 100 \
  --index-backend pq \
  --pq-subquantizers 8 \
  --pq-bits 8 \
  --output-dir mmeb_v2_bench/runs/mm_core12_pq_64b
```

BAPQ baseline:

```bash
python -m mmeb_v2_bench.cli \
  --annotation-backend hf \
  --dataset-root /path/to/MMEB \
  --task MSCOCO_t2i \
  --task ImageNet-1K \
  --task Kinetics-700 \
  --task QVHighlight \
  --task ViDoRe_docvqa \
  --task MMLongBench-doc \
  --train-pool-group mm_core12 \
  --embedder gemini \
  --gemini-model gemini-embedding-2-preview \
  --output-dim 768 \
  --k-values 1 10 50 100 \
  --index-backend bapq \
  --bapq-total-bits 96 \
  --bapq-bmax 12 \
  --bapq-subspace-dim 4 \
  --output-dir mmeb_v2_bench/runs/mm_core12_bapq_64b
```

### CLI responsibilities

`mmeb_v2_bench/cli.py` orchestrates the whole benchmark:

- chooses tasks from a YAML catalog or a local manifest
- builds the embedder
- builds the retrieval backend
- optionally builds a shared train pool for quantizer training
- runs each task through `benchmark.run_benchmark(...)`
- writes per-task summaries and a global `summary.json`

### Package structure

- `cli.py`: command-line entry and experiment orchestration
- `catalog.py` and `catalogs/*.yaml`: task catalogs and group aliases
- `dataset.py`: MMEB-V2 annotation loading, media resolution, manifest loading, and local annotation persistence
- `types.py`: shared dataclasses such as `MediaPart`, `Candidate`, `QueryExample`, and `TaskDataset`
- `embedder.py`: `GeminiEmbedding2Embedder` and `MockEmbedder`
- `embed_cache.py`: SQLite cache for document/query embeddings and unavailable items
- `benchmark.py`: end-to-end benchmark execution for one task
- `metrics.py`: retrieval metrics such as `hit@k`, `precision@k`, `recall@k`, and `mrr@k`
- `exact_index.py`: cosine exact-search baseline
- `pq_index.py`: pure NumPy product quantizer baseline
- `epq_adapter.py`, `opq_adapter.py`, `bapq_adapter.py`: adapters from the multimodal benchmark to the repository-root quantizer implementations
- `quantizer_cache.py`: stable cache paths for trained quantizers

### Caches and outputs

By default, the package uses:

- annotation cache: `mmeb_v2_bench/cache/annotations`
- embedding cache: `mmeb_v2_bench/cache/embeddings.sqlite`
- quantizer cache: `mmeb_v2_bench/cache/quantizers`
- run outputs: `mmeb_v2_bench/runs/latest`

Each task writes a `<task>.summary.json`, and the run directory also gets a merged `summary.json`. If `--save-rankings` is enabled, the package also writes `<task>.rankings.jsonl`.

### Existing MMEB-V2 results

In addition to the default runtime output directory above, this repository already contains checked-in MMEB-V2 benchmark outputs under [result/mmeb_v2_runs](D:\Projects\Python\PythonProject\epq_ano\result\mmeb_v2_runs).

Included runs currently are:

- `mm_core12_epq_64b`
- `mm_core12_pq_64b`
- `mm_core12_opq_64b`
- `mm_core12_bapq_64b`

Each run directory contains:

- one aggregated `summary.json`
- one per-task `<task>.summary.json` file for tasks such as `MSCOCO_t2i`, `ImageNet-1K`, `Kinetics-700`, `QVHighlight`, `ViDoRe_docvqa`, and `MMLongBench-doc`

These directories should be read as archived experiment outputs that were copied into `result/mmeb_v2_runs/` for reference. They are separate from the CLI's default live output location `mmeb_v2_bench/runs/latest`.

### Dependencies

The package-specific dependencies are listed in `mmeb_v2_bench/requirements.txt`:

- `numpy`
- `datasets`
- `google-genai`
- `Pillow`
- `PyYAML`
- `tqdm`
- `modelscope`

For `epq`, `opq`, and `bapq` backends, FAISS and the repository-root quantizer modules must also be importable.

## Notes

- The directory name is `mmeb_v2_bench`, not `mmeb_v2`; this README uses the on-disk package name.
- The root README now documents dataset download and placement explicitly; upstream links are included where the original files are hosted.
