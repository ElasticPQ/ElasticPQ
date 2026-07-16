# EPQ

This repository contains the C++ implementation and evaluation programs for
EPQ. It provides learned product structures, UnevenOPQ training, product and
residual encoding, flat ADC/SDC search, shared coarse-IVF evaluation,
structure serialization, and the baselines used in the experiments.

Datasets, build directories, dependency installations, and experiment logs
are not included in this repository.

## Important: paper names versus code targets

The paper terminology differs from historical class and target names in the
implementation. Use the following mapping when running experiments and
reporting results.

| Paper name | Benchmark target | C++ class | Meaning |
| --- | --- | --- | --- |
| **EPQ** | `arepq` | `epq::IndexAREPQ` | The full method: learned product structure, UnevenOPQ, and a residual tail within the total bit budget. |
| **EPQ with a fixed structure** | `arepq_fixed` | `epq::IndexAREPQ` | The full method using a previously learned product structure. |
| **Product-only EPQ** | `epq` | `epq::IndexEPQ` | Learned product structure and UnevenOPQ, without the residual tail. |
| **EPQ w/o UnevenOPQ** | `arepq` with `index.use_uneven_transform=false` | `epq::IndexAREPQ` | The full residual-tail method with UnevenOPQ disabled. |

In particular, the code target named `epq` is **Product-only EPQ**, not the
full EPQ method reported in the paper. Paper EPQ results must use `arepq` or
`arepq_fixed`.

The command-line bit argument is the total budget. For `arepq`, this budget
contains both the product code and the residual tail. The paper's product-group
count, denoted by `M`, excludes residual-tail stages even though
`IndexAREPQ::component_count()` includes them.

The `repq` target is an additional product-only diagnostic that disables
UnevenOPQ. It should not be used as the paper's EPQ result.

## Repository layout

```text
configs/                  Training and ablation configurations
include/epq/              Public C++ headers
src/                      Index implementations and benchmark programs
python/mmeb_v2_bench/     MMEB data preparation and adapters
docs/benchmark_tables/    Result tables and paper-asset generators
third_party/              Faiss, Eigen, nlohmann-json, and VAQ sources
```

The main executables are:

- `flat_benchmark`: flat-scan evaluation of quantized database codes.
- `joint_benchmark`: shared coarse-IVF evaluation with quantized residual
  payloads.
- `epq_index_smoke_test`: small synthetic correctness test.
- `structure_builder_proxy_smoke_test`: structure-learning proxy test.
- `mmeb_vector_benchmark`: MMEB vector-bundle evaluation.

## System requirements

The implementation is intended for a 64-bit Linux system. The AVX2 release
build is the reference build for performance evaluation.

Required tools and libraries:

- CMake 3.24 or newer;
- a C++20 compiler (GCC or Clang);
- Ninja or Make;
- OpenMP;
- Faiss with a CMake package, preferably built with AVX2;
- Eigen3 and nlohmann-json;
- BLAS and LAPACK;
- GLPK, Armadillo, and SuiteSparse for the VAQ adapter;
- Python development headers for the default AVQ-enabled build.

On Ubuntu or Debian, install the non-Faiss dependencies with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libopenblas-dev liblapack-dev libeigen3-dev nlohmann-json3-dev \
  libglpk-dev libarmadillo-dev libsuitesparse-dev \
  python3-dev
```

The ScaNN Python package is needed only when running the optional `avq`
baseline. It is not needed for `arepq`, `arepq_fixed`, or `epq`.

### Third-party sources

The anonymous source package includes the required third-party source trees.
If they are absent in another checkout that uses Git submodules, initialize
them before building:

```bash
git submodule update --init --recursive
```

### Build an AVX2 Faiss installation

The EPQ CMake project imports an installed Faiss package; it does not build
Faiss automatically. The bundled Faiss source can be installed to
`local-avx2/` as follows:

```bash
cmake -S third_party/faiss -B build-deps/faiss-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/local-avx2" \
  -DFAISS_OPT_LEVEL=avx2 \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF \
  -DBUILD_TESTING=OFF
cmake --build build-deps/faiss-avx2 -j 8
cmake --install build-deps/faiss-avx2
```

This example uses the system BLAS/LAPACK installation. An MKL-based Faiss
build can be used instead by changing the corresponding Faiss options.

## Build EPQ

From the repository root, configure a release build with:

```bash
cmake -S . -B build-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_PREFIX="$PWD/local-avx2" \
  -DDEPS_PREFIX=/usr \
  -DEPQ_FAISS_TARGET=faiss_avx2
```

`DEPS_PREFIX=/usr` matches the standard Ubuntu locations installed by
`libeigen3-dev` and `nlohmann-json3-dev`. For custom installations, set
`DEPS_PREFIX` to a prefix containing:

```text
share/eigen3/cmake/Eigen3Config.cmake
share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake
```

`FAISS_PREFIX` must contain `share/faiss/faiss-config.cmake`. If
`EPQ_FAISS_TARGET` is omitted, CMake selects `faiss_avx2` when available and
otherwise falls back to `faiss`.

When configuring from an activated Python environment, ensure that its C++
runtime is compatible with the system compiler. To force the Ubuntu system
Python, add `-DPython3_EXECUTABLE=/usr/bin/python3`.

Build the primary programs and smoke tests:

```bash
cmake --build build-avx2 -j 8 --target \
  flat_benchmark joint_benchmark \
  epq_index_smoke_test structure_builder_proxy_smoke_test
```

If GLPK, Armadillo, or SuiteSparse are installed in a non-system prefix, pass
`-DVAQ_DEPS_PREFIX=/path/to/prefix` during configuration. At runtime, add that
prefix's library directory to `LD_LIBRARY_PATH` if it is not handled by the
system linker.

## Verify the build

Run the synthetic smoke test first:

```bash
./build-avx2/epq_index_smoke_test
```

A successful run prints JSON with `"status": "ok"`.

For a small data-backed training check, use a deliberately limited training
set:

```bash
./build-avx2/flat_benchmark \
  sift1M 128 arepq \
  --config=configs/epq_train_standard.json \
  --data-root=/path/to/datasets \
  --threads=2 \
  --maxtrain=1000 \
  --train-only
```

The `--maxtrain=1000` command is a smoke test only. Do not use a truncated
training set for reported experiments.

## Download and prepare datasets

Follow the Faiss benchmark data instructions for SIFT1M, GIST1M, and the DEEP
datasets:

<https://github.com/facebookresearch/faiss/tree/main/benchs>

After downloading or converting the files, pass their common parent directory
with `--data-root`. The flat and joint benchmarks expect this layout:

```text
/path/to/datasets/
  sift1M/
    sift_learn.fvecs
    sift_base.fvecs
    sift_query.fvecs
    sift_groundtruth.ivecs
  gist1M/
    gist_learn.fvecs
    gist_base.fvecs
    gist_query.fvecs
    gist_groundtruth.ivecs
  deep1b/
    learn.fvecs
    base.fvecs
    deep1B_queries.fvecs
    deep10M_groundtruth.ivecs
  structures/
    ... learned EPQ structure JSON files ...
```

The `deep10M` dataset name uses the files under `deep1b/` shown above.
`joint_benchmark` also supports the original Deep1B binary files through
`--deep1b-root`:

```text
learn.350M.fbin
base.1B.fbin
query.public.10K.fbin
groundtruth.public.10K.ibin
```

For paper experiments, use every row in the dataset-provided training split.
Set `--maxtrain=0` for `flat_benchmark` and `--train-limit=0` for
`joint_benchmark`. Positive limits are intended only for diagnostics.

## Run paper EPQ with flat search

The standard full-method command is:

```bash
OMP_NUM_THREADS=12 \
OMP_DYNAMIC=false \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
./build-avx2/flat_benchmark \
  sift1M 128 arepq \
  --config=configs/epq_train_standard.json \
  --data-root=/path/to/datasets \
  --threads=12 \
  --maxtrain=0
```

This runs the method named **EPQ in the paper**. With the standard
configuration, one 8-bit residual-tail stage is included in the 128-bit total
budget, leaving 120 bits for the learned product structure.

### Run Product-only EPQ

Change the target to `epq`:

```bash
./build-avx2/flat_benchmark \
  sift1M 128 epq \
  --config=configs/epq_train_standard.json \
  --data-root=/path/to/datasets \
  --threads=12 \
  --maxtrain=0
```

This is **Product-only EPQ** in the paper. It must not be labeled as the full
EPQ method.

### Run EPQ without UnevenOPQ

Keep the `arepq` target so that the residual tail remains enabled, and use the
provided ablation configuration:

```bash
./build-avx2/flat_benchmark \
  sift1M 128 arepq \
  --config=configs/epq_train_ablate_no_uneven.json \
  --data-root=/path/to/datasets \
  --threads=12 \
  --maxtrain=0
```

Using target `epq` for this ablation would also remove the residual tail and
would therefore test a different method.

### Reuse a learned structure

`configs/epq_train_standard_reuse.json` enables automatic structure reuse. A
matching structure is resolved under `<data-root>/structures/` using the
dataset, product-code budget, and configuration fingerprint.

For explicit and auditable reuse, provide the structure directly:

```bash
./build-avx2/flat_benchmark \
  sift1M 128 arepq_fixed \
  --config=configs/epq_train_standard_reuse.json \
  --data-root=/path/to/datasets \
  --epq-structure=/path/to/sift_120bit_structure.json \
  --threads=12 \
  --maxtrain=0
```

For a 128-bit EPQ run with the standard 8-bit tail, the fixed JSON describes
the 120-bit product structure.

## Run paper EPQ with shared coarse IVF

The joint benchmark syntax is:

```text
joint_benchmark <dataset> <bits> <nlist> <nprobe> <target> [options]
```

For example:

```bash
OMP_NUM_THREADS=12 \
OMP_DYNAMIC=false \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
./build-avx2/joint_benchmark \
  sift1M 128 4096 16 arepq \
  --config=configs/epq_train_standard.json \
  --data-root=/path/to/datasets \
  --threads=12 \
  --train-limit=0 \
  --base-limit=0 \
  --query-limit=0 \
  --topk=1000 \
  --metric-topk=1000
```

The coarse quantizer and probed lists are shared across payload methods. Use
`--refine` and `--refine-k-factor=F` only for the explicit exact-vector
reranking protocol.

## Important command-line options

### `flat_benchmark`

- `--config=PATH`: training and structure-builder configuration.
- `--data-root=PATH`: root of the dataset layout described above.
- `--epq-structure=PATH`: load a fixed product structure.
- `--mode=adc|sdc`: query mode; ADC is the default paper path.
- `--threads=N`: requested OpenMP thread count.
- `--topk=N`: number of returned neighbors.
- `--recon-sample=N`: reconstruction-error sample size.
- `--maxtrain=N`: cap training rows; `0` means the full training split.
- `--train-only`: train without adding the database or searching.
- `--skip-search`: train and add, but skip query evaluation.

Multiple targets may be supplied to one flat command, but separate processes
are preferable for controlled timing comparisons.

### `joint_benchmark`

- `--train-limit=N`, `--base-limit=N`, and `--query-limit=N`: row caps; `0`
  means all available rows for SIFT1M, GIST1M, and DEEP10M.
- `--base-batch-size=N`: streaming database-add batch size.
- `--topk=N`: returned-neighbor count.
- `--metric-topk=N`: ground-truth depth used for recall metrics.
- `--json-out=PATH`: write a machine-readable summary.
- `--refine`: enable exact-vector reranking of selected candidates.

## Training configurations

- `configs/epq_train_standard.json`: learn a fresh structure using the standard
  refined builder.
- `configs/epq_train_standard_reuse.json`: reuse a matching learned structure
  when available.
- `configs/epq_train_ablate_no_structure.json`: balanced-structure ablation.
- `configs/epq_train_ablate_no_uneven.json`: full EPQ without UnevenOPQ.
- `configs/epq_train_mmeb_smoke.json`: reduced MMEB smoke configuration.

The standard configuration sets the Faiss BLAS threshold to 20, uses the
refined `grow -> crystallize -> chain_tail` structure builder, and configures
one 8-bit residual-tail stage.

## Reproducible performance runs

For performance comparisons:

1. Use the AVX2 release build.
2. Use the full dataset-provided training split.
3. Pin or record the CPU set.
4. Keep OpenMP and BLAS thread limits identical across methods.
5. Run compared methods in separate processes.
6. Record the configuration path, structure JSON, compiler, Faiss target, and
   effective row counts printed by the benchmark.

A typical thread environment is:

```bash
export OMP_NUM_THREADS=12
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

The benchmark output includes training, add, and search time; QPS; recall at
multiple cutoffs; overlap; reconstruction error; serialized size; memory
statistics; dataset row counts; build metadata; and the resolved method
configuration.

## Optional AVQ runtime

The `avq` target embeds the official ScaNN Python bindings. Install a ScaNN
version compatible with the Python interpreter detected by CMake, then expose
its site-packages directory if it is not already visible:

```bash
export EPQ_AVQ_PYTHONPATH=/path/to/python/site-packages
```

This environment variable is not used by EPQ, Product-only EPQ, or the other
C++ baselines.
