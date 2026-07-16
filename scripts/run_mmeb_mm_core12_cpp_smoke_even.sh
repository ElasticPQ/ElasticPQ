#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BINARY="${BINARY:-${ROOT}/build-avx2/mmeb_vector_benchmark}"
BUNDLE="${BUNDLE:-${ROOT}/mmeb_v2_bench/bundles/mm_core12_gemini2_768_trainpool}"
CONFIG="${CONFIG:-${ROOT}/configs/epq_train_mmeb_smoke.json}"
RUN_NAME="${RUN_NAME:-mm_core12_cpp_smoke_128b_even_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/mmeb_v2_bench/cpp_runs/${RUN_NAME}}"
BITS="${BITS:-128}"
THREADS="${THREADS:-12}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-512}"
CPUSET="${CPUSET:-0,2,4,6,8,10,12,14,16,18,20,22}"
TARGETS="${TARGETS:-arepq}"
TASK="${TASK:-ViDoRe_docvqa}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export EPQ_AREPQ_ICM_ITERS="${EPQ_AREPQ_ICM_ITERS:-0}"
export EPQ_AREPQ_TAIL_ALT_ITERS="${EPQ_AREPQ_TAIL_ALT_ITERS:-0}"
export EPQ_AREPQ_TAIL_KMEANS_NITER="${EPQ_AREPQ_TAIL_KMEANS_NITER:-1}"
export EPQ_AREPQ_TAIL_KMEANS_NREDO="${EPQ_AREPQ_TAIL_KMEANS_NREDO:-1}"
export EPQ_OPQ_NITER="${EPQ_OPQ_NITER:-1}"
export EPQ_OPQ_NITER_PQ="${EPQ_OPQ_NITER_PQ:-1}"
export EPQ_OPQ_NITER_PQ0="${EPQ_OPQ_NITER_PQ0:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"
echo "bundle=${BUNDLE}"
echo "config=${CONFIG}"
echo "bits=${BITS}"
echo "max_train_rows=${MAX_TRAIN_ROWS}"
echo "threads=${THREADS}"
echo "cpuset=${CPUSET}"
echo "targets=${TARGETS}"
echo "task=${TASK}"
echo "started_at=$(date -Is)"

for target in ${TARGETS}; do
  target_out="${OUT_ROOT}/${target}_${BITS}b_trainonly"
  log="${OUT_ROOT}/logs/${target}_${BITS}b_trainonly.log"
  mkdir -p "${target_out}"

  echo "target=${target} started_at=$(date -Is) log=${log}"
  /usr/bin/time -v taskset -c "${CPUSET}" "${BINARY}" \
    --bundle="${BUNDLE}" \
    --task="${TASK}" \
    --target="${target}" \
    --bits="${BITS}" \
    --threads="${THREADS}" \
    --max-train-rows="${MAX_TRAIN_ROWS}" \
    --train-only \
    --config="${CONFIG}" \
    --output-dir="${target_out}" \
    >"${log}" 2>&1
  echo "target=${target} finished_at=$(date -Is)"
done

echo "finished_at=$(date -Is)"
