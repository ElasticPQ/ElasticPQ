#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BINARY="${BINARY:-${ROOT}/build-avx2/mmeb_vector_benchmark}"
BUNDLE="${BUNDLE:-${ROOT}/mmeb_v2_bench/bundles/mm_core12_gemini2_768_trainpool}"
CONFIG="${CONFIG:-${ROOT}/configs/epq_train_standard.json}"
RUN_NAME="${RUN_NAME:-mm_core12_cpp_128b_even_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/mmeb_v2_bench/cpp_runs/${RUN_NAME}}"
BITS="${BITS:-128}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-0,2,4,6,8,10,12,14,16,18,20,22}"
TARGETS="${TARGETS:-pq opq bapq rq lsq arepq}"
TOPK="${TOPK:-100}"
K_VALUES="${K_VALUES:-1,5,10,50,100}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${OUT_ROOT}/logs"

echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"
echo "bundle=${BUNDLE}"
echo "bits=${BITS}"
echo "max_train_rows=${MAX_TRAIN_ROWS:-0}"
echo "threads=${THREADS}"
echo "cpuset=${CPUSET}"
echo "targets=${TARGETS}"
echo "topk=${TOPK}"
echo "k_values=${K_VALUES}"
echo "started_at=$(date -Is)"

for target in ${TARGETS}; do
  target_out="${OUT_ROOT}/${target}_${BITS}b"
  log="${OUT_ROOT}/logs/${target}_${BITS}b.log"
  mkdir -p "${target_out}"

  echo "target=${target} started_at=$(date -Is) log=${log}"
  /usr/bin/time -v taskset -c "${CPUSET}" "${BINARY}" \
    --bundle="${BUNDLE}" \
    --target="${target}" \
    --bits="${BITS}" \
    --threads="${THREADS}" \
    --topk="${TOPK}" \
    --k-values="${K_VALUES}" \
    --max-train-rows="${MAX_TRAIN_ROWS:-0}" \
    --config="${CONFIG}" \
    --output-dir="${target_out}" \
    >"${log}" 2>&1
  echo "target=${target} finished_at=$(date -Is)"
done

echo "finished_at=$(date -Is)"
