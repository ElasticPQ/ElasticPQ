#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
BIN=${BIN:-$ROOT/build-avx2/flat_benchmark}
CONFIG=${CONFIG:-$ROOT/configs/epq_train_standard_reuse.json}
TRACE_DIR=${TRACE_DIR:-$ROOT/docs/benchmark_tables/tmp_jstar_sift128_arepq_trace}
OUT_DIR=${OUT_DIR:-$ROOT/docs/benchmark_tables/tmp_jstar_sift128_arepq_eval}
THREADS=${THREADS:-12}

ids=(0 2 3 5 6 8 9 11 12 14 15 17 18 19 21 23)

mkdir -p "$OUT_DIR"

for id in "${ids[@]}"; do
  printf -v sid "%04d" "$id"
  matches=("$TRACE_DIR"/structure_"$sid"_*.json)
  if [[ ${#matches[@]} -ne 1 || ! -f ${matches[0]} ]]; then
    echo "expected exactly one structure for id=$id under $TRACE_DIR" >&2
    exit 1
  fi

  log="$OUT_DIR/id_${sid}.log"
  if [[ -f "$log" ]] && grep -q "Exit status: 0" "$log"; then
    echo "skip id=$id; completed log exists: $log"
    continue
  fi

  echo "run id=$id structure=${matches[0]}"
  env \
    OMP_NUM_THREADS="$THREADS" \
    OMP_DYNAMIC=false \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MKL_DYNAMIC=false \
    BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    /usr/bin/time -v "$BIN" \
      sift1M 128 arepq_fixed \
      --config="$CONFIG" \
      --data-root="$DATA_ROOT" \
      --epq-structure="${matches[0]}" \
      --threads="$THREADS" \
      --maxtrain=0 \
      > "$log" 2>&1
done
