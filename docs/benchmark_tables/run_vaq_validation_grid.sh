#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$ROOT/build-avx2/flat_benchmark"
CONFIG="${CONFIG:-$ROOT/configs/epq_train_standard.json}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-}"
BITS="${BITS:?set BITS to 64 or 128}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
OUT_DIR="$ROOT/docs/benchmark_tables/logs/$RUN_TAG"
VALIDATION_BASE="${VALIDATION_BASE:-10000}"
VALIDATION_QUERIES="${VALIDATION_QUERIES:-1000}"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/local-vaq/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUT_DIR"

dataset_bounds() {
  case "$1" in
    sift1M) printf '%s %s\n' 2 13 ;;
    gist1M) printf '%s %s\n' 7 13 ;;
    deep10M) printf '%s %s\n' 5 12 ;;
    *) echo "unsupported dataset: $1" >&2; exit 1 ;;
  esac
}

run_candidate() {
  local dataset="$1"
  local tag="$2"
  local subspaces="$3"
  local min_bits="$4"
  local max_bits="$5"
  local stem="validate_${dataset}_${BITS}b_${tag}_m${subspaces}_min${min_bits}_max${max_bits}"
  local log_path="$OUT_DIR/${stem}.log"
  local -a cmd=(
    "$BIN" "$dataset" "$BITS" vaq
    --config="$CONFIG"
    --data-root="$DATA_ROOT"
    --threads="$THREADS"
    --maxtrain=0
    --train-only
    --vaq-subspaces="$subspaces"
    --vaq-min-bits="$min_bits"
    --vaq-max-bits="$max_bits"
    --vaq-validation-base="$VALIDATION_BASE"
    --vaq-validation-queries="$VALIDATION_QUERIES"
  )
  if [[ -n "$CPUSET" ]]; then
    cmd=(taskset -c "$CPUSET" "${cmd[@]}")
  fi
  if [[ -f "$log_path" ]] && rg -q 'Exit status: 0' "$log_path"; then
    echo "=== $stem (skip complete)"
    return
  fi
  echo "=== $stem"
  {
    echo "script.run_tag=$RUN_TAG"
    echo "script.stem=$stem"
    echo "script.cpuset=${CPUSET:-inherit}"
    echo "script.validation_base=$VALIDATION_BASE"
    echo "script.validation_queries=$VALIDATION_QUERIES"
    echo "script.start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    /usr/bin/time -v "${cmd[@]}"
  } >"$log_path" 2>&1
}

for dataset in sift1M gist1M deep10M; do
  uniform_m=$((BITS / 8))
  dense_m=$((BITS / 4))
  read -r official_min official_max < <(dataset_bounds "$dataset")

  run_candidate "$dataset" legacy "$dense_m" 1 8
  run_candidate "$dataset" official "$uniform_m" "$official_min" "$official_max"
  run_candidate "$dataset" balanced "$uniform_m" 7 9
  run_candidate "$dataset" moderate "$uniform_m" 4 12
  run_candidate "$dataset" aggressive "$uniform_m" 1 13
  run_candidate "$dataset" dense_wide "$dense_m" 1 13
  run_candidate "$dataset" dense_moderate "$dense_m" 3 9
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
