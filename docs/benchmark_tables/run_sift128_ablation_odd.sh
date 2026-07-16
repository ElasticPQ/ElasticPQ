#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-$ROOT/build-avx2/flat_benchmark}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-1,3,5,7,9,11,13,15,17,19,21,23}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/docs/benchmark_tables/tmp_sift128_ablation_odd_$STAMP}"

STANDARD_REUSE_CONFIG="$ROOT/configs/epq_train_standard_reuse.json"
NO_STRUCTURE_CONFIG="$ROOT/configs/epq_train_ablate_no_structure.json"
NO_UNEVEN_CONFIG="$ROOT/configs/epq_train_ablate_no_uneven.json"
FULL_MAIN_STRUCTURE="$DATA_ROOT/structures/sift_120B_cfg87a28918fa4b612a_epq_structure.json"

mkdir -p "$OUT_DIR"
echo "$$" > "$OUT_DIR/driver.pid"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

run_case() {
  local label="$1"
  local target="$2"
  local config="$3"
  shift 3

  local log="$OUT_DIR/${label}.log"
  local pidfile="$OUT_DIR/${label}.pid"

  echo "case=$label target=$target config=$config log=$log"
  (
    echo "case=$label"
    echo "target=$target"
    echo "config=$config"
    echo "cpuset=$CPUSET"
    echo "threads=$THREADS"
    echo "data_root=$DATA_ROOT"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    taskset -c "$CPUSET" /usr/bin/time -v "$BIN" \
      sift1M 128 "$target" \
      --config="$config" \
      --data-root="$DATA_ROOT" \
      --threads="$THREADS" \
      --topk=1000 \
      --maxtrain=0 \
      "$@"
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$pidfile"
  echo "pid=$pid"
  wait "$pid"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) case=$label status=0"
}

run_case full_arepq arepq_fixed "$STANDARD_REUSE_CONFIG"
if [[ ! -f "$FULL_MAIN_STRUCTURE" ]]; then
  echo "missing full main structure after full_arepq: $FULL_MAIN_STRUCTURE" >&2
  exit 1
fi
run_case no_structure arepq_fixed "$NO_STRUCTURE_CONFIG"
run_case no_uneven arepq_fixed "$NO_UNEVEN_CONFIG"
run_case no_residual_tail epq "$STANDARD_REUSE_CONFIG"

echo "logs=$OUT_DIR"
