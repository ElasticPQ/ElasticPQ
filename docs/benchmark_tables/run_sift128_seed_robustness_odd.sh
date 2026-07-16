#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-$ROOT/build-avx2/flat_benchmark}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/epq_train_standard_reuse.json}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-1,3,5,7,9,11,13,15,17,19,21,23}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/docs/benchmark_tables/tmp_sift128_seed_robustness_$STAMP}"
SEEDS=("$@")

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(456 789)
fi

mkdir -p "$OUT_DIR/configs"
echo "$$" > "$OUT_DIR/driver.pid"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

make_seed_config() {
  local seed="$1"
  local out="$2"
  python3 - "$BASE_CONFIG" "$out" "$seed" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = json.loads(base_path.read_text())
cfg.setdefault("builder", {})["auto_reuse_structure"] = False
cfg.setdefault("builder", {}).setdefault("refined", {})["seed"] = seed
cfg.setdefault("index", {})["ivf_query_weighted_sampling_seed"] = seed
cfg.setdefault("index", {}).setdefault("transform", {})["transform_seed"] = seed

out_path.write_text(json.dumps(cfg, indent=2) + "\n")
PY
}

run_seed() {
  local seed="$1"
  local config="$OUT_DIR/configs/epq_train_seed_${seed}.json"
  local log="$OUT_DIR/seed_${seed}.log"
  local pidfile="$OUT_DIR/seed_${seed}.pid"

  make_seed_config "$seed" "$config"
  echo "seed=$seed config=$config log=$log"
  (
    echo "case=seed_robustness"
    echo "seed=$seed"
    echo "target=arepq_fixed"
    echo "config=$config"
    echo "cpuset=$CPUSET"
    echo "threads=$THREADS"
    echo "data_root=$DATA_ROOT"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    taskset -c "$CPUSET" /usr/bin/time -v "$BIN" \
      sift1M 128 arepq_fixed \
      --config="$config" \
      --data-root="$DATA_ROOT" \
      --threads="$THREADS" \
      --topk=1000 \
      --maxtrain=0
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$pidfile"
  echo "pid=$pid"
  wait "$pid"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$seed status=0"
}

for seed in "${SEEDS[@]}"; do
  run_seed "$seed"
done

echo "logs=$OUT_DIR"
