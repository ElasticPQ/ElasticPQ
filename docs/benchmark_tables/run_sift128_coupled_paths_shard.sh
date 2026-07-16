#!/usr/bin/env bash
set -euo pipefail

MATRIX_DIR="${1:?usage: run_sift128_coupled_paths_shard.sh MATRIX_DIR SHARD_INDEX SHARD_COUNT}"
SHARD_INDEX="${2:?missing shard index}"
SHARD_COUNT="${3:?missing shard count}"

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$ROOT/build-avx2/flat_benchmark"
DATA_ROOT="$ROOT/data"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-0,2,4,6,8,10,12,14,16,18,20,22}"
LOG_DIR="$MATRIX_DIR/logs"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/local-vaq/usr/lib/x86_64-linux-gnu:$ROOT/local-vaq/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case_index=0
while IFS=$'\t' read -r case_id protocol path_name transform_niter structure_path config_path; do
  if [[ "$case_id" == "case_id" ]]; then
    continue
  fi
  if (( case_index % SHARD_COUNT != SHARD_INDEX )); then
    case_index=$((case_index + 1))
    continue
  fi
  case_index=$((case_index + 1))

  log="$LOG_DIR/${case_id}.log"
  if [[ -f "$log" ]] && rg -q '^\s*Exit status: 0$' "$log"; then
    echo "skip_complete case=$case_id log=$log"
    continue
  fi

  echo "start case=$case_id protocol=$protocol path=$path_name niter=$transform_niter log=$log"
  (
    echo "experiment=sift128_coupled_paths"
    echo "case_id=$case_id"
    echo "protocol=$protocol"
    echo "path_name=$path_name"
    echo "transform_niter=$transform_niter"
    echo "structure_path=$structure_path"
    echo "config_path=$config_path"
    echo "cpuset=$CPUSET"
    echo "threads=$THREADS"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    taskset -c "$CPUSET" /usr/bin/time -v "$BIN" \
      sift1M 128 epq \
      --config="$config_path" \
      --epq-structure="$structure_path" \
      --data-root="$DATA_ROOT" \
      --threads="$THREADS" \
      --topk=1000 \
      --recon-sample=200000 \
      --maxtrain=0
  ) >"$log" 2>&1
  echo "finish case=$case_id log=$log"
done < "$MATRIX_DIR/cases.tsv"

echo "shard_complete index=$SHARD_INDEX count=$SHARD_COUNT logs=$LOG_DIR"
