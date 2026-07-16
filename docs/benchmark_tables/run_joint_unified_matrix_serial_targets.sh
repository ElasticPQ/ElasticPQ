#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$ROOT/build-avx2/joint_benchmark"
CONFIG="${CONFIG:-$ROOT/configs/epq_train_standard.json}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
STRUCTURE_DIR="${STRUCTURE_DIR:-$ROOT/data/structures}"
EPQ_FRESH_STRUCTURE_ON_FIRST_CASE="${EPQ_FRESH_STRUCTURE_ON_FIRST_CASE:-1}"
AREPQ_TAIL_BITS="${AREPQ_TAIL_BITS:-8}"
AREPQ_TAIL_STAGES="${AREPQ_TAIL_STAGES:-1}"
CPUSET_TAG="${CPUSET:-inherit}"
CPUSET_TAG="${CPUSET_TAG//,/x}"
CPUSET_TAG="${CPUSET_TAG//-/_}"
RUN_TAG="${RUN_TAG:-joint_topk1000_fullsplits_serial_t${THREADS}_cpuset${CPUSET_TAG}_${DATE_TAG}}"
OUT_DIR="$ROOT/docs/benchmark_tables/logs/$RUN_TAG"
REFINE_FLAG=""
REFINE_K_FACTOR=""
MATRIX_MODE="full"
DATASETS=(sift1M gist1M deep10M)
TARGETS=(pq opq dpopq epq bapq rq lsq)
BITS_LIST=(64 128)

usage() {
  echo "usage: $0 [--refine] [--refine-k-factor=F] [--matrix=full|representative] [--datasets=a,b] [--targets=a,b] [--bits=a,b]" >&2
}

parse_csv_arg() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$raw"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refine)
      REFINE_FLAG="--refine"
      shift
      ;;
    --refine-k-factor=*)
      REFINE_K_FACTOR="$1"
      shift
      ;;
    --matrix=*)
      MATRIX_MODE="${1#--matrix=}"
      shift
      ;;
    --datasets=*)
      parse_csv_arg "${1#--datasets=}" DATASETS
      shift
      ;;
    --targets=*)
      parse_csv_arg "${1#--targets=}" TARGETS
      shift
      ;;
    --bits=*)
      parse_csv_arg "${1#--bits=}" BITS_LIST
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ "$MATRIX_MODE" != "full" && "$MATRIX_MODE" != "representative" ]]; then
  usage
  exit 1
fi

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
mkdir -p "$STRUCTURE_DIR"

declare -A EPQ_STRUCTURE_PREPARED=()

echo "run_tag=$RUN_TAG"
echo "out_dir=$OUT_DIR"
echo "cpuset=${CPUSET:-inherit}"
echo "threads=$THREADS"
echo "data_root=$DATA_ROOT"
echo "matrix_mode=$MATRIX_MODE"
echo "branch=$(git -C "$ROOT" branch --show-current)"
echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_case() {
  local dataset="$1"
  local bits="$2"
  local nlist="$3"
  local nprobe="$4"
  local target="$5"
  local stem="joint_${dataset}_${bits}b_${target}_nlist${nlist}_nprobe${nprobe}"
  if [[ -n "$REFINE_FLAG" ]]; then
    stem+="_refine"
  fi
  if [[ -n "$REFINE_K_FACTOR" ]]; then
    stem+="_k$(printf '%s' "${REFINE_K_FACTOR#--refine-k-factor=}" | tr '.' 'p')"
  fi
  local log_path="$OUT_DIR/${stem}.log"
  local json_path="$OUT_DIR/${stem}.json"
  local -a cmd=(
    "$BIN"
    "$dataset" "$bits" "$nlist" "$nprobe"
    "$target"
    --config="$CONFIG"
    --data-root="$DATA_ROOT"
    --threads="$THREADS"
    --topk=1000
    --metric-topk=1000
    --json-out="$json_path"
  )
  if [[ "$target" == "vaq" ]]; then
    cmd+=(--vaq-subspaces="$((bits / 8))")
    case "$dataset" in
      sift1M)
        cmd+=(--vaq-min-bits=2 --vaq-max-bits=13)
        ;;
      gist1M)
        cmd+=(--vaq-min-bits=7 --vaq-max-bits=13)
        ;;
      deep10M)
        cmd+=(--vaq-min-bits=5 --vaq-max-bits=12)
        ;;
    esac
  fi
  if [[ "$target" == "epq" || "$target" == "arepq" || "$target" == "arepq_fixed" ]]; then
    local structure_bits="$bits"
    if [[ "$target" == "arepq" || "$target" == "arepq_fixed" ]]; then
      structure_bits=$((bits - AREPQ_TAIL_BITS * AREPQ_TAIL_STAGES))
      if [[ "$structure_bits" -le 0 ]]; then
        echo "invalid AREPQ main bits for bits=$bits tail=${AREPQ_TAIL_BITS}x${AREPQ_TAIL_STAGES}" >&2
        exit 1
      fi
    fi
    local structure_path="$STRUCTURE_DIR/${dataset}_${structure_bits}B_nlist${nlist}_joint_ivf_epq_structure.json"
    local structure_key="${dataset}_${structure_bits}_${nlist}"
    if [[ "$EPQ_FRESH_STRUCTURE_ON_FIRST_CASE" == "1" && -z "${EPQ_STRUCTURE_PREPARED[$structure_key]:-}" ]]; then
      if [[ -f "$structure_path" ]]; then
        local backup_path="$OUT_DIR/$(basename "$structure_path" .json)_pre_rerun_backup_$(date -u +%Y%m%dT%H%M%SZ).json"
        mv "$structure_path" "$backup_path"
        echo "moved stale epq structure: $structure_path -> $backup_path"
      fi
      EPQ_STRUCTURE_PREPARED["$structure_key"]=1
    fi
    cmd+=(--epq-structure="$structure_path")
  fi
  if [[ -n "$REFINE_FLAG" ]]; then
    cmd+=("$REFINE_FLAG")
  fi
  if [[ -n "$REFINE_K_FACTOR" ]]; then
    cmd+=("$REFINE_K_FACTOR")
  fi
  if [[ -n "$CPUSET" ]]; then
    cmd=(taskset -c "$CPUSET" "${cmd[@]}")
  fi

  if [[ -f "$json_path" && -f "$log_path" ]]; then
    echo "=== ${stem} (skip existing)"
    return
  fi

  echo "=== ${stem}"
  {
    echo "script.run_tag=$RUN_TAG"
    echo "script.stem=$stem"
    echo "script.cpuset=${CPUSET:-inherit}"
    echo "script.threads=$THREADS"
    echo "script.config=$CONFIG"
    echo "script.data_root=$DATA_ROOT"
    echo "script.matrix_mode=$MATRIX_MODE"
    echo "script.start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    /usr/bin/time -v "${cmd[@]}"
  } >"$log_path" 2>&1
}

run_sift_matrix() {
  for target in "${TARGETS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      for nprobe in 4 8 16 32; do
        run_case sift1M "$bits" 4096 "$nprobe" "$target"
      done
    done
  done
}

run_gist_matrix() {
  for target in "${TARGETS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      for nprobe in 4 8 16 32; do
        run_case gist1M "$bits" 4096 "$nprobe" "$target"
      done
    done
  done
}

run_deep_matrix() {
  for target in "${TARGETS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      for nprobe in 16 32 64 128; do
        run_case deep10M "$bits" 16384 "$nprobe" "$target"
      done
    done
  done
}

run_representative_case() {
  local dataset="$1"
  for target in "${TARGETS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      case "$dataset" in
        sift1M)
          run_case "$dataset" "$bits" 4096 4 "$target"
          ;;
        gist1M)
          run_case "$dataset" "$bits" 4096 4 "$target"
          ;;
        deep10M)
          run_case "$dataset" "$bits" 16384 16 "$target"
          ;;
        *)
          echo "unknown dataset: $dataset" >&2
          exit 1
          ;;
      esac
    done
  done
}

for dataset in "${DATASETS[@]}"; do
  if [[ "$MATRIX_MODE" == "representative" ]]; then
    run_representative_case "$dataset"
    continue
  fi

  case "$dataset" in
    sift1M)
      run_sift_matrix
      ;;
    gist1M)
      run_gist_matrix
      ;;
    deep10M)
      run_deep_matrix
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
