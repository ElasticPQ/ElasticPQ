#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="$ROOT/build-avx2"
BIN="$BUILD_DIR/flat_benchmark"
CONFIG="${CONFIG:-$ROOT/configs/epq_train_standard.json}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
THREADS="${THREADS:-12}"
CPUSET="${CPUSET:-}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
BUILD_JOBS="${BUILD_JOBS:-4}"
SKIP_BUILD=0
DATASETS=(sift1M gist1M deep10M)
BITS_LIST=(64 128)
TARGETS=(pq opq epq bapq rabitq rq lsq)
ALLOWED_DATASETS=(sift1M gist1M deep10M)
ALLOWED_TARGETS=(pq opq dpopq epq repq arepq bapq rabitq rq lsq vaq)
ALLOWED_BITS=(64 128)

usage() {
  echo "usage: $0 [--targets=a,b] [--datasets=a,b] [--bits=64,128] [--config=PATH] [--data-root=PATH] [--threads=N] [--cpuset=LIST|inherit] [--build-jobs=N] [--skip-build]" >&2
}

parse_csv_arg() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$raw"
}

sanitize_tag() {
  printf '%s' "$1" | tr -cs '[:alnum:]' '_'
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

validate_list() {
  local label="$1"
  local -n values_ref="$2"
  shift 2
  local value
  if [[ "${#values_ref[@]}" -eq 0 ]]; then
    echo "empty ${label} list" >&2
    exit 1
  fi
  for value in "${values_ref[@]}"; do
    if ! contains "$value" "$@"; then
      echo "unsupported ${label}: $value" >&2
      exit 1
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets=*)
      parse_csv_arg "${1#--targets=}" TARGETS
      shift
      ;;
    --datasets=*)
      parse_csv_arg "${1#--datasets=}" DATASETS
      shift
      ;;
    --bits=*)
      parse_csv_arg "${1#--bits=}" BITS_LIST
      shift
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      shift
      ;;
    --data-root=*)
      DATA_ROOT="${1#--data-root=}"
      shift
      ;;
    --threads=*)
      THREADS="${1#--threads=}"
      shift
      ;;
    --cpuset=*)
      CPUSET="${1#--cpuset=}"
      if [[ "$CPUSET" == "inherit" ]]; then
        CPUSET=""
      fi
      shift
      ;;
    --build-jobs=*)
      BUILD_JOBS="${1#--build-jobs=}"
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
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

validate_list "dataset" DATASETS "${ALLOWED_DATASETS[@]}"
validate_list "target" TARGETS "${ALLOWED_TARGETS[@]}"
validate_list "bits" BITS_LIST "${ALLOWED_BITS[@]}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "data root not found: $DATA_ROOT" >&2
  exit 1
fi

CPUSET_TAG="$(sanitize_tag "${CPUSET:-inherit}")"
CONFIG_TAG="$(sanitize_tag "$(basename "${CONFIG%.json}")")"
RUN_TAG="${RUN_TAG:-flat_topk1000_fullsplits_serial_t${THREADS}_cpuset${CPUSET_TAG}_cfg${CONFIG_TAG}_${DATE_TAG}}"
OUT_DIR="$ROOT/docs/benchmark_tables/logs/$RUN_TAG"

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

build_binary() {
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    echo "build=skip"
    return
  fi

  local -a cmd=(
    cmake
    --build "$BUILD_DIR"
    -j "$BUILD_JOBS"
    --target flat_benchmark
  )

  echo "build_cmd=${cmd[*]}"
  "${cmd[@]}"
}

echo "run_tag=$RUN_TAG"
echo "out_dir=$OUT_DIR"
echo "cpuset=${CPUSET:-inherit}"
echo "threads=$THREADS"
echo "config=$CONFIG"
echo "data_root=$DATA_ROOT"
echo "datasets=$(IFS=,; echo "${DATASETS[*]}")"
echo "bits=$(IFS=,; echo "${BITS_LIST[*]}")"
echo "targets=$(IFS=,; echo "${TARGETS[*]}")"
echo "branch=$(git -C "$ROOT" branch --show-current)"
echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

build_binary

if [[ ! -x "$BIN" ]]; then
  echo "flat benchmark binary not found after build: $BIN" >&2
  exit 1
fi

run_case() {
  local dataset="$1"
  local bits="$2"
  local target="$3"
  local stem="flat_${dataset}_${bits}b_${target}"
  local log_path="$OUT_DIR/${stem}.log"
  local -a cmd=(
    "$BIN"
    "$dataset" "$bits" "$target"
    --config="$CONFIG"
    --data-root="$DATA_ROOT"
    --threads="$THREADS"
    --topk=1000
    --maxtrain=0
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
  if [[ -n "$CPUSET" ]]; then
    cmd=(taskset -c "$CPUSET" "${cmd[@]}")
  fi

  if [[ -f "$log_path" ]]; then
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
    echo "script.targets=$(IFS=,; echo "${TARGETS[*]}")"
    echo "script.start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    /usr/bin/time -v "${cmd[@]}"
  } >"$log_path" 2>&1
}

for target in "${TARGETS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      run_case "$dataset" "$bits" "$target"
    done
  done
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
