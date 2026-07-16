#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${ROOT}/.venv-mmeb"
DATA_ROOT="${ROOT}/mmeb_v2_bench/data"
CACHE_ROOT="${ROOT}/mmeb_v2_bench/cache"
TMP_ROOT="${CACHE_ROOT}/tmp"
CACHE_DB="${ROOT}/mmeb_v2_bench/cache/embeddings.sqlite"
ANNOTATION_CACHE="${ROOT}/mmeb_v2_bench/cache/annotations"
BUNDLE="${ROOT}/mmeb_v2_bench/bundles/mm_core12_gemini2_768_trainpool"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-4}"
GEMINI_MIN_REQUEST_INTERVAL_SECONDS="${GEMINI_MIN_REQUEST_INTERVAL_SECONDS:-8}"
GEMINI_MAX_REQUEST_INTERVAL_SECONDS="${GEMINI_MAX_REQUEST_INTERVAL_SECONDS:-30}"
GEMINI_QUOTA_BACKOFF_SECONDS="${GEMINI_QUOTA_BACKOFF_SECONDS:-180}"
GEMINI_MAX_QUOTA_BACKOFF_SECONDS="${GEMINI_MAX_QUOTA_BACKOFF_SECONDS:-900}"
GEMINI_QUOTA_INTERVAL_SCALE="${GEMINI_QUOTA_INTERVAL_SCALE:-1.35}"

if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "set GOOGLE_API_KEY or GEMINI_API_KEY before running this script" >&2
  exit 1
fi
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  export GOOGLE_API_KEY="${GEMINI_API_KEY}"
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="${GOOGLE_API_KEY}"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

cd "${ROOT}"
mkdir -p \
  "${ROOT}/logs" \
  "${DATA_ROOT}" \
  "${CACHE_ROOT}" \
  "${TMP_ROOT}" \
  "${CACHE_ROOT}/hf_home" \
  "${CACHE_ROOT}/hf_hub" \
  "${CACHE_ROOT}/hf_datasets" \
  "${CACHE_ROOT}/xdg" \
  "${CACHE_ROOT}/modelscope" \
  "$(dirname "${CACHE_DB}")" \
  "${ANNOTATION_CACHE}" \
  "$(dirname "${BUNDLE}")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/python"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export GEMINI_MIN_REQUEST_INTERVAL_SECONDS
export GEMINI_MAX_REQUEST_INTERVAL_SECONDS
export GEMINI_QUOTA_BACKOFF_SECONDS
export GEMINI_MAX_QUOTA_BACKOFF_SECONDS
export GEMINI_QUOTA_INTERVAL_SCALE

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

if ! "${VENV}/bin/python" - <<'PY'
import addict
import datasets
import google.genai
import huggingface_hub
import hf_transfer
import numpy
import PIL
from modelscope.msdatasets import MsDataset
import pyarrow
import tqdm
import yaml
PY
then
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/python" -m pip install \
    numpy tqdm pyyaml pillow pyarrow datasets huggingface_hub google-genai modelscope addict hf_transfer
fi

echo "[mmeb] download/resume media into ${DATA_ROOT}"
until env \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    "${VENV}/bin/python" -m mmeb_v2_bench.download_media \
    --output-dir="${DATA_ROOT}" \
    --allow-pattern="image-tasks/**" \
    --allow-pattern="video-tasks/data/**" \
    --allow-pattern="video-tasks/frames/*video_cls*.tar.gz*" \
    --allow-pattern="video-tasks/frames/*video_qa*.tar.gz*" \
    --allow-pattern="video-tasks/frames/*video_mret*.tar.gz*" \
    --allow-pattern="visdoc-tasks/**"; do
  date '+[mmeb] %F %T download failed; retrying in 120s'
  sleep 120
done

echo "[mmeb] export/resume embeddings and vector bundle: ${BUNDLE}"
until "${VENV}/bin/python" -m mmeb_v2_bench.cli \
    --dataset-root="${DATA_ROOT}" \
    --annotation-backend=hf \
    --task MSCOCO_t2i \
    --task ImageNet-1K \
    --task Kinetics-700 \
    --task QVHighlight \
    --task ViDoRe_docvqa \
    --task MMLongBench-doc \
    --train-pool-group mm_core12 \
    --annotation-cache-dir="${ANNOTATION_CACHE}" \
    --embedder=gemini \
    --gemini-model=gemini-embedding-2-preview \
    --output-dim=768 \
    --embed-batch-size="${EMBED_BATCH_SIZE}" \
    --cache-db="${CACHE_DB}" \
    --export-bundle="${BUNDLE}"; do
  date '+[mmeb] %F %T export failed; retrying in 120s'
  sleep 120
done

echo "[mmeb] done: ${BUNDLE}"
