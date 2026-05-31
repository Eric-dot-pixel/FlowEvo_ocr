#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_ROOT="${REPO_ROOT}/.venv"
VENV_SITE_PACKAGES="${VENV_ROOT}/lib/python3.12/site-packages"
NVIDIA_CUDNN_LIB="${VENV_SITE_PACKAGES}/nvidia/cudnn/lib"
NVIDIA_CUBLAS_LIB="${VENV_SITE_PACKAGES}/nvidia/cublas/lib"

DEVICES="${FLOW_TTS_EVAL_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
TEXT_ENCODER_DEVICES="${FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES:-${DEVICES}}"
SCORE_DEVICES="${FLOW_TTS_EVAL_SCORE_DEVICES:-${DEVICES}}"
MODEL="${FLOW_TTS_MODEL:-${REPO_ROOT}/SD_3.5_med}"
OCR_MODEL="${FLOW_TTS_OCR_MODEL:-${REPO_ROOT}/third_party/paddleocr_models}"
DTYPE="${FLOW_TTS_DTYPE:-bfloat16}"
RESOLUTION="${FLOW_TTS_RESOLUTION:-512}"
GUIDANCE_SCALE="${FLOW_TTS_GUIDANCE_SCALE:-4.5}"
NOISE_LEVEL="${FLOW_TTS_NOISE_LEVEL:-0.7}"
SDE_TYPE="${FLOW_TTS_SDE_TYPE:-sde}"
TRAIN_SAMPLE_SIZE="${FLOW_TTS_TRAIN_SAMPLE_SIZE:-500}"
TRAIN_SAMPLE_SEED="${FLOW_TTS_TRAIN_SAMPLE_SEED:-42}"
DATASET="${FLOW_TTS_DATASET:-${REPO_ROOT}/flow_grpo/dataset/ocr}"

if [[ -d "${NVIDIA_CUDNN_LIB}" && -d "${NVIDIA_CUBLAS_LIB}" ]]; then
  if [[ ! -e "${NVIDIA_CUDNN_LIB}/libcudnn.so" && -e "${NVIDIA_CUDNN_LIB}/libcudnn.so.9" ]]; then
    ln -s libcudnn.so.9 "${NVIDIA_CUDNN_LIB}/libcudnn.so"
  fi
  export LD_LIBRARY_PATH="${NVIDIA_CUDNN_LIB}:${NVIDIA_CUBLAS_LIB}:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

TRAIN_ROOT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/ode_b64_train"
TEST_ROOT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/ode_b64_test"

COMMON_ARGS=(
  --devices "${DEVICES}"
  --text-encoder-devices "${TEXT_ENCODER_DEVICES}"
  --score-devices "${SCORE_DEVICES}"
  --betas 0 0.25 0.5 0.75 1.0
  --target-nfes 8 20 36 48 64
  --budget 64
  --dataset "${DATASET}"
  --model "${MODEL}"
  --resolution "${RESOLUTION}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --noise-level "${NOISE_LEVEL}"
  --sde-type "${SDE_TYPE}"
  --dtype "${DTYPE}"
)

if [[ -n "${OCR_MODEL}" ]]; then
  COMMON_ARGS+=(--ocr-model "${OCR_MODEL}")
fi

cd "${REPO_ROOT}"

python -m flow_autotts.experiments.ocr_sd35.ode_baseline \
  "${COMMON_ARGS[@]}" \
  --split train \
  --sample-size "${TRAIN_SAMPLE_SIZE}" \
  --sample-seed "${TRAIN_SAMPLE_SEED}" \
  --source-root "${TRAIN_ROOT}/raw" \
  --output-dir "${TRAIN_ROOT}/compact"

python -m flow_autotts.experiments.ocr_sd35.ode_baseline \
  "${COMMON_ARGS[@]}" \
  --split test \
  --full-split \
  --sample-seed "${TRAIN_SAMPLE_SEED}" \
  --source-root "${TEST_ROOT}/raw" \
  --output-dir "${TEST_ROOT}/compact"
