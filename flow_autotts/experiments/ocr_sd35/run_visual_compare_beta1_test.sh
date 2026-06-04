#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_ROOT="${REPO_ROOT}/.venv"
VENV_SITE_PACKAGES="${VENV_ROOT}/lib/python3.12/site-packages"
NVIDIA_CUDNN_LIB="${VENV_SITE_PACKAGES}/nvidia/cudnn/lib"
NVIDIA_CUBLAS_LIB="${VENV_SITE_PACKAGES}/nvidia/cublas/lib"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
TEST_SAMPLE_SEED="${FLOW_TTS_TEST_SAMPLE_SEED:-42}"
CONTROLLER_NUM_STEPS="${FLOW_TTS_NUM_STEPS:-10}"
DATASET="${FLOW_TTS_DATASET:-${REPO_ROOT}/flow_grpo/dataset/ocr}"
CONTROLLER_PATH="${FLOW_TTS_CONTROLLER_PATH:-${REPO_ROOT}/logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0004_20260530_100756_0a3ad92e/flow_autotts/controllers/optimal.py}"
CONTROLLER_KEY="${FLOW_TTS_CONTROLLER_KEY:-ocr_best_workflow_r0004_0a3ad92e}"
BETA="${FLOW_TTS_BETA:-1.0}"
BUDGET="${FLOW_TTS_BUDGET:-64}"
BASELINE_TOTAL_NFE="${FLOW_TTS_BASELINE_TOTAL_NFE:-64}"
SAMPLE_SIZE="${FLOW_TTS_SAMPLE_SIZE:-100}"
FULL_SPLIT="${FLOW_TTS_FULL_SPLIT:-1}"

if [[ -d "${NVIDIA_CUDNN_LIB}" && -d "${NVIDIA_CUBLAS_LIB}" ]]; then
  if [[ ! -e "${NVIDIA_CUDNN_LIB}/libcudnn.so" && -e "${NVIDIA_CUDNN_LIB}/libcudnn.so.9" ]]; then
    ln -s libcudnn.so.9 "${NVIDIA_CUDNN_LIB}/libcudnn.so"
  fi
  export LD_LIBRARY_PATH="${NVIDIA_CUDNN_LIB}:${NVIDIA_CUBLAS_LIB}:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

RESULT_TAG="${RESULT_TAG:-${CONTROLLER_KEY}_beta1_visual_compare_test}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/flow_autotts/ocr_sd35/${RESULT_TAG}}"
IMAGE_ROOT="${IMAGE_ROOT:-${OUTPUT_DIR}/samples}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-${OUTPUT_DIR}/shards}"
WORKFLOW_LOG_DIR="${WORKFLOW_LOG_DIR:-${OUTPUT_DIR}/workflow_logs}"

BOOTSTRAP_LOG="${WORKFLOW_LOG_DIR}/visual_compare.bootstrap.log"
STDOUT_LOG="${WORKFLOW_LOG_DIR}/visual_compare.stdout.log"
STDERR_LOG="${WORKFLOW_LOG_DIR}/visual_compare.stderr.log"
ENV_LOG="${WORKFLOW_LOG_DIR}/visual_compare.env.log"

mkdir -p "${WORKFLOW_LOG_DIR}" "${OUTPUT_DIR}" "${IMAGE_ROOT}" "${SHARD_OUTPUT_DIR}"
exec >>"${BOOTSTRAP_LOG}" 2>&1

if [[ ! -f "${CONTROLLER_PATH}" ]]; then
  echo "[ocr-visual-compare-beta1] controller not found: ${CONTROLLER_PATH}" >&2
  exit 2
fi

COMMON_ARGS=(
  --devices "${DEVICES}"
  --text-encoder-devices "${TEXT_ENCODER_DEVICES}"
  --score-devices "${SCORE_DEVICES}"
  --dataset "${DATASET}"
  --split test
  --sample-size "${SAMPLE_SIZE}"
  --sample-seed "${TEST_SAMPLE_SEED}"
  --beta "${BETA}"
  --budget "${BUDGET}"
  --baseline-total-nfe "${BASELINE_TOTAL_NFE}"
  --controller-num-steps "${CONTROLLER_NUM_STEPS}"
  --output-dir "${OUTPUT_DIR}"
  --image-root "${IMAGE_ROOT}"
  --shard-output-dir "${SHARD_OUTPUT_DIR}"
  --model "${MODEL}"
  --controller-path "${CONTROLLER_PATH}"
  --controller-key "${CONTROLLER_KEY}"
  --resolution "${RESOLUTION}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --noise-level "${NOISE_LEVEL}"
  --sde-type "${SDE_TYPE}"
  --dtype "${DTYPE}"
)

if [[ "${FULL_SPLIT}" != "0" ]]; then
  COMMON_ARGS+=(--full-split)
fi

if [[ -n "${OCR_MODEL}" ]]; then
  COMMON_ARGS+=(--ocr-model "${OCR_MODEL}")
fi

{
  echo "[ocr-visual-compare-beta1] REPO_ROOT=${REPO_ROOT}"
  echo "[ocr-visual-compare-beta1] PYTHON_BIN=${PYTHON_BIN}"
  echo "[ocr-visual-compare-beta1] CONTROLLER_PATH=${CONTROLLER_PATH}"
  echo "[ocr-visual-compare-beta1] CONTROLLER_KEY=${CONTROLLER_KEY}"
  echo "[ocr-visual-compare-beta1] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[ocr-visual-compare-beta1] IMAGE_ROOT=${IMAGE_ROOT}"
  echo "[ocr-visual-compare-beta1] SHARD_OUTPUT_DIR=${SHARD_OUTPUT_DIR}"
  echo "[ocr-visual-compare-beta1] DATASET=${DATASET}"
  echo "[ocr-visual-compare-beta1] BETA=${BETA}"
  echo "[ocr-visual-compare-beta1] BUDGET=${BUDGET}"
  echo "[ocr-visual-compare-beta1] BASELINE_TOTAL_NFE=${BASELINE_TOTAL_NFE}"
  echo "[ocr-visual-compare-beta1] CONTROLLER_NUM_STEPS=${CONTROLLER_NUM_STEPS}"
  echo "[ocr-visual-compare-beta1] FULL_SPLIT=${FULL_SPLIT}"
  echo "[ocr-visual-compare-beta1] SAMPLE_SIZE=${SAMPLE_SIZE}"
  echo "[ocr-visual-compare-beta1] TEST_SAMPLE_SEED=${TEST_SAMPLE_SEED}"
  echo "[ocr-visual-compare-beta1] DEVICES=${DEVICES}"
  echo "[ocr-visual-compare-beta1] OCR_MODEL=${OCR_MODEL}"
  echo "[ocr-visual-compare-beta1] STDOUT_LOG=${STDOUT_LOG}"
  echo "[ocr-visual-compare-beta1] STDERR_LOG=${STDERR_LOG}"
} >"${ENV_LOG}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m flow_autotts.experiments.ocr_sd35.visual_compare_beta1 \
  "${COMMON_ARGS[@]}" \
  >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
