#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_ROOT="${REPO_ROOT}/.venv"
VENV_SITE_PACKAGES="${VENV_ROOT}/lib/python3.12/site-packages"
NVIDIA_CUDNN_LIB="${VENV_SITE_PACKAGES}/nvidia/cudnn/lib"
NVIDIA_CUBLAS_LIB="${VENV_SITE_PACKAGES}/nvidia/cublas/lib"

BASELINE_DIR_DEFAULT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/bestof4_b64_train/compact"
BASELINE_DIR="${FLOW_TTS_BASELINE_DIR:-${BASELINE_DIR_DEFAULT}}"
BASELINE_SUMMARY="${FLOW_TTS_BASELINE_SUMMARY:-${BASELINE_DIR}/aggregate_summary.json}"

RESULT_TAG="${FLOW_TTS_RESULT_TAG:-ocr_train_codex_b64_bestof4_baseline_injected_r5}"
RESULT_DIR_DEFAULT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/${RESULT_TAG}"
HISTORY_DIR_DEFAULT="logs/flow_autotts/ocr_sd35/history_${RESULT_TAG}"
CODEX_LOG_PARENT_DEFAULT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/codex_logs_${RESULT_TAG}"
LAUNCH_LOG_DEFAULT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/${RESULT_TAG}_launcher.log"

export FLOW_TTS_SPLIT="${FLOW_TTS_SPLIT:-train}"
export FLOW_TTS_SAMPLE_SIZE="${FLOW_TTS_SAMPLE_SIZE:-500}"
export FLOW_TTS_SAMPLE_SEED="${FLOW_TTS_SAMPLE_SEED:-42}"
export FLOW_TTS_BUDGET="${FLOW_TTS_BUDGET:-64}"
export FLOW_TTS_BETAS="${FLOW_TTS_BETAS:-0 0.25 0.5 0.75 1.0}"
export FLOW_TTS_NUM_STEPS="${FLOW_TTS_NUM_STEPS:-10}"
export FLOW_TTS_PROMPT_PROFILE="${FLOW_TTS_PROMPT_PROFILE:-autotts}"
export FLOW_TTS_MODEL="${FLOW_TTS_MODEL:-${REPO_ROOT}/SD_3.5_med}"
export FLOW_TTS_OCR_MODEL="${FLOW_TTS_OCR_MODEL:-${REPO_ROOT}/third_party/paddleocr_models}"
export FLOW_TTS_DTYPE="${FLOW_TTS_DTYPE:-bfloat16}"
export FLOW_TTS_EVAL_DEVICES="${FLOW_TTS_EVAL_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
export FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES="${FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES:-${FLOW_TTS_EVAL_DEVICES}}"
export FLOW_TTS_EVAL_SCORE_DEVICES="${FLOW_TTS_EVAL_SCORE_DEVICES:-${FLOW_TTS_EVAL_DEVICES}}"

export WORKFLOW_RESULT_DIR="${WORKFLOW_RESULT_DIR:-${RESULT_DIR_DEFAULT}}"
export WORKFLOW_HISTORY_DIR="${WORKFLOW_HISTORY_DIR:-${HISTORY_DIR_DEFAULT}}"
export WORKFLOW_CODEX_LOG_PARENT="${WORKFLOW_CODEX_LOG_PARENT:-${CODEX_LOG_PARENT_DEFAULT}}"
export WORKFLOW_BASELINE_SUMMARY="${WORKFLOW_BASELINE_SUMMARY:-${BASELINE_SUMMARY}}"
export WORKFLOW_CONTEXT_PROMOTED_ONLY="${WORKFLOW_CONTEXT_PROMOTED_ONLY:-1}"
export WORKFLOW_ROUNDS="${WORKFLOW_ROUNDS:-5}"
export WORKFLOW_MIN_ROUNDS_BEFORE_STOP="${WORKFLOW_MIN_ROUNDS_BEFORE_STOP:-6}"
export FLOW_TTS_SHARD_OUTPUT_DIR="${FLOW_TTS_SHARD_OUTPUT_DIR:-${WORKFLOW_RESULT_DIR}/shards}"
export FLOW_TTS_LAUNCH_LOG="${FLOW_TTS_LAUNCH_LOG:-${LAUNCH_LOG_DEFAULT}}"

if [[ -d "${NVIDIA_CUDNN_LIB}" && -d "${NVIDIA_CUBLAS_LIB}" ]]; then
  if [[ ! -e "${NVIDIA_CUDNN_LIB}/libcudnn.so" && -e "${NVIDIA_CUDNN_LIB}/libcudnn.so.9" ]]; then
    ln -s libcudnn.so.9 "${NVIDIA_CUDNN_LIB}/libcudnn.so"
  fi
  export LD_LIBRARY_PATH="${NVIDIA_CUDNN_LIB}:${NVIDIA_CUBLAS_LIB}:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -f "${WORKFLOW_BASELINE_SUMMARY}" ]]; then
  echo "[flow_autotts ocr workflow b64] missing baseline summary: ${WORKFLOW_BASELINE_SUMMARY}" >&2
  exit 2
fi

mkdir -p "$(dirname "${FLOW_TTS_LAUNCH_LOG}")"
exec >>"${FLOW_TTS_LAUNCH_LOG}" 2>&1

echo "[flow_autotts ocr workflow b64] WORKFLOW_BASELINE_SUMMARY=${WORKFLOW_BASELINE_SUMMARY}"
echo "[flow_autotts ocr workflow b64] WORKFLOW_RESULT_DIR=${WORKFLOW_RESULT_DIR}"
echo "[flow_autotts ocr workflow b64] WORKFLOW_HISTORY_DIR=${WORKFLOW_HISTORY_DIR}"
echo "[flow_autotts ocr workflow b64] WORKFLOW_CODEX_LOG_PARENT=${WORKFLOW_CODEX_LOG_PARENT}"
echo "[flow_autotts ocr workflow b64] WORKFLOW_ROUNDS=${WORKFLOW_ROUNDS}"
echo "[flow_autotts ocr workflow b64] WORKFLOW_MIN_ROUNDS_BEFORE_STOP=${WORKFLOW_MIN_ROUNDS_BEFORE_STOP}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_SPLIT=${FLOW_TTS_SPLIT}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_SAMPLE_SIZE=${FLOW_TTS_SAMPLE_SIZE}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_SAMPLE_SEED=${FLOW_TTS_SAMPLE_SEED}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_MODEL=${FLOW_TTS_MODEL}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_OCR_MODEL=${FLOW_TTS_OCR_MODEL}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_DTYPE=${FLOW_TTS_DTYPE}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_EVAL_DEVICES=${FLOW_TTS_EVAL_DEVICES}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_SHARD_OUTPUT_DIR=${FLOW_TTS_SHARD_OUTPUT_DIR}"
echo "[flow_autotts ocr workflow b64] FLOW_TTS_LAUNCH_LOG=${FLOW_TTS_LAUNCH_LOG}"

exec "${SCRIPT_DIR}/run_workflow.sh"
