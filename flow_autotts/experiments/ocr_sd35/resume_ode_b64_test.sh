#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
RAW_ROOT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/ode_b64_test/raw"
COMPACT_ROOT="${REPO_ROOT}/logs/flow_autotts/ocr_sd35/ode_b64_test/compact"
MODEL="${FLOW_TTS_MODEL:-${REPO_ROOT}/SD_3.5_med}"
OCR_MODEL="${FLOW_TTS_OCR_MODEL:-${REPO_ROOT}/third_party/paddleocr_models}"
DATASET="${FLOW_TTS_DATASET:-${REPO_ROOT}/flow_grpo/dataset/ocr}"
RESOLUTION="${FLOW_TTS_RESOLUTION:-512}"
GUIDANCE_SCALE="${FLOW_TTS_GUIDANCE_SCALE:-4.5}"
NOISE_LEVEL="${FLOW_TTS_NOISE_LEVEL:-0.7}"
SDE_TYPE="${FLOW_TTS_SDE_TYPE:-sde}"
DTYPE="${FLOW_TTS_DTYPE:-bfloat16}"
TEST_SAMPLE_SEED="${FLOW_TTS_TEST_SAMPLE_SEED:-42}"
DEVICES=(${FLOW_TTS_EVAL_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3})

VENV_SITE_PACKAGES="${REPO_ROOT}/.venv/lib/python3.12/site-packages"
NVIDIA_CUDNN_LIB="${VENV_SITE_PACKAGES}/nvidia/cudnn/lib"
NVIDIA_CUBLAS_LIB="${VENV_SITE_PACKAGES}/nvidia/cublas/lib"

if [[ -d "${NVIDIA_CUDNN_LIB}" && -d "${NVIDIA_CUBLAS_LIB}" ]]; then
  if [[ ! -e "${NVIDIA_CUDNN_LIB}/libcudnn.so" && -e "${NVIDIA_CUDNN_LIB}/libcudnn.so.9" ]]; then
    ln -s libcudnn.so.9 "${NVIDIA_CUDNN_LIB}/libcudnn.so"
  fi
  export LD_LIBRARY_PATH="${NVIDIA_CUDNN_LIB}:${NVIDIA_CUBLAS_LIB}:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

run_one() {
  local beta="$1"
  local target_nfe="$2"
  local shard_index="$3"
  local device="$4"
  local beta_slug="$5"
  local out_dir="${RAW_ROOT}/beta_${beta_slug}_target_${target_nfe}/shard_$(printf "%02d" "${shard_index}")"

  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/history.json" ]]; then
    echo "skip ${out_dir}"
    return 0
  fi

  "${PY}" -m flow_autotts.experiments.ocr_sd35.ode_baseline \
    --worker \
    --repo-root "${REPO_ROOT}" \
    --dataset "${DATASET}" \
    --split test \
    --sample-size 500 \
    --sample-seed "${TEST_SAMPLE_SEED}" \
    --num-shards 4 \
    --shard-index "${shard_index}" \
    --beta "${beta}" \
    --budget 64 \
    --target-total-nfe "${target_nfe}" \
    --total-nfe "${target_nfe}" \
    --output-dir "${out_dir}" \
    --model "${MODEL}" \
    --resolution "${RESOLUTION}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --noise-level "${NOISE_LEVEL}" \
    --sde-type "${SDE_TYPE}" \
    --device "${device}" \
    --full-split \
    --ocr-model "${OCR_MODEL}" \
    --text-encoder-device "${device}" \
    --score-device "${device}" \
    --dtype "${DTYPE}" \
    > "${out_dir}/stdout.log" 2> "${out_dir}/stderr.log"
}

launch_one() {
  local beta="$1"
  local target_nfe="$2"
  local shard_index="$3"
  local device="$4"
  local beta_slug="$5"
  run_one "${beta}" "${target_nfe}" "${shard_index}" "${device}" "${beta_slug}" &
  PIDS+=("$!")
}

cd "${REPO_ROOT}"

PIDS=()
launch_one 0.75 48 0 "${DEVICES[0]}" 0p75
launch_one 1.0 64 0 "${DEVICES[0]}" 1
launch_one 1.0 64 1 "${DEVICES[1]}" 1
launch_one 1.0 64 2 "${DEVICES[2]}" 1
launch_one 1.0 64 3 "${DEVICES[3]}" 1

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

mkdir -p "${COMPACT_ROOT}"
"${PY}" -m flow_autotts.experiments.ocr_sd35.build_ode_baseline_summary \
  --source-root "${RAW_ROOT}" \
  --output-dir "${COMPACT_ROOT}"
