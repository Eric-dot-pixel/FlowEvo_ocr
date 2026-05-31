#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_ROOT="${FLOW_TTS_OCR_MODEL_ROOT:-${REPO_ROOT}/third_party/paddleocr_models}"

mkdir -p "${MODEL_ROOT}/det" "${MODEL_ROOT}/rec" "${MODEL_ROOT}/cls"

download_and_extract() {
  local url="$1"
  local target_dir="$2"
  local tar_name="$3"

  cd "${target_dir}"
  wget --tries=3 --timeout=60 -O "${tar_name}" "${url}"
  tar -xf "${tar_name}"
}

download_and_extract \
  "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar" \
  "${MODEL_ROOT}/det" \
  "en_PP-OCRv3_det_infer.tar"

download_and_extract \
  "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar" \
  "${MODEL_ROOT}/rec" \
  "en_PP-OCRv3_rec_infer.tar"

download_and_extract \
  "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" \
  "${MODEL_ROOT}/cls" \
  "ch_ppocr_mobile_v2.0_cls_infer.tar"

echo "Downloaded PaddleOCR models to ${MODEL_ROOT}"
