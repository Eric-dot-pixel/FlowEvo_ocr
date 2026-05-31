#!/usr/bin/env bash
set -euo pipefail

export FLOW_TTS_CONTROLLER_KEY=pickscore_r0004_ffd4e330_ocr_eval_fixsteps10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_pickscore_controller_on_ocr.sh"
