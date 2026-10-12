#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/train_residual_convlstm_v0_full.py"
"${PYTHON}" "${SCRIPT_DIR}/train_residual_convlstm_v1_no_gradient.py"
"${PYTHON}" "${SCRIPT_DIR}/train_residual_convlstm_v2_no_center_weight.py"
"${PYTHON}" "${SCRIPT_DIR}/train_residual_convlstm_v3_no_residual_l1.py"
