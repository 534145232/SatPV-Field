#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/04_train_swr_anchor_motion_fusion_pvnorm_field_model.py"
