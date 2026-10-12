#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] center-only temporal head: last_mean"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_center_head_last_mean.py"

echo "[2/4] center-only temporal head: last_only"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_center_head_last_only.py"

echo "[3/4] center-only temporal head: flatten"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_center_head_flatten.py"

echo "[4/4] center-only temporal head: conv3d_decoder"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_center_head_conv3d_decoder.py"
