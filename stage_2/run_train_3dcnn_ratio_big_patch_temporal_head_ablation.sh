#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] big-patch temporal head: last_mean"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_big_patch_head_last_mean.py"

echo "[2/4] big-patch temporal head: last_only"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_big_patch_head_last_only.py"

echo "[3/4] big-patch temporal head: flatten"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_big_patch_head_flatten.py"

echo "[4/4] big-patch temporal head: conv3d_decoder"
"${PYTHON}" "${SCRIPT_DIR}/train_3dcnn_ratio_big_patch_head_conv3d_decoder.py"
