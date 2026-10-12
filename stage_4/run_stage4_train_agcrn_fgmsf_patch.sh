#!/bin/sh
set -eu

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==== Train Stage 4 AGCRN + FG-MSF field patch ===="
"${PYTHON}" "${SCRIPT_DIR}/27_train_stage4_agcrn_fgmsf_patch.py" "$@"
