#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Stage 3 frame-wise DeltaCorr re-evaluation ===="
"${PYTHON}" "${SCRIPT_DIR}/reeval_stage3_framewise_delta_corr.py" "$@"
