#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4: 03 temporal baseline vs 05 Stage3 field patch ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/07_eval_compare_stage4_03_vs_05.py" "$@"
