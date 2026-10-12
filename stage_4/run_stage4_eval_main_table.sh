#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4 main station-forecasting table ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/11_eval_stage4_main_table.py" "$@"
