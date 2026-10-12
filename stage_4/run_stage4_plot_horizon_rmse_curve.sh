#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Plot Stage 4 horizon-wise RMSE curve ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/12_plot_stage4_horizon_rmse_curve.py" "$@"
