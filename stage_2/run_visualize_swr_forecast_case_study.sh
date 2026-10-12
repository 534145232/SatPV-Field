#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Visualize Stage 2 SWR forecasting case study ===="
"${PYTHON}" "${SCRIPT_DIR}/visualize_swr_forecast_case_study.py" "$@"
