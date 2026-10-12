#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4 weather-condition analysis ===="
"${PYTHON}" "${SCRIPT_DIR}/18_eval_stage4_weather_condition_analysis.py" "$@"
