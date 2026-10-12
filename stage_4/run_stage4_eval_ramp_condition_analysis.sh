#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Stage 4 ramp-condition evaluation ===="
"${PYTHON}" "${SCRIPT_DIR}/25_eval_stage4_ramp_condition_analysis.py" "$@"
