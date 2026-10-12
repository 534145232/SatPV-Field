#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Stage 4 distance to nearest source station vs guidance improvement ===="
"${PYTHON}" "${SCRIPT_DIR}/24_eval_stage4_distance_vs_improvement.py" "$@"
