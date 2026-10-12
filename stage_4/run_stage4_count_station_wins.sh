#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Count Stage 4 station-wise wins ===="
"${PYTHON}" "${SCRIPT_DIR}/15_count_stage4_station_wins.py" "$@"
