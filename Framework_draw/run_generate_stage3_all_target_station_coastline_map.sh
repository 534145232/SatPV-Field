#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Generate Stage 3 all/target station coastline map ===="
"${PYTHON}" "${SCRIPT_DIR}/generate_stage3_all_target_station_coastline_map.py" "$@"
