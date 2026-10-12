#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
"${PYTHON}" "${SCRIPT_DIR}/generate_stage4_combined_station_input_assets.py" "$@"
