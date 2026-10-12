#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[0/3] Import the one upstream regional station table into Stage 3"
"${PYTHON}" "${SCRIPT_DIR}/00_import_source_station_split.py"

echo "[1/3] Rebuild Stage 1 center SWR legacy adapter with hourly lookup keys"
"${PYTHON}" "${SCRIPT_DIR}/00_export_stage1_center_swr_legacy_cache.py"

echo "[2/3] Rebuild Stage 3 observed-H8 and future-NWP multisource cache"
"${PYTHON}" "${SCRIPT_DIR}/01_build_multisource_nwp_cache.py" --force_rebuild 1
