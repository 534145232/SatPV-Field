#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[0/4] Import the one upstream regional station table into Stage 3"
"${PYTHON}" "${SCRIPT_DIR}/00_import_source_station_split.py"

echo "[1/4] Validate prerequisites for static Stage 3 caches"
"${PYTHON}" "${SCRIPT_DIR}/00_preflight.py" --require_swr_checkpoint 0

echo "[2/4] Export Stage 1 center SWR cache adapter"
"${PYTHON}" "${SCRIPT_DIR}/00_export_stage1_center_swr_legacy_cache.py"

echo "[3/4] Build aligned multisource NWP cache"
"${PYTHON}" "${SCRIPT_DIR}/01_build_multisource_nwp_cache.py"

echo "[4/4] Select 50 target stations and cache the target-excluded PV field"
"${PYTHON}" "${SCRIPT_DIR}/02_select_targets_build_pvnorm_cache.py"
