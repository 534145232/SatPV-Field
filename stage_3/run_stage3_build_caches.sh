#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[0/5] Import the one upstream regional station table into Stage 3"
"${PYTHON}" "${SCRIPT_DIR}/00_import_source_station_split.py"

echo "[1/5] Validate Stage 3 prerequisites and temporal protocol"
"${PYTHON}" "${SCRIPT_DIR}/00_preflight.py" --require_swr_checkpoint 1

echo "[2/5] Export Stage 1 center SWR cache adapter"
"${PYTHON}" "${SCRIPT_DIR}/00_export_stage1_center_swr_legacy_cache.py"

echo "[3/5] Build aligned multisource NWP cache"
"${PYTHON}" "${SCRIPT_DIR}/01_build_multisource_nwp_cache.py"

echo "[4/5] Select 50 target stations and cache the target-excluded PV field"
"${PYTHON}" "${SCRIPT_DIR}/02_select_targets_build_pvnorm_cache.py"

echo "[5/5] Cache center H8 history and three future SWR predictions"
"${PYTHON}" "${SCRIPT_DIR}/03_build_center_h8_swr_anchor_cache.py"
