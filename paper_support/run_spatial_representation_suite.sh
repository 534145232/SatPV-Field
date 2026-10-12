#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON:-python}"
OLD_DATA_ROOT="${OLD_DATA_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)/old_data}"
PV_ROOT="${PV_ROOT:-}"
STAGE3_DATA_DIR="${STAGE3_DATA_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/spatial_representation}"
SPLIT="${SPLIT:-test}"

TRAIN_PV_DIR="${TRAIN_PV_DIR:-}"
VAL_PV_DIR="${VAL_PV_DIR:-}"
BACKGROUND="${BACKGROUND:-contextily}"
BOUNDARY="${BOUNDARY:-${COASTLINE_SHP:-}}"

COMMON_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --old-data-root "${OLD_DATA_ROOT}"
  --split "${SPLIT}"
)

if [[ -n "${PV_ROOT}" ]]; then
  COMMON_ARGS+=(--pv-root "${PV_ROOT}")
fi
if [[ -n "${STAGE3_DATA_DIR}" ]]; then
  COMMON_ARGS+=(--stage3-data-dir "${STAGE3_DATA_DIR}")
fi
if [[ -n "${TRAIN_PV_DIR}" ]]; then
  COMMON_ARGS+=(--train-pv-dir "${TRAIN_PV_DIR}")
fi
if [[ -n "${VAL_PV_DIR}" ]]; then
  COMMON_ARGS+=(--val-pv-dir "${VAL_PV_DIR}")
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[1/2] sampled-station spatial representation consistency"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_spatial_representation_consistency.py" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/consistency"

VIS_ARGS=("${COMMON_ARGS[@]}")
VIS_ARGS+=(--background "${BACKGROUND}")
if [[ -n "${BOUNDARY}" ]]; then
  VIS_ARGS+=(--boundary "${BOUNDARY}")
fi

echo "[2/2] clear/cloudy/rapid visual validation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_spatial_representation_cases.py" \
  "${VIS_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/three_cases"

echo "Completed. Outputs: ${OUTPUT_ROOT}"
