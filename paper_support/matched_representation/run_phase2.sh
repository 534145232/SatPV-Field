#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

cmd=(
  "${PYTHON}"
  "${SCRIPT_DIR}/phase2_sanity_tiny_overfit.py"
  --project-root "${PROJECT_ROOT}"
)

if [[ -n "${OLD_DATA_ROOT:-}" ]]; then
  cmd+=(--old-data-root "${OLD_DATA_ROOT}")
fi
if [[ -n "${STAGE3_DATA_DIR:-}" ]]; then
  cmd+=(--stage3-data-dir "${STAGE3_DATA_DIR}")
fi
if [[ -n "${STAGE3_MODEL_DIR:-}" ]]; then
  cmd+=(--stage3-model-dir "${STAGE3_MODEL_DIR}")
fi
if [[ -n "${PV_ROOT:-}" ]]; then
  cmd+=(--pv-root "${PV_ROOT}")
fi

cmd+=("$@")
"${cmd[@]}"
