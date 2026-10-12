#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] Refresh Stage 1 indices with training-compatible columns"
"${PYTHON}" "${SCRIPT_DIR}/03_build_indices.py"

echo "[2/2] Build center clear-sky SWR cache"
"${PYTHON}" "${SCRIPT_DIR}/07_build_center_clear_sky_cache.py"

echo "Stage 1 index refresh complete."
