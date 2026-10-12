#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${WORKERS:-1}"

echo "[1/5] Build leakage-safe 4-history -> 3-future indices"
"${PYTHON}" "${SCRIPT_DIR}/03_build_indices.py"

echo "[2/5] Build big monthly frame cache"
"${PYTHON}" "${SCRIPT_DIR}/04_build_big_frame_cache.py" --num_workers "${WORKERS}" --time_chunk 4

echo "[3/5] Build center monthly frame cache"
"${PYTHON}" "${SCRIPT_DIR}/05_build_center_frame_cache.py" --num_workers "${WORKERS}"

echo "[4/5] Build center-only sample cache"
"${PYTHON}" "${SCRIPT_DIR}/06_build_center_sample_cache.py" --num_workers "${WORKERS}"

echo "[5/5] Build center clear-sky SWR cache"
"${PYTHON}" "${SCRIPT_DIR}/07_build_center_clear_sky_cache.py"
