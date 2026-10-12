#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/1] Cache center H8 history and versioned future SWR predictions"
"${PYTHON}" "${SCRIPT_DIR}/03_build_center_h8_swr_anchor_cache.py" "$@"
