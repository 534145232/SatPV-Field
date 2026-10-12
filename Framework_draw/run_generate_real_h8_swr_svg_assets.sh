#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Generate real H8/SWR SVG framework assets ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_real_h8_swr_svg_assets.py" "$@"
