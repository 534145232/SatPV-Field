#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Visualize Stage 4 station-level case comparisons ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/13_visualize_stage4_case_comparisons.py" "$@"
