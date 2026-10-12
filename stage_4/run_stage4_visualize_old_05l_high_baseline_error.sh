#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Visualize old-project 05l on high-baseline-error samples ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/09_visualize_old_05l_high_baseline_error.py" "$@"
