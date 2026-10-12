#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Diagnose Stage 4 extreme station samples ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/08_diagnose_stage4_extreme_samples.py" "$@"
