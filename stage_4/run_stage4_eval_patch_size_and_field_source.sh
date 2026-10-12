#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4: patch size and PV-field source ablations ===="
"${PYTHON_BIN}" "${SCRIPT_DIR}/10_eval_stage4_patch_size_and_field_source.py" "$@"
