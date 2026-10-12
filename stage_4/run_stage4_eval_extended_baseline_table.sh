#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4 extended baseline table ===="
"${PYTHON}" "${SCRIPT_DIR}/17_eval_stage4_extended_baseline_table.py" "$@"
