#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/collect_stage3_experiment_tables.py" \
  --out_dir "${SCRIPT_DIR}/paper_tables" \
  "$@"
