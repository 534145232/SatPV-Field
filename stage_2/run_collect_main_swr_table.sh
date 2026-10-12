#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/collect_main_swr_table.py" \
  --batch_size 16 \
  --num_workers 4 \
  --out_csv "${SCRIPT_DIR}/main_swr_table.csv" \
  "$@"
