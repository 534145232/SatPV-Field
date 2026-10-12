#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/visualize_pv_field_case_study.py" \
  --num_cases 8 \
  --case_select_mode random \
  --scan_count 300 \
  --scan_mode even \
  --horizons 1,4,7 \
  --pv_cmap shared_swr \
  --err_cmap shared_flow \
  --pv_vmin 0.0 \
  --pv_vmax 1.2 \
  --err_vmin 0.0 \
  --err_vmax 0.4 \
  "$@"
