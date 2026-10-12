#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/stage3_base_pvnorm_field_model.py" \
  --out_dir "${SCRIPT_DIR}/runs/base_pvnorm_field_model" \
  --models "pv_only,pv_h8,pv_nwp,pv_h8_nwp" \
  "$@"
