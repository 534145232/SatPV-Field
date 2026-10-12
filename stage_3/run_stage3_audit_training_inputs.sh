#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/00_import_source_station_split.py"
"${PYTHON}" "${SCRIPT_DIR}/08_audit_stage3_training_inputs.py" "$@"
