#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Train Stage 4: historical PV-field patch ablation ===="
"${PYTHON}" "${SCRIPT_DIR}/05h_train_per_station_temporal_with_historical_field_patch.py" "$@"
