#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETAIL_CSV="${SCRIPT_DIR}/runs/21_nearest3_station_pv_baseline_seed2026/nearest3_station_pv_eval_detail.csv"

if [[ "${FORCE_STAGE4_NEAREST3:-0}" != "1" && -s "${DETAIL_CSV}" ]]; then
  echo "==== Skip Stage 4 nearest-3 station PV baseline: already finished ===="
  echo "detail: ${DETAIL_CSV}"
  exit 0
fi

echo "==== Train Stage 4 nearest-3 station PV replacement baseline ===="
echo "default neighbor_source: last_observed"
"${PYTHON}" "${SCRIPT_DIR}/21_train_stage4_nearest3_station_pv_baseline.py" "$@"
