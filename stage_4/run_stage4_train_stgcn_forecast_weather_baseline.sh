#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETAIL_CSV="${SCRIPT_DIR}/runs/20_stgcn_forecast_weather_baseline_seed2026/stgcn_forecast_weather_eval_detail.csv"

if [[ "${FORCE_STAGE4_STGCN_WEATHER:-0}" != "1" && -s "${DETAIL_CSV}" ]]; then
  echo "==== Skip Stage 4 STGCN + forecast-weather baseline: already finished ===="
  echo "detail: ${DETAIL_CSV}"
  exit 0
fi

echo "==== Train Stage 4 STGCN + forecast-weather baseline ===="
"${PYTHON}" "${SCRIPT_DIR}/20_train_stage4_stgcn_forecast_weather_baseline.py" "$@"
