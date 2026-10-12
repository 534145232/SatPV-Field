#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${RUN_ALL_STAGE4_BASELINES:-0}" == "1" ]]; then
  for model in lstm bilstm tcn transformer patchtst gcn_lstm stgcn; do
    detail_csv="${SCRIPT_DIR}/runs/16_${model}_baseline_seed2026/${model}_eval_detail.csv"
    if [[ "${FORCE_STAGE4_BASELINES:-0}" != "1" && -s "${detail_csv}" ]]; then
      echo "==== Skip Stage 4 additional baseline: ${model} already finished ===="
      echo "detail: ${detail_csv}"
      continue
    fi
    echo "==== Train Stage 4 additional baseline: ${model} ===="
    "${PYTHON}" "${SCRIPT_DIR}/16_train_stage4_additional_baselines.py" --model "${model}" "$@"
  done
else
  echo "==== Train Stage 4 additional baseline ===="
  "${PYTHON}" "${SCRIPT_DIR}/16_train_stage4_additional_baselines.py" "$@"
fi
