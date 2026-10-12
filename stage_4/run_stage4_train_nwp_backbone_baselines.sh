#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${RUN_ALL_STAGE4_NWP_BASELINES:-0}" == "1" ]]; then
  for model in lstm bilstm tcn transformer patchtst; do
    detail_csv="${SCRIPT_DIR}/runs/19_${model}_nwp_baseline_seed2026/${model}_nwp_eval_detail.csv"
    if [[ "${FORCE_STAGE4_NWP_BASELINES:-0}" != "1" && -s "${detail_csv}" ]]; then
      echo "==== Skip Stage 4 NWP backbone baseline: ${model} already finished ===="
      echo "detail: ${detail_csv}"
      continue
    fi
    echo "==== Train Stage 4 NWP backbone baseline: ${model} ===="
    "${PYTHON}" "${SCRIPT_DIR}/19_train_stage4_nwp_backbone_baselines.py" --model "${model}" "$@"
  done
else
  echo "==== Train Stage 4 NWP backbone baseline ===="
  "${PYTHON}" "${SCRIPT_DIR}/19_train_stage4_nwp_backbone_baselines.py" "$@"
fi
