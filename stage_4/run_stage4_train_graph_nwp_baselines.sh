#!/bin/sh
set -eu

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_one() {
  local model="$1"
  local model_name
  if [[ "${model}" == "agcrn" ]]; then
    model_name="agcrn_nwp"
  elif [[ "${model}" == "graph_wavenet" ]]; then
    model_name="graph_wavenet_nwp"
  else
    echo "Unknown graph/NWP model: ${model}" >&2
    exit 2
  fi

  local detail_csv="${SCRIPT_DIR}/runs/23_${model_name}_baseline_seed2026/${model_name}_eval_detail.csv"
  if [[ "${FORCE_STAGE4_GRAPH_NWP_BASELINES:-0}" != "1" && -s "${detail_csv}" ]]; then
    echo "==== Skip Stage 4 graph/NWP baseline: ${model_name} already finished ===="
    echo "detail: ${detail_csv}"
    return
  fi
  echo "==== Train Stage 4 graph/NWP baseline: ${model_name} ===="
  "${PYTHON}" "${SCRIPT_DIR}/23_train_stage4_graph_nwp_baselines.py" --model "${model}" "$@"
}

if [[ "${RUN_ALL_STAGE4_GRAPH_NWP_BASELINES:-0}" == "1" ]]; then
  if [[ "${1:-}" == "agcrn" || "${1:-}" == "graph_wavenet" ]]; then
    shift
  fi
  run_one agcrn "$@"
  run_one graph_wavenet "$@"
else
  if [[ "${1:-}" == "agcrn" || "${1:-}" == "graph_wavenet" ]]; then
    model="$1"
    shift
  else
    model="agcrn"
  fi
  run_one "${model}" "$@"
fi
