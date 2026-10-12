#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Stage 3 common-subset baseline re-eval ===="
"${PYTHON}" "${SCRIPT_DIR}/eval_stage3_base_pvnorm_field_models.py" \
  --out_dir "${SCRIPT_DIR}/runs/base_pvnorm_field_model" \
  --models "persistence_last,pv_only,pv_h8,pv_nwp,pv_h8_nwp" \
  --common_swr_anchor_subset 1

echo "==== Stage 3 common-subset proposed MFP re-eval ===="
"${PYTHON}" "${SCRIPT_DIR}/ablation_motion_branch/eval_all_motion_branches.py" \
  --motion_branch "warp_direct_gate" \
  --temporal_collapse "last_only" \
  --swr_dependency "full_swr_future" \
  --run_root "${SCRIPT_DIR}/runs/ablation_motion_branch_last_only" \
  --out_dir "${SCRIPT_DIR}/runs/ablation_swr_dependency/eval_full_swr_future"

echo "==== Stage 3 paper-table collection ===="
"${PYTHON}" "${SCRIPT_DIR}/collect_stage3_experiment_tables.py" \
  --out_dir "${SCRIPT_DIR}/paper_tables"
