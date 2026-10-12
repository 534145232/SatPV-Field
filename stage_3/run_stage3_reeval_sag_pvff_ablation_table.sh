#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_optional() {
  local label="$1"
  shift
  echo
  echo "==== ${label} ===="
  if "$@"; then
    echo "[ok] ${label}"
  else
    echo "[skip] ${label} failed or required checkpoints are missing"
  fi
}

echo "==== Re-evaluate SAG-PVFF internal ablations ===="

run_optional "Motion branches with last-only F-TAI setting" \
  "${PYTHON}" "${SCRIPT_DIR}/ablation_motion_branch_last_only/eval_all_motion_branches_last_only.py"

run_optional "Fixed 0.5 fusion from gated checkpoint" \
  "${PYTHON}" "${SCRIPT_DIR}/ablation_motion_branch_last_only/eval_fixed_fusion_from_gate.py"

run_optional "SWR dependency variants" \
  "${PYTHON}" "${SCRIPT_DIR}/ablation_swr_dependency/eval_all_swr_dependency.py"

run_optional "Anchor-weight / F-TAI variants" \
  bash "${SCRIPT_DIR}/ablation_anchor_weights/run_eval_all_anchor_weight_ablation.sh"

echo
echo "==== Collect compact SAG-PVFF ablation table ===="
"${PYTHON}" "${SCRIPT_DIR}/collect_stage3_sag_pvff_ablation_table.py" "$@"

