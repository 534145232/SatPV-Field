#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Evaluate Stage 3 anchor-weight ablation variants ===="
echo "Baseline fixed physical weights: reuse ablation_motion_branch/warp_direct_gate metrics"
echo

echo "[1/3] Learnable anchor weights: random initialization"
"${PYTHON}" "${SCRIPT_DIR}/eval_learnable_random_init.py" "$@"

echo
echo "[2/3] Learnable anchor weights: physical-weight initialization"
"${PYTHON}" "${SCRIPT_DIR}/eval_learnable_physical_init.py" "$@"

echo
echo "[3/3] Merge both variants with the reusable fixed-parameter baseline"
"${PYTHON}" "${SCRIPT_DIR}/combine_anchor_weight_ablation_results.py"
