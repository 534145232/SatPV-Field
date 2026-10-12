#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Train Stage 3 anchor-weight ablation variants ===="
echo "Baseline fixed physical weights: reuse ablation_motion_branch/warp_direct_gate"
echo

echo "[1/2] Learnable anchor weights: random initialization"
"${PYTHON}" "${SCRIPT_DIR}/train_learnable_random_init.py" "$@"

echo
echo "[2/2] Learnable anchor weights: physical-weight initialization"
"${PYTHON}" "${SCRIPT_DIR}/train_learnable_physical_init.py" "$@"

echo
echo "Done. The fixed-parameter baseline is intentionally reused from the motion-branch experiment."
