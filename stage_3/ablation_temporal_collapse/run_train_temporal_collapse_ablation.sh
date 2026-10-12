#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Train Stage 3 temporal-collapse ablation variants ===="
echo "Baseline 0.7 * last + 0.3 * mean: reuse ablation_motion_branch/warp_direct_gate"
echo
echo "[1/1] Train last-only temporal collapse"
"${PYTHON}" "${SCRIPT_DIR}/train_collapse_last_only.py" "$@"

echo
echo "Done. The 0.7-last + 0.3-mean baseline is intentionally reused from the motion-branch experiment."
