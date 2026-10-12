#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Evaluate Stage 3 temporal-collapse ablation variants ===="
echo "Baseline 0.7 * last + 0.3 * mean: reuse ablation_motion_branch/warp_direct_gate metrics"
echo
echo "[1/2] Evaluate last-only temporal collapse"
"${PYTHON}" "${SCRIPT_DIR}/eval_collapse_last_only.py" "$@"

echo
echo "[2/2] Merge last-only metrics with the reusable baseline"
"${PYTHON}" "${SCRIPT_DIR}/combine_temporal_collapse_ablation_results.py"
