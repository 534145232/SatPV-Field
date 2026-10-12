#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODE in direct_only warp_only warp_direct_gate; do
  echo "==== Train Stage 3 motion branch: ${MODE} ===="
  "${PYTHON}" "${SCRIPT_DIR}/train_motion_branch.py" --motion_branch "${MODE}" "$@"
done
