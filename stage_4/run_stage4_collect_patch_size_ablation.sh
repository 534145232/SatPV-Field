#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Collect Stage 4 patch-size ablation metrics ===="
"${PYTHON}" "${SCRIPT_DIR}/14_collect_stage4_patch_size_ablation.py" "$@"
