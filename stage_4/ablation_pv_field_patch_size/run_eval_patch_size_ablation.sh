#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Eval Stage 4 PV-field patch-size ablation ===="
"${PYTHON}" "${SCRIPT_DIR}/eval_patch_size_ablation.py" "$@"
