#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Collect Stage 4 all baseline/results tables ===="
"${PYTHON}" "${SCRIPT_DIR}/22_collect_stage4_all_results.py" "$@"
