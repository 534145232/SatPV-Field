#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Collect Stage 4 AGCRN / Graph WaveNet NWP results ===="
"${PYTHON}" "${SCRIPT_DIR}/26_collect_stage4_graph_nwp_results.py" "$@"
