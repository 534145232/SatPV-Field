#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Diagnose Stage 2 SWR vertical stripe artifacts ===="
"${PYTHON}" "${SCRIPT_DIR}/diagnose_vertical_stripes.py" "$@"
