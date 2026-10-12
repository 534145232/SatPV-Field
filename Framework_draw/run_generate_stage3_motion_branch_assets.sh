#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${SCRIPT_DIR}/generate_stage3_motion_branch_assets.py" "$@"
