#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE4_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${STAGE4_DIR}/05l_train_per_station_temporal_with_stage3_field_patch.py"
RUN_ROOT="${STAGE4_DIR}/runs/ablation_pv_field_patch_size"

# 5x5 is the current full model and is usually already available at:
#   stage_4/runs/05l_per_station_temporal_stage3_field_patch_seed2026
# Set INCLUDE_5_RETRAIN=1 if you want to retrain it inside this ablation folder.
SIZES="${SIZES:-1 3 7}"
if [[ "${INCLUDE_5_RETRAIN:-0}" == "1" ]]; then
  SIZES="${SIZES} 5"
fi

mkdir -p "${RUN_ROOT}"

for SIZE in ${SIZES}; do
  OUT_DIR="${RUN_ROOT}/patch_${SIZE}x${SIZE}"
  echo "==== Stage 4 PV-field patch-size ablation: ${SIZE}x${SIZE} ===="
  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --field_patch_size "${SIZE}" \
    --out_dir "${OUT_DIR}" \
    "$@"
done

echo "==== Patch-size ablation training complete ===="
echo "runs: ${RUN_ROOT}"
