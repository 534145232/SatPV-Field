#!/usr/bin/env python3
"""Print Stage 3 paths and verify the required upstream artifacts."""

import argparse
from pathlib import Path

from stage3_paths import (
    SOURCE_REGIONAL_RUN_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    SWR_RATIO_3DCNN_CHECKPOINT,
    NWP_FARM_INFO_XLSX,
    NWP_STATION_MAP_CSV,
    TRAIN_PV_DIR,
    VAL_PV_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require_swr_checkpoint",
        type=int,
        default=0,
        help="Require the completed Stage 2 SWR checkpoint. Use 0 while preparing static Stage 3 caches.",
    )
    parser.add_argument(
        "--swr_checkpoint",
        type=Path,
        default=SWR_BASE_CHECKPOINT,
        help="Stage 2 SWR checkpoint to validate when --require_swr_checkpoint=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("==== Stage 3 PV-field pipeline preflight ====")
    print("PV protocol:")
    print("  issue T       : hourly 10:00-15:00 BJT")
    print("  PV history    : T-3h ... T-15min, 12 frames")
    print("  PV target     : T ... T+1h45min, 8 frames")
    print("  SWR observed  : T-4h ... T-1h, 4 frames")
    print("  SWR predicted : T ... T+2h, 3 frames")
    print("  SWR source    : Stage 1 center crop only; big 5x5 patches are Stage 2 training augmentation only")
    print("  SWR model     : big-patch Ratio3DCNN conv3d_decoder + residual ConvLSTM v2_no_center_weight")
    print("")
    required = {
        "source regional station split": SOURCE_REGIONAL_RUN_DIR / "station_split.csv",
        "Stage 1 center monthly NetCDF": STAGE1_CENTER_MONTHLY_DIR,
        "Stage 1 center frame cache": STAGE1_CENTER_FRAME_CACHE_DIR,
        "Stage 1 center 4+3 index": STAGE1_CENTER_INDEX_ALL,
        "PV train directory": TRAIN_PV_DIR,
        "PV val directory": VAL_PV_DIR,
        "NWP farm metadata": NWP_FARM_INFO_XLSX,
        "station-to-NWP map": NWP_STATION_MAP_CSV,
    }
    if int(args.require_swr_checkpoint):
        required["Stage 2 SWR checkpoint"] = args.swr_checkpoint
        required["Stage 2 frozen Ratio3DCNN conv3d_decoder base"] = SWR_RATIO_3DCNN_CHECKPOINT
    else:
        print("[INFO] Stage 2 SWR checkpoint check skipped; static Stage 3 caches can be prepared while training runs.")
    missing = []
    for name, path in required.items():
        exists = path.exists()
        print(f"[{'OK' if exists else 'MISSING'}] {name}: {path}")
        if not exists:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required upstream artifacts:\n" + "\n".join(missing))
    print("Preflight passed.")


if __name__ == "__main__":
    main()
