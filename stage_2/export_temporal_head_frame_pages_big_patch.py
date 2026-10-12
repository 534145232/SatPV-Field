"""Export ordered frame pages for big-patch-trained temporal-head checkpoints."""

import sys

from export_temporal_head_frame_pages import main
from train_3dcnn_ratio_swr import STAGE2_RUNS_DIR


BIG_PATCH_RUN_ROOT = (
    STAGE2_RUNS_DIR / "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f"
)


if __name__ == "__main__":
    if "--run_root" not in sys.argv:
        sys.argv.extend(["--run_root", str(BIG_PATCH_RUN_ROOT)])
    main()
