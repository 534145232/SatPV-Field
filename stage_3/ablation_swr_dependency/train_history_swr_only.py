#!/usr/bin/env python3
"""PyCharm entry for the history-SWR-only persistence-anchor ablation."""

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
BASE_TRAIN = STAGE3_DIR / "ablation_motion_branch" / "train_motion_branch.py"
RUN_ROOT = STAGE3_DIR / "runs" / "ablation_swr_dependency" / "history_swr_only"


if __name__ == "__main__":
    subprocess.check_call(
        [
            sys.executable,
            str(BASE_TRAIN),
            "--motion_branch",
            "warp_direct_gate",
            "--temporal_collapse",
            "last_only",
            "--swr_dependency",
            "history_swr_only",
            "--run_root",
            str(RUN_ROOT),
            *sys.argv[1:],
        ]
    )
