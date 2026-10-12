#!/usr/bin/env python3
"""Train Stage 3 SWR-dependency ablations with the current best architecture."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
BASE_TRAIN = STAGE3_DIR / "ablation_motion_branch" / "train_motion_branch.py"
RUN_ROOT = STAGE3_DIR / "runs" / "ablation_swr_dependency"
TRAIN_VARIANTS = ("history_swr_only", "no_swr")


def main() -> None:
    extra_args = sys.argv[1:]
    for variant in TRAIN_VARIANTS:
        print(f"==== Train Stage 3 SWR-dependency ablation: {variant} ====", flush=True)
        cmd = [
            sys.executable,
            str(BASE_TRAIN),
            "--motion_branch",
            "warp_direct_gate",
            "--temporal_collapse",
            "last_only",
            "--swr_dependency",
            variant,
            "--run_root",
            str(RUN_ROOT / variant),
            *extra_args,
        ]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
