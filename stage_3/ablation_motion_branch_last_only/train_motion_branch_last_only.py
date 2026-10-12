#!/usr/bin/env python3
"""Run all Stage 3 motion-branch ablations with last-only temporal collapse."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_paths import RUNS_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
BASE_TRAIN = STAGE3_DIR / "ablation_motion_branch" / "train_motion_branch.py"
RUN_ROOT = RUNS_DIR / "ablation_motion_branch_last_only"
MOTION_BRANCHES = ("direct_only", "warp_only", "warp_direct_gate")


def main() -> None:
    extra_args = sys.argv[1:]
    for mode in MOTION_BRANCHES:
        print(f"==== Train Stage 3 last-only motion branch: {mode} ====", flush=True)
        cmd = [
            sys.executable,
            str(BASE_TRAIN),
            "--temporal_collapse",
            "last_only",
            "--run_root",
            str(RUN_ROOT),
            "--motion_branch",
            mode,
            *extra_args,
        ]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
