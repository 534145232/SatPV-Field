#!/usr/bin/env python3
"""Evaluate Stage 3 motion-branch ablations trained with last-only collapse."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
BASE_EVAL = STAGE3_DIR / "ablation_motion_branch" / "eval_all_motion_branches.py"
RUN_ROOT = STAGE3_DIR / "runs" / "ablation_motion_branch_last_only"
OUT_DIR = RUN_ROOT / "eval_all"


def main() -> None:
    cmd = [
        sys.executable,
        str(BASE_EVAL),
        "--temporal_collapse",
        "last_only",
        "--run_root",
        str(RUN_ROOT),
        "--out_dir",
        str(OUT_DIR),
        *sys.argv[1:],
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
