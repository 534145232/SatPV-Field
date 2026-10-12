#!/usr/bin/env python3
"""Evaluate random-initialized learnable temporal-anchor weights."""

import sys

from eval_learnable_anchor_weights import main
from stage3_paths import RUNS_DIR


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--anchor_weight_init",
        "random",
        "--run_dir",
        str(RUNS_DIR / "ablation_anchor_weights" / "learnable_random_init"),
        "--model_name",
        "swr_anchor_motion_fusion_pv_nwp_learnable_random_init",
    ]
    main()
