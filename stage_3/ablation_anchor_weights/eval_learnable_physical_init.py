#!/usr/bin/env python3
"""Evaluate physically initialized learnable temporal-anchor weights."""

import sys

from eval_learnable_anchor_weights import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--anchor_weight_init",
        "physical",
        "--run_dir",
        "/code/Final_3_stage_pipeline/stage_3/runs/ablation_anchor_weights/learnable_physical_init",
        "--model_name",
        "swr_anchor_motion_fusion_pv_nwp_learnable_physical_init",
    ]
    main()
