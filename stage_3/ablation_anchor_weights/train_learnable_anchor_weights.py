#!/usr/bin/env python3
"""Train a Stage 3 motion-fusion ablation with learnable 8x4 anchor weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

import stage3_motion_fusion_model_impl as module
from stage3_cli import inject_defaults
from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    ensure_stage3_dirs,
)
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, cli_path, cli_path_or_none, resolve_active_cache_dir
from learnable_anchor_weights import install_learnable_anchor_weight_factory


RUN_ROOT = STAGE3_DIR / "runs" / "ablation_anchor_weights"


def export_trained_weights(checkpoint_path: Path, out_dir: Path, init_mode: str) -> None:
    import torch

    if not checkpoint_path.exists():
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    logits = state["model_state"]["future_anchor_logits"].detach().cpu().numpy().astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
    labels = ["A0_T-1h", "A1_T", "A2_T+1h", "A3_T+2h"]
    frame = pd.DataFrame(weights, columns=labels)
    frame.insert(0, "pv_horizon", [f"h{index + 1}" for index in range(len(frame))])
    frame["row_sum"] = weights.sum(axis=1)
    frame.to_csv(out_dir / "trained_anchor_weights.csv", index=False)
    with (out_dir / "anchor_weight_ablation_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "anchor_weight_init": str(init_mode),
                "constraint": "softmax per PV horizon: non-negative weights with row sum 1",
                "anchors": labels,
                "trained_weights": weights.tolist(),
                "checkpoint": str(checkpoint_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("==== Trained Stage 3 anchor weights ====")
    print(frame.to_string(index=False))


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor_weight_init", choices=["random", "physical"], required=True)
    parser.add_argument("--anchor_weight_random_std", type=float, default=0.10)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    wrapper = parse_wrapper_args()
    ensure_stage3_dirs()
    anchor_cache_dir = cli_path_or_none("--swr_anchor_cache_dir")
    if anchor_cache_dir is None:
        anchor_cache_dir = resolve_active_cache_dir(required=True)
    swr_checkpoint = cli_path("--swr_checkpoint", SWR_BASE_CHECKPOINT)
    assert_cache_matches_checkpoint(anchor_cache_dir, swr_checkpoint)
    variant = f"learnable_{wrapper.anchor_weight_init}_init"
    out_dir = RUN_ROOT / variant
    model_name = f"swr_anchor_motion_fusion_pv_nwp_{variant}"
    install_learnable_anchor_weight_factory(
        module,
        fixed_weights=STAGE3_FUTURE_ANCHOR_WEIGHTS,
        init_mode=wrapper.anchor_weight_init,
        seed=2026,
        random_std=float(wrapper.anchor_weight_random_std),
    )
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(out_dir),
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "h8_sample_cache_dir": "",
            "h8_clear_sky_dir": "",
            "swr_checkpoint": str(swr_checkpoint),
            "base_model_name": "pv_nwp",
            "model_name": model_name,
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    print("Stage 3 anchor-weight ablation:", variant)
    print("Output:", out_dir)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()
    export_trained_weights(out_dir / f"{model_name}_best.pt", out_dir, wrapper.anchor_weight_init)


if __name__ == "__main__":
    main()
