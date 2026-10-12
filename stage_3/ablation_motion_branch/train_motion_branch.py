#!/usr/bin/env python3
"""Train one fixed-anchor Stage 3 motion-branch ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))
TEMPORAL_COLLAPSE_DIR = STAGE3_DIR / "ablation_temporal_collapse"
if str(TEMPORAL_COLLAPSE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_COLLAPSE_DIR))
SWR_DEPENDENCY_DIR = STAGE3_DIR / "ablation_swr_dependency"
if str(SWR_DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(SWR_DEPENDENCY_DIR))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import patch_last_only_collapse
from motion_branch import MOTION_BRANCHES, install_motion_branch_factory
from swr_dependency import SWR_DEPENDENCY_MODES, install_swr_dependency_dataset
from stage3_cli import inject_defaults
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    RUNS_DIR,
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


RUN_ROOT = RUNS_DIR / "ablation_motion_branch"
TEMPORAL_COLLAPSES = ("last_mean", "last_only")


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion_branch", choices=MOTION_BRANCHES, required=True)
    parser.add_argument("--temporal_collapse", choices=TEMPORAL_COLLAPSES, default="last_mean")
    parser.add_argument("--swr_dependency", choices=SWR_DEPENDENCY_MODES, default="full_swr_future")
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def install_temporal_collapse_factory(module, temporal_collapse: str) -> None:
    if temporal_collapse == "last_mean":
        return
    if temporal_collapse != "last_only":
        raise ValueError(f"Unknown temporal collapse: {temporal_collapse}")
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        return patch_last_only_collapse(original_factory(*args, **kwargs))

    module.make_swr_anchor_attention_model = make_model


def main() -> None:
    wrapper = parse_wrapper_args()
    ensure_stage3_dirs()
    anchor_cache_dir = cli_path_or_none("--swr_anchor_cache_dir")
    if anchor_cache_dir is None:
        anchor_cache_dir = resolve_active_cache_dir(required=True)
    swr_checkpoint = cli_path("--swr_checkpoint", SWR_BASE_CHECKPOINT)
    assert_cache_matches_checkpoint(anchor_cache_dir, swr_checkpoint)
    if not (anchor_cache_dir / "swr_anchor_64.npy").exists():
        raise FileNotFoundError(f"Missing Stage 3 SWR anchor cache: {anchor_cache_dir}")
    out_dir = Path(wrapper.run_root) / wrapper.motion_branch
    model_name = f"swr_anchor_motion_fusion_pv_nwp_motion_{wrapper.motion_branch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "motion_branch_ablation_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "motion_branch": wrapper.motion_branch,
                "temporal_collapse": wrapper.temporal_collapse,
                "swr_dependency": wrapper.swr_dependency,
                "variants": list(MOTION_BRANCHES),
                "anchor_weights": "unchanged fixed physical 8x4 interpolation weights",
                "swr_anchor_cache_dir": str(anchor_cache_dir),
                "swr_checkpoint": str(swr_checkpoint),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    install_motion_branch_factory(module, wrapper.motion_branch)
    install_temporal_collapse_factory(module, wrapper.temporal_collapse)
    install_swr_dependency_dataset(module, wrapper.swr_dependency)
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
    print("Stage 3 motion-branch ablation:", wrapper.motion_branch)
    print("Temporal collapse:", wrapper.temporal_collapse)
    print("SWR dependency:", wrapper.swr_dependency)
    print("Output:", out_dir)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()
