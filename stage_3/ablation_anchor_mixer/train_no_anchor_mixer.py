#!/usr/bin/env python3
"""Train the selected Stage 3 architecture without the learned anchor mixer."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
for path in (
    STAGE3_DIR,
    STAGE3_DIR / "ablation_motion_branch",
    STAGE3_DIR / "ablation_temporal_collapse",
    STAGE3_DIR / "ablation_swr_dependency",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage3_motion_fusion_model_impl as module
from anchor_mixer_ablation import install_no_anchor_mixer_factory
from last_only_collapse import patch_last_only_collapse
from motion_branch import install_motion_branch_factory
from stage3_cli import inject_defaults
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
from stage3_swr_anchor_cache import (
    assert_cache_matches_checkpoint,
    cli_path,
    cli_path_or_none,
    resolve_active_cache_dir,
)
from swr_dependency import install_swr_dependency_dataset


RUN_DIR = STAGE3_DIR / "runs" / "ablation_anchor_mixer" / "no_anchor_mixer"
MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_no_anchor_mixer"


def install_last_only_factory(module) -> None:
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        return patch_last_only_collapse(original_factory(*args, **kwargs))

    module.make_swr_anchor_attention_model = make_model


def main() -> None:
    ensure_stage3_dirs()
    anchor_cache_dir = cli_path_or_none("--swr_anchor_cache_dir")
    if anchor_cache_dir is None:
        anchor_cache_dir = resolve_active_cache_dir(required=True)
    swr_checkpoint = cli_path("--swr_checkpoint", SWR_BASE_CHECKPOINT)
    assert_cache_matches_checkpoint(anchor_cache_dir, swr_checkpoint)
    if not (anchor_cache_dir / "swr_anchor_64.npy").exists():
        raise FileNotFoundError(f"Missing Stage 3 SWR anchor cache: {anchor_cache_dir}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "anchor_mixer_ablation_meta.json").write_text(
        json.dumps(
            {
                "variant": "no_anchor_mixer",
                "reference": "last_only + warp_direct_gate + full future SWR",
                "intervention": (
                    "Replace the learned 48-channel all-anchor context with zeros. "
                    "Keep fixed horizon-aligned SWR interpolation, past-SWR encoding, "
                    "PV-SWR alignment, NWP conditioning, and warp/direct/gate fusion unchanged."
                ),
                "swr_anchor_cache_dir": str(anchor_cache_dir),
                "swr_checkpoint": str(swr_checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    install_motion_branch_factory(module, "warp_direct_gate")
    install_last_only_factory(module)
    install_swr_dependency_dataset(module, "full_swr_future")
    install_no_anchor_mixer_factory(module)
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(RUN_DIR),
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "h8_sample_cache_dir": "",
            "h8_clear_sky_dir": "",
            "swr_checkpoint": str(swr_checkpoint),
            "base_model_name": "pv_nwp",
            "model_name": MODEL_NAME,
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    print("Stage 3 anchor-mixer ablation: no_anchor_mixer")
    print("Reference architecture: last_only + warp_direct_gate + full_swr_future")
    print("Output:", RUN_DIR)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()

