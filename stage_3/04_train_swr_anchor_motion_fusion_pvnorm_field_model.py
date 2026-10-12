#!/usr/bin/env python3
"""Train the unchanged 04l PV-field model with Stage 3 cache locations."""

import pandas as pd

from stage3_paths import (
    MOTION_FUSION_RUN_DIR,
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
from stage3_cli import inject_defaults
import stage3_motion_fusion_model_impl as module
from stage3_motion_fusion import install_stage3_motion_fusion_factory
from stage3_swr_anchor_cache import (
    assert_cache_matches_checkpoint,
    cli_path,
    cli_path_or_none,
    resolve_active_cache_dir,
)
from stage3_temporal import assert_swr_alignment


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def validate_anchor_cache(anchor_cache_dir) -> None:
    meta_path = anchor_cache_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 3 SWR anchor metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path)
    if meta.empty:
        raise RuntimeError(f"Empty Stage 3 SWR anchor metadata: {meta_path}")
    for _, row in meta.iterrows():
        assert_swr_alignment(
            pd.Timestamp(row["issue_time"]),
            split_times(row["history_label_times"]),
            split_times(row["future_anchor_label_times"]),
        )
    print("Validated SWR anchor cache before PV training: observed T-4h..T-1h, predicted T..T+2h")


def main() -> None:
    ensure_stage3_dirs()
    anchor_cache_dir = cli_path_or_none("--swr_anchor_cache_dir")
    if anchor_cache_dir is None:
        anchor_cache_dir = resolve_active_cache_dir(required=True)
    swr_checkpoint = cli_path("--swr_checkpoint", SWR_BASE_CHECKPOINT)
    assert_cache_matches_checkpoint(anchor_cache_dir, swr_checkpoint)
    anchor_path = anchor_cache_dir / "swr_anchor_64.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"Prebuilt Stage 3 SWR anchor cache not found: {anchor_path}. "
            "Run 03_build_center_h8_swr_anchor_cache.py first."
        )
    validate_anchor_cache(anchor_cache_dir)
    print("Using active Stage 3 future-SWR cache:", anchor_cache_dir)
    print("Matching Stage 2 SWR checkpoint:", swr_checkpoint)
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(MOTION_FUSION_RUN_DIR),
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "h8_sample_cache_dir": "",
            "h8_clear_sky_dir": "",
            "swr_checkpoint": str(swr_checkpoint),
            "base_model_name": "pv_nwp",
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    install_stage3_motion_fusion_factory(module)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()
