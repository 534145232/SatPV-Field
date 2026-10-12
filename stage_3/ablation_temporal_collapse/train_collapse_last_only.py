#!/usr/bin/env python3
"""Train the fixed-anchor Stage 3 motion-fusion model with last-only collapse."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import install_last_only_collapse_factory
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
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, cli_path, cli_path_or_none, resolve_active_cache_dir
from stage3_temporal import assert_swr_alignment


OUT_DIR = STAGE3_DIR / "runs" / "ablation_temporal_collapse" / "last_only"
MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_collapse_last_only"


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def validate_anchor_cache(anchor_cache_dir: Path) -> None:
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
    print("Validated SWR anchor cache: observed T-4h..T-1h, predicted T..T+2h")


def write_ablation_meta(out_dir: Path, anchor_cache_dir: Path, swr_checkpoint: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "temporal_collapse_ablation_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "variant": "last_only",
                "baseline_collapse": "0.7 * x[:, :, -1] + 0.3 * x.mean(dim=2)",
                "ablation_collapse": "x[:, :, -1]",
                "affected_encoders": ["PV encoder", "past-SWR temporal encoder"],
                "anchor_weights": "unchanged fixed physical 8x4 interpolation weights",
                "swr_anchor_cache_dir": str(anchor_cache_dir),
                "swr_checkpoint": str(swr_checkpoint),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


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
            "Run stage_3/run_stage3_build_future_swr_cache.sh first."
        )
    validate_anchor_cache(anchor_cache_dir)
    write_ablation_meta(OUT_DIR, anchor_cache_dir, swr_checkpoint)
    install_last_only_collapse_factory(module)
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(OUT_DIR),
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
    print("Stage 3 temporal-collapse ablation: last_only")
    print("Output:", OUT_DIR)
    print("Using active Stage 3 future-SWR cache:", anchor_cache_dir)
    print("Matching Stage 2 SWR checkpoint:", swr_checkpoint)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()
