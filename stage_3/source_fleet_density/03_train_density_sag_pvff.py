#!/usr/bin/env python3
"""Train SAG-PVFF on one source-fleet density cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
for extra in (
    STAGE3_DIR,
    STAGE3_DIR / "ablation_motion_branch",
    STAGE3_DIR / "ablation_temporal_collapse",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import patch_last_only_collapse
from motion_branch import patch_motion_branch
from stage3_cli import inject_defaults
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    ensure_stage3_dirs,
)
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, cli_path, cli_path_or_none, resolve_active_cache_dir

from density_common import (
    DEFAULT_SEED,
    density_frame_cache_dir,
    density_selection_dir,
    density_stage3_run_dir,
    density_target_run_dir,
    parse_ratios,
    ratio_label,
    write_json,
)


MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate"


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratio", type=float, required=True, help="Source density ratio, e.g. 0.25, 0.50, 0.75.")
    parser.add_argument("--reuse_full_density", action="store_true", help="For r100, do not train; only print the expected full-density reference.")
    wrapper, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return wrapper


def install_best_sag_pvff_factory() -> None:
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        model = patch_motion_branch(model, "warp_direct_gate")
        return patch_last_only_collapse(model)

    module.make_swr_anchor_attention_model = make_model


def main() -> None:
    wrapper = parse_wrapper_args()
    ratio = float(wrapper.ratio)
    if ratio >= 1.0 and wrapper.reuse_full_density:
        print("r100 is configured as the existing full-density reference; no training launched.")
        return
    ensure_stage3_dirs()
    sel_dir = density_selection_dir(wrapper.seed, ratio)
    run_dir = density_target_run_dir(wrapper.seed, ratio)
    frame_cache = density_frame_cache_dir(wrapper.seed, ratio)
    for path in (sel_dir / "issue_time_index.csv", sel_dir / "target_excluded_station_split.csv", run_dir / "station_split.csv", frame_cache / "frame_cache_meta.json"):
        if not path.exists():
            raise FileNotFoundError(f"Missing density artifact before SAG-PVFF training: {path}")

    anchor_cache_dir = cli_path_or_none("--swr_anchor_cache_dir")
    if anchor_cache_dir is None:
        anchor_cache_dir = resolve_active_cache_dir(required=True)
    swr_checkpoint = cli_path("--swr_checkpoint", SWR_BASE_CHECKPOINT)
    assert_cache_matches_checkpoint(anchor_cache_dir, swr_checkpoint)
    if not (anchor_cache_dir / "swr_anchor_64.npy").exists():
        raise FileNotFoundError(f"Missing prebuilt Stage 3 SWR anchor cache: {anchor_cache_dir}")

    out_dir = density_stage3_run_dir(wrapper.seed, ratio)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "density_sag_pvff_meta.json",
        {
            "purpose": "SAG-PVFF retrained from scratch for one source-fleet density level.",
            "seed": int(wrapper.seed),
            "source_density_ratio": float(ratio),
            "source_density_label": ratio_label(ratio),
            "selection_dir": str(sel_dir),
            "target_run_dir": str(run_dir),
            "frame_cache_dir": str(frame_cache),
            "model_variant": "warp_direct_gate + last_only temporal collapse",
            "model_name": MODEL_NAME,
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "swr_checkpoint": str(swr_checkpoint),
        },
    )

    install_best_sag_pvff_factory()
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(sel_dir),
            "target_run_dir": str(run_dir),
            "frame_cache_dir": str(frame_cache),
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
            "model_name": MODEL_NAME,
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    print("==== density SAG-PVFF training ====")
    print("density:", ratio_label(ratio), "seed:", wrapper.seed)
    print("output:", out_dir)
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()
