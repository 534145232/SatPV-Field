#!/usr/bin/env python3
"""Evaluate the fixed-anchor Stage 3 last-only temporal-collapse ablation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import install_last_only_collapse_factory
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir


f04 = module.f04
USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True
RUN_DIR = STAGE3_DIR / "runs" / "ablation_temporal_collapse" / "last_only"
MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_collapse_last_only"

PYCHARM_DEFAULTS = {
    "run_dir": str(RUN_DIR),
    "selection_dir": str(TARGET_SELECTION_DIR),
    "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
    "swr_anchor_cache_dir": "",
    "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
    "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
    "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
    "h8_netcdf_glob": "*.nc",
    "h8_sample_cache_dir": "",
    "h8_clear_sky_dir": "",
    "swr_checkpoint": str(SWR_BASE_CHECKPOINT),
    "checkpoint": "",
    "out_dir": "",
    "model_name": MODEL_NAME,
    "base_model_name": "pv_nwp",
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1.0e-6,
    "freq_minutes": 15,
    "past_frames": 12,
    "future_frames": 8,
    "swr_history_frames": 4,
    "swr_future_anchors": 3,
    "normalize_swr_anchor": 1,
    "pv_hidden_channels": 48,
    "anchor_channels": 48,
    "horizon_embed_channels": 8,
    "dropout": 0.05,
    "large_delta_threshold": 0.10,
    "max_flow_pixels": 3.0,
    "batch_size": 8,
    "num_workers": 0,
    "device": "cuda",
    "sample_count": 8,
    "visual_horizons": "1,4,8",
    "dpi": 150,
    "seed": 2026,
}


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 3 fixed-anchor motion fusion with last-only temporal collapse."
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if key in {"checkpoint", "out_dir", "h8_sample_cache_dir", "h8_clear_sky_dir"}:
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key.endswith("_root") or key == "swr_checkpoint":
            parser.add_argument(f"--{key}", type=Path, default=Path(value) if value else Path(""))
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_validation_state(args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader

    f04._assert_target_exclusion(args)
    station_split = f04._load_station_split(args)
    bounds = module._bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(args.multisource_cache_dir))
    f04._assert_issue_time_alignment(args, ms_cache["time_df"])
    time_df = pd.read_csv(Path(args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(f04._timestamp_key)

    run_dir = Path(args.run_dir)
    pv_cache = f04.ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, time_df, run_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, run_dir, time_df)
    swr_arr_all, swr_meta_all, _swr_info = module._load_prebuilt_swr_anchor_cache(Path(args.swr_anchor_cache_dir))

    base_datasets, swr_arrays, datasets, loaders = {}, {}, {}, {}
    for split in ["train", "val", "test"]:
        sub, cache_indices, _meta_df = module._subset_time_df_by_swr_meta(time_df, swr_meta_all, split)
        base = f04.TargetExcludedPVNormFieldDataset(
            sub,
            pv_cache,
            ms_cache,
            bounds,
            pv_stats,
            ms_stats,
            str(args.base_model_name),
            args,
        )
        base_datasets[split] = base
        swr_arrays[split] = np.asarray(swr_arr_all[cache_indices], dtype=np.float32)

    swr_anchor_stats = (
        module._compute_swr_anchor_stats(swr_arrays["train"], run_dir, force=False)
        if int(args.normalize_swr_anchor)
        else None
    )
    for split in ["train", "val", "test"]:
        datasets[split] = module.SWRAnchorPVFieldDataset(
            base_datasets[split],
            swr_arrays[split],
            args,
            swr_anchor_stats,
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )

    base_in, future_channels = base_datasets["train"].source_dims()
    model = module.make_swr_anchor_attention_model(
        future_frames=int(args.future_frames),
        in_channels=int(base_in) + 2,
        future_channels=int(future_channels) + 1,
        anchor_channels=2,
        pv_hidden=int(args.pv_hidden_channels),
        anchor_hidden=int(args.anchor_channels),
        horizon_embed_channels=int(args.horizon_embed_channels),
        dropout=float(args.dropout),
        swr_history_frames=int(args.swr_history_frames),
        swr_future_anchors=int(args.swr_future_anchors),
        max_flow_pixels=float(args.max_flow_pixels),
    )
    device = module._device(args.device)
    checkpoint = (
        Path(args.checkpoint)
        if str(args.checkpoint).strip()
        else run_dir / f"{args.model_name}_best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, device, datasets, loaders, pv_stats, checkpoint


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if not str(args.swr_anchor_cache_dir).strip() or str(args.swr_anchor_cache_dir) == ".":
        args.swr_anchor_cache_dir = resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(Path(args.swr_anchor_cache_dir), Path(args.swr_checkpoint))
    install_last_only_collapse_factory(module)
    model, device, datasets, loaders, pv_stats, checkpoint = build_validation_state(args)

    out_dir = ensure_dir(
        Path(args.out_dir)
        if str(args.out_dir).strip()
        else Path(args.run_dir) / "05_validation_visuals"
    )
    metric_frames = []
    for split in ["val", "test"]:
        metric_frames.append(
            module.evaluate_model(
                model,
                loaders[split],
                device,
                str(args.model_name),
                split,
                pv_stats,
                float(args.large_delta_threshold),
            )
        )
    summary = pd.concat(metric_frames, ignore_index=True)
    summary_path = out_dir / "swr_anchor_guided_validation_eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    sample_dir = ensure_dir(out_dir / "test_sample_comparisons")
    module.save_sample_comparisons(model, datasets["test"], device, sample_dir, args, pv_stats)
    with (out_dir / "temporal_collapse_validation_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "variant": "last_only",
                "baseline_collapse": "0.7 * x[:, :, -1] + 0.3 * x.mean(dim=2)",
                "ablation_collapse": "x[:, :, -1]",
                "affected_encoders": ["PV encoder", "past-SWR temporal encoder"],
                "checkpoint": str(checkpoint),
                "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
                "metric_summary": str(summary_path),
                "sample_figures": str(sample_dir),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("==== Stage 3 temporal-collapse ablation: last_only ====")
    show = summary[summary["horizon"].isin(["all", "h1", "h4", "h8", "horizon_diversity"])].copy()
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    print("checkpoint:", checkpoint)
    print("summary:", summary_path)
    print("sample figures:", sample_dir)


if __name__ == "__main__":
    main()
