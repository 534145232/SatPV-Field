#!/usr/bin/env python3
"""Rebuild target-excluded PV_norm field caches for density subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import TRAIN_PV_DIR, VAL_PV_DIR
from stage3_select_targets_pvnorm_impl import (
    PVNormPeakAdjustedFrameCache,
    PYCHARM_DEFAULTS,
    estimate_station_pvnorm_peak,
)
from train_fullfield_multisource_residual_unet import PYCHARM_DEFAULTS as MS_PYCHARM_DEFAULTS
from train_fullfield_multisource_residual_unet import _health_filter_station_series
from visualize_regional_split_frames import _bounds, load_station_series

from density_common import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    density_frame_cache_dir,
    density_selection_dir,
    density_target_run_dir,
    ensure_dir,
    parse_ratios,
    ratio_label,
    station_sns_sha1,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PV_norm frame caches for each source-fleet density level. "
            "This reuses the original station-wise q95 normalization and Gaussian splatting implementation."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--train_pv_dir", type=Path, default=TRAIN_PV_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=VAL_PV_DIR)
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--capacity_scale", type=float, default=float(PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--peak_quantile", type=float, default=float(PYCHARM_DEFAULTS["peak_quantile"]))
    parser.add_argument("--peak_min_samples", type=int, default=int(PYCHARM_DEFAULTS["peak_min_samples"]))
    parser.add_argument("--pv_field_clip", type=float, default=float(PYCHARM_DEFAULTS["pv_field_clip"]))
    parser.add_argument("--pv_health_max", type=float, default=float(MS_PYCHARM_DEFAULTS["pv_health_max"]))
    parser.add_argument(
        "--pv_health_active_threshold",
        type=float,
        default=float(MS_PYCHARM_DEFAULTS["pv_health_active_threshold"]),
    )
    parser.add_argument(
        "--pv_health_flatline_window",
        type=int,
        default=int(MS_PYCHARM_DEFAULTS["pv_health_flatline_window"]),
    )
    parser.add_argument(
        "--pv_health_flatline_range",
        type=float,
        default=float(MS_PYCHARM_DEFAULTS["pv_health_flatline_range"]),
    )
    parser.add_argument("--force_rebuild", type=int, default=0)
    parser.add_argument("--disable_health_filter", action="store_true")
    return parser.parse_args()


def _read_subset(seed: int, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sel_dir = density_selection_dir(seed, ratio)
    run_dir = density_target_run_dir(seed, ratio)
    source_path = sel_dir / "target_excluded_station_split.csv"
    station_path = run_dir / "station_split.csv"
    issue_path = sel_dir / "issue_time_index.csv"
    for path in (source_path, station_path, issue_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing density split artifact: {path}. Run 01_build_density_subsets.py first.")
    source = pd.read_csv(source_path, dtype={"station_sn": str})
    station_split = pd.read_csv(station_path, dtype={"station_sn": str})
    time_df = pd.read_csv(issue_path)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    return source, station_split, time_df


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        train_pv_dir=Path(args.train_pv_dir),
        val_pv_dir=Path(args.val_pv_dir),
        grid_size=int(args.grid_size),
        gaussian_sigma_cells=float(args.gaussian_sigma_cells),
        gaussian_radius_cells=int(args.gaussian_radius_cells),
        min_grid_weight=float(args.min_grid_weight),
        capacity_scale=float(args.capacity_scale),
        peak_quantile=float(args.peak_quantile),
        peak_min_samples=int(args.peak_min_samples),
        pv_field_clip=float(args.pv_field_clip),
        pv_health_max=float(args.pv_health_max),
        pv_health_active_threshold=float(args.pv_health_active_threshold),
        pv_health_flatline_window=int(args.pv_health_flatline_window),
        pv_health_flatline_range=float(args.pv_health_flatline_range),
        force_rebuild=int(args.force_rebuild),
        health_filter_frame_cache=not bool(args.disable_health_filter),
    )


def build_one(args: argparse.Namespace, ratio: float) -> dict:
    source, station_split, time_df = _read_subset(int(args.seed), ratio)
    cache_args = _cache_args(args)
    sel_dir = ensure_dir(density_selection_dir(args.seed, ratio))
    frame_cache_dir = ensure_dir(density_frame_cache_dir(args.seed, ratio))
    bounds = _bounds(station_split)
    series, load_status = load_station_series(source, cache_args)
    load_status.to_csv(sel_dir / "station_csv_load_status_for_density_cache.csv", index=False)
    if bool(cache_args.health_filter_frame_cache):
        series = _health_filter_station_series(series, cache_args, sel_dir)
    peak = estimate_station_pvnorm_peak(source, series, time_df, cache_args)
    pd.DataFrame(
        [{"station_sn": k, "pv_norm_q95": v} for k, v in sorted(peak.items())]
    ).to_csv(sel_dir / "target_excluded_station_pv_norm_q95.csv", index=False)
    cache = PVNormPeakAdjustedFrameCache(frame_cache_dir, source, series, peak, bounds, time_df, cache_args)
    cache.build_or_load()
    meta = {
        "seed": int(args.seed),
        "source_density_ratio": float(ratio),
        "source_density_label": ratio_label(ratio),
        "source_station_count": int(len(source)),
        "source_station_sns_sha1": station_sns_sha1(source),
        "frame_cache_dir": str(frame_cache_dir),
        "selection_dir": str(sel_dir),
        "target_run_dir": str(density_target_run_dir(args.seed, ratio)),
        "field_value": "PV_norm divided by station train-period PV_norm q95, clipped before Gaussian splatting",
        "grid_size": int(args.grid_size),
        "gaussian_sigma_cells": float(args.gaussian_sigma_cells),
        "gaussian_radius_cells": int(args.gaussian_radius_cells),
        "min_grid_weight": float(args.min_grid_weight),
        "peak_quantile": float(args.peak_quantile),
        "pv_field_clip": float(args.pv_field_clip),
    }
    write_json(sel_dir / "density_pvnorm_cache_meta.json", meta)
    return meta


def main() -> None:
    args = parse_args()
    rows = []
    for ratio in parse_ratios(args.ratios):
        if float(ratio) >= 1.0:
            print("[density_cache] skip r100 by default; full-density cache already exists in main Stage 3 data.")
            continue
        print(f"==== build density PV_norm cache {ratio_label(ratio)} ====")
        rows.append(build_one(args, ratio))
    if rows:
        out = pd.DataFrame(rows)
        root = ensure_dir(density_selection_dir(args.seed, rows[0]["source_density_ratio"]).parents[1])
        out.to_csv(root / "density_pvnorm_cache_summary.csv", index=False)
        print(out.to_string(index=False))
        print("saved:", root / "density_pvnorm_cache_summary.csv")


if __name__ == "__main__":
    main()
