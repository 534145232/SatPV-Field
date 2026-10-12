#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPT_DIR))

from build_fullfield_multisource_cache import build_multisource_issue_time_table
from train_fullfield_dense_temporal_forecast import _load_split_df, _parse_time_list
from train_fullfield_dense_temporal_residual_unet import _timestamp_key, _unique_timestamps
from train_fullfield_multisource_residual_unet import _health_filter_station_series
from train_regional_dense_reconstruction import splat_values
from visualize_regional_split_frames import _bounds, ensure_dir, load_station_series, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": "/code/Final_3_stage_pipeline/stage_3/data/00_source_regional_split",
    "out_dir": "/code/Final_3_stage_pipeline/stage_3/data/02_target_selection_lhs50",
    "target_run_dir": "/code/Final_3_stage_pipeline/stage_3/data/03_target_excluded_run",
    "frame_cache_dir": "/code/Final_3_stage_pipeline/stage_3/data/04_target_excluded_pvnorm_frame_cache",
    "train_pv_dir": "/data/DayinChen/NWP/PV/train_directory_cleaned_csv_same_day06_nwp",
    "val_pv_dir": "/data/DayinChen/NWP/PV/val_directory_cleaned_csv_same_day06_nwp",
    "time_start": "2022-01-01 07:00:00",
    "time_end": "2023-06-24 16:45:00",
    "train_end": "2023-01-13 23:59:59",
    "val_end": "2023-03-08 23:59:59",
    "freq_minutes": 15,
    "issue_stride_minutes": 60,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "max_targets": 50,
    "min_station_rows": 6000,
    "min_train_issue_samples": 1200,
    "min_val_issue_samples": 180,
    "min_test_issue_samples": 360,
    "density_radius_km": 8.0,
    "min_local_neighbors": 8,
    "min_target_separation_km": 2.0,
    "selection_method": "lhs",
    "lhs_candidate_multiplier": 20,
    "lhs_support_radius_km": 5.0,
    "lhs_min_support_neighbors": 5,
    "selection_grid_size": 10,
    "centrality_weight": 0.6,
    "density_weight": 1.0,
    "sample_count_weight": 0.35,
    "force_northwest_edge_target": True,
    "northwest_lon_quantile": 0.25,
    "northwest_lat_quantile": 0.75,
    "northwest_replace_rank": 3,
    "northwest_edge_weight": 3.0,
    "seed": 2026,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "capacity_scale": 1000.0,
    "peak_quantile": 0.95,
    "peak_min_samples": 120,
    "pv_field_clip": 1.5,
    "health_filter_frame_cache": True,
    "pv_health_max": 1.35,
    "pv_health_active_threshold": 0.05,
    "pv_health_flatline_window": 8,
    "pv_health_flatline_range": 1e-5,
    "force_rebuild": 0,
    "select_only": False,
    "dpi": 170,
}


def _station_sns_sha1(split_df: pd.DataFrame) -> str:
    sns = sorted(str(v) for v in split_df["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(sns).encode("utf-8")).hexdigest()


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select data-sufficient target stations from dense clusters and build a target-excluded "
            "DensePV field cache whose value is capacity-normalized PV corrected by each station's "
            "train-period PV_norm 0.95 quantile. This cache deliberately does not use clear-sky/CSI."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--target_run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["target_run_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["frame_cache_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--time_start", type=str, default=str(PYCHARM_DEFAULTS["time_start"]))
    parser.add_argument("--time_end", type=str, default=str(PYCHARM_DEFAULTS["time_end"]))
    parser.add_argument("--train_end", type=str, default=str(PYCHARM_DEFAULTS["train_end"]))
    parser.add_argument("--val_end", type=str, default=str(PYCHARM_DEFAULTS["val_end"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--issue_stride_minutes", type=int, default=int(PYCHARM_DEFAULTS["issue_stride_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(PYCHARM_DEFAULTS["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(PYCHARM_DEFAULTS["future_frames"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--max_targets", type=int, default=int(PYCHARM_DEFAULTS["max_targets"]))
    parser.add_argument("--min_station_rows", type=int, default=int(PYCHARM_DEFAULTS["min_station_rows"]))
    parser.add_argument("--min_train_issue_samples", type=int, default=int(PYCHARM_DEFAULTS["min_train_issue_samples"]))
    parser.add_argument("--min_val_issue_samples", type=int, default=int(PYCHARM_DEFAULTS["min_val_issue_samples"]))
    parser.add_argument("--min_test_issue_samples", type=int, default=int(PYCHARM_DEFAULTS["min_test_issue_samples"]))
    parser.add_argument("--density_radius_km", type=float, default=float(PYCHARM_DEFAULTS["density_radius_km"]))
    parser.add_argument("--min_local_neighbors", type=int, default=int(PYCHARM_DEFAULTS["min_local_neighbors"]))
    parser.add_argument("--min_target_separation_km", type=float, default=float(PYCHARM_DEFAULTS["min_target_separation_km"]))
    parser.add_argument(
        "--selection_method",
        type=str,
        default=str(PYCHARM_DEFAULTS["selection_method"]),
        choices=["lhs", "score_grid"],
        help="Target selection method. lhs gives spatially stratified Latin-hypercube targets; score_grid keeps the older density/score grid selection.",
    )
    parser.add_argument("--lhs_candidate_multiplier", type=int, default=int(PYCHARM_DEFAULTS["lhs_candidate_multiplier"]))
    parser.add_argument("--lhs_support_radius_km", type=float, default=float(PYCHARM_DEFAULTS["lhs_support_radius_km"]))
    parser.add_argument("--lhs_min_support_neighbors", type=int, default=int(PYCHARM_DEFAULTS["lhs_min_support_neighbors"]))
    parser.add_argument("--selection_grid_size", type=int, default=int(PYCHARM_DEFAULTS["selection_grid_size"]))
    parser.add_argument("--centrality_weight", type=float, default=float(PYCHARM_DEFAULTS["centrality_weight"]))
    parser.add_argument("--density_weight", type=float, default=float(PYCHARM_DEFAULTS["density_weight"]))
    parser.add_argument("--sample_count_weight", type=float, default=float(PYCHARM_DEFAULTS["sample_count_weight"]))
    parser.add_argument(
        "--force_northwest_edge_target",
        action="store_true",
        default=bool(PYCHARM_DEFAULTS["force_northwest_edge_target"]),
        help="Force one selected target into the upper-left / northwest edge dense cluster if available.",
    )
    parser.add_argument("--disable_force_northwest_edge_target", action="store_false", dest="force_northwest_edge_target")
    parser.add_argument("--northwest_lon_quantile", type=float, default=float(PYCHARM_DEFAULTS["northwest_lon_quantile"]))
    parser.add_argument("--northwest_lat_quantile", type=float, default=float(PYCHARM_DEFAULTS["northwest_lat_quantile"]))
    parser.add_argument("--northwest_replace_rank", type=int, default=int(PYCHARM_DEFAULTS["northwest_replace_rank"]))
    parser.add_argument("--northwest_edge_weight", type=float, default=float(PYCHARM_DEFAULTS["northwest_edge_weight"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--capacity_scale", type=float, default=float(PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--peak_quantile", type=float, default=float(PYCHARM_DEFAULTS["peak_quantile"]))
    parser.add_argument("--peak_min_samples", type=int, default=int(PYCHARM_DEFAULTS["peak_min_samples"]))
    parser.add_argument("--pv_field_clip", type=float, default=float(PYCHARM_DEFAULTS["pv_field_clip"]))
    parser.add_argument("--health_filter_frame_cache", action="store_true", default=bool(PYCHARM_DEFAULTS["health_filter_frame_cache"]))
    parser.add_argument("--disable_health_filter_frame_cache", action="store_false", dest="health_filter_frame_cache")
    parser.add_argument("--pv_health_max", type=float, default=float(PYCHARM_DEFAULTS["pv_health_max"]))
    parser.add_argument("--pv_health_active_threshold", type=float, default=float(PYCHARM_DEFAULTS["pv_health_active_threshold"]))
    parser.add_argument("--pv_health_flatline_window", type=int, default=int(PYCHARM_DEFAULTS["pv_health_flatline_window"]))
    parser.add_argument("--pv_health_flatline_range", type=float, default=float(PYCHARM_DEFAULTS["pv_health_flatline_range"]))
    parser.add_argument("--force_rebuild", type=int, default=int(PYCHARM_DEFAULTS["force_rebuild"]))
    parser.add_argument(
        "--select_only",
        action="store_true",
        default=bool(PYCHARM_DEFAULTS["select_only"]),
        # default=True,
        help="Only select/visualize target stations and write selection CSVs; do not build the PV field cache.",
    )
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371.0088
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return r * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _cell_ids(df: pd.DataFrame, bounds: tuple[float, float, float, float], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bounds
    x = np.floor((df["lon"].to_numpy(np.float64) - lon_min) / max(1e-12, lon_max - lon_min) * grid_size).astype(int)
    y = np.floor((df["lat"].to_numpy(np.float64) - lat_min) / max(1e-12, lat_max - lat_min) * grid_size).astype(int)
    return np.clip(y, 0, grid_size - 1), np.clip(x, 0, grid_size - 1)


def _issue_sample_counts(series_df: pd.DataFrame, time_df: pd.DataFrame) -> dict[str, int]:
    if series_df is None or series_df.empty:
        return {"train": 0, "val": 0, "test": 0}
    idx = set(int(pd.Timestamp(t).timestamp()) for t in series_df.index)
    out = {"train": 0, "val": 0, "test": 0}
    for _, row in time_df.iterrows():
        secs = []
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            secs.append(int(pd.Timestamp(ts).timestamp()))
        if all(sec in idx for sec in secs):
            out[str(row["time_split"])] += 1
    return out


def _station_diagnostics(split_df: pd.DataFrame, series: dict[str, pd.DataFrame], time_df: pd.DataFrame, bounds, args) -> pd.DataFrame:
    work = split_df.copy().reset_index(drop=True)
    rows = []
    coords = work[["lon", "lat"]].to_numpy(np.float64)
    dist = _haversine_km(coords[:, None, 0], coords[:, None, 1], coords[None, :, 0], coords[None, :, 1])
    local = ((dist <= float(args.density_radius_km)) & (dist > 0.0)).sum(axis=1)
    center_lon = float(np.nanmedian(work["lon"].to_numpy(np.float64)))
    center_lat = float(np.nanmedian(work["lat"].to_numpy(np.float64)))
    center_dist = _haversine_km(work["lon"].to_numpy(np.float64), work["lat"].to_numpy(np.float64), center_lon, center_lat)
    yy, xx = _cell_ids(work, bounds, int(args.selection_grid_size))
    for i, row in work.iterrows():
        sn = str(row["station_sn"])
        sdf = series.get(sn, pd.DataFrame())
        counts = _issue_sample_counts(sdf, time_df)
        rows.append(
            {
                "station_sn": sn,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "source_split": str(row.get("split", "")),
                "n_rows": int(len(sdf)),
                "date_min": str(sdf.index.min()) if len(sdf) else "",
                "date_max": str(sdf.index.max()) if len(sdf) else "",
                "train_issue_samples": int(counts["train"]),
                "val_issue_samples": int(counts["val"]),
                "test_issue_samples": int(counts["test"]),
                "local_neighbors": int(local[i]),
                "center_distance_km": float(center_dist[i]),
                "selection_cell_y": int(yy[i]),
                "selection_cell_x": int(xx[i]),
                "selection_cell": f"{int(yy[i])}_{int(xx[i])}",
            }
        )
    diag = pd.DataFrame(rows)
    cell_counts = diag.groupby("selection_cell")["station_sn"].transform("count")
    diag["selection_cell_station_count"] = cell_counts.astype(int)
    return diag


def _make_lhs_anchors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(n)
    anchors = np.zeros((n, 2), dtype=np.float64)
    for dim in range(2):
        anchors[:, dim] = (np.arange(n, dtype=np.float64) + rng.random(n)) / max(n, 1)
        rng.shuffle(anchors[:, dim])
    return anchors


def _support_count_after_excluding_targets(
    row: pd.Series,
    pool: pd.DataFrame,
    selected_sns: set[str],
    args: argparse.Namespace,
) -> int:
    support = pool[~pool["station_sn"].astype(str).isin(selected_sns | {str(row["station_sn"])})]
    if support.empty:
        return 0
    dist = _haversine_km(
        float(row["lon"]),
        float(row["lat"]),
        support["lon"].to_numpy(np.float64),
        support["lat"].to_numpy(np.float64),
    )
    return int(np.sum(dist <= float(args.lhs_support_radius_km)))


def _select_targets_lhs(pool: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    n_targets = int(args.max_targets)
    if n_targets <= 0:
        return pool.head(0).copy()
    lon_min, lon_max = float(pool["lon"].min()), float(pool["lon"].max())
    lat_min, lat_max = float(pool["lat"].min()), float(pool["lat"].max())
    n_anchors = max(n_targets, n_targets * max(1, int(args.lhs_candidate_multiplier)))
    anchors01 = _make_lhs_anchors(n_anchors, int(args.seed))
    anchors = np.column_stack(
        [
            lon_min + anchors01[:, 0] * max(1e-12, lon_max - lon_min),
            lat_min + anchors01[:, 1] * max(1e-12, lat_max - lat_min),
        ]
    )
    coords = pool[["lon", "lat"]].to_numpy(np.float64)
    selected_rows = []
    selected_sns: set[str] = set()
    selected_cells: set[str] = set()
    selected_anchor_rows = []

    for anchor_rank, (anchor_lon, anchor_lat) in enumerate(anchors, start=1):
        dist_anchor = _haversine_km(anchor_lon, anchor_lat, coords[:, 0], coords[:, 1])
        order = np.argsort(dist_anchor)
        for idx in order:
            row = pool.iloc[int(idx)]
            sn = str(row["station_sn"])
            if sn in selected_sns:
                continue
            if selected_rows:
                target_dist = [
                    float(_haversine_km(row["lon"], row["lat"], prev["lon"], prev["lat"]))
                    for prev in selected_rows
                ]
                if min(target_dist) < float(args.min_target_separation_km):
                    continue
            support_count = _support_count_after_excluding_targets(row, pool, selected_sns, args)
            if support_count < int(args.lhs_min_support_neighbors):
                continue
            out_row = row.copy()
            out_row["lhs_anchor_rank"] = int(anchor_rank)
            out_row["lhs_anchor_lon"] = float(anchor_lon)
            out_row["lhs_anchor_lat"] = float(anchor_lat)
            out_row["lhs_anchor_distance_km"] = float(dist_anchor[int(idx)])
            out_row["support_neighbors_after_target_exclusion"] = int(support_count)
            selected_rows.append(out_row)
            selected_sns.add(sn)
            selected_cells.add(str(row.get("selection_cell", "")))
            selected_anchor_rows.append(anchor_rank)
            break
        if len(selected_rows) >= n_targets:
            break

    if len(selected_rows) < n_targets:
        for _, row in pool.sort_values(["selection_score", "local_neighbors", "train_issue_samples"], ascending=False).iterrows():
            if len(selected_rows) >= n_targets:
                break
            sn = str(row["station_sn"])
            if sn in selected_sns:
                continue
            if selected_rows:
                target_dist = [
                    float(_haversine_km(row["lon"], row["lat"], prev["lon"], prev["lat"]))
                    for prev in selected_rows
                ]
                if min(target_dist) < float(args.min_target_separation_km):
                    continue
            support_count = _support_count_after_excluding_targets(row, pool, selected_sns, args)
            if support_count < int(args.lhs_min_support_neighbors):
                continue
            out_row = row.copy()
            out_row["lhs_anchor_rank"] = -1
            out_row["lhs_anchor_lon"] = np.nan
            out_row["lhs_anchor_lat"] = np.nan
            out_row["lhs_anchor_distance_km"] = np.nan
            out_row["support_neighbors_after_target_exclusion"] = int(support_count)
            selected_rows.append(out_row)
            selected_sns.add(sn)

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("LHS target selection produced no targets. Lower min_target_separation_km or lhs_min_support_neighbors.")
    out["target_rank"] = np.arange(1, len(out) + 1)
    out["selection_method"] = "lhs"
    return out


def _select_targets_score_grid(pool: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_per_cell = (
        pool.sort_values(["selection_cell_station_count", "selection_score", "train_issue_samples"], ascending=False)
        .groupby("selection_cell", as_index=False)
        .head(2)
        .sort_values("selection_score", ascending=False)
        .reset_index(drop=True)
    )
    selected = []
    for _, row in top_per_cell.iterrows():
        if len(selected) >= int(args.max_targets):
            break
        if not selected:
            selected.append(row)
            continue
        d = [
            float(_haversine_km(row["lon"], row["lat"], prev["lon"], prev["lat"]))
            for prev in selected
        ]
        if min(d) >= float(args.min_target_separation_km):
            selected.append(row)
    if len(selected) < min(int(args.max_targets), len(top_per_cell)):
        for _, row in top_per_cell.iterrows():
            if len(selected) >= int(args.max_targets):
                break
            if str(row["station_sn"]) not in {str(x["station_sn"]) for x in selected}:
                selected.append(row)

    if bool(args.force_northwest_edge_target) and selected:
        lon_cut = float(np.nanquantile(pool["lon"].to_numpy(np.float64), float(args.northwest_lon_quantile)))
        lat_cut = float(np.nanquantile(pool["lat"].to_numpy(np.float64), float(args.northwest_lat_quantile)))
        selected_sns = {str(x["station_sn"]) for x in selected}
        nw_pool = pool[(pool["lon"] <= lon_cut) & (pool["lat"] >= lat_cut)].copy()
        if not nw_pool.empty:
            lon = nw_pool["lon"].to_numpy(np.float64)
            lat = nw_pool["lat"].to_numpy(np.float64)
            westness = (float(pool["lon"].max()) - lon) / max(1e-12, float(pool["lon"].max() - pool["lon"].min()))
            northness = (lat - float(pool["lat"].min())) / max(1e-12, float(pool["lat"].max() - pool["lat"].min()))
            nw_pool["northwest_edge_score"] = (
                nw_pool["selection_score"].to_numpy(np.float64)
                + float(args.northwest_edge_weight) * (westness + northness)
            )
            nw_pool = nw_pool.sort_values(
                ["northwest_edge_score", "local_neighbors", "train_issue_samples"],
                ascending=False,
            ).reset_index(drop=True)
            nw_row = None
            replace_idx = int(np.clip(int(args.northwest_replace_rank) - 1, 0, len(selected) - 1))
            replace_sn = str(selected[replace_idx]["station_sn"])
            for _, cand in nw_pool.iterrows():
                cand_sn = str(cand["station_sn"])
                if cand_sn == replace_sn or cand_sn not in selected_sns:
                    nw_row = cand
                    break
            if nw_row is not None:
                selected[replace_idx] = nw_row

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["target_rank"] = np.arange(1, len(out) + 1)
    out["selection_method"] = "score_grid"
    return out


def _add_final_support_counts(targets: pd.DataFrame, pool: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = targets.copy().reset_index(drop=True)
    target_sns = set(out["station_sn"].astype(str).tolist())
    counts = []
    for _, row in out.iterrows():
        support = pool[~pool["station_sn"].astype(str).isin(target_sns)]
        if support.empty:
            counts.append(0)
            continue
        dist = _haversine_km(
            float(row["lon"]),
            float(row["lat"]),
            support["lon"].to_numpy(np.float64),
            support["lat"].to_numpy(np.float64),
        )
        counts.append(int(np.sum(dist <= float(args.lhs_support_radius_km))))
    out["support_neighbors_after_all_targets_excluded"] = counts
    return out


def select_targets(diag: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pool = diag[
        (diag["n_rows"] >= int(args.min_station_rows))
        & (diag["train_issue_samples"] >= int(args.min_train_issue_samples))
        & (diag["val_issue_samples"] >= int(args.min_val_issue_samples))
        & (diag["test_issue_samples"] >= int(args.min_test_issue_samples))
        & (diag["local_neighbors"] >= int(args.min_local_neighbors))
    ].copy()
    if pool.empty:
        raise RuntimeError(
            "No target candidates passed data/density filters. Lower min_* thresholds or inspect target_candidate_scores.csv."
        )
    local_den = pool["local_neighbors"].to_numpy(np.float64)
    sample = pool["train_issue_samples"].to_numpy(np.float64)
    center = pool["center_distance_km"].to_numpy(np.float64)
    pool["selection_score"] = (
        float(args.density_weight) * (local_den / max(1.0, float(np.nanmax(local_den))))
        + float(args.sample_count_weight) * (sample / max(1.0, float(np.nanmax(sample))))
        - float(args.centrality_weight) * (center / max(1.0, float(np.nanmax(center))))
    )
    if str(args.selection_method).lower() == "lhs":
        return _add_final_support_counts(_select_targets_lhs(pool, args), pool, args)
    return _add_final_support_counts(_select_targets_score_grid(pool, args), pool, args)


def estimate_station_pvnorm_peak(
    split_df: pd.DataFrame,
    series: dict[str, pd.DataFrame],
    time_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, float]:
    train_secs = set()
    train_df = time_df[time_df["time_split"] == "train"]
    for _, row in train_df.iterrows():
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            train_secs.add(int(pd.Timestamp(ts).timestamp()))
    peaks: dict[str, float] = {}
    all_vals = []
    q = float(args.peak_quantile)
    for sn in split_df["station_sn"].astype(str):
        df = series.get(sn)
        if df is None or df.empty:
            continue
        sec = np.asarray([int(pd.Timestamp(t).timestamp()) for t in df.index], dtype=np.int64)
        pv = pd.to_numeric(df["pv_norm"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(pv) & (pv >= 0.0)
        train_valid = valid & np.isin(sec, np.fromiter(train_secs, dtype=np.int64))
        vals = pv[train_valid]
        source = "train"
        if vals.size < int(args.peak_min_samples):
            vals = pv[valid]
            source = "all_fallback"
        vals = vals[np.isfinite(vals) & (vals >= 0.0)]
        if vals.size:
            all_vals.append(vals)
        if vals.size >= max(10, int(args.peak_min_samples) // 4):
            peak = float(np.nanquantile(vals, q))
            peaks[sn] = float(np.clip(peak, 0.05, float(args.pv_field_clip)))
        else:
            peaks[sn] = np.nan
    fallback = 1.0
    if all_vals:
        fallback = float(np.nanquantile(np.concatenate(all_vals), q))
        fallback = float(np.clip(fallback, 0.05, float(args.pv_field_clip)))
    for sn, peak in list(peaks.items()):
        if not np.isfinite(peak):
            peaks[sn] = fallback
    return peaks


def collect_pvnorm_peak_adjusted_values(
    split_df: pd.DataFrame,
    ts: pd.Timestamp,
    series: dict[str, pd.DataFrame],
    peak: dict[str, float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for _, row in split_df.iterrows():
        sn = str(row["station_sn"])
        df = series.get(sn)
        if df is None or ts not in df.index:
            continue
        pv_norm = float(df.loc[ts, "pv_norm"])
        if not np.isfinite(pv_norm) or pv_norm < 0.0:
            continue
        denom = max(float(peak.get(sn, 1.0)), 1e-4)
        value = float(np.clip(pv_norm / denom, 0.0, float(args.pv_field_clip)))
        rows.append({"station_sn": sn, "lon": float(row["lon"]), "lat": float(row["lat"]), "value": value})
    return pd.DataFrame(rows)


class PVNormPeakAdjustedFrameCache:
    def __init__(
        self,
        cache_dir: Path,
        split_df: pd.DataFrame,
        series: dict[str, pd.DataFrame],
        peak: dict[str, float],
        bounds: tuple[float, float, float, float],
        time_df: pd.DataFrame,
        args: argparse.Namespace,
    ):
        self.cache_dir = ensure_dir(cache_dir)
        self.split_df = split_df
        self.series = series
        self.peak = peak
        self.bounds = bounds
        self.args = args
        self.grid_size = int(args.grid_size)
        self.times = _unique_timestamps(time_df)
        self.sec = np.asarray([_timestamp_key(t) for t in self.times], dtype=np.int64)
        self.field_path = self.cache_dir / "fields_float16.npy"
        self.mask_path = self.cache_dir / "masks_uint8.npy"
        self.sec_path = self.cache_dir / "timestamp_sec.npy"
        self.meta_path = self.cache_dir / "frame_cache_meta.json"

    def ready(self) -> bool:
        if int(getattr(self.args, "force_rebuild", 0)):
            return False
        if not (self.field_path.exists() and self.mask_path.exists() and self.sec_path.exists() and self.meta_path.exists()):
            return False
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        expected = {
            "grid_size": int(self.grid_size),
            "gaussian_sigma_cells": float(self.args.gaussian_sigma_cells),
            "gaussian_radius_cells": int(self.args.gaussian_radius_cells),
            "min_grid_weight": float(self.args.min_grid_weight),
            "value_type": "pv_norm_div_station_pv_norm_q95",
            "pv_field_clip": float(self.args.pv_field_clip),
            "source_station_count": int(len(self.split_df)),
            "source_station_sns_sha1": _station_sns_sha1(self.split_df),
        }
        for key, value in expected.items():
            if key not in meta:
                return False
            if isinstance(value, float):
                if abs(float(meta[key]) - value) > 1e-9:
                    return False
            elif str(meta[key]) != str(value):
                return False
        old_sec = np.load(self.sec_path, mmap_mode="r")
        return old_sec.shape == self.sec.shape and np.array_equal(np.asarray(old_sec), self.sec)

    def build_or_load(self) -> None:
        if self.ready():
            print(f"[pvnorm_cache] reuse frame cache: {self.cache_dir}", flush=True)
            return
        print(f"[pvnorm_cache] building frame cache: {self.cache_dir}", flush=True)
        fields = np.lib.format.open_memmap(
            self.field_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(self.times), self.grid_size, self.grid_size),
        )
        masks = np.lib.format.open_memmap(
            self.mask_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(self.times), self.grid_size, self.grid_size),
        )
        for i, ts in enumerate(self.times):
            vals = collect_pvnorm_peak_adjusted_values(self.split_df, ts, self.series, self.peak, self.args)
            field, support = splat_values(vals, self.split_df, self.bounds, self.args)
            mask = support > float(self.args.min_grid_weight)
            fields[i] = field.astype(np.float16)
            masks[i] = mask.astype(np.uint8)
            if (i + 1) % 500 == 0:
                print(f"[pvnorm_cache] cached {i + 1}/{len(self.times)} frames latest={ts}", flush=True)
        fields.flush()
        masks.flush()
        np.save(self.sec_path, self.sec)
        write_json(
            self.meta_path,
            {
                "n_frames": int(len(self.times)),
                "grid_size": int(self.grid_size),
                "gaussian_sigma_cells": float(self.args.gaussian_sigma_cells),
                "gaussian_radius_cells": int(self.args.gaussian_radius_cells),
                "min_grid_weight": float(self.args.min_grid_weight),
                "value_type": "pv_norm_div_station_pv_norm_q95",
                "value_description": "capacity-normalized PV divided by each station's train-period PV_norm quantile; no clear-sky/CSI normalization",
                "peak_quantile": float(self.args.peak_quantile),
                "pv_field_clip": float(self.args.pv_field_clip),
                "source_station_count": int(len(self.split_df)),
                "source_station_sns_sha1": _station_sns_sha1(self.split_df),
                "source_station_policy": "target stations excluded before splatting PV_norm field frames",
                "dtype_field": "float16",
                "dtype_mask": "uint8",
            },
        )


def plot_targets(split_df: pd.DataFrame, candidates: pd.DataFrame, targets: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(split_df["lon"], split_df["lat"], s=24, c="#b8c6d6", edgecolors="#5b6b7a", linewidths=0.4, alpha=0.75, label="all stations")
    if not candidates.empty:
        ax.scatter(candidates["lon"], candidates["lat"], s=34, c="#66c2a5", edgecolors="#2b7b68", linewidths=0.5, alpha=0.8, label="eligible candidates")
    if {"lhs_anchor_lon", "lhs_anchor_lat"}.issubset(targets.columns):
        lhs = targets.dropna(subset=["lhs_anchor_lon", "lhs_anchor_lat"]).copy()
        if not lhs.empty:
            ax.scatter(
                lhs["lhs_anchor_lon"],
                lhs["lhs_anchor_lat"],
                s=62,
                marker="x",
                c="#6a3d9a",
                linewidths=1.3,
                label="LHS anchors",
            )
            for _, row in lhs.iterrows():
                ax.plot(
                    [float(row["lhs_anchor_lon"]), float(row["lon"])],
                    [float(row["lhs_anchor_lat"]), float(row["lat"])],
                    color="#6a3d9a",
                    linewidth=0.7,
                    alpha=0.35,
                )
    ax.scatter(targets["lon"], targets["lat"], s=170, marker="*", c="#d62728", edgecolors="black", linewidths=1.1, label="selected targets")
    for _, row in targets.iterrows():
        ax.text(float(row["lon"]) + 0.012, float(row["lat"]) + 0.012, str(int(row["target_rank"])), color="#b2182b", fontsize=12, fontweight="bold")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Selected target PV stations | method={args.selection_method} | n={len(targets)}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_target_stations_map.png", dpi=int(args.dpi))
    plt.close(fig)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    target_run_dir = ensure_dir(args.target_run_dir)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    time_df = build_multisource_issue_time_table(args)
    if time_df.empty:
        raise RuntimeError("No issue times generated. Check time/freq/past/future settings.")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    time_df.to_csv(out_dir / "issue_time_index.csv", index=False)

    series, load_status = load_station_series(split_df, args)
    load_status.to_csv(out_dir / "station_csv_load_status_before_health_filter.csv", index=False)
    series = _health_filter_station_series(series, args, out_dir)

    diag = _station_diagnostics(split_df, series, time_df, bounds, args)
    local_den = diag["local_neighbors"].to_numpy(np.float64)
    sample = diag["train_issue_samples"].to_numpy(np.float64)
    center = diag["center_distance_km"].to_numpy(np.float64)
    diag["selection_score"] = (
        float(args.density_weight) * (local_den / max(1.0, float(np.nanmax(local_den))))
        + float(args.sample_count_weight) * (sample / max(1.0, float(np.nanmax(sample))))
        - float(args.centrality_weight) * (center / max(1.0, float(np.nanmax(center))))
    )
    eligible = diag[
        (diag["n_rows"] >= int(args.min_station_rows))
        & (diag["train_issue_samples"] >= int(args.min_train_issue_samples))
        & (diag["val_issue_samples"] >= int(args.min_val_issue_samples))
        & (diag["test_issue_samples"] >= int(args.min_test_issue_samples))
        & (diag["local_neighbors"] >= int(args.min_local_neighbors))
    ].copy()
    targets = select_targets(diag, args)
    target_sns = set(targets["station_sn"].astype(str))
    remaining = split_df[~split_df["station_sn"].astype(str).isin(target_sns)].copy().reset_index(drop=True)
    remaining["split"] = "all"
    remaining_series = {sn: df for sn, df in series.items() if sn not in target_sns}

    diag.to_csv(out_dir / "target_candidate_scores.csv", index=False)
    eligible.to_csv(out_dir / "eligible_target_candidates.csv", index=False)
    targets.to_csv(out_dir / "target_stations.csv", index=False)
    remaining.to_csv(out_dir / "target_excluded_station_split.csv", index=False)
    split_df.to_csv(target_run_dir / "station_split.csv", index=False)
    remaining.to_csv(target_run_dir / "target_excluded_station_split.csv", index=False)
    plot_targets(split_df, eligible, targets, out_dir, args)

    if bool(args.select_only):
        write_json(
            out_dir / "01_target_selection_preview_meta.json",
            {
              "purpose": "Preview target station selection before target-excluded PV field cache construction.",
              "selection_method": str(args.selection_method),
              "target_count": int(len(targets)),
              "eligible_candidate_count": int(len(eligible)),
              "remaining_station_count_if_applied": int(len(remaining)),
              "min_target_separation_km": float(args.min_target_separation_km),
              "lhs_support_radius_km": float(args.lhs_support_radius_km),
              "lhs_min_support_neighbors": int(args.lhs_min_support_neighbors),
              "field_cache_built": False,
                "next_step": "Re-run without --select_only after the selected_target_stations_map.png looks acceptable.",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "outputs": {
                    "target_stations": str(out_dir / "target_stations.csv"),
                    "target_map": str(out_dir / "selected_target_stations_map.png"),
                    "candidate_scores": str(out_dir / "target_candidate_scores.csv"),
                    "eligible_candidates": str(out_dir / "eligible_target_candidates.csv"),
                },
            },
        )
        print("==== Target selection preview only ====")
        cols = [
            "target_rank",
            "selection_method",
            "station_sn",
            "lon",
            "lat",
            "n_rows",
            "train_issue_samples",
            "val_issue_samples",
            "test_issue_samples",
            "local_neighbors",
            "support_neighbors_after_all_targets_excluded",
            "center_distance_km",
            "selection_score",
        ]
        cols = [c for c in cols if c in targets.columns]
        print(targets[cols].to_string(index=False))
        print(f"target map: {out_dir / 'selected_target_stations_map.png'}")
        print("PV field cache was NOT built. Re-run without --select_only after confirming the positions.")
        return

    peak = estimate_station_pvnorm_peak(remaining, remaining_series, time_df, args)
    pd.DataFrame([{"station_sn": k, "pv_norm_q95": v} for k, v in sorted(peak.items())]).to_csv(
        out_dir / "target_excluded_station_pv_norm_q95.csv",
        index=False,
    )

    builder = PVNormPeakAdjustedFrameCache(Path(args.frame_cache_dir), remaining, remaining_series, peak, bounds, time_df, args)
    builder.build_or_load()

    write_json(
        out_dir / "01_target_selection_meta.json",
        {
            "purpose": "New target selection and target-excluded PV_norm field cache after same-day06 NWP rebuild.",
            "field_value": "PV_norm / station_train_PV_norm_q95, clipped; no clear-sky/CSI division.",
              "target_selection": {
                  "selection_method": str(args.selection_method),
                  "min_station_rows": int(args.min_station_rows),
                  "min_train_issue_samples": int(args.min_train_issue_samples),
                  "min_val_issue_samples": int(args.min_val_issue_samples),
                  "min_test_issue_samples": int(args.min_test_issue_samples),
                  "density_radius_km": float(args.density_radius_km),
                  "min_local_neighbors": int(args.min_local_neighbors),
                  "lhs_support_radius_km": float(args.lhs_support_radius_km),
                  "lhs_min_support_neighbors": int(args.lhs_min_support_neighbors),
                  "min_target_separation_km": float(args.min_target_separation_km),
                  "max_targets": int(args.max_targets),
              },
            "target_count": int(len(targets)),
            "remaining_station_count": int(len(remaining)),
            "bounds": {"lon_min": bounds[0], "lon_max": bounds[1], "lat_min": bounds[2], "lat_max": bounds[3]},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "outputs": {
                "target_stations": str(out_dir / "target_stations.csv"),
                "target_map": str(out_dir / "selected_target_stations_map.png"),
                "candidate_scores": str(out_dir / "target_candidate_scores.csv"),
                "target_excluded_frame_cache": str(args.frame_cache_dir),
                "target_run_dir": str(target_run_dir),
            },
        },
    )
    print("==== New target selection and PV_norm cache ====")
    cols = [
        "target_rank",
        "selection_method",
        "station_sn",
        "lon",
        "lat",
        "n_rows",
        "train_issue_samples",
        "val_issue_samples",
        "test_issue_samples",
        "local_neighbors",
        "support_neighbors_after_all_targets_excluded",
        "center_distance_km",
        "selection_score",
    ]
    cols = [c for c in cols if c in targets.columns]
    print(targets[cols].to_string(index=False))
    print(f"targets: {out_dir / 'target_stations.csv'}")
    print(f"target map: {out_dir / 'selected_target_stations_map.png'}")
    print(f"target-excluded PV_norm cache: {args.frame_cache_dir}")


if __name__ == "__main__":
    main()
