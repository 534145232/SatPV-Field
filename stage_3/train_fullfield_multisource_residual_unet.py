#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_fullfield_multisource_cache import _resize_bilinear
from train_fullfield_dense_temporal_forecast import _load_split_df, _parse_time_list
from train_fullfield_dense_temporal_residual_unet import FullFieldDiskFrameCache, Residual3DUNet, _grid_lonlat, _time_solar_features
from train_fullfield_dense_temporal_residual_unet_v2 import ReadOnlyDiskFrameCache, _masked_for_plot
from train_regional_dense_reconstruction import masked_huber
from visualize_regional_split_frames import _bounds, ensure_dir, estimate_kpi, load_station_series, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults",
    "train_pv_dir": "/data/DayinChen/NWP/PV/train_directory_cleaned_csv",
    "val_pv_dir": "/data/DayinChen/NWP/PV/val_directory_cleaned_csv",
    "out_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_multisource_h8_nwp_health_weighted_seed2026",
    "frame_cache_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_frame_cache_64_health_seed2026",
    "multisource_cache_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_multisource_cache_issue_h8_seed2026",
    "time_start": "2022-01-01 07:00:00",
    "time_end": "2023-06-24 16:45:00",
    "train_end": "2023-01-13 23:59:59",
    "val_end": "2023-03-08 23:59:59",
    "freq_minutes": 15,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "csi_clip": 2.0,
    "delta_scale": 0.6,
    "batch_size": 8,
    "epochs": 60,
    "patience": 12,
    "early_stop_min_delta": 5e-5,
    "lr": 1.0e-3,
    "weight_decay": 1e-4,
    "lr_patience": 4,
    "lr_factor": 0.5,
    "seed": 2026,
    "device": "cuda",
    "num_workers": 0,
    "models": "pv_h8_nwp",
    "large_delta_threshold": 0.10,
    "big_change_loss_weight": 0.5,
    "health_filter_frame_cache": True,
    "pv_health_max": 1.35,
    "pv_health_active_threshold": 0.03,
    "pv_health_flatline_window": 12,
    "pv_health_flatline_range": 5e-4,
    "clearsky_eps": 0.02,
    "min_clear_sky": 0.03,
    "capacity_scale": 1000.0,
    "kpi_quantile": 0.95,
    "kpi_min_samples": 40,
    "kpi_max_samples_per_station": 2000,
    "sample_count": 6,
    "visual_horizons": "1,4,8",
    "dpi": 150,
}

MODEL_SPECS = {
    "pv_only": {"use_h8": False, "use_nwp": False, "label": "PV only"},
    "pv_h8": {"use_h8": True, "use_nwp": False, "label": "PV + H8"},
    "pv_nwp": {"use_h8": False, "use_nwp": True, "label": "PV + NWP"},
    "pv_h8_nwp": {"use_h8": True, "use_nwp": True, "label": "PV + H8 + NWP"},
}

NWP_NAMES = ["radiation", "direction10", "speed10", "tcc"]


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
            "Fixed residual 3D U-Net V1 multisource ablation: PV only, PV+H8, PV+NWP, PV+H8+NWP. "
            "The backbone is not changed; only source channels are added."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["frame_cache_dir"]))
    parser.add_argument("--multisource_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["multisource_cache_dir"]))
    parser.add_argument("--time_start", type=str, default=str(PYCHARM_DEFAULTS["time_start"]))
    parser.add_argument("--time_end", type=str, default=str(PYCHARM_DEFAULTS["time_end"]))
    parser.add_argument("--train_end", type=str, default=str(PYCHARM_DEFAULTS["train_end"]))
    parser.add_argument("--val_end", type=str, default=str(PYCHARM_DEFAULTS["val_end"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(PYCHARM_DEFAULTS["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(PYCHARM_DEFAULTS["future_frames"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--csi_clip", type=float, default=float(PYCHARM_DEFAULTS["csi_clip"]))
    parser.add_argument("--delta_scale", type=float, default=float(PYCHARM_DEFAULTS["delta_scale"]))
    parser.add_argument("--batch_size", type=int, default=int(PYCHARM_DEFAULTS["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(PYCHARM_DEFAULTS["epochs"]))
    parser.add_argument("--patience", type=int, default=int(PYCHARM_DEFAULTS["patience"]))
    parser.add_argument("--early_stop_min_delta", type=float, default=float(PYCHARM_DEFAULTS["early_stop_min_delta"]))
    parser.add_argument("--lr", type=float, default=float(PYCHARM_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(PYCHARM_DEFAULTS["weight_decay"]))
    parser.add_argument("--lr_patience", type=int, default=int(PYCHARM_DEFAULTS["lr_patience"]))
    parser.add_argument("--lr_factor", type=float, default=float(PYCHARM_DEFAULTS["lr_factor"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--device", type=str, default=str(PYCHARM_DEFAULTS["device"]))
    parser.add_argument("--num_workers", type=int, default=int(PYCHARM_DEFAULTS["num_workers"]))
    parser.add_argument("--models", type=str, default=str(PYCHARM_DEFAULTS["models"]))
    parser.add_argument("--large_delta_threshold", type=float, default=float(PYCHARM_DEFAULTS["large_delta_threshold"]))
    parser.add_argument("--big_change_loss_weight", type=float, default=float(PYCHARM_DEFAULTS["big_change_loss_weight"]))
    parser.add_argument("--health_filter_frame_cache", action="store_true", default=bool(PYCHARM_DEFAULTS["health_filter_frame_cache"]))
    parser.add_argument("--disable_health_filter_frame_cache", action="store_false", dest="health_filter_frame_cache")
    parser.add_argument("--pv_health_max", type=float, default=float(PYCHARM_DEFAULTS["pv_health_max"]))
    parser.add_argument("--pv_health_active_threshold", type=float, default=float(PYCHARM_DEFAULTS["pv_health_active_threshold"]))
    parser.add_argument("--pv_health_flatline_window", type=int, default=int(PYCHARM_DEFAULTS["pv_health_flatline_window"]))
    parser.add_argument("--pv_health_flatline_range", type=float, default=float(PYCHARM_DEFAULTS["pv_health_flatline_range"]))
    parser.add_argument("--clearsky_eps", type=float, default=float(PYCHARM_DEFAULTS["clearsky_eps"]))
    parser.add_argument("--min_clear_sky", type=float, default=float(PYCHARM_DEFAULTS["min_clear_sky"]))
    parser.add_argument("--capacity_scale", type=float, default=float(PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--kpi_quantile", type=float, default=float(PYCHARM_DEFAULTS["kpi_quantile"]))
    parser.add_argument("--kpi_min_samples", type=int, default=int(PYCHARM_DEFAULTS["kpi_min_samples"]))
    parser.add_argument("--kpi_max_samples_per_station", type=int, default=int(PYCHARM_DEFAULTS["kpi_max_samples_per_station"]))
    parser.add_argument("--sample_count", type=int, default=int(PYCHARM_DEFAULTS["sample_count"]))
    parser.add_argument("--visual_horizons", type=str, default=str(PYCHARM_DEFAULTS["visual_horizons"]))
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _parse_visual_horizons(text: str, future_frames: int) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if 1 <= h <= future_frames:
            out.append(h - 1)
    return out or [0, future_frames - 1]


def _load_multisource_cache(cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    required = [
        "issue_time_index.csv",
        "h8_swr_64_float16.npy",
        "nwp_future_fields_float16.npy",
        "nwp_future_valid_uint8.npy",
        "multisource_cache_meta.json",
    ]
    missing = [str(cache_dir / name) for name in required if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Multisource cache is incomplete. Run run_build_fullfield_multisource_cache.sh first. Missing:\n"
            + "\n".join(missing)
        )
    time_df = pd.read_csv(cache_dir / "issue_time_index.csv")
    with open(cache_dir / "multisource_cache_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "dir": cache_dir,
        "time_df": time_df,
        "h8": np.load(cache_dir / "h8_swr_64_float16.npy", mmap_mode="r"),
        "nwp": np.load(cache_dir / "nwp_future_fields_float16.npy", mmap_mode="r"),
        "nwp_valid": np.load(cache_dir / "nwp_future_valid_uint8.npy", mmap_mode="r"),
        "meta": meta,
    }


def _issue_index(time_df: pd.DataFrame) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(time_df["issue_sec"].astype(np.int64).tolist())}


def _compute_or_load_norm_stats(cache: dict, out_dir: Path, time_df: pd.DataFrame, force: bool = False) -> dict:
    path = out_dir / "multisource_norm_stats.json"
    if path.exists() and not force:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    idx = time_df.index[time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
    if idx.size == 0:
        raise ValueError("No train issue rows available for multisource normalization.")
    h8 = np.asarray(cache["h8"][idx], dtype=np.float32)
    h8_finite = h8[np.isfinite(h8)]
    h8_mean = float(np.nanmean(h8_finite)) if h8_finite.size else 0.0
    h8_std = float(np.nanstd(h8_finite)) if h8_finite.size else 1.0
    if not np.isfinite(h8_std) or h8_std < 1e-6:
        h8_std = 1.0

    nwp = np.asarray(cache["nwp"][idx], dtype=np.float32)
    nwp_mean, nwp_std = [], []
    for c in range(nwp.shape[2]):
        vals = nwp[:, :, c]
        vals = vals[np.isfinite(vals)]
        mean = float(np.nanmean(vals)) if vals.size else 0.0
        std = float(np.nanstd(vals)) if vals.size else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        nwp_mean.append(mean)
        nwp_std.append(std)
    stats = {
        "h8_mean": h8_mean,
        "h8_std": h8_std,
        "nwp_names": NWP_NAMES,
        "nwp_mean": nwp_mean,
        "nwp_std": nwp_std,
        "source": str(cache["dir"]),
    }
    write_json(path, stats)
    return stats


def _normalize_h8(arr: np.ndarray, stats: dict) -> np.ndarray:
    out = (arr.astype(np.float32) - float(stats["h8_mean"])) / max(float(stats["h8_std"]), 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize_nwp(arr: np.ndarray, valid: np.ndarray, stats: dict, grid_size: int) -> np.ndarray:
    # arr: [Hf, 4, 16, 16] -> [Hf, 4, 64, 64]
    mean = np.asarray(stats["nwp_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(stats["nwp_std"], dtype=np.float32), 1e-6)
    out = []
    for h in range(arr.shape[0]):
        chans = []
        for c in range(arr.shape[1]):
            coarse = arr[h, c].astype(np.float32)
            if int(valid[h, c]) <= 0:
                resized = np.zeros((grid_size, grid_size), dtype=np.float32)
            else:
                resized = _resize_bilinear(coarse, grid_size, grid_size)
                resized = (resized - mean[c]) / std[c]
                resized = np.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(resized.astype(np.float32))
        out.append(np.stack(chans, axis=0))
    return np.stack(out, axis=0).astype(np.float32)


def _health_filter_station_series(series: dict[str, pd.DataFrame], args: argparse.Namespace, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Remove clear numerical PV pathologies before splatting stations into DensePV fields.

    This is intentionally point-level and conservative. It does not remove valid
    low/zero PV values. It only masks non-finite/out-of-range values and long
    active flatlines, so neighboring stations can fill the affected grid region.
    """

    if not bool(args.health_filter_frame_cache):
        return series
    filtered: dict[str, pd.DataFrame] = {}
    rows = []
    max_pv = float(args.pv_health_max)
    active_thr = float(args.pv_health_active_threshold)
    win = max(2, int(args.pv_health_flatline_window))
    flat_range = float(args.pv_health_flatline_range)
    for sn, df in series.items():
        work = df.copy()
        pv = pd.to_numeric(work["pv_norm"], errors="coerce").astype(float)
        bad = (~np.isfinite(pv.to_numpy(np.float64))) | (pv < -0.03) | (pv > max_pv)
        active = pv >= active_thr
        rolling_range = pv.rolling(win, center=True, min_periods=win).max() - pv.rolling(win, center=True, min_periods=win).min()
        flat_bad = active & (rolling_range <= flat_range)
        bad = np.asarray(bad, dtype=bool) | flat_bad.fillna(False).to_numpy(bool)
        kept = work.loc[~bad].copy()
        filtered[str(sn)] = kept
        rows.append(
            {
                "station_sn": str(sn),
                "before": int(len(work)),
                "after": int(len(kept)),
                "dropped": int(np.sum(bad)),
                "drop_ratio": float(np.mean(bad)) if len(work) else 0.0,
                "out_of_range_or_nonfinite": int(np.sum(np.asarray((~np.isfinite(pv.to_numpy(np.float64))) | (pv < -0.03) | (pv > max_pv), dtype=bool))),
                "active_flatline": int(np.sum(flat_bad.fillna(False).to_numpy(bool))),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "frame_cache_health_filter_station_report.csv", index=False)
    summary = {
        "enabled": True,
        "stations": int(len(report)),
        "rows_before": int(report["before"].sum()) if len(report) else 0,
        "rows_after": int(report["after"].sum()) if len(report) else 0,
        "rows_dropped": int(report["dropped"].sum()) if len(report) else 0,
        "drop_ratio": float(report["dropped"].sum() / max(1, report["before"].sum())) if len(report) else 0.0,
        "pv_health_max": max_pv,
        "pv_health_active_threshold": active_thr,
        "pv_health_flatline_window": win,
        "pv_health_flatline_range": flat_range,
    }
    write_json(out_dir / "frame_cache_health_filter_summary.json", summary)
    print(
        "[health_frame_cache] "
        f"rows_before={summary['rows_before']} rows_after={summary['rows_after']} "
        f"dropped={summary['rows_dropped']} drop_ratio={summary['drop_ratio']:.4f}",
        flush=True,
    )
    return filtered


def _prepare_frame_cache(
    split_df: pd.DataFrame,
    time_df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    frame_cache_dir = Path(args.frame_cache_dir)
    required = ["fields_float16.npy", "masks_uint8.npy", "timestamp_sec.npy", "frame_cache_meta.json"]
    if all((frame_cache_dir / name).exists() for name in required):
        return
    print(f"[frame_cache] building health-filtered PV frame cache: {frame_cache_dir}", flush=True)
    series, load_status = load_station_series(split_df, args)
    load_status.to_csv(out_dir / "frame_cache_station_csv_load_status.csv", index=False)
    series = _health_filter_station_series(series, args, out_dir)
    kpi = estimate_kpi(split_df, series, args)
    pd.DataFrame([{"station_sn": k, "kpi": v} for k, v in sorted(kpi.items())]).to_csv(
        out_dir / "frame_cache_station_kpi_geo_baselines.csv",
        index=False,
    )
    builder = FullFieldDiskFrameCache(frame_cache_dir, split_df, series, kpi, bounds, time_df, args)
    builder.build_or_load()


def weighted_masked_huber(pred, target, mask, x, threshold: float, big_weight: float):
    import torch
    import torch.nn.functional as F

    valid = mask > 0.5
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0
    if float(big_weight) <= 0:
        return F.smooth_l1_loss(pred[valid], target[valid])
    last = x[:, 0, -1].unsqueeze(1).expand_as(target)
    true_delta = torch.abs(target - last)
    weights = torch.ones_like(target)
    weights = weights + float(big_weight) * (true_delta >= float(threshold)).float()
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    weighted = loss * weights * valid.float()
    return weighted.sum() / torch.clamp((weights * valid.float()).sum(), min=1.0)


class FullFieldMultisourceDataset:
    def __init__(
        self,
        time_df: pd.DataFrame,
        pv_cache: ReadOnlyDiskFrameCache,
        ms_cache: dict,
        bounds: tuple[float, float, float, float],
        stats: dict,
        model_name: str,
        args: argparse.Namespace,
    ):
        self.time_df = time_df.reset_index(drop=True)
        self.pv_cache = pv_cache
        self.ms_cache = ms_cache
        self.stats = stats
        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.args = args
        self.global_issue_index = _issue_index(ms_cache["time_df"])
        lon_grid, lat_grid = _grid_lonlat(bounds, int(args.grid_size))
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.feature_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.time_df)

    def _features(self, ts: pd.Timestamp) -> np.ndarray:
        key = _timestamp_key(ts)
        item = self.feature_cache.get(key)
        if item is None:
            item = _time_solar_features(ts, self.lon_grid, self.lat_grid)
            self.feature_cache[key] = item
        return item

    def source_dims(self) -> tuple[int, int]:
        in_channels = 5 + (4 if self.spec["use_h8"] else 0)
        future_channels = 3 + (4 if self.spec["use_nwp"] else 0)
        return in_channels, future_channels

    def __getitem__(self, idx: int):
        row = self.time_df.iloc[idx]
        issue_idx = self.global_issue_index[int(row["issue_sec"])]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])

        h8_extra = None
        if self.spec["use_h8"]:
            h8_extra = _normalize_h8(np.asarray(self.ms_cache["h8"][issue_idx], dtype=np.float32), self.stats)

        nwp_extra = None
        if self.spec["use_nwp"]:
            nwp_extra = _normalize_nwp(
                np.asarray(self.ms_cache["nwp"][issue_idx], dtype=np.float32),
                np.asarray(self.ms_cache["nwp_valid"][issue_idx], dtype=np.uint8),
                self.stats,
                int(self.args.grid_size),
            )

        x_list = []
        for ts in past_times:
            field, mask = self.pv_cache.get(ts)
            channels = [field[None], mask[None], self._features(ts)]
            if h8_extra is not None:
                channels.append(h8_extra)
            x_list.append(np.concatenate(channels, axis=0))

        y_list, m_list, fut_list = [], [], []
        for h, ts in enumerate(future_times):
            field, mask = self.pv_cache.get(ts)
            fut_ch = [self._features(ts)]
            if nwp_extra is not None:
                fut_ch.append(nwp_extra[h])
            y_list.append(field)
            m_list.append(mask)
            fut_list.append(np.concatenate(fut_ch, axis=0))

        x = np.stack(x_list, axis=1).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        m = np.stack(m_list, axis=0).astype(np.float32)
        fut = np.stack(fut_list, axis=0).astype(np.float32)
        return x, fut, y, m


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _metric_rows(model_name: str, split: str, pred: np.ndarray, y: np.ndarray, m: np.ndarray, x: np.ndarray, threshold: float) -> list[dict]:
    rows = []
    mask = m > 0.5
    if not mask.any():
        return rows
    last = x[:, 0, -1]
    last_seq = np.repeat(last[:, None], pred.shape[1], axis=1)
    err = pred - y
    pred_delta = pred - last_seq
    true_delta = y - last_seq
    horizons = [("all", None)] + [(f"h{i + 1}", i) for i in range(pred.shape[1])]
    for label, h in horizons:
        mh = mask if h is None else mask[:, h]
        e = err[mask] if h is None else err[:, h][mh]
        pdlt = pred_delta[mask] if h is None else pred_delta[:, h][mh]
        tdlt = true_delta[mask] if h is None else true_delta[:, h][mh]
        if e.size == 0:
            continue
        large = np.abs(tdlt) >= float(threshold)
        sign_mask = np.abs(tdlt) > 1e-6
        rows.append(
            {
                "model": model_name,
                "eval_split": split,
                "horizon": label,
                "n_points": int(e.size),
                "mae_field": float(np.mean(np.abs(e))),
                "rmse_field": float(np.sqrt(np.mean(e * e))),
                "delta_mae": float(np.mean(np.abs(pdlt - tdlt))),
                "delta_corr": _corrcoef(pdlt, tdlt),
                "delta_sign_accuracy": float(np.mean(np.sign(pdlt[sign_mask]) == np.sign(tdlt[sign_mask]))) if sign_mask.any() else float("nan"),
                "large_delta_threshold": float(threshold),
                "large_delta_n_points": int(large.sum()),
                "large_delta_mae": float(np.mean(np.abs(e[large]))) if large.any() else float("nan"),
            }
        )
    step_mask = mask[:, 1:] & mask[:, :-1]
    if step_mask.any():
        pred_step = np.abs(pred[:, 1:] - pred[:, :-1])[step_mask]
        true_step = np.abs(y[:, 1:] - y[:, :-1])[step_mask]
        rows.append(
            {
                "model": model_name,
                "eval_split": split,
                "horizon": "horizon_diversity",
                "n_points": int(step_mask.sum()),
                "mae_field": float("nan"),
                "rmse_field": float("nan"),
                "delta_mae": float("nan"),
                "delta_corr": float("nan"),
                "delta_sign_accuracy": float("nan"),
                "large_delta_threshold": float(threshold),
                "large_delta_n_points": 0,
                "large_delta_mae": float("nan"),
                "pred_horizon_diversity": float(np.mean(pred_step)),
                "truth_horizon_diversity": float(np.mean(true_step)),
                "horizon_diversity_ratio": float(np.mean(pred_step) / max(float(np.mean(true_step)), 1e-8)),
            }
        )
    return rows


def _combine_metrics(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    weighted_cols = ["mae_field", "rmse_field", "delta_mae", "delta_sign_accuracy", "large_delta_mae"]
    for keys, g in df.groupby(["model", "eval_split", "horizon"], sort=True):
        n = g["n_points"].to_numpy(np.float64)
        total = max(float(n.sum()), 1.0)
        item = {"model": keys[0], "eval_split": keys[1], "horizon": keys[2], "n_points": int(n.sum())}
        for col in weighted_cols:
            vals = g[col].to_numpy(np.float64)
            ok = np.isfinite(vals)
            if not ok.any():
                item[col] = float("nan")
            elif col == "rmse_field":
                item[col] = float(math.sqrt(np.sum((vals[ok] ** 2) * n[ok]) / max(float(n[ok].sum()), 1.0)))
            elif col == "large_delta_mae":
                large_n = g["large_delta_n_points"].to_numpy(np.float64)
                item[col] = float(np.sum(vals[ok] * large_n[ok]) / max(float(large_n[ok].sum()), 1.0))
            else:
                item[col] = float(np.sum(vals[ok] * n[ok]) / max(float(n[ok].sum()), 1.0))
        # Correlations are summarized by median because averaging correlations across batches is misleading.
        corr = g["delta_corr"].to_numpy(np.float64)
        item["delta_corr_median"] = float(np.nanmedian(corr)) if np.isfinite(corr).any() else float("nan")
        item["large_delta_threshold"] = float(g["large_delta_threshold"].iloc[0])
        item["large_delta_n_points"] = int(np.nansum(g["large_delta_n_points"].to_numpy(np.float64)))
        for col in ["pred_horizon_diversity", "truth_horizon_diversity", "horizon_diversity_ratio"]:
            if col in g.columns:
                vals = g[col].to_numpy(np.float64)
                ok = np.isfinite(vals)
                item[col] = float(np.sum(vals[ok] * n[ok]) / max(float(n[ok].sum()), 1.0)) if ok.any() else float("nan")
        out.append(item)
    return pd.DataFrame(out).sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)


def run_epoch(model, loader, optimizer, device: str, train: bool, args: argparse.Namespace) -> float:
    import torch

    model.train(train)
    losses = []
    for step, (x, fut, y, m) in enumerate(loader, start=1):
        x = x.to(device)
        fut = fut.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x, fut)
            if train and float(args.big_change_loss_weight) > 0:
                loss = weighted_masked_huber(
                    pred,
                    y,
                    m,
                    x,
                    threshold=float(args.large_delta_threshold),
                    big_weight=float(args.big_change_loss_weight),
                )
            else:
                loss = masked_huber(pred, y, m)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[multisource][{mode}] batches={step}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_model(model, loader, device: str, model_name: str, split: str, threshold: float) -> pd.DataFrame:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for step, (x, fut, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device), fut.to(device)).detach().cpu().numpy()
            rows.extend(_metric_rows(model_name, split, pred, y.numpy(), m.numpy(), x.numpy(), threshold))
            if step % 100 == 0:
                print(f"[multisource][{model_name}][{split}] eval batches={step}/{len(loader)}", flush=True)
    return _combine_metrics(rows)


def _robust_limits(values: list[np.ndarray], masks: list[np.ndarray]) -> tuple[float, float]:
    chunks = []
    for val, mask in zip(values, masks):
        if np.any(mask > 0.5):
            chunks.append(np.asarray(val)[mask > 0.5])
    if not chunks:
        return 0.0, 1.0
    arr = np.concatenate(chunks)
    lo = float(np.nanquantile(arr, 0.02))
    hi = float(np.nanquantile(arr, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    pad = 0.05 * (hi - lo)
    return max(0.0, lo - pad), min(2.0, hi + pad)


def save_comparison_samples(models: dict, datasets: dict, bounds: tuple[float, float, float, float], device: str, args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sample_dir = ensure_dir(out_dir / "sample_multisource_comparison")
    reference_name = next(iter(datasets))
    test_ds = datasets[reference_name]["test"]
    n = min(int(args.sample_count), len(test_ds))
    if n <= 0:
        return
    horizons = _parse_visual_horizons(str(args.visual_horizons), int(args.future_frames))
    indices = np.linspace(0, len(test_ds) - 1, n, dtype=int)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    rows = []
    for sample_id, idx in enumerate(indices):
        row = test_ds.time_df.iloc[int(idx)]
        issue = pd.Timestamp(row["issue_time"])
        preds = {}
        truth = None
        mask = None
        latest = None
        for name, model in models.items():
            ds = datasets[name]["test"]
            x, fut, y, m = ds[int(idx)]
            if truth is None:
                truth = y
                mask = m
                latest = x[0, -1]
                latest_mask = x[1, -1]
            with torch.no_grad():
                preds[name] = model(torch.from_numpy(x[None]).to(device), torch.from_numpy(fut[None]).to(device)).detach().cpu().numpy()[0]
        values = [truth[h] for h in horizons] + [preds[name][h] for name in models for h in horizons]
        masks = [mask[h] for _ in range(1 + len(models)) for h in horizons]
        vmin, vmax = _robust_limits(values, masks)
        fig, axes = plt.subplots(len(horizons), 2 + len(models), figsize=(3.2 * (2 + len(models)), 3.0 * len(horizons)), squeeze=False)
        for r, h in enumerate(horizons):
            future_time = issue + pd.Timedelta(minutes=int(args.freq_minutes) * (h + 1))
            panels = [("latest", latest, latest_mask), (f"truth h{h + 1}", truth[h], mask[h])]
            panels += [(MODEL_SPECS[name]["label"], preds[name][h], mask[h]) for name in models]
            for c, (title, arr, mm) in enumerate(panels):
                shown = _masked_for_plot(arr, mm)
                im = axes[r, c].imshow(shown, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
                axes[r, c].set_title(f"{title}\n{future_time:%Y-%m-%d %H:%M}" if c > 0 else f"{title}\nissue {issue:%H:%M}", fontsize=9)
                axes[r, c].set_xlabel("lon")
                axes[r, c].set_ylabel("lat")
                fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
        fig.suptitle(f"Multisource residual V1 comparison | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
        fig.tight_layout()
        path = sample_dir / f"multisource_compare_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}.png"
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        rows.append({"sample_id": int(sample_id), "dataset_index": int(idx), "issue_time": str(issue), "path": str(path)})
        print(f"[multisource] saved sample comparison {path}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "sample_multisource_comparison.csv", index=False)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for multisource residual U-Net training") from exc

    requested = [x.strip() for x in str(args.models).split(",") if x.strip()]
    bad = [x for x in requested if x not in MODEL_SPECS]
    if bad:
        raise ValueError(f"unsupported models: {bad}; valid={sorted(MODEL_SPECS)}")

    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    ms_cache = _load_multisource_cache(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"].copy()
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    time_df.to_csv(out_dir / "fullfield_multisource_issue_time_index.csv", index=False)

    _prepare_frame_cache(split_df, time_df, bounds, args, out_dir)
    pv_cache = ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    stats = _compute_or_load_norm_stats(ms_cache, out_dir, time_df)
    write_json(
        out_dir / "run_meta.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "models": requested,
            "backbone": "fixed residual_3d_unet V1",
            "pv_input": "past 12 DensePV CSI frames + support + time/solar channels",
            "h8_input": "four H8/SWR frames resized to 64x64, appended as source channels; H8 latest age is not used",
            "nwp_input": "future radiation, direction10, speed10, tcc from 12 old_NWP points, IDW coarse field then resized for future conditioning",
            "pv_frame_cache": "health-filtered frame cache" if bool(args.health_filter_frame_cache) else "frame cache without extra health filtering",
            "train_loss": (
                f"masked Huber with large-delta weight {float(args.big_change_loss_weight):.3f} "
                f"when |target-last_frame| >= {float(args.large_delta_threshold):.3f}"
            ),
            "cache": str(args.multisource_cache_dir),
            "diagnostics": ["delta_mae", "delta_corr_median", "delta_sign_accuracy", "large_delta_mae", "horizon_diversity_ratio"],
        },
    )

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    datasets: dict[str, dict[str, FullFieldMultisourceDataset]] = {}
    loaders: dict[str, dict[str, DataLoader]] = {}
    models = {}
    all_eval = []
    train_logs = []
    for model_name in requested:
        datasets[model_name] = {}
        loaders[model_name] = {}
        for split in ["train", "val", "test"]:
            tdf = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
            ds = FullFieldMultisourceDataset(tdf, pv_cache, ms_cache, bounds, stats, model_name, args)
            datasets[model_name][split] = ds
            loaders[model_name][split] = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
            )
        in_channels, future_channels = datasets[model_name]["train"].source_dims()
        print(
            f"==== multisource model: {model_name} ({MODEL_SPECS[model_name]['label']}) "
            f"in_channels={in_channels} future_channels={future_channels} ====",
            flush=True,
        )
        model = Residual3DUNet(
            int(args.future_frames),
            in_channels=in_channels,
            future_channels=future_channels,
            delta_scale=float(args.delta_scale),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_factor),
            patience=int(args.lr_patience),
            min_lr=1e-5,
        )
        best_val = float("inf")
        bad_count = 0
        ckpt_path = out_dir / f"{model_name}_best.pt"
        for epoch in range(1, int(args.epochs) + 1):
            tr = run_epoch(model, loaders[model_name]["train"], optimizer, device, train=True, args=args)
            va = run_epoch(model, loaders[model_name]["val"], optimizer, device, train=False, args=args)
            scheduler.step(va)
            lr_now = float(optimizer.param_groups[0]["lr"])
            train_logs.append({"model": model_name, "epoch": int(epoch), "train_loss": tr, "val_loss": va, "lr": lr_now})
            print(f"[multisource][{model_name}] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f} lr={lr_now:.2e}", flush=True)
            if va < best_val - float(args.early_stop_min_delta):
                best_val = va
                bad_count = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "model_name": model_name,
                        "in_channels": in_channels,
                        "future_channels": future_channels,
                    },
                    ckpt_path,
                )
            else:
                bad_count += 1
                if bad_count >= int(args.patience):
                    print(
                        f"[multisource][{model_name}] early stop at epoch={epoch} "
                        f"best_val={best_val:.6f} min_delta={float(args.early_stop_min_delta):.2e}",
                        flush=True,
                    )
                    break
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        models[model_name] = model.eval()
        for split in ["val", "test"]:
            eval_df = evaluate_model(
                model,
                loaders[model_name][split],
                device,
                model_name,
                split,
                threshold=float(args.large_delta_threshold),
            )
            all_eval.append(eval_df)
            print(f"==== {model_name} {split} ====")
            print(eval_df[eval_df["horizon"].isin(["all", "h6", "h7", "h8"])].to_string(index=False))

    pd.DataFrame(train_logs).to_csv(out_dir / "fullfield_multisource_train_log.csv", index=False)
    summary = pd.concat(all_eval, ignore_index=True).sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)
    summary.to_csv(out_dir / "fullfield_multisource_eval_summary.csv", index=False)
    save_comparison_samples(models, datasets, bounds, device, args, out_dir)
    print("==== Full-field multisource residual V1 aggregate summary ====")
    print(summary[summary["horizon"].isin(["all", "h6", "h7", "h8", "horizon_diversity"])].to_string(index=False))
    print(f"saved: {out_dir / 'fullfield_multisource_eval_summary.csv'}")


if __name__ == "__main__":
    main()

