#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
import sys
from typing import Any

STAGE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", STAGE2_DIR.parent)).expanduser().resolve()
WORKSPACE = PROJECT_ROOT
ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
).expanduser().resolve()
STAGE1_DIR = PROJECT_ROOT / "stage_1"
sys.path.insert(0, str(STAGE1_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from build_clear_sky_cache import DEFAULT_CLEAR_SKY_DIR, haurwitz_clear_sky, solar_cos_zenith_bjt
from h8_3d_common import (
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_INDEX_CSV,
    DEFAULT_SAMPLE_CACHE_DIR,
    PROCESSED_ROOT,
    TARGET_CHANNEL,
    _channel_lookup,
    cache_meta_path,
    load_index,
    write_json,
)
from train_3dcnn_swr import seed_everything, serializable_args, unwrap_model


RATIO_CHANNELS = ["SWR", "PAR", "UVA", "UVB", "TAOT_02", "TAAE"]

from stage1_paths import (  # noqa: E402
    BIG_FRAME_CACHE_DIR as STAGE1_BIG_FRAME_CACHE_DIR,
    BIG_INDEX_MAIN as STAGE1_BIG_INDEX_MAIN,
    CENTER_CLEAR_SKY_CACHE_DIR as STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    CENTER_FRAME_CACHE_DIR as STAGE1_CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_ALL as STAGE1_CENTER_INDEX_ALL,
    CENTER_SAMPLE_CACHE_DIR as STAGE1_CENTER_SAMPLE_CACHE_DIR,
)


STAGE2_ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_STAGE2_ROOT", ARTIFACT_ROOT / "stage2")
).expanduser().resolve()
STAGE2_RUNS_DIR = STAGE2_ARTIFACT_ROOT / "runs"
CENTER_CLEAR_SKY_DIR = STAGE1_CENTER_CLEAR_SKY_CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train upgraded 3D CNN to predict SWR / clear-sky-SWR ratio. "
            "Sample_number is excluded and feature outliers are robustly clipped."
        )
    )
    parser.add_argument(
        "--train_mode",
        choices=["center_only", "big_patch_5x5"],
        default="center_only",
        help="Train on the fixed center patch or on the 5x5 overlap patch bank.",
    )
    parser.add_argument("--index_csv", type=Path, default=None, help="Optional train-index override.")
    parser.add_argument("--frame_cache_dir", type=Path, default=None, help="Optional train frame-cache override.")
    parser.add_argument("--sample_cache_dir", type=Path, default=None, help="Optional train sample-cache override.")
    parser.add_argument("--clear_sky_dir", type=Path, default=None, help="Optional train clear-sky cache override.")
    parser.add_argument("--val_index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--val_frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--val_sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--val_clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--test_index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--test_frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--stats_json", type=Path, default=None)
    parser.add_argument("--force_recompute_stats", action="store_true")
    parser.add_argument("--min_clear_sky", type=float, default=50.0)
    parser.add_argument("--ratio_clip", type=float, default=1.6)
    parser.add_argument("--stats_max_samples", type=int, default=512)
    parser.add_argument("--stats_pixels_per_sample", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--temporal_head",
        type=str,
        default="last_mean",
        choices=["last_mean", "last_only", "flatten", "conv3d_decoder"],
        help="Temporal aggregation head for Ratio3DCNN.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def configure_stage1_sources(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_mode == "center_only":
        args.index_csv = args.index_csv or STAGE1_CENTER_INDEX_ALL
        args.frame_cache_dir = args.frame_cache_dir or STAGE1_CENTER_FRAME_CACHE_DIR
        args.sample_cache_dir = args.sample_cache_dir or STAGE1_CENTER_SAMPLE_CACHE_DIR
        args.clear_sky_dir = args.clear_sky_dir or STAGE1_CENTER_CLEAR_SKY_CACHE_DIR
        default_run_name = "h8_swr_3dcnn_ratio_stage1_center_only_4h3f"
    else:
        args.index_csv = args.index_csv or STAGE1_BIG_INDEX_MAIN
        args.frame_cache_dir = args.frame_cache_dir or STAGE1_BIG_FRAME_CACHE_DIR
        # Big patches intentionally read monthly frame cache and crop dynamically.
        args.sample_cache_dir = None
        args.clear_sky_dir = None
        default_run_name = "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f"
    base_out_dir = args.out_dir or (STAGE2_RUNS_DIR / default_run_name)
    head_dir_name = f"head_{args.temporal_head}"
    args.out_dir = (
        base_out_dir
        if base_out_dir.name == head_dir_name
        else base_out_dir / head_dir_name
    )
    return args


def assert_center_only_eval_index(index_csv: Path) -> None:
    if not index_csv.exists():
        raise FileNotFoundError(f"Center evaluation index not found: {index_csv}")
    df = pd.read_csv(index_csv)
    eval_rows = df[df["split"].isin(["val", "test"])].copy()
    if eval_rows.empty:
        raise ValueError(f"No val/test rows in center evaluation index: {index_csv}")
    if "is_center" in eval_rows.columns:
        values = eval_rows["is_center"].astype(str).str.lower()
        if not values.isin(["true", "1"]).all():
            raise AssertionError("Validation/test index contains non-center patches.")
    if "patch_id" in eval_rows.columns and eval_rows["patch_id"].nunique() > 1:
        raise AssertionError("Validation/test index contains multiple spatial patches.")


def assert_train_mode_index(index_csv: Path, train_mode: str) -> None:
    if not index_csv.exists():
        raise FileNotFoundError(f"Training index not found: {index_csv}")
    df = pd.read_csv(index_csv)
    train_rows = df[df["split"] == "train"].copy()
    if train_rows.empty:
        raise ValueError(f"No train rows in training index: {index_csv}")
    if train_mode == "center_only":
        if "patch_id" in train_rows.columns and train_rows["patch_id"].nunique() > 1:
            raise AssertionError("center_only training index contains multiple patches.")
        return
    if "patch_id" not in train_rows.columns:
        raise AssertionError("big_patch_5x5 training index has no patch_id column.")
    patch_count = int(train_rows["patch_id"].nunique())
    if patch_count != 25:
        raise AssertionError(f"Expected 25 training patches, found {patch_count}.")
    if "window_key" in train_rows.columns:
        patches_per_window = train_rows.groupby("window_key")["patch_id"].nunique()
        if not (patches_per_window == 25).all():
            raise AssertionError("Some training windows do not contain all 25 spatial patches.")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_ratio_stats_path(index_csv: Path, min_clear_sky: float, ratio_clip: float) -> Path:
    return index_csv.parent / f"{index_csv.stem}_h8_swr_ratio_3dcnn_stats_clear{int(min_clear_sky)}_clip{str(ratio_clip).replace('.', 'p')}.json"


def load_sample_arrays(sample_cache_dir: Path, clear_sky_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = read_json(sample_cache_dir / "sample_cache_meta.json")
    x = np.load(sample_cache_dir / "x_float32.npy", mmap_mode="r")
    y = np.load(sample_cache_dir / "y_float32.npy", mmap_mode="r")
    mask = np.load(sample_cache_dir / "mask_uint8.npy", mmap_mode="r")
    clear = np.load(clear_sky_dir / "clear_sky_swr_float32.npy", mmap_mode="r")
    return meta, x, y, mask, clear


def sample_cache_available(sample_cache_dir: Path | None, clear_sky_dir: Path | None) -> bool:
    if sample_cache_dir is None or clear_sky_dir is None:
        return False
    return (
        (sample_cache_dir / "sample_cache_meta.json").exists()
        and (sample_cache_dir / "x_float32.npy").exists()
        and (sample_cache_dir / "y_float32.npy").exists()
        and (sample_cache_dir / "mask_uint8.npy").exists()
        and (clear_sky_dir / "clear_sky_swr_float32.npy").exists()
    )


class RatioFrameReader:
    def __init__(self, channels: list[str], frame_cache_dir: Path | None):
        self.channels = channels
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir is not None else None
        self.frame_cache: dict[str, dict[str, Any]] = {}
        self.ds_cache: dict[str, xr.Dataset] = {}
        self.grid_cache: dict[tuple[str, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def close(self) -> None:
        for ds in self.ds_cache.values():
            ds.close()
        self.ds_cache = {}
        self.frame_cache = {}
        self.grid_cache = {}

    def _open_ds(self, nc_path: str) -> xr.Dataset:
        ds = self.ds_cache.get(nc_path)
        if ds is None:
            ds = xr.open_dataset(nc_path)
            self.ds_cache[nc_path] = ds
        return ds

    def _open_frame_cache(self, nc_path: str) -> dict[str, Any] | None:
        if self.frame_cache_dir is None:
            return None
        cached = self.frame_cache.get(nc_path)
        if cached is not None:
            return cached
        meta_path = cache_meta_path(self.frame_cache_dir, nc_path)
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        data_path = Path(meta["data_path"])
        if not data_path.is_absolute():
            data_path = meta_path.parent / data_path
        data = np.load(data_path, mmap_mode="r")
        mask = None
        mask_path_text = meta.get("valid_mask_path")
        if mask_path_text:
            mask_path = Path(mask_path_text)
            if not mask_path.is_absolute():
                mask_path = meta_path.parent / mask_path
            if mask_path.exists():
                mask = np.load(mask_path, mmap_mode="r")
        lookup = {str(name): i for i, name in enumerate(meta["channels"])}
        missing = [c for c in self.channels + [TARGET_CHANNEL] if c not in lookup]
        if missing:
            raise KeyError(f"Missing channels in frame cache {meta_path}: {missing}")
        cached = {"data": data, "mask": mask, "lookup": lookup}
        self.frame_cache[nc_path] = cached
        return cached

    def _slices(self, row: pd.Series) -> tuple[slice, slice, slice, slice]:
        xs = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
        ys = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)
        if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index):
            lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
            lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
        else:
            lat_slice = slice(None)
            lon_slice = slice(None)
        return xs, ys, lat_slice, lon_slice

    def _clear_sky(self, row: pd.Series, ds: xr.Dataset, lat_slice: slice, lon_slice: slice) -> np.ndarray:
        grid_key = (
            str(row["nc_path"]),
            int(lat_slice.start or 0),
            int(lat_slice.stop or len(ds["latitude"])),
            int(lon_slice.start or 0),
            int(lon_slice.stop or len(ds["longitude"])),
        )
        cached_grid = self.grid_cache.get(grid_key)
        if cached_grid is None:
            lat = np.asarray(ds["latitude"].values[lat_slice], dtype=np.float32)
            lon = np.asarray(ds["longitude"].values[lon_slice], dtype=np.float32)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            cached_grid = (lon_grid.astype(np.float32), lat_grid.astype(np.float32))
            self.grid_cache[grid_key] = cached_grid
        lon_grid, lat_grid = cached_grid
        history_frames = int(row["history_frames"]) if "history_frames" in row.index else 4
        future_frames = int(row["future_frames"]) if "future_frames" in row.index else 2
        keys = [f"x_time_{i}_bjt" for i in range(history_frames)]
        keys.extend(f"y_time_{i}_bjt" for i in range(future_frames))
        out = []
        for key in keys:
            cosz = solar_cos_zenith_bjt(pd.Timestamp(row[key]), lat_grid, lon_grid)
            out.append(haurwitz_clear_sky(cosz))
        return np.stack(out, axis=0).astype(np.float32)

    def read(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nc_path = str(row["nc_path"])
        xs, ys, lat_slice, lon_slice = self._slices(row)
        ds = self._open_ds(nc_path)
        fc = self._open_frame_cache(nc_path)
        if fc is not None:
            lookup = fc["lookup"]
            x_idx = [lookup[c] for c in self.channels]
            y_idx = lookup[TARGET_CHANNEL]
            data = fc["data"]
            x = np.asarray(data[xs, :, lat_slice, lon_slice][:, x_idx, :, :], dtype=np.float32)
            y = np.asarray(data[ys, y_idx, lat_slice, lon_slice], dtype=np.float32)
            mask = np.isfinite(y)
            if fc["mask"] is not None:
                mask &= np.asarray(fc["mask"][ys, lat_slice, lon_slice]) > 0
        else:
            lookup = _channel_lookup(ds)
            missing = [c for c in self.channels + [TARGET_CHANNEL] if c not in lookup]
            if missing:
                raise KeyError(f"Missing channels in {nc_path}: {missing}")
            x_idx = [lookup[c] for c in self.channels]
            y_idx = lookup[TARGET_CHANNEL]
            x = np.asarray(
                ds["data"].isel(time=xs, channel=x_idx, latitude=lat_slice, longitude=lon_slice).values,
                dtype=np.float32,
            )
            y = np.asarray(
                ds["data"].isel(time=ys, channel=y_idx, latitude=lat_slice, longitude=lon_slice).values,
                dtype=np.float32,
            )
            mask = np.isfinite(y)
            if "valid_mask" in ds:
                mask &= np.asarray(ds["valid_mask"].isel(time=ys, latitude=lat_slice, longitude=lon_slice).values) > 0
        clear = self._clear_sky(row, ds, lat_slice, lon_slice)
        return x, y, mask.astype(np.uint8), clear


def make_features(
    x_raw: np.ndarray,
    clear_x: np.ndarray,
    channel_idx: list[int],
    swr_pos: int,
    min_clear_sky: float,
    ratio_clip: float,
) -> np.ndarray:
    x = np.asarray(x_raw[:, channel_idx], dtype=np.float32).copy()
    swr = x[:, swr_pos]
    swr_ratio = swr / np.maximum(clear_x, float(min_clear_sky))
    swr_ratio = np.where(clear_x >= float(min_clear_sky), swr_ratio, 0.0)
    x[:, swr_pos] = np.clip(swr_ratio, 0.0, float(ratio_clip))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def make_target_ratio(
    y_raw: np.ndarray,
    y_mask: np.ndarray,
    clear_y: np.ndarray,
    min_clear_sky: float,
    ratio_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (np.asarray(y_mask) > 0) & np.isfinite(y_raw) & np.isfinite(clear_y)
    ratio = np.asarray(y_raw, dtype=np.float32) / np.maximum(clear_y, float(min_clear_sky))
    ratio = np.clip(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(ratio_clip))
    ratio = np.where(valid, ratio, 0.0)
    return ratio.astype(np.float32), valid.astype(np.float32)


def estimate_ratio_stats(args: argparse.Namespace, channels: list[str]) -> dict[str, Any]:
    df = load_index(args.index_csv)
    train_rows = df[df["split"] == "train"].copy()
    if len(train_rows) == 0:
        raise ValueError("No train rows for ratio stats.")
    rng = np.random.default_rng(int(args.seed))
    if len(train_rows) > int(args.stats_max_samples):
        train_rows = train_rows.sample(n=int(args.stats_max_samples), random_state=int(args.seed))

    use_sample_cache = sample_cache_available(args.sample_cache_dir, args.clear_sky_dir)
    if use_sample_cache:
        meta, x_arr, y_arr, mask_arr, clear_arr = load_sample_arrays(args.sample_cache_dir, args.clear_sky_dir)
        cache_channels = [str(x) for x in meta["channels"]]
        channel_idx = [cache_channels.index(c) for c in channels]
        reader = None
    else:
        x_arr = y_arr = mask_arr = clear_arr = None
        channel_idx = list(range(len(channels)))
        reader = RatioFrameReader(channels, args.frame_cache_dir)
    swr_pos = channels.index("SWR")
    history_frames = int(train_rows["history_frames"].iloc[0]) if "history_frames" in train_rows.columns else 4
    future_frames = int(train_rows["future_frames"].iloc[0]) if "future_frames" in train_rows.columns else 2

    samples_by_channel = [[] for _ in channels]
    y_samples = []
    try:
        for _, row in train_rows.iterrows():
            gi = int(row["_index_row"])
            if use_sample_cache:
                x_raw = np.asarray(x_arr[gi], dtype=np.float32)
                y_raw = np.asarray(y_arr[gi], dtype=np.float32)
                y_mask = np.asarray(mask_arr[gi], dtype=np.uint8)
                clear = np.asarray(clear_arr[gi], dtype=np.float32)
            else:
                x_raw, y_raw, y_mask, clear = reader.read(row)
            feat = make_features(
                x_raw,
                np.asarray(clear[:history_frames], dtype=np.float32),
                channel_idx,
                swr_pos,
                float(args.min_clear_sky),
                float(args.ratio_clip),
            )
            target, valid = make_target_ratio(
                y_raw,
                y_mask,
                np.asarray(clear[history_frames:history_frames + future_frames], dtype=np.float32),
                float(args.min_clear_sky),
                float(args.ratio_clip),
            )
            flat_n = feat.shape[0] * feat.shape[2] * feat.shape[3]
            take = min(int(args.stats_pixels_per_sample), flat_n)
            picks = rng.choice(flat_n, size=take, replace=False)
            for ci in range(len(channels)):
                vals = feat[:, ci].reshape(-1)[picks]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    samples_by_channel[ci].append(vals.astype(np.float32))
            tv = target[valid > 0.5]
            if tv.size:
                y_samples.append(rng.choice(tv.reshape(-1), size=min(2048, tv.size), replace=False).astype(np.float32))
    finally:
        if reader is not None:
            reader.close()

    lows, highs, means, stds = [], [], [], []
    for parts in samples_by_channel:
        vals = np.concatenate(parts) if parts else np.asarray([0.0], dtype=np.float32)
        lo = float(np.nanpercentile(vals, 0.5))
        hi = float(np.nanpercentile(vals, 99.5))
        vals_clip = np.clip(vals, lo, hi)
        lows.append(lo)
        highs.append(hi)
        means.append(float(np.mean(vals_clip)))
        stds.append(float(max(np.std(vals_clip), 1e-6)))

    y_vals = np.concatenate(y_samples) if y_samples else np.asarray([0.0], dtype=np.float32)
    return {
        "channels": channels,
        "target": "SWR / clear_sky_SWR",
        "min_clear_sky": float(args.min_clear_sky),
        "ratio_clip": float(args.ratio_clip),
        "x_clip_low": lows,
        "x_clip_high": highs,
        "x_mean": means,
        "x_std": stds,
        "y_mean": float(np.mean(y_vals)),
        "y_std": float(max(np.std(y_vals), 1e-6)),
        "history_frames": history_frames,
        "future_frames": future_frames,
        "data_source": "sample_cache" if use_sample_cache else "frame_cache_or_netcdf",
        "num_train_rows_used": int(len(train_rows)),
        "note": "SWR input channel is converted to SWR/clear_sky; Sample_number is excluded. Other channels are robustly clipped and standardized.",
    }


class H8RatioDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        split: str,
        stats: dict[str, Any],
        index_csv: Path | None = None,
        frame_cache_dir: Path | None = None,
        sample_cache_dir: Path | None = None,
        clear_sky_dir: Path | None = None,
    ):
        self.index_csv = Path(index_csv or args.index_csv)
        self.frame_cache_dir = Path(frame_cache_dir or args.frame_cache_dir)
        self.sample_cache_dir = Path(sample_cache_dir) if sample_cache_dir is not None else None
        self.clear_sky_dir = Path(clear_sky_dir) if clear_sky_dir is not None else None
        self.df = load_index(self.index_csv, split=split)
        self.args = args
        self.stats = stats
        self.use_sample_cache = sample_cache_available(self.sample_cache_dir, self.clear_sky_dir)
        if self.use_sample_cache:
            self.meta, self.x_arr, self.y_arr, self.mask_arr, self.clear_arr = load_sample_arrays(
                self.sample_cache_dir, self.clear_sky_dir
            )
            self.cache_channels = [str(x) for x in self.meta["channels"]]
        else:
            self.meta = {}
            self.x_arr = self.y_arr = self.mask_arr = self.clear_arr = None
            self.cache_channels = [str(x) for x in stats["channels"]]
            self.reader = RatioFrameReader(
                [str(x) for x in stats["channels"]], self.frame_cache_dir
            )
        self.channels = [str(x) for x in stats["channels"]]
        self.channel_idx = [self.cache_channels.index(c) for c in self.channels]
        self.swr_pos = self.channels.index("SWR")
        self.history_frames = int(self.df["history_frames"].iloc[0]) if "history_frames" in self.df.columns else int(stats.get("history_frames", 4))
        self.future_frames = int(self.df["future_frames"].iloc[0]) if "future_frames" in self.df.columns else int(stats.get("future_frames", 2))
        self.x_low = np.asarray(stats["x_clip_low"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_high = np.asarray(stats["x_clip_high"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_mean = np.asarray(stats["x_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_std = np.asarray(stats["x_std"], dtype=np.float32).reshape(1, -1, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def close(self) -> None:
        if not self.use_sample_cache:
            self.reader.close()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        gi = int(row["_index_row"])
        if self.use_sample_cache:
            x_raw = np.asarray(self.x_arr[gi], dtype=np.float32)
            y_raw = np.asarray(self.y_arr[gi], dtype=np.float32)
            y_mask = np.asarray(self.mask_arr[gi], dtype=np.uint8)
            clear = np.asarray(self.clear_arr[gi], dtype=np.float32)
        else:
            x_raw, y_raw, y_mask, clear = self.reader.read(row)
        x = make_features(
            x_raw,
            np.asarray(clear[:self.history_frames], dtype=np.float32),
            self.channel_idx,
            self.swr_pos,
            float(self.stats["min_clear_sky"]),
            float(self.stats["ratio_clip"]),
        )
        x = np.clip(x, self.x_low, self.x_high)
        x = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        y, valid = make_target_ratio(
            y_raw,
            y_mask,
            np.asarray(clear[self.history_frames:self.history_frames + self.future_frames], dtype=np.float32),
            float(self.stats["min_clear_sky"]),
            float(self.stats["ratio_clip"]),
        )
        clear_y = np.asarray(clear[self.history_frames:self.history_frames + self.future_frames], dtype=np.float32).copy()
        raw_y = np.asarray(y_raw, dtype=np.float32).copy()
        future_times = [
            str(row[f"y_time_{j}_bjt"])
            for j in range(self.future_frames)
            if f"y_time_{j}_bjt" in row.index
        ]
        meta = {
            "index": gi,
            "issue_time_bjt": str(row["issue_time_bjt"]),
            "future_times_bjt": " -> ".join(future_times),
        }
        for j, ts in enumerate(future_times):
            meta[f"y_time_{j}_bjt"] = ts
        return {
            "x": torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "mask": torch.from_numpy(valid.astype(np.float32)),
            "clear_y": torch.from_numpy(clear_y),
            "raw_y": torch.from_numpy(raw_y),
            "meta": meta,
        }


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        groups = max(g for g in range(1, min(8, channels) + 1) if channels % g == 0)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Ratio3DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        future_frames: int,
        history_frames: int,
        width: int,
        depth: int,
        dropout: float,
        ratio_clip: float,
        temporal_head: str = "last_mean",
    ):
        super().__init__()
        self.ratio_clip = float(ratio_clip)
        self.future_frames = int(future_frames)
        self.history_frames = int(history_frames)
        self.temporal_head = str(temporal_head)
        self.stem = nn.Conv3d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[Residual3DBlock(width, dropout) for _ in range(depth)])
        if self.temporal_head == "last_mean":
            self.head = nn.Sequential(
                nn.Conv2d(width * 2, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, future_frames, 1),
            )
        elif self.temporal_head == "last_only":
            self.head = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, future_frames, 1),
            )
        elif self.temporal_head == "flatten":
            self.head = nn.Sequential(
                nn.Conv2d(width * self.history_frames, width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, future_frames, 1),
            )
        elif self.temporal_head == "conv3d_decoder":
            groups = max(g for g in range(1, min(8, width) + 1) if width % g == 0)
            self.time_proj = nn.Conv3d(
                width,
                width,
                kernel_size=(self.history_frames, 1, 1),
                padding=(0, 0, 0),
            )
            self.decoder3d = nn.Sequential(
                nn.GroupNorm(groups, width),
                nn.SiLU(inplace=True),
                nn.Conv3d(width, width, 3, padding=1),
                nn.GroupNorm(groups, width),
                nn.SiLU(inplace=True),
                nn.Dropout3d(dropout),
                nn.Conv3d(width, width, 3, padding=1),
            )
            self.future_head = nn.Conv3d(width, 1, 1)
        else:
            raise ValueError(f"Unknown temporal_head: {self.temporal_head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.blocks(self.stem(x))
        if self.temporal_head == "last_mean":
            feat = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
            out = self.head(feat)
        elif self.temporal_head == "last_only":
            out = self.head(h[:, :, -1])
        elif self.temporal_head == "flatten":
            out = self.head(h.flatten(1, 2))
        elif self.temporal_head == "conv3d_decoder":
            z = self.time_proj(h)
            z = z.repeat(1, 1, self.future_frames, 1, 1)
            z = self.decoder3d(z)
            out = self.future_head(z).squeeze(1)
        else:
            raise ValueError(f"Unknown temporal_head: {self.temporal_head}")
        return torch.sigmoid(out) * self.ratio_clip


def make_loader(ds: Dataset, args: argparse.Namespace, train: bool) -> DataLoader:
    kwargs = {}
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=train,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


def masked_ratio_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def metrics(pred_ratio: torch.Tensor, target_ratio: torch.Tensor, clear_y: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    m = mask > 0.5
    if not torch.any(m):
        return {"ratio_mae": float("nan"), "swr_mae": float("nan"), "swr_rmse": float("nan"), "swr_bias": float("nan")}
    pred_swr = pred_ratio * clear_y
    target_swr = target_ratio * clear_y
    err = pred_swr[m] - target_swr[m]
    return {
        "ratio_mae": float((pred_ratio[m] - target_ratio[m]).abs().mean().item()),
        "swr_mae": float(err.abs().mean().item()),
        "swr_rmse": float(torch.sqrt(torch.mean(err * err)).item()),
        "swr_bias": float(err.mean().item()),
    }


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    grad_clip: float,
    epoch_name: str = "",
    log_interval: int = 0,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, total_items = 0.0, 0
    vals = {"ratio_mae": [], "swr_mae": [], "swr_rmse": [], "swr_bias": []}
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        clear_y = batch["clear_y"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = masked_ratio_loss(pred, y, mask)
            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        bs = int(x.shape[0])
        total_loss += float(loss.item()) * bs
        total_items += bs
        mt = metrics(pred.detach(), y, clear_y, mask)
        for k, v in mt.items():
            if math.isfinite(v):
                vals[k].append(v)
        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            phase = epoch_name or ("train" if train else "val")
            print(
                f"{phase} batch {batch_idx}/{total_batches} | "
                f"loss {total_loss / max(total_items, 1):.4f} | "
                f"swr_mae {float(np.mean(vals['swr_mae'])) if vals['swr_mae'] else float('nan'):.2f}",
                flush=True,
            )
    out = {"loss": total_loss / max(total_items, 1)}
    out.update({k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()})
    return out


def save_checkpoint(path: Path, model, optimizer, args, stats, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "stats": stats,
            "channels": stats["channels"],
            "epoch": epoch,
            "best_val_swr_mae": best_val,
            "model_name": "ratio_3dcnn",
            "temporal_head": str(args.temporal_head),
        },
        path,
    )


def main() -> None:
    args = configure_stage1_sources(parse_args())
    seed_everything(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert_train_mode_index(args.index_csv, args.train_mode)
    assert_center_only_eval_index(args.val_index_csv)
    assert_center_only_eval_index(args.test_index_csv)
    print("Training index:", args.index_csv)
    print("Validation index (center only):", args.val_index_csv)
    print("Test index (center only):", args.test_index_csv)
    print("Temporal head:", args.temporal_head)
    print("Output directory:", args.out_dir)
    channels = list(RATIO_CHANNELS)
    stats_path = args.stats_json or default_ratio_stats_path(args.index_csv, float(args.min_clear_sky), float(args.ratio_clip))
    if stats_path.exists() and not args.force_recompute_stats:
        print("Loading ratio stats:", stats_path)
        stats = read_json(stats_path)
    else:
        print("Estimating ratio stats ->", stats_path)
        stats = estimate_ratio_stats(args, channels)
        write_json(stats, stats_path)

    train_ds = H8RatioDataset(
        args,
        "train",
        stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    val_ds = H8RatioDataset(
        args,
        "val",
        stats,
        index_csv=args.val_index_csv,
        frame_cache_dir=args.val_frame_cache_dir,
        sample_cache_dir=args.val_sample_cache_dir,
        clear_sky_dir=args.val_clear_sky_dir,
    )
    future_frames = int(train_ds.future_frames)
    args.future_frames = future_frames
    print(f"Forecast setting: 4 history -> {future_frames} future frames")
    train_loader = make_loader(train_ds, args, True)
    val_loader = make_loader(val_ds, args, False)
    print(
        f"Data: train={len(train_ds)} val={len(val_ds)} | "
        f"batch_size={args.batch_size} workers={args.num_workers} "
        f"prefetch={args.prefetch_factor} | "
        f"train_mode={args.train_mode} | "
        f"train_source={'sample_cache' if train_ds.use_sample_cache else 'frame_cache_or_netcdf'} | "
        f"val_source=center_only",
        flush=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = Ratio3DCNN(
        in_channels=len(channels),
        future_frames=future_frames,
        history_frames=int(train_ds.history_frames),
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
        ratio_clip=float(stats["ratio_clip"]),
        temporal_head=str(args.temporal_head),
    ).to(device)
    if bool(args.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    use_amp = bool(args.amp) and device.type == "cuda"
    write_json({"args": serializable_args(args), "stats": stats}, args.out_dir / "config.json")

    history_path = args.out_dir / "history.csv"
    fields = ["epoch", "train_loss", "train_swr_mae", "train_ratio_mae", "val_loss", "val_swr_mae", "val_swr_rmse", "val_ratio_mae", "val_swr_bias"]
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_val = float("inf")
    bad = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_m = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            float(args.grad_clip),
            epoch_name=f"epoch {epoch:03d} train",
            log_interval=int(args.log_interval),
        )
        with torch.no_grad():
            val_m = run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                False,
                float(args.grad_clip),
                epoch_name=f"epoch {epoch:03d} val",
                log_interval=max(int(args.log_interval), 100),
            )
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_swr_mae": train_m["swr_mae"],
            "train_ratio_mae": train_m["ratio_mae"],
            "val_loss": val_m["loss"],
            "val_swr_mae": val_m["swr_mae"],
            "val_swr_rmse": val_m["swr_rmse"],
            "val_ratio_mae": val_m["ratio_mae"],
            "val_swr_bias": val_m["swr_bias"],
        }
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"ratio3d epoch {epoch:03d} | "
            f"train loss {train_m['loss']:.4f} swr_mae {train_m['swr_mae']:.2f} | "
            f"val loss {val_m['loss']:.4f} swr_mae {val_m['swr_mae']:.2f} rmse {val_m['swr_rmse']:.2f}"
        )
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, args, stats, epoch, best_val)
        if val_m["swr_mae"] < best_val:
            best_val = val_m["swr_mae"]
            bad = 0
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, args, stats, epoch, best_val)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"Early stopping at epoch {epoch}, best val SWR MAE={best_val:.2f}")
                break
    write_json({"best_val_swr_mae": best_val}, args.out_dir / "metrics.json")


if __name__ == "__main__":
    main()
