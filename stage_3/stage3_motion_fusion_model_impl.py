#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", SCRIPT_DIR.parent)).expanduser().resolve()
WORKSPACE = PROJECT_ROOT
STAGE1_DIR = PROJECT_ROOT / "stage_1"
STAGE2_DIR = PROJECT_ROOT / "stage_2"
for p in [STAGE1_DIR, STAGE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_clear_sky_cache import haurwitz_clear_sky, solar_cos_zenith_bjt
from train_3dcnn_ratio_swr import Ratio3DCNN, RatioFrameReader, load_sample_arrays, make_features, sample_cache_available
from visualize_regional_split_frames import _bounds, ensure_dir, write_json
import stage3_base_pvnorm_field_model as f04
import stage3_paths as p3

try:
    from h8_dataset_paths import CENTER_FRAME_CACHE_DIR, CENTER_INDEX_4H3F_ALL, CENTER_PROCESSED_ROOT
except ModuleNotFoundError:
    CENTER_PROCESSED_ROOT = p3.STAGE1_CENTER_MONTHLY_DIR
    CENTER_INDEX_4H3F_ALL = p3.STAGE1_CENTER_INDEX_ALL
    CENTER_FRAME_CACHE_DIR = p3.STAGE1_CENTER_FRAME_CACHE_DIR


USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "selection_dir": str(p3.TARGET_SELECTION_DIR),
    "target_run_dir": str(p3.TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(p3.TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(p3.MULTISOURCE_CACHE_DIR),
    "out_dir": str(p3.MOTION_FUSION_RUN_DIR),
    "swr_anchor_cache_dir": str(p3.CENTER_H8_SWR_ANCHOR_CACHE_ROOT),
    "normalize_swr_anchor": 1,
    "h8_index_csv": str(CENTER_INDEX_4H3F_ALL),
    "h8_frame_cache_dir": str(CENTER_FRAME_CACHE_DIR),
    "h8_processed_root": str(CENTER_PROCESSED_ROOT),
    "h8_netcdf_glob": "H8_region_multivar_*_CENTER_BJT_*.nc",
    "h8_sample_cache_dir": "",
    "h8_clear_sky_dir": "",
    "h8_center_only": 0,
    "h8_missing_history_max_gap_hours": 2.0,
    "swr_checkpoint": str(p3.SWR_BASE_CHECKPOINT),
    "model_name": "swr_anchor_motion_fusion_pv_nwp",
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
    "swr_cache_dtype": "float16",
    "force_rebuild_swr_anchor_cache": 0,
    "pv_hidden_channels": 48,
    "anchor_channels": 48,
    "horizon_embed_channels": 8,
    "dropout": 0.05,
    "batch_size": 8,
    "epochs": 60,
    "patience": 12,
    "early_stop_min_delta": 5.0e-5,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "lr_patience": 4,
    "lr_factor": 0.5,
    "large_delta_threshold": 0.10,
    "big_change_loss_weight": 0.0,
    "delta_loss_weight": 0.2,
    "flow_smooth_loss_weight": 0.0,
    "max_flow_pixels": 3.0,
    "seed": 2026,
    "device": "cuda",
    "swr_device": "cuda",
    "num_workers": 0,
    "sample_count": 8,
    "visual_horizons": "1,4,8",
    "dpi": 150,
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
        description=(
            "Train hourly SWR-anchor guided 15-min PV_norm field forecasting. Future hourly SWR anchors are "
            "predicted by a pretrained large-range 3D-CNN ratio model, cropped/resampled to the PV field grid, "
            "and used with fixed time-weight channels, a past-SWR temporal encoder, and motion/direct gated fusion."
        )
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if key in {"h8_sample_cache_dir", "h8_clear_sky_dir"}:
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key == "swr_checkpoint" or key == "h8_processed_root":
            parser.add_argument(f"--{key}", type=Path, default=Path(value))
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def _device(name: str) -> str:
    import torch

    if str(name).lower().startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(name)


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def _ts_text(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _hour_label_key(ts: pd.Timestamp, tolerance_minutes: float = 10.0) -> int:
    """Map H8 label timestamps such as HH:00:40 to the intended hourly label."""
    ts = pd.Timestamp(ts)
    rounded = ts.round("h")
    offset_min = abs((ts - rounded).total_seconds()) / 60.0
    if offset_min > float(tolerance_minutes):
        raise ValueError(f"H8 label time {ts} is too far from an hourly label: offset={offset_min:.2f} min")
    return _timestamp_key(rounded)


def _parse_visual_horizons(text: str, future_frames: int) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if 1 <= h <= int(future_frames):
            out.append(h - 1)
    return out or [0, min(3, future_frames - 1), future_frames - 1]


def _safe_read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clear_sky_on_grid(ts: pd.Timestamp, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    # The H8 SWR product is hourly mean labelled by start time; for PV 15-min features
    # we use the actual PV timestamp as the clear-sky condition.
    cosz = solar_cos_zenith_bjt(pd.Timestamp(ts), lat_grid, lon_grid)
    return haurwitz_clear_sky(cosz).astype(np.float32)


def _swr_cache_signature(args: argparse.Namespace) -> str:
    return "|".join(
        [
            str(Path(args.swr_checkpoint)),
            str(Path(args.h8_index_csv)),
            str(Path(args.h8_frame_cache_dir)),
            str(Path(args.h8_processed_root)),
            str(args.h8_netcdf_glob),
            str(args.h8_sample_cache_dir),
            str(args.h8_clear_sky_dir),
            f"center_only={int(getattr(args, 'h8_center_only', 1))}",
            f"missing_history_gap={float(getattr(args, 'h8_missing_history_max_gap_hours', 2.0))}",
            f"history={int(args.swr_history_frames)}",
            f"future={int(args.swr_future_anchors)}",
            f"grid={int(args.grid_size)}",
        ]
    )


def _resample_to_grid(arr: np.ndarray, src_lon: np.ndarray, src_lat: np.ndarray, dst_lon: np.ndarray, dst_lat: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from regular H8 center grid to the PV lon-lat grid."""
    arr = np.asarray(arr, dtype=np.float32)
    lon = np.asarray(src_lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(src_lat, dtype=np.float64).reshape(-1)
    if arr.shape[-2] != len(lat) or arr.shape[-1] != len(lon):
        raise ValueError(
            "SWR source grid shape does not match coordinate axes: "
            f"arr={arr.shape}, lat={len(lat)}, lon={len(lon)}. "
            "For bigpatch H8 indexes, the coordinate axes must be sliced by "
            "lat_start_idx/lat_end_idx/lon_start_idx/lon_end_idx before resampling."
        )
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[..., ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr = arr[..., ::-1, :]
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    if dst_lon.ndim == 1 and dst_lat.ndim == 1:
        dst_lon, dst_lat = np.meshgrid(dst_lon, dst_lat)
    else:
        dst_lon, dst_lat = np.broadcast_arrays(dst_lon, dst_lat)
    x = np.interp(dst_lon, lon, np.arange(len(lon), dtype=np.float64))
    y = np.interp(dst_lat, lat, np.arange(len(lat), dtype=np.float64))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.clip(x0, 0, len(lon) - 2)
    y0 = np.clip(y0, 0, len(lat) - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = arr[..., y0, x0]
    v01 = arr[..., y0, x1]
    v10 = arr[..., y1, x0]
    v11 = arr[..., y1, x1]
    return ((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)).astype(np.float32)


class SWRAnchorProvider:
    def __init__(self, args: argparse.Namespace, pv_lon: np.ndarray, pv_lat: np.ndarray, device: str):
        import torch
        import xarray as xr

        self.args = args
        self.device = _device(device)
        self.pv_lon = np.asarray(pv_lon, dtype=np.float32)
        self.pv_lat = np.asarray(pv_lat, dtype=np.float32)
        self.ckpt = torch.load(Path(args.swr_checkpoint), map_location="cpu")
        if "stats" not in self.ckpt:
            raise ValueError(
                "Direct on-the-fly SWR-anchor generation only accepts a bare Ratio3DCNN checkpoint. "
                "The Stage 3 default residual ConvLSTM correction model must be cached first with "
                "03_build_center_h8_swr_anchor_cache.py, then consumed through --swr_anchor_cache_dir."
            )
        self.stats = self.ckpt["stats"]
        self.channels = [str(x) for x in self.stats["channels"]]
        self.swr_pos = self.channels.index("SWR")
        sample_cache_dir = Path(args.h8_sample_cache_dir) if str(args.h8_sample_cache_dir).strip() else None
        clear_sky_dir = Path(args.h8_clear_sky_dir) if str(args.h8_clear_sky_dir).strip() else None
        self.use_sample_cache = (
            sample_cache_dir is not None
            and clear_sky_dir is not None
            and sample_cache_available(sample_cache_dir, clear_sky_dir)
        )
        if self.use_sample_cache:
            self.meta, self.x_arr, self.y_arr, self.mask_arr, self.clear_arr = load_sample_arrays(sample_cache_dir, clear_sky_dir)
            self.cache_channels = [str(x) for x in self.meta["channels"]]
            self.reader = None
        else:
            self.meta = {}
            self.x_arr = self.y_arr = self.mask_arr = self.clear_arr = None
            self.cache_channels = list(self.channels)
            self.reader = RatioFrameReader(self.channels, Path(args.h8_frame_cache_dir))
        # The PV task builds H8 windows from its own issue_time index, so even
        # when a sample cache exists we still need frame-level random access.
        if self.reader is None:
            self.reader = RatioFrameReader(self.channels, Path(args.h8_frame_cache_dir))
        self.channel_idx = [self.cache_channels.index(c) for c in self.channels]
        self.index_df = pd.read_csv(Path(args.h8_index_csv))
        self.index_df["_index_row"] = np.arange(len(self.index_df), dtype=np.int64)
        if int(getattr(args, "h8_center_only", 1)):
            before = len(self.index_df)
            if "is_center" in self.index_df.columns:
                self.index_df = self.index_df[self.index_df["is_center"].astype(int) == 1].copy()
            elif "patch_name" in self.index_df.columns:
                s = self.index_df["patch_name"].astype(str).str.lower()
                self.index_df = self.index_df[s.isin(["center", "centre"])].copy()
            elif "patch_id" in self.index_df.columns:
                s = self.index_df["patch_id"].astype(str).str.lower()
                center_like = s.isin(["center", "centre", "c"])
                if center_like.any():
                    self.index_df = self.index_df[center_like].copy()
            if len(self.index_df) == 0:
                raise ValueError(
                    f"--h8_center_only filtered all rows from {args.h8_index_csv}. "
                    "Check the H8 big-patch index center-patch columns."
                )
            if len(self.index_df) != before:
                print(f"[swr_anchor] filtered H8 index to center patch rows: {before} -> {len(self.index_df)}", flush=True)
            self.index_df = self.index_df.reset_index(drop=True)
        self.index_df["issue_time_bjt"] = pd.to_datetime(self.index_df["issue_time_bjt"])
        for j in range(int(args.swr_future_anchors)):
            col = f"y_time_{j}_bjt"
            if col in self.index_df.columns:
                self.index_df[col] = pd.to_datetime(self.index_df[col])
        for j in range(int(args.swr_history_frames)):
            col = f"x_time_{j}_bjt"
            if col in self.index_df.columns:
                self.index_df[col] = pd.to_datetime(self.index_df[col])
        self.index_df["sample_date_key"] = self.index_df["issue_time_bjt"].dt.strftime("%Y-%m-%d")
        self.patch_template = self._make_patch_template(self.index_df.iloc[0])
        self.time_lookup, self.time_range_text = self._build_time_lookup()
        self.sorted_time_keys = np.asarray(sorted(self.time_lookup.keys()), dtype=np.int64)

        with xr.open_dataset(str(self.index_df.iloc[0]["nc_path"])) as ds:
            self.full_src_lat = np.asarray(ds["latitude"].values, dtype=np.float32)
            self.full_src_lon = np.asarray(ds["longitude"].values, dtype=np.float32)

        width = int(self.ckpt["args"].get("width", 48)) if isinstance(self.ckpt.get("args"), dict) else 48
        depth = int(self.ckpt["args"].get("depth", 4)) if isinstance(self.ckpt.get("args"), dict) else 4
        dropout = float(self.ckpt["args"].get("dropout", 0.05)) if isinstance(self.ckpt.get("args"), dict) else 0.05
        self.model = Ratio3DCNN(
            in_channels=len(self.channels),
            future_frames=int(self.stats.get("future_frames", args.swr_future_anchors)),
            width=width,
            depth=depth,
            dropout=dropout,
            ratio_clip=float(self.stats["ratio_clip"]),
        )
        state = self.ckpt.get("model") or self.ckpt.get("model_state")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.x_low = np.asarray(self.stats["x_clip_low"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_high = np.asarray(self.stats["x_clip_high"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_mean = np.asarray(self.stats["x_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_std = np.asarray(self.stats["x_std"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.min_clear_sky = float(self.stats["min_clear_sky"])
        self.ratio_clip = float(self.stats["ratio_clip"])

    @staticmethod
    def _make_patch_template(row: pd.Series) -> dict[str, Any]:
        keys = [
            "lat_start_idx",
            "lat_end_idx",
            "lon_start_idx",
            "lon_end_idx",
            "patch_id",
            "is_center",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "patch_height",
            "patch_width",
        ]
        return {k: row[k] for k in keys if k in row.index}

    def _build_time_lookup(self) -> tuple[dict[int, dict[str, Any]], str]:
        import xarray as xr

        lookup: dict[int, dict[str, Any]] = {}
        processed_root = Path(getattr(self.args, "h8_processed_root", Path(self.args.h8_frame_cache_dir).parent))
        nc_glob = str(getattr(self.args, "h8_netcdf_glob", "H8_region_multivar_*_CENTER_BJT_*.nc"))
        nc_paths = sorted(str(p) for p in processed_root.glob(nc_glob))
        if not nc_paths:
            # Fallback for unusual deployments where only the sample index knows
            # the processed NetCDF locations.
            nc_paths = sorted({str(p) for p in self.index_df["nc_path"].astype(str).tolist()})
        mins, maxs = [], []
        for nc_path in nc_paths:
            with xr.open_dataset(nc_path) as ds:
                if "time_start_bjt" in ds:
                    label_times = pd.to_datetime(np.asarray(ds["time_start_bjt"].values).astype("datetime64[s]"))
                else:
                    center_times = pd.to_datetime(np.asarray(ds["time"].values).astype("datetime64[s]"))
                    label_times = center_times - pd.Timedelta(minutes=30)
                if len(label_times) == 0:
                    continue
                mins.append(pd.Timestamp(label_times.min()))
                maxs.append(pd.Timestamp(label_times.max()))
                for i, ts in enumerate(label_times):
                    raw_ts = pd.Timestamp(ts)
                    key = _hour_label_key(raw_ts)
                    lookup[key] = {
                        "nc_path": nc_path,
                        "time_idx": int(i),
                        "raw_label_time": _ts_text(raw_ts),
                        "rounded_label_time": _ts_text(raw_ts.round("h")),
                    }
        if not lookup:
            raise RuntimeError(
                f"No H8 timestamps found from processed_root={processed_root}, glob={nc_glob}, "
                f"or fallback index={self.args.h8_index_csv}"
            )
        text = f"{min(mins).strftime('%Y-%m-%d %H:%M:%S')} to {max(maxs).strftime('%Y-%m-%d %H:%M:%S')}"
        print(
            f"[swr_anchor] built BigScale H8 hourly lookup: root={processed_root}, "
            f"files={len(nc_paths)}, frames={len(lookup)}, label_time={text}",
            flush=True,
        )
        return lookup, text

    def _lookup_time(self, ts: pd.Timestamp) -> dict[str, Any]:
        key = _hour_label_key(pd.Timestamp(ts))
        item = self.time_lookup.get(key)
        if item is None:
            raise KeyError(
                f"No BigScale H8 frame found for label_time={_ts_text(ts)}. "
                f"Available BigScale H8 raw label_time range: {self.time_range_text}. "
                "H8 label seconds are rounded to the nearest hour with 10-minute tolerance."
            )
        return item

    @staticmethod
    def _patch_slices(row: pd.Series) -> tuple[slice, slice]:
        if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index):
            lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
            lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
        else:
            lat_slice = slice(None)
            lon_slice = slice(None)
        return lat_slice, lon_slice

    def _read_h8_raw_frame(self, hit: dict[str, Any], row: pd.Series) -> np.ndarray:
        nc_path = str(hit["nc_path"])
        time_idx = int(hit["time_idx"])
        lat_slice, lon_slice = self._patch_slices(row)
        fc = self.reader._open_frame_cache(nc_path) if self.reader is not None else None
        if fc is not None:
            lookup = fc["lookup"]
            missing = [c for c in self.channels if c not in lookup]
            if missing:
                raise KeyError(f"Missing H8 channels in frame cache for {nc_path}: {missing}")
            idx = [lookup[c] for c in self.channels]
            frame = np.asarray(fc["data"][time_idx, :, lat_slice, lon_slice][idx], dtype=np.float32)
        else:
            ds = self.reader._open_ds(nc_path)
            names = [str(x) for x in ds["channel"].values.tolist()]
            lookup = {name: i for i, name in enumerate(names)}
            missing = [c for c in self.channels if c not in lookup]
            if missing:
                raise KeyError(f"Missing H8 channels in NetCDF for {nc_path}: {missing}")
            idx = [lookup[c] for c in self.channels]
            frame = np.asarray(
                ds["data"].isel(time=time_idx, channel=idx, latitude=lat_slice, longitude=lon_slice).values,
                dtype=np.float32,
            )
        return frame

    def _read_or_fill_history_frame(self, ts: pd.Timestamp, row: pd.Series) -> tuple[np.ndarray, str, str]:
        key = _hour_label_key(pd.Timestamp(ts))
        hit = self.time_lookup.get(key)
        if hit is not None:
            return self._read_h8_raw_frame(hit, row), "exact", str(hit.get("raw_label_time", ""))

        keys = self.sorted_time_keys
        pos = int(np.searchsorted(keys, key))
        prev_key = int(keys[pos - 1]) if pos > 0 else None
        next_key = int(keys[pos]) if pos < len(keys) else None
        max_gap = float(getattr(self.args, "h8_missing_history_max_gap_hours", 2.0)) * 3600.0

        if prev_key is not None and next_key is not None:
            prev_gap = float(key - prev_key)
            next_gap = float(next_key - key)
            if prev_gap <= max_gap and next_gap <= max_gap:
                prev_hit = self.time_lookup[prev_key]
                next_hit = self.time_lookup[next_key]
                a = self._read_h8_raw_frame(prev_hit, row)
                b = self._read_h8_raw_frame(next_hit, row)
                w = np.float32(prev_gap / max(prev_gap + next_gap, 1.0))
                raw = f"interp({prev_hit.get('raw_label_time', '')},{next_hit.get('raw_label_time', '')})"
                return ((1.0 - w) * a + w * b).astype(np.float32), "interpolated_missing_history", raw

        candidates = []
        if prev_key is not None:
            candidates.append((abs(key - prev_key), prev_key))
        if next_key is not None:
            candidates.append((abs(next_key - key), next_key))
        if candidates:
            gap, nearest_key = min(candidates)
            if float(gap) <= max_gap:
                nearest_hit = self.time_lookup[int(nearest_key)]
                return (
                    self._read_h8_raw_frame(nearest_hit, row),
                    f"nearest_missing_history_{int(round(float(gap) / 60.0))}min",
                    str(nearest_hit.get("raw_label_time", "")),
                )
        raise KeyError(
            f"No usable BigScale H8 history frame for label_time={_ts_text(ts)}. "
            f"Nearest keys around it are prev={prev_key}, next={next_key}; "
            f"max allowed gap={float(max_gap) / 3600.0:.2f}h."
        )

    def _clear_sky_for_times(self, times: list[pd.Timestamp], row: pd.Series) -> np.ndarray:
        lat_slice, lon_slice = self._patch_slices(row)
        lat = self.full_src_lat[lat_slice]
        lon = self.full_src_lon[lon_slice]
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        out = []
        for ts in times:
            cosz = solar_cos_zenith_bjt(pd.Timestamp(ts), lat_grid, lon_grid)
            out.append(haurwitz_clear_sky(cosz))
        return np.stack(out, axis=0).astype(np.float32)

    def _source_axes_for_row(self, row: pd.Series, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return lon/lat axes matching the spatial crop loaded by RatioFrameReader."""
        lat_slice, lon_slice = self._patch_slices(row)
        lat = self.full_src_lat[lat_slice]
        lon = self.full_src_lon[lon_slice]
        arr = np.asarray(arr)
        if arr.shape[-2] != len(lat) or arr.shape[-1] != len(lon):
            raise ValueError(
                "Loaded SWR patch and H8 coordinate crop are inconsistent: "
                f"arr={arr.shape}, lat={len(lat)}, lon={len(lon)}, "
                f"row_index={int(row['_index_row']) if '_index_row' in row.index else 'NA'}, "
                f"lat_idx=({row.get('lat_start_idx', 'NA')},{row.get('lat_end_idx', 'NA')}), "
                f"lon_idx=({row.get('lon_start_idx', 'NA')},{row.get('lon_end_idx', 'NA')})."
            )
        return lon, lat

    def _row_for_issue(self, issue: pd.Timestamp) -> tuple[pd.Series, str]:
        # Build the exact SWR model window required by the PV issue time rather
        # than relying on the SWR training sample index. The processed H8 time
        # coordinate is center time, while time_start_bjt is the hourly-mean
        # label. We use label times here to match the SWR checkpoint training.
        issue = pd.Timestamp(issue).floor("h")
        history_frames = int(self.stats.get("history_frames", self.args.swr_history_frames))
        future_frames = int(self.stats.get("future_frames", self.args.swr_future_anchors))
        x_times = [issue - pd.Timedelta(hours=history_frames - 1 - j) for j in range(history_frames)]
        y_times = [issue + pd.Timedelta(hours=j + 1) for j in range(future_frames)]
        row = {
            "split": "",
            "nc_path": "",
            "sample_date_bjt": issue.strftime("%Y-%m-%d"),
            "issue_time_bjt": _ts_text(issue),
            "issue_hour_bjt": int(issue.hour),
            "x_start_idx": -1,
            "x_end_idx": -1,
            "y_start_idx": -1,
            "y_end_idx": -1,
            "history_frames": history_frames,
            "future_frames": future_frames,
            "total_frames": history_frames + future_frames,
            "time_reference": "time_start_bjt_dynamic_for_pv_issue",
            "first_input_time_bjt": _ts_text(x_times[0]),
            "last_input_time_bjt": _ts_text(x_times[-1]),
            "first_label_time_bjt": _ts_text(y_times[0]),
            "last_label_time_bjt": _ts_text(y_times[-1]),
            "_index_row": -1,
        }
        row.update(self.patch_template)
        for j, ts in enumerate(x_times):
            row[f"x_time_{j}_bjt"] = _ts_text(ts)
        for j, ts in enumerate(y_times):
            row[f"y_time_{j}_bjt"] = _ts_text(ts)
        return pd.Series(row), "dynamic_bigscale_h8_window_from_pv_issue"

    def predict(self, issue: pd.Timestamp) -> tuple[np.ndarray, dict]:
        import torch

        row, align_mode = self._row_for_issue(pd.Timestamp(issue))
        history_frames = int(self.stats.get("history_frames", self.args.swr_history_frames))
        future_frames = int(self.stats.get("future_frames", self.args.swr_future_anchors))
        x_times = [pd.Timestamp(row[f"x_time_{j}_bjt"]) for j in range(history_frames)]
        y_times = [pd.Timestamp(row[f"y_time_{j}_bjt"]) for j in range(future_frames)]
        frames, frame_modes, raw_times = [], [], []
        for ts in x_times:
            frame, mode, raw = self._read_or_fill_history_frame(ts, row)
            frames.append(frame)
            frame_modes.append(mode)
            raw_times.append(raw)
        x_raw = np.stack(frames, axis=0).astype(np.float32)
        clear = self._clear_sky_for_times(x_times + y_times, row)
        feat = make_features(
            x_raw,
            np.asarray(clear[:history_frames], dtype=np.float32),
            self.channel_idx,
            self.swr_pos,
            self.min_clear_sky,
            self.ratio_clip,
        )
        past_ratio_full = np.asarray(feat[:, self.swr_pos], dtype=np.float32)
        x = np.clip(feat, self.x_low, self.x_high)
        x = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        with torch.no_grad():
            pred_ratio = self.model(torch.from_numpy(x[None]).to(self.device)).detach().cpu().numpy()[0]
        pred_ratio = np.clip(pred_ratio.astype(np.float32), 0.0, self.ratio_clip)
        clear_full = np.asarray(clear, dtype=np.float32)
        past_clear_full = clear_full[:history_frames]
        future_clear_full = clear_full[history_frames:history_frames + future_frames]

        src_lon, src_lat = self._source_axes_for_row(row, past_ratio_full)
        past_ratio = _resample_to_grid(past_ratio_full, src_lon, src_lat, self.pv_lon, self.pv_lat)
        future_ratio = _resample_to_grid(pred_ratio[: int(self.args.swr_future_anchors)], src_lon, src_lat, self.pv_lon, self.pv_lat)
        past_clear = _resample_to_grid(past_clear_full, src_lon, src_lat, self.pv_lon, self.pv_lat) / 1000.0
        future_clear = _resample_to_grid(future_clear_full[: int(self.args.swr_future_anchors)], src_lon, src_lat, self.pv_lon, self.pv_lat) / 1000.0
        anchor_feat = np.stack(
            [
                np.concatenate([past_ratio, future_ratio], axis=0),
                np.concatenate([past_clear, future_clear], axis=0),
            ],
            axis=1,
        ).astype(np.float32)
        meta = {
            "pv_issue_time": str(pd.Timestamp(issue)),
            "h8_index_row": -1,
            "h8_issue_time_bjt": str(row["issue_time_bjt"]),
            "align_mode": align_mode,
            "history_frame_modes": "|".join(frame_modes),
            "history_raw_label_times": "|".join(raw_times),
            "anchor_time_reference": "hourly SWR mean label is interpreted as center time label+30min",
        }
        for j in range(history_frames):
            if f"x_time_{j}_bjt" in row.index:
                meta[f"past_swr_label_{j}"] = str(row[f"x_time_{j}_bjt"])
        for j in range(future_frames):
            if f"y_time_{j}_bjt" in row.index:
                meta[f"future_swr_label_{j}"] = str(row[f"y_time_{j}_bjt"])
        return anchor_feat, meta


def build_or_load_swr_anchor_cache(
    provider: SWRAnchorProvider,
    time_df: pd.DataFrame,
    split: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ensure_dir(out_dir / "swr_anchor_cache")
    dtype = np.float16 if str(args.swr_cache_dtype).lower() == "float16" else np.float32
    data_path = cache_dir / f"swr_anchor_{split}_{args.grid_size}.npy"
    meta_path = cache_dir / f"swr_anchor_{split}_meta.csv"
    if data_path.exists() and meta_path.exists() and not int(args.force_rebuild_swr_anchor_cache):
        meta_df = pd.read_csv(meta_path)
        sig = _swr_cache_signature(args)
        if "swr_cache_signature" in meta_df.columns and len(meta_df) and str(meta_df["swr_cache_signature"].iloc[0]) == sig:
            return np.load(data_path, mmap_mode="r"), meta_df
        print(f"[swr_anchor_cache][{split}] source signature changed; rebuilding cache", flush=True)
    n = len(time_df)
    a = int(args.swr_history_frames) + int(args.swr_future_anchors)
    shape = (n, a, 2, int(args.grid_size), int(args.grid_size))
    arr = np.lib.format.open_memmap(data_path, mode="w+", dtype=dtype, shape=shape)
    rows = []
    sig = _swr_cache_signature(args)
    for i, (_, row) in enumerate(time_df.iterrows()):
        feat, meta = provider.predict(pd.Timestamp(row["issue_time"]))
        arr[i] = feat.astype(dtype)
        item = dict(meta)
        item["swr_cache_signature"] = sig
        item["swr_source_index_csv"] = str(Path(args.h8_index_csv))
        item["swr_source_frame_cache_dir"] = str(Path(args.h8_frame_cache_dir))
        item["swr_source_processed_root"] = str(Path(args.h8_processed_root))
        item["swr_source_netcdf_glob"] = str(args.h8_netcdf_glob)
        item["swr_checkpoint"] = str(Path(args.swr_checkpoint))
        item["h8_center_only"] = int(getattr(args, "h8_center_only", 1))
        item["dataset_idx"] = int(i)
        item["time_split"] = split
        rows.append(item)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[swr_anchor_cache][{split}] cached {i + 1}/{n}", flush=True)
    arr.flush()
    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(meta_path, index=False)
    return np.load(data_path, mmap_mode="r"), meta_df


def _load_prebuilt_swr_anchor_cache(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    cache_dir = Path(cache_dir)
    anchor_path = cache_dir / "swr_anchor_64.npy"
    meta_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    json_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.json"
    if not anchor_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing prebuilt SWR anchor cache under {cache_dir}")
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    if "issue_sec" not in meta.columns:
        meta["issue_sec"] = meta["issue_time"].map(_timestamp_key)
    info = _safe_read_json(json_path) if json_path.exists() else {}
    return np.load(anchor_path, mmap_mode="r"), meta, info


def _subset_time_df_by_swr_meta(time_df: pd.DataFrame, meta_df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    sub_meta = meta_df[meta_df["time_split"].astype(str) == str(split)].copy().reset_index(drop=True)
    if sub_meta.empty:
        raise ValueError(f"No prebuilt SWR-anchor cache rows for split={split}")
    work = time_df.copy()
    work["issue_time"] = pd.to_datetime(work["issue_time"])
    if "issue_sec" not in work.columns:
        work["issue_sec"] = work["issue_time"].map(_timestamp_key)
    by_sec = {int(row["issue_sec"]): row for _, row in work.iterrows()}
    rows = []
    cache_indices = []
    missing = []
    for _, row in sub_meta.iterrows():
        sec = int(row["issue_sec"])
        base = by_sec.get(sec)
        if base is None:
            missing.append(str(row["issue_time"]))
            continue
        item = base.to_dict()
        item["swr_cache_idx"] = int(row["cache_idx"])
        rows.append(item)
        cache_indices.append(int(row["cache_idx"]))
    if missing:
        raise KeyError(f"{len(missing)} SWR-cache issue_time rows are missing from PV issue_time_index, e.g. {missing[:5]}")
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(cache_indices, dtype=np.int64), sub_meta


def _compute_swr_anchor_stats(train_swr: np.ndarray, out_dir: Path, force: bool = False) -> dict:
    path = Path(out_dir) / "swr_anchor_norm_stats.json"
    if path.exists() and not force:
        return _safe_read_json(path)
    arr = np.asarray(train_swr, dtype=np.float32)
    # Shape [N, anchors, 2, H, W]. Normalize each anchor channel separately:
    # channel 0 = SWR ratio, channel 1 = clear-sky SWR kW/m2.
    mean = np.nanmean(arr, axis=(0, 1, 3, 4)).astype(np.float32)
    std = np.nanstd(arr, axis=(0, 1, 3, 4)).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    stats = {
        "anchor_channel_names": ["swr_ratio", "clear_sky_swr_kwm2"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "note": (
            "Computed on train split over all anchor times and grid cells. "
            "This is separate from PV field z-score and from the SWR checkpoint input normalization."
        ),
    }
    write_json(path, stats)
    return stats


class SWRAnchorPVFieldDataset:
    def __init__(self, base_dataset, swr_anchor_arr: np.ndarray, args: argparse.Namespace, swr_anchor_stats: dict | None = None):
        self.base = base_dataset
        self.swr = swr_anchor_arr
        self.args = args
        self.swr_anchor_stats = swr_anchor_stats
        self.time_df = base_dataset.time_df
        self.lon_grid = base_dataset.lon_grid
        self.lat_grid = base_dataset.lat_grid

    def __len__(self) -> int:
        return len(self.base)

    def _clear_stack(self, times: list[pd.Timestamp]) -> np.ndarray:
        return np.stack([_clear_sky_on_grid(ts, self.lon_grid, self.lat_grid) / 1000.0 for ts in times], axis=0).astype(np.float32)

    @staticmethod
    def _pv_delta_channel(x: np.ndarray) -> np.ndarray:
        pv = np.asarray(x[0], dtype=np.float32)
        mask = np.asarray(x[1], dtype=np.float32) > 0.5
        out = np.zeros_like(pv, dtype=np.float32)
        for i in range(1, pv.shape[0]):
            valid = mask[i] & mask[i - 1]
            out[i, valid] = pv[i, valid] - pv[i - 1, valid]
        return out

    def normalize_swr(self, swr: np.ndarray) -> np.ndarray:
        if not int(getattr(self.args, "normalize_swr_anchor", 1)) or self.swr_anchor_stats is None:
            return swr.astype(np.float32)
        mean = np.asarray(self.swr_anchor_stats["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(self.swr_anchor_stats["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        return ((swr.astype(np.float32) - mean) / np.maximum(std, 1.0e-6)).astype(np.float32)

    def denorm_swr(self, swr: np.ndarray) -> np.ndarray:
        if not int(getattr(self.args, "normalize_swr_anchor", 1)) or self.swr_anchor_stats is None:
            return swr.astype(np.float32)
        mean = np.asarray(self.swr_anchor_stats["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(self.swr_anchor_stats["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        return (swr.astype(np.float32) * np.maximum(std, 1.0e-6) + mean).astype(np.float32)

    def __getitem__(self, idx: int):
        x, future_feat, y, m = self.base[idx]
        row = self.time_df.iloc[int(idx)]
        past_times = f04._parse_time_list(row["past_times"])
        future_times = f04._parse_time_list(row["future_times"])
        pv_delta = self._pv_delta_channel(x)[None]
        past_clear = self._clear_stack(past_times)[None]
        future_clear = self._clear_stack(future_times)[:, None]
        x_aug = np.concatenate([x, pv_delta, past_clear], axis=0).astype(np.float32)
        future_aug = np.concatenate([future_feat, future_clear], axis=1).astype(np.float32)
        swr = self.normalize_swr(np.asarray(self.swr[int(idx)], dtype=np.float32))
        return x_aug, future_aug, swr, y.astype(np.float32), m.astype(np.float32)


def make_swr_anchor_attention_model(
    future_frames: int,
    in_channels: int,
    future_channels: int,
    anchor_channels: int,
    pv_hidden: int,
    anchor_hidden: int,
    horizon_embed_channels: int,
    dropout: float,
    swr_history_frames: int = 4,
    swr_future_anchors: int = 3,
    max_flow_pixels: float = 3.0,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def fixed_temporal_weights(n_future: int, n_anchor: int) -> np.ndarray:
        weights = np.zeros((int(n_future), int(n_anchor)), dtype=np.float32)
        if int(n_future) == 8 and int(n_anchor) >= 4:
            weights[:, :4] = np.asarray(
                [
                    [0.25, 0.75, 0.00, 0.00],
                    [0.00, 1.00, 0.00, 0.00],
                    [0.00, 0.75, 0.25, 0.00],
                    [0.00, 0.50, 0.50, 0.00],
                    [0.00, 0.25, 0.75, 0.00],
                    [0.00, 0.00, 1.00, 0.00],
                    [0.00, 0.00, 0.75, 0.25],
                    [0.00, 0.00, 0.50, 0.50],
                ],
                dtype=np.float32,
            )
            return weights
        anchor_pos = np.arange(int(n_anchor), dtype=np.float32)
        for h in range(int(n_future)):
            # Generic fallback: linearly interpolate across future anchors.
            pos = h / max(int(n_future) - 1, 1) * max(int(n_anchor) - 1, 0)
            lo = int(np.floor(pos))
            hi = min(lo + 1, int(n_anchor) - 1)
            if lo == hi:
                weights[h, lo] = 1.0
            else:
                weights[h, lo] = float(hi - pos)
                weights[h, hi] = float(pos - lo)
        return weights

    class Block3D(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class Block2D(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.future_frames = int(future_frames)
            self.anchor_channels = int(anchor_channels)
            self.swr_history_frames = int(swr_history_frames)
            self.swr_future_anchors = int(swr_future_anchors)
            self.temporal_anchor_count = 4
            self.max_flow_pixels = float(max_flow_pixels)
            self.horizon_embed = nn.Embedding(int(future_frames), int(horizon_embed_channels))
            self.register_buffer(
                "future_anchor_weights",
                torch.from_numpy(fixed_temporal_weights(int(future_frames), self.temporal_anchor_count)),
                persistent=False,
            )
            self.enc1 = Block3D(in_channels, 32)
            self.down1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
            self.enc2 = Block3D(64, 64)
            self.down2 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
            self.enc3 = Block3D(96, 96)
            self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
            self.dec2 = Block2D(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = Block2D(64, pv_hidden)
            self.future_proj = nn.Sequential(
                nn.Conv2d(future_channels, pv_hidden, 3, padding=1),
                nn.GroupNorm(4, pv_hidden),
                nn.GELU(),
            )
            self.anchor_encoder = nn.Sequential(
                nn.Conv2d(anchor_channels, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
                nn.Conv2d(anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
            )
            self.past_swr_temporal_encoder = nn.Sequential(
                nn.Conv3d(anchor_channels, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
                nn.Conv3d(anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
            )
            self.alignment_encoder = nn.Sequential(
                nn.Conv2d(pv_hidden + anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
                nn.Conv2d(anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
            )
            self.alignment_proj = nn.Sequential(
                nn.Conv2d(anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
            )
            self.swr_alignment_gate = nn.Sequential(
                nn.Conv2d(anchor_hidden + anchor_channels, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
                nn.Conv2d(anchor_hidden, anchor_channels, 1),
                nn.Sigmoid(),
            )
            anchor_mixer_in = self.temporal_anchor_count * anchor_hidden + self.temporal_anchor_count * anchor_channels + self.temporal_anchor_count
            self.anchor_mixer = nn.Sequential(
                nn.Conv2d(anchor_mixer_in, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
                nn.Conv2d(anchor_hidden, anchor_hidden, 3, padding=1),
                nn.GroupNorm(4, anchor_hidden),
                nn.GELU(),
            )
            motion_in_channels = pv_hidden * 2 + anchor_hidden * 3 + anchor_channels * 2 + horizon_embed_channels
            self.flow_head = nn.Sequential(
                nn.Conv2d(motion_in_channels, 96, 3, padding=1),
                nn.GroupNorm(4, 96),
                nn.GELU(),
                nn.Conv2d(96, 48, 3, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
                nn.Conv2d(48, 2, 1),
            )
            self.direct_decoder = nn.Sequential(
                nn.Conv2d(motion_in_channels, 96, 3, padding=1),
                nn.GroupNorm(4, 96),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
                nn.Conv2d(96, 48, 3, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
                nn.Conv2d(48, 1, 1),
            )
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(motion_in_channels + 2, 96, 3, padding=1),
                nn.GroupNorm(4, 96),
                nn.GELU(),
                nn.Conv2d(96, 48, 3, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
                nn.Conv2d(48, 1, 1),
                nn.Sigmoid(),
            )

        @staticmethod
        def collapse(x):
            return 0.7 * x[:, :, -1] + 0.3 * x.mean(dim=2)

        @staticmethod
        def _match(x, ref):
            if x.shape[-2:] != ref.shape[-2:]:
                return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            return x

        def encode_pv(self, x):
            e1_3d = self.enc1(x)
            e2_3d = self.enc2(self.down1(e1_3d))
            e3_3d = self.enc3(self.down2(e2_3d))
            e1 = self.collapse(e1_3d)
            e2 = self.collapse(e2_3d)
            e3 = self.collapse(e3_3d)
            d2 = self._match(self.up2(e3), e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self._match(self.up1(d2), e1)
            return self.dec1(torch.cat([d1, e1], dim=1))

        def encode_anchors(self, swr_anchor):
            b, a, c, h, w = swr_anchor.shape
            feat = self.anchor_encoder(swr_anchor.reshape(b * a, c, h, w))
            return feat.reshape(b, a, -1, h, w)

        def encode_past_swr_temporal(self, swr_anchor):
            past_swr = swr_anchor[:, : self.swr_history_frames]
            seq = past_swr.permute(0, 2, 1, 3, 4).contiguous()
            feat = self.past_swr_temporal_encoder(seq)
            return self.collapse(feat)

        def encode_alignment(self, pv_feat, past_swr_temporal_feat):
            return self.alignment_encoder(torch.cat([pv_feat, past_swr_temporal_feat], dim=1))

        @staticmethod
        def warp_with_flow(img, flow):
            b, _c, h, w = img.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype),
                torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype),
                indexing="ij",
            )
            base = torch.stack([xx, yy], dim=-1)[None].expand(b, -1, -1, -1)
            norm = torch.empty_like(flow)
            norm[:, 0] = flow[:, 0] * (2.0 / max(w - 1, 1))
            norm[:, 1] = flow[:, 1] * (2.0 / max(h - 1, 1))
            grid = base - norm.permute(0, 2, 3, 1)
            return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

        def forward(self, x, future_feat, swr_anchor, return_attention: bool = False):
            pv_feat = self.encode_pv(x)
            anchor_feat = self.encode_anchors(swr_anchor)
            # Temporal anchors for 15-min PV horizons are:
            # A-1 = past_swr[-2], A0 = past_swr[-1], A1 = future_swr[0],
            # A2 = future_swr[1]. The last future_swr[2] is kept in the cache
            # but not used for h1-h8 in this center-time interpolation variant.
            anchor_start = max(self.swr_history_frames - 2, 0)
            anchor_end = anchor_start + self.temporal_anchor_count
            temporal_anchor_feat = anchor_feat[:, anchor_start:anchor_end]
            temporal_anchor_map = swr_anchor[:, anchor_start:anchor_end]
            if temporal_anchor_feat.shape[1] != self.temporal_anchor_count:
                raise ValueError(
                    f"Expected {self.temporal_anchor_count} temporal SWR anchors, got {temporal_anchor_feat.shape[1]}"
                )
            past_swr_temporal_feat = self.encode_past_swr_temporal(swr_anchor)
            alignment_ctx = self.alignment_proj(self.encode_alignment(pv_feat, past_swr_temporal_feat))
            preds, fixed_weight_maps, aligned_swr_maps = [], [], []
            warped_maps, direct_maps, gate_maps, flow_maps = [], [], [], []
            b, _, h, w = pv_feat.shape
            last_pv = x[:, 0, -1][:, None]
            anchor_feat_flat = temporal_anchor_feat.reshape(b, self.temporal_anchor_count * temporal_anchor_feat.shape[2], h, w)
            anchor_map_flat = temporal_anchor_map.reshape(b, self.temporal_anchor_count * self.anchor_channels, h, w)
            for t in range(self.future_frames):
                fut = self.future_proj(future_feat[:, t])
                he = self.horizon_embed(torch.full((b,), t, dtype=torch.long, device=x.device))[:, :, None, None].expand(-1, -1, h, w)
                weights = self.future_anchor_weights[t].to(x.device, dtype=temporal_anchor_feat.dtype)
                weight_map = weights[None, :, None, None].expand(b, -1, h, w)
                anchor_ctx = self.anchor_mixer(torch.cat([anchor_feat_flat, anchor_map_flat, weight_map], dim=1))
                future_swr_map = (temporal_anchor_map * weights[None, :, None, None, None]).sum(dim=1)
                swr_gate = self.swr_alignment_gate(torch.cat([alignment_ctx, future_swr_map], dim=1))
                aligned_swr_map = future_swr_map * (0.5 + swr_gate)
                motion_in = torch.cat(
                    [
                        pv_feat,
                        fut,
                        past_swr_temporal_feat,
                        alignment_ctx,
                        anchor_ctx,
                        future_swr_map,
                        aligned_swr_map,
                        he,
                    ],
                    dim=1,
                )
                flow = self.max_flow_pixels * torch.tanh(self.flow_head(motion_in))
                warped = self.warp_with_flow(last_pv, flow)
                direct = self.direct_decoder(motion_in)
                gate = self.fusion_gate(torch.cat([motion_in, warped, direct], dim=1))
                pred = (gate * warped + (1.0 - gate) * direct).squeeze(1)
                preds.append(pred)
                if return_attention:
                    full = torch.zeros(
                        b,
                        self.swr_history_frames + self.swr_future_anchors,
                        h,
                        w,
                        dtype=temporal_anchor_feat.dtype,
                        device=x.device,
                    )
                    full[:, anchor_start:anchor_end] = (
                        weights[None, :, None, None].expand(b, -1, h, w)
                    )
                    fixed_weight_maps.append(full)
                    aligned_swr_maps.append(aligned_swr_map)
                    warped_maps.append(warped.squeeze(1))
                    direct_maps.append(direct.squeeze(1))
                    gate_maps.append(gate.squeeze(1))
                    flow_maps.append(flow)
            pred_stack = torch.stack(preds, dim=1)
            if return_attention:
                return (
                    pred_stack,
                    torch.stack(fixed_weight_maps, dim=1),
                    torch.stack(aligned_swr_maps, dim=1),
                    torch.stack(warped_maps, dim=1),
                    torch.stack(direct_maps, dim=1),
                    torch.stack(gate_maps, dim=1),
                    torch.stack(flow_maps, dim=1),
                )
            return pred_stack

    return Model()


def masked_huber(pred, target, mask):
    import torch
    import torch.nn.functional as F

    valid = mask.float()
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)


def weighted_field_loss(pred, target, mask, x, pv_stats: dict, args: argparse.Namespace):
    import torch
    import torch.nn.functional as F

    mean = float(pv_stats["pv_field_mean"])
    std = max(float(pv_stats["pv_field_std"]), 1e-6)
    last_raw = x[:, 0, -1] * std + mean
    target_raw = target * std + mean
    true_delta = torch.abs(target_raw - last_raw[:, None])
    weights = 1.0 + float(args.big_change_loss_weight) * (true_delta >= float(args.large_delta_threshold)).float()
    valid = mask.float()
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights * valid
    return loss.sum() / torch.clamp((weights * valid).sum(), min=1.0)


def motion_fusion_loss(pred, target, mask, x, args: argparse.Namespace):
    import torch
    import torch.nn.functional as F

    valid = mask.float()
    loss_abs = F.smooth_l1_loss(pred, target, reduction="none")
    last = x[:, 0, -1][:, None]
    loss_delta = F.smooth_l1_loss(pred - last, target - last, reduction="none")
    loss = loss_abs + float(args.delta_loss_weight) * loss_delta
    return (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)


def flow_smooth_loss(flow):
    import torch

    du = torch.abs(flow[..., :, 1:] - flow[..., :, :-1]).mean()
    dv = torch.abs(flow[..., 1:, :] - flow[..., :-1, :]).mean()
    return du + dv


def run_epoch(model, loader, optimizer, device: str, train: bool, args: argparse.Namespace, pv_stats: dict) -> float:
    import torch

    model.train(train)
    losses = []
    for step, (x, future_feat, swr, y, m) in enumerate(loader, start=1):
        x = x.to(device)
        future_feat = future_feat.to(device)
        swr = swr.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            if train and float(args.flow_smooth_loss_weight) > 0:
                pred, _attn, _aligned, _warped, _direct, _gate, flow = model(x, future_feat, swr, return_attention=True)
            else:
                pred = model(x, future_feat, swr)
                flow = None
            if train and float(args.big_change_loss_weight) > 0:
                loss = weighted_field_loss(pred, y, m, x, pv_stats, args)
            elif float(args.delta_loss_weight) > 0:
                loss = motion_fusion_loss(pred, y, m, x, args)
            else:
                loss = masked_huber(pred, y, m)
            if train and flow is not None:
                loss = loss + float(args.flow_smooth_loss_weight) * flow_smooth_loss(flow)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[swr_anchor_pv][{mode}] batches={step}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_model(model, loader, device: str, model_name: str, split: str, pv_stats: dict, threshold: float) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for step, (x, future_feat, swr, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device), future_feat.to(device), swr.to(device)).detach().cpu().numpy()
            rows.extend(f04._metric_rows(model_name, split, pred, y.numpy(), m.numpy(), x.numpy(), pv_stats, threshold))
    return f04._combine_metrics(rows)


def save_sample_comparisons(model, dataset, device: str, out_dir: Path, args: argparse.Namespace, pv_stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sample_dir = ensure_dir(out_dir / "sample_swr_anchor_guided_field_comparison")
    horizons = _parse_visual_horizons(args.visual_horizons, int(args.future_frames))
    n = min(int(args.sample_count), len(dataset))
    if n <= 0:
        return
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    rows = []
    for rank, idx in enumerate(indices, start=1):
        x, future_feat, swr, y, m = dataset[int(idx)]
        with torch.no_grad():
            pred_z, attn, aligned_swr, warped_z, direct_z, gate_np, flow_np = model(
                torch.from_numpy(x[None]).to(device),
                torch.from_numpy(future_feat[None]).to(device),
                torch.from_numpy(swr[None]).to(device),
                return_attention=True,
            )
        pred = f04._denorm_field(pred_z.detach().cpu().numpy()[0], pv_stats)
        truth = f04._denorm_field(y, pv_stats)
        last = f04._denorm_field(x[0, -1], pv_stats)
        mask = m
        swr_show = dataset.denorm_swr(swr)
        attn_np = attn.detach().cpu().numpy()[0]
        aligned_swr_np = aligned_swr.detach().cpu().numpy()[0, :, 0]
        swr_context = np.asarray(
            [(swr_show[:, 0] * attn_np[h].mean(axis=(1, 2))[:, None, None]).sum(axis=0) for h in range(int(args.future_frames))],
            dtype=np.float32,
        )
        vals = np.concatenate([truth[mask > 0.5], pred[mask > 0.5]])
        vmin, vmax = float(np.nanquantile(vals, 0.02)), float(np.nanquantile(vals, 0.98))
        dvals = np.concatenate([(truth - last[None])[mask > 0.5], (pred - last[None])[mask > 0.5]])
        dmax = max(0.05, float(np.nanquantile(np.abs(dvals), 0.98)))
        align_lo = float(np.nanquantile(aligned_swr_np, 0.02))
        align_hi = float(np.nanquantile(aligned_swr_np, 0.98))
        if not np.isfinite(align_lo) or not np.isfinite(align_hi) or align_hi <= align_lo:
            align_lo, align_hi = -1.0, 1.0
        fig, axes = plt.subplots(7, len(horizons), figsize=(4.0 * len(horizons), 19.0), squeeze=False)
        panels = [
            ("truth", truth, "viridis", vmin, vmax),
            ("pred", pred, "viridis", vmin, vmax),
            ("truth-latest", truth - last[None], "coolwarm", -dmax, dmax),
            ("pred-latest", pred - last[None], "coolwarm", -dmax, dmax),
            ("fixed center-time SWR context", swr_context, "viridis", 0.0, float(np.nanquantile(swr_show[:, 0], 0.98))),
            ("alignment-modulated SWR feature", aligned_swr_np, "viridis", align_lo, align_hi),
            ("fixed anchor weights", attn_np.mean(axis=(2, 3)), "bar", 0.0, 1.0),
        ]
        for c, h in enumerate(horizons):
            for r, (title, arrs, cmap, lo, hi) in enumerate(panels):
                ax = axes[r, c]
                if cmap == "bar":
                    vals_bar = arrs[h]
                    ax.bar(np.arange(len(vals_bar)), vals_bar, color="#3b6ea8")
                    ax.set_ylim(0.0, 1.0)
                    ax.set_title(f"{title} h{h + 1}")
                else:
                    show = np.asarray(arrs[h], dtype=np.float32).copy()
                    if r < 4:
                        show[mask[h] <= 0.5] = np.nan
                    im = ax.imshow(show, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                    ax.set_title(f"{title} h{h + 1}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                ax.set_xticks([])
                ax.set_yticks([])
        issue = str(dataset.time_df.iloc[int(idx)]["issue_time"])
        fig.suptitle(f"SWR-anchor guided PV field comparison | issue={issue}", y=0.995)
        fig.tight_layout()
        path = sample_dir / f"swr_anchor_case_{rank:03d}_idx_{idx}_issue_{issue.replace(':', '').replace(' ', '_')}.png"
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        rows.append({"rank": rank, "dataset_idx": int(idx), "issue_time": issue, "figure": str(path)})
    pd.DataFrame(rows).to_csv(out_dir / "sample_swr_anchor_guided_field_comparison.csv", index=False)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    import torch
    from torch.utils.data import DataLoader

    out_dir = ensure_dir(args.out_dir)
    device = _device(args.device)
    swr_device = _device(args.swr_device)
    f04._assert_target_exclusion(args)

    station_split = f04._load_station_split(args)
    bounds = _bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(args.multisource_cache_dir))
    f04._assert_issue_time_alignment(args, ms_cache["time_df"])
    time_df = pd.read_csv(Path(args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(f04._timestamp_key)
    time_df.to_csv(out_dir / "pvnorm_field_issue_time_index.csv", index=False)

    pv_cache = f04.ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, out_dir, time_df)

    prebuilt_swr_arr = None
    prebuilt_swr_meta = None
    prebuilt_swr_info = {}
    if str(args.swr_anchor_cache_dir).strip():
        prebuilt_swr_arr, prebuilt_swr_meta, prebuilt_swr_info = _load_prebuilt_swr_anchor_cache(Path(args.swr_anchor_cache_dir))
        print(
            f"[swr_anchor_pv] using prebuilt center H8/SWR-anchor cache: {args.swr_anchor_cache_dir} "
            f"rows={len(prebuilt_swr_meta)}",
            flush=True,
        )
        swr_provider = None
    else:
        lon_grid, lat_grid = f04._grid_lonlat(bounds, int(args.grid_size))
        swr_provider = SWRAnchorProvider(args, lon_grid, lat_grid, swr_device)

    base_datasets, swr_arrays, datasets, loaders, swr_meta = {}, {}, {}, {}, {}
    for split in ["train", "val", "test"]:
        if prebuilt_swr_arr is not None:
            sub, cache_indices, meta_df = _subset_time_df_by_swr_meta(time_df, prebuilt_swr_meta, split)
            swr_arr = np.asarray(prebuilt_swr_arr[cache_indices], dtype=np.float32)
        else:
            sub = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
            swr_arr, meta_df = build_or_load_swr_anchor_cache(swr_provider, sub, split, out_dir, args)
        base = f04.TargetExcludedPVNormFieldDataset(sub, pv_cache, ms_cache, bounds, pv_stats, ms_stats, str(args.base_model_name), args)
        base_datasets[split] = base
        swr_arrays[split] = swr_arr
        swr_meta[split] = meta_df

    swr_anchor_stats = None
    if int(args.normalize_swr_anchor):
        swr_anchor_stats = _compute_swr_anchor_stats(swr_arrays["train"], out_dir, force=False)
        print(
            f"[swr_anchor_pv] SWR-anchor train stats mean={swr_anchor_stats['mean']} std={swr_anchor_stats['std']}",
            flush=True,
        )

    for split in ["train", "val", "test"]:
        datasets[split] = SWRAnchorPVFieldDataset(base_datasets[split], swr_arrays[split], args, swr_anchor_stats)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
        )

    align_summary = {}
    for split, meta_df in swr_meta.items():
        align_summary[split] = meta_df["align_mode"].value_counts().to_dict() if "align_mode" in meta_df.columns else {}

    base_in, future_channels = base_datasets["train"].source_dims()
    in_channels = int(base_in) + 2  # PV delta + past clear-sky
    future_channels = int(future_channels) + 1  # future clear-sky
    model = make_swr_anchor_attention_model(
        future_frames=int(args.future_frames),
        in_channels=in_channels,
        future_channels=future_channels,
        anchor_channels=2,
        pv_hidden=int(args.pv_hidden_channels),
        anchor_hidden=int(args.anchor_channels),
        horizon_embed_channels=int(args.horizon_embed_channels),
        dropout=float(args.dropout),
        swr_history_frames=int(args.swr_history_frames),
        swr_future_anchors=int(args.swr_future_anchors),
        max_flow_pixels=float(args.max_flow_pixels),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.lr_factor),
        patience=int(args.lr_patience),
    )
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    ckpt_path = out_dir / f"{args.model_name}_best.pt"
    logs = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, loaders["train"], optimizer, device, True, args, pv_stats)
        val_loss = run_epoch(model, loaders["val"], optimizer, device, False, args, pv_stats)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - float(args.early_stop_min_delta)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "improved": bool(improved)})
        print(f"[swr_anchor_pv] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.3e}", flush=True)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "base_model_name": str(args.base_model_name),
                    "model_name": str(args.model_name),
                    "in_channels": int(in_channels),
                    "base_in_channels": int(base_in),
                    "future_channels": int(future_channels),
                    "swr_anchor_channels": 2,
                    "swr_anchor_count": int(args.swr_history_frames) + int(args.swr_future_anchors),
                    "pv_field_norm_stats": pv_stats,
                    "multisource_norm_stats": ms_stats,
                    "swr_checkpoint": str(args.swr_checkpoint),
                    "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
                    "swr_anchor_norm_stats": swr_anchor_stats,
                    "best_val_loss": float(best_val),
                    "best_epoch": int(best_epoch),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[swr_anchor_pv] early stop at epoch={epoch}, best_epoch={best_epoch}", flush=True)
                break

    pd.DataFrame(logs).to_csv(out_dir / "swr_anchor_guided_pvnorm_field_train_log.csv", index=False)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

    metric_frames = []
    for split in ["val", "test"]:
        metric_frames.append(evaluate_model(model, loaders[split], device, str(args.model_name), split, pv_stats, float(args.large_delta_threshold)))
    summary = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    summary.to_csv(out_dir / "swr_anchor_guided_pvnorm_field_eval_summary.csv", index=False)
    save_sample_comparisons(model, datasets["test"], device, out_dir, args, pv_stats)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Hourly SWR-anchor guided 15-min PV_norm field forecasting with center-time temporal anchor stack and past-SWR temporal encoding.",
            "model_name": str(args.model_name),
            "base_model_name": str(args.base_model_name),
            "swr_anchor_definition": {
                "swr_checkpoint": str(args.swr_checkpoint),
                "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
                "prebuilt_swr_anchor_cache_meta": prebuilt_swr_info,
                "swr_anchor_norm_stats": swr_anchor_stats,
                "h8_index_csv": str(args.h8_index_csv),
                "h8_processed_root": str(args.h8_processed_root),
                "h8_netcdf_glob": str(args.h8_netcdf_glob),
                "h8_sample_cache_dir": str(args.h8_sample_cache_dir),
                "h8_clear_sky_dir": str(args.h8_clear_sky_dir),
                "center_domain": "BigScale center dataset lon 116-126.5, lat 26-34, then resampled/cropped to current Ningbo PV field grid",
                "time_reference": "hourly SWR mean labelled HH:00 is interpreted as center HH:30",
                "alignment_policy": "fixed center-time assignment uses A-1=past_swr[-2], A0=past_swr[-1], A1=future_swr[0], A2=future_swr[1]. h1 uses A-1/A0 = 0.25/0.75; h2 uses A0; h3-h5 interpolate A0/A1; h6 uses A1; h7-h8 interpolate A1/A2.",
                "alignment_summary": align_summary,
                "anchor_channels": ["SWR_ratio", "clear_sky_SWR_kWm2"],
            },
            "input_channels": {
                "historical": "past PV_norm z, PV mask, time/solar, past PV delta, past clear-sky; no historical NWP. Past SWR is encoded as a 4-step low-frequency temporal sequence by a small 3D CNN.",
                "future": "future time/solar, future NWP, future clear-sky, independent SWR temporal anchors with fixed weight channels, and direct future/alignment-modulated SWR spatial maps",
                "swr_anchors": "4 hourly SWR input anchors x_time_0..3 plus 3 predicted hourly SWR anchors y_time_0..2. Decoder receives independent [x_time_2, x_time_3, y_time_0, y_time_1] anchor features/maps plus fixed temporal weight channels; y_time_2 is not used for h1-h8 in this variant.",
            },
            "architecture": "PV 3D encoder + past SWR 3D temporal encoder + learned PV/SWR alignment context + multi-anchor stack mixer with fixed temporal weight channels + SWR-guided flow head + warped/latest PV branch + direct PV branch + fusion gate",
            "pv_field_norm_stats": pv_stats,
            "multisource_norm_stats": ms_stats,
            "checkpoint": str(ckpt_path),
        },
    )
    print("==== SWR-anchor guided PV_norm field model summary ====")
    show = summary[summary["horizon"].isin(["all", "h1", "h4", "h8", "horizon_diversity"])].copy()
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    print(f"saved: {out_dir / 'swr_anchor_guided_pvnorm_field_eval_summary.csv'}")
    print(f"checkpoint: {ckpt_path}")
    print(f"figures: {out_dir / 'sample_swr_anchor_guided_field_comparison'}")


if __name__ == "__main__":
    main()
