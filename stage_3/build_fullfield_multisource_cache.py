#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_fullfield_dense_temporal_forecast import build_issue_time_table, _load_split_df, _parse_time_list
from train_fullfield_dense_temporal_residual_unet_v2 import ReadOnlyDiskFrameCache, _masked_for_plot, _timestamp_key
from visualize_regional_split_frames import _bounds, ensure_dir, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults",
    "out_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_multisource_cache_issue_h8_seed2026",
    "pv_frame_cache_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_frame_cache_64_seed2026",
    "h8_cache_dir": "/data/DayinChen/NWP/H8_cache_swr_bjt",
    "train_pv_dir": "/data/DayinChen/NWP/PV/train_directory_cleaned_csv",
    "val_pv_dir": "/data/DayinChen/NWP/PV/val_directory_cleaned_csv",
    "nwp_farm_info_xlsx": "/data/DayinChen/NWP/old_NWP/farm_basic_infomation.xlsx",
    "nwp_station_map_csv": "/data/DayinChen/NWP/old_NWP/basic_info_500_with_station.csv",
    "time_start": "2022-01-01 07:00:00",
    "time_end": "2023-06-24 16:45:00",
    "train_end": "2023-01-13 23:59:59",
    "val_end": "2023-03-08 23:59:59",
    "freq_minutes": 15,
    "issue_stride_minutes": 60,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "nwp_grid_size": 16,
    "h8_frames": 4,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "nwp_idw_power": 2.0,
    "nwp_distance_eps_km": 0.2,
    "force_rebuild": 0,
    "sample_count": 8,
    "sample_split": "test",
    "dpi": 150,
}

NWP_ALIASES = {
    "radiation": ["fore_radiation", "nwp_radiation", "radiation", "SSRD", "ssrd"],
    "direction10": ["fore_direction10", "nwp_direction10", "direction10", "fore_wind_direction10"],
    "speed10": ["fore_speed10", "nwp_speed10", "speed10", "fore_wind_speed10"],
    "tcc": ["fore_tcc", "nwp_tcc", "tcc", "TCC"],
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
            "Build aligned full-field multisource caches for the fixed residual 3D U-Net V1 line: "
            "PV field cache is reused, H8/SWR is geospatially resampled from its lat/lon arrays to the "
            "PV/NWP lon-lat grid, and the 12-point old_NWP prior is cached from representative station CSV "
            "fore_* columns."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--pv_frame_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["pv_frame_cache_dir"]))
    parser.add_argument("--h8_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["h8_cache_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--nwp_farm_info_xlsx", type=Path, default=Path(PYCHARM_DEFAULTS["nwp_farm_info_xlsx"]))
    parser.add_argument("--nwp_station_map_csv", type=Path, default=Path(PYCHARM_DEFAULTS["nwp_station_map_csv"]))
    parser.add_argument("--time_start", type=str, default=str(PYCHARM_DEFAULTS["time_start"]))
    parser.add_argument("--time_end", type=str, default=str(PYCHARM_DEFAULTS["time_end"]))
    parser.add_argument("--train_end", type=str, default=str(PYCHARM_DEFAULTS["train_end"]))
    parser.add_argument("--val_end", type=str, default=str(PYCHARM_DEFAULTS["val_end"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--issue_stride_minutes", type=int, default=int(PYCHARM_DEFAULTS["issue_stride_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(PYCHARM_DEFAULTS["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(PYCHARM_DEFAULTS["future_frames"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--nwp_grid_size", type=int, default=int(PYCHARM_DEFAULTS["nwp_grid_size"]))
    parser.add_argument("--h8_frames", type=int, default=int(PYCHARM_DEFAULTS["h8_frames"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--nwp_idw_power", type=float, default=float(PYCHARM_DEFAULTS["nwp_idw_power"]))
    parser.add_argument("--nwp_distance_eps_km", type=float, default=float(PYCHARM_DEFAULTS["nwp_distance_eps_km"]))
    parser.add_argument("--force_rebuild", type=int, default=int(PYCHARM_DEFAULTS["force_rebuild"]))
    parser.add_argument("--sample_count", type=int, default=int(PYCHARM_DEFAULTS["sample_count"]))
    parser.add_argument("--sample_split", type=str, default=str(PYCHARM_DEFAULTS["sample_split"]), choices=["train", "val", "test"])
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def _issue_times(time_df: pd.DataFrame) -> list[pd.Timestamp]:
    return [pd.Timestamp(t) for t in pd.to_datetime(time_df["issue_time"]).tolist()]


def _is_source_daytime(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    return ts.hour >= 7 and (ts.hour < 16 or (ts.hour == 16 and ts.minute <= 45))


def build_multisource_issue_time_table(args: argparse.Namespace) -> pd.DataFrame:
    step = pd.Timedelta(minutes=int(args.freq_minutes))
    issue_step = pd.Timedelta(minutes=int(args.issue_stride_minutes))
    start = pd.Timestamp(args.time_start)
    end = pd.Timestamp(args.time_end)
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    rows = []
    for issue in pd.date_range(start, end, freq=issue_step):
        if int(args.issue_stride_minutes) == 60 and issue.minute != 0:
            continue
        past = [issue - step * (int(args.past_frames) - 1 - i) for i in range(int(args.past_frames))]
        future = [issue + step * (i + 1) for i in range(int(args.future_frames))]
        all_times = past + future
        if not all(_is_source_daytime(t) for t in all_times):
            continue
        if past[0].date() != issue.date() or future[-1].date() != issue.date():
            continue
        if issue <= train_end:
            split = "train"
        elif issue <= val_end:
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "issue_time": issue,
                "issue_sec": int(issue.timestamp()),
                "time_split": split,
                "past_start": past[0].strftime("%Y-%m-%d %H:%M:%S"),
                "past_end": past[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "future_start": future[0].strftime("%Y-%m-%d %H:%M:%S"),
                "future_end": future[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "past_times": "|".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in past),
                "future_times": "|".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in future),
            }
        )
    return pd.DataFrame(rows)


def _load_h8_time_to_index(h8_cache_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[pd.Timestamp, int]]:
    lat_values = np.load(h8_cache_dir / "lat_values.npy")
    lon_values = np.load(h8_cache_dir / "lon_values.npy")
    ts_df = pd.read_csv(h8_cache_dir / "timestamps.csv")
    out: dict[pd.Timestamp, int] = {}
    for _, row in ts_df.iterrows():
        ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.notna(ts):
            out[pd.Timestamp(ts)] = int(row["index"])
    return lat_values, lon_values, out


def _select_issue_h8_times(issue_time: pd.Timestamp, frame_count: int) -> list[pd.Timestamp]:
    anchor = pd.Timestamp(issue_time).floor("h")
    return [anchor - pd.Timedelta(hours=i) for i in range(int(frame_count) - 1, -1, -1)]


def _resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    in_h, in_w = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr.astype(np.float32, copy=True)
    y = np.linspace(0.0, in_h - 1.0, out_h, dtype=np.float32)
    x = np.linspace(0.0, in_w - 1.0, out_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    top = arr[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :] + arr[y0[:, None], x1[None, :]] * wx[None, :]
    bot = arr[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :] + arr[y1[:, None], x1[None, :]] * wx[None, :]
    return (top * (1.0 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)


def _h8_rectilinear_axes(lat_values: np.ndarray, lon_values: np.ndarray, frame_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D latitude/longitude axes for an H8 frame.

    The H8 cache normally stores explicit geolocation arrays.  Depending on
    the cache writer, those arrays can be either 1D rectilinear axes or 2D
    meshgrids.  The PV/NWP field grid is rectilinear, so we collapse 2D
    geolocation arrays to representative row/column axes.
    """

    h, w = int(frame_shape[0]), int(frame_shape[1])
    lat = np.asarray(lat_values, dtype=np.float64)
    lon = np.asarray(lon_values, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        lat_axis = lat
        lon_axis = lon
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != (h, w) or lon.shape != (h, w):
            if lat.size == h * w and lon.size == h * w:
                lat = lat.reshape(h, w)
                lon = lon.reshape(h, w)
            else:
                raise ValueError(
                    f"H8 lat/lon mesh shape does not match frame shape: "
                    f"lat={lat.shape}, lon={lon.shape}, frame={(h, w)}"
                )
        lat_axis = np.nanmedian(lat, axis=1)
        lon_axis = np.nanmedian(lon, axis=0)
    elif lat.size == h and lon.size == w:
        lat_axis = lat.reshape(-1)
        lon_axis = lon.reshape(-1)
    elif lat.size == h * w and lon.size == h * w:
        lat = lat.reshape(h, w)
        lon = lon.reshape(h, w)
        lat_axis = np.nanmedian(lat, axis=1)
        lon_axis = np.nanmedian(lon, axis=0)
    else:
        raise ValueError(
            f"cannot infer H8 rectilinear geolocation axes: "
            f"lat_shape={lat.shape}, lon_shape={lon.shape}, frame={(h, w)}"
        )

    lat_axis = np.asarray(lat_axis, dtype=np.float64).reshape(-1)
    lon_axis = np.asarray(lon_axis, dtype=np.float64).reshape(-1)
    if lat_axis.size != h or lon_axis.size != w:
        raise ValueError(
            f"H8 geolocation axis length mismatch: "
            f"lat_len={lat_axis.size}, lon_len={lon_axis.size}, frame={(h, w)}"
        )
    if not np.isfinite(lat_axis).all() or not np.isfinite(lon_axis).all():
        raise ValueError("H8 lat/lon axes contain non-finite values")
    return lat_axis, lon_axis


def _resample_h8_to_bounds(
    arr: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    bounds: tuple[float, float, float, float],
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Geospatially resample an H8 frame onto the PV/NWP lon-lat grid."""

    src = np.asarray(arr, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"H8 frame must be 2D, got shape={src.shape}")
    lat_axis, lon_axis = _h8_rectilinear_axes(lat_values, lon_values, src.shape)

    # Make axes ascending for np.interp; flip the data consistently.
    if lat_axis[0] > lat_axis[-1]:
        lat_axis = lat_axis[::-1]
        src = src[::-1, :]
    if lon_axis[0] > lon_axis[-1]:
        lon_axis = lon_axis[::-1]
        src = src[:, ::-1]

    lon_min, lon_max, lat_min, lat_max = bounds
    target_lons = lon_min + (np.arange(int(out_w), dtype=np.float64) + 0.5) / int(out_w) * (lon_max - lon_min)
    target_lats = lat_min + (np.arange(int(out_h), dtype=np.float64) + 0.5) / int(out_h) * (lat_max - lat_min)

    x = np.interp(target_lons, lon_axis, np.arange(lon_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    y = np.interp(target_lats, lat_axis, np.arange(lat_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    if not valid_x.any() or not valid_y.any():
        return np.full((int(out_h), int(out_w)), np.nan, dtype=np.float32)

    x0 = np.floor(np.nan_to_num(x, nan=0.0)).astype(np.int64)
    y0 = np.floor(np.nan_to_num(y, nan=0.0)).astype(np.int64)
    x0 = np.clip(x0, 0, src.shape[1] - 1)
    y0 = np.clip(y0, 0, src.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, src.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, src.shape[0] - 1)
    wx = (np.nan_to_num(x, nan=0.0) - x0).astype(np.float32)
    wy = (np.nan_to_num(y, nan=0.0) - y0).astype(np.float32)

    top = src[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :] + src[y0[:, None], x1[None, :]] * wx[None, :]
    bot = src[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :] + src[y1[:, None], x1[None, :]] * wx[None, :]
    out = top * (1.0 - wy)[:, None] + bot * wy[:, None]
    out[~valid_y, :] = np.nan
    out[:, ~valid_x] = np.nan
    return out.astype(np.float32)


def _nwp_file_lonlat(path: Path) -> tuple[float, float] | None:
    m = re.search(r"lon_(?P<lon>-?\d+p\d+)_lat_(?P<lat>-?\d+p\d+)\.csv$", path.name, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group("lon").replace("p", ".")), float(m.group("lat").replace("p", "."))


def _choose_time_col(columns: list[str]) -> str | None:
    for name in ["time", "date", "date_time", "timestamp"]:
        if name in columns:
            return name
    return None


def _choose_col(columns: list[str], aliases: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _station_csv_path(station_sn: str, train_pv_dir: Path, val_pv_dir: Path) -> Path | None:
    sn = str(station_sn)
    for base in [Path(train_pv_dir), Path(val_pv_dir)]:
        path = base / f"{sn}.csv"
        if path.exists():
            return path
    return None


def _load_nwp_point_locations(nwp_farm_info_xlsx: Path) -> pd.DataFrame:
    if not Path(nwp_farm_info_xlsx).exists():
        raise FileNotFoundError(f"NWP farm info not found: {nwp_farm_info_xlsx}")
    df = pd.read_excel(nwp_farm_info_xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = _choose_col(list(df.columns), ["NO", "no", "farm_id", "station", "station_id"])
    lon_col = _choose_col(list(df.columns), ["lon", "longitude", "Longitude"])
    lat_col = _choose_col(list(df.columns), ["lat", "latitude", "Latitude"])
    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"cannot identify NWP point id/lon/lat columns in {nwp_farm_info_xlsx}; columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "nwp_station": df[id_col].astype(str).str.strip().str.lower(),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["nwp_station", "lon", "lat"]).drop_duplicates("nwp_station").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no usable NWP point locations loaded from {nwp_farm_info_xlsx}")
    return out


def _load_station_nwp_map(nwp_station_map_csv: Path) -> pd.DataFrame:
    if not Path(nwp_station_map_csv).exists():
        raise FileNotFoundError(f"station-to-NWP map not found: {nwp_station_map_csv}")
    df = pd.read_csv(nwp_station_map_csv, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    sn_col = _choose_col(list(df.columns), ["SN", "station_sn", "sn", "StationSN"])
    nearest_col = _choose_col(
        list(df.columns),
        ["nearest_station_NO", "nearest_st", "nearest_station", "nearest", "nwp_station", "NWP"],
    )
    if sn_col is None or nearest_col is None:
        raise ValueError(f"cannot identify SN/nearest NWP columns in {nwp_station_map_csv}; columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "station_sn": df[sn_col].astype(str).str.strip(),
            "nwp_station": df[nearest_col].astype(str).str.strip().str.lower(),
        }
    )
    out = out[(out["station_sn"] != "") & (out["nwp_station"] != "")]
    return out.drop_duplicates(["station_sn", "nwp_station"]).reset_index(drop=True)


def _read_station_nwp_values(path: Path, needed_set: set[int]) -> tuple[pd.DataFrame | None, dict]:
    header = pd.read_csv(path, nrows=0)
    columns = [str(c).strip() for c in header.columns]
    time_col = _choose_time_col(columns)
    if time_col is None:
        return None, {"status": "missing_time_col", "columns": columns}
    col_map = {key: _choose_col(columns, aliases) for key, aliases in NWP_ALIASES.items()}
    if any(col_map[key] is None for key in ["radiation", "direction10", "speed10", "tcc"]):
        return None, {"status": "missing_nwp_cols", "columns": columns, "selected": col_map}
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        return None, {"status": "empty_after_time_parse", "selected": col_map}
    df["timestamp_sec"] = (df[time_col].astype("int64") // 10**9).astype(np.int64)
    df = df[df["timestamp_sec"].isin(needed_set)].copy()
    if df.empty:
        return None, {"status": "no_needed_timestamps", "selected": col_map}
    for key in ["radiation", "direction10", "speed10", "tcc"]:
        df[key] = pd.to_numeric(df[col_map[key]], errors="coerce")
    df = df[["timestamp_sec", "radiation", "direction10", "speed10", "tcc"]].drop_duplicates("timestamp_sec")
    return df, {"status": "ok", "time_col": time_col, "selected": col_map, "n_rows": int(len(df))}


def _read_nwp_points_from_station_csvs(
    nwp_farm_info_xlsx: Path,
    nwp_station_map_csv: Path,
    train_pv_dir: Path,
    val_pv_dir: Path,
    needed_times: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict]:
    needed_sec = np.asarray([_timestamp_key(t) for t in needed_times], dtype=np.int64)
    needed_set = set(int(v) for v in needed_sec)
    points = _load_nwp_point_locations(nwp_farm_info_xlsx)
    station_map = _load_station_nwp_map(nwp_station_map_csv)
    values_by_time = {int(s): [] for s in needed_sec}
    point_rows = []
    selected_sources = {}

    for point_idx, point in points.reset_index(drop=True).iterrows():
        nwp_station = str(point["nwp_station"])
        candidates = station_map[station_map["nwp_station"] == nwp_station]["station_sn"].astype(str).tolist()
        chosen = None
        chosen_df = None
        chosen_meta = None
        for sn in candidates:
            path = _station_csv_path(sn, train_pv_dir, val_pv_dir)
            if path is None:
                continue
            df, meta = _read_station_nwp_values(path, needed_set)
            if df is None:
                continue
            chosen = {"station_sn": sn, "csv_path": str(path)}
            chosen_df = df
            chosen_meta = meta
            break
        point_rows.append(
            {
                "point_idx": int(point_idx),
                "nwp_station": nwp_station,
                "lon": float(point["lon"]),
                "lat": float(point["lat"]),
                "representative_station_sn": "" if chosen is None else chosen["station_sn"],
                "representative_csv_path": "" if chosen is None else chosen["csv_path"],
                "n_candidate_stations": int(len(candidates)),
            }
        )
        selected_sources[nwp_station] = {
            "candidate_count": int(len(candidates)),
            "representative": chosen,
            "read_meta": chosen_meta,
        }
        if chosen_df is None:
            print(f"[multisource_cache] warning: no representative station CSV found for {nwp_station}", flush=True)
            continue
        for _, row in chosen_df.iterrows():
            vals = [float(row[key]) if pd.notna(row[key]) else np.nan for key in ["radiation", "direction10", "speed10", "tcc"]]
            values_by_time[int(row["timestamp_sec"])].append((int(point_idx), vals))

    point_df = pd.DataFrame(point_rows)
    if point_df.empty:
        raise RuntimeError("no NWP points loaded from old_NWP metadata")
    by_time_arr: dict[int, np.ndarray] = {}
    for sec in needed_sec:
        arr = np.full((len(point_df), 4), np.nan, dtype=np.float32)
        for point_idx, vals in values_by_time[int(sec)]:
            arr[int(point_idx)] = np.asarray(vals, dtype=np.float32)
        by_time_arr[int(sec)] = arr
    meta = {
        "source": "old_NWP farm point coordinates + representative station CSV fore_* columns",
        "nwp_farm_info_xlsx": str(nwp_farm_info_xlsx),
        "nwp_station_map_csv": str(nwp_station_map_csv),
        "nwp_variables": ["radiation", "direction10", "speed10", "tcc"],
        "selected_sources_by_nwp_station": selected_sources,
    }
    return point_df, by_time_arr, meta


def _grid_lonlat(bounds: tuple[float, float, float, float], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bounds
    xs = lon_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lon_max - lon_min)
    ys = lat_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lat_max - lat_min)
    return np.meshgrid(xs, ys)


def _distance_weights(points: pd.DataFrame, bounds: tuple[float, float, float, float], grid_size: int, power: float, eps_km: float):
    lon_grid, lat_grid = _grid_lonlat(bounds, grid_size)
    lon0 = float(np.nanmean(points["lon"]))
    lat0 = float(np.nanmean(points["lat"]))
    km_lon = 111.32 * math.cos(math.radians(lat0))
    km_lat = 111.32
    gx = (lon_grid.reshape(-1) - lon0) * km_lon
    gy = (lat_grid.reshape(-1) - lat0) * km_lat
    px = (points["lon"].to_numpy(np.float64) - lon0) * km_lon
    py = (points["lat"].to_numpy(np.float64) - lat0) * km_lat
    dist = np.sqrt((gx[:, None] - px[None, :]) ** 2 + (gy[:, None] - py[None, :]) ** 2)
    nearest = np.min(dist, axis=1).reshape(grid_size, grid_size).astype(np.float32)
    weights = 1.0 / np.maximum(dist, float(eps_km)) ** float(power)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32), nearest


def _idw_field(point_values: np.ndarray, weights: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((point_values.shape[1], grid_size, grid_size), np.nan, dtype=np.float32)
    valid_by_var = np.zeros((point_values.shape[1],), dtype=np.uint8)
    for c in range(point_values.shape[1]):
        vals = point_values[:, c].astype(np.float32)
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        valid_by_var[c] = 1
        if ok.all():
            flat = weights @ vals
        else:
            w = weights[:, ok]
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
            flat = w @ vals[ok]
        out[c] = flat.reshape(grid_size, grid_size).astype(np.float32)
    return out, valid_by_var


def _cache_ready(out_dir: Path, args: argparse.Namespace) -> bool:
    meta_path = out_dir / "multisource_cache_meta.json"
    required = [
        out_dir / "issue_time_index.csv",
        out_dir / "h8_swr_64_float16.npy",
        out_dir / "h8_age_minutes_float16.npy",
        out_dir / "h8_frame_sec.npy",
        out_dir / "nwp_future_fields_float16.npy",
        out_dir / "nwp_future_valid_uint8.npy",
        out_dir / "nwp_nearest_distance_km_float16.npy",
        meta_path,
    ]
    if any(not p.exists() for p in required):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    checks = {
        "grid_size": int(args.grid_size),
        "nwp_grid_size": int(args.nwp_grid_size),
        "h8_frames": int(args.h8_frames),
        "future_frames": int(args.future_frames),
        "nwp_source_version": "old_nwp_station_csv_v1",
    }
    for k, v in checks.items():
        if isinstance(v, str):
            if str(meta.get(k, "")) != v:
                return False
        elif int(meta.get(k, -1)) != v:
            return False
    if int(meta.get("issue_stride_minutes", 15)) != int(args.issue_stride_minutes):
        return False
    return True


def build_or_load_cache(args: argparse.Namespace, time_df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> dict:
    out_dir = ensure_dir(args.out_dir)
    if not bool(int(args.force_rebuild)) and _cache_ready(out_dir, args):
        print(f"[multisource_cache] reuse cache: {out_dir}", flush=True)
        return {
            "h8": np.load(out_dir / "h8_swr_64_float16.npy", mmap_mode="r"),
            "h8_age": np.load(out_dir / "h8_age_minutes_float16.npy", mmap_mode="r"),
            "h8_sec": np.load(out_dir / "h8_frame_sec.npy", mmap_mode="r"),
            "nwp": np.load(out_dir / "nwp_future_fields_float16.npy", mmap_mode="r"),
            "nwp_valid": np.load(out_dir / "nwp_future_valid_uint8.npy", mmap_mode="r"),
            "nwp_dist": np.load(out_dir / "nwp_nearest_distance_km_float16.npy", mmap_mode="r"),
        }

    issues = _issue_times(time_df)
    n_issue = len(issues)
    h8_path = out_dir / "h8_swr_64_float16.npy"
    h8_age_path = out_dir / "h8_age_minutes_float16.npy"
    h8_sec_path = out_dir / "h8_frame_sec.npy"
    nwp_path = out_dir / "nwp_future_fields_float16.npy"
    nwp_valid_path = out_dir / "nwp_future_valid_uint8.npy"
    nwp_dist_path = out_dir / "nwp_nearest_distance_km_float16.npy"

    h8_arr = np.lib.format.open_memmap(h8_path, mode="w+", dtype=np.float16, shape=(n_issue, int(args.h8_frames), int(args.grid_size), int(args.grid_size)))
    h8_age = np.lib.format.open_memmap(h8_age_path, mode="w+", dtype=np.float16, shape=(n_issue,))
    h8_sec = np.lib.format.open_memmap(h8_sec_path, mode="w+", dtype=np.int64, shape=(n_issue, int(args.h8_frames)))
    nwp_arr = np.lib.format.open_memmap(
        nwp_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_issue, int(args.future_frames), 4, int(args.nwp_grid_size), int(args.nwp_grid_size)),
    )
    nwp_valid = np.lib.format.open_memmap(nwp_valid_path, mode="w+", dtype=np.uint8, shape=(n_issue, int(args.future_frames), 4))

    print("[multisource_cache] loading H8 index", flush=True)
    h8_lat_values, h8_lon_values, h8_lookup = _load_h8_time_to_index(Path(args.h8_cache_dir))
    swr = np.load(Path(args.h8_cache_dir) / "swr_3d.npy", mmap_mode="r")
    missing_h8 = 0
    h8_outside_grid = 0
    for i, issue in enumerate(issues):
        h8_times = _select_issue_h8_times(issue, int(args.h8_frames))
        anchor = h8_times[-1]
        h8_age[i] = np.float16((pd.Timestamp(issue) - pd.Timestamp(anchor)).total_seconds() / 60.0)
        for j, ts in enumerate(h8_times):
            h8_sec[i, j] = _timestamp_key(ts)
            idx = h8_lookup.get(pd.Timestamp(ts))
            if idx is None:
                missing_h8 += 1
                h8_arr[i, j] = np.nan
            else:
                geo_frame = _resample_h8_to_bounds(
                    np.asarray(swr[int(idx)], dtype=np.float32),
                    h8_lat_values,
                    h8_lon_values,
                    bounds,
                    int(args.grid_size),
                    int(args.grid_size),
                )
                if not np.isfinite(geo_frame).any():
                    h8_outside_grid += 1
                h8_arr[i, j] = geo_frame.astype(np.float16)
        if (i + 1) % 1000 == 0:
            print(f"[multisource_cache] H8 cached {i + 1}/{n_issue}", flush=True)

    future_times: list[pd.Timestamp] = []
    for _, row in time_df.iterrows():
        future_times.extend(_parse_time_list(row["future_times"]))
    unique_future = [pd.Timestamp(v, unit="s") for v in sorted(set(_timestamp_key(t) for t in future_times))]
    print(f"[multisource_cache] loading 12-point old_NWP priors for {len(unique_future)} future timestamps", flush=True)
    nwp_points, nwp_values_by_time, nwp_meta = _read_nwp_points_from_station_csvs(
        Path(args.nwp_farm_info_xlsx),
        Path(args.nwp_station_map_csv),
        Path(args.train_pv_dir),
        Path(args.val_pv_dir),
        unique_future,
    )
    nwp_points.to_csv(out_dir / "nwp_point_table.csv", index=False)
    weights, nearest_dist = _distance_weights(
        nwp_points,
        bounds,
        int(args.nwp_grid_size),
        power=float(args.nwp_idw_power),
        eps_km=float(args.nwp_distance_eps_km),
    )
    np.save(nwp_dist_path, nearest_dist.astype(np.float16))
    nwp_field_by_sec: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, ts in enumerate(unique_future, start=1):
        vals = nwp_values_by_time.get(_timestamp_key(ts))
        if vals is None:
            field = np.full((4, int(args.nwp_grid_size), int(args.nwp_grid_size)), np.nan, dtype=np.float32)
            valid = np.zeros((4,), dtype=np.uint8)
        else:
            field, valid = _idw_field(vals, weights, int(args.nwp_grid_size))
        nwp_field_by_sec[_timestamp_key(ts)] = (field, valid)
        if k % 1000 == 0:
            print(f"[multisource_cache] NWP interpolated {k}/{len(unique_future)} timestamps", flush=True)

    for i, row in time_df.iterrows():
        for h, ts in enumerate(_parse_time_list(row["future_times"])):
            field, valid = nwp_field_by_sec[_timestamp_key(ts)]
            nwp_arr[i, h] = field.astype(np.float16)
            nwp_valid[i, h] = valid.astype(np.uint8)
        if (i + 1) % 1000 == 0:
            print(f"[multisource_cache] NWP packed {i + 1}/{n_issue}", flush=True)

    h8_arr.flush()
    h8_age.flush()
    h8_sec.flush()
    nwp_arr.flush()
    nwp_valid.flush()
    time_df.to_csv(out_dir / "issue_time_index.csv", index=False)
    write_json(
        out_dir / "multisource_cache_meta.json",
        {
            "grid_size": int(args.grid_size),
            "nwp_grid_size": int(args.nwp_grid_size),
            "h8_frames": int(args.h8_frames),
            "future_frames": int(args.future_frames),
            "issue_stride_minutes": int(args.issue_stride_minutes),
            "h8_alignment": "issue_time_floor, frames=[t_floor-3h, ..., t_floor]",
            "h8_spatial_alignment": (
                "H8 SWR is geospatially resampled from H8 lat_values/lon_values "
                "onto the same PV/NWP lon-lat bounds and grid_size using bilinear interpolation."
            ),
            "h8_age_minutes": (
                "issue_time - latest_h8_time; expected value is 0 for issue_stride_minutes=60, "
                "and values can include 0,15,30,45 for issue_stride_minutes=15"
            ),
            "nwp_variables": ["radiation", "direction10", "speed10", "tcc"],
            "nwp_source_version": "old_nwp_station_csv_v1",
            "nwp_interpolation": f"IDW over the 12 old_NWP points, power={float(args.nwp_idw_power)}, eps_km={float(args.nwp_distance_eps_km)}",
            "nwp_meta": nwp_meta,
            "missing_h8_frames": int(missing_h8),
            "h8_frames_outside_target_grid": int(h8_outside_grid),
            "n_issue_times": int(n_issue),
        },
    )
    return {
        "h8": np.load(h8_path, mmap_mode="r"),
        "h8_age": np.load(h8_age_path, mmap_mode="r"),
        "h8_sec": np.load(h8_sec_path, mmap_mode="r"),
        "nwp": np.load(nwp_path, mmap_mode="r"),
        "nwp_valid": np.load(nwp_valid_path, mmap_mode="r"),
        "nwp_dist": np.load(nwp_dist_path, mmap_mode="r"),
    }


def _render_samples(args: argparse.Namespace, time_df: pd.DataFrame, bounds: tuple[float, float, float, float], pv_cache, cache: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(Path(args.out_dir) / "sample_multisource_frames")
    split_df = time_df[time_df["time_split"] == str(args.sample_split)].reset_index(drop=True)
    n = min(int(args.sample_count), len(split_df))
    if n <= 0:
        return
    indices = np.linspace(0, len(split_df) - 1, n, dtype=int)
    lon_min, lon_max, lat_min, lat_max = bounds
    pv_extent = [lon_min, lon_max, lat_min, lat_max]
    nwp_extent = pv_extent
    h8 = cache["h8"]
    h8_age = cache["h8_age"]
    nwp = cache["nwp"]
    nwp_valid = cache["nwp_valid"]
    issue_to_global = {int(sec): i for i, sec in enumerate(time_df["issue_sec"].astype(np.int64).tolist())}
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    nwp_vars = ["radiation", "direction10", "speed10", "tcc"]

    rows = []
    for sample_id, local_idx in enumerate(indices):
        row = split_df.iloc[int(local_idx)]
        issue = pd.Timestamp(row["issue_time"])
        global_idx = issue_to_global[int(row["issue_sec"])]
        latest_field, latest_mask = pv_cache.get(issue)
        h8_frames = np.asarray(h8[global_idx], dtype=np.float32)
        nwp_frames = np.asarray(nwp[global_idx], dtype=np.float32)

        fig, axes = plt.subplots(5, 5, figsize=(17, 15), squeeze=False)
        im = axes[0, 0].imshow(
            _masked_for_plot(latest_field, latest_mask),
            origin="lower",
            extent=pv_extent,
            vmin=0.0,
            vmax=max(0.3, float(np.nanquantile(latest_field[latest_mask > 0.5], 0.98)) if np.any(latest_mask > 0.5) else 1.0),
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
        )
        axes[0, 0].set_title(f"PV latest field\n{issue:%Y-%m-%d %H:%M}")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

        age_map = np.full((int(args.grid_size), int(args.grid_size)), float(h8_age[global_idx]), dtype=np.float32)
        im = axes[0, 1].imshow(age_map, origin="lower", extent=pv_extent, vmin=0.0, vmax=45.0, cmap="magma", aspect="auto")
        axes[0, 1].set_title(f"H8 latest age\n{float(h8_age[global_idx]):.0f} min")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)
        for ax in axes[0, 2:]:
            ax.axis("off")

        for j in range(int(args.h8_frames)):
            arr = h8_frames[j]
            ax = axes[1, j]
            finite = np.isfinite(arr)
            vmax = float(np.nanquantile(arr[finite], 0.98)) if finite.any() else 1.0
            im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=pv_extent, cmap=cmap, vmin=0.0, vmax=max(0.1, vmax), aspect="auto")
            h8_time = pd.Timestamp(int(cache["h8_sec"][global_idx, j]), unit="s")
            ax.set_title(f"H8/SWR frame {j + 1}\n{h8_time:%H:%M}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        axes[1, 4].axis("off")

        for r, h in enumerate([0, min(3, int(args.future_frames) - 1), int(args.future_frames) - 1], start=2):
            future_time = issue + pd.Timedelta(minutes=int(args.freq_minutes) * (h + 1))
            for c, var in enumerate(nwp_vars):
                ax = axes[r, c]
                arr = nwp_frames[h, c]
                valid = bool(nwp_valid[global_idx, h, c])
                finite = np.isfinite(arr)
                lo = float(np.nanquantile(arr[finite], 0.02)) if finite.any() else 0.0
                hi = float(np.nanquantile(arr[finite], 0.98)) if finite.any() else 1.0
                if not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
                im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=nwp_extent, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
                ax.set_title(f"NWP {var} h{h + 1}\n{future_time:%H:%M} valid={int(valid)}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            axes[r, 4].axis("off")
        for ax in axes.reshape(-1):
            if ax.axison:
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
        fig.suptitle(f"Full-field multisource sample | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
        fig.tight_layout()
        path = out_dir / f"multisource_sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}.png"
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        rows.append({"sample_id": sample_id, "issue_time": str(issue), "path": str(path)})
        print(f"[multisource_cache] saved sample {path}", flush=True)
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "sample_multisource_frames.csv", index=False)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    time_df = build_multisource_issue_time_table(args)
    time_df.to_csv(out_dir / "issue_time_index.csv", index=False)
    pv_cache = ReadOnlyDiskFrameCache(Path(args.pv_frame_cache_dir), time_df, args)
    cache = build_or_load_cache(args, time_df, bounds)
    _render_samples(args, time_df, bounds, pv_cache, cache)
    print("==== Full-field multisource cache ready ====")
    print(f"issue_times={len(time_df)}")
    print(f"H8 cache: {out_dir / 'h8_swr_64_float16.npy'}")
    print(f"NWP cache: {out_dir / 'nwp_future_fields_float16.npy'}")
    print(f"figures: {out_dir / 'sample_multisource_frames'}")


if __name__ == "__main__":
    main()

