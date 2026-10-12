#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from h8_3d_common import DEFAULT_INDEX_CSV, DEFAULT_SAMPLE_CACHE_DIR, load_index, write_json


PROCESSED_ROOT = Path(
    "/data/DayinChen/NWP/H8_processed/"
    "region_26_34_116_126p5_multivar_daytime_06_18_center_bjt"
)
DEFAULT_CLEAR_SKY_DIR = PROCESSED_ROOT / "clear_sky_sample_cache_4hist_3future"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired clear-sky/ideal SWR fields for each 4-history + multi-future H8 sample "
            "from BJT timestamp, latitude and longitude."
        )
    )
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--sample_cache_dir", type=Path, default=DEFAULT_SAMPLE_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_CLEAR_SKY_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_lat_lon(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first_nc = Path(str(index_df.iloc[0]["nc_path"]))
    ds = xr.open_dataset(first_nc)
    try:
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    finally:
        ds.close()
    return lat, lon


def solar_cos_zenith_bjt(ts: pd.Timestamp, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """NOAA-style solar geometry. Input timestamp is Beijing time UTC+8."""
    ts = pd.Timestamp(ts)
    doy = int(ts.dayofyear)
    hour_bjt = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_bjt - 12.0) / 24.0)

    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    # BJT standard meridian is 120E. Positive correction means later local solar time.
    time_offset_min = eq_time + 4.0 * (lon_deg - 120.0)
    true_solar_min = hour_bjt * 60.0 + time_offset_min
    hour_angle = np.deg2rad(true_solar_min / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat_deg)
    cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    return np.maximum(cosz.astype(np.float32), 0.0)


def haurwitz_clear_sky(cosz: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cosz, dtype=np.float32)
    mask = cosz > 1e-4
    # Haurwitz clear-sky GHI, W/m2. It depends only on solar zenith.
    out[mask] = 1098.0 * cosz[mask] * np.exp(-0.059 / np.maximum(cosz[mask], 1e-4))
    return out.astype(np.float32)


def time_keys(row: pd.Series) -> list[str]:
    history_frames = int(row["history_frames"]) if "history_frames" in row.index else 4
    future_frames = int(row["future_frames"]) if "future_frames" in row.index else 2
    keys = [f"x_time_{i}_bjt" for i in range(history_frames)]
    keys.extend(f"y_time_{i}_bjt" for i in range(future_frames))
    return keys


def row_times(row: pd.Series) -> list[pd.Timestamp]:
    keys = time_keys(row)
    return [pd.Timestamp(row[k]) for k in keys]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "clear_sky_swr_float32.npy"
    meta_path = args.out_dir / "clear_sky_meta.json"
    if out_path.exists() and meta_path.exists() and not args.overwrite:
        print("clear-sky cache already exists:", args.out_dir)
        return

    df = load_index(args.index_csv)
    first_keys = time_keys(df.iloc[0])
    lat_1d, lon_1d = load_lat_lon(df)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    n = len(df)
    shape = (n, len(first_keys), len(lat_1d), len(lon_1d))
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)

    print("index:", args.index_csv)
    print("out:", out_path)
    print("shape:", shape)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, ts in enumerate(row_times(row)):
            cosz = solar_cos_zenith_bjt(ts, lat_grid, lon_grid)
            arr[i, j] = haurwitz_clear_sky(cosz)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"cached clear-sky {i + 1}/{n}", flush=True)

    arr.flush()
    write_json(
        {
            "index_csv": str(args.index_csv),
            "sample_cache_dir": str(args.sample_cache_dir),
            "clear_sky_path": out_path.name,
            "shape": list(shape),
            "dtype": "float32",
            "time_order": first_keys,
            "history_frames": int(df.iloc[0]["history_frames"]) if "history_frames" in df.columns else 4,
            "future_frames": int(df.iloc[0]["future_frames"]) if "future_frames" in df.columns else 2,
            "formula": "Haurwitz clear-sky GHI: 1098*cosZ*exp(-0.059/cosZ), cosZ from BJT timestamp, lat, lon.",
            "lat_min": float(np.nanmin(lat_1d)),
            "lat_max": float(np.nanmax(lat_1d)),
            "lon_min": float(np.nanmin(lon_1d)),
            "lon_max": float(np.nanmax(lon_1d)),
        },
        meta_path,
    )
    print("done:", args.out_dir)


if __name__ == "__main__":
    main()
