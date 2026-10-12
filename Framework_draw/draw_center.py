#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE1_DIR = PIPELINE_DIR / "stage_1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from stage1_paths import BIG_INDEX_MAIN  # noqa: E402
from generate_real_h8_swr_svg_assets import COLOR_STOPS  # noqa: E402

BIG_ROOT = Path(
    os.environ.get("H8_BIG_PROCESSED_ROOT", STAGE1_DIR / "data" / "01_big_monthly_netcdf")
)

OUT_DIR = Path("./patch_augmentation_ppt_assets")
NETCDF_GLOB = "H8_region_multivar_*_CENTER_BJT_*.nc"
SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"

CENTER_LAT_MIN, CENTER_LAT_MAX = 26.0, 34.0
CENTER_LON_MIN, CENTER_LON_MAX = 116.0, 126.5

NINGBO_LAT_MIN, NINGBO_LAT_MAX = 29.1, 30.3
NINGBO_LON_MIN, NINGBO_LON_MAX = 121.0, 122.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--big_root", type=Path, default=BIG_ROOT)
    p.add_argument("--big_index_csv", type=Path, default=BIG_INDEX_MAIN)
    p.add_argument("--nc_path", type=Path, default=None)
    p.add_argument("--sampled_issue_times_csv", type=Path, default=SAMPLED_ISSUE_TIMES_CSV)
    p.add_argument("--issue_time_column", type=str, default="stage1_issue_time_bjt")
    p.add_argument("--sample_rank", type=int, default=1)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--time_index", type=int, default=-1)
    p.add_argument("--vmax", type=float, default=1000.0)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def choose_nc(big_root: Path) -> Path:
    files = sorted(big_root.glob(NETCDF_GLOB))
    if not files:
        raise FileNotFoundError(f"No NetCDF found under {big_root}")
    return files[len(files) // 2]


def shared_real_h8_cmap():
    stops = np.asarray([rgb for _, rgb in COLOR_STOPS], dtype=np.float32)
    x = np.asarray([pos for pos, _ in COLOR_STOPS], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    palette = np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.float32) / 255.0
    return ListedColormap(palette, name="shared_real_h8")


def channel_index(ds: xr.Dataset, name: str) -> int:
    channels = [str(x) for x in ds["channel"].values.tolist()]
    if name not in channels:
        raise KeyError(f"{name} not found in channels: {channels}")
    return channels.index(name)


def choose_time_index(ds: xr.Dataset, requested: int) -> int:
    n = int(ds.sizes["time"])

    if requested >= 0:
        return min(requested, n - 1)

    if "time_start_bjt" in ds:
        times = np.asarray(ds["time_start_bjt"].values).astype("datetime64[s]")
        hours = (
            times.astype("datetime64[h]") - times.astype("datetime64[D]")
        ).astype("timedelta64[h]").astype(int)

        candidates = np.where((hours >= 10) & (hours <= 14))[0]
        if candidates.size:
            return int(candidates[candidates.size // 2])

    return n // 2


def load_swr(nc_path: Path, time_index: int):
    ds = xr.open_dataset(nc_path)
    try:
        swr_idx = channel_index(ds, "SWR")
        ti = choose_time_index(ds, time_index)

        arr = np.asarray(
            ds["data"].isel(time=ti, channel=swr_idx).values,
            dtype=np.float32,
        )
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    finally:
        ds.close()

    return arr, lat, lon, int(ti)


def resolve_aligned_index_row(
    big_index_csv: Path,
    sampled_issue_times_csv: Path,
    issue_time_column: str,
    sample_rank: int,
) -> pd.Series | None:
    if not Path(big_index_csv).exists() or not Path(sampled_issue_times_csv).exists():
        return None
    req = pd.read_csv(sampled_issue_times_csv)
    if issue_time_column not in req.columns:
        if issue_time_column == "stage1_issue_time_bjt" and "issue_time" in req.columns:
            issue_time_column = "issue_time"
        else:
            raise KeyError(
                f"Column {issue_time_column!r} not found in {sampled_issue_times_csv}. "
                f"Available columns: {list(req.columns)}"
            )
    rank = max(1, min(int(sample_rank), len(req)))
    requested = pd.Timestamp(req.iloc[rank - 1][issue_time_column])

    df = pd.read_csv(big_index_csv)
    if "issue_time_bjt" not in df.columns:
        raise KeyError(f"{big_index_csv} lacks issue_time_bjt")
    df["issue_time_bjt"] = pd.to_datetime(df["issue_time_bjt"])
    exact = df[df["issue_time_bjt"] == requested].copy()
    if exact.empty:
        return None
    if "is_center" in exact.columns:
        center = exact[exact["is_center"].astype(int) == 1]
        if not center.empty:
            exact = center
    return exact.iloc[0].copy()


def load_swr_from_index_row(row: pd.Series):
    nc_path = Path(str(row["nc_path"]))
    issue_time = pd.Timestamp(row["issue_time_bjt"])
    ds = xr.open_dataset(nc_path)
    try:
        swr_idx = channel_index(ds, "SWR")
        ti = None
        if "time_start_bjt" in ds:
            times = pd.to_datetime(np.asarray(ds["time_start_bjt"].values))
            match = np.where(times == issue_time)[0]
            if match.size:
                ti = int(match[0])
        if ti is None:
            ti = choose_time_index(ds, -1)
        arr = np.asarray(ds["data"].isel(time=ti, channel=swr_idx).values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    finally:
        ds.close()
    return arr, lat, lon, nc_path, issue_time, int(ti)

def save_no_margin(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    plt.close(fig)

def crop_by_lonlat(arr, lat, lon, lon_min, lon_max, lat_min, lat_max):
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    if not lat_mask.any() or not lon_mask.any():
        raise ValueError("Center patch range does not overlap SWR grid.")

    return arr[np.ix_(lat_mask, lon_mask)], lat[lat_mask], lon[lon_mask]


def draw_center_swr(arr, lat, lon, out_path: Path, vmax: float, dpi: int):
    center_arr, center_lat, center_lon = crop_by_lonlat(
        arr,
        lat,
        lon,
        CENTER_LON_MIN,
        CENTER_LON_MAX,
        CENTER_LAT_MIN,
        CENTER_LAT_MAX,
    )

    extent = [
        float(center_lon.min()),
        float(center_lon.max()),
        float(center_lat.min()),
        float(center_lat.max()),
    ]

    fig, ax = plt.subplots(
        figsize=(5.8, 4.0),
        facecolor="none"
    )

    fig.patch.set_alpha(0)

    ax.imshow(
        center_arr,
        origin="upper",
        extent=extent,
        vmin=0.0,
        vmax=vmax,
        cmap=shared_real_h8_cmap(),
        aspect="auto",
    )

    # Ningbo Region 红色虚线框
    ax.add_patch(
        Rectangle(
            (NINGBO_LON_MIN, NINGBO_LAT_MIN),
            NINGBO_LON_MAX - NINGBO_LON_MIN,
            NINGBO_LAT_MAX - NINGBO_LAT_MIN,
            fill=False,
            edgecolor="#ff3030",
            linewidth=2.8,
            linestyle=(0, (2.2, 1.5)),
            alpha=0.95,
            zorder=10,
        )
    )

    ax.set_xlim(CENTER_LON_MIN, CENTER_LON_MAX)
    ax.set_ylim(CENTER_LAT_MIN, CENTER_LAT_MAX)

    # 经纬度坐标
    ax.set_xticks([116, 118, 120, 122, 124, 126.5])
    ax.set_yticks([26, 28, 30, 32, 34])

    ax.set_xticklabels(
        ["116°E", "118°E", "120°E", "122°E", "124°E", "126.5°E"],
        fontsize=10,
    )

    ax.set_yticklabels(
        ["26°N", "28°N", "30°N", "32°N", "34°N"],
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        length=3,
        width=0.8,
        pad=2,
    )

    # 坐标轴边框细一点
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # 去掉所有额外留白
    plt.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.10,
        top=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    plt.close(fig)


def draw_ningbo_zoom(arr, lat, lon, out_path: Path, vmax: float, dpi: int):
    ningbo_arr, ningbo_lat, ningbo_lon = crop_by_lonlat(
        arr,
        lat,
        lon,
        NINGBO_LON_MIN,
        NINGBO_LON_MAX,
        NINGBO_LAT_MIN,
        NINGBO_LAT_MAX,
    )

    extent = [
        float(ningbo_lon.min()),
        float(ningbo_lon.max()),
        float(ningbo_lat.min()),
        float(ningbo_lat.max()),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 4.8), facecolor="none")
    fig.patch.set_alpha(0)

    ax.imshow(
        ningbo_arr,
        origin="upper",
        extent=extent,
        vmin=0.0,
        vmax=vmax,
        cmap=shared_real_h8_cmap(),
        aspect="auto",
    )

    ax.set_xlim(NINGBO_LON_MIN, NINGBO_LON_MAX)
    ax.set_ylim(NINGBO_LAT_MIN, NINGBO_LAT_MAX)
    ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    used_issue_time = None
    used_time_index = None
    aligned_row = None
    if args.nc_path is None:
        aligned_row = resolve_aligned_index_row(
            Path(args.big_index_csv),
            Path(args.sampled_issue_times_csv),
            str(args.issue_time_column),
            int(args.sample_rank),
        )
    if aligned_row is not None:
        arr, lat, lon, nc_path, used_issue_time, used_time_index = load_swr_from_index_row(aligned_row)
    else:
        nc_path = args.nc_path or choose_nc(args.big_root)
        arr, lat, lon, used_time_index = load_swr(nc_path, args.time_index)

    out_path = args.out_dir / "center_swr_with_latlon_ticks.png"
    ningbo_zoom_path = args.out_dir / "ningbo_swr_zoom.png"

    draw_center_swr(
        arr,
        lat,
        lon,
        out_path,
        args.vmax,
        args.dpi,
    )
    draw_ningbo_zoom(
        arr,
        lat,
        lon,
        ningbo_zoom_path,
        args.vmax,
        args.dpi,
    )

    print("Using NetCDF:", nc_path)
    if used_issue_time is not None:
        print("Aligned Stage1 issue_time_bjt:", used_issue_time)
    print("saved:", out_path)
    print("saved:", ningbo_zoom_path)
    meta = {
        "nc_path": str(nc_path),
        "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
        "issue_time_column": str(args.issue_time_column),
        "sample_rank": int(args.sample_rank),
        "matched_issue_time_bjt": None if used_issue_time is None else str(used_issue_time),
        "matched_time_index": used_time_index,
        "shared_palette_source": "generate_real_h8_swr_svg_assets.COLOR_STOPS",
        "ningbo_zoom_path": str(ningbo_zoom_path),
    }
    (args.out_dir / "center_swr_with_latlon_ticks_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
