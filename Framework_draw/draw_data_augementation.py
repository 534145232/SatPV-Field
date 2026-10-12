#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE1_DIR = PIPELINE_DIR / "stage_1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from stage1_paths import BIG_INDEX_MAIN, PATCH_BANK_CSV  # noqa: E402
from generate_real_h8_swr_svg_assets import COLOR_STOPS  # noqa: E402

BIG_ROOT = Path(
    os.environ.get("H8_BIG_PROCESSED_ROOT", STAGE1_DIR / "data" / "01_big_monthly_netcdf")
)

PATCH_BANK = PATCH_BANK_CSV
OUT_DIR = Path("./patch_augmentation_ppt_assets")
NETCDF_GLOB = "H8_region_multivar_*_CENTER_BJT_*.nc"
SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"

BIG_LAT_MIN, BIG_LAT_MAX = 18.0, 42.0
BIG_LON_MIN, BIG_LON_MAX = 105.5, 137.0

CENTER_LAT_MIN, CENTER_LAT_MAX = 26.0, 34.0
CENTER_LON_MIN, CENTER_LON_MAX = 116.0, 126.5

NINGBO_LAT_MIN, NINGBO_LAT_MAX = 29.1, 30.3
NINGBO_LON_MIN, NINGBO_LON_MAX = 121.0, 122.0

OCEAN_COLOR = "#dceeff"
LAND_COLOR = "#f1f1f1"
LAND_EDGE_COLOR = "#ffffff"

COLOR_ITEMS = [
    ("blue", "#2b6cff"),
    ("green", "#20a464"),
    ("red", "#ff3030"),
    ("orange", "#ff9f00"),
    ("purple", "#8b5cf6"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--big_root", type=Path, default=BIG_ROOT)
    p.add_argument("--big_index_csv", type=Path, default=BIG_INDEX_MAIN)
    p.add_argument("--patch_bank", type=Path, default=PATCH_BANK)
    p.add_argument("--nc_path", type=Path, default=None)
    p.add_argument("--sampled_issue_times_csv", type=Path, default=SAMPLED_ISSUE_TIMES_CSV)
    p.add_argument("--issue_time_column", type=str, default="stage1_issue_time_bjt")
    p.add_argument("--sample_rank", type=int, default=1)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--time_index", type=int, default=-1)
    p.add_argument("--vmax", type=float, default=1000.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--land_shp", type=Path, default=None)
    p.add_argument("--lakes_shp", type=Path, default=None)
    return p.parse_args()


def choose_nc(big_root: Path) -> Path:
    files = sorted(big_root.glob(NETCDF_GLOB))
    if not files:
        raise FileNotFoundError(f"No NetCDF found under {big_root}")
    return files[len(files) // 2]


def shared_real_h8_cmap():
    from matplotlib.colors import ListedColormap

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

    return arr, lat, lon


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
            history_cols = sorted(
                [col for col in row.index if re.fullmatch(r"x\d+_idx", str(col))],
                key=lambda name: int(str(name).split("_")[0][1:]),
            )
            if history_cols:
                issue_col = history_cols[min(3, len(history_cols) - 1)]
                ti = int(row[issue_col])
        if ti is None:
            ti = choose_time_index(ds, -1)
        arr = np.asarray(ds["data"].isel(time=ti, channel=swr_idx).values, dtype=np.float32)
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    finally:
        ds.close()
    return arr, lat, lon, nc_path, issue_time, int(ti)


def select_example_patches(patch_bank: Path) -> pd.DataFrame:
    df = pd.read_csv(patch_bank)

    center = df[df["is_center"].astype(int) == 1]
    if len(center) != 1:
        raise ValueError("Expected exactly one center patch.")

    c = center.iloc[0]
    cr, cc = int(c["patch_row"]), int(c["patch_col"])

    wanted = [
        (cr, cc),
        (cr - 1, cc - 1),
        (cr - 1, cc),
        (cr, cc + 1),
        (cr + 1, cc),
    ]

    sub = df[
        df.apply(
            lambda r: (int(r["patch_row"]), int(r["patch_col"])) in wanted,
            axis=1,
        )
    ].copy()

    offsets = {
        (cr, cc): (0.00, 0.00),
        (cr - 1, cc - 1): (-0.18, 0.16),
        (cr - 1, cc): (0.15, 0.22),
        (cr, cc + 1): (0.22, -0.12),
        (cr + 1, cc): (-0.12, -0.20),
    }

    sub["dx"] = sub.apply(
        lambda r: offsets[(int(r["patch_row"]), int(r["patch_col"]))][0],
        axis=1,
    )
    sub["dy"] = sub.apply(
        lambda r: offsets[(int(r["patch_row"]), int(r["patch_col"]))][1],
        axis=1,
    )

    order_map = {wanted[i]: i for i in range(len(wanted))}
    sub["vis_order"] = sub.apply(
        lambda r: order_map[(int(r["patch_row"]), int(r["patch_col"]))],
        axis=1,
    )

    sub = sub.sort_values("vis_order").reset_index(drop=True)

    sub["color_name"] = [COLOR_ITEMS[i][0] for i in range(len(sub))]
    sub["color_hex"] = [COLOR_ITEMS[i][1] for i in range(len(sub))]

    return sub

def save_no_margin(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

def ensure_natural_earth(out_dir: Path):
    try:
        import shapefile  # noqa
    except ImportError as e:
        raise ImportError("Please install pyshp first:\n\npip install pyshp\n") from e

    cache_dir = out_dir / "_natural_earth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    land_zip = cache_dir / "ne_10m_land.zip"
    land_dir = cache_dir / "ne_10m_land"
    land_shp = land_dir / "ne_10m_land.shp"

    lakes_zip = cache_dir / "ne_10m_lakes.zip"
    lakes_dir = cache_dir / "ne_10m_lakes"
    lakes_shp = lakes_dir / "ne_10m_lakes.shp"

    def download_and_unzip(url: str, zip_path: Path, extract_dir: Path):
        shp_files = list(extract_dir.glob("*.shp"))
        if shp_files:
            return shp_files[0]

        print("Downloading:", url)
        urllib.request.urlretrieve(url, zip_path)

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp found after extracting {zip_path}")

        return shp_files[0]

    land = download_and_unzip(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        land_zip,
        land_dir,
    )

    lakes = download_and_unzip(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip",
        lakes_zip,
        lakes_dir,
    )

    return land, lakes


def add_shapefile_polygons(
    ax,
    shp_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    zorder: int,
):
    import shapefile

    sf = shapefile.Reader(str(shp_path))
    poly_list = []

    for shape in sf.shapes():
        pts = np.asarray(shape.points, dtype=float)
        if pts.size == 0:
            continue

        x = pts[:, 0]
        y = pts[:, 1]

        if x.max() < lon_min or x.min() > lon_max:
            continue
        if y.max() < lat_min or y.min() > lat_max:
            continue

        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i]:parts[i + 1]]
            if len(seg) >= 3:
                poly_list.append(Polygon(seg, closed=True))

    pc = PatchCollection(
        poly_list,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_collection(pc)


def draw_map_background(
    ax,
    land_shp: Path,
    lakes_shp: Path | None,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    ax.set_facecolor(OCEAN_COLOR)

    add_shapefile_polygons(
        ax,
        land_shp,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        facecolor=LAND_COLOR,
        edgecolor=LAND_EDGE_COLOR,
        linewidth=0.35,
        zorder=1,
    )

    if lakes_shp is not None and lakes_shp.exists():
        add_shapefile_polygons(
            ax,
            lakes_shp,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            facecolor=OCEAN_COLOR,
            edgecolor=LAND_EDGE_COLOR,
            linewidth=0.25,
            zorder=2,
        )


def draw_main_map(
    patches: pd.DataFrame,
    out_path: Path,
    dpi: int,
    land_shp: Path,
    lakes_shp: Path | None,
):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.patch.set_facecolor(OCEAN_COLOR)

    draw_map_background(
        ax,
        land_shp,
        lakes_shp,
        BIG_LON_MIN,
        BIG_LON_MAX,
        BIG_LAT_MIN,
        BIG_LAT_MAX,
    )

    for _, row in patches.iterrows():
        x = float(row["lon_min"]) + float(row["dx"])
        y = float(row["lat_min"]) + float(row["dy"])
        w = float(row["lon_max"] - row["lon_min"])
        h = float(row["lat_max"] - row["lat_min"])

        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=str(row["color_hex"]),
                linewidth=2.8,
                linestyle=(0, (2.2, 1.5)),
                alpha=0.96,
                zorder=10,
            )
        )

    ax.set_xlim(BIG_LON_MIN, BIG_LON_MAX)
    ax.set_ylim(BIG_LAT_MIN, BIG_LAT_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    save_no_margin(fig, out_path, dpi)


def draw_center_patch_map(
    out_path: Path,
    dpi: int,
    land_shp: Path,
    lakes_shp: Path | None,
):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fig.patch.set_facecolor(OCEAN_COLOR)
    draw_map_background(
        ax,
        land_shp,
        lakes_shp,
        CENTER_LON_MIN,
        CENTER_LON_MAX,
        CENTER_LAT_MIN,
        CENTER_LAT_MAX,
    )

    ax.add_patch(
        Rectangle(
            (CENTER_LON_MIN, CENTER_LAT_MIN),
            CENTER_LON_MAX - CENTER_LON_MIN,
            CENTER_LAT_MAX - CENTER_LAT_MIN,
            fill=False,
            edgecolor="#9a9a9a",
            linewidth=1.0,
            linestyle="-",
            alpha=0.75,
            zorder=5,
        )
    )

    ax.add_patch(
        Rectangle(
            (NINGBO_LON_MIN, NINGBO_LAT_MIN),
            NINGBO_LON_MAX - NINGBO_LON_MIN,
            NINGBO_LAT_MAX - NINGBO_LAT_MIN,
            fill=False,
            edgecolor="#ff3030",
            linewidth=2.2,
            linestyle=(0, (2.2, 1.5)),
            zorder=10,
        )
    )

    ax.set_xlim(CENTER_LON_MIN, CENTER_LON_MAX)
    ax.set_ylim(CENTER_LAT_MIN, CENTER_LAT_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    save_no_margin(fig, out_path, dpi)


def crop_by_lonlat(arr, lat, lon, lon_min, lon_max, lat_min, lat_max):
    lat_low, lat_high = sorted([lat_min, lat_max])
    lon_low, lon_high = sorted([lon_min, lon_max])

    lat_mask = (lat >= lat_low) & (lat <= lat_high)
    lon_mask = (lon >= lon_low) & (lon <= lon_high)

    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            f"Patch range does not overlap grid: "
            f"lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}"
        )

    return arr[np.ix_(lat_mask, lon_mask)], lat[lat_mask], lon[lon_mask]


def draw_swr_crop(
    crop,
    crop_lat,
    crop_lon,
    out_path: Path,
    edge_color: str,
    vmax: float,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(3.2, 2.25))

    extent = [
        float(np.min(crop_lon)),
        float(np.max(crop_lon)),
        float(np.min(crop_lat)),
        float(np.max(crop_lat)),
    ]

    ax.imshow(
        crop,
        origin="upper",
        extent=extent,
        vmin=0.0,
        vmax=vmax,
        cmap=shared_real_h8_cmap(),
        aspect="auto",
    )

    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            fill=False,
            edgecolor=edge_color,
            linewidth=3.2,
            linestyle=(0, (2.2, 1.5)),
            zorder=10,
        )
    )

    overlap_lon_min = max(float(np.min(crop_lon)), NINGBO_LON_MIN)
    overlap_lon_max = min(float(np.max(crop_lon)), NINGBO_LON_MAX)
    overlap_lat_min = max(float(np.min(crop_lat)), NINGBO_LAT_MIN)
    overlap_lat_max = min(float(np.max(crop_lat)), NINGBO_LAT_MAX)
    if overlap_lon_min < overlap_lon_max and overlap_lat_min < overlap_lat_max:
        ax.add_patch(
            Rectangle(
                (overlap_lon_min, overlap_lat_min),
                overlap_lon_max - overlap_lon_min,
                overlap_lat_max - overlap_lat_min,
                fill=False,
                edgecolor="#ff3030",
                linewidth=2.4,
                linestyle=(0, (2.2, 1.5)),
                zorder=11,
            )
        )

    ax.set_axis_off()
    save_no_margin(fig, out_path, dpi)


def draw_layout_preview(
    main_map_path: Path,
    crop_paths: list[Path],
    center_map_path: Path,
    out_path: Path,
    dpi: int,
):
    main_img = plt.imread(main_map_path)
    center_img = plt.imread(center_map_path)
    crop_imgs = [plt.imread(p) for p in crop_paths[:3]]

    fig = plt.figure(figsize=(9.2, 5.4))

    ax_main = fig.add_axes([0.03, 0.08, 0.50, 0.84])
    ax_main.imshow(main_img)
    ax_main.set_axis_off()

    ys = [0.67, 0.39, 0.11]
    for img, y in zip(crop_imgs, ys):
        ax = fig.add_axes([0.60, y, 0.22, 0.22])
        ax.imshow(img)
        ax.set_axis_off()

    ax_center = fig.add_axes([0.82, 0.31, 0.16, 0.36])
    ax_center.imshow(center_img)
    ax_center.set_axis_off()

    save_no_margin(fig, out_path, dpi)


def save_mapping_csv(patches: pd.DataFrame, out_path: Path):
    rows = []

    for i, (_, row) in enumerate(patches.iterrows(), start=1):
        patch_id = int(row["patch_id"])
        filename = f"{i:02d}_{row['color_name']}_swr_crop_patch_id_{patch_id}.png"

        rows.append(
            {
                "order": i,
                "color_name": row["color_name"],
                "color_hex": row["color_hex"],
                "patch_id": patch_id,
                "patch_row": int(row["patch_row"]),
                "patch_col": int(row["patch_col"]),
                "is_center": int(row["is_center"]),
                "lon_min": float(row["lon_min"]),
                "lon_max": float(row["lon_max"]),
                "lat_min": float(row["lat_min"]),
                "lat_max": float(row["lat_max"]),
                "swr_file": filename,
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.land_shp is not None and args.land_shp.exists():
        land_shp = args.land_shp
        lakes_shp = args.lakes_shp
    else:
        land_shp, lakes_shp = ensure_natural_earth(args.out_dir)

    aligned_row = None
    used_issue_time = None
    used_time_index = None
    if args.nc_path is None:
        aligned_row = resolve_aligned_index_row(
            Path(args.big_index_csv),
            Path(args.sampled_issue_times_csv),
            str(args.issue_time_column),
            int(args.sample_rank),
        )
    if aligned_row is not None:
        arr, lat, lon, nc_path, used_issue_time, used_time_index = load_swr_from_index_row(aligned_row)
        print("Using aligned Stage1 issue_time_bjt:", used_issue_time)
        print("Using NetCDF:", nc_path)
    else:
        nc_path = args.nc_path or choose_nc(args.big_root)
        print("Using fallback NetCDF:", nc_path)
        arr, lat, lon = load_swr(nc_path, args.time_index)
    patches = select_example_patches(args.patch_bank)

    main_map_path = args.out_dir / "00_main_map_with_boxes.png"
    center_map_path = args.out_dir / "center_patch_map_with_ningbo_region.png"

    draw_main_map(
        patches,
        main_map_path,
        args.dpi,
        land_shp,
        lakes_shp,
    )

    draw_center_patch_map(
        center_map_path,
        args.dpi,
        land_shp,
        lakes_shp,
    )

    crop_paths = []

    for i, (_, row) in enumerate(patches.iterrows(), start=1):
        patch_id = int(row["patch_id"])
        color_name = str(row["color_name"])
        color_hex = str(row["color_hex"])

        crop, crop_lat, crop_lon = crop_by_lonlat(
            arr,
            lat,
            lon,
            float(row["lon_min"]),
            float(row["lon_max"]),
            float(row["lat_min"]),
            float(row["lat_max"]),
        )

        out = args.out_dir / f"{i:02d}_{color_name}_swr_crop_patch_id_{patch_id}.png"

        draw_swr_crop(
            crop,
            crop_lat,
            crop_lon,
            out,
            color_hex,
            args.vmax,
            args.dpi,
        )

        crop_paths.append(out)

    mapping_path = args.out_dir / "patch_color_mapping.csv"
    save_mapping_csv(patches, mapping_path)

    preview_path = args.out_dir / "99_ppt_layout_preview.png"
    draw_layout_preview(
        main_map_path,
        crop_paths,
        center_map_path,
        preview_path,
        args.dpi,
    )

    print("saved assets to:", args.out_dir)
    print("main map:", main_map_path)
    print("center map:", center_map_path)
    for p in crop_paths:
        print("swr crop:", p)
    print("mapping:", mapping_path)
    print("preview:", preview_path)
    meta = {
        "nc_path": str(nc_path),
        "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
        "issue_time_column": str(args.issue_time_column),
        "sample_rank": int(args.sample_rank),
        "matched_issue_time_bjt": None if used_issue_time is None else str(used_issue_time),
        "matched_time_index": used_time_index,
        "shared_palette_source": "generate_real_h8_swr_svg_assets.COLOR_STOPS",
    }
    (args.out_dir / "asset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
