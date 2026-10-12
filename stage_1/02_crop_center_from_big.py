"""Crop center monthly NetCDF files from the Stage 1 big monthly source."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from stage1_paths import BIG_MONTHLY_DIR, CENTER_MONTHLY_DIR, ensure_stage1_dirs


CENTER_LAT_MIN = 26.0
CENTER_LAT_MAX = 34.0
CENTER_LON_MIN = 116.0
CENTER_LON_MAX = 126.5


def coord_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    return slice(upper, lower) if values[0] > values[-1] else slice(lower, upper)


def output_path(src: Path) -> Path:
    name = src.name.replace("lat18_42_lon105.5_137", "lat26_34_lon116_126.5")
    if name == src.name:
        name = f"{src.stem}_center_lat26_34_lon116_126.5.nc"
    return CENTER_MONTHLY_DIR / name


def crop_one(src_str: str, overwrite: bool) -> tuple[str, str, tuple[int, ...]]:
    src = Path(src_str)
    dst = output_path(src)
    if dst.exists() and not overwrite:
        return "exists", str(dst), ()

    with xr.open_dataset(src) as ds:
        cropped = ds.sel(
            latitude=coord_slice(ds["latitude"].values, CENTER_LAT_MIN, CENTER_LAT_MAX),
            longitude=coord_slice(ds["longitude"].values, CENTER_LON_MIN, CENTER_LON_MAX),
        ).load()

    cropped.attrs.update(
        {
            "dataset_variant": "stage1_center_crop_from_big",
            "source_big_netcdf": str(src),
            "region_lat": f"{CENTER_LAT_MIN}-{CENTER_LAT_MAX}",
            "region_lon": f"{CENTER_LON_MIN}-{CENTER_LON_MAX}",
        }
    )
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in cropped.data_vars
        if cropped[name].ndim >= 2
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.to_netcdf(dst, engine="netcdf4", encoding=encoding)
    return "saved", str(dst), tuple(cropped["data"].shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_stage1_dirs()
    sources = sorted(BIG_MONTHLY_DIR.glob("*.nc"))
    if not sources:
        raise FileNotFoundError(f"No Stage 1 big monthly NetCDF files: {BIG_MONTHLY_DIR}")

    print(f"big source: {BIG_MONTHLY_DIR}")
    print(f"center out: {CENTER_MONTHLY_DIR}")
    print(f"monthly files: {len(sources)} | workers: {args.num_workers}")

    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {
            executor.submit(crop_one, str(src), args.overwrite): src for src in sources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            status, dst, shape = future.result()
            print(f"[{index}/{len(sources)}] {status}: {dst} shape={shape}")


if __name__ == "__main__":
    main()
