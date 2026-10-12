#!/usr/bin/env python3
"""Expose Stage 1 center SWR frame cache in the legacy multisource-cache format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from stage3_paths import (
    LEGACY_CENTER_SWR_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_MONTHLY_DIR,
    STAGE2_DIR,
    TIME_END,
    TIME_START,
    ensure_stage3_dirs,
)


sys.path.insert(0, str(STAGE2_DIR))
from train_3dcnn_ratio_swr import RatioFrameReader  # noqa: E402


ADAPTER_SCHEMA_VERSION = 2
HOUR_ROUND_TOLERANCE_MINUTES = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Stage 1 center SWR frame-cache values for the unchanged PV multisource builder."
    )
    parser.add_argument("--monthly_dir", type=Path, default=STAGE1_CENTER_MONTHLY_DIR)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=LEGACY_CENTER_SWR_CACHE_DIR)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--time_start",
        type=str,
        default=TIME_START,
        help="PV experiment start. The adapter exports H8 frames from the same BJT calendar day.",
    )
    parser.add_argument(
        "--time_end",
        type=str,
        default=TIME_END,
        help="PV experiment end. Frames after this BJT calendar day are outside Stage 3.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def hourly_lookup_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    raw = pd.Timestamp(value)
    rounded = raw.round("h")
    offset_minutes = abs((raw - rounded).total_seconds()) / 60.0
    if offset_minutes > HOUR_ROUND_TOLERANCE_MINUTES:
        raise ValueError(
            f"H8 start label is too far from an hourly boundary: "
            f"raw={raw}, rounded={rounded}, offset_minutes={offset_minutes:.3f}"
        )
    return rounded


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    swr_path = args.out_dir / "swr_3d.npy"
    timestamps_path = args.out_dir / "timestamps.csv"
    lat_path = args.out_dir / "lat_values.npy"
    lon_path = args.out_dir / "lon_values.npy"
    meta_path = args.out_dir / "stage1_center_swr_adapter_meta.json"
    required = (swr_path, timestamps_path, lat_path, lon_path, meta_path)
    expected_day_start = str(pd.Timestamp(args.time_start).normalize().date())
    expected_day_end = str(pd.Timestamp(args.time_end).normalize().date())
    if all(path.exists() for path in required) and not bool(args.overwrite):
        with meta_path.open("r", encoding="utf-8") as handle:
            existing_meta = json.load(handle)
        if (
            int(existing_meta.get("adapter_schema_version", -1)) == ADAPTER_SCHEMA_VERSION
            and
            str(existing_meta.get("export_bjt_day_start", "")) == expected_day_start
            and str(existing_meta.get("export_bjt_day_end", "")) == expected_day_end
        ):
            print("reuse:", args.out_dir)
            return
        print("rebuild: existing adapter does not match the Stage 3 BJT day range")

    monthly_paths = sorted(args.monthly_dir.glob("*.nc"))
    if not monthly_paths:
        raise FileNotFoundError(f"No Stage 1 center monthly NetCDF files: {args.monthly_dir}")

    export_day_start = pd.Timestamp(args.time_start).normalize()
    export_day_end = pd.Timestamp(args.time_end).normalize() + pd.Timedelta(days=1)
    month_specs: list[tuple[Path, np.ndarray, int]] = []
    lat = lon = None
    for path in monthly_paths:
        with xr.open_dataset(path) as dataset:
            times = pd.to_datetime(np.asarray(dataset["time_start_bjt"].values).astype("datetime64[s]"))
            keep = np.flatnonzero((times >= export_day_start) & (times < export_day_end))
            if keep.size == 0:
                continue
            current_lat = np.asarray(dataset["latitude"].values, dtype=np.float32)
            current_lon = np.asarray(dataset["longitude"].values, dtype=np.float32)
            if lat is None:
                lat, lon = current_lat, current_lon
            elif not np.array_equal(lat, current_lat) or not np.array_equal(lon, current_lon):
                raise ValueError(f"Inconsistent Stage 1 center grid: {path}")
            month_specs.append((path, keep, int(keep.size)))

    if not month_specs:
        raise RuntimeError(
            f"No Stage 1 center frames within BJT days {export_day_start.date()} to {args.time_end}"
        )
    total_frames = int(sum(frame_count for _, _, frame_count in month_specs))
    dtype = np.dtype(args.dtype)
    swr = np.lib.format.open_memmap(
        swr_path,
        mode="w+",
        dtype=dtype,
        shape=(total_frames, len(lat), len(lon)),
    )
    reader = RatioFrameReader(["SWR"], args.frame_cache_dir)
    timestamp_rows = []
    offset = 0
    try:
        for month_index, (path, keep, frame_count) in enumerate(month_specs, start=1):
            with xr.open_dataset(path) as dataset:
                times = pd.to_datetime(
                    np.asarray(dataset["time_start_bjt"].values).astype("datetime64[s]")
                )[keep]
            frame_cache = reader._open_frame_cache(str(path))
            if frame_cache is None:
                raise FileNotFoundError(
                    f"Stage 1 center frame cache missing for {path}. Run stage_1/05_build_center_frame_cache.py first."
                )
            swr_channel = int(frame_cache["lookup"]["SWR"])
            values = np.asarray(frame_cache["data"][keep, swr_channel], dtype=dtype)
            if values.shape[0] != frame_count:
                raise ValueError(f"Frame count mismatch for {path}: {values.shape[0]} != {frame_count}")
            swr[offset : offset + frame_count] = values
            for index, timestamp in enumerate(times):
                raw_timestamp = pd.Timestamp(timestamp)
                aligned_timestamp = hourly_lookup_timestamp(raw_timestamp)
                timestamp_rows.append(
                    {
                        "index": int(offset + index),
                        "timestamp": aligned_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "raw_timestamp": raw_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "hour_round_offset_seconds": float(
                            (raw_timestamp - aligned_timestamp).total_seconds()
                        ),
                    }
                )
            offset += frame_count
            print(f"[{month_index}/{len(month_specs)}] exported {path.name} | frames={offset}/{total_frames}", flush=True)
    finally:
        reader.close()
    swr.flush()
    timestamp_df = pd.DataFrame(timestamp_rows)
    duplicated = timestamp_df["timestamp"].duplicated(keep=False)
    if duplicated.any():
        examples = timestamp_df.loc[duplicated, ["timestamp", "raw_timestamp"]].head(10)
        raise ValueError(
            "Multiple H8 frames map to the same hourly lookup key:\n"
            + examples.to_string(index=False)
        )
    np.save(lat_path, lat.astype(np.float32))
    np.save(lon_path, lon.astype(np.float32))
    timestamp_df.to_csv(timestamps_path, index=False)
    write_json(
        {
            "purpose": "Legacy-format SWR adapter backed by the leakage-safe Stage 1 center frame cache.",
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "timestamp_alignment": (
                "timestamp is the nearest hourly BJT lookup key; raw_timestamp preserves "
                "the original H8 time_start_bjt label."
            ),
            "hour_round_tolerance_minutes": HOUR_ROUND_TOLERANCE_MINUTES,
            "source_monthly_dir": str(args.monthly_dir),
            "source_frame_cache_dir": str(args.frame_cache_dir),
            "shape": [total_frames, int(len(lat)), int(len(lon))],
            "dtype": str(dtype),
            "export_bjt_day_start": str(export_day_start.date()),
            "export_bjt_day_end": str(pd.Timestamp(args.time_end).normalize().date()),
            "time_start": timestamp_rows[0]["timestamp"],
            "time_end": timestamp_rows[-1]["timestamp"],
        },
        meta_path,
    )
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
