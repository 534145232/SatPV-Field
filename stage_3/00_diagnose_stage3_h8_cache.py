#!/usr/bin/env python3
"""Diagnose Stage 3 center-SWR adapter and 64x64 multisource H8 values."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_paths import LEGACY_CENTER_SWR_CACHE_DIR, MULTISOURCE_CACHE_DIR
import build_fullfield_multisource_cache as base
from stage3_temporal import swr_history_times


DEFAULT_ISSUE = "2023-04-17 12:00:00"


def stats(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    finite = np.isfinite(values)
    ratio = float(finite.mean()) if values.size else 0.0
    print(f"{name}: shape={values.shape} dtype={values.dtype} finite_ratio={ratio:.6f}")
    if finite.any():
        selected = np.asarray(values[finite], dtype=np.float32)
        print(
            f"  min={float(np.nanmin(selected)):.3f} "
            f"max={float(np.nanmax(selected)):.3f} "
            f"mean={float(np.nanmean(selected)):.3f}"
        )


def main() -> None:
    issue = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp(DEFAULT_ISSUE)
    requested = swr_history_times(issue)
    print("==== Stage 3 H8 cache diagnosis ====")
    print("PV issue T:", issue)
    print("required observed SWR:", [str(value) for value in requested])
    print("")

    adapter_meta_path = LEGACY_CENTER_SWR_CACHE_DIR / "stage1_center_swr_adapter_meta.json"
    timestamp_path = LEGACY_CENTER_SWR_CACHE_DIR / "timestamps.csv"
    swr_path = LEGACY_CENTER_SWR_CACHE_DIR / "swr_3d.npy"
    lat_path = LEGACY_CENTER_SWR_CACHE_DIR / "lat_values.npy"
    lon_path = LEGACY_CENTER_SWR_CACHE_DIR / "lon_values.npy"
    for path in (adapter_meta_path, timestamp_path, swr_path, lat_path, lon_path):
        print(f"[{'OK' if path.exists() else 'MISSING'}] {path}")
    if not all(path.exists() for path in (adapter_meta_path, timestamp_path, swr_path, lat_path, lon_path)):
        raise FileNotFoundError("Legacy center SWR adapter is incomplete.")

    with adapter_meta_path.open("r", encoding="utf-8") as handle:
        print("adapter meta:", json.load(handle))
    timestamps = pd.read_csv(timestamp_path)
    timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
    by_time = {pd.Timestamp(row["timestamp"]): int(row["index"]) for _, row in timestamps.iterrows()}
    swr = np.load(swr_path, mmap_mode="r")
    lat = np.load(lat_path)
    lon = np.load(lon_path)
    stats("adapter latitude", lat)
    stats("adapter longitude", lon)
    print(f"adapter lat range: {float(np.nanmin(lat)):.4f} .. {float(np.nanmax(lat)):.4f}")
    print(f"adapter lon range: {float(np.nanmin(lon)):.4f} .. {float(np.nanmax(lon)):.4f}")
    source_frames = []
    for timestamp in requested:
        index = by_time.get(timestamp)
        print(f"lookup {timestamp}: {'missing' if index is None else f'index={index}'}")
        if index is None:
            nearest_rows = timestamps.iloc[
                np.argsort(np.abs((timestamps["timestamp"] - timestamp).dt.total_seconds().to_numpy()))[:3]
            ]
            print(
                "  nearest adapter timestamps:",
                nearest_rows[[column for column in ["timestamp", "raw_timestamp"] if column in nearest_rows.columns]]
                .astype(str)
                .to_dict(orient="records"),
            )
        if index is not None:
            frame = np.asarray(swr[index], dtype=np.float32)
            stats(f"  adapter SWR {timestamp}", frame)
            source_frames.append(frame)

    ms_time_path = MULTISOURCE_CACHE_DIR / "issue_time_index.csv"
    ms_h8_path = MULTISOURCE_CACHE_DIR / "h8_swr_64_float16.npy"
    ms_sec_path = MULTISOURCE_CACHE_DIR / "h8_frame_sec.npy"
    if all(path.exists() for path in (ms_time_path, ms_h8_path, ms_sec_path)):
        time_df = pd.read_csv(ms_time_path)
        time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
        matches = time_df.index[time_df["issue_time"] == issue].tolist()
        if matches:
            index = int(matches[0])
            h8_64 = np.load(ms_h8_path, mmap_mode="r")
            sec = np.load(ms_sec_path, mmap_mode="r")
            print("")
            print("multisource row:", index)
            print("multisource h8 timestamps:", [str(pd.Timestamp(value, unit="s")) for value in sec[index]])
            stats("multisource H8 64x64", np.asarray(h8_64[index], dtype=np.float32))
        else:
            print("multisource issue row missing:", issue)

    if len(source_frames) == len(requested):
        bounds = (120.9, 122.0, 29.0, 30.4)
        print("")
        print("manual Ningbo-like resample bounds:", bounds)
        for timestamp, frame in zip(requested, source_frames):
            out = base._resample_h8_to_bounds(frame, lat, lon, bounds, 64, 64)
            stats(f"manual resample {timestamp}", out)


if __name__ == "__main__":
    main()
