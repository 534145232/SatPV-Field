#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from h8_3d_common import (
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_INDEX_CSV,
    DEFAULT_SAMPLE_CACHE_DIR,
    TARGET_CHANNEL,
    cache_meta_path,
    load_index,
    parse_channels,
    read_json,
    write_json,
    _channel_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sample-level float32 cache for H8 3D SWR forecasting. "
            "The output is three contiguous memmap-friendly .npy files: x/y/mask."
        )
    )
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_SAMPLE_CACHE_DIR)
    parser.add_argument("--channels", type=str, default=",".join(parse_channels(None)))
    parser.add_argument("--frame_cache_dir", type=Path, default=DEFAULT_FRAME_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class SourceReader:
    def __init__(self, channels: list[str], frame_cache_dir: Path | None):
        self.channels = channels
        self.frame_cache_dir = frame_cache_dir
        self.ds_cache: dict[str, xr.Dataset] = {}
        self.frame_cache: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        for ds in self.ds_cache.values():
            ds.close()
        self.ds_cache = {}
        self.frame_cache = {}

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

        valid_mask = None
        mask_path_text = meta.get("valid_mask_path")
        if mask_path_text:
            mask_path = Path(mask_path_text)
            if not mask_path.is_absolute():
                mask_path = meta_path.parent / mask_path
            if mask_path.exists():
                valid_mask = np.load(mask_path, mmap_mode="r")

        lookup = {str(name): i for i, name in enumerate(meta["channels"])}
        self._check_channels(lookup, str(meta_path))
        cached = {"data": data, "valid_mask": valid_mask, "lookup": lookup}
        self.frame_cache[nc_path] = cached
        return cached

    def _open_dataset(self, nc_path: str) -> tuple[xr.Dataset, dict[str, int]]:
        ds = self.ds_cache.get(nc_path)
        if ds is None:
            ds = xr.open_dataset(nc_path)
            self.ds_cache[nc_path] = ds
        lookup = _channel_lookup(ds)
        self._check_channels(lookup, nc_path)
        return ds, lookup

    def _check_channels(self, lookup: dict[str, int], source: str) -> None:
        missing = [c for c in self.channels + [TARGET_CHANNEL] if c not in lookup]
        if missing:
            raise KeyError(f"Missing channels in {source}: {missing}")

    def read_sample(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nc_path = str(row["nc_path"])
        xs = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
        ys = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)

        fc = self._open_frame_cache(nc_path)
        if fc is not None:
            lookup = fc["lookup"]
            x_idx = [lookup[c] for c in self.channels]
            y_idx = lookup[TARGET_CHANNEL]
            data = fc["data"]
            x = np.asarray(data[xs, :, :, :][:, x_idx, :, :], dtype=np.float32)
            y = np.asarray(data[ys, y_idx, :, :], dtype=np.float32)
            mask = np.isfinite(y)
            if fc["valid_mask"] is not None:
                mask &= np.asarray(fc["valid_mask"][ys]) > 0
            return x, y, mask.astype(np.uint8)

        ds, lookup = self._open_dataset(nc_path)
        x_idx = [lookup[c] for c in self.channels]
        y_idx = lookup[TARGET_CHANNEL]
        x = np.asarray(ds["data"].isel(time=xs, channel=x_idx).values, dtype=np.float32)
        y = np.asarray(ds["data"].isel(time=ys, channel=y_idx).values, dtype=np.float32)
        mask = np.isfinite(y)
        if "valid_mask" in ds:
            mask &= np.asarray(ds["valid_mask"].isel(time=ys).values) > 0
        return x, y, mask.astype(np.uint8)


def _cache_ready(out_dir: Path, channels: list[str], n: int) -> bool:
    meta_path = out_dir / "sample_cache_meta.json"
    x_path = out_dir / "x_float32.npy"
    y_path = out_dir / "y_float32.npy"
    mask_path = out_dir / "mask_uint8.npy"
    if not (meta_path.exists() and x_path.exists() and y_path.exists() and mask_path.exists()):
        return False
    meta = read_json(meta_path)
    return meta.get("channels") == channels and int(meta.get("num_samples", -1)) == int(n)


def main() -> None:
    args = parse_args()
    channels = parse_channels(args.channels)
    if not args.index_csv.exists():
        raise FileNotFoundError(
            f"Index CSV not found: {args.index_csv}\n"
            "Generate the 4-history + 3-future index first, for example:\n"
            "  bash /code/Final_3_stage_pipeline/stage_1/run_stage1_build_indices_and_cache.sh"
        )
    df = load_index(args.index_csv)
    n = len(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if _cache_ready(args.out_dir, channels, n) and not args.overwrite:
        print("sample cache already exists:", args.out_dir)
        return

    reader = SourceReader(channels, args.frame_cache_dir)
    try:
        first_x, first_y, first_mask = reader.read_sample(df.iloc[0])
        x_shape = (n,) + tuple(first_x.shape)
        y_shape = (n,) + tuple(first_y.shape)
        mask_shape = (n,) + tuple(first_mask.shape)

        x_path = args.out_dir / "x_float32.npy"
        y_path = args.out_dir / "y_float32.npy"
        mask_path = args.out_dir / "mask_uint8.npy"

        print("index:", args.index_csv)
        print("out_dir:", args.out_dir)
        print("channels:", channels)
        print("num_samples:", n)
        print("x_shape:", x_shape, "float32")
        print("y_shape:", y_shape, "float32")
        print("mask_shape:", mask_shape, "uint8")

        x_arr = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=x_shape)
        y_arr = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=y_shape)
        mask_arr = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=mask_shape)

        x_arr[0] = first_x
        y_arr[0] = first_y
        mask_arr[0] = first_mask

        for i in range(1, n):
            x, y, mask = reader.read_sample(df.iloc[i])
            x_arr[i] = x
            y_arr[i] = y
            mask_arr[i] = mask
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"cached {i + 1}/{n}", flush=True)

        x_arr.flush()
        y_arr.flush()
        mask_arr.flush()
        df.drop(columns=["_index_row"]).to_csv(args.out_dir / "sample_index.csv", index=False)
        write_json(
            {
                "index_csv": str(args.index_csv),
                "frame_cache_dir": str(args.frame_cache_dir) if args.frame_cache_dir else None,
                "channels": channels,
                "target_channel": TARGET_CHANNEL,
                "num_samples": int(n),
                "x_path": x_path.name,
                "y_path": y_path.name,
                "mask_path": mask_path.name,
                "x_shape": list(x_shape),
                "y_shape": list(y_shape),
                "mask_shape": list(mask_shape),
                "dtype": "float32",
                "mask_dtype": "uint8",
            },
            args.out_dir / "sample_cache_meta.json",
        )
        print("done:", args.out_dir)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
