#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from h8_3d_common import (
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_INDEX_CSV,
    cache_key_for_nc,
    load_index,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert processed monthly H8 NetCDF files to uncompressed .npy frame cache. "
            "This is faster for random-window PyTorch training than repeated NetCDF slicing."
        )
    )
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_FRAME_CACHE_DIR)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_one(nc_path: Path, cache_dir: Path, dtype: str, overwrite: bool) -> Path:
    key = cache_key_for_nc(nc_path)
    meta_path = cache_dir / f"{key}.json"
    data_path = cache_dir / f"{key}_data.npy"
    mask_path = cache_dir / f"{key}_valid_mask.npy"

    if meta_path.exists() and data_path.exists() and not overwrite:
        print("exists:", meta_path)
        return meta_path

    print("reading:", nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        channels = [str(x) for x in ds["channel"].values.tolist()]
        data = np.asarray(ds["data"].values, dtype=np.dtype(dtype))
        np.save(data_path, data)

        has_mask = "valid_mask" in ds
        if has_mask:
            mask = np.asarray(ds["valid_mask"].values, dtype=np.uint8)
            np.save(mask_path, mask)

        meta = {
            "source_nc_path": str(nc_path),
            "data_path": data_path.name,
            "valid_mask_path": mask_path.name if has_mask else None,
            "channels": channels,
            "data_shape": list(data.shape),
            "dtype": str(data.dtype),
            "has_valid_mask": bool(has_mask),
        }
        write_json(meta, meta_path)
        print("saved:", meta_path, "shape=", data.shape, "dtype=", data.dtype)
        return meta_path
    finally:
        ds.close()


def build_one_star(args: tuple[Path, Path, str, bool]) -> Path:
    return build_one(*args)


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_index(args.index_csv)
    nc_paths = sorted({Path(p) for p in df["nc_path"].astype(str).tolist()})
    print("index:", args.index_csv)
    print("cache_dir:", args.cache_dir)
    print("unique NetCDF files:", len(nc_paths))

    num_workers = max(1, int(args.num_workers))
    print("num_workers:", num_workers)

    if num_workers == 1 or len(nc_paths) <= 1:
        for i, nc_path in enumerate(nc_paths, start=1):
            print(f"\n[{i}/{len(nc_paths)}]")
            build_one(nc_path, args.cache_dir, args.dtype, args.overwrite)
    else:
        tasks = [(nc_path, args.cache_dir, args.dtype, bool(args.overwrite)) for nc_path in nc_paths]
        with ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as pool:
            future_to_path = {pool.submit(build_one_star, task): task[0] for task in tasks}
            done = 0
            for fut in as_completed(future_to_path):
                nc_path = future_to_path[fut]
                done += 1
                try:
                    out = fut.result()
                    print(f"[{done}/{len(tasks)}] done: {nc_path.name} -> {out}", flush=True)
                except Exception as e:
                    print(f"[{done}/{len(tasks)}] failed: {nc_path}", flush=True)
                    raise e

    print("\nDone.")


if __name__ == "__main__":
    main()
