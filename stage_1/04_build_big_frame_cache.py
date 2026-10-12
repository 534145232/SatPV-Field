"""Build the big frame cache incrementally to keep memory bounded."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from stage1_paths import BIG_FRAME_CACHE_DIR, BIG_INDEX_MAIN, ensure_stage1_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=Path, default=BIG_INDEX_MAIN)
    parser.add_argument("--cache_dir", type=Path, default=BIG_FRAME_CACHE_DIR)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--time_chunk", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cache_key_for_nc(nc_path: Path) -> str:
    digest = hashlib.sha1(str(nc_path).encode("utf-8")).hexdigest()[:12]
    return f"{nc_path.stem}_{digest}"


def write_json(data: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_one(
    nc_path: Path,
    cache_dir: Path,
    dtype_name: str,
    time_chunk: int,
    overwrite: bool,
) -> Path:
    key = cache_key_for_nc(nc_path)
    meta_path = cache_dir / f"{key}.json"
    data_path = cache_dir / f"{key}_data.npy"
    mask_path = cache_dir / f"{key}_valid_mask.npy"
    if meta_path.exists() and data_path.exists() and not overwrite:
        print("exists:", meta_path, flush=True)
        return meta_path

    tmp_data = data_path.with_suffix(data_path.suffix + ".partial")
    tmp_mask = mask_path.with_suffix(mask_path.suffix + ".partial")
    for path in (tmp_data, tmp_mask):
        if path.exists():
            path.unlink()

    print("reading incrementally:", nc_path, flush=True)
    dtype = np.dtype(dtype_name)
    try:
        with xr.open_dataset(nc_path) as ds:
            data_source = ds["data"]
            data_shape = tuple(int(size) for size in data_source.shape)
            channels = [str(value) for value in ds["channel"].values.tolist()]
            data_out = np.lib.format.open_memmap(tmp_data, mode="w+", dtype=dtype, shape=data_shape)
            has_mask = "valid_mask" in ds
            mask_out = None
            if has_mask:
                mask_shape = tuple(int(size) for size in ds["valid_mask"].shape)
                mask_out = np.lib.format.open_memmap(
                    tmp_mask, mode="w+", dtype=np.uint8, shape=mask_shape
                )
            for start in range(0, data_shape[0], max(1, time_chunk)):
                end = min(start + max(1, time_chunk), data_shape[0])
                data_out[start:end] = np.asarray(
                    data_source.isel(time=slice(start, end)).values, dtype=dtype
                )
                if mask_out is not None:
                    mask_out[start:end] = np.asarray(
                        ds["valid_mask"].isel(time=slice(start, end)).values,
                        dtype=np.uint8,
                    )
                print(f"  cached frames {end}/{data_shape[0]}", flush=True)
            data_out.flush()
            if mask_out is not None:
                mask_out.flush()

        os.replace(tmp_data, data_path)
        if tmp_mask.exists():
            os.replace(tmp_mask, mask_path)
        write_json(
            {
                "source_nc_path": str(nc_path),
                "data_path": data_path.name,
                "valid_mask_path": mask_path.name if has_mask else None,
                "channels": channels,
                "data_shape": list(data_shape),
                "dtype": dtype_name,
                "has_valid_mask": has_mask,
                "writer": "incremental_time_chunk",
                "time_chunk": max(1, time_chunk),
            },
            meta_path,
        )
        print("saved:", meta_path, flush=True)
        return meta_path
    except Exception:
        for path in (tmp_data, tmp_mask):
            if path.exists():
                path.unlink()
        raise


def build_one_star(args: tuple[Path, Path, str, int, bool]) -> Path:
    return build_one(*args)


def main() -> None:
    args = parse_args()
    ensure_stage1_dirs()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.index_csv)
    nc_paths = sorted({Path(path) for path in df["nc_path"].astype(str)})
    print("index:", args.index_csv)
    print("cache_dir:", args.cache_dir)
    print("unique NetCDF files:", len(nc_paths))
    print("num_workers:", args.num_workers)
    print("time_chunk:", args.time_chunk)

    tasks = [
        (path, args.cache_dir, args.dtype, max(1, args.time_chunk), args.overwrite)
        for path in nc_paths
    ]
    if max(1, args.num_workers) == 1:
        for index, task in enumerate(tasks, start=1):
            print(f"\n[{index}/{len(tasks)}]", flush=True)
            build_one_star(task)
    else:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(tasks))) as executor:
            futures = {executor.submit(build_one_star, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                print(f"[{index}/{len(tasks)}] done: {future.result()}", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
