"""Build the center-only sample cache with parallel chunk writers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


from build_sample_cache import SourceReader
from h8_3d_common import TARGET_CHANNEL, load_index, parse_channels, read_json, write_json
from stage1_paths import (  # noqa: E402
    CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_ALL,
    CENTER_SAMPLE_CACHE_DIR,
    ensure_stage1_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=Path, default=CENTER_INDEX_ALL)
    parser.add_argument("--out_dir", type=Path, default=CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--frame_cache_dir", type=Path, default=CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--channels", type=str, default=",".join(parse_channels(None)))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cache_ready(out_dir: Path, channels: list[str], n: int) -> bool:
    paths = (
        out_dir / "sample_cache_meta.json",
        out_dir / "x_float32.npy",
        out_dir / "y_float32.npy",
        out_dir / "mask_uint8.npy",
    )
    if not all(path.exists() for path in paths):
        return False
    meta = read_json(paths[0])
    return meta.get("channels") == channels and int(meta.get("num_samples", -1)) == n


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, max(1, chunk_size))
    ]


def write_chunk(
    index_csv: str,
    frame_cache_dir: str,
    channels: list[str],
    x_path: str,
    y_path: str,
    mask_path: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    df = pd.read_csv(index_csv)
    x_arr = np.load(x_path, mmap_mode="r+")
    y_arr = np.load(y_path, mmap_mode="r+")
    mask_arr = np.load(mask_path, mmap_mode="r+")
    reader = SourceReader(channels, Path(frame_cache_dir))
    try:
        for row_index in range(start, end):
            x, y, mask = reader.read_sample(df.iloc[row_index])
            x_arr[row_index] = x
            y_arr[row_index] = y
            mask_arr[row_index] = mask
        x_arr.flush()
        y_arr.flush()
        mask_arr.flush()
        return start, end
    finally:
        reader.close()


def main() -> None:
    args = parse_args()
    ensure_stage1_dirs()
    channels = parse_channels(args.channels)
    df = load_index(args.index_csv)
    n = len(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if cache_ready(args.out_dir, channels, n) and not args.overwrite:
        print("sample cache already exists:", args.out_dir)
        return

    reader = SourceReader(channels, args.frame_cache_dir)
    try:
        first_x, first_y, first_mask = reader.read_sample(df.iloc[0])
    finally:
        reader.close()

    x_shape = (n,) + tuple(first_x.shape)
    y_shape = (n,) + tuple(first_y.shape)
    mask_shape = (n,) + tuple(first_mask.shape)
    x_path = args.out_dir / "x_float32.npy"
    y_path = args.out_dir / "y_float32.npy"
    mask_path = args.out_dir / "mask_uint8.npy"

    print("index:", args.index_csv)
    print("out_dir:", args.out_dir)
    print("frame_cache_dir:", args.frame_cache_dir)
    print("channels:", channels)
    print("num_samples:", n)
    print("num_workers:", args.num_workers)
    print("chunk_size:", args.chunk_size)
    print("x_shape:", x_shape, "float32")
    print("y_shape:", y_shape, "float32")
    print("mask_shape:", mask_shape, "uint8")

    np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=x_shape).flush()
    np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=y_shape).flush()
    np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=mask_shape).flush()

    ranges = chunk_ranges(n, args.chunk_size)
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [
            executor.submit(
                write_chunk,
                str(args.index_csv),
                str(args.frame_cache_dir),
                channels,
                str(x_path),
                str(y_path),
                str(mask_path),
                start,
                end,
            )
            for start, end in ranges
        ]
        for future in as_completed(futures):
            start, end = future.result()
            completed += end - start
            print(f"cached {completed}/{n} | finished rows [{start}, {end})", flush=True)

    df.drop(columns=["_index_row"]).to_csv(args.out_dir / "sample_index.csv", index=False)
    write_json(
        {
            "index_csv": str(args.index_csv),
            "frame_cache_dir": str(args.frame_cache_dir),
            "channels": channels,
            "target_channel": TARGET_CHANNEL,
            "num_samples": n,
            "num_workers": max(1, args.num_workers),
            "chunk_size": max(1, args.chunk_size),
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


if __name__ == "__main__":
    main()
