"""Visualize raw Stage 1 H8 features for a few sampled forecast windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stage1_paths import (
    BIG_FRAME_CACHE_DIR,
    BIG_INDEX_MAIN,
    CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_ALL,
    DATA_DIR,
)


DEFAULT_CHANNELS = [
    "SWR",
    "PAR",
    "UVA",
    "UVB",
    "TAOT_02",
    "TAAE",
    "Sample_number",
]
TARGET_CHANNEL = "SWR"
DEFAULT_OUT_DIR = DATA_DIR / "05_visualizations" / "sample_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw raw 4-frame H8 input features and 3-frame SWR truth."
    )
    parser.add_argument(
        "--mode",
        choices=["center", "big_patch_5x5"],
        default="center",
        help="Use center samples or sample patches from the augmented train index.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--index_csv", type=Path, default=None)
    parser.add_argument("--frame_cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    if args.mode == "center":
        args.index_csv = args.index_csv or CENTER_INDEX_ALL
        args.frame_cache_dir = args.frame_cache_dir or CENTER_FRAME_CACHE_DIR
    else:
        args.index_csv = args.index_csv or BIG_INDEX_MAIN
        args.frame_cache_dir = args.frame_cache_dir or BIG_FRAME_CACHE_DIR


def cache_key_for_nc(nc_path: str | Path) -> str:
    path = Path(nc_path)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RawFrameReader:
    def __init__(self, frame_cache_dir: Path):
        self.frame_cache_dir = frame_cache_dir
        self.frame_cache: dict[str, dict] = {}
        self.ds_cache: dict[str, xr.Dataset] = {}

    def close(self) -> None:
        for ds in self.ds_cache.values():
            ds.close()
        self.frame_cache = {}
        self.ds_cache = {}

    def open_dataset(self, nc_path: str) -> xr.Dataset:
        ds = self.ds_cache.get(nc_path)
        if ds is None:
            ds = xr.open_dataset(nc_path)
            self.ds_cache[nc_path] = ds
        return ds

    def open_frame_cache(self, nc_path: str) -> dict | None:
        cached = self.frame_cache.get(nc_path)
        if cached is not None:
            return cached
        meta_path = self.frame_cache_dir / f"{cache_key_for_nc(nc_path)}.json"
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        data_path = Path(meta["data_path"])
        if not data_path.is_absolute():
            data_path = meta_path.parent / data_path
        cached = {
            "data": np.load(data_path, mmap_mode="r"),
            "lookup": {str(name): index for index, name in enumerate(meta["channels"])},
        }
        self.frame_cache[nc_path] = cached
        return cached

    @staticmethod
    def slices(row: pd.Series) -> tuple[slice, slice, slice, slice]:
        x_slice = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
        y_slice = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)
        if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index):
            lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
            lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
        else:
            lat_slice = slice(None)
            lon_slice = slice(None)
        return x_slice, y_slice, lat_slice, lon_slice

    def read(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        nc_path = str(row["nc_path"])
        x_slice, y_slice, lat_slice, lon_slice = self.slices(row)
        frame_cache = self.open_frame_cache(nc_path)
        if frame_cache is not None:
            lookup = frame_cache["lookup"]
            channel_indices = [lookup[channel] for channel in DEFAULT_CHANNELS]
            swr_index = lookup[TARGET_CHANNEL]
            data = frame_cache["data"]
            x = np.asarray(
                data[x_slice, :, lat_slice, lon_slice][:, channel_indices],
                dtype=np.float32,
            )
            y = np.asarray(data[y_slice, swr_index, lat_slice, lon_slice], dtype=np.float32)
            return x, y

        ds = self.open_dataset(nc_path)
        lookup = {
            str(name): index for index, name in enumerate(ds["channel"].values.tolist())
        }
        channel_indices = [lookup[channel] for channel in DEFAULT_CHANNELS]
        swr_index = lookup[TARGET_CHANNEL]
        x = np.asarray(
            ds["data"].isel(
                time=x_slice,
                channel=channel_indices,
                latitude=lat_slice,
                longitude=lon_slice,
            ).values,
            dtype=np.float32,
        )
        y = np.asarray(
            ds["data"].isel(
                time=y_slice,
                channel=swr_index,
                latitude=lat_slice,
                longitude=lon_slice,
            ).values,
            dtype=np.float32,
        )
        return x, y


def robust_limits(values: np.ndarray, channel: str) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0.0, 1.0
    if channel == "Sample_number":
        return 0.0, max(4.0, float(np.nanmax(valid)))
    low = float(np.nanpercentile(valid, 1))
    high = float(np.nanpercentile(valid, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(valid))
        high = float(np.nanmax(valid))
    if high <= low:
        high = low + 1.0
    return low, high


def plot_sample(
    out_path: Path,
    row: pd.Series,
    x: np.ndarray,
    y: np.ndarray,
    dpi: int,
) -> None:
    history_frames = int(x.shape[0])
    future_frames = int(y.shape[0])
    columns = max(history_frames, future_frames)
    total_rows = len(DEFAULT_CHANNELS) + 1
    fig, axes = plt.subplots(
        total_rows,
        columns,
        figsize=(3.5 * columns, 2.65 * total_rows),
        constrained_layout=True,
    )

    for channel_index, channel in enumerate(DEFAULT_CHANNELS):
        vmin, vmax = robust_limits(x[:, channel_index], channel)
        cmap = "viridis" if channel == "Sample_number" else "turbo"
        for time_index in range(columns):
            ax = axes[channel_index, time_index]
            if time_index < history_frames:
                image = ax.imshow(
                    x[time_index, channel_index],
                    cmap=cmap,
                    origin="upper",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"{channel} | history x{time_index}", fontsize=9)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    y_vmin, y_vmax = robust_limits(y, TARGET_CHANNEL)
    for time_index in range(columns):
        ax = axes[-1, time_index]
        if time_index < future_frames:
            image = ax.imshow(y[time_index], cmap="turbo", origin="upper", vmin=y_vmin, vmax=y_vmax)
            ax.set_title(f"truth SWR | future y{time_index}", fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    patch_text = ""
    if "patch_id" in row.index:
        patch_text = (
            f" | patch={row['patch_id']} "
            f"lat={row['lat_min']}-{row['lat_max']} lon={row['lon_min']}-{row['lon_max']}"
        )
    issue = row.get("issue_time_bjt", row.get("issue_bjt", ""))
    future = " -> ".join(
        str(row[f"y_time_{index}_bjt"])
        for index in range(future_frames)
        if f"y_time_{index}_bjt" in row.index
    )
    fig.suptitle(f"issue {issue} | future {future}{patch_text}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_paths(args)
    if not args.index_csv.exists():
        raise FileNotFoundError(f"Index not found: {args.index_csv}")
    df = pd.read_csv(args.index_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No samples for split={args.split}: {args.index_csv}")

    rng = np.random.default_rng(args.seed)
    take = min(int(args.num_samples), len(df))
    selected_indices = sorted(rng.choice(len(df), size=take, replace=False).tolist())
    out_dir = args.out_dir / f"{args.mode}_{args.split}"
    reader = RawFrameReader(args.frame_cache_dir)
    try:
        print("mode:", args.mode)
        print("split:", args.split)
        print("index:", args.index_csv)
        print("frame_cache:", args.frame_cache_dir)
        print("selected rows:", selected_indices)
        for sample_number, row_index in enumerate(selected_indices):
            row = df.iloc[row_index]
            x, y = reader.read(row)
            out_path = out_dir / f"{args.mode}_{args.split}_sample_{sample_number:02d}_row_{row_index:05d}.png"
            plot_sample(out_path, row, x, y, args.dpi)
            print("saved:", out_path)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
