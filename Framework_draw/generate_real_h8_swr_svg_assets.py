#!/usr/bin/env python3
"""Generate PPT/article-ready image assets from real Stage 1 H8/SWR samples.

The script samples real Stage 1 forecast windows and writes clean image assets
with no axes, labels, legends, or color bars:

* 4 historical multi-channel H8 frames, drawn as offset channel sheets.
* 3 future SWR frames, drawn as single-field maps.
* one 4-frame history strip and one 3-frame future strip per sampled group.

It reads the same index/cache format used by Stage 1/2 training. Frame-cache
NPY files are preferred; if they are missing, the script falls back to the
NetCDF path stored in the index CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE1_DIR = PIPELINE_DIR / "stage_1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from stage1_paths import BIG_FRAME_CACHE_DIR, BIG_INDEX_MAIN, CENTER_FRAME_CACHE_DIR, CENTER_INDEX_ALL  # noqa: E402


DEFAULT_CHANNELS = ["SWR", "PAR", "UVA", "UVB", "TAOT_02", "TAAE", "Sample_number"]
DEFAULT_DRAW_CHANNELS = ["PAR", "TAOT_02", "TAAE"]
TARGET_CHANNEL = "SWR"

COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["center", "big_patch_5x5"],
        default="center",
        help="Read center training samples or the big-patch training index.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index_csv", type=Path, default=None)
    parser.add_argument("--frame_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=SCRIPT_DIR / "real_h8_swr_png_assets",
        help="Output directory for generated assets.",
    )
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--png_dpi", type=int, default=600)
    parser.add_argument(
        "--png_scale",
        type=float,
        default=4.0,
        help="Pixel scale applied to nominal drawing dimensions. 4 gives crisp PPT/article PNGs.",
    )
    parser.add_argument("--num_groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--issue_times_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing issue timestamps to render. Pass the "
            "sampled_issue_times.csv written by generate_stage3_multisource_input_assets.py "
            "to align these Stage 1 H8/SWR assets with the Stage 3 multisource samples."
        ),
    )
    parser.add_argument(
        "--issue_time_column",
        type=str,
        default="issue_time",
        help="Timestamp column in --issue_times_csv. Values are matched to Stage 1 issue_time_bjt.",
    )
    parser.add_argument(
        "--prefer_center_patch",
        type=int,
        default=1,
        help="When a big-patch index has multiple rows for the same issue time, prefer rows with is_center=1.",
    )
    parser.add_argument(
        "--issue_match_tolerance_minutes",
        type=float,
        default=75.0,
        help="Fallback nearest-time tolerance for matching requested issue times to Stage 1 issue_time_bjt.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=",".join(DEFAULT_CHANNELS),
        help="Input channels to read from real H8 data.",
    )
    parser.add_argument(
        "--draw_channels",
        type=str,
        default=",".join(DEFAULT_DRAW_CHANNELS),
        help=(
            "Inner channels to draw as offset historical sheets. "
            "SWR is always appended as the outer/front sheet."
        ),
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=36,
        help="Downsampled heatmap grid size. Lower values make smaller files.",
    )
    parser.add_argument(
        "--channel_offset_px",
        type=float,
        default=12.0,
        help="Offset between historical channel sheets.",
    )
    parser.add_argument("--history_width", type=float, default=180.0)
    parser.add_argument("--future_width", type=float, default=190.0)
    parser.add_argument("--history_strip_gap", type=float, default=28.0)
    parser.add_argument("--future_strip_gap", type=float, default=46.0)
    return parser.parse_args()


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def history_draw_channels(inner_channels: list[str]) -> list[str]:
    """Return back-to-front historical sheet order, with SWR as the front sheet."""
    ordered = []
    for channel in inner_channels:
        if channel == TARGET_CHANNEL:
            continue
        if channel not in ordered:
            ordered.append(channel)
    ordered.append(TARGET_CHANNEL)
    return ordered


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


class RealH8Reader:
    def __init__(self, frame_cache_dir: Path, channels: list[str]):
        self.frame_cache_dir = Path(frame_cache_dir)
        self.channels = list(channels)
        self.frame_cache: dict[str, dict] = {}
        self.ds_cache: dict[str, object] = {}

    def close(self) -> None:
        for ds in self.ds_cache.values():
            close = getattr(ds, "close", None)
            if close is not None:
                close()
        self.frame_cache = {}
        self.ds_cache = {}

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
        valid_mask_path = meta.get("valid_mask_path")
        valid_mask = None
        if valid_mask_path:
            mask_path = Path(valid_mask_path)
            if not mask_path.is_absolute():
                mask_path = meta_path.parent / mask_path
            if mask_path.exists():
                valid_mask = np.load(mask_path, mmap_mode="r")
        cached = {
            "data": np.load(data_path, mmap_mode="r"),
            "valid_mask": valid_mask,
            "lookup": {str(name): index for index, name in enumerate(meta["channels"])},
        }
        self._check_lookup(cached["lookup"], str(meta_path))
        self.frame_cache[nc_path] = cached
        return cached

    def open_dataset(self, nc_path: str):
        cached = self.ds_cache.get(nc_path)
        if cached is not None:
            return cached
        import xarray as xr

        ds = xr.open_dataset(nc_path)
        lookup = {str(name): index for index, name in enumerate(ds["channel"].values.tolist())}
        self._check_lookup(lookup, nc_path)
        cached = {"ds": ds, "lookup": lookup}
        self.ds_cache[nc_path] = cached
        return cached

    def _check_lookup(self, lookup: dict[str, int], source: str) -> None:
        missing = [channel for channel in list(self.channels) + [TARGET_CHANNEL] if channel not in lookup]
        if missing:
            raise KeyError(f"Missing channels in {source}: {missing}")

    @staticmethod
    def row_slices(row: pd.Series) -> tuple[slice, slice, slice, slice]:
        x_slice = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
        y_slice = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)
        if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index):
            lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
            lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
        else:
            lat_slice = slice(None)
            lon_slice = slice(None)
        return x_slice, y_slice, lat_slice, lon_slice

    def read_sample(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nc_path = str(row["nc_path"])
        x_slice, y_slice, lat_slice, lon_slice = self.row_slices(row)
        frame_cache = self.open_frame_cache(nc_path)
        if frame_cache is not None:
            lookup = frame_cache["lookup"]
            x_idx = [lookup[channel] for channel in self.channels]
            y_idx = lookup[TARGET_CHANNEL]
            data = frame_cache["data"]
            x = np.asarray(data[x_slice, :, lat_slice, lon_slice][:, x_idx], dtype=np.float32)
            y = np.asarray(data[y_slice, y_idx, lat_slice, lon_slice], dtype=np.float32)
            mask = np.isfinite(y)
            if frame_cache["valid_mask"] is not None:
                mask &= np.asarray(frame_cache["valid_mask"][y_slice, lat_slice, lon_slice]) > 0
            return x, y, mask.astype(np.uint8)

        opened = self.open_dataset(nc_path)
        ds = opened["ds"]
        lookup = opened["lookup"]
        x_idx = [lookup[channel] for channel in self.channels]
        y_idx = lookup[TARGET_CHANNEL]
        x = np.asarray(
            ds["data"].isel(time=x_slice, channel=x_idx, latitude=lat_slice, longitude=lon_slice).values,
            dtype=np.float32,
        )
        y = np.asarray(
            ds["data"].isel(time=y_slice, channel=y_idx, latitude=lat_slice, longitude=lon_slice).values,
            dtype=np.float32,
        )
        mask = np.isfinite(y)
        if "valid_mask" in ds:
            mask &= np.asarray(ds["valid_mask"].isel(time=y_slice, latitude=lat_slice, longitude=lon_slice).values) > 0
        return x, y, mask.astype(np.uint8)


def normalize_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).floor("s")


def load_requested_issue_times(issue_times_csv: Path, column: str) -> list[pd.Timestamp]:
    if not Path(issue_times_csv).exists():
        raise FileNotFoundError(f"--issue_times_csv not found: {issue_times_csv}")
    df = pd.read_csv(issue_times_csv)
    if column not in df.columns:
        if column == "stage1_issue_time_bjt" and "issue_time" in df.columns:
            print(
                f"[align] Column {column!r} not found in {issue_times_csv}; falling back to 'issue_time'.",
                flush=True,
            )
            column = "issue_time"
        else:
            raise KeyError(f"Column {column!r} not found in {issue_times_csv}. Available columns: {list(df.columns)}")
    values = [normalize_timestamp(value) for value in df[column].dropna().tolist()]
    if not values:
        raise ValueError(f"No timestamps found in {issue_times_csv}:{column}")
    return values


def load_rows(
    index_csv: Path,
    split: str,
    n: int,
    seed: int,
    issue_times_csv: Path | None = None,
    issue_time_column: str = "issue_time",
    prefer_center_patch: bool = True,
    issue_match_tolerance_minutes: float = 75.0,
) -> pd.DataFrame:
    if not Path(index_csv).exists():
        raise FileNotFoundError(
            f"Index CSV not found: {index_csv}\n"
            "Build Stage 1 indices/caches first, for example:\n"
            "  bash /code/Final_3_stage_pipeline/stage_1/run_stage1_build_indices_and_cache.sh"
    )
    df = pd.read_csv(index_csv)
    if issue_times_csv is not None:
        requested = load_requested_issue_times(issue_times_csv, issue_time_column)
        if "issue_time_bjt" not in df.columns:
            raise KeyError(f"Stage 1 index lacks issue_time_bjt, cannot align to {issue_times_csv}")
        df = df.copy()
        df["_issue_match_key"] = pd.to_datetime(df["issue_time_bjt"]).map(normalize_timestamp)
        if prefer_center_patch and "is_center" in df.columns:
            df = df.sort_values(["_issue_match_key", "is_center"], ascending=[True, False])
        else:
            df = df.sort_values("_issue_match_key")
        grouped = {key: group.iloc[0] for key, group in df.groupby("_issue_match_key", sort=False)}
        unique_times = sorted(grouped)
        tolerance = pd.Timedelta(minutes=float(issue_match_tolerance_minutes))
        rows = []
        missing = []
        matched = []
        for issue in requested:
            row = grouped.get(issue)
            matched_time = issue
            if row is None and unique_times:
                nearest = min(unique_times, key=lambda candidate: abs(candidate - issue))
                if abs(nearest - issue) <= tolerance:
                    row = grouped.get(nearest)
                    matched_time = nearest
            if row is None:
                missing.append(str(issue))
            else:
                out = row.drop(labels=["_issue_match_key"], errors="ignore").copy()
                out["requested_issue_time"] = str(issue)
                out["matched_issue_time_bjt"] = str(matched_time)
                out["issue_match_delta_minutes"] = float((matched_time - issue).total_seconds() / 60.0)
                matched.append((issue, matched_time))
                rows.append(out)
        if missing:
            examples = ", ".join(missing[:8])
            raise ValueError(
                f"Could not match {len(missing)} requested issue timestamps to Stage 1 issue_time_bjt. "
                f"Examples: {examples}. Matching tolerance={float(issue_match_tolerance_minutes):.1f} min."
            )
        for requested, actual in matched:
            if requested != actual:
                delta = (actual - requested).total_seconds() / 60.0
                print(f"[align] requested {requested} matched Stage1 issue_time_bjt {actual} (delta {delta:+.0f} min)", flush=True)
        return pd.DataFrame(rows).reset_index(drop=True)

    if "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)].copy()
    if df.empty:
        raise ValueError(f"No rows found in {index_csv} for split={split}")
    if len(df) <= int(n):
        return df.reset_index(drop=True)
    return df.sample(n=int(n), random_state=int(seed)).sort_index().reset_index(drop=True)


def finite_percentiles(arrays: list[np.ndarray], low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(vals, low))
    hi = float(np.nanpercentile(vals, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def downsample_mean(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape[-2], arr.shape[-1]
    y_edges = np.linspace(0, h, int(out_h) + 1).round().astype(int)
    x_edges = np.linspace(0, w, int(out_w) + 1).round().astype(int)
    out = np.zeros((int(out_h), int(out_w)), dtype=np.float32)
    for iy in range(int(out_h)):
        ys = slice(y_edges[iy], max(y_edges[iy + 1], y_edges[iy] + 1))
        for ix in range(int(out_w)):
            xs = slice(x_edges[ix], max(x_edges[ix + 1], x_edges[ix] + 1))
            block = arr[ys, xs]
            valid = block[np.isfinite(block)]
            out[iy, ix] = float(valid.mean()) if valid.size else float("nan")
    return out


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def color_for_value(value: float, lo: float, hi: float) -> str:
    if not np.isfinite(value):
        value = lo
    t = clamp((float(value) - float(lo)) / max(float(hi) - float(lo), 1e-6))
    for index in range(len(COLOR_STOPS) - 1):
        x0, c0 = COLOR_STOPS[index]
        x1, c1 = COLOR_STOPS[index + 1]
        if x0 <= t <= x1:
            local = (t - x0) / max(x1 - x0, 1e-9)
            rgb = [int(round(c0[i] + (c1[i] - c0[i]) * local)) for i in range(3)]
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    rgb = COLOR_STOPS[-1][1]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def rgb_for_value(value: float, lo: float, hi: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        value = lo
    t = clamp((float(value) - float(lo)) / max(float(hi) - float(lo), 1e-6))
    for index in range(len(COLOR_STOPS) - 1):
        x0, c0 = COLOR_STOPS[index]
        x1, c1 = COLOR_STOPS[index + 1]
        if x0 <= t <= x1:
            local = (t - x0) / max(x1 - x0, 1e-9)
            return tuple(int(round(c0[i] + (c1[i] - c0[i]) * local)) for i in range(3))
    return COLOR_STOPS[-1][1]


def heatmap_svg(arr: np.ndarray, x: float, y: float, width: float, height: float, lo: float, hi: float, opacity: float) -> str:
    grid_h, grid_w = arr.shape
    cell_w = float(width) / float(grid_w)
    cell_h = float(height) / float(grid_h)
    parts = []
    for iy in range(grid_h):
        for ix in range(grid_w):
            parts.append(
                f'<rect x="{x + ix * cell_w:.3f}" y="{y + iy * cell_h:.3f}" '
                f'width="{cell_w + 0.04:.3f}" height="{cell_h + 0.04:.3f}" '
                f'fill="{color_for_value(float(arr[iy, ix]), lo, hi)}" fill-opacity="{opacity:.3f}"/>'
            )
    return "\n".join(parts)


def svg_header(width: float, height: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}" '
        f'width="{width:.0f}" height="{height:.0f}">'
        '<defs><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.18"/>'
        '</filter></defs>'
    )


def outline_svg(x: float, y: float, width: float, height: float, rx: float = 4.0) -> str:
    return (
        f'<rect x="{x + 0.5:.3f}" y="{y + 0.5:.3f}" width="{width - 1:.3f}" height="{height - 1:.3f}" '
        f'rx="{rx:.3f}" fill="none" stroke="#334155" stroke-opacity="0.32" stroke-width="1"/>'
    )


def sheet_svg(arr: np.ndarray, x: float, y: float, width: float, height: float, lo: float, hi: float, opacity: float) -> str:
    return "\n".join(
        [
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" rx="4" fill="#ffffff" filter="url(#shadow)"/>',
            heatmap_svg(arr, x, y, width, height, lo, hi, opacity),
            outline_svg(x, y, width, height),
        ]
    )


def make_history_frame_svg(
    x_sample: np.ndarray,
    frame_index: int,
    channel_indices: list[int],
    limits: dict[int, tuple[float, float]],
    aspect: float,
    args: argparse.Namespace,
) -> str:
    panel_w = float(args.history_width)
    panel_h = panel_w / max(float(aspect), 1e-6)
    offset = float(args.channel_offset_px)
    svg_w = panel_w + offset * (len(channel_indices) - 1) + 18
    svg_h = panel_h + offset * (len(channel_indices) - 1) + 18
    grid_w = int(args.grid)
    grid_h = max(4, int(round(grid_w / max(float(aspect), 1e-6))))
    parts = [svg_header(svg_w, svg_h)]
    for order, channel_index in enumerate(channel_indices):
        dx = order * offset
        dy = order * offset
        arr = downsample_mean(x_sample[frame_index, channel_index], grid_h, grid_w)
        lo, hi = limits[channel_index]
        opacity = 0.82 if order < len(channel_indices) - 1 else 1.0
        parts.append(sheet_svg(arr, 8 + dx, 8 + dy, panel_w, panel_h, lo, hi, opacity))
    parts.append("</svg>")
    return "\n".join(parts)


def make_future_frame_svg(y_sample: np.ndarray, frame_index: int, limits: tuple[float, float], aspect: float, args: argparse.Namespace) -> str:
    panel_w = float(args.future_width)
    panel_h = panel_w / max(float(aspect), 1e-6)
    svg_w = panel_w + 18
    svg_h = panel_h + 18
    grid_w = int(args.grid)
    grid_h = max(4, int(round(grid_w / max(float(aspect), 1e-6))))
    arr = downsample_mean(y_sample[frame_index], grid_h, grid_w)
    parts = [svg_header(svg_w, svg_h), sheet_svg(arr, 9, 9, panel_w, panel_h, limits[0], limits[1], 1.0), "</svg>"]
    return "\n".join(parts)


def strip_svg(items: list[str], gap: float) -> str:
    bodies = []
    sizes = []
    for item in items:
        head = item.split(">", 1)[0]
        view_box = head.split('viewBox="', 1)[1].split('"', 1)[0].split()
        width, height = float(view_box[2]), float(view_box[3])
        body = item.split(">", 1)[1].rsplit("</svg>", 1)[0]
        bodies.append(body)
        sizes.append((width, height))
    out_w = sum(width for width, _height in sizes) + float(gap) * (len(sizes) - 1)
    out_h = max(height for _width, height in sizes)
    parts = [svg_header(out_w, out_h)]
    cursor = 0.0
    for body, (width, _height) in zip(bodies, sizes):
        parts.append(f'<g transform="translate({cursor:.3f},0)">{body}</g>')
        cursor += width + float(gap)
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap_png(arr: np.ndarray, width_px: int, height_px: int, lo: float, hi: float, opacity: float) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    grid_h, grid_w = arr.shape
    rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
    alpha = int(round(255 * clamp(float(opacity))))
    for iy in range(grid_h):
        for ix in range(grid_w):
            rgba[iy, ix, :3] = rgb_for_value(float(arr[iy, ix]), lo, hi)
            rgba[iy, ix, 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA")
    return image.resize((int(width_px), int(height_px)), Image.Resampling.BICUBIC)


def paste_shadowed_sheet(
    canvas: Image.Image,
    arr: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    lo: float,
    hi: float,
    opacity: float,
    scale: float,
) -> None:
    x_px = int(round(float(x) * float(scale)))
    y_px = int(round(float(y) * float(scale)))
    width_px = int(round(float(width) * float(scale)))
    height_px = int(round(float(height) * float(scale)))
    radius_px = max(1, int(round(4.0 * float(scale))))

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [x_px, y_px, x_px + width_px, y_px + height_px],
        radius=radius_px,
        fill=(15, 23, 42, 42),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1.0, 2.0 * float(scale))))
    canvas.alpha_composite(shadow, (0, 0))

    sheet = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 255))
    heat = heatmap_png(arr, width_px, height_px, lo, hi, opacity)
    sheet.alpha_composite(heat, (0, 0))

    rounded_mask = Image.new("L", (width_px, height_px), 0)
    mask_draw = ImageDraw.Draw(rounded_mask)
    mask_draw.rounded_rectangle([0, 0, width_px - 1, height_px - 1], radius=radius_px, fill=255)
    clipped = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    clipped.paste(sheet, (0, 0), rounded_mask)
    border = ImageDraw.Draw(clipped)
    border.rounded_rectangle(
        [0, 0, width_px - 1, height_px - 1],
        radius=radius_px,
        outline=(51, 65, 85, 82),
        width=max(1, int(round(float(scale)))),
    )
    canvas.alpha_composite(clipped, (x_px, y_px))


def make_history_frame_png(
    x_sample: np.ndarray,
    frame_index: int,
    channel_indices: list[int],
    limits: dict[int, tuple[float, float]],
    aspect: float,
    args: argparse.Namespace,
) -> Image.Image:
    panel_w = float(args.history_width)
    panel_h = panel_w / max(float(aspect), 1e-6)
    offset = float(args.channel_offset_px)
    nominal_w = panel_w + offset * (len(channel_indices) - 1) + 18
    nominal_h = panel_h + offset * (len(channel_indices) - 1) + 18
    scale = float(args.png_scale)
    canvas = Image.new("RGBA", (int(round(nominal_w * scale)), int(round(nominal_h * scale))), (255, 255, 255, 0))
    grid_w = int(args.grid)
    grid_h = max(4, int(round(grid_w / max(float(aspect), 1e-6))))
    for order, channel_index in enumerate(channel_indices):
        dx = order * offset
        dy = order * offset
        arr = downsample_mean(x_sample[frame_index, channel_index], grid_h, grid_w)
        lo, hi = limits[channel_index]
        opacity = 0.82 if order < len(channel_indices) - 1 else 1.0
        paste_shadowed_sheet(canvas, arr, 8 + dx, 8 + dy, panel_w, panel_h, lo, hi, opacity, scale)
    return canvas


def make_future_frame_png(
    y_sample: np.ndarray,
    frame_index: int,
    limits: tuple[float, float],
    aspect: float,
    args: argparse.Namespace,
) -> Image.Image:
    panel_w = float(args.future_width)
    panel_h = panel_w / max(float(aspect), 1e-6)
    nominal_w = panel_w + 18
    nominal_h = panel_h + 18
    scale = float(args.png_scale)
    canvas = Image.new("RGBA", (int(round(nominal_w * scale)), int(round(nominal_h * scale))), (255, 255, 255, 0))
    grid_w = int(args.grid)
    grid_h = max(4, int(round(grid_w / max(float(aspect), 1e-6))))
    arr = downsample_mean(y_sample[frame_index], grid_h, grid_w)
    paste_shadowed_sheet(canvas, arr, 9, 9, panel_w, panel_h, limits[0], limits[1], 1.0, scale)
    return canvas


def strip_png(items: list[Image.Image], gap: float, scale: float) -> Image.Image:
    gap_px = int(round(float(gap) * float(scale)))
    width = sum(item.width for item in items) + gap_px * (len(items) - 1)
    height = max(item.height for item in items)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    cursor = 0
    for item in items:
        canvas.alpha_composite(item, (cursor, 0))
        cursor += item.width + gap_px
    return canvas


def save_png(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(int(dpi), int(dpi)))


def row_meta(row: pd.Series) -> dict:
    keep = [
        "split",
        "nc_path",
        "issue_time_bjt",
        "requested_issue_time",
        "matched_issue_time_bjt",
        "issue_match_delta_minutes",
        "patch_id",
        "is_center",
        "x_start_idx",
        "x_end_idx",
        "y_start_idx",
        "y_end_idx",
        "lat_start_idx",
        "lat_end_idx",
        "lon_start_idx",
        "lon_end_idx",
    ]
    return {key: str(row[key]) for key in keep if key in row.index}


def main() -> None:
    args = parse_args()
    configure_paths(args)
    channels = split_csv(args.channels)
    inner_draw_channels = split_csv(args.draw_channels)
    draw_channels = history_draw_channels(inner_draw_channels)
    missing_draw = [channel for channel in draw_channels if channel not in channels]
    if missing_draw:
        raise ValueError(f"--draw_channels must be included in --channels; missing: {missing_draw}")

    rows = load_rows(
        args.index_csv,
        args.split,
        int(args.num_groups),
        int(args.seed),
        args.issue_times_csv,
        str(args.issue_time_column),
        bool(int(args.prefer_center_patch)),
        float(args.issue_match_tolerance_minutes),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reader = RealH8Reader(args.frame_cache_dir, channels)
    manifest = {
        "mode": args.mode,
        "split": args.split,
        "index_csv": str(args.index_csv),
        "frame_cache_dir": str(args.frame_cache_dir),
        "channels": channels,
        "inner_draw_channels": inner_draw_channels,
        "rendered_history_channels_back_to_front": draw_channels,
        "history_front_sheet": TARGET_CHANNEL,
        "history_stack_direction": "top_left_to_bottom_right",
        "format": args.format,
        "png_dpi": int(args.png_dpi),
        "png_scale": float(args.png_scale),
        "grid": int(args.grid),
        "issue_times_csv": str(args.issue_times_csv) if args.issue_times_csv else None,
        "issue_time_column": str(args.issue_time_column),
        "groups": [],
    }
    try:
        for group_index, row in rows.reset_index(drop=True).iterrows():
            x, y, mask = reader.read_sample(row)
            if x.shape[0] < 4 or y.shape[0] < 3:
                raise ValueError(f"Expected at least 4 history and 3 future frames, got x={x.shape}, y={y.shape}")
            draw_indices = [channels.index(channel) for channel in draw_channels]
            aspect = float(x.shape[-1]) / max(float(x.shape[-2]), 1.0)
            channel_limits = {
                channel_index: finite_percentiles([x[:, channel_index]], 2.0, 98.0)
                for channel_index in draw_indices
            }
            y_for_limits = np.where(mask[:3].astype(bool), y[:3], np.nan)
            y_limits = finite_percentiles([y_for_limits], 2.0, 98.0)

            group_dir = args.out_dir / f"group_{group_index + 1:02d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            history_svgs = []
            future_svgs = []
            history_pngs = []
            future_pngs = []
            for frame_index in range(4):
                if args.format in {"png", "both"}:
                    image = make_history_frame_png(x, frame_index, draw_indices, channel_limits, aspect, args)
                    name = f"history_h8_multichannel_frame_{frame_index + 1:02d}.png"
                    save_png(image, group_dir / name, int(args.png_dpi))
                    history_pngs.append(image)
                if args.format in {"svg", "both"}:
                    svg = make_history_frame_svg(x, frame_index, draw_indices, channel_limits, aspect, args)
                    name = f"history_h8_multichannel_frame_{frame_index + 1:02d}.svg"
                    (group_dir / name).write_text(svg, encoding="utf-8")
                    history_svgs.append(svg)
            for frame_index in range(3):
                if args.format in {"png", "both"}:
                    image = make_future_frame_png(y, frame_index, y_limits, aspect, args)
                    name = f"future_swr_frame_{frame_index + 1:02d}.png"
                    save_png(image, group_dir / name, int(args.png_dpi))
                    future_pngs.append(image)
                if args.format in {"svg", "both"}:
                    svg = make_future_frame_svg(y, frame_index, y_limits, aspect, args)
                    name = f"future_swr_frame_{frame_index + 1:02d}.svg"
                    (group_dir / name).write_text(svg, encoding="utf-8")
                    future_svgs.append(svg)
            if args.format in {"png", "both"}:
                save_png(
                    strip_png(history_pngs, float(args.history_strip_gap), float(args.png_scale)),
                    group_dir / "history_h8_multichannel_4frame_strip.png",
                    int(args.png_dpi),
                )
                save_png(
                    strip_png(future_pngs, float(args.future_strip_gap), float(args.png_scale)),
                    group_dir / "future_swr_3frame_strip.png",
                    int(args.png_dpi),
                )
            if args.format in {"svg", "both"}:
                (group_dir / "history_h8_multichannel_4frame_strip.svg").write_text(
                    strip_svg(history_svgs, float(args.history_strip_gap)),
                    encoding="utf-8",
                )
                (group_dir / "future_swr_3frame_strip.svg").write_text(
                    strip_svg(future_svgs, float(args.future_strip_gap)),
                    encoding="utf-8",
                )
            (group_dir / "sample_meta.json").write_text(
                json.dumps(
                    {
                        "sample": row_meta(row),
                        "x_shape": list(x.shape),
                        "y_shape": list(y.shape),
                        "aspect_width_over_height": aspect,
                        "channel_limits": {
                            channels[channel_index]: [float(v) for v in limits]
                            for channel_index, limits in channel_limits.items()
                        },
                        "future_swr_limits": [float(v) for v in y_limits],
                        "format": args.format,
                        "png_dpi": int(args.png_dpi),
                        "png_scale": float(args.png_scale),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifest["groups"].append({"group": f"group_{group_index + 1:02d}", "sample": row_meta(row)})
            print(f"wrote {group_dir}", flush=True)
    finally:
        reader.close()

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: {args.out_dir}")


if __name__ == "__main__":
    main()
