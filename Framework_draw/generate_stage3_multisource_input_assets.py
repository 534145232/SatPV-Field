#!/usr/bin/env python3
"""Export clean framework-figure assets for Stage 3 multisource inputs.

The script reads real Stage 3 training caches and writes axis-free PNG assets:
historical PV fields, historical PV support masks, coarse NWP fields, past SWR
fields, future clear-sky GHI/time-encoding fields, and future SWR anchors.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import (  # noqa: E402
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
    MULTISOURCE_CACHE_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
)
from train_fullfield_dense_temporal_forecast import _parse_time_list  # noqa: E402
from train_fullfield_dense_temporal_residual_unet import _grid_lonlat, _time_solar_features  # noqa: E402
from train_fullfield_dense_temporal_residual_unet_v2 import ReadOnlyDiskFrameCache  # noqa: E402
from visualize_regional_split_frames import _bounds  # noqa: E402
from generate_real_h8_swr_svg_assets import COLOR_STOPS  # noqa: E402


NWP_NAMES = ["radiation", "direction10", "speed10", "tcc"]
DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_multisource_input_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean PNG assets from real Stage 3 multisource input caches."
    )
    parser.add_argument("--multisource_cache_dir", type=Path, default=MULTISOURCE_CACHE_DIR)
    parser.add_argument("--pv_frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--stack_size", type=int, default=300)
    parser.add_argument("--stack_offset", type=int, default=16)
    parser.add_argument("--nwp_horizons", type=str, default="1,4,8")
    parser.add_argument("--pv_frames", type=str, default="all", help="'all' or comma-separated 1-based history frames.")
    parser.add_argument("--swr_history_frames", type=int, default=4)
    parser.add_argument("--swr_future_anchors", type=int, default=3)
    parser.add_argument("--preview_labels", type=int, default=0, help="Draw small labels on preview sheets only.")
    parser.add_argument("--gaussian_sigma_cells", type=float, default=1.0)
    parser.add_argument("--gaussian_radius_cells", type=int, default=2)
    parser.add_argument("--min_grid_weight", type=float, default=1e-6)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def load_multisource_cache(cache_dir: Path) -> dict:
    required = {
        "time_df": "issue_time_index.csv",
        "h8": "h8_swr_64_float16.npy",
        "nwp": "nwp_future_fields_float16.npy",
        "nwp_valid": "nwp_future_valid_uint8.npy",
        "meta": "multisource_cache_meta.json",
    }
    missing = [str(cache_dir / name) for name in required.values() if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Stage 3 multisource cache is incomplete:\n" + "\n".join(missing))
    with (cache_dir / required["meta"]).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    time_df = pd.read_csv(cache_dir / required["time_df"])
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    return {
        "dir": cache_dir,
        "time_df": time_df,
        "h8": np.load(cache_dir / required["h8"], mmap_mode="r"),
        "nwp": np.load(cache_dir / required["nwp"], mmap_mode="r"),
        "nwp_valid": np.load(cache_dir / required["nwp_valid"], mmap_mode="r"),
        "meta": meta,
    }


def resolve_anchor_cache(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    manifest = CENTER_H8_SWR_ANCHOR_CACHE_ROOT / "active_cache.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            candidate = Path(json.load(handle)["active_cache_dir"])
        if candidate.exists():
            return candidate
    return CENTER_H8_SWR_ANCHOR_CACHE_ROOT


def load_swr_anchor_cache(cache_dir: Path) -> dict:
    required = [
        "swr_anchor_64.npy",
        "center_h8_swr_anchor_input_cache_meta.csv",
        "center_h8_swr_anchor_input_cache_meta.json",
    ]
    missing = [str(cache_dir / name) for name in required if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Stage 3 future-SWR anchor cache is incomplete:\n" + "\n".join(missing))
    meta = pd.read_csv(cache_dir / "center_h8_swr_anchor_input_cache_meta.csv")
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    meta["issue_sec"] = meta["issue_time"].map(timestamp_key)
    return {
        "dir": cache_dir,
        "anchor": np.load(cache_dir / "swr_anchor_64.npy", mmap_mode="r"),
        "meta": meta,
        "by_issue_sec": {int(row["issue_sec"]): row for _, row in meta.iterrows()},
    }


def load_h8_frame_seconds(cache_dir: Path) -> np.ndarray | None:
    path = Path(cache_dir) / "h8_frame_sec.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def split_pipe_times(text: object) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def sample_rows(time_df: pd.DataFrame, split: str, count: int, seed: int) -> pd.DataFrame:
    sub = time_df[time_df["time_split"].astype(str) == str(split)].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No issue rows found for split={split}")
    rng = np.random.default_rng(int(seed))
    n = min(int(count), len(sub))
    idx = np.sort(rng.choice(len(sub), size=n, replace=False))
    return sub.iloc[idx].reset_index(drop=True)


def robust_limits(arrays: list[np.ndarray], qlo: float = 0.02, qhi: float = 0.98) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float32)
        vals.append(a[np.isfinite(a)])
    finite = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.asarray([])
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanquantile(finite, qlo))
    hi = float(np.nanquantile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite) + 1e-6)
    return lo, hi


def parse_indices(text: str, max_count: int) -> list[int]:
    if str(text).strip().lower() == "all":
        return list(range(max_count))
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part) - 1
        if 0 <= idx < max_count:
            out.append(idx)
    return out or list(range(max_count))


def parse_horizons(text: str, max_count: int) -> list[int]:
    return parse_indices(text, max_count)


def shared_real_h8_palette() -> np.ndarray:
    stops = np.asarray([rgb for _, rgb in COLOR_STOPS], dtype=np.float32)
    x = np.asarray([pos for pos, _ in COLOR_STOPS], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.uint8)


PALETTES = {
    "swr": shared_real_h8_palette(),
    "pv": shared_real_h8_palette(),
    "nwp": shared_real_h8_palette(),
}


def field_to_image(
    field: np.ndarray,
    palette: np.ndarray,
    vmin: float,
    vmax: float,
    size: int,
    mask: np.ndarray | None = None,
    interpolation: str = "nearest",
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    arr = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=np.float32) > 0.5
    masked = np.ma.masked_where(~valid, arr)

    cmap = ListedColormap(np.asarray(palette, dtype=np.float32) / 255.0)
    cmap = cmap.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    dpi = 100
    fig = plt.figure(figsize=(float(size) / dpi, float(size) / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    ax.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation=str(interpolation),
        aspect="auto",
    )
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def mask_to_image(mask: np.ndarray, size: int) -> Image.Image:
    arr = np.asarray(mask, dtype=np.float32) > 0.5
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[~arr] = np.asarray([236, 240, 245, 150], dtype=np.uint8)
    rgba[arr] = np.asarray([32, 93, 156, 245], dtype=np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    return image.resize((int(size), int(size)), Image.Resampling.NEAREST)


def solar_cos_zenith_bjt(ts: pd.Timestamp, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    ts = pd.Timestamp(ts)
    doy = int(ts.dayofyear)
    hour_bjt = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_bjt - 12.0) / 24.0)
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    time_offset_min = eq_time + 4.0 * (lon_deg - 120.0)
    true_solar_min = hour_bjt * 60.0 + time_offset_min
    hour_angle = np.deg2rad(true_solar_min / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat_deg)
    cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    return np.maximum(cosz.astype(np.float32), 0.0)


def clear_sky_ghi(ts: pd.Timestamp, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    cosz = solar_cos_zenith_bjt(pd.Timestamp(ts), lat_grid, lon_grid)
    out = np.zeros_like(cosz, dtype=np.float32)
    mask = cosz > 1.0e-4
    out[mask] = 1098.0 * cosz[mask] * np.exp(-0.059 / np.maximum(cosz[mask], 1.0e-4))
    return out.astype(np.float32)


def save_image(path: Path, image: Image.Image) -> str:
    ensure_dir(path.parent)
    image.save(path, format="PNG", optimize=True)
    return str(path)


def make_stack(images: list[Image.Image], size: int, offset: int) -> Image.Image:
    if not images:
        return Image.new("RGBA", (size, size), (255, 255, 255, 0))
    w = int(size) + int(offset) * (len(images) - 1)
    h = int(size) + int(offset) * (len(images) - 1)
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    for order, image in enumerate(images):
        x = order * int(offset)
        y = order * int(offset)
        tile = image.convert("RGBA").resize((int(size), int(size)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile, (x, y))
        draw.rectangle([x, y, x + int(size) - 1, y + int(size) - 1], outline=(215, 221, 230, 230), width=2)
    return canvas


def make_strip(images: list[Image.Image], tile: int, gap: int = 16) -> Image.Image:
    if not images:
        return Image.new("RGBA", (tile, tile), (255, 255, 255, 0))
    w = len(images) * int(tile) + (len(images) - 1) * int(gap)
    canvas = Image.new("RGBA", (w, int(tile)), (255, 255, 255, 0))
    for i, image in enumerate(images):
        tile_image = image.convert("RGBA").resize((int(tile), int(tile)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile_image, (i * (int(tile) + int(gap)), 0))
    return canvas


def draw_preview(sample_dir: Path, sections: list[tuple[str, Image.Image]], labels: bool) -> str:
    tile_h = max(image.height for _, image in sections)
    gap = 28
    label_h = 28 if labels else 0
    width = sum(image.width for _, image in sections) + gap * (len(sections) - 1)
    canvas = Image.new("RGBA", (width, tile_h + label_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, image in sections:
        if labels:
            draw.text((x, 4), title, fill=(20, 25, 35, 255))
        canvas.alpha_composite(image.convert("RGBA"), (x, label_h))
        x += image.width + gap
    path = sample_dir / "overview_multisource_inputs.png"
    canvas.save(path, format="PNG", optimize=True)
    return str(path)


def make_pv_cache(frame_cache_dir: Path, time_df: pd.DataFrame, args: argparse.Namespace) -> ReadOnlyDiskFrameCache:
    return ReadOnlyDiskFrameCache(frame_cache_dir, time_df, args)


def load_station_split(target_run_dir: Path) -> pd.DataFrame:
    candidates = [
        Path(target_run_dir) / "station_split.csv",
        Path(target_run_dir) / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, dtype={"station_sn": str})
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            return df.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)
    raise FileNotFoundError("Cannot build time/clear-sky grids; station split not found in " + str(target_run_dir))


def render_sample(
    sample_id: int,
    row: pd.Series,
    global_idx: int,
    pv_cache: ReadOnlyDiskFrameCache,
    ms_cache: dict,
    swr_cache: dict,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    issue = pd.Timestamp(row["issue_time"])
    issue_text = issue.strftime("%Y%m%d_%H%M%S")
    sample_dir = ensure_dir(Path(args.out_dir) / f"sample_{sample_id:02d}_{issue_text}")

    past_times = _parse_time_list(row["past_times"])
    future_times = _parse_time_list(row["future_times"])
    pv_indices = parse_indices(str(args.pv_frames), len(past_times))

    pv_fields, pv_masks = [], []
    for idx in pv_indices:
        field, mask = pv_cache.get(past_times[idx])
        pv_fields.append(field)
        pv_masks.append(mask)
    pv_lo, pv_hi = 0.0, max(0.3, min(1.5, robust_limits(pv_fields, 0.01, 0.99)[1]))
    pv_images = []
    pv_paths = []
    for local_i, idx in enumerate(pv_indices):
        image = field_to_image(pv_fields[local_i], PALETTES["pv"], pv_lo, pv_hi, int(args.image_size), pv_masks[local_i])
        pv_images.append(image)
        pv_paths.append(save_image(sample_dir / "historical_pv_field" / f"pv_hist_{idx + 1:02d}.png", image))
    pv_stack = make_stack(pv_images, int(args.stack_size), int(args.stack_offset))
    pv_stack_path = save_image(sample_dir / "historical_pv_field_stack.png", pv_stack)
    pv_strip_path = save_image(sample_dir / "historical_pv_field_strip.png", make_strip(pv_images, int(args.stack_size)))

    pv_mask_images = []
    pv_mask_paths = []
    for local_i, idx in enumerate(pv_indices):
        image = mask_to_image(pv_masks[local_i], int(args.image_size))
        pv_mask_images.append(image)
        pv_mask_paths.append(save_image(sample_dir / "historical_support_mask" / f"support_mask_hist_{idx + 1:02d}.png", image))
    pv_mask_stack = make_stack(pv_mask_images, int(args.stack_size), int(args.stack_offset))
    pv_mask_stack_path = save_image(sample_dir / "historical_support_mask_stack.png", pv_mask_stack)
    pv_mask_strip_path = save_image(
        sample_dir / "historical_support_mask_strip.png",
        make_strip(pv_mask_images, int(args.stack_size)),
    )
    latest_mask_path = save_image(sample_dir / "latest_support_mask.png", pv_mask_images[-1])

    swr_row = swr_cache["by_issue_sec"].get(int(row["issue_sec"]))
    if swr_row is None:
        raise KeyError(f"No SWR-anchor cache row for issue_time={issue}")
    cache_idx = int(swr_row["cache_idx"])
    anchor_item = np.asarray(swr_cache["anchor"][cache_idx], dtype=np.float32)
    restored_swr = anchor_item[:, 0] * anchor_item[:, 1] * 1000.0

    history_end = min(int(args.swr_history_frames), restored_swr.shape[0])
    past_swr_fields = [restored_swr[j] for j in range(history_end)]
    future_start = int(args.swr_history_frames)
    future_end = min(future_start + int(args.swr_future_anchors), restored_swr.shape[0])
    future_swr_fields = [restored_swr[j] for j in range(future_start, future_end)]
    swr_lo, swr_hi = 0.0, max(300.0, robust_limits(past_swr_fields + future_swr_fields, 0.01, 0.99)[1])

    past_swr_images, past_swr_paths = [], []
    for j, field in enumerate(past_swr_fields):
        image = field_to_image(field, PALETTES["swr"], swr_lo, swr_hi, int(args.image_size))
        past_swr_images.append(image)
        past_swr_paths.append(save_image(sample_dir / "past_swr_field" / f"past_swr_{j + 1:02d}.png", image))
    past_swr_stack = make_stack(past_swr_images, int(args.stack_size), int(args.stack_offset))
    past_swr_stack_path = save_image(sample_dir / "past_swr_field_stack.png", past_swr_stack)
    past_swr_strip_path = save_image(sample_dir / "past_swr_field_strip.png", make_strip(past_swr_images, int(args.stack_size)))

    nwp_raw = np.asarray(ms_cache["nwp"][global_idx], dtype=np.float32)
    nwp_valid = np.asarray(ms_cache["nwp_valid"][global_idx], dtype=np.uint8)
    nwp_horizons = parse_horizons(str(args.nwp_horizons), nwp_raw.shape[0])
    nwp_paths: dict[str, list[str]] = {}
    nwp_stack_paths: dict[str, str] = {}
    nwp_preview_images = []
    for channel, name in enumerate(NWP_NAMES[: nwp_raw.shape[1]]):
        channel_fields = [nwp_raw[h, channel] for h in nwp_horizons if int(nwp_valid[h, channel]) > 0]
        if not channel_fields:
            channel_fields = [np.zeros_like(nwp_raw[0, channel], dtype=np.float32)]
        lo, hi = robust_limits(channel_fields, 0.02, 0.98)
        images = []
        paths = []
        for h in nwp_horizons:
            field = nwp_raw[h, channel] if int(nwp_valid[h, channel]) > 0 else np.full_like(nwp_raw[h, channel], np.nan)
            image = field_to_image(field, PALETTES["nwp"], lo, hi, int(args.image_size))
            images.append(image)
            paths.append(save_image(sample_dir / "coarse_nwp_fields" / name / f"{name}_h{h + 1:02d}.png", image))
        stack = make_stack(images, int(args.stack_size), int(args.stack_offset))
        nwp_paths[name] = paths
        nwp_stack_paths[name] = save_image(sample_dir / "coarse_nwp_fields" / f"{name}_stack.png", stack)
        if channel < 2:
            nwp_preview_images.extend(images[:2])
    nwp_preview = make_stack(nwp_preview_images[:4], int(args.stack_size), int(args.stack_offset))

    future_condition_indices = nwp_horizons
    clear_fields = [clear_sky_ghi(future_times[h], lon_grid, lat_grid) for h in future_condition_indices]
    clear_hi = max(300.0, robust_limits(clear_fields, 0.01, 0.99)[1])
    clear_images, clear_paths = [], []
    for h, field in zip(future_condition_indices, clear_fields):
        image = field_to_image(field, PALETTES["swr"], 0.0, clear_hi, int(args.image_size))
        clear_images.append(image)
        clear_paths.append(save_image(sample_dir / "future_clear_sky_ghi" / f"clear_sky_ghi_h{h + 1:02d}.png", image))
    clear_stack = make_stack(clear_images, int(args.stack_size), int(args.stack_offset))
    clear_stack_path = save_image(sample_dir / "future_clear_sky_ghi_stack.png", clear_stack)
    clear_strip_path = save_image(sample_dir / "future_clear_sky_ghi_strip.png", make_strip(clear_images, int(args.stack_size)))

    time_feature_names = ["tod_sin", "tod_cos", "solar_elevation"]
    time_feature_paths: dict[str, list[str]] = {}
    time_feature_stack_paths: dict[str, str] = {}
    time_preview_images = []
    for channel, name in enumerate(time_feature_names):
        fields = [_time_solar_features(future_times[h], lon_grid, lat_grid)[channel] for h in future_condition_indices]
        if name == "tod_sin" or name == "tod_cos":
            lo, hi = -1.0, 1.0
        else:
            lo, hi = 0.0, max(0.3, robust_limits(fields, 0.01, 0.99)[1])
        images, paths = [], []
        for h, field in zip(future_condition_indices, fields):
            image = field_to_image(field, PALETTES["nwp"], lo, hi, int(args.image_size))
            images.append(image)
            paths.append(save_image(sample_dir / "future_time_encoding" / name / f"{name}_h{h + 1:02d}.png", image))
        time_feature_paths[name] = paths
        time_feature_stack_paths[name] = save_image(sample_dir / "future_time_encoding" / f"{name}_stack.png", make_stack(images, int(args.stack_size), int(args.stack_offset)))
        time_preview_images.extend(images[:1])
    time_preview = make_stack(time_preview_images[:3], int(args.stack_size), int(args.stack_offset))

    future_swr_images, future_swr_paths = [], []
    for local_i, field in enumerate(future_swr_fields):
        image = field_to_image(field, PALETTES["swr"], swr_lo, swr_hi, int(args.image_size))
        future_swr_images.append(image)
        future_swr_paths.append(save_image(sample_dir / "future_swr_anchors" / f"future_swr_tplus{local_i + 1:02d}.png", image))
    future_swr_stack = make_stack(future_swr_images, int(args.stack_size), int(args.stack_offset))
    future_swr_stack_path = save_image(sample_dir / "future_swr_anchors_stack.png", future_swr_stack)
    future_swr_strip_path = save_image(sample_dir / "future_swr_anchors_strip.png", make_strip(future_swr_images, int(args.stack_size)))

    preview_path = draw_preview(
        sample_dir,
        [
            ("Historical PV Field", pv_stack),
            ("Support Mask", pv_mask_stack),
            ("Coarse NWP Fields", nwp_preview),
            ("Clear-sky GHI", clear_stack),
            ("Time Encoding", time_preview),
            ("Past SWR Field", past_swr_stack),
            ("Future SWR Anchors", future_swr_stack),
        ],
        labels=bool(int(args.preview_labels)),
    )

    return {
        "sample_id": int(sample_id),
        "issue_time": str(issue),
        "global_issue_index": int(global_idx),
        "swr_anchor_cache_idx": int(cache_idx),
        "swr_anchor_history_label_times": str(swr_row.get("history_label_times", "")),
        "swr_anchor_future_label_times": str(swr_row.get("future_anchor_label_times", "")),
        "past_swr_source": "swr_anchor_64.npy restored_swr[:,0] * clear_sky[:,1] * 1000",
        "future_swr_source": "swr_anchor_64.npy restored_swr[:,0] * clear_sky[:,1] * 1000",
        "sample_dir": str(sample_dir),
        "historical_pv_frames": pv_paths,
        "historical_pv_stack": pv_stack_path,
        "historical_pv_strip": pv_strip_path,
        "historical_support_mask_frames": pv_mask_paths,
        "historical_support_mask_stack": pv_mask_stack_path,
        "historical_support_mask_strip": pv_mask_strip_path,
        "latest_support_mask": latest_mask_path,
        "coarse_nwp_frames": nwp_paths,
        "coarse_nwp_stacks": nwp_stack_paths,
        "future_clear_sky_ghi_frames": clear_paths,
        "future_clear_sky_ghi_stack": clear_stack_path,
        "future_clear_sky_ghi_strip": clear_strip_path,
        "future_time_encoding_frames": time_feature_paths,
        "future_time_encoding_stacks": time_feature_stack_paths,
        "past_swr_frames": past_swr_paths,
        "past_swr_stack": past_swr_stack_path,
        "past_swr_strip": past_swr_strip_path,
        "future_swr_anchor_frames": future_swr_paths,
        "future_swr_anchor_stack": future_swr_stack_path,
        "future_swr_anchor_strip": future_swr_strip_path,
        "overview": preview_path,
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    ms_cache = load_multisource_cache(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"]
    issue_to_idx = {int(sec): i for i, sec in enumerate(time_df["issue_sec"].astype(np.int64).tolist())}
    h8_frame_sec = load_h8_frame_seconds(Path(args.multisource_cache_dir))
    pv_cache = make_pv_cache(Path(args.pv_frame_cache_dir), time_df, args)
    swr_cache = load_swr_anchor_cache(resolve_anchor_cache(args.swr_anchor_cache_dir))
    station_split = load_station_split(Path(args.target_run_dir))
    lon_grid, lat_grid = _grid_lonlat(_bounds(station_split), int(args.grid_size))

    samples = sample_rows(time_df, str(args.sample_split), int(args.sample_count), int(args.seed))
    samples = samples.copy()
    stage1_times = []
    anchor_future_times = []
    multisource_latest_swr_times = []
    for _, row in samples.iterrows():
        global_idx = issue_to_idx[int(row["issue_sec"])]
        swr_row = swr_cache["by_issue_sec"].get(int(row["issue_sec"]))
        if swr_row is not None:
            history_times = split_pipe_times(swr_row.get("history_label_times", ""))
            future_times = split_pipe_times(swr_row.get("future_anchor_label_times", ""))
            if history_times:
                stage1_times.append(history_times[-1].strftime("%Y-%m-%d %H:%M:%S"))
            else:
                stage1_times.append(pd.Timestamp(row["issue_time"]).strftime("%Y-%m-%d %H:%M:%S"))
            anchor_future_times.append("|".join(ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_times))
        else:
            stage1_times.append(pd.Timestamp(row["issue_time"]).strftime("%Y-%m-%d %H:%M:%S"))
            anchor_future_times.append("")

        if h8_frame_sec is not None:
            secs = np.asarray(h8_frame_sec[global_idx], dtype=np.int64)
            secs = secs[secs > 0]
            if secs.size:
                multisource_latest_swr_times.append(pd.Timestamp(int(secs[-1]), unit="s").strftime("%Y-%m-%d %H:%M:%S"))
                continue
        multisource_latest_swr_times.append((pd.Timestamp(row["issue_time"]) - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))
    samples["stage1_issue_time_bjt"] = stage1_times
    samples["future_swr_anchor_label_times"] = anchor_future_times
    samples["multisource_latest_swr_time_bjt"] = multisource_latest_swr_times
    sample_index_path = out_dir / "sampled_issue_times.csv"
    samples.to_csv(sample_index_path, index=False)

    rows = []
    for sample_id, row in samples.iterrows():
        global_idx = issue_to_idx[int(row["issue_sec"])]
        print(f"[framework_inputs] sample {sample_id + 1}/{len(samples)} issue={row['issue_time']}", flush=True)
        rows.append(render_sample(sample_id + 1, row, global_idx, pv_cache, ms_cache, swr_cache, lon_grid, lat_grid, args))

    index_path = out_dir / "asset_index.json"
    write_json(index_path, {"samples": rows})
    write_json(
        out_dir / "generation_meta.json",
        {
            "purpose": "Clean framework-figure assets for Stage 3 multisource PV-field model inputs.",
            "source": "Real Stage 3 caches: target-excluded PV field and support mask, multisource NWP/H8-SWR cache, future-SWR anchor cache.",
            "multisource_cache_dir": str(Path(args.multisource_cache_dir)),
            "pv_frame_cache_dir": str(Path(args.pv_frame_cache_dir)),
            "target_run_dir": str(Path(args.target_run_dir)),
            "swr_anchor_cache_dir": str(swr_cache["dir"]),
            "sample_split": str(args.sample_split),
            "sample_count": int(len(samples)),
            "image_size": int(args.image_size),
            "stack_size": int(args.stack_size),
            "nwp_names": NWP_NAMES,
            "shared_palette_source": "generate_real_h8_swr_svg_assets.COLOR_STOPS",
            "time_encoding_channels": ["tod_sin", "tod_cos", "solar_elevation"],
            "clear_sky_ghi_formula": "Haurwitz clear-sky GHI: 1098*cosZ*exp(-0.059/cosZ), cosZ from BJT timestamp, lat, lon.",
            "alignment_note": "sampled_issue_times.csv includes stage1_issue_time_bjt for aligning generate_real_h8_swr_svg_assets.py.",
        },
    )
    print("==== Stage 3 multisource input assets complete ====")
    print(f"samples: {sample_index_path}")
    print(f"asset index: {index_path}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
