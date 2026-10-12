#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPT_DIR))

from train_fullfield_dense_temporal_forecast import _load_split_df, _parse_time_list
from train_fullfield_dense_temporal_residual_unet import _grid_lonlat, _time_solar_features
from train_fullfield_dense_temporal_residual_unet_v2 import ReadOnlyDiskFrameCache
from train_fullfield_multisource_residual_unet import (
    NWP_NAMES,
    _compute_or_load_norm_stats,
    _load_multisource_cache,
    _normalize_h8,
    _normalize_nwp,
)
from visualize_regional_split_frames import _bounds, ensure_dir, write_json
import stage3_paths as p3

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": str(p3.TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(p3.TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(p3.MULTISOURCE_CACHE_DIR),
    "out_dir": str(p3.MULTISOURCE_VIS_DIR),
    "sample_split": "test",
    "sample_count": 5,
    "seed": 2026,
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "past_frames": 12,
    "future_frames": 8,
    "freq_minutes": 15,
    "nwp_horizons": "1,4,8",
    "dpi": 150,
}


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly visualize all cached PV/H8/NWP/time-solar features for the new station-field-gain setup."
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["frame_cache_dir"]))
    parser.add_argument("--multisource_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["multisource_cache_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--sample_split", type=str, default=str(PYCHARM_DEFAULTS["sample_split"]), choices=["train", "val", "test"])
    parser.add_argument("--sample_count", type=int, default=int(PYCHARM_DEFAULTS["sample_count"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--past_frames", type=int, default=int(PYCHARM_DEFAULTS["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(PYCHARM_DEFAULTS["future_frames"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--nwp_horizons", type=str, default=str(PYCHARM_DEFAULTS["nwp_horizons"]))
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def _masked(field: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    arr = np.asarray(field, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32) <= 0.5
    return np.ma.masked_where(m, arr)


def _plot_grid_images(fig, axes, arrays, titles, extent, cmap, vmin=None, vmax=None, colorbar=True):
    for ax, arr, title in zip(axes.reshape(-1), arrays, titles):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.reshape(-1)[len(arrays):]:
        ax.axis("off")


def _parse_horizons(text: str, future_frames: int) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if 1 <= h <= future_frames:
            out.append(h - 1)
    return out or [0, min(3, future_frames - 1), future_frames - 1]


def _time_df_from_ms(ms_cache: dict) -> pd.DataFrame:
    df = ms_cache["time_df"].copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    return df


def _issue_index(time_df: pd.DataFrame) -> dict[int, int]:
    return {int(sec): i for i, sec in enumerate(time_df["issue_sec"].astype(np.int64).tolist())}


def _sample_rows(time_df: pd.DataFrame, split: str, count: int, seed: int) -> pd.DataFrame:
    sub = time_df[time_df["time_split"].astype(str) == str(split)].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No issue rows for split={split}")
    rng = np.random.default_rng(int(seed))
    n = min(int(count), len(sub))
    idx = np.sort(rng.choice(len(sub), size=n, replace=False))
    return sub.iloc[idx].reset_index(drop=True)


def _load_optional_ms_arrays(cache_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for key, name in {
        "h8_age": "h8_age_minutes_float16.npy",
        "h8_sec": "h8_frame_sec.npy",
        "nwp_dist": "nwp_nearest_distance_km_float16.npy",
    }.items():
        path = cache_dir / name
        if path.exists():
            out[key] = np.load(path, mmap_mode="r")
    return out


def render_pv_sequence(sample_id: int, row: pd.Series, pv_cache: ReadOnlyDiskFrameCache, extent, out_dir: Path, args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    issue = pd.Timestamp(row["issue_time"])
    past_times = _parse_time_list(row["past_times"])
    future_times = _parse_time_list(row["future_times"])
    fields, masks, titles = [], [], []
    for ts in past_times:
        f, m = pv_cache.get(ts)
        fields.append(_masked(f, m))
        masks.append(m)
        titles.append(f"past {pd.Timestamp(ts):%H:%M}")
    for ts in future_times:
        f, m = pv_cache.get(ts)
        fields.append(_masked(f, m))
        masks.append(m)
        titles.append(f"future {pd.Timestamp(ts):%H:%M}")

    finite = np.concatenate([np.asarray(f.compressed(), dtype=np.float32) for f in fields if f.count() > 0])
    vmax = float(np.nanquantile(finite, 0.98)) if finite.size else 1.0
    vmax = max(0.3, min(float(args.pv_field_clip) if hasattr(args, "pv_field_clip") else 1.5, vmax))

    fig, axes = plt.subplots(5, 4, figsize=(15, 16), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    _plot_grid_images(fig, axes, fields, titles, extent, cmap, vmin=0.0, vmax=vmax)
    fig.suptitle(f"PV_norm q95-adjusted field sequence | sample={sample_id:03d} | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
    fig.tight_layout()
    path = out_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_pv_sequence.png"
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)

    fig, axes = plt.subplots(5, 4, figsize=(15, 16), squeeze=False)
    _plot_grid_images(fig, axes, masks, titles, extent, plt.get_cmap("magma"), vmin=0.0, vmax=1.0)
    fig.suptitle(f"PV field support masks | sample={sample_id:03d} | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
    fig.tight_layout()
    mask_path = out_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_pv_masks.png"
    fig.savefig(mask_path, dpi=int(args.dpi))
    plt.close(fig)
    return {"pv_sequence": str(path), "pv_masks": str(mask_path)}


def render_h8_time_features(
    sample_id: int,
    row: pd.Series,
    ms_cache: dict,
    ms_optional: dict,
    stats: dict,
    issue_to_idx: dict[int, int],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    extent,
    out_dir: Path,
    args,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    issue = pd.Timestamp(row["issue_time"])
    idx = issue_to_idx[int(row["issue_sec"])]
    h8_raw = np.asarray(ms_cache["h8"][idx], dtype=np.float32)
    h8_norm = _normalize_h8(h8_raw, stats)
    h8_sec = np.asarray(ms_optional.get("h8_sec", np.zeros((len(ms_cache["time_df"]), h8_raw.shape[0]), dtype=np.int64))[idx])
    h8_age = float(np.asarray(ms_optional.get("h8_age", np.zeros((len(ms_cache["time_df"]),), dtype=np.float32))[idx]))

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    raw_titles, norm_titles = [], []
    for j in range(h8_raw.shape[0]):
        if h8_sec is not None and np.asarray(h8_sec).size > j and int(h8_sec[j]) > 0:
            ts_label = pd.Timestamp(int(h8_sec[j]), unit="s").strftime("%H:%M")
        else:
            ts_label = f"frame {j + 1}"
        raw_titles.append(f"H8/SWR raw {j + 1}\n{ts_label}")
        norm_titles.append(f"H8/SWR z {j + 1}\n{ts_label}")
    vmax = float(np.nanquantile(h8_raw[np.isfinite(h8_raw)], 0.98)) if np.isfinite(h8_raw).any() else 1.0
    _plot_grid_images(fig, axes[0:1], h8_raw, raw_titles, extent, cmap, vmin=0.0, vmax=max(0.1, vmax))
    zn = float(np.nanquantile(np.abs(h8_norm[np.isfinite(h8_norm)]), 0.98)) if np.isfinite(h8_norm).any() else 2.0
    _plot_grid_images(fig, axes[1:2], h8_norm, norm_titles, extent, plt.get_cmap("coolwarm"), vmin=-max(1.0, zn), vmax=max(1.0, zn))

    future_times = _parse_time_list(row["future_times"])
    selected_times = [pd.Timestamp(row["issue_time"]), future_times[0], future_times[min(3, len(future_times) - 1)], future_times[-1]]
    arrays, titles = [], []
    for ts in selected_times:
        feat = _time_solar_features(ts, lon_grid, lat_grid)
        arrays.append(feat[2])
        titles.append(f"solar elevation scale\n{pd.Timestamp(ts):%H:%M}")
    _plot_grid_images(fig, axes[2:3], arrays, titles, extent, plt.get_cmap("inferno"), vmin=0.0, vmax=1.0)
    fig.suptitle(f"H8 and time/solar features | sample={sample_id:03d} | issue={issue:%Y-%m-%d %H:%M:%S} | H8 age={h8_age:.0f} min", y=0.995)
    fig.tight_layout()
    path = out_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_h8_time_features.png"
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)
    return {"h8_time_features": str(path)}


def render_nwp_features(
    sample_id: int,
    row: pd.Series,
    ms_cache: dict,
    stats: dict,
    issue_to_idx: dict[int, int],
    extent,
    out_dir: Path,
    args,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    issue = pd.Timestamp(row["issue_time"])
    idx = issue_to_idx[int(row["issue_sec"])]
    nwp_raw = np.asarray(ms_cache["nwp"][idx], dtype=np.float32)
    nwp_valid = np.asarray(ms_cache["nwp_valid"][idx], dtype=np.uint8)
    nwp_norm = _normalize_nwp(nwp_raw, nwp_valid, stats, int(args.grid_size))
    horizons = _parse_horizons(str(args.nwp_horizons), int(args.future_frames))
    future_times = _parse_time_list(row["future_times"])

    fig, axes = plt.subplots(len(horizons) * 2, 4, figsize=(17, 4.2 * len(horizons) * 2), squeeze=False)
    for rr, h in enumerate(horizons):
        for c, name in enumerate(NWP_NAMES):
            raw = nwp_raw[h, c]
            ax = axes[rr * 2, c]
            finite = np.isfinite(raw)
            lo = float(np.nanquantile(raw[finite], 0.02)) if finite.any() else 0.0
            hi = float(np.nanquantile(raw[finite], 0.98)) if finite.any() else 1.0
            if not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            im = ax.imshow(np.ma.masked_invalid(raw), origin="lower", extent=extent, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
            ax.set_title(f"raw {name} h{h + 1}\n{pd.Timestamp(future_times[h]):%H:%M} valid={int(nwp_valid[h, c])}", fontsize=9)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            norm = nwp_norm[h, c]
            ax = axes[rr * 2 + 1, c]
            z = float(np.nanquantile(np.abs(norm[np.isfinite(norm)]), 0.98)) if np.isfinite(norm).any() else 2.0
            im = ax.imshow(np.ma.masked_invalid(norm), origin="lower", extent=extent, cmap="coolwarm", vmin=-max(1.0, z), vmax=max(1.0, z), interpolation="nearest", aspect="auto")
            ax.set_title(f"z-score {name} h{h + 1}", fontsize=9)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"NWP future features | sample={sample_id:03d} | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.997)
    fig.tight_layout()
    path = out_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_nwp_features.png"
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)
    return {"nwp_features": str(path)}


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "sample_feature_figures")

    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    lon_grid, lat_grid = _grid_lonlat(bounds, int(args.grid_size))

    ms_cache = _load_multisource_cache(Path(args.multisource_cache_dir))
    ms_cache["time_df"] = _time_df_from_ms(ms_cache)
    ms_optional = _load_optional_ms_arrays(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"]
    stats = _compute_or_load_norm_stats(ms_cache, out_dir, time_df, force=False)
    issue_to_idx = _issue_index(time_df)
    pv_cache = ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)

    samples = _sample_rows(time_df, str(args.sample_split), int(args.sample_count), int(args.seed))
    samples.to_csv(out_dir / "sampled_issue_times.csv", index=False)
    rows = []
    for i, row in samples.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        print(f"[visualize_cache_features] sample {i + 1}/{len(samples)} issue={issue}", flush=True)
        paths = {}
        paths.update(render_pv_sequence(i, row, pv_cache, extent, fig_dir, args))
        paths.update(render_h8_time_features(i, row, ms_cache, ms_optional, stats, issue_to_idx, lon_grid, lat_grid, extent, fig_dir, args))
        paths.update(render_nwp_features(i, row, ms_cache, stats, issue_to_idx, extent, fig_dir, args))
        rows.append({"sample_id": int(i), "issue_time": str(issue), **paths})

    pd.DataFrame(rows).to_csv(out_dir / "sample_feature_figure_index.csv", index=False)
    write_json(
        out_dir / "02_cache_feature_visualization_meta.json",
        {
            "purpose": "Random visual QA for new target-excluded PV_norm field cache and aligned H8/NWP/time-solar features.",
            "field_value": "PV_norm / station_train_PV_norm_q95; no clear-sky/CSI normalization",
            "sample_split": str(args.sample_split),
            "sample_count": int(len(samples)),
            "frame_cache_dir": str(args.frame_cache_dir),
            "multisource_cache_dir": str(args.multisource_cache_dir),
            "norm_stats": stats,
            "figures": str(fig_dir),
        },
    )
    print("==== New cache feature visualization complete ====")
    print(f"samples: {out_dir / 'sampled_issue_times.csv'}")
    print(f"figure index: {out_dir / 'sample_feature_figure_index.csv'}")
    print(f"figures: {fig_dir}")


if __name__ == "__main__":
    main()
