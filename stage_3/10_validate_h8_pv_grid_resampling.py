#!/usr/bin/env python3
"""Visualize and numerically validate center-H8 to PV-grid interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_build_swr_anchor_cache_impl import _resample_to_grid
from stage3_paths import (
    H8_RESAMPLING_AUDIT_DIR,
    LEGACY_CENTER_SWR_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    ensure_stage3_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the production center-H8 to 64x64 PV-grid bilinear resampling."
    )
    parser.add_argument("--adapter_dir", type=Path, default=LEGACY_CENTER_SWR_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=H8_RESAMPLING_AUDIT_DIR)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument(
        "--timestamps",
        type=str,
        default="",
        help="Optional comma-separated hourly BJT labels, for example 2023-04-17 08:00:00,2023-04-17 11:00:00.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_station_split() -> pd.DataFrame:
    candidates = [
        TARGET_EXCLUDED_RUN_DIR / "station_split.csv",
        TARGET_SELECTION_DIR / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, dtype={"station_sn": str})
            frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
            frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
            return frame.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    raise FileNotFoundError(f"Target-excluded station split not found. Tried: {candidates}")


def pv_bounds(stations: pd.DataFrame) -> tuple[float, float, float, float]:
    lon_min = float(stations["lon"].min())
    lon_max = float(stations["lon"].max())
    lat_min = float(stations["lat"].min())
    lat_max = float(stations["lat"].max())
    lon_pad = max(1.0e-6, (lon_max - lon_min) * 0.04)
    lat_pad = max(1.0e-6, (lat_max - lat_min) * 0.04)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def target_mesh(bounds: tuple[float, float, float, float], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bounds
    lon = lon_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lon_max - lon_min)
    lat = lat_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lat_max - lat_min)
    return np.meshgrid(lon, lat)


def ascending_axes(
    values: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    lon = np.asarray(src_lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(src_lat, dtype=np.float64).reshape(-1)
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[..., ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr = arr[..., ::-1, :]
    return arr, lon, lat


def reference_bilinear(
    values: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon_axis: np.ndarray,
    dst_lat_axis: np.ndarray,
) -> np.ndarray:
    """Independent separable np.interp reference implementation for a 2D field."""
    arr, lon, lat = ascending_axes(values, src_lon, src_lat)
    if arr.ndim != 2:
        raise ValueError(f"Reference implementation expects one 2D field, got {arr.shape}")
    dst_lon_axis = np.asarray(dst_lon_axis, dtype=np.float64).reshape(-1)
    dst_lat_axis = np.asarray(dst_lat_axis, dtype=np.float64).reshape(-1)
    lon_pass = np.stack([np.interp(dst_lon_axis, lon, row) for row in arr], axis=0)
    return np.stack([np.interp(dst_lat_axis, lat, lon_pass[:, index]) for index in range(lon_pass.shape[1])], axis=1).astype(np.float32)


def nearest_neighbor(
    values: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon_axis: np.ndarray,
    dst_lat_axis: np.ndarray,
) -> np.ndarray:
    arr, lon, lat = ascending_axes(values, src_lon, src_lat)
    x = np.abs(lon[:, None] - np.asarray(dst_lon_axis, dtype=np.float64)[None, :]).argmin(axis=0)
    y = np.abs(lat[:, None] - np.asarray(dst_lat_axis, dtype=np.float64)[None, :]).argmin(axis=0)
    return arr[y[:, None], x[None, :]].astype(np.float32)


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    error = left[valid] - right[valid]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def gradient_stats(values: np.ndarray, lon_step: float, lat_step: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "lon_gradient_wm2_per_degree": float(np.nanmean(np.abs(np.diff(values, axis=-1))) / max(abs(lon_step), 1.0e-12)),
        "lat_gradient_wm2_per_degree": float(np.nanmean(np.abs(np.diff(values, axis=-2))) / max(abs(lat_step), 1.0e-12)),
    }


def legacy_shape_probe(
    prefix_shape: tuple[int, ...],
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
) -> tuple[int, ...]:
    """Return the shape produced by the old mesh-incompatible indexing pattern."""
    x = np.interp(np.asarray(dst_lon, dtype=np.float64), np.asarray(src_lon, dtype=np.float64), np.arange(len(src_lon)))
    y = np.interp(np.asarray(dst_lat, dtype=np.float64), np.asarray(src_lat, dtype=np.float64), np.arange(len(src_lat)))
    legacy_spatial = np.broadcast_shapes(y[:, None].shape, x[None, :].shape)
    return (*prefix_shape, *legacy_spatial)


def choose_rows(timestamps: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if str(args.timestamps).strip():
        requested = [pd.Timestamp(value.strip()) for value in str(args.timestamps).split(",") if value.strip()]
        indexed = timestamps.set_index("timestamp")
        missing = [str(value) for value in requested if value not in indexed.index]
        if missing:
            raise KeyError(f"Requested adapter timestamps not found: {missing}")
        return indexed.loc[requested].reset_index()
    daytime = timestamps[timestamps["timestamp"].dt.hour.between(7, 16)].reset_index(drop=True)
    if daytime.empty:
        raise RuntimeError("No daytime H8 timestamps available in the Stage 3 adapter.")
    count = min(max(int(args.num_samples), 1), len(daytime))
    indices = np.linspace(0, len(daytime) - 1, count, dtype=int)
    return daytime.iloc[indices].reset_index(drop=True)


def render_case(
    out_path: Path,
    timestamp: pd.Timestamp,
    source: np.ndarray,
    crop: np.ndarray,
    bilinear: np.ndarray,
    nearest: np.ndarray,
    reference: np.ndarray,
    reconstruction: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    crop_lon: np.ndarray,
    crop_lat: np.ndarray,
    bounds: tuple[float, float, float, float],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lon_min, lon_max, lat_min, lat_max = bounds
    source, src_lon, src_lat = ascending_axes(source, src_lon, src_lat)
    original_crop_lon = crop_lon
    original_crop_lat = crop_lat
    crop, crop_lon, crop_lat = ascending_axes(crop, original_crop_lon, original_crop_lat)
    reconstruction, _, _ = ascending_axes(reconstruction, original_crop_lon, original_crop_lat)
    valid = np.concatenate([source[np.isfinite(source)], bilinear[np.isfinite(bilinear)]])
    vmax = max(1.0, float(np.nanquantile(valid, 0.995)))
    compare_error = bilinear - reference
    nearest_error = bilinear - nearest
    reconstruct_error = reconstruction - crop
    diff_vmax = max(1.0, float(np.nanquantile(np.abs(nearest_error), 0.995)))
    reconstruct_vmax = max(1.0, float(np.nanquantile(np.abs(reconstruct_error), 0.995)))
    reference_vmax = max(1.0e-5, float(np.nanmax(np.abs(compare_error))))
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)

    image = axes[0, 0].imshow(source, origin="lower", extent=[src_lon.min(), src_lon.max(), src_lat.min(), src_lat.max()], cmap="turbo", vmin=0.0, vmax=vmax, aspect="auto")
    axes[0, 0].add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, fill=False, edgecolor="white", linewidth=2.0))
    axes[0, 0].set_title("Stage 1 center H8 SWR | PV crop box | north-up")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.02)

    panels = [
        (axes[0, 1], crop, "Native H8 pixels inside PV box | north-up", "turbo", 0.0, vmax, [crop_lon.min(), crop_lon.max(), crop_lat.min(), crop_lat.max()]),
        (axes[0, 2], bilinear, "Production bilinear mesh v3 | 64x64 | north-up", "turbo", 0.0, vmax, [lon_min, lon_max, lat_min, lat_max]),
        (axes[0, 3], nearest, "Nearest-neighbor comparison | 64x64", "turbo", 0.0, vmax, [lon_min, lon_max, lat_min, lat_max]),
        (axes[1, 0], compare_error, "Production - independent reference", "coolwarm", -reference_vmax, reference_vmax, [lon_min, lon_max, lat_min, lat_max]),
        (axes[1, 1], nearest_error, "Bilinear - nearest neighbor", "coolwarm", -diff_vmax, diff_vmax, [lon_min, lon_max, lat_min, lat_max]),
        (axes[1, 2], reconstruct_error, "Back-project bilinear - native crop", "coolwarm", -reconstruct_vmax, reconstruct_vmax, [crop_lon.min(), crop_lon.max(), crop_lat.min(), crop_lat.max()]),
        (axes[1, 3], reconstruction, "Back-projected bilinear on native crop", "turbo", 0.0, vmax, [crop_lon.min(), crop_lon.max(), crop_lat.min(), crop_lat.max()]),
    ]
    for ax, values, title, cmap, vmin, vmax_panel, extent in panels:
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax_panel, aspect="auto")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.reshape(-1):
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.suptitle(f"H8 -> PV-grid interpolation validation | BJT label={timestamp}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = args.adapter_dir / "timestamps.csv"
    swr_path = args.adapter_dir / "swr_3d.npy"
    lat_path = args.adapter_dir / "lat_values.npy"
    lon_path = args.adapter_dir / "lon_values.npy"
    for path in (timestamps_path, swr_path, lat_path, lon_path):
        if not path.exists():
            raise FileNotFoundError(f"Stage 3 center-SWR adapter file not found: {path}")

    timestamps = pd.read_csv(timestamps_path)
    timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
    selected = choose_rows(timestamps, args)
    swr = np.load(swr_path, mmap_mode="r")
    src_lat = np.load(lat_path).astype(np.float64)
    src_lon = np.load(lon_path).astype(np.float64)
    src_lat_order = "ascending" if src_lat[0] <= src_lat[-1] else "descending"
    src_lon_order = "ascending" if src_lon[0] <= src_lon[-1] else "descending"
    bounds = pv_bounds(load_station_split())
    dst_lon, dst_lat = target_mesh(bounds, int(args.grid_size))
    dst_lon_axis = dst_lon[0]
    dst_lat_axis = dst_lat[:, 0]
    crop_lat_mask = (src_lat >= bounds[2]) & (src_lat <= bounds[3])
    crop_lon_mask = (src_lon >= bounds[0]) & (src_lon <= bounds[1])
    crop_lat = src_lat[crop_lat_mask]
    crop_lon = src_lon[crop_lon_mask]
    if crop_lat.size < 2 or crop_lon.size < 2:
        raise RuntimeError(f"PV bounds do not overlap enough native H8 pixels: bounds={bounds}")

    rows = []
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in selected.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        source = np.asarray(swr[int(row["index"])], dtype=np.float32)
        bilinear = _resample_to_grid(source, src_lon, src_lat, dst_lon, dst_lat)
        reference = reference_bilinear(source, src_lon, src_lat, dst_lon_axis, dst_lat_axis)
        nearest = nearest_neighbor(source, src_lon, src_lat, dst_lon_axis, dst_lat_axis)
        crop = source[np.ix_(crop_lat_mask, crop_lon_mask)]
        reconstruction = reference_bilinear(bilinear, dst_lon_axis, dst_lat_axis, crop_lon, crop_lat)
        stacked = _resample_to_grid(
            np.broadcast_to(source, (4, 6, *source.shape)),
            src_lon,
            src_lat,
            dst_lon,
            dst_lat,
        )
        if stacked.shape != (4, 6, int(args.grid_size), int(args.grid_size)):
            raise AssertionError(f"Production stacked resample has unexpected shape: {stacked.shape}")
        production_reference = metrics(bilinear, reference)
        bilinear_nearest = metrics(bilinear, nearest)
        reconstruction_metrics = metrics(reconstruction, crop)
        figure_path = figure_dir / f"resample_{rank + 1:02d}_{timestamp:%Y%m%d_%H%M}.png"
        render_case(
            figure_path,
            timestamp,
            source,
            crop,
            bilinear,
            nearest,
            reference,
            reconstruction,
            src_lon,
            src_lat,
            crop_lon,
            crop_lat,
            bounds,
            int(args.dpi),
        )
        rows.append(
            {
                "timestamp": str(timestamp),
                "source_shape": str(tuple(source.shape)),
                "source_latitude_order": src_lat_order,
                "source_longitude_order": src_lon_order,
                "native_crop_shape": str(tuple(crop.shape)),
                "production_shape": str(tuple(bilinear.shape)),
                "stacked_production_shape": str(tuple(stacked.shape)),
                "legacy_hypothetical_stacked_shape": str(
                    legacy_shape_probe((4, 6), src_lon, src_lat, dst_lon, dst_lat)
                ),
                "production_vs_reference_mae": production_reference["mae"],
                "production_vs_reference_rmse": production_reference["rmse"],
                "production_vs_reference_max_abs": production_reference["max_abs"],
                "bilinear_vs_nearest_mae": bilinear_nearest["mae"],
                "bilinear_vs_nearest_rmse": bilinear_nearest["rmse"],
                "backproject_vs_native_crop_mae": reconstruction_metrics["mae"],
                "backproject_vs_native_crop_rmse": reconstruction_metrics["rmse"],
                **{f"source_crop_{key}": value for key, value in gradient_stats(crop, float(np.mean(np.diff(crop_lon))), float(np.mean(np.diff(crop_lat)))).items()},
                **{f"bilinear_{key}": value for key, value in gradient_stats(bilinear, float(np.mean(np.diff(dst_lon_axis))), float(np.mean(np.diff(dst_lat_axis)))).items()},
                "figure": str(figure_path),
            }
        )

    report = pd.DataFrame(rows)
    csv_path = args.out_dir / "h8_pv_grid_resampling_metrics.csv"
    report.to_csv(csv_path, index=False)
    summary = {
        "purpose": "Validate center-H8 to PV-grid interpolation after bilinear_mesh_v3 fix.",
        "adapter_dir": str(args.adapter_dir),
        "pv_bounds": [float(value) for value in bounds],
        "grid_size": int(args.grid_size),
        "source_latitude_order": src_lat_order,
        "source_longitude_order": src_lon_order,
        "num_samples": int(len(report)),
        "expected_stacked_shape": [4, 6, int(args.grid_size), int(args.grid_size)],
        "legacy_hypothetical_stacked_shape": list(
            legacy_shape_probe((4, 6), src_lon, src_lat, dst_lon, dst_lat)
        ),
        "production_vs_reference_mae_mean": float(report["production_vs_reference_mae"].mean()),
        "production_vs_reference_rmse_mean": float(report["production_vs_reference_rmse"].mean()),
        "production_vs_reference_max_abs_max": float(report["production_vs_reference_max_abs"].max()),
        "metrics_csv": str(csv_path),
        "figure_dir": str(figure_dir),
    }
    json_path = args.out_dir / "h8_pv_grid_resampling_summary.json"
    write_json(summary, json_path)
    print("==== Stage 3 H8 -> PV-grid interpolation validation ====")
    print("PV bounds:", bounds)
    print("source coordinate order: latitude=", src_lat_order, "longitude=", src_lon_order)
    print("expected stacked output:", tuple(summary["expected_stacked_shape"]))
    print("legacy hypothetical output:", tuple(summary["legacy_hypothetical_stacked_shape"]))
    print("production vs independent reference MAE:", summary["production_vs_reference_mae_mean"])
    print("production vs independent reference RMSE:", summary["production_vs_reference_rmse_mean"])
    print("production vs independent reference max abs:", summary["production_vs_reference_max_abs_max"])
    print("metrics:", csv_path)
    print("figures:", figure_dir)


if __name__ == "__main__":
    main()
