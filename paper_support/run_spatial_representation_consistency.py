#!/usr/bin/env python3
"""Source-station consistency check for the explicit regional PV field.

This is a representation sanity check, not a forecasting or unseen-site
interpolation experiment.  It samples the already-constructed, target-excluded
PV field at the source-station grid cells and compares those values with the
same source measurements used to construct each field.

Default server usage
--------------------
python paper_support/run_spatial_representation_consistency.py

All outputs are written below
``paper_support/spatial_representation_consistency`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Publication defaults.  SVG text remains editable.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_support_paths import (  # noqa: E402
    resolve_old_data_root,
    resolve_pv_directories,
    resolve_stage3_data_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "spatial_representation_consistency",
    )
    parser.add_argument(
        "--old-data-root",
        type=Path,
        default=None,
        help=(
            "External cache root. By default it is inferred as the old_data "
            "directory beside the code directory."
        ),
    )
    parser.add_argument(
        "--pv-root",
        type=Path,
        default=None,
        help=(
            "Root containing the train/validation station-CSV directories. "
            "Default: <old-data-root>/PV (or BIGPV_PV_ROOT)."
        ),
    )
    parser.add_argument(
        "--stage3-data-dir",
        type=Path,
        default=None,
        help="Optional explicit Stage-3 data directory; otherwise local and old_data are checked.",
    )
    parser.add_argument(
        "--train-pv-dir",
        type=Path,
        default=None,
        help="Optional explicit train station-CSV directory; overrides --pv-root.",
    )
    parser.add_argument(
        "--val-pv-dir",
        type=Path,
        default=None,
        help="Optional explicit validation/test station-CSV directory; overrides --pv-root.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument(
        "--example-time",
        type=str,
        default="",
        help="Optional exact example timestamp. Default selects a deterministic representative frame.",
    )
    parser.add_argument(
        "--example-variability-quantile",
        type=float,
        default=0.75,
        help="Deterministic example target quantile of across-station standard deviation.",
    )
    parser.add_argument(
        "--max-timestamps",
        type=int,
        default=0,
        help="Debug-only deterministic cap. Zero uses every timestamp in the requested split.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with require(path, "JSON metadata").open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def station_sns_sha1(frame: pd.DataFrame) -> str:
    station_ids = sorted(str(value) for value in frame["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(station_ids).encode("utf-8")).hexdigest()


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if y.size < 2:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if y.size == 0:
        return {"n_points": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    error = pred - y
    return {
        "n_points": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2_score(y, pred),
        "bias": float(np.mean(error)),
    }


def timestamp_seconds(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return np.asarray([int(pd.Timestamp(value).timestamp()) for value in values], dtype=np.int64)


def split_mask(timestamp_index: pd.DatetimeIndex, split: str, train_end: pd.Timestamp, val_end: pd.Timestamp) -> np.ndarray:
    if split == "all":
        return np.ones(len(timestamp_index), dtype=bool)
    values = np.asarray(timestamp_index)
    if split == "train":
        return values <= np.datetime64(train_end)
    if split == "val":
        return (values > np.datetime64(train_end)) & (values <= np.datetime64(val_end))
    return values > np.datetime64(val_end)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")


def load_pipeline_helpers(stage3_dir: Path):
    if str(stage3_dir) not in sys.path:
        sys.path.insert(0, str(stage3_dir))
    try:
        from visualize_regional_split_frames import (  # type: ignore
            CAPACITY_CANDIDATES,
            PV_CANDIDATES,
            TIME_CANDIDATES,
            _bounds,
            _find_col,
            _station_csv_path,
            _xy_to_ij,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Stage-3 field-construction helpers. Run this script "
            "inside the complete SatPV-Field environment."
        ) from exc
    return SimpleNamespace(
        CAPACITY_CANDIDATES=CAPACITY_CANDIDATES,
        PV_CANDIDATES=PV_CANDIDATES,
        TIME_CANDIDATES=TIME_CANDIDATES,
        bounds=_bounds,
        find_col=_find_col,
        station_csv_path=_station_csv_path,
        xy_to_ij=_xy_to_ij,
    )


def choose_column(columns: list[str], preferred: str, candidates: list[str], helpers) -> str | None:
    if preferred in columns:
        return preferred
    return helpers.find_col(columns, candidates)


def health_filter_pv(series: pd.Series, settings: dict[str, Any]) -> tuple[pd.Series, dict[str, int | float]]:
    """Mirror the point-level health filter used to build the Stage-3 field cache."""
    pv = pd.to_numeric(series, errors="coerce").astype(float)
    if not bool(settings.get("health_filter_frame_cache", True)):
        finite = np.isfinite(pv.to_numpy(np.float64))
        kept = pv.loc[finite].copy()
        return kept, {
            "before": int(len(pv)),
            "after": int(len(kept)),
            "n_before": int(len(pv)),
            "n_after": int(len(kept)),
            "dropped": int(np.sum(~finite)),
            "out_of_range_or_nonfinite": int(np.sum(~finite)),
            "active_flatline": 0,
            "health_filter_enabled": 0,
        }
    maximum = float(settings.get("pv_health_max", 1.35))
    active_threshold = float(settings.get("pv_health_active_threshold", 0.05))
    window = max(2, int(settings.get("pv_health_flatline_window", 8)))
    flat_range = float(settings.get("pv_health_flatline_range", 1e-5))
    base_bad = (~np.isfinite(pv.to_numpy(np.float64))) | (pv < -0.03) | (pv > maximum)
    rolling_range = pv.rolling(window, center=True, min_periods=window).max() - pv.rolling(
        window, center=True, min_periods=window
    ).min()
    flat_bad = (pv >= active_threshold) & (rolling_range <= flat_range)
    bad = np.asarray(base_bad, dtype=bool) | flat_bad.fillna(False).to_numpy(bool)
    kept = pv.loc[~bad].copy()
    report = {
        "before": int(len(pv)),
        "after": int(len(kept)),
        "n_before": int(len(pv)),
        "n_after": int(len(kept)),
        "dropped": int(np.sum(bad)),
        "out_of_range_or_nonfinite": int(np.sum(base_bad)),
        "active_flatline": int(np.sum(flat_bad.fillna(False).to_numpy(bool))),
        "health_filter_enabled": 1,
    }
    return kept, report


def load_station_values(
    station_sn: str,
    requested_secs: np.ndarray,
    q95: float,
    helpers,
    args: argparse.Namespace,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = helpers.station_csv_path(station_sn, args.train_pv_dir, args.val_pv_dir)
    empty = np.full(len(requested_secs), np.nan, dtype=np.float32)
    if path is None:
        return empty, {"station_sn": station_sn, "status": "missing_csv"}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return empty, {"station_sn": station_sn, "status": f"read_failed:{type(exc).__name__}", "path": str(path)}

    columns = [str(column) for column in df.columns]
    time_col = choose_column(columns, "date", helpers.TIME_CANDIDATES, helpers)
    pv_col = choose_column(columns, "elec", helpers.PV_CANDIDATES, helpers)
    cap_col = choose_column(columns, "Capacity", helpers.CAPACITY_CANDIDATES, helpers)
    if time_col is None or pv_col is None or cap_col is None:
        return empty, {
            "station_sn": station_sn,
            "status": "missing_date_elec_or_capacity_col",
            "path": str(path),
            "columns": "|".join(columns),
        }

    ts = pd.to_datetime(df[time_col], errors="coerce")
    pv = pd.to_numeric(df[pv_col], errors="coerce").astype(float)
    capacity = pd.to_numeric(df[cap_col], errors="coerce").astype(float)
    cap_value = float(capacity.dropna().median()) if capacity.dropna().size else np.nan
    if not np.isfinite(cap_value) or cap_value <= 0:
        return empty, {"station_sn": station_sn, "status": "invalid_capacity", "path": str(path)}

    pv_norm = pv / max(cap_value * float(settings.get("capacity_scale", 1000.0)), 1e-6)
    work = pd.DataFrame({"timestamp": ts, "pv_norm": pv_norm}).dropna(subset=["timestamp", "pv_norm"])
    work = work[(work["pv_norm"] >= 0.0) & (work["pv_norm"] <= 1.3)]
    work = work.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    filtered, health = health_filter_pv(work["pv_norm"], settings)

    seconds = timestamp_seconds(filtered.index)
    values = np.clip(
        filtered.to_numpy(np.float64) / max(float(q95), 1e-4),
        0.0,
        float(settings.get("pv_field_clip", 1.5)),
    )
    lookup = pd.Series(values.astype(np.float32), index=seconds)
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    aligned = lookup.reindex(requested_secs).to_numpy(np.float32)
    return aligned, {
        "station_sn": station_sn,
        "status": "ok",
        "path": str(path),
        "valid_requested_points": int(np.isfinite(aligned).sum()),
        "capacity_raw": cap_value,
        "q95": float(q95),
        **health,
    }


def load_settings(selection_meta: dict[str, Any], frame_meta: dict[str, Any]) -> dict[str, Any]:
    source = selection_meta.get("args", {})
    if not isinstance(source, dict):
        source = {}
    settings = {
        "capacity_scale": float(source.get("capacity_scale", 1000.0)),
        "pv_field_clip": float(frame_meta.get("pv_field_clip", source.get("pv_field_clip", 1.5))),
        "grid_size": int(frame_meta.get("grid_size", source.get("grid_size", 64))),
        "gaussian_sigma_cells": float(frame_meta.get("gaussian_sigma_cells", source.get("gaussian_sigma_cells", 1.0))),
        "gaussian_radius_cells": int(frame_meta.get("gaussian_radius_cells", source.get("gaussian_radius_cells", 2))),
        "min_grid_weight": float(frame_meta.get("min_grid_weight", source.get("min_grid_weight", 1e-6))),
        "health_filter_frame_cache": str(source.get("health_filter_frame_cache", "True")).lower() not in {"0", "false", "no"},
        "pv_health_max": float(source.get("pv_health_max", 1.35)),
        "pv_health_active_threshold": float(source.get("pv_health_active_threshold", 0.05)),
        "pv_health_flatline_window": int(source.get("pv_health_flatline_window", 8)),
        "pv_health_flatline_range": float(source.get("pv_health_flatline_range", 1e-5)),
    }
    return settings


def aggregate_by_station(
    station_ids: np.ndarray, actual: np.ndarray, sampled: np.ndarray, valid: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, station_sn in enumerate(station_ids):
        mask = valid[:, index]
        row = {"station_sn": str(station_sn)}
        row.update(metrics(actual[mask, index], sampled[mask, index]))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_timestamp(
    timestamps: pd.DatetimeIndex, actual: np.ndarray, sampled: np.ndarray, valid: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        mask = valid[index]
        values = actual[index, mask]
        row: dict[str, Any] = {
            "timestamp": str(pd.Timestamp(timestamp)),
            "timestamp_sec": int(pd.Timestamp(timestamp).timestamp()),
            "n_source_measurements": int(mask.sum()),
            "actual_mean": float(np.mean(values)) if values.size else np.nan,
            "actual_std": float(np.std(values)) if values.size else np.nan,
            "actual_p10": float(np.quantile(values, 0.10)) if values.size else np.nan,
            "actual_p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
        }
        row.update(metrics(actual[index, mask], sampled[index, mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def collision_diagnostics(cell_y: np.ndarray, cell_x: np.ndarray, grid_size: int) -> dict[str, Any]:
    flat = cell_y.astype(np.int64) * int(grid_size) + cell_x.astype(np.int64)
    _cells, counts = np.unique(flat, return_counts=True)
    occupied = int(len(counts))
    colliding_cells = int(np.sum(counts > 1))
    stations_in_colliding_cells = int(np.sum(counts[counts > 1]))
    return {
        "n_source_stations": int(len(flat)),
        "n_occupied_station_cells": occupied,
        "n_colliding_cells": colliding_cells,
        "n_stations_in_colliding_cells": stations_in_colliding_cells,
        "fraction_stations_in_colliding_cells": float(stations_in_colliding_cells / max(1, len(flat))),
        "max_stations_per_cell": int(np.max(counts)) if counts.size else 0,
    }


def choose_example(
    per_time: pd.DataFrame,
    requested: str,
    quantile: float,
    minimum_stations: int,
) -> int:
    if requested:
        target = pd.Timestamp(requested)
        times = pd.to_datetime(per_time["timestamp"])
        exact = np.flatnonzero(times == target)
        if exact.size == 0:
            raise ValueError(f"--example-time {target} is not present in the evaluated timestamps")
        return int(exact[0])
    eligible = per_time[
        (per_time["n_source_measurements"] >= int(minimum_stations))
        & np.isfinite(per_time["actual_std"])
        & (per_time["actual_mean"] >= 0.05)
    ].copy()
    if eligible.empty:
        eligible = per_time[np.isfinite(per_time["actual_std"])].copy()
    if eligible.empty:
        raise RuntimeError("No timestamp has enough valid source measurements for the example figure")
    q = float(np.clip(quantile, 0.0, 1.0))
    target_std = float(eligible["actual_std"].quantile(q))
    eligible["distance_to_target_std"] = np.abs(eligible["actual_std"] - target_std)
    chosen = eligible.sort_values(["distance_to_target_std", "timestamp_sec"]).iloc[0]
    return int(per_time.index[per_time["timestamp_sec"] == int(chosen["timestamp_sec"])][0])


def save_figure(
    out_dir: Path,
    bounds: tuple[float, float, float, float],
    source_df: pd.DataFrame,
    field: np.ndarray,
    field_mask: np.ndarray,
    example_actual: np.ndarray,
    example_sampled: np.ndarray,
    actual_all: np.ndarray,
    sampled_all: np.ndarray,
    pooled: dict[str, Any],
    timestamp: pd.Timestamp,
    field_clip: float,
    dpi: int,
) -> list[str]:
    lon_min, lon_max, lat_min, lat_max = bounds
    fig = plt.figure(figsize=(7.20, 2.65), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], wspace=0.18)
    ax_station = fig.add_subplot(grid[0, 0])
    ax_field = fig.add_subplot(grid[0, 1])
    ax_scatter = fig.add_subplot(grid[0, 2])

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("white")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(field_clip))

    valid_example = np.isfinite(example_actual)
    station_plot = ax_station.scatter(
        source_df.loc[valid_example, "lon"],
        source_df.loc[valid_example, "lat"],
        c=example_actual[valid_example],
        cmap=cmap,
        norm=norm,
        s=9,
        linewidths=0.15,
        edgecolors="#303030",
        rasterized=True,
    )
    ax_station.set_title("Source measurements", pad=5)
    ax_station.set_xlabel("Longitude (°E)")
    ax_station.set_ylabel("Latitude (°N)")
    ax_station.set_xlim(lon_min, lon_max)
    ax_station.set_ylim(lat_min, lat_max)
    ax_station.set_aspect("equal", adjustable="box")
    add_panel_label(ax_station, "a")

    display_field = np.ma.masked_where(~field_mask.astype(bool), field)
    ax_field.imshow(
        display_field,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_field.scatter(source_df["lon"], source_df["lat"], s=1.3, c="#333333", alpha=0.35, linewidths=0)
    ax_field.set_title("Constructed PV field", pad=5)
    ax_field.set_xlabel("Longitude (°E)")
    ax_field.set_yticklabels([])
    ax_field.set_xlim(lon_min, lon_max)
    ax_field.set_ylim(lat_min, lat_max)
    colorbar = fig.colorbar(station_plot, ax=[ax_station, ax_field], orientation="horizontal", fraction=0.08, pad=0.14)
    colorbar.set_label("Normalized PV generation (p.u.)")

    valid = np.isfinite(actual_all) & np.isfinite(sampled_all)
    x = actual_all[valid].astype(np.float64)
    y = sampled_all[valid].astype(np.float64)
    hb = ax_scatter.hexbin(
        x,
        y,
        gridsize=62,
        extent=(0.0, field_clip, 0.0, field_clip),
        mincnt=1,
        cmap="Blues",
        norm=LogNorm(),
        linewidths=0,
        rasterized=True,
    )
    ax_scatter.plot([0.0, field_clip], [0.0, field_clip], color="#B64342", linestyle="--", linewidth=1.1, label="Identity")
    ax_scatter.set_xlim(0.0, field_clip)
    ax_scatter.set_ylim(0.0, field_clip)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Measured source PV (p.u.)")
    ax_scatter.set_ylabel("Field sampled at source location (p.u.)")
    ax_scatter.set_title("All test measurements", pad=5)
    add_panel_label(ax_scatter, "b")
    ax_scatter.text(
        0.04,
        0.96,
        f"n = {int(pooled['n_points']):,}\nMAE = {float(pooled['mae']):.4f}\nRMSE = {float(pooled['rmse']):.4f}\n$R^2$ = {float(pooled['r2']):.4f}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "#C7C7C7", "boxstyle": "round,pad=0.3", "alpha": 0.92},
    )
    ax_scatter.legend(loc="lower right", fontsize=6.5)
    density_bar = fig.colorbar(hb, ax=ax_scatter, orientation="vertical", fraction=0.046, pad=0.03)
    density_bar.set_label("Point count")

    fig.suptitle(f"Spatial representation consistency | {timestamp:%Y-%m-%d %H:%M} BJT", fontsize=8, y=1.035)
    svg_path = out_dir / "spatial_representation_consistency.svg"
    pdf_path = out_dir / "spatial_representation_consistency.pdf"
    tiff_path = out_dir / "spatial_representation_consistency.tiff"
    png_path = out_dir / "spatial_representation_consistency.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    saved = [str(svg_path), str(pdf_path), str(tiff_path), str(png_path)]
    plt.close(fig)
    return saved


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    pooled: dict[str, Any],
    n_stations: int,
    n_timestamps: int,
    collision: dict[str, Any],
    example_time: pd.Timestamp,
    example_metrics: dict[str, Any],
    settings: dict[str, Any],
    selection_note: str,
    outputs: list[str],
) -> None:
    lines = [
        "# Spatial representation consistency",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Purpose and interpretation boundary",
        "",
        "This analysis tests whether the explicit regional PV field retains the source fleet's measured generation information at the original source locations. It is a representation sanity check. It does not evaluate unseen-site interpolation, target-station reconstruction, or forecasting.",
        "",
        "Each comparison is self-inclusive: the source measurement being checked also contributed to the constructed field. The reported error therefore quantifies information distortion introduced by grid assignment, Gaussian smoothing, overlap averaging, support masking, and float16 field storage.",
        "",
        "## Data and protocol",
        "",
        f"- Split: `{args.split}`",
        f"- Source stations with metadata: {n_stations}",
        f"- Evaluated field timestamps: {n_timestamps}",
        f"- Timestamp selection: {selection_note}",
        f"- Valid station-time comparisons: {int(pooled['n_points'])}",
        "- Target stations: excluded, consistent with the source-supported field definition",
        "- Sampling rule: value of the cached field at the exact grid cell assigned by the production coordinate-to-grid function",
        f"- Field grid: {settings['grid_size']} x {settings['grid_size']}",
        f"- Gaussian splatting: sigma={settings['gaussian_sigma_cells']} cell, radius={settings['gaussian_radius_cells']} cells",
        f"- Field normalization: capacity-normalized PV divided by each source station's train-period q95 and clipped to [0, {settings['pv_field_clip']}]",
        "- Inclusion rule: every finite source measurement retained by the production-equivalent point health filter at every available timestamp in the requested split",
        "",
        "## Pooled consistency metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {int(pooled['n_points'])} |",
        f"| MAE | {fmt(pooled['mae'])} |",
        f"| RMSE | {fmt(pooled['rmse'])} |",
        f"| R² | {fmt(pooled['r2'])} |",
        f"| Bias (sampled - measured) | {fmt(pooled['bias'])} |",
        "",
        "## Grid-collision diagnostic",
        "",
        "| Diagnostic | Value |",
        "|---|---:|",
        f"| Occupied source-station cells | {collision['n_occupied_station_cells']} |",
        f"| Cells containing >1 source station | {collision['n_colliding_cells']} |",
        f"| Source stations in colliding cells | {collision['n_stations_in_colliding_cells']} |",
        f"| Fraction of stations in colliding cells | {fmt(collision['fraction_stations_in_colliding_cells'])} |",
        f"| Maximum stations in one cell | {collision['max_stations_per_cell']} |",
        "",
        "## Representative visualization",
        "",
        f"Example timestamp: `{example_time}`.",
        "",
        f"When `--example-time` is not supplied, the example is selected deterministically as the eligible timestamp closest to the {float(args.example_variability_quantile):.2f} quantile of across-station PV standard deviation. It is not selected by reconstruction error.",
        "",
        f"Example-frame MAE={fmt(example_metrics['mae'])}, RMSE={fmt(example_metrics['rmse'])}, R²={fmt(example_metrics['r2'])}, n={int(example_metrics['n_points'])}.",
        "",
        "### Suggested figure legend",
        "",
        "**Fig. X | Station-level consistency of the explicit regional PV generation field.** **a**, Source-station PV measurements and the corresponding target-excluded regional PV generation field at a deterministically selected test timestamp. Both maps use the same value scale; points overlaid on the field indicate source-station locations. **b**, Density representation of measured source-station PV output versus values sampled from the generated field at the corresponding station grid cells across all valid test timestamps. The dashed line denotes identity. Metrics are pooled over all valid source-station–timestamp pairs. Target stations are excluded from both field construction and this consistency analysis.",
        "",
        "## Manuscript-safe interpretation",
        "",
        "Suggested wording: *To verify that the spatialized representation preserves the generation information of the source fleet, we sampled each constructed field at the original source-station locations and compared the sampled values with the corresponding measurements.*",
        "",
        "Use a result sentence of the form: *The close station-level agreement (MAE = XX, RMSE = XX, R² = XX; n = XX station–timestamp pairs) indicates that the field retains the spatial organization of realized fleet generation while introducing only the smoothing associated with its grid-based construction.*",
        "",
        "Do not call this field reconstruction accuracy, unseen-site estimation, or evidence that Gaussian splatting is superior to another interpolation method.",
        "",
        "## Output files",
        "",
    ]
    lines.extend(f"- `{item}`" for item in outputs)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    old_data_root = resolve_old_data_root(root, args.old_data_root)
    args.train_pv_dir, args.val_pv_dir = resolve_pv_directories(
        args.pv_root,
        args.train_pv_dir,
        args.val_pv_dir,
        project_root=root,
        old_data_root=old_data_root,
    )
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage3 = require(root / "stage_3", "Stage-3 directory")
    helpers = load_pipeline_helpers(stage3)

    stage3_data_dir = resolve_stage3_data_dir(
        root,
        old_data_root,
        args.stage3_data_dir,
    )
    selection_dir = stage3_data_dir / "02_target_selection_lhs50"
    run_dir = stage3_data_dir / "03_target_excluded_run"
    cache_dir = stage3_data_dir / "04_target_excluded_pvnorm_frame_cache"
    all_path = require(run_dir / "station_split.csv", "full station split")
    source_path = require(run_dir / "target_excluded_station_split.csv", "target-excluded source split")
    target_path = require(selection_dir / "target_stations.csv", "held-out target list")
    q95_path = require(selection_dir / "target_excluded_station_pv_norm_q95.csv", "source-station q95 values")
    selection_meta_path = require(selection_dir / "01_target_selection_meta.json", "target-selection metadata")
    field_path = require(cache_dir / "fields_float16.npy", "cached PV fields")
    mask_path = require(cache_dir / "masks_uint8.npy", "cached support masks")
    sec_path = require(cache_dir / "timestamp_sec.npy", "cached field timestamps")
    frame_meta_path = require(cache_dir / "frame_cache_meta.json", "field-cache metadata")

    all_df = pd.read_csv(all_path, dtype={"station_sn": str})
    source_df = pd.read_csv(source_path, dtype={"station_sn": str}).reset_index(drop=True)
    target_df = pd.read_csv(target_path, dtype={"station_sn": str})
    q95_df = pd.read_csv(q95_path, dtype={"station_sn": str})
    selection_meta = read_json(selection_meta_path)
    frame_meta = read_json(frame_meta_path)
    settings = load_settings(selection_meta, frame_meta)

    required_columns = {"station_sn", "lon", "lat"}
    for label, frame in [("all stations", all_df), ("source stations", source_df), ("targets", target_df)]:
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{label} file is missing columns: {missing}")
    overlap = set(source_df["station_sn"]) & set(target_df["station_sn"])
    if overlap:
        raise AssertionError(f"Source-target overlap detected: {sorted(overlap)[:10]}")
    if int(frame_meta.get("source_station_count", len(source_df))) != len(source_df):
        raise AssertionError(
            f"Field-cache source count {frame_meta.get('source_station_count')} != source CSV count {len(source_df)}"
        )
    expected_source_hash = str(frame_meta.get("source_station_sns_sha1", "")).strip()
    actual_source_hash = station_sns_sha1(source_df)
    if expected_source_hash and expected_source_hash != actual_source_hash:
        raise AssertionError(
            "Field-cache source station hash does not match target_excluded_station_split.csv: "
            f"cache={expected_source_hash}, csv={actual_source_hash}"
        )

    q95_lookup = q95_df.set_index("station_sn")["pv_norm_q95"].astype(float).to_dict()
    missing_q95 = [sn for sn in source_df["station_sn"] if sn not in q95_lookup]
    if missing_q95:
        raise ValueError(f"Missing q95 values for {len(missing_q95)} source stations")

    fields = np.load(field_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    all_secs = np.asarray(np.load(sec_path, mmap_mode="r"), dtype=np.int64)
    if fields.shape != masks.shape or fields.shape[0] != len(all_secs):
        raise AssertionError(f"Cache shape mismatch: fields={fields.shape}, masks={masks.shape}, times={len(all_secs)}")
    if fields.shape[1:] != (int(settings["grid_size"]), int(settings["grid_size"])):
        raise AssertionError(f"Unexpected field grid: {fields.shape[1:]}, settings={settings['grid_size']}")

    timestamp_index = pd.to_datetime(all_secs, unit="s")
    source_args = selection_meta.get("args", {}) if isinstance(selection_meta.get("args", {}), dict) else {}
    train_end = pd.Timestamp(source_args.get("train_end", "2023-01-13 23:59:59"))
    val_end = pd.Timestamp(source_args.get("val_end", "2023-03-08 23:59:59"))
    selected = np.flatnonzero(split_mask(timestamp_index, args.split, train_end, val_end))
    if selected.size == 0:
        raise RuntimeError(f"No cached timestamps found for split={args.split}")
    if int(args.max_timestamps) > 0 and selected.size > int(args.max_timestamps):
        # Debug-only, deterministic coverage of the full split; never used by default.
        cap_indices = np.linspace(0, selected.size - 1, num=int(args.max_timestamps), dtype=int)
        selected = selected[np.unique(cap_indices)]
        selection_note = f"debug cap used: {args.max_timestamps}/{int(split_mask(timestamp_index, args.split, train_end, val_end).sum())} timestamps"
    else:
        selection_note = "all timestamps in requested split"
    secs = all_secs[selected]
    timestamps = timestamp_index[selected]

    n_times, n_stations = len(selected), len(source_df)
    actual = np.full((n_times, n_stations), np.nan, dtype=np.float32)
    load_rows: list[dict[str, Any]] = []
    for station_index, row in source_df.iterrows():
        station_sn = str(row["station_sn"])
        values, report = load_station_values(
            station_sn,
            secs,
            float(q95_lookup[station_sn]),
            helpers,
            args,
            settings,
        )
        actual[:, station_index] = values
        load_rows.append(report)
        if (station_index + 1) % 50 == 0 or station_index + 1 == n_stations:
            print(f"[station data] {station_index + 1}/{n_stations}", flush=True)
    pd.DataFrame(load_rows).to_csv(out_dir / "station_load_and_health_filter_report.csv", index=False)

    if not np.isfinite(actual).any():
        raise RuntimeError("No valid source-station measurements aligned with the selected field timestamps")

    bounds = helpers.bounds(all_df)
    cells = [helpers.xy_to_ij(float(row.lon), float(row.lat), bounds, int(settings["grid_size"])) for row in source_df.itertuples()]
    cell_y = np.asarray([item[0] for item in cells], dtype=np.int32)
    cell_x = np.asarray([item[1] for item in cells], dtype=np.int32)
    collision = collision_diagnostics(cell_y, cell_x, int(settings["grid_size"]))

    sampled = np.full_like(actual, np.nan)
    supported = np.zeros_like(actual, dtype=bool)
    chunk_size = max(1, int(args.chunk_size))
    for start in range(0, n_times, chunk_size):
        stop = min(n_times, start + chunk_size)
        cache_indices = selected[start:stop]
        field_chunk = np.asarray(fields[cache_indices], dtype=np.float32)
        mask_chunk = np.asarray(masks[cache_indices], dtype=bool)
        sampled[start:stop] = field_chunk[:, cell_y, cell_x]
        supported[start:stop] = mask_chunk[:, cell_y, cell_x]
        print(f"[field sampling] {stop}/{n_times}", flush=True)

    valid = np.isfinite(actual) & np.isfinite(sampled) & supported
    if not valid.any():
        raise RuntimeError("No valid source measurement / sampled-field pairs were produced")
    pooled = metrics(actual[valid], sampled[valid])
    pooled.update(
        {
            "n_stations": int(np.sum(valid.any(axis=0))),
            "n_timestamps": int(np.sum(valid.any(axis=1))),
            "split": args.split,
            "timestamp_selection": selection_note,
        }
    )

    per_station = aggregate_by_station(source_df["station_sn"].astype(str).to_numpy(), actual, sampled, valid)
    per_time = aggregate_by_timestamp(timestamps, actual, sampled, valid)
    per_station.to_csv(out_dir / "source_data_metrics_by_station.csv", index=False)
    per_time.to_csv(out_dir / "source_data_metrics_by_timestamp.csv", index=False)

    valid_time_index, valid_station_index = np.nonzero(valid)
    np.savez_compressed(
        out_dir / "source_data_all_valid_pairs.npz",
        actual=actual[valid].astype(np.float32),
        sampled=sampled[valid].astype(np.float32),
        timestamp_sec=secs[valid_time_index].astype(np.int64),
        station_index=valid_station_index.astype(np.int32),
        station_sn=source_df["station_sn"].astype(str).to_numpy(),
    )

    example_index = choose_example(
        per_time,
        args.example_time,
        float(args.example_variability_quantile),
        max(20, int(math.ceil(n_stations * 0.5))),
    )
    example_cache_index = int(selected[example_index])
    example_time = pd.Timestamp(timestamps[example_index])
    example_valid = valid[example_index]
    example_metrics = metrics(actual[example_index, example_valid], sampled[example_index, example_valid])
    example_source = source_df[["station_sn", "lon", "lat"]].copy()
    example_source["cell_y"] = cell_y
    example_source["cell_x"] = cell_x
    example_source["measured_pv_norm"] = actual[example_index]
    example_source["sampled_field_pv_norm"] = sampled[example_index]
    example_source["supported"] = supported[example_index]
    example_source["valid_comparison"] = example_valid
    example_source.to_csv(out_dir / "source_data_example_frame.csv", index=False)

    field_example = np.asarray(fields[example_cache_index], dtype=np.float32)
    mask_example = np.asarray(masks[example_cache_index], dtype=bool)
    figure_paths = save_figure(
        out_dir,
        bounds,
        source_df,
        field_example,
        mask_example,
        actual[example_index],
        sampled[example_index],
        actual[valid],
        sampled[valid],
        pooled,
        example_time,
        float(settings["pv_field_clip"]),
        int(args.dpi),
    )

    run_meta = {
        "purpose": "source-station spatial representation consistency; not reconstruction forecasting or unseen-site interpolation",
        "project_root": str(root),
        "old_data_root": str(old_data_root),
        "stage3_data_dir": str(stage3_data_dir),
        "pv_root": str(Path(args.pv_root).resolve()) if args.pv_root is not None else str(old_data_root / "PV"),
        "train_pv_dir": str(args.train_pv_dir),
        "val_pv_dir": str(args.val_pv_dir),
        "split": args.split,
        "timestamp_selection": selection_note,
        "source_station_count": int(n_stations),
        "target_station_count": int(len(target_df)),
        "source_target_overlap": 0,
        "pooled_metrics": pooled,
        "example_timestamp": str(example_time),
        "example_selection_rule": (
            "explicit --example-time" if args.example_time else f"closest to q={float(args.example_variability_quantile):.3f} of across-station actual PV std; not selected by reconstruction error"
        ),
        "example_metrics": example_metrics,
        "field_settings": settings,
        "collision_diagnostics": collision,
        "sampling_rule": "cached field value at production _xy_to_ij station cell",
        "self_inclusion_warning": "each evaluated source measurement also contributed to its field",
        "artifacts": {
            "field_cache": str(field_path),
            "support_mask": str(mask_path),
            "timestamp_index": str(sec_path),
            "source_station_file": str(source_path),
            "target_station_file": str(target_path),
            "q95_file": str(q95_path),
        },
    }
    write_json(out_dir / "spatial_representation_consistency_metrics.json", run_meta)
    pd.DataFrame([pooled]).to_csv(out_dir / "spatial_representation_consistency_metrics.csv", index=False)

    report_outputs = [
        *figure_paths,
        str(out_dir / "spatial_representation_consistency_metrics.csv"),
        str(out_dir / "spatial_representation_consistency_metrics.json"),
        str(out_dir / "source_data_metrics_by_station.csv"),
        str(out_dir / "source_data_metrics_by_timestamp.csv"),
        str(out_dir / "source_data_example_frame.csv"),
        str(out_dir / "source_data_all_valid_pairs.npz"),
        str(out_dir / "station_load_and_health_filter_report.csv"),
    ]
    report_path = out_dir / "spatial_representation_consistency.md"
    write_report(
        report_path,
        args,
        pooled,
        n_stations,
        n_times,
        collision,
        example_time,
        example_metrics,
        settings,
        selection_note,
        report_outputs,
    )
    print("==== Spatial representation consistency ====")
    print(json.dumps(pooled, ensure_ascii=False, indent=2))
    print(f"example_timestamp={example_time}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
