#!/usr/bin/env python3
"""Create the III-B.1 visual validation figure for the regional PV field.

The script selects three deterministic test-set cases from source-station PV
measurements only:

1. a high-output, spatially uniform, temporally stable case;
2. a spatially heterogeneous case with intermediate regional output;
3. a rapidly varying case with non-trivial spatial heterogeneity.

For every case, one independently exported panel shows the source measurements
and the next shows the corresponding cached, target-excluded regional PV
field.  The six numbered panels use the same color scale and each embeds its
own horizontal color bar.  A separate combined figure uses one shared color
bar and panel labels (a)--(f), without embedded titles. Target stations are
never plotted or used in case selection.

Default server usage
--------------------
python paper_support/visualize_spatial_representation_cases.py

The default map geometry uses the same detailed contextily tile workflow as
the Ramp_FinalFinal geographical figures.  Use an offline Natural Earth
background with:

python paper_support/visualize_spatial_representation_cases.py \
  --background natural-earth

The script reuses the production-equivalent loading and health-filter helpers
from ``run_spatial_representation_consistency.py`` in the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_spatial_representation_consistency as consistency  # noqa: E402
from paper_support_paths import (  # noqa: E402
    resolve_old_data_root,
    resolve_pv_directories,
    resolve_stage3_data_dir,
)
from paper_support_basemap import (  # noqa: E402
    DEFAULT_BOUNDARY,
    DEFAULT_NATURAL_EARTH_DIR,
    draw_basemap,
    draw_boundary_on_top,
    load_boundary,
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)


CASE_ORDER = ("clear", "cloudy", "rapid")
DEFAULT_EXPORT_DPI = 600
CASE_DISPLAY = {
    "clear": "Stable high-generation",
    "cloudy": "Spatially heterogeneous generation",
    "rapid": "Rapidly varying generation",
}
CASE_SLUG = {
    "clear": "stable_high_generation",
    "cloudy": "spatially_heterogeneous_generation",
    "rapid": "rapidly_varying_generation",
}

# Same blue-to-red field palette used by the Stage-3 framework assets.  It
# avoids a red/green-only encoding and visually links this validation figure
# to the methodology diagram.
FRAMEWORK_PV_COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]


def framework_pv_colormap() -> mpl.colors.Colormap:
    colors = [
        (position, tuple(channel / 255.0 for channel in rgb))
        for position, rgb in FRAMEWORK_PV_COLOR_STOPS
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "framework_pv",
        colors,
        N=256,
    )
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "spatial_representation_visualization",
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
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--clear-time", type=str, default="", help="Optional exact timestamp override.")
    parser.add_argument("--cloudy-time", type=str, default="", help="Optional exact timestamp override.")
    parser.add_argument("--rapid-time", type=str, default="", help="Optional exact timestamp override.")
    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.70,
        help="Minimum fraction of source stations required for case selection.",
    )
    parser.add_argument(
        "--minimum-case-gap-hours",
        type=float,
        default=24.0,
        help="Prefer selected cases separated by at least this many hours.",
    )
    parser.add_argument("--clear-mean-quantile", type=float, default=0.70)
    parser.add_argument("--clear-std-quantile", type=float, default=0.40)
    parser.add_argument("--clear-delta-quantile", type=float, default=0.40)
    parser.add_argument("--cloudy-mean-low-quantile", type=float, default=0.20)
    parser.add_argument("--cloudy-mean-high-quantile", type=float, default=0.80)
    parser.add_argument("--cloudy-std-quantile", type=float, default=0.75)
    parser.add_argument("--rapid-delta-quantile", type=float, default=0.80)
    parser.add_argument("--rapid-std-quantile", type=float, default=0.50)
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Shared upper color limit. Zero uses the field-cache clip value (normally 1.5).",
    )
    parser.add_argument(
        "--boundary",
        "--coastline-shp",
        dest="boundary",
        type=Path,
        default=DEFAULT_BOUNDARY,
        help=(
            "Ningbo land boundary in GeoJSON/SHP/GPKG format. The legacy "
            "--coastline-shp name is retained as an alias."
        ),
    )
    parser.add_argument(
        "--boundary-crs",
        default="EPSG:4326",
        help="CRS assigned only when the boundary file does not declare one.",
    )
    parser.add_argument(
        "--background",
        choices=("natural-earth", "contextily", "vector", "none"),
        default="contextily",
        help=(
            "Map background: Ramp-aligned detailed web geography (default), "
            "offline Natural Earth 10m, legacy Ningbo vector fill, or no fill."
        ),
    )
    parser.add_argument(
        "--natural-earth-dir",
        type=Path,
        default=DEFAULT_NATURAL_EARTH_DIR,
        help="Directory containing ne_10m_land/lakes/rivers shapefile folders.",
    )
    parser.add_argument(
        "--no-coastline",
        action="store_true",
        help="Legacy alias for --background none.",
    )
    parser.add_argument("--station-size", type=float, default=10.0)
    parser.add_argument("--field-station-size", type=float, default=2.0)
    parser.add_argument(
        "--figure-width",
        type=float,
        default=3.15,
        help="Width in inches of each independently exported map panel.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=3.30,
        help="Height in inches of each independently exported map panel.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_EXPORT_DPI)
    parser.add_argument(
        "--max-timestamps",
        type=int,
        default=0,
        help="Debug only: deterministic cap over the requested split; zero uses all timestamps.",
    )
    return parser.parse_args()


def require(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with require(path, "JSON metadata").open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return result


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def station_sns_sha1(frame: pd.DataFrame) -> str:
    n_before = len(frame)
    valid_station_ids = frame["station_sn"].notna()
    n_after = int(valid_station_ids.sum())
    if n_after != n_before:
        raise ValueError(
            f"Station table contains {n_before - n_after} missing station_sn values; "
            "refusing to silently exclude them from the source-fleet hash"
        )
    values = sorted(frame.loc[valid_station_ids, "station_sn"].astype(str).tolist())
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True)


def compute_timestamp_statistics(
    timestamps: pd.DatetimeIndex,
    seconds: np.ndarray,
    station_values: np.ndarray,
    minimum_valid_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sec_to_index = {int(value): index for index, value in enumerate(seconds)}
    for index, (timestamp, timestamp_sec) in enumerate(zip(timestamps, seconds)):
        current = station_values[index]
        valid = np.isfinite(current)
        values = current[valid].astype(np.float64)
        previous_index = sec_to_index.get(int(timestamp_sec) - 15 * 60)
        delta = np.nan
        n_delta_pairs = 0
        if previous_index is not None:
            previous = station_values[previous_index]
            paired = valid & np.isfinite(previous)
            n_delta_pairs = int(paired.sum())
            if n_delta_pairs >= minimum_valid_count:
                delta = float(np.mean(np.abs(current[paired] - previous[paired])))
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "timestamp_sec": int(timestamp_sec),
                "n_valid_stations": int(valid.sum()),
                "coverage_fraction": float(valid.mean()),
                "regional_mean": float(np.mean(values)) if values.size else np.nan,
                "spatial_std": float(np.std(values)) if values.size else np.nan,
                "spatial_p10": float(np.quantile(values, 0.10)) if values.size else np.nan,
                "spatial_p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
                "spatial_contrast_p90_p10": (
                    float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
                    if values.size
                    else np.nan
                ),
                "temporal_change_mae_15min": delta,
                "n_temporal_pairs": n_delta_pairs,
            }
        )
    frame = pd.DataFrame(rows)
    frame["eligible"] = (
        (frame["n_valid_stations"] >= int(minimum_valid_count))
        & np.isfinite(frame["regional_mean"])
        & np.isfinite(frame["spatial_std"])
        & np.isfinite(frame["temporal_change_mae_15min"])
    )
    rank_sources = {
        "mean_percentile": "regional_mean",
        "std_percentile": "spatial_std",
        "contrast_percentile": "spatial_contrast_p90_p10",
        "delta_percentile": "temporal_change_mae_15min",
    }
    for rank_column, value_column in rank_sources.items():
        frame[rank_column] = np.nan
        frame.loc[frame["eligible"], rank_column] = percentile_rank(
            frame.loc[frame["eligible"], value_column]
        )
    frame["stable_high_generation_score"] = (
        frame["mean_percentile"]
        - 0.65 * frame["std_percentile"]
        - 0.65 * frame["delta_percentile"]
    )
    frame["spatially_heterogeneous_generation_score"] = (
        frame["std_percentile"]
        + frame["contrast_percentile"]
        - 0.20 * np.abs(frame["mean_percentile"] - 0.50)
        - 0.10 * frame["delta_percentile"]
    )
    frame["rapidly_varying_generation_score"] = (
        frame["delta_percentile"]
        + 0.40 * frame["std_percentile"]
        + 0.20 * frame["contrast_percentile"]
    )
    return frame


def threshold_candidates(frame: pd.DataFrame, case: str, args: argparse.Namespace) -> pd.DataFrame:
    eligible = frame[frame["eligible"]].copy()
    if case == "clear":
        return eligible[
            (eligible["mean_percentile"] >= float(args.clear_mean_quantile))
            & (eligible["std_percentile"] <= float(args.clear_std_quantile))
            & (eligible["delta_percentile"] <= float(args.clear_delta_quantile))
        ]
    if case == "cloudy":
        return eligible[
            (eligible["mean_percentile"] >= float(args.cloudy_mean_low_quantile))
            & (eligible["mean_percentile"] <= float(args.cloudy_mean_high_quantile))
            & (eligible["std_percentile"] >= float(args.cloudy_std_quantile))
        ]
    if case == "rapid":
        return eligible[
            (eligible["delta_percentile"] >= float(args.rapid_delta_quantile))
            & (eligible["std_percentile"] >= float(args.rapid_std_quantile))
        ]
    raise ValueError(f"Unknown case: {case}")


def case_override(args: argparse.Namespace, case: str) -> str:
    return str(getattr(args, f"{case}_time"))


def select_cases(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, int], dict[str, str]]:
    score_columns = {
        "clear": "stable_high_generation_score",
        "cloudy": "spatially_heterogeneous_generation_score",
        "rapid": "rapidly_varying_generation_score",
    }
    chosen: dict[str, int] = {}
    notes: dict[str, str] = {}
    gap_seconds = float(args.minimum_case_gap_hours) * 3600.0

    for case in CASE_ORDER:
        override = case_override(args, case).strip()
        if override:
            target = pd.Timestamp(override)
            match = frame.index[pd.to_datetime(frame["timestamp"]) == target].tolist()
            if not match:
                raise ValueError(f"--{case}-time {target} is not present in split={args.split}")
            index = int(match[0])
            if index in chosen.values():
                raise ValueError(f"Timestamp {target} was assigned to more than one case")
            chosen[case] = index
            notes[case] = "explicit timestamp override"
            continue

        candidates = threshold_candidates(frame, case, args)
        note = "strict quantile thresholds"
        if candidates.empty:
            candidates = frame[frame["eligible"]].copy()
            note = "thresholds relaxed because the strict candidate set was empty"
        if candidates.empty:
            raise RuntimeError(f"No eligible timestamps are available for the {case} case")

        if chosen:
            prior_seconds = frame.loc[list(chosen.values()), "timestamp_sec"].to_numpy(np.int64)
            keep = candidates["timestamp_sec"].apply(
                lambda value: bool(np.all(np.abs(prior_seconds - int(value)) >= gap_seconds))
            )
            separated = candidates[keep]
            if not separated.empty:
                candidates = separated
            else:
                candidates = candidates[~candidates.index.isin(chosen.values())]
                note += "; minimum time gap relaxed"
        if candidates.empty:
            raise RuntimeError(f"No distinct timestamp remains for the {case} case")

        candidates = candidates.sort_values(
            [score_columns[case], "timestamp_sec"],
            ascending=[False, True],
        )
        chosen[case] = int(candidates.index[0])
        notes[case] = note

    if len(set(chosen.values())) != len(CASE_ORDER):
        raise AssertionError(f"Case timestamps are not distinct: {chosen}")
    return chosen, notes


def style_map_axis(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    row_index: int,
    column_index: int,
) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax.tick_params(labelsize=6.5, pad=2)
    if row_index == 2:
        ax.set_xlabel("Longitude (°E)")
    ax.tick_params(axis="x", labelbottom=(row_index == 2))
    if column_index == 0:
        ax.set_ylabel("Latitude (°N)")
    ax.tick_params(axis="y", labelleft=(column_index == 0))


def save_case_figure(
    out_dir: Path,
    args: argparse.Namespace,
    source_df: pd.DataFrame,
    station_values: np.ndarray,
    selected_cache_indices: dict[str, int],
    selected_stat_indices: dict[str, int],
    stats: pd.DataFrame,
    fields: np.ndarray,
    masks: np.ndarray,
    bounds: tuple[float, float, float, float],
    region: Any,
    region_geometry: Any,
    vmax: float,
) -> tuple[list[str], list[str]]:
    cmap = framework_pv_colormap()
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(vmax), clip=True)
    outputs: list[str] = []
    background_sources: set[str] = set()

    def save_bundle(fig: plt.Figure, base_name: str) -> list[str]:
        base = out_dir / base_name
        paths = [
            base.with_suffix(".svg"),
            base.with_suffix(".pdf"),
            base.with_suffix(".tiff"),
            base.with_suffix(".png"),
        ]
        fig.savefig(paths[0], bbox_inches="tight")
        fig.savefig(paths[1], bbox_inches="tight")
        fig.savefig(paths[2], dpi=int(args.dpi), bbox_inches="tight")
        fig.savefig(paths[3], dpi=min(300, int(args.dpi)), bbox_inches="tight")
        plt.close(fig)
        return [str(path) for path in paths]

    def panel_canvas() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        fig = plt.figure(
            figsize=(float(args.figure_width), float(args.figure_height)),
            constrained_layout=False,
        )
        axis = fig.add_axes([0.18, 0.29, 0.79, 0.68])
        colorbar_axis = fig.add_axes([0.18, 0.105, 0.79, 0.035])
        return fig, axis, colorbar_axis

    def draw_panel(
        fig: plt.Figure,
        axis: plt.Axes,
        colorbar_axis: plt.Axes | None,
        case: str,
        kind: str,
        row_index: int,
        column_index: int,
        panel_label: str | None = None,
        map_anchor: str = "C",
    ) -> None:
        station_index = selected_stat_indices[case]
        cache_index = selected_cache_indices[case]
        values = station_values[station_index]
        field = np.asarray(fields[cache_index], dtype=np.float32)
        mask = np.asarray(masks[cache_index], dtype=bool)
        style_map_axis(axis, bounds, row_index, column_index)
        background_sources.add(
            draw_basemap(
                axis,
                region,
                region_geometry,
                args.background,
                args.natural_earth_dir,
            )
        )
        if kind == "source":
            axis.scatter(
                source_df["lon"],
                source_df["lat"],
                s=max(2.0, float(args.station_size) * 0.55),
                c="#CBD5E1",
                linewidths=0,
                alpha=0.70,
                zorder=2,
            )
            valid = np.isfinite(values)
            mappable = axis.scatter(
                source_df.loc[valid, "lon"],
                source_df.loc[valid, "lat"],
                c=values[valid],
                cmap=cmap,
                norm=norm,
                s=float(args.station_size),
                linewidths=0.18,
                edgecolors="#334155",
                alpha=0.96,
                zorder=3,
                rasterized=True,
            )
        elif kind == "field":
            display_field = np.ma.masked_where(~mask, field)
            mappable = axis.imshow(
                display_field,
                origin="lower",
                extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
                zorder=3,
            )
            axis.scatter(
                source_df["lon"],
                source_df["lat"],
                s=float(args.field_station_size),
                c="#334155",
                linewidths=0,
                alpha=0.36,
                zorder=4,
                rasterized=True,
            )
        else:
            raise ValueError(f"Unknown panel kind: {kind}")
        draw_boundary_on_top(axis, region)
        # Basemap and scientific rasters must not override the geographic
        # aspect.  Reapply it after every artist so source and field panels
        # occupy the same map box and one degree has the same scale.
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_anchor(map_anchor)
        if panel_label is not None:
            axis.text(
                -0.065,
                1.015,
                f"({panel_label})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#111827",
                clip_on=False,
                zorder=20,
            )
        if colorbar_axis is not None:
            colorbar = fig.colorbar(
                mappable,
                cax=colorbar_axis,
                orientation="horizontal",
            )
            colorbar.set_label("Normalized PV generation (p.u.)", fontsize=7, labelpad=2)
            colorbar.ax.tick_params(labelsize=6.2, pad=1.5)
            colorbar.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

    panel_specs = [
        (case, kind)
        for case in CASE_ORDER
        for kind in ("source", "field")
    ]
    for panel_number, (case, kind) in enumerate(panel_specs, start=1):
        fig, axis, colorbar_axis = panel_canvas()
        draw_panel(fig, axis, colorbar_axis, case, kind, 2, 0)
        suffix = "source_stations" if kind == "source" else "constructed_field"
        outputs.extend(
            save_bundle(fig, f"{panel_number:02d}_{CASE_SLUG[case]}_{suffix}")
        )

    preview = plt.figure(figsize=(7.16, 8.55), constrained_layout=False)
    outer = preview.add_gridspec(
        3,
        2,
        left=0.090,
        right=0.985,
        bottom=0.145,
        top=0.985,
        wspace=0.055,
        hspace=0.105,
    )
    for panel_number, (case, kind) in enumerate(panel_specs):
        row_index, column_index = divmod(panel_number, 2)
        axis = preview.add_subplot(outer[row_index, column_index])
        draw_panel(
            preview,
            axis,
            None,
            case,
            kind,
            row_index,
            column_index,
            panel_label=chr(ord("a") + panel_number),
            map_anchor="E" if column_index == 0 else "W",
        )
    shared_colorbar_axis = preview.add_axes([0.245, 0.052, 0.555, 0.018])
    shared_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    shared_mappable.set_array([])
    shared_colorbar = preview.colorbar(
        shared_mappable,
        cax=shared_colorbar_axis,
        orientation="horizontal",
    )
    shared_colorbar.set_label(
        "Normalized PV generation (p.u.)",
        fontsize=7,
        labelpad=2,
    )
    shared_colorbar.ax.tick_params(labelsize=6.2, pad=1.5)
    shared_colorbar.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))

    combined_base = out_dir / "spatial_representation_combined"
    combined_paths = [
        combined_base.with_suffix(".svg"),
        combined_base.with_suffix(".pdf"),
        combined_base.with_suffix(".tiff"),
        combined_base.with_suffix(".png"),
    ]
    preview.savefig(combined_paths[0], bbox_inches="tight")
    preview.savefig(combined_paths[1], bbox_inches="tight")
    preview.savefig(combined_paths[2], dpi=int(args.dpi), bbox_inches="tight")
    preview.savefig(combined_paths[3], dpi=min(300, int(args.dpi)), bbox_inches="tight")
    preview_path = out_dir / "spatial_representation_combined_preview.png"
    preview.savefig(preview_path, dpi=min(300, int(args.dpi)), bbox_inches="tight")
    plt.close(preview)
    outputs.extend(str(path) for path in combined_paths)
    outputs.append(str(preview_path))

    return (
        outputs,
        sorted(background_sources),
    )


def selected_case_table(
    stats: pd.DataFrame,
    selected: dict[str, int],
    notes: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in CASE_ORDER:
        row = stats.loc[selected[case]].to_dict()
        row["case"] = CASE_SLUG[case]
        row["display_label"] = CASE_DISPLAY[case]
        row["selection_note"] = notes[case]
        rows.append(row)
    columns = ["case", "display_label", "selection_note"] + [
        column for column in stats.columns if column not in {"selected_case", "selection_note"}
    ]
    return pd.DataFrame(rows)[columns]


def save_source_data(
    out_dir: Path,
    source_df: pd.DataFrame,
    station_values: np.ndarray,
    stats: pd.DataFrame,
    selected: dict[str, int],
    cell_y: np.ndarray,
    cell_x: np.ndarray,
) -> None:
    rows: list[pd.DataFrame] = []
    for case in CASE_ORDER:
        index = selected[case]
        frame = source_df[["station_sn", "lon", "lat"]].copy()
        frame.insert(0, "case", CASE_SLUG[case])
        frame.insert(1, "timestamp", str(pd.Timestamp(stats.loc[index, "timestamp"])))
        frame["cell_y"] = cell_y
        frame["cell_x"] = cell_x
        frame["measured_pv_normalized"] = station_values[index]
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(
        out_dir / "source_data_selected_station_measurements.csv",
        index=False,
    )


def save_neighbor_bundle(
    out_dir: Path,
    fields: np.ndarray,
    masks: np.ndarray,
    all_seconds: np.ndarray,
    selected_cache_indices: dict[str, int],
    station_values: np.ndarray,
    selected_stat_indices: dict[str, int],
    selected_seconds: np.ndarray,
) -> None:
    cache_lookup = {int(value): index for index, value in enumerate(all_seconds)}
    stat_lookup = {int(value): index for index, value in enumerate(selected_seconds)}
    field_triplet = np.full((3, 3, fields.shape[1], fields.shape[2]), np.nan, dtype=np.float32)
    mask_triplet = np.zeros((3, 3, masks.shape[1], masks.shape[2]), dtype=np.uint8)
    station_triplet = np.full((3, 3, station_values.shape[1]), np.nan, dtype=np.float32)
    timestamp_triplet = np.full((3, 3), -1, dtype=np.int64)
    for case_index, case in enumerate(CASE_ORDER):
        center_sec = int(all_seconds[selected_cache_indices[case]])
        for neighbor_index, offset in enumerate((-15 * 60, 0, 15 * 60)):
            timestamp_sec = center_sec + offset
            timestamp_triplet[case_index, neighbor_index] = timestamp_sec
            cache_index = cache_lookup.get(timestamp_sec)
            if cache_index is not None:
                field_triplet[case_index, neighbor_index] = np.asarray(fields[cache_index], dtype=np.float32)
                mask_triplet[case_index, neighbor_index] = np.asarray(masks[cache_index], dtype=np.uint8)
            stat_index = stat_lookup.get(timestamp_sec)
            if stat_index is not None:
                station_triplet[case_index, neighbor_index] = station_values[stat_index]
    np.savez_compressed(
        out_dir / "source_data_selected_cases_with_neighbors.npz",
        case=np.asarray([CASE_SLUG[case] for case in CASE_ORDER]),
        relative_minutes=np.asarray([-15, 0, 15], dtype=np.int16),
        timestamp_sec=timestamp_triplet,
        field=field_triplet,
        support_mask=mask_triplet,
        station_values=station_triplet,
    )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    cases: pd.DataFrame,
    n_sources: int,
    n_targets: int,
    geography_source: str,
    background_sources: list[str],
    vmax: float,
    outputs: list[str],
) -> None:
    lines = [
        "# Visual validation of the spatial representation",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: the source-supported regional field retains meaningful spatial organization across stable, heterogeneous, and rapidly varying fleet states.",
        "- Figure archetype: six independently exported map panels + quantitative selection metadata.",
        "- Hero evidence: direct source-measurement versus constructed-field comparison at the same timestamp.",
        "- Layout: numbered files 01-06 remain independent maps with embedded horizontal color bars; the combined 3x2 figure uses panel labels (a)-(f) and one shared horizontal color bar. No titles are embedded.",
        "- Validation evidence: deterministic fleet-statistic selection over the full requested split.",
        "- Interpretation boundary: the case labels are operational categories inferred from source PV statistics, not independent meteorological ground-truth labels.",
        "",
        "## Protocol",
        "",
        f"- Evaluated split: `{args.split}`",
        f"- Source stations: {n_sources}",
        f"- Held-out targets excluded from selection and plotting: {n_targets}",
        "- Temporal change: mean source-station absolute change relative to the preceding 15-min frame.",
        f"- Shared color scale: [0, {vmax:g}]",
        f"- Display geography source: `{geography_source}`",
        f"- Background source: {', '.join(background_sources)}",
        "- Unsupported field cells are masked rather than displayed as zero generation.",
        "- The rapidly varying case's -15/0/+15 min neighbors are retained in the exported NPZ for a separate temporal-evolution panel if needed; they are not inserted into the compact 3x2 main figure.",
        "",
        "## Selected cases",
        "",
        "| Case | Timestamp | n | Mean | Spatial SD | P90-P10 | 15-min change | Selection |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in cases.itertuples(index=False):
        lines.append(
            f"| {row.display_label} | {pd.Timestamp(row.timestamp)} | {int(row.n_valid_stations)} | "
            f"{float(row.regional_mean):.4f} | {float(row.spatial_std):.4f} | "
            f"{float(row.spatial_contrast_p90_p10):.4f} | "
            f"{float(row.temporal_change_mae_15min):.4f} | {row.selection_note} |"
        )
    lines.extend(
        [
            "",
            "## Suggested manuscript caption",
            "",
            "**Fig. X | Visual validation of the source-supported regional PV generation field.** Panels (a,c,e) show normalized PV observations from the source fleet, and panels (b,d,f) show the corresponding source-supported regional fields under stable high-generation, spatially heterogeneous generation, and rapidly varying generation regimes, respectively. The regimes are selected deterministically from the test period using regional mean generation, across-station standard deviation, and mean 15-min absolute generation change; they are operating regimes rather than independently verified weather classes. All panels use one shared color scale, unsupported grid cells are masked, and held-out target stations are excluded from both field construction and visualization.",
            "",
            "## Reviewer-risk note",
            "",
            (
                "Do not present the three operational labels as independently verified weather classes unless they are cross-checked against satellite/NWP cloud information. "
                + (
                    "The default detailed web basemap follows the Ramp contextily presentation and is not clipped to an administrative polygon. "
                    if args.background == "contextily"
                    else "The selected offline/vector background is display-only and may contain less detailed coastal geometry than the web basemap. "
                )
                + "The basemap does not mask, move, or modify the scientific data layers."
            ),
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in outputs)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.no_coastline:
        args.background = "none"
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
    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        for obsolete_stem in (
            "spatial_representation_three_cases",
            "07_shared_colorbar",
            "01_clear_source_stations",
            "02_clear_constructed_field",
            "03_cloudy_source_stations",
            "04_cloudy_constructed_field",
            "05_rapid_source_stations",
            "06_rapid_constructed_field",
        ):
            obsolete_output = out_dir / f"{obsolete_stem}{suffix}"
            if obsolete_output.is_file():
                obsolete_output.unlink()
    stage3 = require(root / "stage_3", "Stage-3 directory")
    helpers = consistency.load_pipeline_helpers(stage3)

    stage3_data_dir = resolve_stage3_data_dir(
        root,
        old_data_root,
        args.stage3_data_dir,
    )
    selection_dir = stage3_data_dir / "02_target_selection_lhs50"
    run_dir = stage3_data_dir / "03_target_excluded_run"
    cache_dir = stage3_data_dir / "04_target_excluded_pvnorm_frame_cache"
    all_path = require(run_dir / "station_split.csv", "full station split")
    source_path = require(run_dir / "target_excluded_station_split.csv", "source station split")
    target_path = require(selection_dir / "target_stations.csv", "held-out target list")
    q95_path = require(selection_dir / "target_excluded_station_pv_norm_q95.csv", "source q95 table")
    selection_meta_path = require(selection_dir / "01_target_selection_meta.json", "selection metadata")
    field_path = require(cache_dir / "fields_float16.npy", "PV field cache")
    mask_path = require(cache_dir / "masks_uint8.npy", "support mask cache")
    second_path = require(cache_dir / "timestamp_sec.npy", "field timestamp cache")
    frame_meta_path = require(cache_dir / "frame_cache_meta.json", "field cache metadata")

    all_df = pd.read_csv(all_path, dtype={"station_sn": str})
    source_df = pd.read_csv(source_path, dtype={"station_sn": str}).reset_index(drop=True)
    target_df = pd.read_csv(target_path, dtype={"station_sn": str})
    q95_df = pd.read_csv(q95_path, dtype={"station_sn": str})
    selection_meta = read_json(selection_meta_path)
    frame_meta = read_json(frame_meta_path)
    settings = consistency.load_settings(selection_meta, frame_meta)

    for label, frame in (("all", all_df), ("source", source_df), ("target", target_df)):
        missing = {"station_sn", "lon", "lat"} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} station table is missing columns: {sorted(missing)}")
    overlap = set(source_df["station_sn"].astype(str)) & set(target_df["station_sn"].astype(str))
    if overlap:
        raise AssertionError(f"Source-target overlap detected: {sorted(overlap)[:10]}")
    if int(frame_meta.get("source_station_count", len(source_df))) != len(source_df):
        raise AssertionError("Source count in field metadata does not match the source station table")
    cache_hash = str(frame_meta.get("source_station_sns_sha1", "")).strip()
    source_hash = station_sns_sha1(source_df)
    if cache_hash and cache_hash != source_hash:
        raise AssertionError(f"Source station hash mismatch: cache={cache_hash}, csv={source_hash}")

    q95_lookup = q95_df.set_index("station_sn")["pv_norm_q95"].astype(float).to_dict()
    missing_q95 = [station for station in source_df["station_sn"] if station not in q95_lookup]
    if missing_q95:
        raise ValueError(f"Missing q95 for {len(missing_q95)} source stations")

    fields = np.load(field_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    all_seconds = np.asarray(np.load(second_path, mmap_mode="r"), dtype=np.int64)
    if fields.shape != masks.shape or fields.shape[0] != len(all_seconds):
        raise AssertionError(f"Field-cache shape mismatch: fields={fields.shape}, masks={masks.shape}")

    all_timestamps = pd.to_datetime(all_seconds, unit="s")
    meta_args = selection_meta.get("args", {}) if isinstance(selection_meta.get("args", {}), dict) else {}
    train_end = pd.Timestamp(meta_args.get("train_end", "2023-01-13 23:59:59"))
    val_end = pd.Timestamp(meta_args.get("val_end", "2023-03-08 23:59:59"))
    selected_cache = np.flatnonzero(
        consistency.split_mask(all_timestamps, args.split, train_end, val_end)
    )
    if selected_cache.size == 0:
        raise RuntimeError(f"No timestamps found for split={args.split}")
    timestamp_selection_note = "all timestamps in requested split"
    if int(args.max_timestamps) > 0 and len(selected_cache) > int(args.max_timestamps):
        selected_cache = selected_cache[: int(args.max_timestamps)]
        timestamp_selection_note = (
            f"debug contiguous-prefix cap: {len(selected_cache)} timestamps; "
            "not suitable for manuscript case selection"
        )

    seconds = all_seconds[selected_cache]
    timestamps = all_timestamps[selected_cache]
    station_values = np.full((len(selected_cache), len(source_df)), np.nan, dtype=np.float32)
    load_rows: list[dict[str, Any]] = []
    for station_index, row in source_df.iterrows():
        station_sn = str(row["station_sn"])
        values, load_report = consistency.load_station_values(
            station_sn,
            seconds,
            float(q95_lookup[station_sn]),
            helpers,
            args,
            settings,
        )
        station_values[:, station_index] = values
        load_rows.append(load_report)
        if (station_index + 1) % 50 == 0 or station_index + 1 == len(source_df):
            print(f"[station data] {station_index + 1}/{len(source_df)}", flush=True)
    pd.DataFrame(load_rows).to_csv(out_dir / "station_load_and_health_filter_report.csv", index=False)
    if not np.isfinite(station_values).any():
        raise RuntimeError("No source-station PV values aligned with the requested split")

    minimum_valid_count = max(
        20,
        int(math.ceil(float(args.minimum_valid_fraction) * len(source_df))),
    )
    stats = compute_timestamp_statistics(timestamps, seconds, station_values, minimum_valid_count)
    selected_stat_indices, selection_notes = select_cases(stats, args)
    stats["selected_case"] = ""
    stats["selection_note"] = ""
    for case, index in selected_stat_indices.items():
        stats.loc[index, "selected_case"] = CASE_SLUG[case]
        stats.loc[index, "selection_note"] = selection_notes[case]
    stats.to_csv(out_dir / "source_data_case_selection_metrics.csv", index=False)

    selected_cache_indices = {
        case: int(selected_cache[index]) for case, index in selected_stat_indices.items()
    }
    cases = selected_case_table(stats, selected_stat_indices, selection_notes)
    cases.to_csv(out_dir / "selected_cases.csv", index=False)

    bounds = helpers.bounds(all_df)
    cells = [
        helpers.xy_to_ij(float(row.lon), float(row.lat), bounds, int(settings["grid_size"]))
        for row in source_df.itertuples()
    ]
    cell_y = np.asarray([cell[0] for cell in cells], dtype=np.int32)
    cell_x = np.asarray([cell[1] for cell in cells], dtype=np.int32)
    save_source_data(
        out_dir,
        source_df,
        station_values,
        stats,
        selected_stat_indices,
        cell_y,
        cell_x,
    )
    save_neighbor_bundle(
        out_dir,
        fields,
        masks,
        all_seconds,
        selected_cache_indices,
        station_values,
        selected_stat_indices,
        seconds,
    )

    if args.background in {"natural-earth", "contextily"}:
        region, region_geometry = None, None
        geography_source = (
            "online detailed basemap; Esri WorldGrayCanvas with OSM fallback"
            if args.background == "contextily"
            else str(Path(args.natural_earth_dir).resolve())
        )
    else:
        region, region_geometry = load_boundary(args.boundary, args.boundary_crs)
        geography_source = str(Path(args.boundary).resolve())
    vmax = float(args.vmax) if float(args.vmax) > 0 else float(settings["pv_field_clip"])
    figure_outputs, background_sources = save_case_figure(
        out_dir,
        args,
        source_df,
        station_values,
        selected_cache_indices,
        selected_stat_indices,
        stats,
        fields,
        masks,
        bounds,
        region,
        region_geometry,
        vmax,
    )

    metadata = {
        "purpose": "visual validation of the source-supported regional PV generation field",
        "interpretation_boundary": (
            "operational case labels inferred from source PV statistics; not independent weather ground truth"
        ),
        "project_root": str(root),
        "old_data_root": str(old_data_root),
        "stage3_data_dir": str(stage3_data_dir),
        "pv_root": str(Path(args.pv_root).resolve()) if args.pv_root is not None else str(old_data_root / "PV"),
        "train_pv_dir": str(args.train_pv_dir),
        "val_pv_dir": str(args.val_pv_dir),
        "split": args.split,
        "timestamp_selection": timestamp_selection_note,
        "source_station_count": int(len(source_df)),
        "held_out_target_count": int(len(target_df)),
        "source_target_overlap": 0,
        "minimum_valid_station_count": int(minimum_valid_count),
        "minimum_case_gap_hours": float(args.minimum_case_gap_hours),
        "field_value_definition": frame_meta.get("value_type", "pv_norm_div_station_pv_norm_q95"),
        "shared_color_scale": [0.0, vmax],
        "colormap": "framework_pv_blue_cream_orange_red",
        "colormap_stops": [
            {"position": float(position), "rgb_8bit": list(rgb)}
            for position, rgb in FRAMEWORK_PV_COLOR_STOPS
        ],
        "unsupported_cells": "masked",
        "layout": "six numbered independent map panels with embedded colorbars plus a 3x2 combined figure with one shared colorbar",
        "embedded_titles": False,
        "embedded_panel_letters": {
            "standalone": False,
            "combined": True,
            "labels": ["a", "b", "c", "d", "e", "f"],
        },
        "embedded_case_annotations": False,
        "support_mask_display": "unsupported cells transparent; no explicit support-boundary outline",
        "rapid_neighbor_frames_exported": [-15, 0, 15],
        "display_geography_source": geography_source,
        "background_mode": args.background,
        "background_sources": background_sources,
        "selected_cases": cases.to_dict(orient="records"),
        "selection_thresholds": {
            "stable_high_generation": {
                "mean_quantile_min": float(args.clear_mean_quantile),
                "std_quantile_max": float(args.clear_std_quantile),
                "delta_quantile_max": float(args.clear_delta_quantile),
            },
            "spatially_heterogeneous_generation": {
                "mean_quantile_min": float(args.cloudy_mean_low_quantile),
                "mean_quantile_max": float(args.cloudy_mean_high_quantile),
                "std_quantile_min": float(args.cloudy_std_quantile),
            },
            "rapidly_varying_generation": {
                "delta_quantile_min": float(args.rapid_delta_quantile),
                "std_quantile_min": float(args.rapid_std_quantile),
            },
        },
        "selection_scores": {
            "stable_high_generation": "mean_percentile - 0.65*std_percentile - 0.65*delta_percentile",
            "spatially_heterogeneous_generation": "std_percentile + contrast_percentile - 0.20*abs(mean_percentile-0.5) - 0.10*delta_percentile",
            "rapidly_varying_generation": "delta_percentile + 0.40*std_percentile + 0.20*contrast_percentile",
        },
        "artifacts": {
            "field_cache": str(field_path),
            "support_mask_cache": str(mask_path),
            "source_station_table": str(source_path),
            "target_station_table": str(target_path),
            "q95_table": str(q95_path),
        },
        "outputs": figure_outputs,
    }
    write_json(out_dir / "visualization_metadata.json", metadata)

    report_outputs = [
        *figure_outputs,
        str(out_dir / "selected_cases.csv"),
        str(out_dir / "source_data_case_selection_metrics.csv"),
        str(out_dir / "source_data_selected_station_measurements.csv"),
        str(out_dir / "source_data_selected_cases_with_neighbors.npz"),
        str(out_dir / "station_load_and_health_filter_report.csv"),
        str(out_dir / "visualization_metadata.json"),
    ]
    report_path = out_dir / "spatial_representation_visualization.md"
    write_summary(
        report_path,
        args,
        cases,
        len(source_df),
        len(target_df),
        geography_source,
        background_sources,
        vmax,
        report_outputs,
    )

    print("==== Spatial representation visualization complete ====")
    print(cases[["case", "timestamp", "regional_mean", "spatial_std", "temporal_change_mae_15min"]].to_string(index=False))
    print(f"figure: {figure_outputs[0]}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
