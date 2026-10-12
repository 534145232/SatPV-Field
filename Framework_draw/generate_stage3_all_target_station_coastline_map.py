#!/usr/bin/env python3
"""Draw all Stage 3 PV stations and highlight the selected target stations.

The coastline/background style intentionally follows
generate_stage3_remaining_station_coastline_map.py.  The output is designed as
a clean framework-map asset with optional lightweight lon/lat ticks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import TARGET_EXCLUDED_RUN_DIR, TARGET_SELECTION_DIR  # noqa: E402
import generate_stage3_remaining_station_coastline_map as coastmap  # noqa: E402
from generate_stage3_remaining_station_coastline_map import (  # noqa: E402
    draw_map_background,
    ensure_dir,
    map_bounds,
    pv_field_like_bounds,
    read_station_csv,
    resolve_background_sources,
)


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_all_target_station_map"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--station_csv", type=Path, default=None, help="All-station CSV. Defaults to station_split.csv.")
    parser.add_argument("--target_csv", type=Path, default=None, help="Selected target-station CSV. Defaults to target_stations.csv.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_name", type=str, default="all_stations_with_target_stations")
    parser.add_argument("--dpi", type=int, default=650)
    parser.add_argument("--width", type=float, default=5.0)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--pad_frac", type=float, default=0.18)
    parser.add_argument(
        "--match_pv_field_style",
        type=int,
        default=1,
        help="Use the same 4%% padded bounds and square/non-equal display style as Stage 3 PV field assets.",
    )
    parser.add_argument("--transparent_background", type=int, default=0)
    parser.add_argument("--figure_bg_color", type=str, default="#ffffff")
    parser.add_argument("--water_color", type=str, default="#e9f3f8")
    parser.add_argument("--land_color", type=str, default="#f7f7f5")
    parser.add_argument("--land_edge_color", type=str, default="#d8dee4")
    parser.add_argument("--river_color", type=str, default="#c6deec")
    parser.add_argument("--all_station_size", type=float, default=7.0)
    parser.add_argument("--all_station_color", type=str, default="#6f8fa8")
    parser.add_argument("--all_station_alpha", type=float, default=0.62)
    parser.add_argument("--target_size", type=float, default=24.0)
    parser.add_argument("--target_color", type=str, default="#d64a45")
    parser.add_argument("--target_edge_color", type=str, default="#9d252b")
    parser.add_argument("--target_edge_width", type=float, default=0.85)
    parser.add_argument("--target_marker", type=str, default="o")
    parser.add_argument("--target_fill", type=int, default=0)
    parser.add_argument("--draw_target_ring", type=int, default=0)
    parser.add_argument("--target_ring_size", type=float, default=34.0)
    parser.add_argument("--target_ring_color", type=str, default="#8f1d1d")
    parser.add_argument("--target_ring_width", type=float, default=0.55)
    parser.add_argument("--draw_latlon_ticks", type=int, default=1)
    parser.add_argument("--tick_count", type=int, default=3)
    parser.add_argument("--tick_fontsize", type=float, default=7.2)
    parser.add_argument("--tick_color", type=str, default="#4a545c")
    parser.add_argument("--draw_grid", type=int, default=1)
    parser.add_argument("--grid_color", type=str, default="#9ca8b0")
    parser.add_argument("--grid_alpha", type=float, default=0.18)
    parser.add_argument("--grid_width", type=float, default=0.38)
    parser.add_argument("--frame_color", type=str, default="#5e6a72")
    parser.add_argument("--frame_width", type=float, default=0.65)
    parser.add_argument("--draw_domain_box", type=int, default=1)
    parser.add_argument("--domain_box_color", type=str, default="#7b8790")
    parser.add_argument("--domain_box_width", type=float, default=0.9)
    parser.add_argument("--domain_box_alpha", type=float, default=0.78)
    parser.add_argument("--domain_box_inset_frac", type=float, default=0.008)
    parser.add_argument("--draw_legend", type=int, default=1)
    parser.add_argument("--legend_loc", type=str, default="lower left")
    parser.add_argument("--legend_fontsize", type=float, default=7.2)
    parser.add_argument("--land_shp", type=Path, default=None)
    parser.add_argument("--lakes_shp", type=Path, default=None)
    parser.add_argument("--rivers_shp", type=Path, default=None)
    parser.add_argument(
        "--coastline_source",
        choices=["auto", "natural_earth", "built_in", "none"],
        default="auto",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_all_station_csv(args: argparse.Namespace) -> Path:
    if args.station_csv is not None:
        return Path(args.station_csv)
    candidates = [
        Path(args.target_run_dir) / "station_split.csv",
        Path(args.selection_dir) / "station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_target_csv(args: argparse.Namespace) -> Path:
    if args.target_csv is not None:
        return Path(args.target_csv)
    return Path(args.selection_dir) / "target_stations.csv"


def deduplicate_targets(all_stations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Prefer exact target CSV coordinates, but keep only targets present in all-station list."""
    if targets.empty:
        return targets
    target_sns = set(targets["station_sn"].astype(str))
    all_sns = set(all_stations["station_sn"].astype(str))
    missing = sorted(target_sns - all_sns)
    if missing:
        print(f"[warn] {len(missing)} target stations are not in all-station table; examples: {missing[:5]}", flush=True)
    return targets[targets["station_sn"].astype(str).isin(all_sns)].drop_duplicates("station_sn").reset_index(drop=True)


def configure_lonlat_axes(ax, bounds: tuple[float, float, float, float], args: argparse.Namespace) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    count = max(2, int(args.tick_count))
    lon_ticks = np.linspace(lon_min, lon_max, count)
    lat_ticks = np.linspace(lat_min, lat_max, count)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f"{x:.2f}E" for x in lon_ticks])
    ax.set_yticklabels([f"{y:.2f}N" for y in lat_ticks])
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=float(args.tick_fontsize),
        colors=str(args.tick_color),
        length=2.8,
        width=0.65,
        pad=2.0,
        direction="out",
    )
    if int(args.draw_grid):
        ax.grid(
            True,
            color=str(args.grid_color),
            alpha=float(args.grid_alpha),
            linewidth=float(args.grid_width),
            linestyle="-",
            zorder=1,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(str(args.frame_color))
        spine.set_linewidth(float(args.frame_width))


def apply_map_colors(args: argparse.Namespace) -> None:
    coastmap.OCEAN_COLOR = str(args.water_color)
    coastmap.LAND_COLOR = str(args.land_color)
    coastmap.LAND_EDGE_COLOR = str(args.land_edge_color)
    coastmap.RIVER_COLOR = str(args.river_color)


def draw_domain_box(ax, bounds: tuple[float, float, float, float], args: argparse.Namespace) -> None:
    if not int(args.draw_domain_box):
        return
    from matplotlib.patches import Rectangle

    lon_min, lon_max, lat_min, lat_max = bounds
    inset = float(args.domain_box_inset_frac)
    dx = (lon_max - lon_min) * inset
    dy = (lat_max - lat_min) * inset
    rect = Rectangle(
        (lon_min + dx, lat_min + dy),
        (lon_max - lon_min) - 2.0 * dx,
        (lat_max - lat_min) - 2.0 * dy,
        fill=False,
        linestyle=(0, (4.0, 3.0)),
        linewidth=float(args.domain_box_width),
        edgecolor=str(args.domain_box_color),
        alpha=float(args.domain_box_alpha),
        zorder=4,
    )
    ax.add_patch(rect)


def draw_legend(ax, args: argparse.Namespace) -> None:
    if not int(args.draw_legend):
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=str(args.all_station_color),
            markeredgecolor="none",
            alpha=float(args.all_station_alpha),
            markersize=4.2,
            label="Source stations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=str(args.target_color) if int(args.target_fill) else "none",
            markeredgecolor=str(args.target_edge_color),
            markeredgewidth=float(args.target_edge_width),
            markersize=5.2,
            label="Held-out target stations",
        ),
    ]
    if int(args.draw_domain_box):
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                linestyle=(0, (4.0, 3.0)),
                linewidth=float(args.domain_box_width),
                edgecolor=str(args.domain_box_color),
                label="PV-field domain",
            )
        )
    legend = ax.legend(
        handles=handles,
        loc=str(args.legend_loc),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#d4d8dc",
        framealpha=0.88,
        fontsize=float(args.legend_fontsize),
        handlelength=1.45,
        borderpad=0.35,
        labelspacing=0.35,
    )
    legend.set_zorder(20)


def plot_map(
    all_stations: pd.DataFrame,
    targets: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    land_shp: Path | None,
    lakes_shp: Path | None,
    rivers_shp: Path | None,
    background_source: str,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_map_colors(args)
    lon_min, lon_max, lat_min, lat_max = bounds
    if int(args.match_pv_field_style):
        side = max(float(args.width), float(args.height))
        fig, ax = plt.subplots(figsize=(side, side))
    else:
        fig, ax = plt.subplots(figsize=(float(args.width), float(args.height)))
    fig.patch.set_facecolor(str(args.figure_bg_color))
    fig.patch.set_alpha(0.0 if int(args.transparent_background) else 1.0)
    ax.set_facecolor(str(args.water_color))
    draw_map_background(
        ax,
        bounds,
        land_shp,
        lakes_shp,
        rivers_shp,
        use_builtin=(background_source == "built_in_fallback"),
    )
    draw_domain_box(ax, bounds, args)

    ax.scatter(
        all_stations["lon"],
        all_stations["lat"],
        s=float(args.all_station_size),
        c=str(args.all_station_color),
        alpha=float(args.all_station_alpha),
        linewidths=0.0,
        zorder=5,
    )

    if not targets.empty and int(args.draw_target_ring):
        ax.scatter(
            targets["lon"],
            targets["lat"],
            s=float(args.target_ring_size),
            facecolors="none",
            edgecolors=str(args.target_ring_color),
            linewidths=float(args.target_ring_width),
            alpha=0.92,
            marker="o",
            zorder=6,
        )

    if not targets.empty:
        target_face = str(args.target_color) if int(args.target_fill) else "none"
        ax.scatter(
            targets["lon"],
            targets["lat"],
            s=float(args.target_size),
            facecolors=target_face,
            edgecolors=str(args.target_edge_color),
            linewidths=float(args.target_edge_width),
            alpha=0.96,
            marker=str(args.target_marker),
            zorder=7,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    if int(args.match_pv_field_style):
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
    if int(args.draw_latlon_ticks):
        configure_lonlat_axes(ax, bounds, args)
        draw_legend(ax, args)
        fig.subplots_adjust(left=0.125, right=0.985, bottom=0.095, top=0.985)
    else:
        draw_legend(ax, args)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)

    png_path = out_base.with_suffix(".png")
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    transparent = bool(int(args.transparent_background))
    if int(args.match_pv_field_style) and not int(args.draw_latlon_ticks):
        fig.savefig(png_path, dpi=int(args.dpi), transparent=transparent, pad_inches=0)
        fig.savefig(svg_path, transparent=transparent, pad_inches=0)
        fig.savefig(pdf_path, transparent=transparent, pad_inches=0)
    else:
        fig.savefig(png_path, dpi=int(args.dpi), transparent=transparent, bbox_inches="tight", pad_inches=0.025)
        fig.savefig(svg_path, transparent=transparent, bbox_inches="tight", pad_inches=0.025)
        fig.savefig(pdf_path, transparent=transparent, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    station_csv = resolve_all_station_csv(args)
    target_csv = resolve_target_csv(args)

    all_stations = read_station_csv(station_csv)
    targets = read_station_csv(target_csv) if target_csv.exists() else pd.DataFrame(columns=all_stations.columns)
    targets = deduplicate_targets(all_stations, targets)

    bounds = pv_field_like_bounds(all_stations) if int(args.match_pv_field_style) else map_bounds(all_stations, float(args.pad_frac))
    land_shp, lakes_shp, rivers_shp, coast_source = resolve_background_sources(args, out_dir)
    out_base = out_dir / str(args.out_name)
    plot_map(all_stations, targets, bounds, land_shp, lakes_shp, rivers_shp, coast_source, out_base, args)

    write_json(
        out_dir / f"{args.out_name}_meta.json",
        {
            "purpose": "Transparent framework asset: all Stage 3 stations with selected target stations highlighted.",
            "station_csv": str(station_csv),
            "target_csv": str(target_csv) if target_csv.exists() else None,
            "all_station_count": int(len(all_stations)),
            "target_station_count": int(len(targets)),
            "bounds": {
                "lon_min": float(bounds[0]),
                "lon_max": float(bounds[1]),
                "lat_min": float(bounds[2]),
                "lat_max": float(bounds[3]),
            },
            "coastline_source": coast_source,
            "match_pv_field_style": bool(int(args.match_pv_field_style)),
            "draw_latlon_ticks": bool(int(args.draw_latlon_ticks)),
            "draw_legend": bool(int(args.draw_legend)),
            "draw_domain_box": bool(int(args.draw_domain_box)),
            "transparent_background": bool(int(args.transparent_background)),
            "land_shp": str(land_shp) if land_shp is not None else None,
            "lakes_shp": str(lakes_shp) if lakes_shp is not None else None,
            "rivers_shp": str(rivers_shp) if rivers_shp is not None else None,
            "outputs": {
                "png": str(out_base.with_suffix(".png")),
                "svg": str(out_base.with_suffix(".svg")),
                "pdf": str(out_base.with_suffix(".pdf")),
            },
        },
    )
    print("==== Stage 3 all/target station coastline map complete ====")
    print(f"all stations: {len(all_stations)} from {station_csv}")
    print(f"target stations: {len(targets)} from {target_csv}")
    print(f"coastline source: {coast_source}")
    print(f"png: {out_base.with_suffix('.png')}")
    print(f"svg: {out_base.with_suffix('.svg')}")
    print(f"pdf: {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
