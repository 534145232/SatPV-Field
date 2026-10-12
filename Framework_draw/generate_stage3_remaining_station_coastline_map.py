#!/usr/bin/env python3
"""Draw a clean transparent map of Stage 3 remaining PV-field stations.

The default station table is the Stage 3 target-excluded station split, i.e.
the source stations left after selecting and removing the 50 target stations.
The output intentionally has no title, axes, legend, ticks, labels or colorbar.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import TARGET_EXCLUDED_RUN_DIR, TARGET_SELECTION_DIR  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_remaining_station_map"
OCEAN_COLOR = "#dceeff"
LAND_COLOR = "#f1f1f1"
LAND_EDGE_COLOR = "#ffffff"
RIVER_COLOR = "#b9dcff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--station_csv", type=Path, default=None)
    parser.add_argument("--target_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_name", type=str, default="remaining_stations_coastline")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=4.8)
    parser.add_argument("--height", type=float, default=4.8)
    parser.add_argument("--pad_frac", type=float, default=0.18)
    parser.add_argument(
        "--match_pv_field_style",
        type=int,
        default=1,
        help="Use the same 4%% padded bounds and square/non-equal display style as Stage 3 PV field assets.",
    )
    parser.add_argument("--station_size", type=float, default=18.0)
    parser.add_argument("--station_color", type=str, default="#1565a9")
    parser.add_argument("--station_alpha", type=float, default=0.88)
    parser.add_argument("--coast_color", type=str, default="#475569")
    parser.add_argument("--coast_width", type=float, default=1.05)
    parser.add_argument("--draw_targets", type=int, default=0, help="Optional debug overlay for excluded 50 targets.")
    parser.add_argument("--target_size", type=float, default=30.0)
    parser.add_argument("--target_color", type=str, default="#d94848")
    parser.add_argument("--land_shp", type=Path, default=None)
    parser.add_argument("--lakes_shp", type=Path, default=None)
    parser.add_argument("--rivers_shp", type=Path, default=None)
    parser.add_argument(
        "--coastline_source",
        choices=["auto", "natural_earth", "built_in", "none"],
        default="auto",
        help="auto tries Natural Earth land/lakes/rivers first, then falls back to a built-in sketch.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_station_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"station CSV not found: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = {"station_sn", "lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["station_sn", "lon", "lat"]).copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return df


def resolve_station_csv(args: argparse.Namespace) -> Path:
    if args.station_csv is not None:
        return Path(args.station_csv)
    candidates = [
        Path(args.target_run_dir) / "target_excluded_station_split.csv",
        Path(args.selection_dir) / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_target_csv(args: argparse.Namespace) -> Path:
    if args.target_csv is not None:
        return Path(args.target_csv)
    return Path(args.selection_dir) / "target_stations.csv"


def map_bounds(stations: pd.DataFrame, pad_frac: float) -> tuple[float, float, float, float]:
    lon_min = float(stations["lon"].min())
    lon_max = float(stations["lon"].max())
    lat_min = float(stations["lat"].min())
    lat_max = float(stations["lat"].max())
    lon_pad = max(0.08, (lon_max - lon_min) * float(pad_frac))
    lat_pad = max(0.08, (lat_max - lat_min) * float(pad_frac))
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def pv_field_like_bounds(stations: pd.DataFrame) -> tuple[float, float, float, float]:
    """Match Stage 3 PV-field bound construction exactly."""
    lon_min = float(stations["lon"].min())
    lon_max = float(stations["lon"].max())
    lat_min = float(stations["lat"].min())
    lat_max = float(stations["lat"].max())
    lon_pad = max(1.0e-6, (lon_max - lon_min) * 0.04)
    lat_pad = max(1.0e-6, (lat_max - lat_min) * 0.04)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def ensure_natural_earth(out_dir: Path) -> tuple[Path, Path | None, Path | None]:
    try:
        import shapefile  # noqa: F401
    except ImportError as exc:
        raise ImportError("Please install pyshp first: pip install pyshp") from exc

    cache_dir = out_dir / "_natural_earth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_unzip(url: str, stem: str) -> Path:
        extract_dir = cache_dir / stem
        shp_path = extract_dir / f"{stem}.shp"
        if shp_path.exists():
            return shp_path
        zip_path = cache_dir / f"{stem}.zip"
        print(f"Downloading Natural Earth: {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        if not shp_path.exists():
            raise FileNotFoundError(f"No shapefile found after extracting {zip_path}")
        return shp_path

    land = download_and_unzip(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        "ne_10m_land",
    )
    lakes = download_and_unzip(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip",
        "ne_10m_lakes",
    )
    rivers = download_and_unzip(
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_rivers_lake_centerlines.zip",
        "ne_10m_rivers_lake_centerlines",
    )
    return land, lakes, rivers


def add_shapefile_polygons(
    ax,
    shp_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    zorder: int,
) -> None:
    import shapefile
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    sf = shapefile.Reader(str(shp_path))
    poly_list = []
    for shape in sf.shapes():
        pts = np.asarray(shape.points, dtype=float)
        if pts.size == 0:
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        if x.max() < lon_min or x.min() > lon_max or y.max() < lat_min or y.min() > lat_max:
            continue
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i]:parts[i + 1]]
            if len(seg) >= 3:
                poly_list.append(Polygon(seg, closed=True))
    if not poly_list:
        return
    pc = PatchCollection(
        poly_list,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_collection(pc)


def add_shapefile_lines(
    ax,
    shp_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    import shapefile
    from matplotlib.collections import LineCollection

    sf = shapefile.Reader(str(shp_path))
    segments = []
    for shape in sf.shapes():
        pts = np.asarray(shape.points, dtype=float)
        if pts.size == 0:
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        if x.max() < lon_min or x.min() > lon_max or y.max() < lat_min or y.min() > lat_max:
            continue
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i]:parts[i + 1]]
            if len(seg) >= 2:
                segments.append(seg)
    if not segments:
        return
    lc = LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder, capstyle="round")
    ax.add_collection(lc)


def built_in_ningbo_coastlines() -> list[np.ndarray]:
    """A compact hand-sketched coastline fallback for Ningbo/Zhoushan context.

    Coordinates are intentionally coarse and used only when a proper coastline
    provider is unavailable in the runtime.
    """
    mainland = np.asarray(
        [
            [120.55, 30.42],
            [120.73, 30.31],
            [120.91, 30.17],
            [121.05, 30.05],
            [121.18, 29.92],
            [121.28, 29.80],
            [121.37, 29.67],
            [121.48, 29.55],
            [121.58, 29.42],
            [121.70, 29.29],
            [121.82, 29.16],
            [121.91, 29.03],
            [121.98, 28.90],
            [121.99, 28.76],
            [121.91, 28.63],
        ],
        dtype=np.float32,
    )
    hangzhou_bay = np.asarray(
        [
            [120.42, 30.66],
            [120.67, 30.58],
            [120.95, 30.50],
            [121.22, 30.38],
            [121.49, 30.24],
            [121.75, 30.10],
            [122.02, 29.96],
            [122.24, 29.86],
        ],
        dtype=np.float32,
    )
    zhoushan_1 = np.asarray(
        [
            [121.84, 30.05],
            [121.98, 30.00],
            [122.15, 30.01],
            [122.25, 29.94],
            [122.12, 29.87],
            [121.94, 29.88],
            [121.82, 29.95],
            [121.84, 30.05],
        ],
        dtype=np.float32,
    )
    zhoushan_2 = np.asarray(
        [
            [122.05, 29.78],
            [122.22, 29.73],
            [122.36, 29.65],
            [122.26, 29.56],
            [122.08, 29.59],
            [121.98, 29.68],
            [122.05, 29.78],
        ],
        dtype=np.float32,
    )
    return [mainland, hangzhou_bay, zhoushan_1, zhoushan_2]


def resolve_background_sources(args: argparse.Namespace, out_dir: Path) -> tuple[Path | None, Path | None, Path | None, str]:
    if args.land_shp and Path(args.land_shp).exists():
        land = Path(args.land_shp)
        lakes = Path(args.lakes_shp) if args.lakes_shp and Path(args.lakes_shp).exists() else None
        rivers = Path(args.rivers_shp) if args.rivers_shp and Path(args.rivers_shp).exists() else None
        return land, lakes, rivers, "user_supplied"
    if str(args.coastline_source) in {"auto", "natural_earth"}:
        try:
            land, lakes, rivers = ensure_natural_earth(out_dir)
            return land, lakes, rivers, "natural_earth_10m"
        except Exception:
            if str(args.coastline_source) == "natural_earth":
                raise
    return None, None, None, "built_in_fallback"


def draw_map_background(
    ax,
    bounds: tuple[float, float, float, float],
    land_shp: Path | None,
    lakes_shp: Path | None,
    rivers_shp: Path | None,
    use_builtin: bool,
) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    ax.set_facecolor(OCEAN_COLOR)
    if land_shp is not None and land_shp.exists():
        add_shapefile_polygons(
            ax,
            land_shp,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            facecolor=LAND_COLOR,
            edgecolor=LAND_EDGE_COLOR,
            linewidth=0.35,
            zorder=1,
        )
        if lakes_shp is not None and lakes_shp.exists():
            add_shapefile_polygons(
                ax,
                lakes_shp,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                facecolor=OCEAN_COLOR,
                edgecolor=LAND_EDGE_COLOR,
                linewidth=0.25,
                zorder=2,
            )
        if rivers_shp is not None and rivers_shp.exists():
            add_shapefile_lines(
                ax,
                rivers_shp,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                color=RIVER_COLOR,
                linewidth=0.5,
                zorder=3,
            )
        return

    if use_builtin:
        for line in built_in_ningbo_coastlines():
            arr = np.asarray(line, dtype=np.float32)
            ax.fill(
                arr[:, 0],
                arr[:, 1],
                facecolor=LAND_COLOR,
                edgecolor=LAND_EDGE_COLOR,
                linewidth=0.6,
                zorder=1,
            )


def plot_map(
    stations: pd.DataFrame,
    targets: pd.DataFrame | None,
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

    lon_min, lon_max, lat_min, lat_max = bounds
    if int(args.match_pv_field_style):
        side = max(float(args.width), float(args.height))
        fig, ax = plt.subplots(figsize=(side, side))
    else:
        fig, ax = plt.subplots(figsize=(float(args.width), float(args.height)))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(OCEAN_COLOR)
    draw_map_background(
        ax,
        bounds,
        land_shp,
        lakes_shp,
        rivers_shp,
        use_builtin=(background_source == "built_in_fallback"),
    )

    ax.scatter(
        stations["lon"],
        stations["lat"],
        s=float(args.station_size),
        c=str(args.station_color),
        alpha=float(args.station_alpha),
        linewidths=0.0,
        zorder=5,
    )

    if int(args.draw_targets) and targets is not None and not targets.empty:
        ax.scatter(
            targets["lon"],
            targets["lat"],
            s=float(args.target_size),
            c=str(args.target_color),
            alpha=0.9,
            marker="x",
            linewidths=1.2,
            zorder=6,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    if int(args.match_pv_field_style):
        # PV-field PNGs are rendered from square 64x64 grids, so match that
        # visual convention instead of preserving geographic aspect ratio.
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    png_path = out_base.with_suffix(".png")
    svg_path = out_base.with_suffix(".svg")
    if int(args.match_pv_field_style):
        fig.savefig(png_path, dpi=int(args.dpi), transparent=True, pad_inches=0)
        fig.savefig(svg_path, transparent=True, pad_inches=0)
    else:
        fig.savefig(png_path, dpi=int(args.dpi), transparent=True, bbox_inches="tight", pad_inches=0)
        fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    station_csv = resolve_station_csv(args)
    target_csv = resolve_target_csv(args)

    stations = read_station_csv(station_csv)
    targets = read_station_csv(target_csv) if target_csv.exists() else pd.DataFrame()

    bounds = pv_field_like_bounds(stations) if int(args.match_pv_field_style) else map_bounds(stations, float(args.pad_frac))
    land_shp, lakes_shp, rivers_shp, coast_source = resolve_background_sources(args, out_dir)
    out_base = out_dir / str(args.out_name)
    plot_map(stations, targets, bounds, land_shp, lakes_shp, rivers_shp, coast_source, out_base, args)
    write_json(
        out_dir / f"{args.out_name}_meta.json",
        {
            "purpose": "Transparent framework asset: base map plus Stage 3 remaining PV-field source stations after excluding 50 targets.",
            "station_csv": str(station_csv),
            "target_csv": str(target_csv) if target_csv.exists() else None,
            "remaining_station_count": int(len(stations)),
            "target_station_count": int(len(targets)) if not targets.empty else None,
            "bounds": {
                "lon_min": float(bounds[0]),
                "lon_max": float(bounds[1]),
                "lat_min": float(bounds[2]),
                "lat_max": float(bounds[3]),
            },
            "coastline_source": coast_source,
            "match_pv_field_style": bool(int(args.match_pv_field_style)),
            "land_shp": str(land_shp) if land_shp is not None else None,
            "lakes_shp": str(lakes_shp) if lakes_shp is not None else None,
            "rivers_shp": str(rivers_shp) if rivers_shp is not None else None,
            "outputs": {
                "png": str(out_base.with_suffix(".png")),
                "svg": str(out_base.with_suffix(".svg")),
            },
        },
    )
    print("==== Stage 3 remaining-station coastline map complete ====")
    print(f"stations: {len(stations)} from {station_csv}")
    print(f"coastline source: {coast_source}")
    print(f"png: {out_base.with_suffix('.png')}")
    print(f"svg: {out_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
