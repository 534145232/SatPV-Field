#!/usr/bin/env python3
"""Manuscript basemaps shared by the paper-support figures.

The default uses the same detailed contextily tile workflow as the Ramp
spatial maps, then recolors the tiles to the manuscript land/sea palette.  It
supplies display geography only and never changes station coordinates, field
support masks, or scientific values.  Natural Earth remains an offline
fallback and the older Ningbo polygon mode remains available explicitly.
"""

from __future__ import annotations

import inspect
import struct
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOUNDARY = SCRIPT_DIR / "data" / "boundaries" / "ningbo_land_boundary.geojson"
DEFAULT_NATURAL_EARTH_DIR = SCRIPT_DIR / "data" / "natural_earth"

# Match the three Ramp manuscript maps exactly.
OCEAN_COLOR = "#EAF3F7"
LAND_COLOR = "#F3F1EA"
COAST_COLOR = "#667781"
RIVER_COLOR = "#C5D9E4"

# One online tile download is reused by all independently exported panels.
_WEB_TILE_CACHE: dict[tuple[float, float, float, float], tuple[np.ndarray, tuple[float, ...], str]] = {}
_SHAPE_PART_CACHE: dict[tuple[str, tuple[float, float, float, float]], list[np.ndarray]] = {}


def _natural_earth_paths(root: Path) -> tuple[Path, Path | None, Path | None]:
    root = Path(root)
    land = root / "ne_10m_land" / "ne_10m_land.shp"
    lakes = root / "ne_10m_lakes" / "ne_10m_lakes.shp"
    rivers = (
        root
        / "ne_10m_rivers_lake_centerlines"
        / "ne_10m_rivers_lake_centerlines.shp"
    )
    if not land.is_file():
        raise FileNotFoundError(
            f"Natural Earth 10m land file not found: {land}\n"
            "Keep paper_support/data/natural_earth with the script."
        )
    return land, lakes if lakes.is_file() else None, rivers if rivers.is_file() else None


def _read_shapefile_parts(
    path: Path,
    bounds: tuple[float, float, float, float],
) -> list[np.ndarray]:
    key = (str(Path(path).resolve()), tuple(round(value, 8) for value in bounds))
    cached = _SHAPE_PART_CACHE.get(key)
    if cached is not None:
        return cached

    lon_min, lon_max, lat_min, lat_max = bounds
    parts_out: list[np.ndarray] = []
    with Path(path).open("rb") as handle:
        handle.seek(100)
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            _, length_words = struct.unpack(">2i", header)
            payload = handle.read(length_words * 2)
            if len(payload) < 44:
                continue
            shape_type = struct.unpack_from("<i", payload, 0)[0]
            if shape_type not in {3, 5, 13, 15, 23, 25}:
                continue
            xmin, ymin, xmax, ymax = struct.unpack_from("<4d", payload, 4)
            if xmax < lon_min or xmin > lon_max or ymax < lat_min or ymin > lat_max:
                continue
            n_parts, n_points = struct.unpack_from("<2i", payload, 36)
            parts_offset = 44
            points_offset = parts_offset + 4 * n_parts
            if (
                n_parts <= 0
                or n_points <= 0
                or len(payload) < points_offset + 16 * n_points
            ):
                continue
            starts = list(
                struct.unpack_from(f"<{n_parts}i", payload, parts_offset)
            ) + [n_points]
            points = np.asarray(
                struct.unpack_from(f"<{2 * n_points}d", payload, points_offset),
                dtype=float,
            ).reshape(-1, 2)
            for start, stop in zip(starts[:-1], starts[1:]):
                if stop - start >= 2:
                    parts_out.append(points[start:stop])
    _SHAPE_PART_CACHE[key] = parts_out
    return parts_out


def _add_polygon_shapefile(axis, path: Path | None, bounds, face, edge, zorder) -> bool:
    if path is None:
        return False
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    polygons = [
        Polygon(part, closed=True)
        for part in _read_shapefile_parts(path, bounds)
        if len(part) >= 3
    ]
    if not polygons:
        return False
    axis.add_collection(
        PatchCollection(
            polygons,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.28,
            zorder=zorder,
        )
    )
    return True


def _add_line_shapefile(axis, path: Path | None, bounds, color, zorder) -> bool:
    if path is None:
        return False
    from matplotlib.collections import LineCollection

    lines = _read_shapefile_parts(path, bounds)
    if not lines:
        return False
    axis.add_collection(
        LineCollection(lines, colors=color, linewidths=0.32, zorder=zorder)
    )
    return True


def _add_natural_earth(axis, root: Path) -> str:
    bounds = (
        float(axis.get_xlim()[0]),
        float(axis.get_xlim()[1]),
        float(axis.get_ylim()[0]),
        float(axis.get_ylim()[1]),
    )
    land, lakes, rivers = _natural_earth_paths(root)
    axis.set_facecolor(OCEAN_COLOR)
    if not _add_polygon_shapefile(axis, land, bounds, LAND_COLOR, COAST_COLOR, 1):
        raise RuntimeError(f"No Natural Earth land polygon intersects {bounds}")
    _add_polygon_shapefile(axis, lakes, bounds, OCEAN_COLOR, COAST_COLOR, 2)
    _add_line_shapefile(axis, rivers, bounds, RIVER_COLOR, 2)
    return "Natural Earth 10m land/lakes/rivers (Ramp-aligned)"


def load_boundary(path: Path, boundary_crs: str = "EPSG:4326") -> tuple[Any, Any]:
    """Load and merge a polygon boundary in longitude/latitude coordinates."""

    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The manuscript basemap requires GeoPandas. Install it with the same "
            "interpreter:\n  python -m pip install geopandas shapely pyogrio"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ningbo land boundary not found: {path}\n"
            "Keep paper_support/data/boundaries/ningbo_land_boundary.geojson "
            "with the script, or pass --boundary explicitly."
        )
    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError(f"Boundary contains no features: {path}")
    if frame.crs is None:
        frame = frame.set_crs(boundary_crs)
    frame = frame.to_crs("EPSG:4326")
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise ValueError(f"Boundary contains no valid geometry: {path}")
    merged = frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.geometry.unary_union
    if not merged.is_valid:
        merged = merged.buffer(0)
    region = gpd.GeoDataFrame(
        {"name": ["Ningbo land area"]},
        geometry=[merged],
        crs="EPSG:4326",
    )
    return region, merged


def _geometry_clip_patch(axis, geometry):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    polygons = (
        [geometry]
        if geometry.geom_type == "Polygon"
        else [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]
    )
    paths = []
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices = np.asarray(ring.coords, dtype=float)
            if len(vertices) < 3:
                continue
            codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            paths.append(MplPath(vertices, codes))
    if not paths:
        raise ValueError("Cannot construct a clip path from the boundary geometry")
    compound = paths[0] if len(paths) == 1 else MplPath.make_compound_path(*paths)
    patch = PathPatch(
        compound,
        transform=axis.transData,
        facecolor="none",
        edgecolor="none",
        linewidth=0.0,
    )
    axis.add_patch(patch)
    return patch


def _recolor_contextily_image(image: np.ndarray) -> np.ndarray:
    """Apply the same muted land/sea palette used by the Ramp spatial maps."""

    from matplotlib.colors import to_rgb

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] not in (3, 4):
        return source
    scale = 255.0 if source.dtype.kind in "ui" or np.nanmax(source[..., :3]) > 1.5 else 1.0
    rgba = source.astype(float) / scale
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    saturation = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    land = np.asarray(to_rgb(LAND_COLOR), dtype=float)
    ocean = np.asarray(to_rgb(OCEAN_COLOR), dtype=float)
    line = np.asarray(to_rgb(COAST_COLOR), dtype=float)
    water_mask = (
        ((luminance >= 0.62) & (luminance < 0.885) & (saturation < 0.20))
        | ((rgb[..., 2] - rgb[..., 0] > 0.035) & (luminance >= 0.62))
    )
    line_mask = luminance < 0.62
    styled = np.empty_like(rgb)
    styled[...] = land
    styled[water_mask] = ocean
    land_detail = np.clip((luminance - 0.937) * 0.18, -0.018, 0.018)
    water_detail = np.clip((luminance - 0.816) * 0.12, -0.014, 0.014)
    styled += land_detail[..., None]
    styled[water_mask] = ocean + water_detail[water_mask][:, None]
    coast_mask = np.zeros_like(water_mask, dtype=bool)
    coast_mask[1:, :] |= water_mask[1:, :] != water_mask[:-1, :]
    coast_mask[:-1, :] |= water_mask[:-1, :] != water_mask[1:, :]
    coast_mask[:, 1:] |= water_mask[:, 1:] != water_mask[:, :-1]
    coast_mask[:, :-1] |= water_mask[:, :-1] != water_mask[:, 1:]
    styled[coast_mask | line_mask] = line
    styled = np.clip(styled, 0.0, 1.0)
    if source.shape[-1] == 4:
        styled = np.concatenate([styled, rgba[..., 3:4]], axis=-1)
    if scale == 255.0:
        styled = np.rint(styled * 255.0).astype(source.dtype)
    return styled


def _add_contextily(axis) -> str:
    try:
        import contextily as ctx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The online manuscript basemap requires contextily. Install it with:\n"
            "  python -m pip install contextily"
        ) from exc

    # CartoDB.Positron was removed because the provider now requires an API
    # key and can leave batch jobs waiting on repeated unauthorized requests.
    providers = (
        ("Esri.WorldGrayCanvas", ctx.providers.Esri.WorldGrayCanvas),
        ("OpenStreetMap.Mapnik", ctx.providers.OpenStreetMap.Mapnik),
    )
    xlim = tuple(float(value) for value in axis.get_xlim())
    ylim = tuple(float(value) for value in axis.get_ylim())
    cache_key = tuple(round(value, 8) for value in (*xlim, *ylim))

    cached = _WEB_TILE_CACHE.get(cache_key)
    if cached is None:
        west, east = xlim
        south, north = ylim
        errors: list[str] = []
        for name, provider in providers:
            try:
                download_options = {
                    "source": provider,
                    "ll": True,
                    "wait": 0,
                    "max_retries": 1,
                    "n_connections": 1,
                    "use_cache": True,
                }
                if "timeout" in inspect.signature(ctx.bounds2img).parameters:
                    download_options["timeout"] = (5, 15)
                image, extent = ctx.bounds2img(
                    west,
                    south,
                    east,
                    north,
                    **download_options,
                )
                image, extent = ctx.warp_tiles(
                    image,
                    extent,
                    t_crs="EPSG:4326",
                )
                image = _recolor_contextily_image(image)
                cached = (image, tuple(float(value) for value in extent), name)
                _WEB_TILE_CACHE[cache_key] = cached
                break
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if cached is None:
            raise RuntimeError(
                "All web basemap providers failed. Use --background natural-earth "
                "for an offline result.\n  - " + "\n  - ".join(errors)
            )

    image, extent, name = cached
    axis.imshow(
        image,
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    return name


def draw_basemap(
    axis,
    region,
    geometry,
    background: str,
    natural_earth_dir: Path = DEFAULT_NATURAL_EARTH_DIR,
) -> str:
    """Draw Ramp-aligned geography without clipping scientific data layers."""

    axis.set_facecolor(OCEAN_COLOR)
    if background == "natural-earth":
        return _add_natural_earth(axis, natural_earth_dir)

    if background == "none":
        region.boundary.plot(
            ax=axis,
            color=COAST_COLOR,
            linewidth=0.65,
            zorder=6,
        )
        return "local Ningbo land boundary (outline only)"

    if background == "contextily":
        provider = _add_contextily(axis)
        source = f"contextily:{provider}, Ramp-aligned palette, no administrative clipping"
    elif background == "vector":
        region.plot(
            ax=axis,
            facecolor=LAND_COLOR,
            edgecolor="none",
            zorder=1,
        )
        source = "local Ningbo land-only GeoJSON"
    else:
        raise ValueError(f"Unknown background mode: {background}")

    if region is not None:
        region.boundary.plot(
            ax=axis,
            color=COAST_COLOR,
            linewidth=0.65,
            zorder=6,
        )
    return source


def draw_boundary_on_top(axis, region) -> None:
    if region is None:
        return
    region.boundary.plot(
        ax=axis,
        color="#56636D",
        linewidth=0.62,
        alpha=0.92,
        zorder=8,
    )
