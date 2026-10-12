#!/usr/bin/env python3
"""Draw the Stage-3 station split with the current manuscript basemap.

This compatibility entry point delegates data loading, target styling,
domain-box drawing, legend construction, axis formatting, and export settings
to the original framework-map script. It replaces the basemap provider and
slightly strengthens the source-station marks for readability on that basemap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
FRAMEWORK_DIR = PIPELINE_DIR / "Framework_draw"
for module_dir in (FRAMEWORK_DIR, SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import generate_stage3_all_target_station_coastline_map as original  # noqa: E402
from paper_support_basemap import (  # noqa: E402
    DEFAULT_NATURAL_EARTH_DIR,
    draw_basemap,
)


BASEMAP_MODE = os.environ.get("BIGPV_MAP_BACKGROUND", "contextily").strip().lower()
if BASEMAP_MODE not in {"contextily", "natural-earth"}:
    raise ValueError(
        "BIGPV_MAP_BACKGROUND must be 'contextily' or 'natural-earth', "
        f"got {BASEMAP_MODE!r}"
    )


def draw_latest_background(
    axis,
    bounds,
    _land_shp,
    _lakes_shp,
    _rivers_shp,
    use_builtin: bool = False,
) -> None:
    """Replace only the original basemap artist; preserve all other styling."""
    del use_builtin
    lon_min, lon_max, lat_min, lat_max = bounds
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    draw_basemap(
        axis,
        None,
        None,
        BASEMAP_MODE,
        DEFAULT_NATURAL_EARTH_DIR,
    )


def resolve_latest_background(_args, _out_dir):
    """Bypass the legacy coastline resolver without changing plot arguments."""
    source = (
        "latest manuscript basemap: contextily/Esri with OSM fallback"
        if BASEMAP_MODE == "contextily"
        else "latest manuscript basemap: Natural Earth 10m offline fallback"
    )
    return None, None, None, source


def inject_source_station_visibility_defaults() -> None:
    """Improve source-station contrast while preserving explicit CLI choices."""
    defaults = {
        "--all_station_color": "#52758c",
        "--all_station_alpha": "0.88",
        "--all_station_size": "8.0",
    }
    supplied = sys.argv[1:]
    for option, value in defaults.items():
        if not any(arg == option or arg.startswith(f"{option}=") for arg in supplied):
            sys.argv.extend([option, value])


def main() -> None:
    original.draw_map_background = draw_latest_background
    original.resolve_background_sources = resolve_latest_background
    inject_source_station_visibility_defaults()
    original.main()


if __name__ == "__main__":
    main()
