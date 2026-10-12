#!/usr/bin/env python3
"""Combine cached real Stage 4 station-input curves into two compact SVG assets."""

from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR / "stage4_station_temporal_curve_assets"
DEFAULT_OUT_DIR = SCRIPT_DIR / "stage4_combined_station_input_assets"

HIST_WEATHER_NAMES = ["Radiation", "Temperature", "Cloud cover", "Wind speed"]
FUTURE_NWP_NAMES = ["Radiation", "Temperature", "Cloud cover", "Wind speed"]
TIME_SOLAR_NAMES = ["Time sine", "Time cosine", "Solar elevation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--sample_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def find_sample_dir(source_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        sample_dir = Path(explicit)
        if not sample_dir.exists():
            raise FileNotFoundError(sample_dir)
        return sample_dir
    candidates = sorted(
        path for path in Path(source_root).glob("sample_*") if (path / "curve_meta.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No cached real-data sample found below {source_root}")
    return candidates[0]


def parse_path_points(path_data: str) -> np.ndarray:
    numbers = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", path_data)]
    if len(numbers) < 4 or len(numbers) % 2:
        raise ValueError(f"Unsupported SVG path: {path_data[:80]!r}")
    return np.asarray(numbers, dtype=np.float64).reshape(-1, 2)


def extract_real_curves(svg_path: Path) -> tuple[list[np.ndarray], list[str]]:
    if not svg_path.exists():
        raise FileNotFoundError(svg_path)
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    curves: list[np.ndarray] = []
    colors: list[str] = []
    for group in root.findall(".//svg:g", namespace):
        if not str(group.attrib.get("id", "")).startswith("line2d_"):
            continue
        path = group.find("svg:path", namespace)
        if path is None or "d" not in path.attrib:
            continue
        style = str(path.attrib.get("style", ""))
        color_match = re.search(r"stroke:\s*(#[0-9a-fA-F]{6})", style)
        if color_match is None:
            continue
        points = parse_path_points(path.attrib["d"])
        y_svg = points[:, 1]
        lo = float(np.min(y_svg))
        hi = float(np.max(y_svg))
        if hi - lo < 1.0e-9:
            y = np.full_like(y_svg, 0.5, dtype=np.float64)
        else:
            y = 1.0 - (y_svg - lo) / (hi - lo)
        curves.append(y.astype(np.float32))
        colors.append(color_match.group(1).lower())
    if not curves:
        raise ValueError(f"No Matplotlib line paths found in {svg_path}")
    return curves, colors


def svg_text(x: float, y: float, text: str, size: float, color: str, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size:.2f}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def curve_polyline(
    curve: np.ndarray,
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    stroke_width: float,
    markers: bool,
) -> str:
    values = np.asarray(curve, dtype=np.float64)
    denom = max(len(values) - 1, 1)
    points = []
    circles = []
    for index, value in enumerate(values):
        x = x0 + width * index / denom
        y = y0 + height * (1.0 - float(value))
        points.append(f"{x:.2f},{y:.2f}")
        if markers:
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.1" fill="{color}"/>')
    line = (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    return line + "".join(circles)


def axis_line(x0: float, y: float, width: float) -> str:
    return f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + width:.2f}" y2="{y:.2f}" stroke="#94a3b8" stroke-width="1.2"/>'


def svg_document(width: int, height: int, body: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<g id="station-input-asset">\n'
        + "\n".join(body)
        + '\n</g>\n</svg>\n'
    )


def draw_historical(sample_dir: Path, out_dir: Path, dpi: int) -> Path:
    pv_curves, pv_colors = extract_real_curves(sample_dir / "hist_pv_curve.svg")
    weather_curves, weather_colors = extract_real_curves(sample_dir / "hist_weather_curves.svg")
    width, height = 900, 400
    x0, plot_width = 155.0, 700.0
    body = [svg_text(width / 2, 34, "Historical Station Sequence", 22, "#7f1d1d", "middle", "700")]
    body += [svg_text(125, 112, "PV", 16, "#8b1e2d", "end", "700")]
    body += [curve_polyline(pv_curves[0], pv_colors[0], x0, 68, plot_width, 86, 4.2, True)]
    body += [axis_line(x0, 160, plot_width)]
    body += [svg_text(125, 242, "Weather", 16, "#475569", "end", "700")]
    for curve, color in zip(weather_curves[:4], weather_colors[:4]):
        body.append(curve_polyline(curve, color, x0, 188, plot_width, 102, 2.5, False))
    body += [axis_line(x0, 298, plot_width)]
    body += [svg_text(x0, 320, "t−12", 12, "#475569", "middle")]
    body += [svg_text(x0 + plot_width, 320, "t−1", 12, "#475569", "middle")]
    legend_x = [235, 390, 550, 710]
    for x, name, color in zip(legend_x, HIST_WEATHER_NAMES, weather_colors[:4]):
        body.append(f'<line x1="{x - 23}" y1="346" x2="{x + 1}" y2="346" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        body.append(svg_text(x + 8, 351, name, 11.5, "#475569"))
    body += [svg_text(width / 2, 384, "PV + historical weather  |  12 historical steps", 13, "#64748b", "middle")]
    svg_path = out_dir / "historical_station_sequence.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_document(width, height, body), encoding="utf-8")
    return svg_path


def draw_future(sample_dir: Path, out_dir: Path, dpi: int) -> Path:
    nwp_curves, nwp_colors = extract_real_curves(sample_dir / "future_nwp_curves.svg")
    time_curves, time_colors = extract_real_curves(sample_dir / "time_encoding_curves.svg")
    clear_curves, clear_colors = extract_real_curves(sample_dir / "clearsky_curve.svg")
    width, height = 900, 500
    x0, plot_width = 155.0, 700.0
    body = [svg_text(width / 2, 34, "Future Conditions", 22, "#1e3a8a", "middle", "700")]
    body += [svg_text(125, 125, "NWP", 16, "#1d4ed8", "end", "700")]
    for curve, color in zip(nwp_curves[:4], nwp_colors[:4]):
        body.append(curve_polyline(curve, color, x0, 80, plot_width, 92, 2.5, True))
    body += [axis_line(x0, 180, plot_width)]
    body += [svg_text(125, 263, "Time/Solar", 16, "#6d28d9", "end", "700")]
    for curve, color in zip(time_curves[:3], time_colors[:3]):
        body.append(curve_polyline(curve, color, x0, 218, plot_width, 88, 2.5, False))
    body += [axis_line(x0, 314, plot_width)]
    body += [svg_text(125, 382, "Clear-sky", 16, "#b45309", "end", "700")]
    body += [curve_polyline(clear_curves[0], clear_colors[0], x0, 344, plot_width, 66, 3.5, True)]
    body += [axis_line(x0, 418, plot_width)]
    steps = len(nwp_curves[0])
    for index in range(steps):
        x = x0 + plot_width * index / max(steps - 1, 1)
        body.append(svg_text(x, 442, f"h{index + 1}", 11.5, "#475569", "middle"))
    legend_x = [235, 390, 550, 710]
    for x, name, color in zip(legend_x, FUTURE_NWP_NAMES, nwp_colors[:4]):
        body.append(f'<line x1="{x - 23}" y1="58" x2="{x + 1}" y2="58" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        body.append(svg_text(x + 8, 63, name, 11.5, "#475569"))
    body += [svg_text(width / 2, 484, "Horizon-aligned NWP + time/solar + clear-sky GHI", 13, "#64748b", "middle")]
    svg_path = out_dir / "future_station_conditions.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_document(width, height, body), encoding="utf-8")
    return svg_path


def main() -> None:
    args = parse_args()
    sample_dir = find_sample_dir(Path(args.source_root), args.sample_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    historical_svg = draw_historical(sample_dir, out_dir, int(args.dpi))
    future_svg = draw_future(sample_dir, out_dir, int(args.dpi))

    source_meta = json.loads((sample_dir / "curve_meta.json").read_text(encoding="utf-8"))
    payload = {
        "source_sample_dir": str(sample_dir),
        "source_is_real_data": True,
        "issue_time": source_meta.get("issue_time"),
        "station_sn": source_meta.get("station_sn"),
        "station_lon": source_meta.get("station_lon"),
        "station_lat": source_meta.get("station_lat"),
        "historical_weather_features": HIST_WEATHER_NAMES,
        "future_nwp_features": FUTURE_NWP_NAMES,
        "future_time_solar_features": TIME_SOLAR_NAMES,
        "normalization_note": (
            "Curves are recovered from the cached SVG assets generated from the real Stage 4 station inputs. "
            "Each line retains the source asset's independent display normalization."
        ),
        "outputs": {
            "historical_svg": str(historical_svg),
            "future_svg": str(future_svg),
        },
    }
    (out_dir / "asset_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Historical sequence:", historical_svg)
    print("Future conditions:", future_svg)
    print("Source station:", payload["station_sn"])
    print("Issue time:", payload["issue_time"])


if __name__ == "__main__":
    main()
