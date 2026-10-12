#!/usr/bin/env python3
"""Generate clean Stage 4 station temporal curve assets for framework figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE4_DIR = PIPELINE_DIR / "stage_4"
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_paths import TARGET_EXCLUDED_RUN_DIR, TARGET_ISSUE_TIME_INDEX, TARGET_SELECTION_DIR, TRAIN_PV_DIR, VAL_PV_DIR  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage4_station_temporal_curve_assets"
DEFAULT_SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE4_BASELINE = _load_local_module(
    "framework_stage4_temporal_baseline",
    STAGE4_DIR / "03_train_per_station_temporal_baseline.py",
)


HIST_WEATHER_VARS = ["radiation", "temperature", "tcc", "speed10"]
FUTURE_NWP_VARS = ["radiation", "temperature", "tcc", "speed10"]

COLOR_THEME = {
    "hist_pv": "#8b1e2d",
    "hist_weather": {
        "radiation": "#c67a00",
        "temperature": "#d95f02",
        "tcc": "#3a7ca5",
        "speed10": "#2a9d8f",
    },
    "future_nwp": {
        "radiation": "#b36a00",
        "temperature": "#c94c1a",
        "tcc": "#356f9a",
        "speed10": "#208b7a",
    },
    "time_encoding": {
        "tod_sin": "#7c3aed",
        "tod_cos": "#4f46e5",
        "solar_elevation": "#d4a017",
    },
    "clearsky": "#f59e0b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--time_index_csv", type=Path, default=TARGET_ISSUE_TIME_INDEX)
    parser.add_argument("--train_pv_dir", type=Path, default=TRAIN_PV_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=VAL_PV_DIR)
    parser.add_argument("--sampled_issue_times_csv", type=Path, default=DEFAULT_SAMPLED_ISSUE_TIMES_CSV)
    parser.add_argument("--issue_time_column", type=str, default="issue_time")
    parser.add_argument("--sample_rank", type=int, default=1)
    parser.add_argument("--station_sn", type=str, default="")
    parser.add_argument("--station_rank", type=int, default=1, help="Representative-station rank when --station_sn is not set.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--capacity_scale", type=float, default=1000.0)
    parser.add_argument("--past_frames", type=int, default=12)
    parser.add_argument("--future_frames", type=int, default=8)
    parser.add_argument("--pv_health_max", type=float, default=1.35)
    parser.add_argument("--pv_health_active_threshold", type=float, default=0.05)
    parser.add_argument("--pv_health_flatline_window", type=int, default=8)
    parser.add_argument("--pv_health_flatline_range", type=float, default=1e-5)
    parser.add_argument("--health_filter", type=int, default=1)
    parser.add_argument("--width", type=float, default=4.6)
    parser.add_argument("--height", type=float, default=1.45)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--line_width_primary", type=float, default=4.4)
    parser.add_argument("--line_width_secondary", type=float, default=3.4)
    parser.add_argument("--axis_color", type=str, default="#0f172a")
    parser.add_argument("--axis_width", type=float, default=1.8)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_target_csv(selection_dir: Path, target_run_dir: Path) -> Path:
    candidates = [
        Path(selection_dir) / "target_stations.csv",
        Path(target_run_dir) / "target_stations.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"target_stations.csv not found. Checked: {[str(p) for p in candidates]}")


def load_requested_issue_row(sampled_issue_times_csv: Path, issue_time_column: str, sample_rank: int, time_index_csv: Path) -> pd.Series:
    req = pd.read_csv(sampled_issue_times_csv)
    if issue_time_column not in req.columns:
        raise KeyError(
            f"Column {issue_time_column!r} not found in {sampled_issue_times_csv}. Available columns: {list(req.columns)}"
        )
    rank = max(1, min(int(sample_rank), len(req)))
    ts = pd.Timestamp(req.iloc[rank - 1][issue_time_column])

    time_df = pd.read_csv(time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    match = time_df[time_df["issue_time"] == ts].copy()
    if match.empty:
        raise ValueError(f"Could not find issue_time={ts} in {time_index_csv}")
    return match.iloc[0].copy()


def choose_representative_stations(targets: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or targets.empty:
        return targets.iloc[0:0].copy()
    coords = targets[["lon", "lat"]].to_numpy(np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    scale = np.where((hi - lo) > 1.0e-9, hi - lo, 1.0)
    norm = (coords - lo) / scale
    centroid = norm.mean(axis=0)
    chosen = [int(np.argmin(np.sum((norm - centroid[None]) ** 2, axis=1)))]
    while len(chosen) < min(int(count), len(targets)):
        remaining = [i for i in range(len(targets)) if i not in chosen]
        d2 = []
        for idx in remaining:
            dist = min(float(np.sum((norm[idx] - norm[c]) ** 2)) for c in chosen)
            d2.append((dist, idx))
        d2.sort(reverse=True)
        chosen.append(int(d2[0][1]))
    return targets.iloc[chosen].reset_index(drop=True)


def station_has_window(df: pd.DataFrame, issue_row: pd.Series) -> bool:
    past = STAGE4_BASELINE._parse_time_list(issue_row["past_times"])
    future = STAGE4_BASELINE._parse_time_list(issue_row["future_times"])
    return all(pd.Timestamp(t) in df.index for t in (past + future))


def pick_station(targets: pd.DataFrame, issue_row: pd.Series, args: argparse.Namespace) -> tuple[pd.Series, pd.DataFrame, dict]:
    if str(args.station_sn).strip():
        row = targets[targets["station_sn"].astype(str) == str(args.station_sn).strip()]
        if row.empty:
            raise ValueError(f"station_sn {args.station_sn!r} not found in target stations")
        station_row = row.iloc[0]
        station_df, meta = STAGE4_BASELINE.load_station_table(str(station_row["station_sn"]), args)
        if not station_has_window(station_df, issue_row):
            raise ValueError(f"Station {station_row['station_sn']} does not have a complete window for issue_time={issue_row['issue_time']}")
        return station_row, station_df, meta

    reps = choose_representative_stations(targets, max(5, int(args.station_rank)))
    ordered = list(reps["station_sn"].astype(str).tolist()) + [
        sn for sn in targets["station_sn"].astype(str).tolist() if sn not in set(reps["station_sn"].astype(str).tolist())
    ]
    for sn in ordered:
        station_row = targets[targets["station_sn"].astype(str) == sn].iloc[0]
        station_df, meta = STAGE4_BASELINE.load_station_table(str(sn), args)
        if station_has_window(station_df, issue_row):
            return station_row, station_df, meta
    raise RuntimeError("No target station with a complete past/future window was found for the selected issue time.")


def normalize_for_display(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanmin(arr[valid]))
    hi = float(np.nanmax(arr[valid]))
    if hi - lo < 1.0e-6:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def plot_curve_group(
    series_list: list[np.ndarray],
    colors: list[str],
    out_path: Path,
    width: float,
    height: float,
    dpi: int,
    line_width_primary: float,
    line_width_secondary: float,
    axis_color: str,
    axis_width: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(float(width), float(height)), dpi=int(dpi))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    max_len = max(len(np.asarray(s)) for s in series_list) if series_list else 1
    x = np.arange(max_len, dtype=np.float32)
    for idx, (series, color) in enumerate(zip(series_list, colors)):
        y = normalize_for_display(np.asarray(series, dtype=np.float32))
        xx = np.arange(len(y), dtype=np.float32)
        lw = float(line_width_primary if idx == 0 and len(series_list) == 1 else line_width_secondary)
        ax.plot(xx, y, color=str(color), linewidth=lw, solid_capstyle="round", solid_joinstyle="round")

    ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    ax.set_ylim(-0.04, 1.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(str(axis_color))
    ax.spines["bottom"].set_color(str(axis_color))
    ax.spines["left"].set_linewidth(float(axis_width))
    ax.spines["bottom"].set_linewidth(float(axis_width))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.08, top=0.995)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, format="svg", dpi=int(dpi), bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    target_csv = resolve_target_csv(Path(args.selection_dir), Path(args.target_run_dir))
    targets = pd.read_csv(target_csv, dtype={"station_sn": str})
    issue_row = load_requested_issue_row(
        Path(args.sampled_issue_times_csv),
        str(args.issue_time_column),
        int(args.sample_rank),
        Path(args.time_index_csv),
    )
    issue_time = pd.Timestamp(issue_row["issue_time"])
    station_row, station_df, station_meta = pick_station(targets, issue_row, args)

    past_times = STAGE4_BASELINE._parse_time_list(issue_row["past_times"])
    future_times = STAGE4_BASELINE._parse_time_list(issue_row["future_times"])
    hist_raw = station_df.loc[past_times, ["pv_norm", *STAGE4_BASELINE.FUTURE_NWP_NAMES]].to_numpy(np.float32)
    fut_raw = station_df.loc[future_times, STAGE4_BASELINE.FUTURE_NWP_NAMES].to_numpy(np.float32)
    future_time_solar = np.stack(
        [STAGE4_BASELINE._time_solar_point(ts, float(station_row["lon"]), float(station_row["lat"])) for ts in future_times],
        axis=0,
    ).astype(np.float32)

    station_dir = ensure_dir(out_dir / f"sample_{issue_time:%Y%m%d_%H%M%S}_{station_row['station_sn']}")

    hist_pv_path = station_dir / "hist_pv_curve.svg"
    plot_curve_group(
        [hist_raw[:, 0]],
        [COLOR_THEME["hist_pv"]],
        hist_pv_path,
        args.width,
        args.height,
        args.dpi,
        args.line_width_primary,
        args.line_width_secondary,
        args.axis_color,
        args.axis_width,
    )

    hist_weather_series = [hist_raw[:, 1 + STAGE4_BASELINE.FUTURE_NWP_NAMES.index(name)] for name in HIST_WEATHER_VARS]
    hist_weather_colors = [COLOR_THEME["hist_weather"][name] for name in HIST_WEATHER_VARS]
    hist_weather_path = station_dir / "hist_weather_curves.svg"
    plot_curve_group(
        hist_weather_series,
        hist_weather_colors,
        hist_weather_path,
        args.width,
        args.height,
        args.dpi,
        args.line_width_primary,
        args.line_width_secondary,
        args.axis_color,
        args.axis_width,
    )

    future_nwp_series = [fut_raw[:, STAGE4_BASELINE.FUTURE_NWP_NAMES.index(name)] for name in FUTURE_NWP_VARS]
    future_nwp_colors = [COLOR_THEME["future_nwp"][name] for name in FUTURE_NWP_VARS]
    future_nwp_path = station_dir / "future_nwp_curves.svg"
    plot_curve_group(
        future_nwp_series,
        future_nwp_colors,
        future_nwp_path,
        args.width,
        args.height,
        args.dpi,
        args.line_width_primary,
        args.line_width_secondary,
        args.axis_color,
        args.axis_width,
    )

    time_encoding_series = [
        future_time_solar[:, 0],
        future_time_solar[:, 1],
        future_time_solar[:, 2],
    ]
    time_encoding_colors = [
        COLOR_THEME["time_encoding"]["tod_sin"],
        COLOR_THEME["time_encoding"]["tod_cos"],
        COLOR_THEME["time_encoding"]["solar_elevation"],
    ]
    time_encoding_path = station_dir / "time_encoding_curves.svg"
    plot_curve_group(
        time_encoding_series,
        time_encoding_colors,
        time_encoding_path,
        args.width,
        args.height,
        args.dpi,
        args.line_width_primary,
        args.line_width_secondary,
        args.axis_color,
        args.axis_width,
    )

    clearsky_series = [fut_raw[:, STAGE4_BASELINE.FUTURE_NWP_NAMES.index("clearsky_ghi")]]
    clearsky_path = station_dir / "clearsky_curve.svg"
    plot_curve_group(
        clearsky_series,
        [COLOR_THEME["clearsky"]],
        clearsky_path,
        args.width,
        args.height,
        args.dpi,
        args.line_width_primary,
        args.line_width_secondary,
        args.axis_color,
        args.axis_width,
    )

    meta = {
        "issue_time": str(issue_time),
        "sample_rank": int(args.sample_rank),
        "station_sn": str(station_row["station_sn"]),
        "station_lon": float(station_row["lon"]),
        "station_lat": float(station_row["lat"]),
        "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
        "issue_time_column": str(args.issue_time_column),
        "target_csv": str(target_csv),
        "station_csv_meta": station_meta,
        "past_times": [str(pd.Timestamp(ts)) for ts in past_times],
        "future_times": [str(pd.Timestamp(ts)) for ts in future_times],
        "outputs": {
            "hist_pv_curve": str(hist_pv_path),
            "hist_weather_curves": str(hist_weather_path),
            "future_nwp_curves": str(future_nwp_path),
            "time_encoding_curves": str(time_encoding_path),
            "clearsky_curve": str(clearsky_path),
        },
        "curve_groups": {
            "hist_weather_curves": HIST_WEATHER_VARS,
            "future_nwp_curves": FUTURE_NWP_VARS,
            "time_encoding_curves": ["tod_sin", "tod_cos", "solar_elevation"],
            "clearsky_curve": ["clearsky_ghi"],
        },
        "display_note": "All curves are drawn from real Stage 4 station inputs and min-max normalized independently within each line for clean framework display.",
    }
    write_json(station_dir / "curve_meta.json", meta)

    print("==== Stage 4 station temporal curve assets complete ====")
    print(f"station: {station_row['station_sn']}")
    print(f"issue: {issue_time}")
    print(f"output dir: {station_dir}")


if __name__ == "__main__":
    main()
