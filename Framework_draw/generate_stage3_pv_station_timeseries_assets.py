#!/usr/bin/env python3
"""Generate transparent Stage 3 PV station multi-day sparkline assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import TARGET_EXCLUDED_RUN_DIR, TRAIN_PV_DIR, VAL_PV_DIR  # noqa: E402
from visualize_regional_split_frames import load_station_series  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_pv_station_timeseries_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station_csv", type=Path, default=TARGET_EXCLUDED_RUN_DIR / "target_excluded_station_split.csv")
    parser.add_argument("--train_pv_dir", type=Path, default=TRAIN_PV_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=VAL_PV_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--station_count", type=int, default=5)
    parser.add_argument("--candidate_batch_size", type=int, default=24)
    parser.add_argument("--window_points", type=int, default=280)
    parser.add_argument("--points_per_day", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--capacity_scale", type=float, default=1000.0)
    parser.add_argument("--line_color", type=str, default="#3468ff")
    parser.add_argument("--line_width", type=float, default=1.6)
    parser.add_argument("--glow_width", type=float, default=4.6)
    parser.add_argument("--glow_alpha", type=float, default=0.08)
    parser.add_argument("--panel_width", type=float, default=4.6)
    parser.add_argument("--panel_height_per_series", type=float, default=0.75)
    parser.add_argument("--single_width", type=float, default=4.0)
    parser.add_argument("--single_height", type=float, default=0.72)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--ymax", type=float, default=1.05)
    parser.add_argument("--draw_last_dot", type=int, default=1)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_station_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"station csv not found: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = {"station_sn", "lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)


def list_window_start_indices(index: pd.Index, window_points: int) -> list[int]:
    if len(index) < window_points:
        return []
    return list(range(0, len(index) - window_points + 1))


def choose_station_windows(
    station_df: pd.DataFrame,
    series: dict[str, pd.DataFrame],
    window_points: int,
    count: int,
    seed: int,
    ymax: float,
) -> list[dict]:
    eligible: list[dict] = []
    for _, row in station_df.iterrows():
        sn = str(row["station_sn"])
        sdf = series.get(sn)
        if sdf is None or sdf.empty:
            continue
        starts = list_window_start_indices(sdf.index, int(window_points))
        if not starts:
            continue
        arr = sdf["pv_norm"].to_numpy(np.float32)
        variability = []
        for start_idx in starts:
            values = arr[start_idx : start_idx + int(window_points)]
            value_range = float(np.nanmax(values) - np.nanmin(values))
            value_std = float(np.nanstd(values))
            value_peak = float(np.nanmax(values))
            score = 1.2 * value_range + 0.9 * value_std + 0.25 * min(value_peak, float(ymax))
            variability.append((start_idx, score))
        eligible.append(
            {
                "station_sn": sn,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "series": sdf,
                "starts": [int(v[0]) for v in variability],
                "scores": [float(v[1]) for v in variability],
                "best_score": float(max(v[1] for v in variability)),
            }
        )
    if not eligible:
        raise RuntimeError("No stations with enough PV points for the requested window length were found.")

    ranked = sorted(eligible, key=lambda item: item["best_score"], reverse=True)
    candidate_pool = ranked[: max(int(count) * 10, min(80, len(ranked)))]
    rng = np.random.default_rng(int(seed))
    pick_count = min(int(count), len(candidate_pool))
    pick_idx = np.sort(rng.choice(len(candidate_pool), size=pick_count, replace=False))
    chosen: list[dict] = []
    for order, idx in enumerate(pick_idx, start=1):
        item = candidate_pool[int(idx)]
        start_weights = np.asarray(item["scores"], dtype=np.float64)
        start_weights = np.clip(start_weights, 1e-6, None)
        start_weights = start_weights / start_weights.sum()
        start_choice = int(rng.choice(len(item["starts"]), p=start_weights))
        start_idx = int(item["starts"][start_choice])
        sdf = item["series"]
        window = sdf.iloc[start_idx : start_idx + int(window_points)]
        chosen.append(
            {
                "rank": int(order),
                "station_sn": str(item["station_sn"]),
                "lon": float(item["lon"]),
                "lat": float(item["lat"]),
                "start_time": str(pd.Timestamp(window.index[0])),
                "end_time": str(pd.Timestamp(window.index[-1])),
                "times": [pd.Timestamp(ts) for ts in window.index],
                "values": window["pv_norm"].to_numpy(np.float32),
            }
        )
    return chosen


def render_single_series(times: list[pd.Timestamp], values: np.ndarray, out_path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(times), dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(float(args.single_width), float(args.single_height)))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.plot(x, y, color=str(args.line_color), linewidth=float(args.glow_width), alpha=float(args.glow_alpha), solid_capstyle="round")
    ax.plot(x, y, color=str(args.line_color), linewidth=float(args.line_width), alpha=0.98, solid_capstyle="round")
    if int(args.draw_last_dot):
        ax.scatter([x[-1]], [y[-1]], s=10.0, color=str(args.line_color), alpha=0.98, zorder=3)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, float(args.ymax))
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(args.dpi), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_panel(series_values: list[np.ndarray], out_path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(series_values)
    fig_h = max(0.6, float(args.panel_height_per_series) * max(n, 1))
    fig, axes = plt.subplots(n, 1, figsize=(float(args.panel_width), fig_h), squeeze=False)
    fig.patch.set_alpha(0.0)
    for ax, values in zip(axes[:, 0], series_values):
        x = np.arange(len(values), dtype=np.float32)
        y = np.asarray(values, dtype=np.float32)
        ax.set_facecolor((1, 1, 1, 0))
        ax.plot(x, y, color=str(args.line_color), linewidth=float(args.glow_width), alpha=float(args.glow_alpha), solid_capstyle="round")
        ax.plot(x, y, color=str(args.line_color), linewidth=float(args.line_width), alpha=0.98, solid_capstyle="round")
        if int(args.draw_last_dot):
            ax.scatter([x[-1]], [y[-1]], s=10.0, color=str(args.line_color), alpha=0.98, zorder=3)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(0.0, float(args.ymax))
        ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, hspace=0.28)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=int(args.dpi), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    station_df = read_station_table(Path(args.station_csv))
    window_points = int(args.window_points)

    # Reuse the exact Stage 3 PV normalization/loading path.
    load_args = argparse.Namespace(
        train_pv_dir=Path(args.train_pv_dir),
        val_pv_dir=Path(args.val_pv_dir),
        capacity_scale=float(args.capacity_scale),
    )
    rng = np.random.default_rng(int(args.seed))
    shuffled_df = station_df.iloc[rng.permutation(len(station_df))].reset_index(drop=True)
    batch_size = max(1, int(args.candidate_batch_size))
    loaded_station_parts: list[pd.DataFrame] = []
    status_parts: list[pd.DataFrame] = []
    series: dict[str, pd.DataFrame] = {}
    chosen: list[dict] | None = None
    last_error: Exception | None = None

    for batch_start in range(0, len(shuffled_df), batch_size):
        batch_id = batch_start // batch_size + 1
        batch_df = shuffled_df.iloc[batch_start : batch_start + batch_size].reset_index(drop=True)
        if batch_df.empty:
            continue
        print(
            f"[pv_timeseries] loading candidate batch {batch_id} "
            f"({batch_start + 1}-{batch_start + len(batch_df)} / {len(shuffled_df)})",
            flush=True,
        )
        batch_series, batch_status = load_station_series(batch_df, load_args)
        loaded_station_parts.append(batch_df)
        status_parts.append(batch_status.assign(candidate_batch=int(batch_id)))
        series.update(batch_series)
        loaded_df = pd.concat(loaded_station_parts, ignore_index=True)
        try:
            chosen = choose_station_windows(
                station_df=loaded_df,
                series=series,
                window_points=window_points,
                count=int(args.station_count),
                seed=int(args.seed),
                ymax=float(args.ymax),
            )
            print(
                f"[pv_timeseries] found {len(chosen)} usable station windows after "
                f"{len(loaded_df)} candidate stations",
                flush=True,
            )
            break
        except RuntimeError as exc:
            last_error = exc
            print(
                f"[pv_timeseries] usable windows not enough after {len(loaded_df)} candidates, continue...",
                flush=True,
            )

    status = pd.concat(status_parts, ignore_index=True) if status_parts else pd.DataFrame()
    status.to_csv(out_dir / "station_load_status.csv", index=False)
    if chosen is None:
        raise RuntimeError(str(last_error) if last_error is not None else "Failed to find usable station windows.")

    all_values = []
    station_rows = []
    for item in chosen:
        values = np.asarray(item["values"], dtype=np.float32)
        all_values.append(values)
        station_rows.append(
            {
                "rank": int(item["rank"]),
                "station_sn": str(item["station_sn"]),
                "lon": float(item["lon"]),
                "lat": float(item["lat"]),
                "start_time": str(item["start_time"]),
                "end_time": str(item["end_time"]),
                "values": values.tolist(),
            }
        )
        render_single_series(
            item["times"],
            values,
            out_dir / f"pv_station_timeseries_{int(item['rank']):02d}_{item['station_sn']}.png",
            args,
        )
        print(
            f"[pv_timeseries] station {int(item['rank'])}/{len(chosen)} "
            f"sn={item['station_sn']} {item['start_time']} -> {item['end_time']}",
            flush=True,
        )
    panel_path = out_dir / "pv_station_timeseries_panel.png"
    render_panel(all_values, panel_path, args)

    write_json(
        out_dir / "manifest.json",
        {
            "purpose": "Transparent Stage 3 PV station multi-day sparkline assets sampled from real station CSVs.",
            "station_csv": str(args.station_csv),
            "station_count": int(args.station_count),
            "window_points": int(args.window_points),
            "points_per_day": int(args.points_per_day),
            "approx_days": float(args.window_points) / max(float(args.points_per_day), 1.0),
            "panel_png": str(panel_path),
            "stations": station_rows,
        },
    )
    print("==== Stage 3 PV station timeseries assets complete ====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
