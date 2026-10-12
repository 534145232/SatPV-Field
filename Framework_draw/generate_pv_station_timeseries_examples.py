#!/usr/bin/env python3
"""Generate a few transparent PV station sparkline examples for framework figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_DIR = WORKSPACE / "data" / "temp" / "same_day06_nwp_train_test"
DEFAULT_VAL_DIR = WORKSPACE / "data" / "temp" / "same_day06_nwp_val_test"
DEFAULT_OUT_DIR = SCRIPT_DIR / "pv_station_timeseries_examples"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

TIME_CANDIDATES = ["date", "datetime", "timestamp", "time", "ds"]
PV_CANDIDATES = ["elec", "pv", "power", "active_power"]
CAPACITY_CANDIDATES = ["Capacity", "capacity", "cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_pv_dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--history_steps", type=int, default=12)
    parser.add_argument("--freq_minutes", type=int, default=15)
    parser.add_argument("--capacity_scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--line_color", type=str, default="#3468ff")
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--glow_width", type=float, default=5.5)
    parser.add_argument("--glow_alpha", type=float, default=0.10)
    parser.add_argument("--single_width", type=float, default=2.35)
    parser.add_argument("--single_height", type=float, default=0.68)
    parser.add_argument("--panel_width", type=float, default=3.2)
    parser.add_argument("--panel_height_per_series", type=float, default=0.76)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--ymax", type=float, default=1.02)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in columns:
        low = str(col).strip().lower()
        if any(cand.lower() in low for cand in candidates):
            return str(col)
    return None


def list_station_csvs(*dirs: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for folder in dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            stem = path.stem.lower()
            if stem.startswith("audit_") or stem.startswith("summary_"):
                continue
            files.setdefault(path.name, path)
    return list(files.values())


def load_series(path: Path, capacity_scale: float) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    time_col = _find_col(list(df.columns), TIME_CANDIDATES)
    pv_col = _find_col(list(df.columns), PV_CANDIDATES)
    cap_col = _find_col(list(df.columns), CAPACITY_CANDIDATES)
    if time_col is None or pv_col is None or cap_col is None:
        return None
    ts = pd.to_datetime(df[time_col], errors="coerce")
    pv = pd.to_numeric(df[pv_col], errors="coerce").astype(float)
    cap = pd.to_numeric(df[cap_col], errors="coerce").astype(float)
    cap_value = float(cap.dropna().median()) if cap.dropna().size else np.nan
    if not np.isfinite(cap_value) or cap_value <= 0:
        return None
    pv_norm = pv / max(cap_value * float(capacity_scale), 1e-6)
    out = pd.DataFrame({"timestamp": ts, "pv_norm": pv_norm})
    out = out.dropna(subset=["timestamp", "pv_norm"])
    out = out[(out["pv_norm"] >= 0.0) & (out["pv_norm"] <= 1.3)]
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out


def collect_candidate_windows(
    series: pd.DataFrame,
    station_sn: str,
    source_path: Path,
    history_steps: int,
    freq_minutes: int,
) -> list[dict]:
    if len(series) < history_steps:
        return []
    timestamps = pd.to_datetime(series["timestamp"])
    values = series["pv_norm"].to_numpy(np.float32)
    delta = pd.Timedelta(minutes=int(freq_minutes))
    windows: list[dict] = []
    for start in range(0, len(series) - history_steps + 1):
        end = start + history_steps
        ts_window = timestamps.iloc[start:end]
        if not ((ts_window.diff().iloc[1:] == delta).all()):
            continue
        value_window = values[start:end]
        v_max = float(np.nanmax(value_window))
        v_std = float(np.nanstd(value_window))
        v_range = float(np.nanmax(value_window) - np.nanmin(value_window))
        if v_max < 0.06 or v_std < 0.01:
            continue
        slope = float(value_window[-1] - value_window[0])
        peak_index = int(np.nanargmax(value_window))
        score = 1.7 * v_range + 1.2 * v_std + 0.4 * v_max
        windows.append(
            {
                "station_sn": station_sn,
                "source_csv": str(source_path),
                "start_time": str(pd.Timestamp(ts_window.iloc[0])),
                "end_time": str(pd.Timestamp(ts_window.iloc[-1])),
                "values": value_window.tolist(),
                "slope": float(slope),
                "peak_index": int(peak_index),
                "score": float(score),
            }
        )
    return windows


def _time_gap_ok(candidate: dict, chosen: list[dict], min_gap_minutes: int = 60) -> bool:
    cand_start = pd.Timestamp(candidate["start_time"])
    for item in chosen:
        if item["station_sn"] != candidate["station_sn"]:
            continue
        other_start = pd.Timestamp(item["start_time"])
        if abs((cand_start - other_start).total_seconds()) < min_gap_minutes * 60:
            return False
    return True


def choose_windows(candidates: list[dict], count: int, seed: int) -> list[dict]:
    if not candidates:
        raise RuntimeError("No valid PV sparkline windows found in the available CSV files.")
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top_k = ranked[: max(int(count) * 10, min(60, len(ranked)))]

    rising = [x for x in top_k if x["slope"] >= 0.18]
    falling = [x for x in top_k if x["slope"] <= -0.18]
    peaked = [x for x in top_k if 2 <= int(x["peak_index"]) <= max(2, len(x["values"]) - 3)]

    chosen: list[dict] = []
    for bucket in (rising, falling, peaked):
        for item in bucket:
            if item not in chosen and _time_gap_ok(item, chosen):
                chosen.append(item)
                break
        if len(chosen) >= count:
            return chosen[:count]

    rng = np.random.default_rng(int(seed))
    remaining = [x for x in top_k if x not in chosen]
    if remaining:
        jitter = rng.uniform(-0.03, 0.03, size=len(remaining))
        remaining = [
            item
            for _, item in sorted(
                zip(
                    [float(item["score"]) + float(noise) for item, noise in zip(remaining, jitter)],
                    remaining,
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]
    for item in remaining:
        if _time_gap_ok(item, chosen):
            chosen.append(item)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        for item in ranked:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) >= count:
                break
    return chosen[:count]


def render_single(values: np.ndarray, out_path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(values), dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(float(args.single_width), float(args.single_height)))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.plot(x, y, color=str(args.line_color), linewidth=float(args.glow_width), alpha=float(args.glow_alpha), solid_capstyle="round")
    ax.plot(x, y, color=str(args.line_color), linewidth=float(args.line_width), alpha=0.98, solid_capstyle="round")
    ax.scatter([x[-1]], [y[-1]], s=9.0, color=str(args.line_color), alpha=0.98, zorder=3)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, float(args.ymax))
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=int(args.dpi), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_panel(all_values: list[np.ndarray], out_path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(all_values)
    fig_h = max(0.8, float(args.panel_height_per_series) * max(count, 1))
    fig, axes = plt.subplots(count, 1, figsize=(float(args.panel_width), fig_h), squeeze=False)
    fig.patch.set_alpha(0.0)
    for ax, values in zip(axes[:, 0], all_values):
        x = np.arange(len(values), dtype=np.float32)
        y = np.asarray(values, dtype=np.float32)
        ax.set_facecolor((1, 1, 1, 0))
        ax.plot(x, y, color=str(args.line_color), linewidth=float(args.glow_width), alpha=float(args.glow_alpha), solid_capstyle="round")
        ax.plot(x, y, color=str(args.line_color), linewidth=float(args.line_width), alpha=0.98, solid_capstyle="round")
        ax.scatter([x[-1]], [y[-1]], s=9.0, color=str(args.line_color), alpha=0.98, zorder=3)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(0.0, float(args.ymax))
        ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, hspace=0.33)
    fig.savefig(out_path, dpi=int(args.dpi), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    csv_files = list_station_csvs(Path(args.train_pv_dir), Path(args.val_pv_dir))
    if not csv_files:
        raise FileNotFoundError("No station CSV files found in the provided train/val directories.")

    candidates: list[dict] = []
    for path in csv_files:
        series = load_series(path, float(args.capacity_scale))
        if series is None or series.empty:
            continue
        candidates.extend(
            collect_candidate_windows(
                series=series,
                station_sn=path.stem,
                source_path=path,
                history_steps=int(args.history_steps),
                freq_minutes=int(args.freq_minutes),
            )
        )
    chosen = choose_windows(candidates, int(args.count), int(args.seed))

    all_values: list[np.ndarray] = []
    entries: list[dict] = []
    for idx, item in enumerate(chosen, start=1):
        values = np.asarray(item["values"], dtype=np.float32)
        all_values.append(values)
        png_path = out_dir / f"pv_station_timeseries_{idx:02d}.png"
        render_single(values, png_path, args)
        entries.append(
            {
                "rank": int(idx),
                "station_sn": str(item["station_sn"]),
                "source_csv": str(item["source_csv"]),
                "start_time": str(item["start_time"]),
                "end_time": str(item["end_time"]),
                "png": str(png_path),
                "values": [float(v) for v in values],
            }
        )
        print(
            f"[pv_timeseries_examples] {idx}/{len(chosen)} station={item['station_sn']} "
            f"{item['start_time']} -> {item['end_time']}",
            flush=True,
        )

    panel_path = out_dir / "pv_station_timeseries_panel.png"
    render_panel(all_values, panel_path, args)
    write_json(
        out_dir / "manifest.json",
        {
            "purpose": "Transparent PV station sparkline examples for framework figures.",
            "train_pv_dir": str(args.train_pv_dir),
            "val_pv_dir": str(args.val_pv_dir),
            "count": int(args.count),
            "history_steps": int(args.history_steps),
            "freq_minutes": int(args.freq_minutes),
            "panel_png": str(panel_path),
            "entries": entries,
        },
    )
    print("==== PV station timeseries examples complete ====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
