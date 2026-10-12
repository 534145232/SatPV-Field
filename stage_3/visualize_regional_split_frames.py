#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stage3_paths as p3

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": str(p3.DATA_DIR / "_local_core_defaults"),
    "train_pv_dir": str(p3.TRAIN_PV_DIR),
    "val_pv_dir": str(p3.VAL_PV_DIR),
    "out_dir": "",
    "start_time": "",
    "candidate_time_start": "2023-06-14 08:30:00",
    "candidate_time_end": "2023-06-14 14:00:00",
    "auto_select_start": True,
    "frame_count": 8,
    "freq_minutes": 15,
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "clearsky_eps": 0.02,
    "min_clear_sky": 0.03,
    "csi_clip": 2.0,
    "capacity_scale": 1000.0,
    "kpi_quantile": 0.95,
    "kpi_min_samples": 40,
    "kpi_max_samples_per_station": 2000,
    "dpi": 170,
}


TIME_CANDIDATES = [
    "date",
    "datetime",
    "timestamp",
    "time",
    "ds",
    "采集时间",
    "时间",
    "日期",
]
PV_CANDIDATES = [
    "elec",
    "pv",
    "power",
    "active_power",
    "发电功率",
    "功率",
    "出力",
]
CAPACITY_CANDIDATES = [
    "Capacity",
    "capacity",
    "cap",
    "瑁呮満瀹归噺",
    "瀹归噺",
]


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        if value == "":
            continue
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize consecutive DensePV CSI frames under the new fixed regional train/val/test split. "
            "Only train stations are splatted into the input field; val/test stations are shown as held-out truth."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--start_time", type=str, default=str(PYCHARM_DEFAULTS["start_time"]))
    parser.add_argument("--candidate_time_start", type=str, default=str(PYCHARM_DEFAULTS["candidate_time_start"]))
    parser.add_argument("--candidate_time_end", type=str, default=str(PYCHARM_DEFAULTS["candidate_time_end"]))
    parser.add_argument("--auto_select_start", action="store_true", default=bool(PYCHARM_DEFAULTS["auto_select_start"]))
    parser.add_argument("--no_auto_select_start", action="store_false", dest="auto_select_start")
    parser.add_argument("--frame_count", type=int, default=int(PYCHARM_DEFAULTS["frame_count"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--clearsky_eps", type=float, default=float(PYCHARM_DEFAULTS["clearsky_eps"]))
    parser.add_argument("--min_clear_sky", type=float, default=float(PYCHARM_DEFAULTS["min_clear_sky"]))
    parser.add_argument("--csi_clip", type=float, default=float(PYCHARM_DEFAULTS["csi_clip"]))
    parser.add_argument("--capacity_scale", type=float, default=float(PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--kpi_quantile", type=float, default=float(PYCHARM_DEFAULTS["kpi_quantile"]))
    parser.add_argument("--kpi_min_samples", type=int, default=int(PYCHARM_DEFAULTS["kpi_min_samples"]))
    parser.add_argument("--kpi_max_samples_per_station", type=int, default=int(PYCHARM_DEFAULTS["kpi_max_samples_per_station"]))
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def _geo_solar_elevation_scale(timestamp: pd.Timestamp, lon: float, lat: float) -> float:
    ts = pd.Timestamp(timestamp)
    day = int(ts.dayofyear)
    hour = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    time_offset = eq_time + 4.0 * float(lon) - 60.0 * 8.0
    true_solar_min = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_min / 4.0 - 180.0)
    lat_rad = math.radians(float(lat))
    sin_elev = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    return float(np.clip(sin_elev, 0.0, 1.2))


def _station_csv_path(station_sn: str, train_pv_dir: Path, val_pv_dir: Path) -> Path | None:
    train_path = train_pv_dir / f"{station_sn}.csv"
    val_path = val_pv_dir / f"{station_sn}.csv"
    if train_path.exists():
        return train_path
    if val_path.exists():
        return val_path
    return None


def _normalize_pv_with_capacity(pv: pd.Series, raw_capacity: float | None) -> tuple[pd.Series, float, str]:
    """Return capacity-normalized PV with a robust fallback.

    Some source CSVs do not expose a capacity column, and the station metadata
    column names differ across exports. If capacity is missing, using raw kW as
    normalized PV would incorrectly drop most daytime points. The fallback
    estimates a scale from the station's high PV quantile, only for visualization
    and this regional-grid prototype.
    """
    pv_num = pd.to_numeric(pv, errors="coerce").astype(float)
    raw = pv_num.to_numpy(np.float64)
    finite = raw[np.isfinite(raw) & (raw >= 0.0)]
    q99 = float(np.nanquantile(finite, 0.995)) if finite.size else np.nan

    if raw_capacity is not None and np.isfinite(raw_capacity) and raw_capacity > 0:
        cap_scale = float(raw_capacity)
        pv_norm = pv_num / max(cap_scale, 1e-6)
        norm_q99 = float(np.nanquantile(pv_norm.to_numpy(np.float64), 0.995)) if pv_norm.notna().any() else np.nan
        if np.isfinite(norm_q99) and norm_q99 <= 1.8:
            return pv_norm, cap_scale, "capacity_column_or_metadata"
        if np.isfinite(q99) and q99 <= 1.8:
            return pv_num, 1.0, "already_normalized_fallback"

    if np.isfinite(q99) and q99 > 1.8:
        cap_scale = max(q99, 1e-6)
        return pv_num / cap_scale, cap_scale, "estimated_from_station_pv_q995"
    return pv_num, 1.0, "already_normalized_or_unknown"


def load_station_series(split_df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    rows = []
    for i, row in split_df.iterrows():
        sn = str(row["station_sn"])
        path = _station_csv_path(sn, args.train_pv_dir, args.val_pv_dir)
        if path is None:
            rows.append({"station_sn": sn, "status": "missing_csv"})
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({"station_sn": sn, "status": f"read_failed:{exc}"})
            continue
        time_col = "date" if "date" in df.columns else _find_col([str(c) for c in df.columns], TIME_CANDIDATES)
        pv_col = "elec" if "elec" in df.columns else _find_col([str(c) for c in df.columns], PV_CANDIDATES)
        cap_col = "Capacity" if "Capacity" in df.columns else _find_col([str(c) for c in df.columns], CAPACITY_CANDIDATES)
        if time_col is None or pv_col is None or cap_col is None:
            rows.append({"station_sn": sn, "status": "missing_date_elec_or_capacity_col", "columns": "|".join(map(str, df.columns))})
            continue
        ts = pd.to_datetime(df[time_col], errors="coerce")
        pv = pd.to_numeric(df[pv_col], errors="coerce").astype(float)
        cap = pd.to_numeric(df[cap_col], errors="coerce").astype(float)
        cap_value = float(cap.dropna().median()) if cap.dropna().size else np.nan
        if not np.isfinite(cap_value) or cap_value <= 0:
            rows.append({"station_sn": sn, "status": "invalid_capacity", "csv_path": str(path)})
            continue
        cap_scale = float(cap_value) * float(args.capacity_scale)
        pv_norm = pv / max(cap_scale, 1e-6)
        cap_source = "csv_capacity_times_capacity_scale"
        out = pd.DataFrame({"timestamp": ts, "pv_norm": pv_norm})
        out = out.dropna(subset=["timestamp", "pv_norm"])
        out = out[(out["pv_norm"] >= 0.0) & (out["pv_norm"] <= 1.3)]
        out = out.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        series[sn] = out
        rows.append(
            {
                "station_sn": sn,
                "status": "ok",
                "n_rows": int(len(out)),
                "csv_path": str(path),
                "time_col": str(time_col),
                "pv_col": str(pv_col),
                "capacity_col": str(cap_col) if cap_col is not None else "",
            "capacity_scale_used": float(cap_scale),
            "capacity_scale_source": cap_source,
            "capacity_value_raw": float(cap_value),
            "capacity_scale_factor": float(args.capacity_scale),
            "pv_norm_q995": float(np.nanquantile(out["pv_norm"].to_numpy(np.float64), 0.995)) if len(out) else np.nan,
        }
        )
        if (i + 1) % 100 == 0:
            print(f"[regional_frames] loaded {i + 1}/{len(split_df)} station CSVs")
    return series, pd.DataFrame(rows)


def estimate_kpi(split_df: pd.DataFrame, series: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, float]:
    kpi: dict[str, float] = {}
    values_all = []
    q = float(args.kpi_quantile)
    max_samples = int(getattr(args, "kpi_max_samples_per_station", 2000))
    for i, (_, row) in enumerate(split_df.iterrows(), start=1):
        sn = str(row["station_sn"])
        df = series.get(sn)
        if df is None or df.empty:
            continue
        work = df
        if max_samples > 0 and len(work) > max_samples:
            step = max(1, int(math.ceil(len(work) / max_samples)))
            work = work.iloc[::step]
        vals = []
        lon = float(row["lon"])
        lat = float(row["lat"])
        for ts, pv_norm in zip(work.index, work["pv_norm"].to_numpy(np.float64)):
            clear = _geo_solar_elevation_scale(pd.Timestamp(ts), lon, lat)
            if clear < float(args.min_clear_sky) or not np.isfinite(pv_norm):
                continue
            vals.append(float(pv_norm) / max(float(args.clearsky_eps), clear))
        arr = np.clip(np.asarray(vals, dtype=np.float64), 0.0, float(args.csi_clip))
        if arr.size:
            values_all.append(arr)
        if arr.size >= int(args.kpi_min_samples):
            kpi[sn] = float(np.nanquantile(arr, q))
        if i % 100 == 0:
            print(f"[regional_frames] estimated KPI for {i}/{len(split_df)} stations", flush=True)
    fallback = float(np.nanquantile(np.concatenate(values_all), q)) if values_all else 1.0
    for sn in split_df["station_sn"].astype(str):
        kpi.setdefault(sn, fallback)
    return kpi


def _bounds(split_df: pd.DataFrame) -> tuple[float, float, float, float]:
    lon_min = float(split_df["lon"].min())
    lon_max = float(split_df["lon"].max())
    lat_min = float(split_df["lat"].min())
    lat_max = float(split_df["lat"].max())
    lon_pad = max(1e-6, (lon_max - lon_min) * 0.04)
    lat_pad = max(1e-6, (lat_max - lat_min) * 0.04)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def _xy_to_ij(lon: float, lat: float, bounds: tuple[float, float, float, float], g: int) -> tuple[int, int]:
    lon_min, lon_max, lat_min, lat_max = bounds
    x = int(round((float(lon) - lon_min) / max(1e-12, lon_max - lon_min) * (g - 1)))
    y = int(round((float(lat) - lat_min) / max(1e-12, lat_max - lat_min) * (g - 1)))
    return int(np.clip(y, 0, g - 1)), int(np.clip(x, 0, g - 1))


def station_value(row: pd.Series, ts: pd.Timestamp, series: dict[str, pd.DataFrame], kpi: dict[str, float], args: argparse.Namespace) -> float | None:
    sn = str(row["station_sn"])
    df = series.get(sn)
    if df is None or ts not in df.index:
        return None
    pv_norm = float(df.loc[ts, "pv_norm"])
    clear = _geo_solar_elevation_scale(ts, float(row["lon"]), float(row["lat"]))
    if clear < float(args.min_clear_sky) or not np.isfinite(pv_norm) or pv_norm < 0.0:
        return None
    val = pv_norm / max(float(args.clearsky_eps), clear * max(float(kpi.get(sn, 1.0)), 1e-3))
    if not np.isfinite(val):
        return None
    return float(np.clip(val, 0.0, float(args.csi_clip)))


def splat_train_field(split_df: pd.DataFrame, ts: pd.Timestamp, series: dict[str, pd.DataFrame], kpi: dict[str, float], bounds: tuple[float, float, float, float], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    g = int(args.grid_size)
    field_sum = np.zeros((g, g), dtype=np.float32)
    weight_sum = np.zeros((g, g), dtype=np.float32)
    rows = []
    sigma = max(1e-6, float(args.gaussian_sigma_cells))
    radius = max(0, int(args.gaussian_radius_cells))
    train_df = split_df[split_df["split"] == "train"]
    for _, row in train_df.iterrows():
        val = station_value(row, ts, series, kpi, args)
        if val is None:
            continue
        cy, cx = _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, g)
        for yy in range(max(0, cy - radius), min(g, cy + radius + 1)):
            for xx in range(max(0, cx - radius), min(g, cx + radius + 1)):
                dist2 = float((yy - cy) ** 2 + (xx - cx) ** 2)
                if dist2 > float(radius * radius):
                    continue
                w = math.exp(-dist2 / (2.0 * sigma * sigma))
                field_sum[yy, xx] += float(val) * w
                weight_sum[yy, xx] += w
        rows.append(
            {
                "timestamp": str(ts),
                "split": "train",
                "station_sn": str(row["station_sn"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "value": float(val),
            }
        )
    field = np.zeros_like(field_sum)
    mask = weight_sum > float(args.min_grid_weight)
    field[mask] = field_sum[mask] / weight_sum[mask]
    return field, np.clip(weight_sum, 0.0, 1.0), pd.DataFrame(rows)


def sparse_truth(split_df: pd.DataFrame, split: str, ts: pd.Timestamp, series: dict[str, pd.DataFrame], kpi: dict[str, float], bounds: tuple[float, float, float, float], args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    g = int(args.grid_size)
    truth = np.full((g, g), np.nan, dtype=np.float32)
    rows = []
    for _, row in split_df[split_df["split"] == split].iterrows():
        val = station_value(row, ts, series, kpi, args)
        if val is None:
            continue
        y, x = _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, g)
        if np.isnan(truth[y, x]):
            truth[y, x] = float(val)
        else:
            truth[y, x] = float(np.nanmean([truth[y, x], val]))
        rows.append(
            {
                "timestamp": str(ts),
                "split": split,
                "station_sn": str(row["station_sn"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "value": float(val),
            }
        )
    return truth, pd.DataFrame(rows)


def count_valid_values(split_df: pd.DataFrame, split: str, ts: pd.Timestamp, series: dict[str, pd.DataFrame], kpi: dict[str, float], args: argparse.Namespace) -> int:
    n = 0
    for _, row in split_df[split_df["split"] == split].iterrows():
        if station_value(row, ts, series, kpi, args) is not None:
            n += 1
    return n


def choose_visual_times(split_df: pd.DataFrame, series: dict[str, pd.DataFrame], kpi: dict[str, float], args: argparse.Namespace) -> list[pd.Timestamp]:
    frame_count = int(args.frame_count)
    freq = pd.Timedelta(minutes=int(args.freq_minutes))
    if str(args.start_time).strip() and not bool(args.auto_select_start):
        start = pd.Timestamp(args.start_time)
        return [start + freq * i for i in range(frame_count)]
    if str(args.start_time).strip() and not bool(args.auto_select_start):
        start = pd.Timestamp(args.start_time)
        return [start + freq * i for i in range(frame_count)]

    candidate_times = list(pd.date_range(pd.Timestamp(args.candidate_time_start), pd.Timestamp(args.candidate_time_end), freq=f"{int(args.freq_minutes)}min"))
    if not candidate_times:
        start = pd.Timestamp(args.start_time or args.candidate_time_start)
        return [start + freq * i for i in range(frame_count)]

    count_rows = []
    for ts in candidate_times:
        count_rows.append(
            {
                "timestamp": ts,
                "n_train_values": count_valid_values(split_df, "train", ts, series, kpi, args),
                "n_val_values": count_valid_values(split_df, "val", ts, series, kpi, args),
                "n_test_values": count_valid_values(split_df, "test", ts, series, kpi, args),
            }
        )
    counts = pd.DataFrame(count_rows)
    best_i = 0
    best_score = -1
    for i in range(max(1, len(counts) - frame_count + 1)):
        w = counts.iloc[i : i + frame_count]
        score = float(w["n_train_values"].sum() * 2.0 + w["n_val_values"].sum() + w["n_test_values"].sum())
        if score > best_score:
            best_score = score
            best_i = i
    chosen = counts.iloc[best_i : best_i + frame_count].copy()
    print("==== Auto-selected visualization window ====")
    print(chosen.to_string(index=False))
    return [pd.Timestamp(x) for x in chosen["timestamp"].tolist()]


def render_frame(
    ts: pd.Timestamp,
    split_df: pd.DataFrame,
    field: np.ndarray,
    support: np.ndarray,
    val_truth: np.ndarray,
    test_truth: np.ndarray,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    lon_min, lon_max, lat_min, lat_max = bounds
    extent = [lon_min, lon_max, lat_min, lat_max]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    field_display = field.astype(np.float32, copy=True)
    field_display[support <= 0.0] = np.nan
    panels = [
        ("train-station input field", field_display, 0.0, 2.0, "viridis", None),
        ("train support", support, 0.0, 1.0, "viridis", None),
        ("val-region truth", val_truth, 0.0, 2.0, "viridis", val_rows),
        ("test-region truth", test_truth, 0.0, 2.0, "viridis", test_rows),
    ]
    for ax, (title, arr, vmin, vmax, cmap, truth_rows) in zip(axes, panels):
        shown = np.ma.masked_invalid(arr)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#f2f2f2")
        im = ax.imshow(shown, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cm, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(alpha=0.18)
        if title.endswith("truth"):
            for split, marker, color in [("val", "D", "#ff7f00"), ("test", "^", "#e41a1c")]:
                g = split_df[split_df["split"] == split]
                ax.scatter(g["lon"], g["lat"], s=14, marker=marker, c="lightgray", edgecolor="none", alpha=0.45, label=f"{split} all")
            if truth_rows is not None and not truth_rows.empty:
                marker = "D" if title.startswith("val") else "^"
                ax.scatter(
                    truth_rows["lon"],
                    truth_rows["lat"],
                    s=38,
                    marker=marker,
                    c=truth_rows["value"],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=0.95,
                    label="valid truth",
                )
            ax.legend(loc="best", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"Regional DensePV split frame | {ts} | KPI-adjusted geo CSI", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    run_dir = Path(args.run_dir)
    split_path = run_dir / "station_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"station split not found: {split_path}; run build_regional_station_split.py first")
    out_dir = ensure_dir(args.out_dir if args.out_dir is not None else run_dir / "regional_frame_visualization")
    frame_dir = ensure_dir(out_dir / "frames")

    split_df = pd.read_csv(split_path, dtype={"station_sn": str})
    split_df["station_sn"] = split_df["station_sn"].astype(str)
    split_df["lon"] = pd.to_numeric(split_df["lon"], errors="coerce")
    split_df["lat"] = pd.to_numeric(split_df["lat"], errors="coerce")
    split_df = split_df.dropna(subset=["lon", "lat", "split"])

    series, load_table = load_station_series(split_df, args)
    load_table.to_csv(out_dir / "station_csv_load_status.csv", index=False)
    if not load_table.empty:
        merged_load = load_table.merge(split_df[["station_sn", "split", "source_folder"]], on="station_sn", how="left")
        load_summary = (
            merged_load.groupby(["split", "source_folder", "status", "capacity_scale_source"], dropna=False)
            .size()
            .reset_index(name="n_stations")
            .sort_values(["split", "source_folder", "status", "capacity_scale_source"])
        )
        load_summary.to_csv(out_dir / "station_csv_load_summary.csv", index=False)
        print("==== Station CSV load summary ====")
        print(load_summary.to_string(index=False))
    kpi = estimate_kpi(split_df, series, args)
    pd.DataFrame([{"station_sn": k, "kpi": v} for k, v in sorted(kpi.items())]).to_csv(out_dir / "station_kpi_geo_baselines.csv", index=False)

    bounds = _bounds(split_df)
    times = choose_visual_times(split_df, series, kpi, args)
    station_value_rows = []
    frame_rows = []
    for i, ts in enumerate(times):
        field, support, train_rows = splat_train_field(split_df, ts, series, kpi, bounds, args)
        val_truth, val_rows = sparse_truth(split_df, "val", ts, series, kpi, bounds, args)
        test_truth, test_rows = sparse_truth(split_df, "test", ts, series, kpi, bounds, args)
        station_value_rows.extend([train_rows, val_rows, test_rows])
        out_path = frame_dir / f"regional_split_frame_{i:03d}_{ts.strftime('%Y%m%d_%H%M')}.png"
        render_frame(ts, split_df, field, support, val_truth, test_truth, val_rows, test_rows, bounds, out_path, int(args.dpi))
        frame_rows.append(
            {
                "frame_idx": i,
                "timestamp": str(ts),
                "n_train_values": int(len(train_rows)),
                "n_val_values": int(len(val_rows)),
                "n_test_values": int(len(test_rows)),
                "n_train_stations_total": int((split_df["split"] == "train").sum()),
                "n_val_stations_total": int((split_df["split"] == "val").sum()),
                "n_test_stations_total": int((split_df["split"] == "test").sum()),
                "input_support_fraction": float(np.mean(support > float(args.min_grid_weight))),
                "frame_png": str(out_path),
            }
        )
        print(
            f"[regional_frames] {i + 1}/{len(times)} {ts} "
            f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    station_values = pd.concat([x for x in station_value_rows if not x.empty], ignore_index=True) if station_value_rows else pd.DataFrame()
    station_values.to_csv(out_dir / "frame_station_values.csv", index=False)
    frame_summary = pd.DataFrame(frame_rows)
    frame_summary.to_csv(out_dir / "frame_summary.csv", index=False)
    write_json(
        out_dir / "visualization_meta.json",
        {
            "run_dir": str(run_dir),
            "station_split_csv": str(split_path),
            "train_pv_dir": str(args.train_pv_dir),
            "val_pv_dir": str(args.val_pv_dir),
            "value": "KPI-adjusted CSI = PV_norm / max(geo_clear_sky * station_KPI, eps)",
            "grid_size": int(args.grid_size),
            "gaussian_sigma_cells": float(args.gaussian_sigma_cells),
            "gaussian_radius_cells": int(args.gaussian_radius_cells),
            "auto_select_start": bool(args.auto_select_start),
            "candidate_time_start": str(args.candidate_time_start),
            "candidate_time_end": str(args.candidate_time_end),
            "bounds": {
                "lon_min": bounds[0],
                "lon_max": bounds[1],
                "lat_min": bounds[2],
                "lat_max": bounds[3],
            },
        },
    )
    print("==== Regional split frame visualization ====")
    print(frame_summary.to_string(index=False))
    print(f"saved: {out_dir / 'frame_summary.csv'}")
    print(f"frames: {frame_dir}")


if __name__ == "__main__":
    main()

