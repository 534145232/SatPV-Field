#!/usr/bin/env python3
"""Analyze Stage 4 station forecast performance under weather/PV regimes.

The default regime definition uses the station-level historical weather columns
as ERA5/reanalysis variables. Observation-side PV regimes are also available
through --condition_source pv_clear_sky / pv_quantile / pv_cluster.

External ERA5 CSV support is kept only as an optional extension.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras  # noqa: E402
from stage4_paths import (  # noqa: E402
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)


def _load_stage4_temporal_module():
    path = SCRIPT_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_temporal_baseline_for_weather", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_temporal_baseline_for_weather"] = module
    spec.loader.exec_module(module)
    return module


f03 = _load_stage4_temporal_module()

DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]

METHOD_SPECS = [
    {
        "key": "temporal_baseline",
        "method": "Temporal baseline",
        "path_arg": "baseline_detail_csv",
    },
    {
        "key": "future_swr_patch",
        "method": "+ Future SWR patch",
        "path_arg": "future_swr_patch_detail_csv",
    },
    {
        "key": "historical_pv_field_patch",
        "method": "+ Historical PV-field patch",
        "path_arg": "historical_field_detail_csv",
    },
    {
        "key": "fg_msf",
        "method": "FG-MSF",
        "path_arg": "future_field_detail_csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument("--extra_detail_csv", action="append", default=[], help="Optional extra method as key:label:path")
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "18_eval_stage4_weather_condition_analysis")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--align_mode", choices=["intersection", "reference", "none"], default="intersection")
    parser.add_argument("--valid_minutes_per_horizon", type=int, default=15)
    parser.add_argument("--min_group_points", type=int, default=100)
    parser.add_argument(
        "--condition_source",
        choices=[
            "station_era5_threshold",
            "station_era5_cluster",
            "pv_clear_sky",
            "pv_quantile",
            "pv_cluster",
            "external_era5_threshold",
            "external_era5_cluster",
        ],
        default="station_era5_threshold",
        help="Default uses ERA5/reanalysis weather columns in the station-level history CSVs.",
    )
    parser.add_argument("--era5_csv", type=Path, default=None, help="Optional external ERA5/reanalysis CSV.")
    parser.add_argument("--era5_time_col", type=str, default="", help="External ERA5 time column; auto-detected if empty.")
    parser.add_argument("--era5_station_col", type=str, default="", help="Optional external ERA5 station id column; auto-detected if empty.")
    parser.add_argument("--era5_tcc_col", type=str, default="", help="External ERA5 cloud-cover column; auto-detected if empty.")
    parser.add_argument("--era5_radiation_col", type=str, default="", help="External ERA5 radiation/SSRD column; auto-detected if empty.")
    parser.add_argument("--era5_precip_col", type=str, default="", help="Optional external ERA5 precipitation column; auto-detected if empty.")
    parser.add_argument("--cluster_k", type=int, default=3)
    parser.add_argument("--cluster_seed", type=int, default=2026)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)

    # Arguments required by 03_train_per_station_temporal_baseline.load_station_table.
    parser.add_argument("--train_pv_dir", type=Path, default=Path(f03.PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(f03.PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--capacity_scale", type=float, default=float(f03.PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--health_filter", action="store_true", default=bool(f03.PYCHARM_DEFAULTS["health_filter"]))
    parser.add_argument("--disable_health_filter", action="store_false", dest="health_filter")
    parser.add_argument("--pv_health_max", type=float, default=float(f03.PYCHARM_DEFAULTS["pv_health_max"]))
    parser.add_argument("--pv_health_active_threshold", type=float, default=float(f03.PYCHARM_DEFAULTS["pv_health_active_threshold"]))
    parser.add_argument("--pv_health_flatline_window", type=int, default=int(f03.PYCHARM_DEFAULTS["pv_health_flatline_window"]))
    parser.add_argument("--pv_health_flatline_range", type=float, default=float(f03.PYCHARM_DEFAULTS["pv_health_flatline_range"]))
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def safe_label(label: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "model"


def parse_extra_specs(extra_args: list[str]) -> list[dict[str, str]]:
    specs = []
    for item in extra_args:
        parts = str(item).split(":", 2)
        if len(parts) != 3:
            raise ValueError("--extra_detail_csv must be key:label:path")
        key, label, path = parts
        specs.append({"key": safe_label(key), "method": label, "path": path})
    return specs


def read_detail(path: Path, key: str, method: str, split: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing eval-detail CSV for {method}: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [*KEY_COLS, "horizon", "pred_pv_norm", "true_pv_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = attach_capacity_from_station_meta(df, path)
    df = df[df["eval_split"].astype(str) == str(split)].copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    df["h"] = df["h"].astype(int)
    df["horizon"] = df["horizon"].astype(str)
    if "large_change" in df.columns:
        df["large_change"] = coerce_bool(df["large_change"])
    df["method_key"] = key
    df["method"] = method
    print(f"[read] {method}: {path} rows={len(df)}", flush=True)
    return df


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEY_COLS].drop_duplicates().copy()


def align_details(details: dict[str, pd.DataFrame], mode: str) -> dict[str, pd.DataFrame]:
    if mode == "none":
        return details
    if mode == "reference":
        if "temporal_baseline" not in details:
            raise ValueError("--align_mode reference requires temporal_baseline")
        keys = key_frame(details["temporal_baseline"])
    else:
        keys = None
        for df in details.values():
            k = key_frame(df)
            keys = k if keys is None else keys.merge(k, on=KEY_COLS, how="inner")
    assert keys is not None
    out = {}
    for key, df in details.items():
        before = len(df)
        out[key] = df.merge(keys, on=KEY_COLS, how="inner")
        if len(out[key]) != before:
            print(f"[align] {key}: rows {before} -> {len(out[key])}", flush=True)
    return out


def attach_valid_time(keys: pd.DataFrame, minutes_per_h: int) -> pd.DataFrame:
    out = keys.copy()
    out["issue_time"] = out["issue_time"].astype(str)
    out["valid_time"] = pd.to_datetime(out["issue_time"]) + pd.to_timedelta(out["h"].astype(int) * int(minutes_per_h), unit="min")
    return out


def _find_col(columns: list[str], candidates: list[str], explicit: str = "") -> str | None:
    if explicit and explicit in columns:
        return explicit
    lower = {str(c).strip().lower(): str(c) for c in columns}
    if explicit and explicit.lower() in lower:
        return lower[explicit.lower()]
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in columns:
        low = str(col).strip().lower()
        if any(cand.lower() in low for cand in candidates):
            return str(col)
    return None


def build_weather_context(reference: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keys = attach_valid_time(reference, int(args.valid_minutes_per_horizon))
    context_rows = []
    for sn, group in keys.groupby("station_sn", sort=True):
        station_df, _meta = f03.load_station_table(str(sn), args)
        wanted = pd.DatetimeIndex(pd.to_datetime(group["valid_time"].drop_duplicates()))
        cols = [c for c in ["pv_norm", "radiation", "tcc", "clearsky_ghi", "solar_zenith", "humidity", "temperature", "speed10"] if c in station_df.columns]
        if not cols:
            raise ValueError(f"Station {sn} has no PV/weather columns after loading.")
        sub = station_df.reindex(wanted)[cols].reset_index().rename(columns={"timestamp": "valid_time", "index": "valid_time"})
        if "valid_time" not in sub.columns:
            sub = sub.rename(columns={sub.columns[0]: "valid_time"})
        sub["valid_time"] = pd.to_datetime(sub["valid_time"])
        sub["station_sn"] = str(sn)
        pv = pd.to_numeric(station_df["pv_norm"], errors="coerce").astype(float)
        active_pv = pv[np.isfinite(pv) & (pv > 0.02)]
        sub["station_pv_q95"] = float(active_pv.quantile(0.95)) if active_pv.size else float("nan")
        if "clearsky_ghi" in station_df.columns:
            clear = pd.to_numeric(station_df["clearsky_ghi"], errors="coerce").astype(float)
            active_clear = clear[np.isfinite(clear) & (clear > 1.0)]
            sub["station_clearsky_q95"] = float(active_clear.quantile(0.95)) if active_clear.size else float("nan")
        context_rows.append(sub)
    weather = pd.concat(context_rows, ignore_index=True) if context_rows else pd.DataFrame()
    merged = keys.merge(weather, on=["station_sn", "valid_time"], how="left")
    if "true_pv_norm" not in merged.columns and "pv_norm" in merged.columns:
        merged["true_pv_norm"] = merged["pv_norm"]
    merged = attach_era5_context(merged, args)
    return assign_weather_conditions(merged, args)


def attach_era5_context(context: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.era5_csv is None or not Path(args.era5_csv).exists():
        return context
    era5 = pd.read_csv(args.era5_csv)
    cols = [str(c) for c in era5.columns]
    time_col = _find_col(cols, ["valid_time", "time", "datetime", "timestamp", "date"], str(args.era5_time_col))
    if time_col is None:
        raise ValueError(f"Could not detect ERA5 time column in {args.era5_csv}. Columns={cols}")
    station_col = _find_col(cols, ["station_sn", "station", "site", "id"], str(args.era5_station_col))
    tcc_col = _find_col(cols, ["tcc", "cloud_cover", "total_cloud_cover", "cloud"], str(args.era5_tcc_col))
    rad_col = _find_col(cols, ["ssrd", "surface_solar_radiation_downwards", "radiation", "ghi", "swr"], str(args.era5_radiation_col))
    precip_col = _find_col(cols, ["tp", "precip", "precipitation", "rain"], str(args.era5_precip_col))

    keep = [time_col]
    rename = {time_col: "valid_time"}
    if station_col is not None:
        keep.append(station_col)
        rename[station_col] = "station_sn"
    for src, dst in [(tcc_col, "era5_tcc"), (rad_col, "era5_radiation"), (precip_col, "era5_precip")]:
        if src is not None:
            keep.append(src)
            rename[src] = dst
    e = era5[keep].copy().rename(columns=rename)
    e["valid_time"] = pd.to_datetime(e["valid_time"])
    if "station_sn" in e.columns:
        e["station_sn"] = e["station_sn"].astype(str)
        merged = context.merge(e, on=["station_sn", "valid_time"], how="left")
    else:
        e = e.groupby("valid_time", as_index=False).mean(numeric_only=True)
        merged = context.merge(e, on="valid_time", how="left")
    return merged


def normalize_tcc(tcc: pd.Series) -> pd.Series:
    values = pd.to_numeric(tcc, errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    if float(finite.quantile(0.95)) > 1.5:
        values = values / 100.0
    return values.clip(0.0, 1.0)


def pv_clear_sky_condition(out: pd.DataFrame) -> pd.DataFrame:
    truth = pd.to_numeric(out["true_pv_norm"], errors="coerce").astype(float)
    if "clearsky_ghi" in out.columns and "station_clearsky_q95" in out.columns:
        clear = pd.to_numeric(out["clearsky_ghi"], errors="coerce").astype(float)
        clear_q95 = pd.to_numeric(out["station_clearsky_q95"], errors="coerce").astype(float)
        clear_norm = clear / clear_q95.where(clear_q95 > 1.0)
    else:
        clear_norm = pd.Series(np.nan, index=out.index, dtype=float)
    pv_q95 = pd.to_numeric(out.get("station_pv_q95", np.nan), errors="coerce").astype(float)
    expected_clear_pv = pv_q95 * clear_norm
    csi = truth / expected_clear_pv.where(expected_clear_pv > 0.02)
    out["pv_clear_index"] = csi.clip(0.0, 2.0)
    valid = out["pv_clear_index"].notna()
    out.loc[valid & (out["pv_clear_index"] >= 0.75), "weather_condition"] = "PV-clear"
    out.loc[valid & (out["pv_clear_index"] >= 0.35) & (out["pv_clear_index"] < 0.75), "weather_condition"] = "PV-variable"
    out.loc[valid & (out["pv_clear_index"] < 0.35), "weather_condition"] = "PV-cloudy"
    out.loc[valid, "weather_rule"] = "true_pv_over_station_clear_sky_proxy"
    return out


def quantile_condition(out: pd.DataFrame, value_col: str, prefix: str, rule: str) -> pd.DataFrame:
    values = pd.to_numeric(out[value_col], errors="coerce").astype(float)
    valid = values.notna()
    if int(valid.sum()) < 10:
        return out
    q1, q2 = np.nanquantile(values[valid].to_numpy(np.float64), [1.0 / 3.0, 2.0 / 3.0])
    out.loc[valid & (values >= q2), "weather_condition"] = f"{prefix}-high"
    out.loc[valid & (values >= q1) & (values < q2), "weather_condition"] = f"{prefix}-mid"
    out.loc[valid & (values < q1), "weather_condition"] = f"{prefix}-low"
    out.loc[valid, "weather_rule"] = rule
    return out


def simple_kmeans(features: np.ndarray, k: int, seed: int, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(features, dtype=np.float64)
    valid = np.all(np.isfinite(x), axis=1)
    labels = np.full(x.shape[0], -1, dtype=int)
    if int(valid.sum()) < max(3, int(k)):
        return labels
    xv = x[valid]
    mean = np.nanmean(xv, axis=0)
    std = np.nanstd(xv, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    z = (xv - mean) / std
    init_idx = rng.choice(np.arange(z.shape[0]), size=int(k), replace=False)
    centers = z[init_idx].copy()
    lab = np.zeros(z.shape[0], dtype=int)
    for _ in range(int(iters)):
        dist = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_lab = np.argmin(dist, axis=1)
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
        for j in range(int(k)):
            if np.any(lab == j):
                centers[j] = np.mean(z[lab == j], axis=0)
    labels[np.where(valid)[0]] = lab
    return labels


def cluster_condition(out: pd.DataFrame, feature_cols: list[str], score_col: str, prefix: str, args: argparse.Namespace) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in out.columns]
    if not cols:
        return out
    features = np.column_stack([pd.to_numeric(out[c], errors="coerce").astype(float).to_numpy() for c in cols])
    labels = simple_kmeans(features, int(args.cluster_k), int(args.cluster_seed))
    out[f"{prefix}_cluster_id"] = labels
    valid = labels >= 0
    if not np.any(valid):
        return out
    score = pd.to_numeric(out[score_col], errors="coerce").astype(float) if score_col in out.columns else pd.Series(-labels, index=out.index)
    cluster_scores = []
    for lab in sorted(set(labels[valid])):
        cluster_scores.append((lab, float(score[labels == lab].mean())))
    ordered = [lab for lab, _ in sorted(cluster_scores, key=lambda item: item[1], reverse=True)]
    names = ["clear-like", "variable", "cloudy-like", "cluster4", "cluster5"]
    mapping = {lab: f"{prefix}-{names[i] if i < len(names) else f'cluster{i + 1}'}" for i, lab in enumerate(ordered)}
    out.loc[valid, "weather_condition"] = [mapping[int(v)] for v in labels[valid]]
    out.loc[valid, "weather_rule"] = f"{prefix}_kmeans"
    return out


def assign_external_era5_threshold(out: pd.DataFrame) -> pd.DataFrame:
    if "era5_tcc" in out.columns and out["era5_tcc"].notna().any():
        tcc = normalize_tcc(out["era5_tcc"])
        out["era5_tcc_norm"] = tcc
        out.loc[tcc <= 0.30, "weather_condition"] = "Ext-ERA5-clear"
        out.loc[(tcc > 0.30) & (tcc <= 0.70), "weather_condition"] = "Ext-ERA5-partly cloudy"
        out.loc[tcc > 0.70, "weather_condition"] = "Ext-ERA5-overcast"
        out.loc[tcc.notna(), "weather_rule"] = "external_era5_tcc"
        return out
    if "era5_radiation" in out.columns:
        return quantile_condition(out, "era5_radiation", "Ext-ERA5-radiation", "external_era5_radiation_tertile")
    return out


def assign_station_era5_threshold(out: pd.DataFrame) -> pd.DataFrame:
    if "tcc" in out.columns and out["tcc"].notna().any():
        tcc = normalize_tcc(out["tcc"])
        out["station_era5_tcc_norm"] = tcc
        out.loc[tcc <= 0.30, "weather_condition"] = "ERA5-clear"
        out.loc[(tcc > 0.30) & (tcc <= 0.70), "weather_condition"] = "ERA5-partly cloudy"
        out.loc[tcc > 0.70, "weather_condition"] = "ERA5-overcast"
        out.loc[tcc.notna(), "weather_rule"] = "station_era5_tcc"
        return out
    if "radiation" in out.columns:
        return quantile_condition(out, "radiation", "ERA5-radiation", "station_era5_radiation_tertile")
    return out


def assign_weather_conditions(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["weather_condition"] = "Unknown"
    out["weather_rule"] = "missing"

    source = str(args.condition_source)
    if source == "station_era5_threshold":
        out = assign_station_era5_threshold(out)
    elif source == "station_era5_cluster":
        out = assign_station_era5_threshold(out)
        out = cluster_condition(
            out,
            ["station_era5_tcc_norm", "radiation", "clearsky_ghi", "humidity", "temperature", "speed10"],
            "radiation",
            "ERA5",
            args,
        )
    elif source == "pv_clear_sky":
        out = pv_clear_sky_condition(out)
    elif source == "pv_quantile":
        out = quantile_condition(out, "true_pv_norm", "PV", "true_pv_quantile")
    elif source == "pv_cluster":
        out = pv_clear_sky_condition(out)
        out = cluster_condition(out, ["true_pv_norm", "pv_clear_index", "true_delta_pv_norm"], "pv_clear_index", "PV", args)
    elif source == "external_era5_threshold":
        if args.era5_csv is None:
            raise ValueError("--condition_source external_era5_threshold requires --era5_csv")
        out = assign_external_era5_threshold(out)
    elif source == "external_era5_cluster":
        if args.era5_csv is None:
            raise ValueError("--condition_source external_era5_cluster requires --era5_csv")
        out = assign_external_era5_threshold(out)
        out = cluster_condition(out, ["era5_tcc_norm", "era5_radiation", "era5_precip"], "era5_radiation", "Ext-ERA5", args)
    else:
        raise ValueError(f"Unknown condition source: {source}")

    return out


def to_long(details: dict[str, pd.DataFrame], specs: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    labels = {spec["key"]: spec["method"] for spec in specs}
    for key, df in details.items():
        keep = [
            c
            for c in [
                *KEY_COLS,
                "horizon",
                "pred_pv_norm",
                "true_pv_norm",
                "true_delta_pv_norm",
                "large_change",
                "capacity_raw",
                "capacity_scale",
            ]
            if c in df.columns
        ]
        sub = df[keep].copy()
        sub["method_key"] = key
        sub["method"] = labels.get(key, key)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def metric_row(df: pd.DataFrame, group_cols: dict[str, Any]) -> dict[str, Any]:
    err = df["pred_pv_norm"].astype(float) - df["true_pv_norm"].astype(float)
    row = {
        **group_cols,
        "n_points": int(len(df)),
        "n_stations": int(df["station_sn"].nunique()),
        "mae_pv_norm": float(err.abs().mean()),
        "rmse_pv_norm": float(math.sqrt(float((err**2).mean()))),
        "bias_pv_norm": float(err.mean()),
        "p95_abs_err_pv_norm": float(err.abs().quantile(0.95)),
    }
    row.update(regression_metric_extras(df, "pred_pv_norm"))
    return row


def summarize(long_df: pd.DataFrame, min_group_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (condition, method_key, method), g in long_df.groupby(["weather_condition", "method_key", "method"], sort=False):
        if len(g) >= int(min_group_points):
            rows.append(metric_row(g, {"weather_condition": condition, "method_key": method_key, "method": method}))
    condition_summary = pd.DataFrame(rows)

    rows = []
    for (condition, horizon, method_key, method), g in long_df.groupby(["weather_condition", "horizon", "method_key", "method"], sort=False):
        if len(g) >= int(min_group_points):
            rows.append(
                metric_row(
                    g,
                    {
                        "weather_condition": condition,
                        "horizon": horizon,
                        "method_key": method_key,
                        "method": method,
                    },
                )
            )
    horizon_summary = pd.DataFrame(rows)
    return condition_summary, horizon_summary


def add_gain_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "temporal_baseline" not in set(summary["method_key"].astype(str)):
        return summary
    key_cols = [c for c in ["weather_condition", "horizon"] if c in summary.columns]
    base = summary[summary["method_key"] == "temporal_baseline"][
        [*key_cols, "mae_pv_norm", "rmse_pv_norm", "r2_pv_norm"]
    ].rename(
        columns={
            "mae_pv_norm": "baseline_mae_pv_norm",
            "rmse_pv_norm": "baseline_rmse_pv_norm",
            "r2_pv_norm": "baseline_r2_pv_norm",
        }
    )
    out = summary.merge(base, on=key_cols, how="left")
    out["mae_gain_vs_temporal"] = out["baseline_mae_pv_norm"] - out["mae_pv_norm"]
    out["rmse_gain_vs_temporal"] = out["baseline_rmse_pv_norm"] - out["rmse_pv_norm"]
    out["r2_gain_vs_temporal"] = out["r2_pv_norm"] - out["baseline_r2_pv_norm"]
    return out


def condition_counts(weather_context: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["weather_condition", "weather_rule"]
    rows = []
    total = max(int(len(weather_context)), 1)
    for keys, g in weather_context.groupby(group_cols, sort=False):
        rows.append(
            {
                "weather_condition": keys[0],
                "weather_rule": keys[1],
                "n_points": int(len(g)),
                "percentage": float(len(g) / total * 100.0),
                "n_stations": int(g["station_sn"].nunique()),
                "mean_radiation": float(pd.to_numeric(g.get("radiation"), errors="coerce").mean()) if "radiation" in g else float("nan"),
                "mean_tcc": float(pd.to_numeric(g.get("tcc"), errors="coerce").mean()) if "tcc" in g else float("nan"),
                "mean_station_era5_tcc_norm": (
                    float(pd.to_numeric(g.get("station_era5_tcc_norm"), errors="coerce").mean())
                    if "station_era5_tcc_norm" in g
                    else float("nan")
                ),
                "mean_pv_clear_index": (
                    float(pd.to_numeric(g.get("pv_clear_index"), errors="coerce").mean())
                    if "pv_clear_index" in g
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def print_markdown(summary: pd.DataFrame, digits: int) -> None:
    print("\n=== Stage 4 Weather-Condition Summary ===")
    cols = ["weather_condition", "method", "n_points", "mae_pv_norm", "rmse_pv_norm", "r2_pv_norm", "mae_gain_vs_temporal"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        cells = []
        for col in cols:
            value = row.get(col, "")
            if col in {"mae_pv_norm", "rmse_pv_norm", "r2_pv_norm", "mae_gain_vs_temporal"}:
                cells.append("" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        print("| " + " | ".join(cells) + " |")


def plot_condition_mae(summary: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    if summary.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skip plot: {exc}", flush=True)
        return

    preferred = [
        "ERA5-clear",
        "ERA5-partly cloudy",
        "ERA5-overcast",
        "PV-clear",
        "PV-variable",
        "PV-cloudy",
        "ERA5-radiation-high",
        "ERA5-radiation-mid",
        "ERA5-radiation-low",
        "PV-high",
        "PV-mid",
        "PV-low",
    ]
    conditions = [c for c in preferred if c in set(summary["weather_condition"])]
    if not conditions:
        conditions = list(summary["weather_condition"].drop_duplicates())
    methods = list(summary["method"].drop_duplicates())
    colors = {
        "Temporal baseline": "#516b7c",
        "+ Future SWR patch": "#2b9e93",
        "+ Historical PV-field patch": "#7aa14a",
        "FG-MSF": "#df493d",
    }
    x = np.arange(len(conditions), dtype=float)
    width = min(0.18, 0.78 / max(1, len(methods)))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(conditions)), 4.0))
    for i, method in enumerate(methods):
        vals = []
        for cond in conditions:
            sub = summary[(summary["weather_condition"] == cond) & (summary["method"] == method)]
            vals.append(float(sub["mae_pv_norm"].iloc[0]) if not sub.empty else np.nan)
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=method, color=colors.get(method, None), alpha=0.92)
    ax.set_ylabel("MAE (p.u.)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "stage4_weather_condition_mae.png", dpi=int(args.dpi))
    fig.savefig(out_dir / "stage4_weather_condition_mae.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    specs = list(METHOD_SPECS)
    for extra in parse_extra_specs(list(args.extra_detail_csv)):
        specs.append({"key": extra["key"], "method": extra["method"], "path_arg": None, "path": extra["path"]})

    details = {}
    for spec in specs:
        path = Path(spec["path"]) if spec.get("path_arg") is None else Path(getattr(args, spec["path_arg"]))
        details[spec["key"]] = read_detail(path, spec["key"], spec["method"], str(args.split))

    details = align_details(details, str(args.align_mode))
    if not details:
        raise RuntimeError("No details loaded.")

    reference = details["temporal_baseline"] if "temporal_baseline" in details else next(iter(details.values()))
    weather_context = build_weather_context(key_frame(reference), args)
    long_df = to_long(details, specs)
    long_df = long_df.merge(
        weather_context.drop(columns=["eval_split", "horizon", "true_pv_norm"], errors="ignore"),
        on=["station_sn", "issue_time", "h"],
        how="left",
    )
    long_df = long_df[long_df["weather_condition"].notna()].copy()

    condition_summary, horizon_summary = summarize(long_df, int(args.min_group_points))
    condition_summary = add_gain_vs_baseline(condition_summary)
    horizon_summary = add_gain_vs_baseline(horizon_summary)
    counts = condition_counts(weather_context)

    long_df.to_csv(out_dir / "stage4_weather_condition_aligned_detail_long.csv", index=False)
    weather_context.to_csv(out_dir / "stage4_weather_context.csv", index=False)
    counts.to_csv(out_dir / "stage4_weather_condition_counts.csv", index=False)
    condition_summary.to_csv(out_dir / "stage4_weather_condition_summary.csv", index=False)
    horizon_summary.to_csv(out_dir / "stage4_weather_condition_horizon_summary.csv", index=False)
    plot_condition_mae(condition_summary, out_dir, args) if int(args.plot) else None

    write_json(
        out_dir / "stage4_weather_condition_analysis_meta.json",
        {
            "split": str(args.split),
            "align_mode": str(args.align_mode),
            "valid_minutes_per_horizon": int(args.valid_minutes_per_horizon),
            "min_group_points": int(args.min_group_points),
            "condition_source": str(args.condition_source),
            "methods": [{"key": spec["key"], "method": spec["method"]} for spec in specs],
            "note": "Station-level historical weather columns are treated as ERA5/reanalysis variables.",
            "recommended_sources": ["station_era5_threshold", "station_era5_cluster", "pv_clear_sky", "pv_cluster"],
        },
    )

    print("==== Stage 4 weather-condition analysis ====")
    print(f"aligned long rows: {len(long_df)}")
    print("\n=== Weather condition counts ===")
    print(counts.to_string(index=False) if not counts.empty else "EMPTY")
    if not counts.empty:
        print("\n=== Weather condition LaTeX-ready rows ===")
        for _, row in counts.iterrows():
            print(
                f"{row['weather_condition']} & {int(row['n_points'])} & "
                f"{float(row['percentage']):.2f}\\% & {int(row['n_stations'])} \\\\"
            )
    print_markdown(condition_summary, int(args.digits))
    print(f"\n[write] {out_dir / 'stage4_weather_condition_summary.csv'}")
    print(f"[write] {out_dir / 'stage4_weather_condition_horizon_summary.csv'}")
    print(f"[write] {out_dir / 'stage4_weather_condition_mae.png'}")


if __name__ == "__main__":
    main()
