#!/usr/bin/env python3
"""Collect an extended Stage 4 baseline table for station-level PV forecasting."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras
from stage4_paths import (
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)

BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
FIELD_DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]


METHOD_SPECS = [
    {
        "key": "lstm",
        "method": "LSTM",
        "spatial_type": "Station-only",
        "path_attr": "lstm_detail_csv",
    },
    {
        "key": "bilstm",
        "method": "BiLSTM",
        "spatial_type": "Station-only",
        "path_attr": "bilstm_detail_csv",
    },
    {
        "key": "tcn",
        "method": "TCN",
        "spatial_type": "Station-only",
        "path_attr": "tcn_detail_csv",
    },
    {
        "key": "transformer",
        "method": "Transformer",
        "spatial_type": "Station-only",
        "path_attr": "transformer_detail_csv",
    },
    {
        "key": "patchtst",
        "method": "PatchTST",
        "spatial_type": "Station-only",
        "path_attr": "patchtst_detail_csv",
    },
    {
        "key": "lstm_nwp",
        "method": "LSTM + forecast weather",
        "spatial_type": "Station + weather forecast",
        "path_attr": "lstm_nwp_detail_csv",
    },
    {
        "key": "bilstm_nwp",
        "method": "BiLSTM + forecast weather",
        "spatial_type": "Station + weather forecast",
        "path_attr": "bilstm_nwp_detail_csv",
    },
    {
        "key": "tcn_nwp",
        "method": "TCN + forecast weather",
        "spatial_type": "Station + weather forecast",
        "path_attr": "tcn_nwp_detail_csv",
    },
    {
        "key": "transformer_nwp",
        "method": "Transformer + forecast weather",
        "spatial_type": "Station + weather forecast",
        "path_attr": "transformer_nwp_detail_csv",
    },
    {
        "key": "patchtst_nwp",
        "method": "PatchTST + forecast weather",
        "spatial_type": "Station + weather forecast",
        "path_attr": "patchtst_nwp_detail_csv",
    },
    {
        "key": "gcn_lstm",
        "method": "GCN-LSTM",
        "spatial_type": "Implicit multi-site",
        "path_attr": "gcn_lstm_detail_csv",
    },
    {
        "key": "stgcn",
        "method": "STGCN",
        "spatial_type": "Implicit multi-site",
        "path_attr": "stgcn_detail_csv",
    },
    {
        "key": "stgcn_forecast_weather",
        "method": "STGCN + forecast weather",
        "spatial_type": "Implicit multi-site + weather forecast",
        "path_attr": "stgcn_forecast_weather_detail_csv",
    },
    {
        "key": "agcrn_nwp",
        "method": "AGCRN + forecast weather",
        "spatial_type": "Implicit multi-site + weather forecast",
        "path_attr": "agcrn_nwp_detail_csv",
    },
    {
        "key": "graph_wavenet_nwp",
        "method": "Graph WaveNet + forecast weather",
        "spatial_type": "Implicit multi-site + weather forecast",
        "path_attr": "graph_wavenet_nwp_detail_csv",
    },
    {
        "key": "agcrn_fgmsf_patch",
        "method": "AGCRN + FG-MSF field",
        "spatial_type": "Graph + predicted future PV field",
        "path_attr": "agcrn_fgmsf_patch_detail_csv",
    },
    {
        "key": "temporal_baseline",
        "method": "Temporal baseline",
        "spatial_type": "Station + NWP",
        "path_attr": "temporal_detail_csv",
    },
    {
        "key": "future_swr_patch",
        "method": "+ Future SWR patch",
        "spatial_type": "Environmental field",
        "path_attr": "future_swr_patch_detail_csv",
    },
    {
        "key": "historical_pv_field_patch",
        "method": "+ Historical PV-field patch",
        "spatial_type": "Historical generation field",
        "path_attr": "historical_field_detail_csv",
    },
    {
        "key": "fg_msf",
        "method": "FG-MSF",
        "spatial_type": "Predicted future PV field",
        "path_attr": "future_field_detail_csv",
    },
    {
        "key": "nearest3_station_pv",
        "method": "+ Nearest-3 station PV",
        "spatial_type": "Neighbor station PV",
        "path_attr": "nearest3_station_pv_detail_csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ["lstm", "bilstm", "tcn", "transformer", "patchtst", "gcn_lstm", "stgcn"]:
        parser.add_argument(
            f"--{model}_detail_csv",
            type=Path,
            default=RUNS_DIR / f"16_{model}_baseline_seed2026" / f"{model}_eval_detail.csv",
        )
    for model in ["lstm", "bilstm", "tcn", "transformer", "patchtst"]:
        parser.add_argument(
            f"--{model}_nwp_detail_csv",
            type=Path,
            default=RUNS_DIR / f"19_{model}_nwp_baseline_seed2026" / f"{model}_nwp_eval_detail.csv",
        )
    parser.add_argument(
        "--stgcn_forecast_weather_detail_csv",
        type=Path,
        default=RUNS_DIR / "20_stgcn_forecast_weather_baseline_seed2026" / "stgcn_forecast_weather_eval_detail.csv",
    )
    parser.add_argument(
        "--agcrn_nwp_detail_csv",
        type=Path,
        default=RUNS_DIR / "23_agcrn_nwp_baseline_seed2026" / "agcrn_nwp_eval_detail.csv",
    )
    parser.add_argument(
        "--graph_wavenet_nwp_detail_csv",
        type=Path,
        default=RUNS_DIR / "23_graph_wavenet_nwp_baseline_seed2026" / "graph_wavenet_nwp_eval_detail.csv",
    )
    parser.add_argument(
        "--agcrn_fgmsf_patch_detail_csv",
        type=Path,
        default=RUNS_DIR / "27_agcrn_fgmsf_patch_seed2026" / "agcrn_fgmsf_patch_eval_detail.csv",
    )
    parser.add_argument(
        "--temporal_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--nearest3_station_pv_detail_csv",
        type=Path,
        default=RUNS_DIR / "21_nearest3_station_pv_baseline_seed2026" / "nearest3_station_pv_eval_detail.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "17_eval_stage4_extended_baseline_table")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--horizon", type=str, default="all", help="all or h1/h2/...")
    parser.add_argument(
        "--align_mode",
        choices=["intersection", "reference", "none"],
        default="intersection",
        help="intersection: all loaded methods share identical keys; reference: align each method to temporal baseline keys; none: evaluate each file as-is.",
    )
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--strict_missing", type=int, default=0)
    return parser.parse_args()


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def read_detail(path: Path, key: str, strict: bool) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        message = f"[skip] {key}: missing {path}"
        if strict:
            raise FileNotFoundError(message)
        print(message, flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [*KEY_COLS, "horizon", "pred_pv_norm", "true_pv_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = attach_capacity_from_station_meta(df, path)
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["horizon"] = out["horizon"].astype(str)
    if "large_change" in out.columns:
        out["large_change"] = coerce_bool(out["large_change"])
    out["method_key"] = key
    print(f"[read] {key}: {path} rows={len(out)}", flush=True)
    return out


def filter_scope(df: pd.DataFrame, split: str, horizon: str) -> pd.DataFrame:
    out = df[df["eval_split"].astype(str) == str(split)].copy()
    if str(horizon) != "all":
        out = out[out["horizon"].astype(str) == str(horizon)].copy()
    return out


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEY_COLS].drop_duplicates().copy()


def align_details(details: dict[str, pd.DataFrame], mode: str) -> dict[str, pd.DataFrame]:
    if mode == "none" or not details:
        return details
    if mode == "reference":
        if "temporal_baseline" not in details:
            raise ValueError("--align_mode reference requires temporal_baseline detail")
        keys = key_frame(details["temporal_baseline"])
    else:
        keys = None
        for df in details.values():
            k = key_frame(df)
            keys = k if keys is None else keys.merge(k, on=KEY_COLS, how="inner")
    out = {}
    assert keys is not None
    for key, df in details.items():
        before = len(df)
        out[key] = df.merge(keys, on=KEY_COLS, how="inner")
        if len(out[key]) != before:
            print(f"[align] {key}: rows {before} -> {len(out[key])}", flush=True)
    return out


def metric_row(df: pd.DataFrame, spec: dict[str, str]) -> dict[str, Any]:
    pred = df["pred_pv_norm"].astype(float)
    truth = df["true_pv_norm"].astype(float)
    err = pred - truth
    row = {
        "Method": spec["method"],
        "Spatial type": spec["spatial_type"],
        "n_points": int(len(df)),
        "n_stations": int(df["station_sn"].nunique()) if "station_sn" in df.columns else 0,
        "MAE": float(err.abs().mean()),
        "RMSE": float(math.sqrt(float((err**2).mean()))),
        "Bias": float(err.mean()),
    }
    row["R2"] = float(regression_metric_extras(df, "pred_pv_norm")["r2_pv_norm"])
    return row


def format_value(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def print_markdown(table: pd.DataFrame, digits: int) -> None:
    print("\n=== Stage 4 Extended Baseline Table ===")
    cols = ["Method", "Spatial type", "MAE", "RMSE", "R2", "n_points"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        cells = []
        for col in cols:
            cells.append(format_value(row[col], digits) if col in {"MAE", "RMSE", "R2"} else str(row[col]))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    args = parse_args()
    strict = bool(int(args.strict_missing))
    details = {}
    loaded_specs = []
    for spec in METHOD_SPECS:
        path = getattr(args, spec["path_attr"])
        df = read_detail(path, spec["key"], strict)
        if df is None:
            continue
        df = filter_scope(df, str(args.split), str(args.horizon))
        if df.empty:
            print(f"[skip] {spec['key']}: empty after split/horizon filter", flush=True)
            continue
        details[spec["key"]] = df
        loaded_specs.append(spec)
    details = align_details(details, str(args.align_mode))

    rows = []
    for spec in loaded_specs:
        df = details.get(spec["key"])
        if df is not None and not df.empty:
            rows.append(metric_row(df, spec))
    table = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stage4_extended_baseline_table.csv", index=False)
    with (out_dir / "stage4_extended_baseline_table.md").open("w", encoding="utf-8") as f:
        cols = ["Method", "Spatial type", "MAE", "RMSE", "R2", "n_points"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in table.iterrows():
            f.write(
                "| "
                + " | ".join(format_value(row[col], int(args.digits)) if col in {"MAE", "RMSE", "R2"} else str(row[col]) for col in cols)
                + " |\n"
            )

    print("==== Stage 4 extended baseline eval ====")
    print(f"split={args.split} horizon={args.horizon} align_mode={args.align_mode}")
    print_markdown(table, int(args.digits))
    print(f"\n[write] {out_dir / 'stage4_extended_baseline_table.csv'}")
    print(f"[write] {out_dir / 'stage4_extended_baseline_table.md'}")


if __name__ == "__main__":
    main()
