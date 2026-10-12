#!/usr/bin/env python3
"""Build the Stage 4 main station-forecasting table on a common aligned subset."""

from __future__ import annotations

import argparse
import csv
import math
import re
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


DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
CORE_DETAIL_COLS = [
    "station_sn",
    "eval_split",
    "issue_time",
    "h",
    "horizon",
    "pred_pv_norm",
    "true_pv_norm",
    "true_delta_pv_norm",
    "large_change",
]
CAPACITY_DETAIL_COLS = ["capacity_raw", "capacity_scale"]


METHOD_SPECS = [
    {
        "model": "temporal_baseline",
        "method": "Temporal baseline",
        "station_history": "yes",
        "nwp": "yes",
        "future_swr": "no",
        "pv_field_patch": "no",
    },
    {
        "model": "future_swr_patch",
        "method": "+ future SWR patch",
        "station_history": "yes",
        "nwp": "yes",
        "future_swr": "yes",
        "pv_field_patch": "no",
    },
    {
        "model": "historical_pv_field_patch",
        "method": "+ historical PV-field patch",
        "station_history": "yes",
        "nwp": "yes",
        "future_swr": "no",
        "pv_field_patch": "historical",
    },
    {
        "model": "future_pv_field_patch",
        "method": "Proposed",
        "station_history": "yes",
        "nwp": "yes",
        "future_swr": "via SAG-PVFF",
        "pv_field_patch": "future predicted",
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
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=RUNS_DIR / "11_eval_stage4_main_table",
    )
    parser.add_argument("--large_delta_threshold", type=float, default=0.10)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--horizon", type=str, default="all")
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def safe_label(label: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "model"


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def history_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")])


def read_detail(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing eval-detail CSV for {label}: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = [c for c in CORE_DETAIL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    df["h"] = df["h"].astype(int)
    df["horizon"] = df["horizon"].astype(str)
    df["large_change"] = coerce_bool(df["large_change"])
    df["model_label"] = label
    return attach_capacity_from_station_meta(df, path)


def truth_variant_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in [
            "field_patch_center_pv_norm",
            "field_patch_mean_pv_norm",
            "field_patch_support_mean",
        ]
        if c in df.columns
    ]


def align_details(details: dict[str, pd.DataFrame], large_delta_threshold: float) -> pd.DataFrame:
    baseline = details["temporal_baseline"]
    base_cols = [
        "station_sn",
        "eval_split",
        "issue_time",
        "h",
        "horizon",
        "true_pv_norm",
        "true_delta_pv_norm",
        "large_change",
        "pred_pv_norm",
    ]
    base_cols = list(dict.fromkeys(base_cols + [c for c in CAPACITY_DETAIL_COLS if c in baseline.columns] + history_cols(baseline)))
    merged = baseline[[c for c in base_cols if c in baseline.columns]].rename(
        columns={"pred_pv_norm": "pred_temporal_baseline"}
    )
    merged["large_change"] = coerce_bool(merged["large_change"])

    for label in ["future_swr_patch", "historical_pv_field_patch", "future_pv_field_patch"]:
        df = details[label]
        tag = safe_label(label)
        cols = list(dict.fromkeys([c for c in CORE_DETAIL_COLS if c != "horizon"] + truth_variant_cols(df)))
        cols = [c for c in cols if c in df.columns]
        right = df[cols].copy().rename(
            columns={
                "pred_pv_norm": f"pred_{tag}",
                "true_pv_norm": f"true_{tag}",
                "true_delta_pv_norm": f"true_delta_{tag}",
                "large_change": f"large_change_{tag}",
                "field_patch_center_pv_norm": f"field_center_{tag}",
                "field_patch_mean_pv_norm": f"field_mean_{tag}",
                "field_patch_support_mean": f"field_support_{tag}",
            }
        )
        before = len(merged)
        merged = merged.merge(right, on=KEY_COLS, how="inner", validate="one_to_one")
        if len(merged) != before:
            print(f"[warn] alignment with {label}: rows {before} -> {len(merged)}", flush=True)
        truth_diff = (merged["true_pv_norm"].astype(float) - merged[f"true_{tag}"].astype(float)).abs()
        max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
        if np.isfinite(max_diff) and max_diff > 1.0e-5:
            print(f"[warn] max truth mismatch for {label}: {max_diff:.6g}", flush=True)
        merged["large_change"] = merged["large_change"] | coerce_bool(merged[f"large_change_{tag}"])

    if merged["large_change"].isna().any():
        merged["large_change"] = merged["true_delta_pv_norm"].astype(float).abs() >= float(large_delta_threshold)
    for model in [spec["model"] for spec in METHOD_SPECS]:
        pred_col = f"pred_{safe_label(model)}"
        if pred_col not in merged.columns:
            continue
        err = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        merged[f"err_{safe_label(model)}"] = err
        merged[f"abs_err_{safe_label(model)}"] = err.abs()
        merged[f"sq_err_{safe_label(model)}"] = err**2
    return merged


def metric_row(df: pd.DataFrame, model: str, scope: dict[str, Any]) -> dict[str, Any]:
    tag = safe_label(model)
    pred_col = f"pred_{tag}"
    err = df[pred_col].astype(float) - df["true_pv_norm"].astype(float)
    abs_err = err.abs()
    sq_err = err**2
    large = df["large_change"].astype(bool)
    hist_cols = history_cols(df)
    if hist_cols:
        hist_last = df[hist_cols[-1]].astype(float)
        pred_delta = df[pred_col].astype(float) - hist_last
        true_delta = df["true_delta_pv_norm"].astype(float)
        valid_delta = np.isfinite(pred_delta) & np.isfinite(true_delta)
        sign_valid = valid_delta & (true_delta.abs() >= 1.0e-6)
        delta_sign_accuracy = (
            float((np.sign(pred_delta[sign_valid]) == np.sign(true_delta[sign_valid])).mean())
            if bool(sign_valid.any())
            else float("nan")
        )
        delta_corr = (
            float(np.corrcoef(pred_delta[valid_delta], true_delta[valid_delta])[0, 1])
            if int(valid_delta.sum()) >= 3
            else float("nan")
        )
        delta_mae = float((pred_delta[valid_delta] - true_delta[valid_delta]).abs().mean()) if bool(valid_delta.any()) else float("nan")
    else:
        delta_sign_accuracy = float("nan")
        delta_corr = float("nan")
        delta_mae = float("nan")
    row = {
        "model": model,
        "n_points": int(len(df)),
        "mae_pv_norm": float(abs_err.mean()),
        "rmse_pv_norm": float(math.sqrt(float(sq_err.mean()))),
        "bias_pv_norm": float(err.mean()),
        "p95_abs_err_pv_norm": float(abs_err.quantile(0.95)),
        "max_abs_err_pv_norm": float(abs_err.max()),
        "delta_mae_pv_norm": delta_mae,
        "delta_sign_accuracy": delta_sign_accuracy,
        "delta_corr": delta_corr,
        "large_change_mae_pv_norm": float(abs_err[large].mean()) if bool(large.any()) else float("nan"),
        "large_change_n": int(large.sum()),
        **regression_metric_extras(df, pred_col),
    }
    row.update(scope)
    return row


def build_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = [spec["model"] for spec in METHOD_SPECS]
    for (split, horizon), g in merged.groupby(["eval_split", "horizon"], sort=True):
        for model in models:
            rows.append(metric_row(g, model, {"eval_split": split, "horizon": horizon, "dimension": "horizon"}))
    for split, g in merged.groupby("eval_split", sort=True):
        for model in models:
            rows.append(metric_row(g, model, {"eval_split": split, "horizon": "all", "dimension": "split"}))
    rows.extend(metric_row(merged, model, {"eval_split": "all_splits", "horizon": "all", "dimension": "all"}) for model in models)
    return pd.DataFrame(rows)


def add_main_metadata(summary: pd.DataFrame, split: str, horizon: str) -> pd.DataFrame:
    work = summary[(summary["eval_split"].astype(str) == split) & (summary["horizon"].astype(str) == horizon)].copy()
    meta = pd.DataFrame(METHOD_SPECS)
    out = meta.merge(work, on="model", how="left", validate="one_to_one")
    out["mae_gain_vs_baseline"] = np.nan
    out["rmse_gain_vs_baseline"] = np.nan
    base = out[out["model"] == "temporal_baseline"]
    if not base.empty:
        base_mae = float(base["mae_pv_norm"].iloc[0])
        base_rmse = float(base["rmse_pv_norm"].iloc[0])
        out["mae_gain_vs_baseline"] = base_mae - out["mae_pv_norm"].astype(float)
        out["rmse_gain_vs_baseline"] = base_rmse - out["rmse_pv_norm"].astype(float)
    return out


def format_value(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def print_markdown(df: pd.DataFrame, digits: int) -> None:
    print("\n=== Stage 4 Main Station-Forecasting Table ===")
    cols = [
        ("Method", "method"),
        ("Station Hist.", "station_history"),
        ("NWP", "nwp"),
        ("Future SWR", "future_swr"),
        ("PV Field Patch", "pv_field_patch"),
        ("MAE", "mae_pv_norm"),
        ("RMSE", "rmse_pv_norm"),
        ("R2", "r2_pv_norm"),
        ("ACC", "acc_installed_capacity"),
        ("MAE Gain", "mae_gain_vs_baseline"),
    ]
    print("| " + " | ".join(label for label, _ in cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        cells = []
        for _label, key in cols:
            if key in {"mae_pv_norm", "rmse_pv_norm", "r2_pv_norm", "acc_installed_capacity", "mae_gain_vs_baseline"}:
                cells.append(format_value(row.get(key, np.nan), digits))
            else:
                cells.append(str(row.get(key, "")))
        print("| " + " | ".join(cells) + " |")


def csv_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if any(ch in text for ch in [",", "\"", "\n"]):
        text = "\"" + text.replace("\"", "\"\"") + "\""
    return text


def print_csv_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} CSV ===")
    cols = list(df.columns)
    print(",".join(cols))
    for _, row in df.iterrows():
        print(",".join(csv_escape(row[col]) for col in cols))


def main() -> None:
    args = parse_args()
    details = {
        "temporal_baseline": read_detail(args.baseline_detail_csv, "temporal_baseline"),
        "future_swr_patch": read_detail(args.future_swr_patch_detail_csv, "future_swr_patch"),
        "historical_pv_field_patch": read_detail(args.historical_field_detail_csv, "historical_pv_field_patch"),
        "future_pv_field_patch": read_detail(args.future_field_detail_csv, "future_pv_field_patch"),
    }
    merged = align_details(details, float(args.large_delta_threshold))
    summary = build_summary(merged)
    main_table = add_main_metadata(summary, str(args.split), str(args.horizon))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "stage4_main_aligned_detail.csv", index=False)
    summary.to_csv(out_dir / "stage4_main_eval_summary.csv", index=False)
    main_table.to_csv(out_dir / "stage4_main_table.csv", index=False)

    print("==== Stage 4 main table eval ====")
    print(f"baseline detail         : {args.baseline_detail_csv}")
    print(f"future SWR patch detail : {args.future_swr_patch_detail_csv}")
    print(f"historical field detail : {args.historical_field_detail_csv}")
    print(f"future field detail     : {args.future_field_detail_csv}")
    print(f"aligned rows            : {len(merged)}")
    print_markdown(main_table, int(args.digits))
    print_csv_block("Stage 4 Main Table", main_table)
    print(f"\n[write] {out_dir / 'stage4_main_table.csv'}")
    print(f"[write] {out_dir / 'stage4_main_eval_summary.csv'}")
    print(f"[write] {out_dir / 'stage4_main_aligned_detail.csv'}")


if __name__ == "__main__":
    main()

