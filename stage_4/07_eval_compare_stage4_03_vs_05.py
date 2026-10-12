"""Compare Stage 4 station baseline against the Stage 3 field-patch model.

This script is evaluation-only. It consumes the eval_detail CSV files written by:
  03_train_per_station_temporal_baseline.py
  05l_train_per_station_temporal_with_stage3_field_patch.py

The comparison is aligned by station, split, issue time, and forecast horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage4_paths import (
    MULTISOURCE_CACHE_DIR,
    PER_STATION_03_VS_05_EVAL_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    ensure_stage4_dirs,
)
from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras


BASELINE_MODEL = "03_temporal_baseline"
FIELD_PATCH_MODEL = "05_stage3_field_patch"
SWR_PATCH_MODEL = "06_stage3_swr_patch"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
MODEL_SPECS = [
    (BASELINE_MODEL, "pred_baseline", "err_baseline", "abs_err_baseline"),
    (FIELD_PATCH_MODEL, "pred_field_patch", "err_field_patch", "abs_err_field_patch"),
    (SWR_PATCH_MODEL, "pred_swr_patch", "err_swr_patch", "abs_err_swr_patch"),
]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / "per_station_temporal_baseline_eval_detail.csv",
        help="Eval-detail CSV from 03_train_per_station_temporal_baseline.py.",
    )
    parser.add_argument(
        "--field_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
        help="Eval-detail CSV from 05l_train_per_station_temporal_with_stage3_field_patch.py.",
    )
    parser.add_argument(
        "--swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
        help="Eval-detail CSV from 06l_train_per_station_temporal_with_stage3_swr_patch.py. Skipped if missing.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PER_STATION_03_VS_05_EVAL_DIR,
        help="Directory for comparison tables and sample figures.",
    )
    parser.add_argument(
        "--random_sample_count",
        type=int,
        default=36,
        help="Number of random test issue samples to visualize.",
    )
    parser.add_argument(
        "--worst_sample_count",
        type=int,
        default=18,
        help="Number of high-error test issue samples to visualize separately for diagnosis.",
    )
    parser.add_argument(
        "--multisource_sample_count",
        type=int,
        default=5,
        help="Number of random test issue samples to render with full PV/H8/NWP multisource inputs.",
    )
    parser.add_argument("--sample_count", type=int, default=None, help="Deprecated alias for --random_sample_count.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--multisource_cache_dir", type=Path, default=MULTISOURCE_CACHE_DIR)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--past_frames", type=int, default=12)
    parser.add_argument("--future_frames", type=int, default=8)
    parser.add_argument("--freq_minutes", type=int, default=15)
    parser.add_argument("--nwp_horizons", type=str, default="1,4,8")
    parser.add_argument("--gaussian_sigma_cells", type=float, default=1.0)
    parser.add_argument("--gaussian_radius_cells", type=int, default=2)
    parser.add_argument("--min_grid_weight", type=float, default=1e-6)
    parser.add_argument("--pv_field_clip", type=float, default=1.5)
    parser.add_argument(
        "--large_delta_threshold",
        type=float,
        default=0.1,
        help="Large-change threshold used when large_change is absent or needs recomputation.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required eval-detail CSV: {path}\n"
            "Please run the corresponding Stage 4 training/eval script first."
        )


def read_detail(path: Path, model_label: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = [c for c in CORE_DETAIL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    df["h"] = df["h"].astype(int)
    df["horizon"] = df["horizon"].astype(str)
    df["large_change"] = coerce_bool(df["large_change"])
    df["model_label"] = model_label
    df = attach_capacity_from_station_meta(df, path)
    return df


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    normalized = values.astype(str).str.lower().str.strip()
    return normalized.isin(["true", "1", "yes", "y"])


def optional_read_detail(path: Path, model_label: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        print(f"[warn] skip missing optional eval-detail CSV for {model_label}: {path}", flush=True)
        return None
    return read_detail(path, model_label)


def _variant_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in [
            "field_patch_center_pv_norm",
            "field_patch_mean_pv_norm",
            "field_patch_support_mean",
        ]
        if c in df.columns
    ]


def aligned_detail(
    baseline: pd.DataFrame,
    field_patch: pd.DataFrame,
    large_delta_threshold: float,
    swr_patch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_cols = list(dict.fromkeys(CORE_DETAIL_COLS + CAPACITY_DETAIL_COLS + history_cols(baseline)))
    field_cols = list(dict.fromkeys(CORE_DETAIL_COLS + _variant_cols(field_patch)))
    base_cols = [c for c in base_cols if c in baseline.columns]
    field_cols = [c for c in field_cols if c in field_patch.columns]

    left = baseline[base_cols].rename(
        columns={
            "pred_pv_norm": "pred_baseline",
            "true_pv_norm": "true_baseline",
            "horizon": "horizon_baseline",
            "true_delta_pv_norm": "true_delta_baseline",
            "large_change": "large_change_baseline",
        }
    )
    right = field_patch[field_cols].rename(
        columns={
            "pred_pv_norm": "pred_field_patch",
            "true_pv_norm": "true_field_patch",
            "horizon": "horizon_field_patch",
            "true_delta_pv_norm": "true_delta_field_patch",
            "large_change": "large_change_field_patch",
        }
    )
    merged = left.merge(right, on=KEY_COLS, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No rows aligned between 03 baseline and 05 field-patch eval details.")

    baseline_keys = baseline[KEY_COLS].drop_duplicates()
    field_keys = field_patch[KEY_COLS].drop_duplicates()
    missing_field = len(baseline_keys.merge(field_keys, on=KEY_COLS, how="left", indicator=True).query("_merge == 'left_only'"))
    missing_base = len(field_keys.merge(baseline_keys, on=KEY_COLS, how="left", indicator=True).query("_merge == 'left_only'"))

    if swr_patch is not None:
        swr_cols = list(dict.fromkeys(CORE_DETAIL_COLS + _variant_cols(swr_patch)))
        swr_cols = [c for c in swr_cols if c in swr_patch.columns]
        swr_right = swr_patch[swr_cols].rename(
            columns={
                "pred_pv_norm": "pred_swr_patch",
                "true_pv_norm": "true_swr_patch",
                "horizon": "horizon_swr_patch",
                "true_delta_pv_norm": "true_delta_swr_patch",
                "large_change": "large_change_swr_patch",
                "field_patch_center_pv_norm": "swr_patch_center_pv_norm",
                "field_patch_mean_pv_norm": "swr_patch_mean_pv_norm",
                "field_patch_support_mean": "swr_patch_support_mean",
            }
        )
        swr_keys = swr_patch[KEY_COLS].drop_duplicates()
        missing_swr = len(baseline_keys.merge(swr_keys, on=KEY_COLS, how="left", indicator=True).query("_merge == 'left_only'"))
        merged = merged.merge(swr_right, on=KEY_COLS, how="inner", validate="one_to_one")
    else:
        missing_swr = None

    truth_diff = (merged["true_baseline"].astype(float) - merged["true_field_patch"].astype(float)).abs()
    max_truth_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
    if np.isfinite(max_truth_diff) and max_truth_diff > 1.0e-5:
        print(f"[warn] max truth mismatch after alignment: {max_truth_diff:.6g}", flush=True)
    if "true_swr_patch" in merged.columns:
        swr_truth_diff = (merged["true_baseline"].astype(float) - merged["true_swr_patch"].astype(float)).abs()
        max_swr_truth_diff = float(swr_truth_diff.max()) if len(swr_truth_diff) else float("nan")
        if np.isfinite(max_swr_truth_diff) and max_swr_truth_diff > 1.0e-5:
            print(f"[warn] max SWR-patch truth mismatch after alignment: {max_swr_truth_diff:.6g}", flush=True)
    else:
        max_swr_truth_diff = float("nan")

    delta_diff = (merged["true_delta_baseline"].astype(float) - merged["true_delta_field_patch"].astype(float)).abs()
    max_delta_diff = float(delta_diff.max()) if len(delta_diff) else float("nan")
    if np.isfinite(max_delta_diff) and max_delta_diff > 1.0e-5:
        print(f"[warn] max true_delta mismatch after alignment: {max_delta_diff:.6g}", flush=True)

    merged["horizon"] = merged["horizon_baseline"].where(merged["horizon_baseline"].notna(), merged["horizon_field_patch"])
    merged["true_pv_norm"] = merged["true_baseline"].astype(float)
    merged["true_delta_pv_norm"] = merged["true_delta_baseline"].astype(float)
    merged["large_change"] = merged["large_change_baseline"] | merged["large_change_field_patch"]
    if "large_change_swr_patch" in merged.columns:
        merged["large_change"] = merged["large_change"] | merged["large_change_swr_patch"]
    if "large_change" not in merged or merged["large_change"].isna().any():
        merged["large_change"] = merged["true_delta_pv_norm"].abs() >= float(large_delta_threshold)

    for _model, pred_col, _err_col, _abs_col in available_model_specs(merged):
        err_col = pred_col.replace("pred_", "err_")
        abs_col = pred_col.replace("pred_", "abs_err_")
        sq_col = pred_col.replace("pred_", "sq_err_")
        merged[err_col] = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        merged[abs_col] = merged[err_col].abs()
        merged[sq_col] = merged[err_col] ** 2
    merged["abs_err_gain"] = merged["abs_err_baseline"] - merged["abs_err_field_patch"]
    if "abs_err_swr_patch" in merged.columns:
        merged["abs_err_gain_swr_patch"] = merged["abs_err_baseline"] - merged["abs_err_swr_patch"]
    merged.attrs["missing_field_rows"] = int(missing_field)
    merged.attrs["missing_baseline_rows"] = int(missing_base)
    merged.attrs["missing_swr_patch_rows_from_baseline_keys"] = None if missing_swr is None else int(missing_swr)
    merged.attrs["max_truth_diff"] = max_truth_diff
    merged.attrs["max_swr_patch_truth_diff"] = max_swr_truth_diff
    return merged


def history_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")])


def available_model_specs(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    return [(model, pred, err, abs_err) for model, pred, err, abs_err in MODEL_SPECS if pred in df.columns]


def metric_dict(df: pd.DataFrame, model: str, pred_col: str) -> dict:
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
        pred_sign = np.sign(pred_delta[sign_valid])
        true_sign = np.sign(true_delta[sign_valid])
        delta_sign_accuracy = float((pred_sign == true_sign).mean()) if len(pred_sign) else float("nan")
        if int(valid_delta.sum()) >= 3:
            corr = float(np.corrcoef(pred_delta[valid_delta], true_delta[valid_delta])[0, 1])
        else:
            corr = float("nan")
        pred_delta_err = (pred_delta - true_delta).abs()
        delta_mae = float(pred_delta_err[valid_delta].mean()) if bool(valid_delta.any()) else float("nan")
    else:
        delta_sign_accuracy = float("nan")
        corr = float("nan")
        delta_mae = float("nan")
    return {
        "model": model,
        "n_points": int(len(df)),
        "mae_pv_norm": float(abs_err.mean()),
        "rmse_pv_norm": float(math.sqrt(float(sq_err.mean()))),
        "bias_pv_norm": float(err.mean()),
        "p95_abs_err_pv_norm": float(abs_err.quantile(0.95)),
        "max_abs_err_pv_norm": float(abs_err.max()),
        "delta_mae_pv_norm": delta_mae,
        "delta_sign_accuracy": delta_sign_accuracy,
        "delta_corr": corr,
        "large_change_mae_pv_norm": float(abs_err[large].mean()) if bool(large.any()) else float("nan"),
        "large_change_n": int(large.sum()),
        **regression_metric_extras(df, pred_col),
    }


def _metric_pair_rows(df: pd.DataFrame, scope: dict) -> list[dict]:
    rows = []
    for model, pred_col, _err_col, _abs_col in available_model_specs(df):
        row = metric_dict(df, model, pred_col)
        row.update(scope)
        rows.append(row)
    return rows


def build_overall_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["eval_split", "horizon"]
    groups = list(merged.groupby(group_keys, sort=True)) + [(("all_splits", "all"), merged)]
    for keys, g in groups:
        split, horizon = keys
        rows.extend(_metric_pair_rows(g, {"eval_split": split, "horizon": horizon}))
    for split, g_split in merged.groupby("eval_split", sort=True):
        rows.extend(_metric_pair_rows(g_split, {"eval_split": split, "horizon": "all"}))
    return pd.DataFrame(rows).sort_values(["eval_split", "horizon", "model"]).reset_index(drop=True)


def build_dimension_metrics(merged: pd.DataFrame, large_delta_threshold: float) -> pd.DataFrame:
    work = merged.copy()
    truth_bins = [-np.inf, 0.05, 0.2, 0.5, 0.8, np.inf]
    truth_labels = ["truth_lt_0p05", "truth_0p05_0p2", "truth_0p2_0p5", "truth_0p5_0p8", "truth_ge_0p8"]
    change_bins = [-np.inf, 0.05, float(large_delta_threshold), 0.25, 0.5, np.inf]
    change_labels = ["delta_lt_0p05", f"delta_0p05_{large_delta_threshold:g}", "delta_mid", "delta_large", "delta_extreme"]
    err_bins = [-np.inf, 0.05, 0.1, 0.2, 0.4, np.inf]
    err_labels = ["base_err_lt_0p05", "base_err_0p05_0p1", "base_err_0p1_0p2", "base_err_0p2_0p4", "base_err_ge_0p4"]

    work["truth_bin"] = pd.cut(work["true_pv_norm"].astype(float), bins=truth_bins, labels=truth_labels)
    work["change_bin"] = pd.cut(work["true_delta_pv_norm"].astype(float).abs(), bins=change_bins, labels=change_labels)
    work["baseline_error_bin"] = pd.cut(work["abs_err_baseline"].astype(float), bins=err_bins, labels=err_labels)

    rows = []
    specs = [
        ("truth_bin", ["eval_split", "truth_bin"]),
        ("change_bin", ["eval_split", "change_bin"]),
        ("baseline_error_bin", ["eval_split", "baseline_error_bin"]),
        ("station", ["eval_split", "station_sn"]),
        ("horizon_truth_bin", ["eval_split", "horizon", "truth_bin"]),
        ("horizon_change_bin", ["eval_split", "horizon", "change_bin"]),
    ]
    for dimension, keys in specs:
        for key_values, g in work.groupby(keys, sort=True, observed=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            scope = {"dimension": dimension}
            scope.update({k: str(v) for k, v in zip(keys, key_values)})
            rows.extend(_metric_pair_rows(g, scope))
    return pd.DataFrame(rows)


def build_issue_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["station_sn", "eval_split", "issue_time"]
    for key_values, g in merged.groupby(keys, sort=True):
        station, split, issue = key_values
        base_mae = float(g["abs_err_baseline"].mean())
        field_mae = float(g["abs_err_field_patch"].mean())
        row = {
            "station_sn": str(station),
            "eval_split": str(split),
            "issue_time": str(issue),
            "n_horizons": int(len(g)),
            "baseline_mae": base_mae,
            "field_patch_mae": field_mae,
            "mae_gain": base_mae - field_mae,
            "truth_mean": float(g["true_pv_norm"].mean()),
            "truth_min": float(g["true_pv_norm"].min()),
            "truth_max": float(g["true_pv_norm"].max()),
            "large_change_ratio": float(g["large_change"].astype(bool).mean()),
        }
        if "abs_err_swr_patch" in g.columns:
            swr_mae = float(g["abs_err_swr_patch"].mean())
            row["swr_patch_mae"] = swr_mae
            row["swr_patch_mae_gain"] = base_mae - swr_mae
        for model, pred_col, _err_col, _abs_col in available_model_specs(g):
            extras = regression_metric_extras(g, pred_col)
            for metric in ["r2_pv_norm", "acc_installed_capacity", "nrmse_installed_capacity"]:
                row[f"{model}_{metric}"] = extras[metric]
        for col in ["field_patch_center_pv_norm", "field_patch_mean_pv_norm", "field_patch_support_mean"]:
            if col in g.columns:
                row[col] = float(g[col].mean())
        for col in ["swr_patch_center_pv_norm", "swr_patch_mean_pv_norm", "swr_patch_support_mean"]:
            if col in g.columns:
                row[col] = float(g[col].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eval_split", "baseline_mae"], ascending=[True, False]).reset_index(drop=True)


def build_horizon_diversity(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, station, issue), g in merged.groupby(["eval_split", "station_sn", "issue_time"], sort=True):
        if len(g) < 2:
            continue
        for model, pred_col, _err_col, _abs_col in available_model_specs(g):
            rows.append(
                {
                    "eval_split": str(split),
                    "station_sn": str(station),
                    "issue_time": str(issue),
                    "model": model,
                    "pred_horizon_std": float(g[pred_col].astype(float).std(ddof=0)),
                    "truth_horizon_std": float(g["true_pv_norm"].astype(float).std(ddof=0)),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["eval_split", "model"], sort=True)
        .agg(
            n_issues=("issue_time", "count"),
            pred_horizon_diversity=("pred_horizon_std", "mean"),
            truth_horizon_diversity=("truth_horizon_std", "mean"),
        )
        .reset_index()
    )
    summary["horizon_diversity_ratio"] = np.where(
        summary["truth_horizon_diversity"] > 0,
        summary["pred_horizon_diversity"] / summary["truth_horizon_diversity"],
        np.nan,
    )
    return summary


def build_improvement(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["model"] == BASELINE_MODEL].copy()
    suffix_cols = ["eval_split", "horizon"]
    rows = []
    for candidate in sorted(metrics["model"].astype(str).unique()):
        if candidate == BASELINE_MODEL:
            continue
        other = metrics[metrics["model"] == candidate].copy()
        joined = base.merge(other, on=suffix_cols, suffixes=("_baseline", "_candidate"), validate="one_to_one")
        if joined.empty:
            continue
        joined["candidate_model"] = candidate
        joined["mae_gain"] = joined["mae_pv_norm_baseline"] - joined["mae_pv_norm_candidate"]
        joined["rmse_gain"] = joined["rmse_pv_norm_baseline"] - joined["rmse_pv_norm_candidate"]
        joined["r2_gain"] = joined["r2_pv_norm_candidate"] - joined["r2_pv_norm_baseline"]
        joined["acc_gain"] = joined["acc_installed_capacity_candidate"] - joined["acc_installed_capacity_baseline"]
        joined["mae_gain_pct"] = np.where(
            joined["mae_pv_norm_baseline"] > 0,
            100.0 * joined["mae_gain"] / joined["mae_pv_norm_baseline"],
            np.nan,
        )
        joined["rmse_gain_pct"] = np.where(
            joined["rmse_pv_norm_baseline"] > 0,
            100.0 * joined["rmse_gain"] / joined["rmse_pv_norm_baseline"],
            np.nan,
        )
        keep = [
            "eval_split",
            "horizon",
            "candidate_model",
            "n_points_baseline",
            "mae_pv_norm_baseline",
            "mae_pv_norm_candidate",
            "mae_gain",
            "mae_gain_pct",
            "rmse_pv_norm_baseline",
            "rmse_pv_norm_candidate",
            "rmse_gain",
            "rmse_gain_pct",
            "r2_pv_norm_baseline",
            "r2_pv_norm_candidate",
            "r2_gain",
            "acc_installed_capacity_baseline",
            "acc_installed_capacity_candidate",
            "acc_gain",
            "nrmse_installed_capacity_baseline",
            "nrmse_installed_capacity_candidate",
            "bias_pv_norm_baseline",
            "bias_pv_norm_candidate",
            "large_change_mae_pv_norm_baseline",
            "large_change_mae_pv_norm_candidate",
            "large_change_n_baseline",
        ]
        rows.append(
            joined[keep].rename(
                columns={
                    "n_points_baseline": "n_points",
                    "mae_pv_norm_candidate": "mae_pv_norm_model",
                    "rmse_pv_norm_candidate": "rmse_pv_norm_model",
                    "r2_pv_norm_candidate": "r2_pv_norm_model",
                    "acc_installed_capacity_candidate": "acc_installed_capacity_model",
                    "nrmse_installed_capacity_candidate": "nrmse_installed_capacity_model",
                    "bias_pv_norm_candidate": "bias_pv_norm_model",
                    "large_change_mae_pv_norm_candidate": "large_change_mae_pv_norm_model",
                    "large_change_n_baseline": "large_change_n",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_per_station_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, split), g in merged.groupby(["station_sn", "eval_split"], sort=True):
        for horizon, h_group in list(g.groupby("horizon", sort=True)) + [("all", g)]:
            for model, pred_col, _err_col, _abs_col in available_model_specs(h_group):
                row = metric_dict(h_group, model, pred_col)
                row.update({"station_sn": station, "eval_split": split, "horizon": horizon})
                rows.append(row)
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    return metrics.sort_values(["eval_split", "station_sn", "horizon", "model"]).reset_index(drop=True)


def select_issue_samples(
    merged: pd.DataFrame,
    sample_count: int,
    seed: int,
    split: str = "test",
    strategy: str = "random",
) -> pd.DataFrame:
    sub = merged[merged["eval_split"] == str(split)].copy()
    if sub.empty:
        return pd.DataFrame()
    issue_scores = (
        sub.groupby(["station_sn", "issue_time"], sort=False)
        .agg(
            baseline_mae=("abs_err_baseline", "mean"),
            field_patch_mae=("abs_err_field_patch", "mean"),
            swr_patch_mae=("abs_err_swr_patch", "mean") if "abs_err_swr_patch" in sub.columns else ("abs_err_field_patch", "mean"),
            mean_truth=("true_pv_norm", "mean"),
            min_truth=("true_pv_norm", "min"),
            max_truth=("true_pv_norm", "max"),
            mean_gain=("abs_err_gain", "mean"),
            swr_mean_gain=("abs_err_gain_swr_patch", "mean") if "abs_err_gain_swr_patch" in sub.columns else ("abs_err_gain", "mean"),
        )
        .reset_index()
    )
    issue_scores["abs_gap"] = (issue_scores["baseline_mae"] - issue_scores["field_patch_mae"]).abs()
    n = min(max(int(sample_count), 0), len(issue_scores))
    if n == 0:
        return pd.DataFrame()
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(issue_scores.index.to_numpy(), size=n, replace=False))
        return issue_scores.loc[idx].reset_index(drop=True)
    if strategy == "worst":
        return issue_scores.sort_values("baseline_mae", ascending=False).head(n).reset_index(drop=True)
    if strategy == "largest_gap":
        return issue_scores.sort_values("abs_gap", ascending=False).head(n).reset_index(drop=True)
    raise ValueError(f"Unknown sample selection strategy: {strategy}")


def save_sample_figures(
    merged: pd.DataFrame,
    sampled: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    subdir: str = "random_sample_comparisons",
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(out_dir / subdir)
    test = merged[merged["eval_split"] == "test"].copy()
    if test.empty or sampled.empty:
        return pd.DataFrame()

    hist_columns = history_cols(merged)
    rows = []
    for i, sample in sampled.reset_index(drop=True).iterrows():
        station = str(sample["station_sn"])
        issue = str(sample["issue_time"])
        sub = test[(test["station_sn"].astype(str) == station) & (test["issue_time"].astype(str) == issue)].sort_values("h")
        if sub.empty:
            continue
        x = sub["h"].to_numpy(np.int64)
        fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.8), sharex=False)

        if hist_columns:
            hist = sub.iloc[0][hist_columns].to_numpy(np.float64)
            hist_x = np.arange(-len(hist) + 1, 1)
            axes[0].plot(hist_x, hist, color="#7f7f7f", marker=".", linewidth=2.0, label="history PV_norm")
            axes[0].axvline(0, color="0.7", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.2, label="truth")
        axes[0].plot(x, sub["pred_baseline"].to_numpy(np.float64), color="#1f77b4", marker="o", linewidth=2.0, label="03 baseline")
        axes[0].plot(x, sub["pred_field_patch"].to_numpy(np.float64), color="#d62728", marker="o", linewidth=2.0, label="05 field patch")
        if "pred_swr_patch" in sub.columns:
            axes[0].plot(x, sub["pred_swr_patch"].to_numpy(np.float64), color="#ff7f0e", marker="o", linewidth=2.0, label="06 SWR patch")
        if "field_patch_center_pv_norm" in sub.columns:
            axes[0].plot(
                x,
                sub["field_patch_center_pv_norm"].to_numpy(np.float64),
                color="#2ca02c",
                marker=".",
                linewidth=1.4,
                label="field patch center",
            )
        axes[0].set_title(f"Stage 4 03 vs 05 | station={station} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", ncol=2)

        width = 0.36
        axes[1].bar(x - width / 2, sub["err_baseline"].to_numpy(np.float64), width=width, color="#1f77b4", alpha=0.85, label="03 error")
        axes[1].bar(x + width / 2, sub["err_field_patch"].to_numpy(np.float64), width=width, color="#d62728", alpha=0.85, label="05 error")
        if "err_swr_patch" in sub.columns:
            axes[1].plot(x, sub["err_swr_patch"].to_numpy(np.float64), color="#ff7f0e", marker="s", linewidth=1.6, label="06 error")
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("pred - truth")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        axes[2].plot(x, sub["abs_err_baseline"].to_numpy(np.float64), color="#1f77b4", marker="o", label="03 abs error")
        axes[2].plot(x, sub["abs_err_field_patch"].to_numpy(np.float64), color="#d62728", marker="o", label="05 abs error")
        if "abs_err_swr_patch" in sub.columns:
            axes[2].plot(x, sub["abs_err_swr_patch"].to_numpy(np.float64), color="#ff7f0e", marker="s", label="06 abs error")
        if "field_patch_mean_pv_norm" in sub.columns:
            ax2 = axes[2].twinx()
            ax2.plot(
                x,
                sub["field_patch_mean_pv_norm"].to_numpy(np.float64),
                color="#2ca02c",
                linestyle="--",
                marker=".",
                label="field patch mean",
            )
            ax2.set_ylabel("field patch mean")
            handles1, labels1 = axes[2].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            axes[2].legend(handles1 + handles2, labels1 + labels2, loc="best")
        else:
            axes[2].legend(loc="best")
        axes[2].set_ylabel("absolute error")
        axes[2].set_xlabel("forecast horizon")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"h{k}" for k in x])
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        safe_issue = issue.replace(":", "").replace(" ", "_").replace("/", "-")
        file_name = f"stage4_03_vs_05_{i:03d}_{station}_{safe_issue}.png"
        fig.savefig(fig_dir / file_name, dpi=int(dpi))
        plt.close(fig)
        rows.append(
            {
                "station_sn": station,
                "issue_time": issue,
                "baseline_mae": float(sample["baseline_mae"]),
                "field_patch_mae": float(sample["field_patch_mae"]),
                "swr_patch_mae": float(sample["swr_patch_mae"]) if "swr_patch_mae" in sample.index else float("nan"),
                "mean_gain": float(sample["mean_gain"]),
                "swr_mean_gain": float(sample["swr_mean_gain"]) if "swr_mean_gain" in sample.index else float("nan"),
                "figure": str(fig_dir / file_name),
            }
        )
    return pd.DataFrame(rows)


def save_summary_plot(improvement: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = improvement[(improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))]
    if test.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    labels = ["all", "h1", "h4", "h8"]
    x = np.arange(len(labels))
    base = test.drop_duplicates("horizon").set_index("horizon").reindex(labels)
    axes[0].bar(x - 0.25, base["mae_pv_norm_baseline"], width=0.24, color="#1f77b4", label="03 baseline")
    colors = {FIELD_PATCH_MODEL: "#d62728", SWR_PATCH_MODEL: "#ff7f0e"}
    offsets = {FIELD_PATCH_MODEL: 0.0, SWR_PATCH_MODEL: 0.25}
    labels_map = {FIELD_PATCH_MODEL: "05 field patch", SWR_PATCH_MODEL: "06 SWR patch"}
    for candidate in test["candidate_model"].astype(str).drop_duplicates():
        g = test[test["candidate_model"].astype(str) == candidate].set_index("horizon").reindex(labels)
        axes[0].bar(x + offsets.get(candidate, 0.25), g["mae_pv_norm_model"], width=0.24, color=colors.get(candidate, "0.5"), label=labels_map.get(candidate, candidate))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("MAE PV_norm")
    axes[0].set_title("Test MAE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    width = 0.32 if test["candidate_model"].nunique() > 1 else 0.5
    for j, candidate in enumerate(test["candidate_model"].astype(str).drop_duplicates()):
        g = test[test["candidate_model"].astype(str) == candidate].set_index("horizon").reindex(labels)
        offset = (j - (test["candidate_model"].nunique() - 1) / 2.0) * width
        axes[1].bar(x + offset, g["mae_gain"], width=width, color=colors.get(candidate, "#2ca02c"), label=labels_map.get(candidate, candidate))
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("03 MAE - model MAE")
    axes[1].set_title("Positive = candidate improves")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "stage4_03_vs_05_test_mae_summary.png", dpi=int(dpi))
    plt.close(fig)


def save_multisource_sample_figures(sampled: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    if sampled.empty or int(args.multisource_sample_count) <= 0:
        return pd.DataFrame()
    if not Path(args.multisource_cache_dir).exists():
        print(f"[warn] skip multisource figures: missing {args.multisource_cache_dir}", flush=True)
        return pd.DataFrame()
    if not Path(args.frame_cache_dir).exists():
        print(f"[warn] skip multisource figures: missing {args.frame_cache_dir}", flush=True)
        return pd.DataFrame()

    from stage3_visualize_multisource_impl import (
        ReadOnlyDiskFrameCache,
        _bounds,
        _compute_or_load_norm_stats,
        _grid_lonlat,
        _issue_index,
        _load_multisource_cache,
        _load_optional_ms_arrays,
        _load_split_df,
        _time_df_from_ms,
        render_h8_time_features,
        render_nwp_features,
        render_pv_sequence,
    )

    fig_dir = ensure_dir(out_dir / "random_multisource_inputs")
    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    lon_grid, lat_grid = _grid_lonlat(bounds, int(args.grid_size))

    ms_cache = _load_multisource_cache(Path(args.multisource_cache_dir))
    ms_cache["time_df"] = _time_df_from_ms(ms_cache)
    ms_optional = _load_optional_ms_arrays(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"].copy()
    stats_dir = ensure_dir(out_dir / "multisource_norm_for_compare")
    stats = _compute_or_load_norm_stats(ms_cache, stats_dir, time_df, force=False)
    issue_to_idx = _issue_index(time_df)
    pv_cache = ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)

    wanted = sampled.head(int(args.multisource_sample_count)).copy()
    rows = []
    for i, row in wanted.reset_index(drop=True).iterrows():
        issue = pd.Timestamp(row["issue_time"])
        matches = time_df[time_df["issue_time"] == issue]
        if matches.empty:
            rows.append(
                {
                    "sample_id": int(i),
                    "station_sn": str(row.get("station_sn", "")),
                    "issue_time": str(issue),
                    "status": "missing_issue_in_multisource_cache",
                }
            )
            continue
        ms_row = matches.iloc[0]
        print(f"[multisource_sample] {i + 1}/{len(wanted)} issue={issue}", flush=True)
        paths = {}
        paths.update(render_pv_sequence(i, ms_row, pv_cache, extent, fig_dir, args))
        paths.update(render_h8_time_features(i, ms_row, ms_cache, ms_optional, stats, issue_to_idx, lon_grid, lat_grid, extent, fig_dir, args))
        paths.update(render_nwp_features(i, ms_row, ms_cache, stats, issue_to_idx, extent, fig_dir, args))
        rows.append(
            {
                "sample_id": int(i),
                "station_sn": str(row.get("station_sn", "")),
                "issue_time": str(issue),
                "baseline_mae": float(row.get("baseline_mae", np.nan)),
                "field_patch_mae": float(row.get("field_patch_mae", np.nan)),
                "status": "ok",
                **paths,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "stage4_03_vs_05_multisource_sample_figures.csv", index=False)
    return out


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sample_count is not None:
        args.random_sample_count = int(args.sample_count)
    ensure_stage4_dirs()
    out_dir = ensure_dir(args.out_dir)

    baseline = read_detail(args.baseline_detail_csv, BASELINE_MODEL)
    field_patch = read_detail(args.field_patch_detail_csv, FIELD_PATCH_MODEL)
    swr_patch = optional_read_detail(args.swr_patch_detail_csv, SWR_PATCH_MODEL)
    merged = aligned_detail(baseline, field_patch, float(args.large_delta_threshold), swr_patch=swr_patch)

    overall = build_overall_metrics(merged)
    improvement = build_improvement(overall)
    dimension_metrics = build_dimension_metrics(merged, float(args.large_delta_threshold))
    issue_metrics = build_issue_metrics(merged)
    horizon_diversity = build_horizon_diversity(merged)
    per_station = build_per_station_metrics(merged)
    random_samples = select_issue_samples(merged, int(args.random_sample_count), int(args.seed), split="test", strategy="random")
    worst_samples = select_issue_samples(merged, int(args.worst_sample_count), int(args.seed), split="test", strategy="worst")
    random_figures = save_sample_figures(merged, random_samples, out_dir, int(args.dpi), subdir="random_sample_comparisons")
    worst_figures = save_sample_figures(merged, worst_samples, out_dir, int(args.dpi), subdir="worst_sample_comparisons")
    multisource_figures = save_multisource_sample_figures(random_samples, out_dir, args)
    save_summary_plot(improvement, out_dir, int(args.dpi))

    overall.to_csv(out_dir / "stage4_03_vs_05_overall_metrics.csv", index=False)
    improvement.to_csv(out_dir / "stage4_03_vs_05_improvement.csv", index=False)
    dimension_metrics.to_csv(out_dir / "stage4_03_vs_05_dimension_metrics.csv", index=False)
    issue_metrics.to_csv(out_dir / "stage4_03_vs_05_issue_metrics.csv", index=False)
    horizon_diversity.to_csv(out_dir / "stage4_03_vs_05_horizon_diversity.csv", index=False)
    per_station.to_csv(out_dir / "stage4_03_vs_05_per_station_metrics.csv", index=False)
    random_samples.to_csv(out_dir / "stage4_03_vs_05_random_sample_index.csv", index=False)
    worst_samples.to_csv(out_dir / "stage4_03_vs_05_worst_sample_index.csv", index=False)
    random_figures.to_csv(out_dir / "stage4_03_vs_05_random_sample_figures.csv", index=False)
    worst_figures.to_csv(out_dir / "stage4_03_vs_05_worst_sample_figures.csv", index=False)

    compact_cols = [
        "station_sn",
        "eval_split",
        "issue_time",
        "h",
        "horizon",
        "true_pv_norm",
        "capacity_raw",
        "capacity_scale",
        "pred_baseline",
        "pred_field_patch",
        "abs_err_baseline",
        "abs_err_field_patch",
        "abs_err_gain",
        "true_delta_pv_norm",
        "large_change",
    ]
    if "pred_swr_patch" in merged.columns:
        compact_cols.extend(["pred_swr_patch", "abs_err_swr_patch", "abs_err_gain_swr_patch"])
    compact_cols = [c for c in compact_cols if c in merged.columns]
    extra_cols = [
        c
        for c in [
            "field_patch_center_pv_norm",
            "field_patch_mean_pv_norm",
            "field_patch_support_mean",
            "swr_patch_center_pv_norm",
            "swr_patch_mean_pv_norm",
            "swr_patch_support_mean",
        ]
        if c in merged.columns
    ]
    merged[compact_cols + extra_cols].to_csv(out_dir / "stage4_03_vs_05_aligned_detail_compact.csv", index=False)

    meta = {
        "baseline_detail_csv": str(args.baseline_detail_csv),
        "field_patch_detail_csv": str(args.field_patch_detail_csv),
        "swr_patch_detail_csv": str(args.swr_patch_detail_csv),
        "swr_patch_loaded": swr_patch is not None,
        "out_dir": str(out_dir),
        "n_baseline_rows": int(len(baseline)),
        "n_field_patch_rows": int(len(field_patch)),
        "n_swr_patch_rows": None if swr_patch is None else int(len(swr_patch)),
        "n_aligned_rows": int(len(merged)),
        "missing_field_rows_from_baseline_keys": int(merged.attrs.get("missing_field_rows", 0)),
        "missing_baseline_rows_from_field_keys": int(merged.attrs.get("missing_baseline_rows", 0)),
        "missing_swr_patch_rows_from_baseline_keys": merged.attrs.get("missing_swr_patch_rows_from_baseline_keys"),
        "max_truth_diff": float(merged.attrs.get("max_truth_diff", float("nan"))),
        "max_swr_patch_truth_diff": float(merged.attrs.get("max_swr_patch_truth_diff", float("nan"))),
        "random_sample_count": int(len(random_samples)),
        "worst_sample_count": int(len(worst_samples)),
        "multisource_sample_count": int(len(multisource_figures)),
        "run_dir": str(args.run_dir),
        "frame_cache_dir": str(args.frame_cache_dir),
        "multisource_cache_dir": str(args.multisource_cache_dir),
    }
    write_json(out_dir / "stage4_03_vs_05_eval_meta.json", meta)

    print("==== Stage 4 03 baseline vs 05 field-patch comparison ====")
    print(f"baseline detail : {args.baseline_detail_csv}")
    print(f"field detail    : {args.field_patch_detail_csv}")
    print(f"swr detail      : {args.swr_patch_detail_csv if swr_patch is not None else 'SKIPPED'}")
    print(f"aligned rows    : {len(merged)}")
    show = improvement[
        (improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))
    ].copy()
    if show.empty:
        show = improvement[improvement["horizon"].isin(["all", "h1", "h4", "h8"])].copy()
    if not show.empty:
        display_cols = [
            "eval_split",
            "horizon",
            "candidate_model",
            "n_points",
            "mae_pv_norm_baseline",
            "mae_pv_norm_model",
            "mae_gain",
            "mae_gain_pct",
            "rmse_pv_norm_baseline",
            "rmse_pv_norm_model",
            "rmse_gain",
            "large_change_mae_pv_norm_baseline",
            "large_change_mae_pv_norm_model",
        ]
        print(show[[c for c in display_cols if c in show.columns]].to_string(index=False))
    if not horizon_diversity.empty:
        print("==== Horizon diversity ====")
        print(horizon_diversity[horizon_diversity["eval_split"] == "test"].to_string(index=False))
    if not issue_metrics.empty:
        print("==== Worst test issue samples by 03 baseline MAE ====")
        cols = [
            "station_sn",
            "issue_time",
            "baseline_mae",
            "field_patch_mae",
            "mae_gain",
            "truth_min",
            "truth_max",
            "field_patch_center_pv_norm",
            "swr_patch_mae",
            "swr_patch_mae_gain",
            "swr_patch_center_pv_norm",
        ]
        worst_show = issue_metrics[issue_metrics["eval_split"] == "test"].head(12)
        print(worst_show[[c for c in cols if c in worst_show.columns]].to_string(index=False))
    print(f"saved tables    : {out_dir}")
    print(f"random figures  : {out_dir / 'random_sample_comparisons'}")
    print(f"worst figures   : {out_dir / 'worst_sample_comparisons'}")
    print(f"multisource figs: {out_dir / 'random_multisource_inputs'}")


if __name__ == "__main__":
    main()
