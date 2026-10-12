#!/usr/bin/env python3
"""Evaluate Stage 4 patch-size and PV-field-source ablations.

This is a small evaluation suite around the existing Stage 4 eval-detail CSV
format. It runs two aligned comparisons:

1. PV-field patch size: 1x1 / 3x3 / 5x5 / 7x7 against the 03 baseline.
2. PV-field source: historical PV-field patch vs future Stage 3 PV-field patch.

The script intentionally accepts direct CSV paths so previously trained runs in
other folders can be reused without moving files into stage_4/runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_EVAL_DIR = SCRIPT_DIR / "ablation_pv_field_patch_size"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PATCH_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_EVAL_DIR))

from stage4_paths import (  # noqa: E402
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
    ensure_stage4_dirs,
)
from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras


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


def _load_patch_eval_module():
    path = PATCH_EVAL_DIR / "eval_patch_size_ablation.py"
    spec = importlib.util.spec_from_file_location("stage4_patch_size_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import patch-size eval module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_patch_size_eval"] = module
    spec.loader.exec_module(module)
    return module


patch_eval = _load_patch_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
        help="03 temporal baseline eval-detail CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=RUNS_DIR / "10_eval_patch_size_and_field_source_ablation",
        help="Output root. The script writes patch_size/ and field_source/ under this folder.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--large_delta_threshold", type=float, default=0.10)

    parser.add_argument(
        "--patch_run_root",
        type=Path,
        default=RUNS_DIR / "ablation_pv_field_patch_size",
        help="Run root containing patch_1x1, patch_3x3, patch_7x7.",
    )
    parser.add_argument("--patch_sizes", type=str, default="1,3,5,7")
    parser.add_argument(
        "--patch_detail",
        action="append",
        default=[],
        metavar="SIZE=CSV",
        help="Override a patch-size eval-detail CSV. Can repeat, e.g. --patch_detail 5=/path/to/eval_detail.csv.",
    )
    parser.add_argument("--patch_1_detail_csv", type=Path, default=None)
    parser.add_argument("--patch_3_detail_csv", type=Path, default=None)
    parser.add_argument("--patch_5_detail_csv", type=Path, default=None)
    parser.add_argument("--patch_7_detail_csv", type=Path, default=None)
    parser.add_argument(
        "--patch_5_dir",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
        help="Fallback 5x5 run directory when --patch_5_detail_csv is not supplied.",
    )
    parser.add_argument("--patch_random_sample_count", type=int, default=32)
    parser.add_argument("--patch_worst_sample_count", type=int, default=16)

    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=None,
        help="05h historical PV-field eval-detail CSV. Defaults to the standard Stage 4 historical run.",
    )
    parser.add_argument("--historical_field_dir", type=Path, default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR)
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=None,
        help=(
            "05l future Stage 3 PV-field eval-detail CSV. If omitted, the script reuses "
            "--patch_5_detail_csv, then --patch_5_dir, then the standard Stage 4 05l run."
        ),
    )
    parser.add_argument("--future_field_dir", type=Path, default=None)
    parser.add_argument("--field_random_sample_count", type=int, default=32)
    parser.add_argument("--field_worst_sample_count", type=int, default=16)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_sizes(text: str) -> list[int]:
    sizes: list[int] = []
    for part in str(text).split(","):
        part = part.strip().lower().replace("x", "")
        if part:
            sizes.append(int(part))
    return sizes


def safe_label(label: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip())
    label = re.sub(r"_+", "_", label).strip("_")
    if not label:
        raise ValueError("empty model label")
    return label


def parse_patch_overrides(args: argparse.Namespace) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for size, path in [
        (1, args.patch_1_detail_csv),
        (3, args.patch_3_detail_csv),
        (5, args.patch_5_detail_csv),
        (7, args.patch_7_detail_csv),
    ]:
        if path is not None:
            out[int(size)] = Path(path)
    for item in args.patch_detail:
        if "=" not in str(item):
            raise ValueError(f"--patch_detail must be SIZE=CSV, got: {item}")
        size_text, path_text = str(item).split("=", 1)
        size = int(size_text.lower().strip().replace("x", ""))
        out[size] = Path(path_text)
    return out


def patch_detail_path(args: argparse.Namespace, overrides: dict[int, Path], size: int) -> Path:
    if int(size) in overrides:
        return Path(overrides[int(size)])
    local = Path(args.patch_run_root) / f"patch_{size}x{size}" / DETAIL_NAME
    if local.exists():
        return local
    if int(size) == 5:
        return Path(args.patch_5_dir) / DETAIL_NAME
    return local


def field_detail_path(args: argparse.Namespace, kind: str) -> Path:
    if kind == "historical":
        if args.historical_field_detail_csv is not None:
            return Path(args.historical_field_detail_csv)
        return Path(args.historical_field_dir) / DETAIL_NAME
    if kind == "future":
        if args.future_field_detail_csv is not None:
            return Path(args.future_field_detail_csv)
        if args.patch_5_detail_csv is not None:
            return Path(args.patch_5_detail_csv)
        if args.future_field_dir is not None:
            return Path(args.future_field_dir) / DETAIL_NAME
        return Path(args.patch_5_dir) / DETAIL_NAME
    raise ValueError(f"unknown field source kind: {kind}")


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def history_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")])


def read_detail(path: Path, label: str, required: bool = True) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        msg = f"missing eval-detail CSV for {label}: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"[warn] skip {msg}", flush=True)
        return None
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
    df = attach_capacity_from_station_meta(df, path)
    return df


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


def align_field_sources(
    baseline: pd.DataFrame | None,
    candidates: list[tuple[str, pd.DataFrame]],
    large_delta_threshold: float,
) -> pd.DataFrame:
    if not candidates:
        raise RuntimeError("No field-source candidate details are available.")
    ref = baseline if baseline is not None else candidates[0][1]
    ref_cols = [
        "station_sn",
        "eval_split",
        "issue_time",
        "h",
        "horizon",
        "true_pv_norm",
        "true_delta_pv_norm",
        "large_change",
    ]
    ref_cols = list(dict.fromkeys(ref_cols + [c for c in CAPACITY_DETAIL_COLS if c in ref.columns]))
    if baseline is not None:
        ref_cols = list(dict.fromkeys(ref_cols + ["pred_pv_norm"] + history_cols(baseline)))
    merged = ref[ref_cols].copy()
    if baseline is not None:
        merged = merged.rename(columns={"pred_pv_norm": "pred_03_baseline"})
    merged["large_change"] = coerce_bool(merged["large_change"])

    source_paths: dict[str, int] = {}
    for label, df in candidates:
        tag = safe_label(label)
        if tag in source_paths:
            raise ValueError(f"duplicate model label after sanitizing: {label} -> {tag}")
        source_paths[tag] = len(df)
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
    for pred_col in [c for c in merged.columns if c.startswith("pred_")]:
        tag = pred_col.replace("pred_", "", 1)
        err = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        merged[f"err_{tag}"] = err
        merged[f"abs_err_{tag}"] = err.abs()
        merged[f"sq_err_{tag}"] = err**2
    return merged


def prediction_columns(merged: pd.DataFrame) -> list[tuple[str, str]]:
    cols = []
    if "pred_03_baseline" in merged.columns:
        cols.append(("03_baseline", "pred_03_baseline"))
    for col in sorted([c for c in merged.columns if c.startswith("pred_") and c != "pred_03_baseline"]):
        cols.append((col.replace("pred_", "", 1), col))
    return cols


def metric_row(df: pd.DataFrame, model: str, pred_col: str, scope: dict) -> dict:
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


def build_metrics(merged: pd.DataFrame, large_delta_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (split, horizon), g in merged.groupby(["eval_split", "horizon"], sort=True):
        for model, pred_col in prediction_columns(g):
            rows.append(metric_row(g, model, pred_col, {"eval_split": split, "horizon": horizon, "dimension": "horizon"}))
    for split, g in merged.groupby("eval_split", sort=True):
        for model, pred_col in prediction_columns(g):
            rows.append(metric_row(g, model, pred_col, {"eval_split": split, "horizon": "all", "dimension": "split"}))
    overall = pd.DataFrame(rows)

    work = merged.copy()
    work["truth_bin"] = pd.cut(
        work["true_pv_norm"].astype(float),
        bins=[-np.inf, 0.05, 0.2, 0.5, 0.8, np.inf],
        labels=["truth_lt_0p05", "truth_0p05_0p2", "truth_0p2_0p5", "truth_0p5_0p8", "truth_ge_0p8"],
    )
    work["change_bin"] = pd.cut(
        work["true_delta_pv_norm"].astype(float).abs(),
        bins=[-np.inf, 0.05, float(large_delta_threshold), 0.25, 0.5, np.inf],
        labels=["delta_lt_0p05", f"delta_0p05_{large_delta_threshold:g}", "delta_mid", "delta_large", "delta_extreme"],
    )
    dim_rows = []
    for dimension, keys in [
        ("truth_bin", ["eval_split", "truth_bin"]),
        ("change_bin", ["eval_split", "change_bin"]),
        ("station", ["eval_split", "station_sn"]),
        ("horizon_truth_bin", ["eval_split", "horizon", "truth_bin"]),
        ("horizon_change_bin", ["eval_split", "horizon", "change_bin"]),
    ]:
        for key_values, g in work.groupby(keys, sort=True, observed=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            scope = {"dimension": dimension}
            scope.update({k: str(v) for k, v in zip(keys, key_values)})
            for model, pred_col in prediction_columns(g):
                dim_rows.append(metric_row(g, model, pred_col, scope))
    dimension = pd.DataFrame(dim_rows)

    issue_rows = []
    for (station, split, issue), g in merged.groupby(["station_sn", "eval_split", "issue_time"], sort=True):
        row = {
            "station_sn": str(station),
            "eval_split": str(split),
            "issue_time": str(issue),
            "n_horizons": int(len(g)),
            "truth_min": float(g["true_pv_norm"].min()),
            "truth_max": float(g["true_pv_norm"].max()),
            "truth_mean": float(g["true_pv_norm"].mean()),
            "large_change_ratio": float(g["large_change"].astype(bool).mean()),
        }
        if "abs_err_03_baseline" in g.columns:
            row["baseline_mae"] = float(g["abs_err_03_baseline"].mean())
        for model, pred_col in prediction_columns(g):
            err_col = f"abs_err_{safe_label(model)}"
            if err_col in g.columns:
                row[f"{safe_label(model)}_mae"] = float(g[err_col].mean())
                if "baseline_mae" in row and model != "03_baseline":
                    row[f"{safe_label(model)}_gain"] = row["baseline_mae"] - row[f"{safe_label(model)}_mae"]
            if pred_col in g.columns:
                extras = regression_metric_extras(g, pred_col)
                for metric in ["r2_pv_norm", "acc_installed_capacity", "nrmse_installed_capacity"]:
                    row[f"{safe_label(model)}_{metric}"] = extras[metric]
            for field_col, out_prefix in [
                (f"field_center_{safe_label(model)}", "field_center"),
                (f"field_mean_{safe_label(model)}", "field_mean"),
                (f"field_support_{safe_label(model)}", "field_support"),
            ]:
                if field_col in g.columns:
                    row[f"{safe_label(model)}_{out_prefix}"] = float(g[field_col].mean())
        issue_rows.append(row)
    issue = pd.DataFrame(issue_rows)
    if not issue.empty:
        sort_col = "baseline_mae" if "baseline_mae" in issue.columns else issue.filter(like="_mae").columns[0]
        issue = issue.sort_values(["eval_split", sort_col], ascending=[True, False]).reset_index(drop=True)
    return overall, dimension, issue


def build_improvement(overall: pd.DataFrame, baseline_model: str = "03_baseline") -> pd.DataFrame:
    base = overall[overall["model"] == baseline_model].copy()
    if base.empty:
        return pd.DataFrame()
    rows = []
    for model in sorted([m for m in overall["model"].astype(str).unique() if m != baseline_model]):
        cand = overall[overall["model"] == model].copy()
        joined = base.merge(cand, on=["eval_split", "horizon"], suffixes=("_baseline", "_model"), validate="one_to_one")
        if joined.empty:
            continue
        joined["candidate_model"] = model
        joined["mae_gain"] = joined["mae_pv_norm_baseline"] - joined["mae_pv_norm_model"]
        joined["rmse_gain"] = joined["rmse_pv_norm_baseline"] - joined["rmse_pv_norm_model"]
        joined["r2_gain"] = joined["r2_pv_norm_model"] - joined["r2_pv_norm_baseline"]
        joined["acc_gain"] = joined["acc_installed_capacity_model"] - joined["acc_installed_capacity_baseline"]
        joined["mae_gain_pct"] = np.where(joined["mae_pv_norm_baseline"] > 0, 100.0 * joined["mae_gain"] / joined["mae_pv_norm_baseline"], np.nan)
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_horizon_diversity(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, station, issue), g in merged.groupby(["eval_split", "station_sn", "issue_time"], sort=True):
        if len(g) < 2:
            continue
        truth_std = float(g["true_pv_norm"].astype(float).std(ddof=0))
        for model, pred_col in prediction_columns(g):
            rows.append(
                {
                    "eval_split": str(split),
                    "station_sn": str(station),
                    "issue_time": str(issue),
                    "model": model,
                    "pred_horizon_std": float(g[pred_col].astype(float).std(ddof=0)),
                    "truth_horizon_std": truth_std,
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


def select_samples(merged: pd.DataFrame, count: int, seed: int, strategy: str) -> pd.DataFrame:
    test = merged[merged["eval_split"] == "test"].copy()
    if test.empty or int(count) <= 0:
        return pd.DataFrame()
    rows = []
    for (station, issue), g in test.groupby(["station_sn", "issue_time"], sort=False):
        row = {"station_sn": str(station), "issue_time": str(issue), "truth_mean": float(g["true_pv_norm"].mean())}
        for model, _pred_col in prediction_columns(g):
            err_col = f"abs_err_{safe_label(model)}"
            if err_col in g.columns:
                row[f"{safe_label(model)}_mae"] = float(g[err_col].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n = min(int(count), len(df))
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        return df.loc[np.sort(rng.choice(df.index.to_numpy(), size=n, replace=False))].reset_index(drop=True)
    sort_col = "03_baseline_mae" if "03_baseline_mae" in df.columns else df.filter(like="_mae").columns[0]
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def abs_error_col(model: str) -> str:
    return f"abs_err_{safe_label(model)}"


def save_field_source_figures(merged: pd.DataFrame, samples: pd.DataFrame, out_dir: Path, subdir: str, dpi: int) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(out_dir / subdir)
    test = merged[merged["eval_split"] == "test"].copy()
    hist_cols = history_cols(merged)
    colors = {
        "03_baseline": "#1f77b4",
        "historical_pv_field_patch": "#ff7f0e",
        "future_pv_field_patch": "#d62728",
    }
    rows = []
    for i, sample in samples.reset_index(drop=True).iterrows():
        station = str(sample["station_sn"])
        issue = str(sample["issue_time"])
        sub = test[(test["station_sn"].astype(str) == station) & (test["issue_time"].astype(str) == issue)].sort_values("h")
        if sub.empty:
            continue
        x = sub["h"].to_numpy(np.int64)
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=False)
        if hist_cols:
            hist = sub.iloc[0][hist_cols].to_numpy(np.float64)
            axes[0].plot(np.arange(-len(hist) + 1, 1), hist, color="#7f7f7f", marker=".", linewidth=2.0, label="history PV_norm")
            axes[0].axvline(0, color="0.7", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.3, label="truth")
        for model, pred_col in prediction_columns(sub):
            axes[0].plot(x, sub[pred_col].to_numpy(np.float64), marker="o", linewidth=1.8, color=colors.get(model, "0.4"), label=model)
        axes[0].set_title(f"PV-field source ablation | station={station} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", ncol=2)

        for model, _pred_col in prediction_columns(sub):
            err_col = abs_error_col(model)
            if err_col in sub.columns:
                axes[1].plot(x, sub[err_col].to_numpy(np.float64), marker="o", linewidth=1.8, color=colors.get(model, "0.4"), label=model)
        axes[1].set_ylabel("absolute error")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", ncol=2)

        for model, _pred_col in prediction_columns(sub):
            mean_col = f"field_mean_{safe_label(model)}"
            center_col = f"field_center_{safe_label(model)}"
            if mean_col in sub.columns:
                axes[2].plot(x, sub[mean_col].to_numpy(np.float64), marker="o", linewidth=1.7, color=colors.get(model, "0.4"), label=f"{model} mean")
            if center_col in sub.columns:
                axes[2].plot(x, sub[center_col].to_numpy(np.float64), marker=".", linestyle="--", linewidth=1.2, color=colors.get(model, "0.4"), label=f"{model} center")
        axes[2].set_ylabel("field PV_norm")
        axes[2].set_xlabel("forecast horizon")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"h{k}" for k in x])
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best", ncol=2)
        fig.tight_layout()
        safe_issue = issue.replace(":", "").replace(" ", "_").replace("/", "-")
        path = fig_dir / f"field_source_{i:03d}_{station}_{safe_issue}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        rows.append({"sample_id": int(i), "station_sn": station, "issue_time": issue, "figure": str(path)})
    return pd.DataFrame(rows)


def save_field_source_summary_plot(improvement: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if improvement.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = improvement[(improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))].copy()
    if test.empty:
        return
    labels = ["all", "h1", "h4", "h8"]
    models = sorted(test["candidate_model"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(labels))
    width = min(0.30, 0.78 / max(len(models), 1))
    for j, model in enumerate(models):
        g = test[test["candidate_model"] == model].set_index("horizon").reindex(labels)
        offset = (j - (len(models) - 1) / 2.0) * width
        axes[0].bar(x + offset, g["mae_pv_norm_model"], width=width, label=model)
        axes[1].bar(x + offset, g["mae_gain"], width=width, label=model)
    base = test.drop_duplicates("horizon").set_index("horizon").reindex(labels)
    axes[0].plot(x, base["mae_pv_norm_baseline"], color="black", marker="o", linewidth=1.6, label="03 baseline")
    axes[0].set_title("Test MAE by PV-field source")
    axes[0].set_ylabel("MAE PV_norm")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Positive = better than 03 baseline")
    axes[1].set_ylabel("03 MAE - model MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "field_source_test_mae_summary.png", dpi=int(dpi))
    plt.close(fig)


def run_patch_size_eval(args: argparse.Namespace, baseline: pd.DataFrame | None) -> dict:
    out_dir = ensure_dir(Path(args.out_dir) / "patch_size")
    sizes = parse_sizes(args.patch_sizes)
    overrides = parse_patch_overrides(args)
    patch_details = {}
    source_paths = {}
    for size in sizes:
        path = patch_detail_path(args, overrides, int(size))
        df = patch_eval.read_detail(path, f"patch_{size}x{size}")
        if df is not None:
            patch_details[int(size)] = df
            source_paths[f"patch_{size}x{size}"] = str(path)
    merged = patch_eval.align_details(baseline, patch_details)
    overall, dimension, issue = patch_eval.build_metrics(merged, float(args.large_delta_threshold))
    improvement = patch_eval.build_improvement(overall)
    diversity = patch_eval.build_horizon_diversity(merged)
    random_samples = patch_eval.select_samples(merged, int(args.patch_random_sample_count), int(args.seed), "random")
    worst_samples = patch_eval.select_samples(merged, int(args.patch_worst_sample_count), int(args.seed), "worst")
    random_figures = patch_eval.save_sample_figures(merged, random_samples, out_dir, "random_sample_comparisons", int(args.dpi))
    worst_figures = patch_eval.save_sample_figures(merged, worst_samples, out_dir, "worst_sample_comparisons", int(args.dpi))
    patch_eval.save_summary_plot(improvement, out_dir, int(args.dpi))

    merged.to_csv(out_dir / "patch_size_aligned_detail.csv", index=False)
    overall.to_csv(out_dir / "patch_size_eval_summary.csv", index=False)
    improvement.to_csv(out_dir / "patch_size_improvement_vs_baseline.csv", index=False)
    dimension.to_csv(out_dir / "patch_size_dimension_metrics.csv", index=False)
    issue.to_csv(out_dir / "patch_size_issue_metrics.csv", index=False)
    diversity.to_csv(out_dir / "patch_size_horizon_diversity.csv", index=False)
    random_samples.to_csv(out_dir / "patch_size_random_sample_index.csv", index=False)
    worst_samples.to_csv(out_dir / "patch_size_worst_sample_index.csv", index=False)
    random_figures.to_csv(out_dir / "patch_size_random_sample_figures.csv", index=False)
    worst_figures.to_csv(out_dir / "patch_size_worst_sample_figures.csv", index=False)
    meta = {
        "sizes_requested": sizes,
        "sizes_loaded": sorted(patch_details),
        "baseline_detail_csv": str(args.baseline_detail_csv) if baseline is not None else None,
        "source_paths": source_paths,
        "n_aligned_rows": int(len(merged)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "patch_size_eval_meta.json", meta)
    print("==== Stage 4 patch-size ablation ====")
    print(f"loaded sizes: {sorted(patch_details)}")
    print(f"aligned rows: {len(merged)}")
    show_table(improvement, "patch size")
    return meta


def run_field_source_eval(args: argparse.Namespace, baseline: pd.DataFrame | None) -> dict:
    out_dir = ensure_dir(Path(args.out_dir) / "field_source")
    hist_path = field_detail_path(args, "historical")
    future_path = field_detail_path(args, "future")
    historical = read_detail(hist_path, "historical_pv_field_patch", required=False)
    future = read_detail(future_path, "future_pv_field_patch", required=False)
    candidates = []
    source_paths = {}
    if historical is not None:
        candidates.append(("historical_pv_field_patch", historical))
        source_paths["historical_pv_field_patch"] = str(hist_path)
    if future is not None:
        candidates.append(("future_pv_field_patch", future))
        source_paths["future_pv_field_patch"] = str(future_path)
    if not candidates:
        raise RuntimeError(
            "No field-source eval-detail CSVs were found. Provide --historical_field_detail_csv "
            "and/or --future_field_detail_csv."
        )

    merged = align_field_sources(baseline, candidates, float(args.large_delta_threshold))
    overall, dimension, issue = build_metrics(merged, float(args.large_delta_threshold))
    improvement = build_improvement(overall)
    diversity = build_horizon_diversity(merged)
    random_samples = select_samples(merged, int(args.field_random_sample_count), int(args.seed), "random")
    worst_samples = select_samples(merged, int(args.field_worst_sample_count), int(args.seed), "worst")
    random_figures = save_field_source_figures(merged, random_samples, out_dir, "random_sample_comparisons", int(args.dpi))
    worst_figures = save_field_source_figures(merged, worst_samples, out_dir, "worst_sample_comparisons", int(args.dpi))
    save_field_source_summary_plot(improvement, out_dir, int(args.dpi))

    compact_cols = [
        "station_sn",
        "eval_split",
        "issue_time",
        "h",
        "horizon",
        "true_pv_norm",
        "true_delta_pv_norm",
        "large_change",
    ]
    compact_cols.extend([c for c in merged.columns if c.startswith("pred_") or c.startswith("abs_err_")])
    compact_cols.extend([c for c in merged.columns if c.startswith("field_center_") or c.startswith("field_mean_") or c.startswith("field_support_")])
    merged[list(dict.fromkeys([c for c in compact_cols if c in merged.columns]))].to_csv(out_dir / "field_source_aligned_detail_compact.csv", index=False)
    overall.to_csv(out_dir / "field_source_eval_summary.csv", index=False)
    improvement.to_csv(out_dir / "field_source_improvement_vs_baseline.csv", index=False)
    dimension.to_csv(out_dir / "field_source_dimension_metrics.csv", index=False)
    issue.to_csv(out_dir / "field_source_issue_metrics.csv", index=False)
    diversity.to_csv(out_dir / "field_source_horizon_diversity.csv", index=False)
    random_samples.to_csv(out_dir / "field_source_random_sample_index.csv", index=False)
    worst_samples.to_csv(out_dir / "field_source_worst_sample_index.csv", index=False)
    random_figures.to_csv(out_dir / "field_source_random_sample_figures.csv", index=False)
    worst_figures.to_csv(out_dir / "field_source_worst_sample_figures.csv", index=False)
    meta = {
        "models_loaded": [label for label, _df in candidates],
        "baseline_detail_csv": str(args.baseline_detail_csv) if baseline is not None else None,
        "source_paths": source_paths,
        "n_aligned_rows": int(len(merged)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "field_source_eval_meta.json", meta)
    print("==== Stage 4 PV-field source ablation ====")
    print(f"loaded models: {meta['models_loaded']}")
    print(f"aligned rows: {len(merged)}")
    show_table(improvement, "field source")
    return meta


def show_table(improvement: pd.DataFrame, label: str) -> None:
    show = improvement[(improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))].copy()
    cols = [
        "eval_split",
        "horizon",
        "candidate_model",
        "n_points_baseline",
        "n_points",
        "mae_pv_norm_baseline",
        "mae_pv_norm_model",
        "mae_gain",
        "mae_gain_pct",
        "rmse_pv_norm_baseline",
        "rmse_pv_norm_model",
        "rmse_gain",
    ]
    cols = [c for c in cols if c in show.columns]
    print(f"==== Test {label} improvement ====")
    print(show[cols].to_string(index=False) if not show.empty else "EMPTY")


def main() -> None:
    args = parse_args()
    ensure_stage4_dirs()
    out_dir = ensure_dir(Path(args.out_dir))
    baseline = read_detail(Path(args.baseline_detail_csv), "03_baseline", required=False)

    patch_meta = run_patch_size_eval(args, baseline)
    field_meta = run_field_source_eval(args, baseline)
    write_json(
        out_dir / "eval_suite_meta.json",
        {
            "baseline_detail_csv": str(args.baseline_detail_csv) if baseline is not None else None,
            "patch_size": patch_meta,
            "field_source": field_meta,
        },
    )
    print("==== Stage 4 ablation eval suite complete ====")
    print(f"patch-size outputs : {out_dir / 'patch_size'}")
    print(f"field-source outputs: {out_dir / 'field_source'}")


if __name__ == "__main__":
    main()
