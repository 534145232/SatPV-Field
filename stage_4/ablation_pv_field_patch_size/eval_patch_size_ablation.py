#!/usr/bin/env python3
"""Evaluate Stage 4 PV-field patch-size ablations.

Compares station-level models that differ only by the spatial size of the
predicted future PV-field patch around the target station:

    1x1, 3x3, 5x5, 7x7

The 5x5 default reuses the existing 05l full-model run unless a local
ablation run exists.
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
STAGE4_DIR = SCRIPT_DIR.parent
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_paths import (  # noqa: E402
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
)
from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras


KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
CAPACITY_DETAIL_COLS = ["capacity_raw", "capacity_scale"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root",
        type=Path,
        default=STAGE4_DIR / "runs" / "ablation_pv_field_patch_size",
        help="Ablation run root containing patch_1x1, patch_3x3, patch_7x7.",
    )
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / "per_station_temporal_baseline_eval_detail.csv",
    )
    parser.add_argument(
        "--patch_5_dir",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
        help="Existing 5x5 full-model run directory to reuse.",
    )
    parser.add_argument("--sizes", type=str, default="1,3,5,7")
    parser.add_argument("--out_dir", type=Path, default=STAGE4_DIR / "runs" / "ablation_pv_field_patch_size" / "eval")
    parser.add_argument("--random_sample_count", type=int, default=32)
    parser.add_argument("--worst_sample_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--large_delta_threshold", type=float, default=0.10)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_sizes(text: str) -> list[int]:
    sizes = []
    for part in str(text).split(","):
        part = part.strip().lower().replace("x", "")
        if part:
            sizes.append(int(part))
    return sizes


def detail_path_for_size(args: argparse.Namespace, size: int) -> Path:
    local = Path(args.run_root) / f"patch_{size}x{size}" / DETAIL_NAME
    if local.exists():
        return local
    if int(size) == 5:
        return Path(args.patch_5_dir) / DETAIL_NAME
    return local


def read_detail(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] skip missing patch-size detail for {label}: {path}", flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [
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
    missing = [c for c in required if c not in df.columns]
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


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def history_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")])


def align_details(baseline: pd.DataFrame | None, patch_details: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not patch_details:
        raise RuntimeError("No patch-size details are available.")
    base_size = 5 if 5 in patch_details else sorted(patch_details)[0]
    ref = baseline if baseline is not None else patch_details[base_size]
    ref_cols = ["station_sn", "eval_split", "issue_time", "h", "horizon", "true_pv_norm", "true_delta_pv_norm", "large_change"]
    ref_cols = list(dict.fromkeys(ref_cols + [c for c in CAPACITY_DETAIL_COLS if c in ref.columns]))
    if baseline is not None:
        ref_cols = list(dict.fromkeys(ref_cols + ["pred_pv_norm"] + history_cols(baseline)))
    merged = ref[ref_cols].copy()
    if baseline is not None:
        merged = merged.rename(columns={"pred_pv_norm": "pred_baseline"})
    merged["large_change"] = coerce_bool(merged["large_change"])

    for size, df in sorted(patch_details.items()):
        cols = [
            "station_sn",
            "eval_split",
            "issue_time",
            "h",
            "pred_pv_norm",
            "true_pv_norm",
            "true_delta_pv_norm",
            "large_change",
            "field_patch_center_pv_norm",
            "field_patch_mean_pv_norm",
            "field_patch_support_mean",
        ]
        cols = [c for c in cols if c in df.columns]
        right = df[cols].copy().rename(
            columns={
                "pred_pv_norm": f"pred_patch_{size}",
                "true_pv_norm": f"true_patch_{size}",
                "true_delta_pv_norm": f"true_delta_patch_{size}",
                "large_change": f"large_change_patch_{size}",
                "field_patch_center_pv_norm": f"field_center_patch_{size}",
                "field_patch_mean_pv_norm": f"field_mean_patch_{size}",
                "field_patch_support_mean": f"field_support_patch_{size}",
            }
        )
        before = len(merged)
        merged = merged.merge(right, on=KEY_COLS, how="inner", validate="one_to_one")
        if len(merged) != before:
            print(f"[warn] alignment with patch {size}x{size}: rows {before} -> {len(merged)}", flush=True)
        truth_diff = (merged["true_pv_norm"].astype(float) - merged[f"true_patch_{size}"].astype(float)).abs()
        max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
        if np.isfinite(max_diff) and max_diff > 1.0e-5:
            print(f"[warn] max truth mismatch for patch {size}x{size}: {max_diff:.6g}", flush=True)

    for pred_col in [c for c in merged.columns if c.startswith("pred_")]:
        tag = pred_col.replace("pred_", "")
        err = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        merged[f"err_{tag}"] = err
        merged[f"abs_err_{tag}"] = err.abs()
        merged[f"sq_err_{tag}"] = err**2
    return merged


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
        valid = np.isfinite(pred_delta) & np.isfinite(true_delta)
        sign_valid = valid & (true_delta.abs() >= 1.0e-6)
        delta_sign_acc = float((np.sign(pred_delta[sign_valid]) == np.sign(true_delta[sign_valid])).mean()) if bool(sign_valid.any()) else float("nan")
        delta_corr = float(np.corrcoef(pred_delta[valid], true_delta[valid])[0, 1]) if int(valid.sum()) >= 3 else float("nan")
    else:
        delta_sign_acc = float("nan")
        delta_corr = float("nan")
    row = {
        "model": model,
        "n_points": int(len(df)),
        "mae_pv_norm": float(abs_err.mean()),
        "rmse_pv_norm": float(math.sqrt(float(sq_err.mean()))),
        "bias_pv_norm": float(err.mean()),
        "p95_abs_err_pv_norm": float(abs_err.quantile(0.95)),
        "max_abs_err_pv_norm": float(abs_err.max()),
        "large_change_mae_pv_norm": float(abs_err[large].mean()) if bool(large.any()) else float("nan"),
        "large_change_n": int(large.sum()),
        "delta_sign_accuracy": delta_sign_acc,
        "delta_corr": delta_corr,
        **regression_metric_extras(df, pred_col),
    }
    row.update(scope)
    return row


def prediction_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    if "pred_baseline" in df.columns:
        out.append(("03_baseline", "pred_baseline"))
    for col in sorted([c for c in df.columns if c.startswith("pred_patch_")], key=lambda x: int(x.rsplit("_", 1)[-1])):
        size = int(col.rsplit("_", 1)[-1])
        out.append((f"patch_{size}x{size}", col))
    return out


def model_patch_size(model: str) -> int | None:
    if not str(model).startswith("patch_"):
        return None
    return int(str(model).split("_", 1)[1].split("x", 1)[0])


def abs_error_col_for_model(model: str) -> str:
    if model == "03_baseline":
        return "abs_err_baseline"
    size = model_patch_size(model)
    if size is None:
        return f"abs_err_{model}"
    return f"abs_err_patch_{size}"


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
            "truth_min": float(g["true_pv_norm"].min()),
            "truth_max": float(g["true_pv_norm"].max()),
            "truth_mean": float(g["true_pv_norm"].mean()),
        }
        if "abs_err_baseline" in g.columns:
            row["baseline_mae"] = float(g["abs_err_baseline"].mean())
        for model, pred_col in prediction_columns(g):
            extras = regression_metric_extras(g, pred_col)
            for metric in ["r2_pv_norm", "acc_installed_capacity", "nrmse_installed_capacity"]:
                row[f"{model}_{metric}"] = extras[metric]
        for size in sorted([int(c.rsplit("_", 1)[-1]) for c in g.columns if c.startswith("abs_err_patch_")]):
            row[f"patch_{size}x{size}_mae"] = float(g[f"abs_err_patch_{size}"].mean())
            if "baseline_mae" in row:
                row[f"patch_{size}x{size}_gain"] = row["baseline_mae"] - row[f"patch_{size}x{size}_mae"]
            mean_col = f"field_mean_patch_{size}"
            center_col = f"field_center_patch_{size}"
            if mean_col in g.columns:
                row[f"patch_{size}x{size}_field_mean"] = float(g[mean_col].mean())
            if center_col in g.columns:
                row[f"patch_{size}x{size}_field_center"] = float(g[center_col].mean())
        issue_rows.append(row)
    issue = pd.DataFrame(issue_rows)
    if not issue.empty:
        sort_col = "baseline_mae" if "baseline_mae" in issue.columns else issue.filter(like="_mae").columns[0]
        issue = issue.sort_values(["eval_split", sort_col], ascending=[True, False]).reset_index(drop=True)
    return overall, dimension, issue


def build_improvement(overall: pd.DataFrame) -> pd.DataFrame:
    base = overall[overall["model"] == "03_baseline"].copy()
    if base.empty:
        return pd.DataFrame()
    rows = []
    for model in sorted([m for m in overall["model"].astype(str).unique() if m != "03_baseline"]):
        cand = overall[overall["model"] == model].copy()
        joined = base.merge(cand, on=["eval_split", "horizon"], suffixes=("_baseline", "_model"), validate="one_to_one")
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
    if test.empty or count <= 0:
        return pd.DataFrame()
    issue = []
    for (station, issue_time), g in test.groupby(["station_sn", "issue_time"], sort=False):
        row = {"station_sn": str(station), "issue_time": str(issue_time), "truth_mean": float(g["true_pv_norm"].mean())}
        if "abs_err_baseline" in g.columns:
            row["baseline_mae"] = float(g["abs_err_baseline"].mean())
        for size in sorted([int(c.rsplit("_", 1)[-1]) for c in g.columns if c.startswith("abs_err_patch_")]):
            row[f"patch_{size}x{size}_mae"] = float(g[f"abs_err_patch_{size}"].mean())
        issue.append(row)
    df = pd.DataFrame(issue)
    n = min(int(count), len(df))
    if n <= 0:
        return pd.DataFrame()
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(df.index.to_numpy(), size=n, replace=False))
        return df.loc[idx].reset_index(drop=True)
    sort_col = "baseline_mae" if "baseline_mae" in df.columns else df.filter(like="_mae").columns[0]
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def save_sample_figures(merged: pd.DataFrame, samples: pd.DataFrame, out_dir: Path, subdir: str, dpi: int) -> pd.DataFrame:
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
        "patch_1x1": "#9467bd",
        "patch_3x3": "#ff7f0e",
        "patch_5x5": "#d62728",
        "patch_7x7": "#2ca02c",
    }
    rows = []
    for i, sample in samples.reset_index(drop=True).iterrows():
        station = str(sample["station_sn"])
        issue = str(sample["issue_time"])
        sub = test[(test["station_sn"].astype(str) == station) & (test["issue_time"].astype(str) == issue)].sort_values("h")
        if sub.empty:
            continue
        x = sub["h"].to_numpy(np.int64)
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=False)
        if hist_cols:
            hist = sub.iloc[0][hist_cols].to_numpy(np.float64)
            hist_x = np.arange(-len(hist) + 1, 1)
            axes[0].plot(hist_x, hist, color="#7f7f7f", marker=".", linewidth=2.0, label="history PV_norm")
            axes[0].axvline(0, color="0.7", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.3, label="truth")
        for model, pred_col in prediction_columns(sub):
            axes[0].plot(x, sub[pred_col].to_numpy(np.float64), marker="o", linewidth=1.8, color=colors.get(model, "0.4"), label=model)
        axes[0].set_title(f"PV-field patch-size ablation | station={station} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", ncol=3)

        for model, _pred_col in prediction_columns(sub):
            abs_col = abs_error_col_for_model(model)
            if abs_col in sub.columns:
                axes[1].plot(x, sub[abs_col].to_numpy(np.float64), marker="o", linewidth=1.8, color=colors.get(model, "0.4"), label=model)
        axes[1].set_ylabel("absolute error")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", ncol=3)

        for size in sorted([int(c.rsplit("_", 1)[-1]) for c in sub.columns if c.startswith("field_mean_patch_")]):
            axes[2].plot(x, sub[f"field_mean_patch_{size}"].to_numpy(np.float64), marker="o", linewidth=1.7, color=colors.get(f"patch_{size}x{size}", "0.4"), label=f"{size}x{size} mean")
        axes[2].set_ylabel("field patch mean")
        axes[2].set_xlabel("forecast horizon")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"h{k}" for k in x])
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best", ncol=3)
        fig.tight_layout()

        safe_issue = issue.replace(":", "").replace(" ", "_").replace("/", "-")
        path = fig_dir / f"patch_size_{i:03d}_{station}_{safe_issue}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        rows.append({"sample_id": int(i), "station_sn": station, "issue_time": issue, "figure": str(path)})
    return pd.DataFrame(rows)


def save_summary_plot(improvement: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if improvement.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = improvement[(improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))].copy()
    if test.empty:
        return
    labels = ["all", "h1", "h4", "h8"]
    models = sorted(test["candidate_model"].astype(str).unique(), key=lambda s: model_patch_size(s) or 999)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(labels))
    width = min(0.18, 0.78 / max(len(models), 1))
    for j, model in enumerate(models):
        g = test[test["candidate_model"] == model].set_index("horizon").reindex(labels)
        offset = (j - (len(models) - 1) / 2.0) * width
        axes[0].bar(x + offset, g["mae_pv_norm_model"], width=width, label=model)
        axes[1].bar(x + offset, g["mae_gain"], width=width, label=model)
    base = test.drop_duplicates("horizon").set_index("horizon").reindex(labels)
    axes[0].plot(x, base["mae_pv_norm_baseline"], color="black", marker="o", linewidth=1.6, label="03 baseline")
    axes[0].set_title("Test MAE by patch size")
    axes[0].set_ylabel("MAE PV_norm")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", ncol=2)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Positive = better than 03 baseline")
    axes[1].set_ylabel("03 MAE - patch MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "patch_size_test_mae_summary.png", dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    sizes = parse_sizes(args.sizes)
    baseline = read_detail(Path(args.baseline_detail_csv), "03_baseline") if Path(args.baseline_detail_csv).exists() else None
    patch_details = {}
    source_paths = {}
    for size in sizes:
        path = detail_path_for_size(args, size)
        df = read_detail(path, f"patch_{size}x{size}")
        if df is not None:
            patch_details[int(size)] = df
            source_paths[f"patch_{size}x{size}"] = str(path)
    merged = align_details(baseline, patch_details)
    overall, dimension, issue = build_metrics(merged, float(args.large_delta_threshold))
    improvement = build_improvement(overall)
    diversity = build_horizon_diversity(merged)
    random_samples = select_samples(merged, int(args.random_sample_count), int(args.seed), "random")
    worst_samples = select_samples(merged, int(args.worst_sample_count), int(args.seed), "worst")
    random_figures = save_sample_figures(merged, random_samples, out_dir, "random_sample_comparisons", int(args.dpi))
    worst_figures = save_sample_figures(merged, worst_samples, out_dir, "worst_sample_comparisons", int(args.dpi))
    save_summary_plot(improvement, out_dir, int(args.dpi))

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
    write_json(
        out_dir / "patch_size_eval_meta.json",
        {
            "sizes_requested": sizes,
            "sizes_loaded": sorted(patch_details),
            "baseline_detail_csv": str(args.baseline_detail_csv) if baseline is not None else None,
            "source_paths": source_paths,
            "n_aligned_rows": int(len(merged)),
            "out_dir": str(out_dir),
        },
    )

    print("==== Stage 4 PV-field patch-size ablation ====")
    print(f"loaded sizes: {sorted(patch_details)}")
    print(f"aligned rows: {len(merged)}")
    show = improvement[(improvement["eval_split"] == "test") & (improvement["horizon"].isin(["all", "h1", "h4", "h8"]))].copy()
    cols = [
        "eval_split",
        "horizon",
        "candidate_model",
        "n_points_baseline",
        "mae_pv_norm_baseline",
        "mae_pv_norm_model",
        "mae_gain",
        "mae_gain_pct",
        "rmse_pv_norm_baseline",
        "rmse_pv_norm_model",
        "rmse_gain",
    ]
    print(show[[c for c in cols if c in show.columns]].to_string(index=False) if not show.empty else "EMPTY")
    if not diversity.empty:
        print("==== Horizon diversity ====")
        print(diversity[diversity["eval_split"] == "test"].to_string(index=False))
    print(f"saved: {out_dir}")
    print(f"figures: {out_dir / 'random_sample_comparisons'}")


if __name__ == "__main__":
    main()
