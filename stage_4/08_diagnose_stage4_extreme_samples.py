"""Diagnose extreme Stage 4 station-forecast samples.

This script inspects the worst samples from the Stage 4 03-vs-05 comparison.
It is meant for debugging large visual errors, not for model training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stage4_paths import (
    ARTIFACT_ROOT,
    PER_STATION_03_VS_05_EVAL_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    TRAIN_PV_DIR,
    VAL_PV_DIR,
    ensure_stage4_dirs,
)
from stage4_metrics import regression_metric_extras


SCRIPT_DIR = Path(__file__).resolve().parent
OLD_EXPERIMENT_ROOT = Path(
    os.environ.get(
        "SATPV_LEGACY_EXPERIMENT_ROOT",
        ARTIFACT_ROOT / "legacy" / "station_field_gain_lhs50",
    )
).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison_csv",
        type=Path,
        default=PER_STATION_03_VS_05_EVAL_DIR / "stage4_03_vs_05_aligned_detail_compact.csv",
        help="Compact aligned comparison CSV produced by 07_eval_compare_stage4_03_vs_05.py.",
    )
    parser.add_argument(
        "--new_baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / "per_station_temporal_baseline_eval_detail.csv",
    )
    parser.add_argument(
        "--new_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
    )
    parser.add_argument(
        "--old_baseline_detail_csv",
        type=Path,
        default=OLD_EXPERIMENT_ROOT / "03_per_station_temporal_baseline_seed2026" / "per_station_temporal_baseline_eval_detail.csv",
    )
    parser.add_argument(
        "--old_field_detail_csv",
        type=Path,
        default=OLD_EXPERIMENT_ROOT
        / "05l_per_station_temporal_swr_anchor04l_field_patch_seed2026"
        / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
    )
    parser.add_argument("--train_pv_dir", type=Path, default=TRAIN_PV_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=VAL_PV_DIR)
    parser.add_argument("--out_dir", type=Path, default=PER_STATION_03_VS_05_EVAL_DIR / "extreme_sample_diagnosis")
    parser.add_argument("--station_sn", action="append", default=[], help="Specific station_sn to inspect. Can repeat.")
    parser.add_argument("--issue_time", action="append", default=[], help="Specific issue_time to inspect. Can repeat.")
    parser.add_argument("--top_n", type=int, default=24, help="If no station/issue list is given, inspect top-N worst test issues.")
    parser.add_argument("--bad_abs_error_threshold", type=float, default=0.30)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_stage4_baseline_module():
    path = SCRIPT_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_03_temporal_baseline_for_diagnosis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def station_loader_args(args: argparse.Namespace, module) -> SimpleNamespace:
    defaults = dict(module.PYCHARM_DEFAULTS)
    defaults.update({"train_pv_dir": args.train_pv_dir, "val_pv_dir": args.val_pv_dir})
    return SimpleNamespace(**defaults)


def read_csv_if_exists(path: Path, dtype: dict | None = None) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, dtype=dtype)
    return None


def load_comparison(args: argparse.Namespace) -> pd.DataFrame:
    if args.comparison_csv.exists():
        df = pd.read_csv(args.comparison_csv, dtype={"station_sn": str})
    else:
        raise FileNotFoundError(
            f"Comparison CSV not found: {args.comparison_csv}\n"
            "Run run_stage4_eval_compare_03_vs_05.sh first."
        )
    required = ["station_sn", "eval_split", "issue_time", "h", "true_pv_norm", "pred_baseline", "pred_field_patch"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{args.comparison_csv} missing columns: {missing}")
    df["station_sn"] = df["station_sn"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    df["h"] = df["h"].astype(int)
    for pred in ["baseline", "field_patch"]:
        err_col = f"err_{pred}"
        abs_col = f"abs_err_{pred}"
        if err_col not in df.columns:
            df[err_col] = df[f"pred_{pred}"].astype(float) - df["true_pv_norm"].astype(float)
        if abs_col not in df.columns:
            df[abs_col] = df[err_col].abs()
    return df


def select_samples(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    test = df[df["eval_split"].astype(str) == "test"].copy()
    if args.station_sn or args.issue_time:
        out = test
        if args.station_sn:
            stations = set(str(v) for v in args.station_sn)
            out = out[out["station_sn"].astype(str).isin(stations)]
        if args.issue_time:
            issues = set(str(pd.Timestamp(v)) for v in args.issue_time)
            out = out[out["issue_time"].map(lambda v: str(pd.Timestamp(v))).isin(issues)]
        if out.empty:
            raise RuntimeError("No rows matched the requested --station_sn/--issue_time filters.")
        return out[["station_sn", "issue_time"]].drop_duplicates().reset_index(drop=True)
    score = (
        test.groupby(["station_sn", "issue_time"], sort=False)
        .agg(
            baseline_mae=("abs_err_baseline", "mean"),
            field_patch_mae=("abs_err_field_patch", "mean"),
            max_truth=("true_pv_norm", "max"),
            min_truth=("true_pv_norm", "min"),
            mean_field_center=("field_patch_center_pv_norm", "mean") if "field_patch_center_pv_norm" in test.columns else ("true_pv_norm", "mean"),
        )
        .reset_index()
    )
    score["worst_mae"] = score[["baseline_mae", "field_patch_mae"]].max(axis=1)
    score["truth_range"] = score["max_truth"] - score["min_truth"]
    return score.sort_values(["worst_mae", "truth_range"], ascending=False).head(int(args.top_n))[["station_sn", "issue_time"]]


def station_day_frame(module, loader_args: SimpleNamespace, station_sn: str, issue_time: str) -> tuple[pd.DataFrame, dict]:
    station_df, meta = module.load_station_table(station_sn, loader_args)
    issue = pd.Timestamp(issue_time)
    start = issue.normalize() + pd.Timedelta(hours=7)
    end = issue.normalize() + pd.Timedelta(hours=16, minutes=45)
    day = station_df.loc[(station_df.index >= start) & (station_df.index <= end)].copy()
    day = day.reset_index().rename(columns={"index": "timestamp"})
    return day, meta


def old_same_clock_rows(old_df: pd.DataFrame | None, station_sn: str, issue_time: str, prefix: str) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return pd.DataFrame()
    df = old_df.copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    issue = str(pd.Timestamp(issue_time))
    sub = df[(df["station_sn"] == str(station_sn)) & (df["issue_time"].map(lambda v: str(pd.Timestamp(v))) == issue)].copy()
    if sub.empty:
        return sub
    keep = ["station_sn", "issue_time", "h", "horizon", "pred_pv_norm", "true_pv_norm", "abs_err_pv_norm"]
    keep = [c for c in keep if c in sub.columns]
    sub = sub[keep].copy()
    sub = sub.rename(
        columns={
            "pred_pv_norm": f"{prefix}_pred_pv_norm",
            "true_pv_norm": f"{prefix}_true_pv_norm",
            "abs_err_pv_norm": f"{prefix}_abs_err_pv_norm",
        }
    )
    return sub.sort_values("h").reset_index(drop=True)


def build_station_extreme_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    test = df[df["eval_split"].astype(str) == "test"].copy()
    rows = []
    for station, g in test.groupby("station_sn", sort=True):
        row = {"station_sn": str(station), "n_points": int(len(g)), "n_issues": int(g["issue_time"].nunique())}
        for name in ["baseline", "field_patch"]:
            abs_err = g[f"abs_err_{name}"].astype(float)
            row[f"{name}_mae"] = float(abs_err.mean())
            row[f"{name}_p95_abs_err"] = float(abs_err.quantile(0.95))
            row[f"{name}_bad_rate"] = float((abs_err >= float(threshold)).mean())
            pred_col = f"pred_{name}"
            if pred_col in g.columns:
                extras = regression_metric_extras(g, pred_col)
                row[f"{name}_r2_pv_norm"] = extras["r2_pv_norm"]
                row[f"{name}_acc_installed_capacity"] = extras["acc_installed_capacity"]
                row[f"{name}_nrmse_installed_capacity"] = extras["nrmse_installed_capacity"]
        if "field_patch_center_pv_norm" in g.columns:
            valid = np.isfinite(g["field_patch_center_pv_norm"].astype(float)) & np.isfinite(g["true_pv_norm"].astype(float))
            row["field_center_truth_corr"] = float(np.corrcoef(g.loc[valid, "field_patch_center_pv_norm"], g.loc[valid, "true_pv_norm"])[0, 1]) if int(valid.sum()) > 3 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("baseline_mae", ascending=False).reset_index(drop=True)


def save_sample_plot(
    sample_rows: pd.DataFrame,
    day: pd.DataFrame,
    old_baseline: pd.DataFrame,
    old_field: pd.DataFrame,
    station_sn: str,
    issue_time: str,
    out_path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    issue = pd.Timestamp(issue_time)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=False)
    if not day.empty:
        axes[0].plot(pd.to_datetime(day["timestamp"]), day["pv_norm"], color="0.45", marker=".", linewidth=1.8, label="raw station PV_norm")
    axes[0].axvline(issue, color="black", linestyle="--", linewidth=1.0, label="new issue T")
    future_times = [issue + pd.Timedelta(minutes=15 * (int(h) - 1)) for h in sample_rows["h"].to_numpy()]
    axes[0].plot(future_times, sample_rows["true_pv_norm"], color="black", marker="o", linewidth=2.2, label="new truth")
    axes[0].plot(future_times, sample_rows["pred_baseline"], color="#1f77b4", marker="o", linewidth=1.8, label="new 03")
    axes[0].plot(future_times, sample_rows["pred_field_patch"], color="#d62728", marker="o", linewidth=1.8, label="new 05")
    if "field_patch_center_pv_norm" in sample_rows.columns:
        axes[0].plot(future_times, sample_rows["field_patch_center_pv_norm"], color="#2ca02c", marker=".", linewidth=1.4, label="field center")
    axes[0].set_title(f"Extreme Stage 4 sample diagnosis | station={station_sn} | issue={issue}")
    axes[0].set_ylabel("PV_norm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=2)

    x = sample_rows["h"].to_numpy(np.int64)
    axes[1].bar(x - 0.18, sample_rows["err_baseline"], width=0.36, color="#1f77b4", alpha=0.85, label="new 03 error")
    axes[1].bar(x + 0.18, sample_rows["err_field_patch"], width=0.36, color="#d62728", alpha=0.85, label="new 05 error")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("pred - truth")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"h{k}" for k in x])
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    # Old same-clock rows have future h1 = T+15min, so compare visually against new h2..h8.
    if not old_baseline.empty:
        axes[2].plot(old_baseline["h"], old_baseline.get("old03_true_pv_norm"), color="black", linestyle="--", marker=".", label="old truth same-clock")
        axes[2].plot(old_baseline["h"], old_baseline.get("old03_pred_pv_norm"), color="#1f77b4", linestyle="--", marker=".", label="old 03 same-clock")
    if not old_field.empty:
        axes[2].plot(old_field["h"], old_field.get("old05_pred_pv_norm"), color="#d62728", linestyle="--", marker=".", label="old 05 same-clock")
    axes[2].set_title("Old same issue_time rows, if available. Old h1 starts at T+15min.")
    axes[2].set_xlabel("old forecast horizon")
    axes[2].set_ylabel("PV_norm")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_stage4_dirs()
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    day_dir = ensure_dir(out_dir / "station_day_series")

    comparison = load_comparison(args)
    samples = select_samples(comparison, args)
    station_summary = build_station_extreme_summary(comparison, float(args.bad_abs_error_threshold))
    station_summary.to_csv(out_dir / "stage4_station_extreme_error_summary.csv", index=False)

    old_baseline = read_csv_if_exists(args.old_baseline_detail_csv, dtype={"station_sn": str})
    old_field = read_csv_if_exists(args.old_field_detail_csv, dtype={"station_sn": str})
    module = load_stage4_baseline_module()
    loader_args = station_loader_args(args, module)

    diag_rows = []
    old_rows = []
    meta_rows = []
    for sample_id, sample in samples.reset_index(drop=True).iterrows():
        station_sn = str(sample["station_sn"])
        issue_time = str(pd.Timestamp(sample["issue_time"]))
        sub = comparison[
            (comparison["station_sn"].astype(str) == station_sn)
            & (comparison["issue_time"].map(lambda v: str(pd.Timestamp(v))) == issue_time)
            & (comparison["eval_split"].astype(str) == "test")
        ].sort_values("h")
        if sub.empty:
            continue

        day, meta = station_day_frame(module, loader_args, station_sn, issue_time)
        day_file = day_dir / f"station_day_{sample_id:03d}_{station_sn}_{issue_time.replace(':', '').replace(' ', '_')}.csv"
        day.to_csv(day_file, index=False)
        meta_rows.append({"sample_id": int(sample_id), **meta})

        old03 = old_same_clock_rows(old_baseline, station_sn, issue_time, "old03")
        old05 = old_same_clock_rows(old_field, station_sn, issue_time, "old05")
        if not old03.empty:
            old03.insert(0, "sample_id", int(sample_id))
            old_rows.append(old03)
        if not old05.empty:
            old05.insert(0, "sample_id", int(sample_id))
            old_rows.append(old05)

        out_png = fig_dir / f"extreme_diag_{sample_id:03d}_{station_sn}_{issue_time.replace(':', '').replace(' ', '_')}.png"
        save_sample_plot(sub, day, old03, old05, station_sn, issue_time, out_png, int(args.dpi))

        summary = {
            "sample_id": int(sample_id),
            "station_sn": station_sn,
            "issue_time": issue_time,
            "new_baseline_mae": float(sub["abs_err_baseline"].mean()),
            "new_field_patch_mae": float(sub["abs_err_field_patch"].mean()),
            "truth_min": float(sub["true_pv_norm"].min()),
            "truth_max": float(sub["true_pv_norm"].max()),
            "history_last": float(sub[[c for c in sub.columns if c.startswith("hist_pv_norm_")]].iloc[0, -1]) if any(c.startswith("hist_pv_norm_") for c in sub.columns) else float("nan"),
            "field_center_mean": float(sub["field_patch_center_pv_norm"].mean()) if "field_patch_center_pv_norm" in sub.columns else float("nan"),
            "station_day_csv": str(day_file),
            "figure": str(out_png),
            "old03_same_issue_available": bool(not old03.empty),
            "old05_same_issue_available": bool(not old05.empty),
        }
        diag_rows.append(summary)

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out_dir / "stage4_extreme_sample_diagnosis.csv", index=False)
    if old_rows:
        pd.concat(old_rows, ignore_index=True).to_csv(out_dir / "old_experiment_same_clock_rows.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "station_load_meta_for_extreme_samples.csv", index=False)
    write_json(
        out_dir / "diagnosis_meta.json",
        {
            "comparison_csv": str(args.comparison_csv),
            "old_baseline_detail_csv": str(args.old_baseline_detail_csv),
            "old_field_detail_csv": str(args.old_field_detail_csv),
            "note": (
                "Old station_field_gain_new issue_time uses history ending at T and future starting at T+15min; "
                "Stage 4 now uses history ending at T-15min and future starting at T."
            ),
        },
    )

    print("==== Stage 4 extreme-sample diagnosis ====")
    if not diag.empty:
        print(diag[["sample_id", "station_sn", "issue_time", "new_baseline_mae", "new_field_patch_mae", "truth_min", "truth_max", "field_center_mean"]].to_string(index=False))
    print(f"station summary: {out_dir / 'stage4_station_extreme_error_summary.csv'}")
    print(f"sample diagnosis: {out_dir / 'stage4_extreme_sample_diagnosis.csv'}")
    print(f"figures: {fig_dir}")


if __name__ == "__main__":
    main()
