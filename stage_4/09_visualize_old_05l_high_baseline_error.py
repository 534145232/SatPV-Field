"""Visualize old-project 05l behavior on high-baseline-error samples.

This script answers a specific diagnostic question:
  When the old station-level temporal baseline has very large test error,
  does the old 05l field-patch model show similarly extreme failures?

It only reads saved eval_detail CSVs from the old project and does not retrain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_paths import ARTIFACT_ROOT, PER_STATION_03_VS_05_EVAL_DIR, PROJECT_ROOT, ensure_stage4_dirs
from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras


OLD_EXPERIMENT_ROOT = Path(
    os.environ.get(
        "SATPV_LEGACY_EXPERIMENT_ROOT",
        ARTIFACT_ROOT / "legacy" / "station_field_gain_lhs50",
    )
).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old_baseline_detail_csv",
        type=Path,
        default=OLD_EXPERIMENT_ROOT / "03_per_station_temporal_baseline_seed2026" / "per_station_temporal_baseline_eval_detail.csv",
        help="Old-project 03 baseline eval_detail CSV.",
    )
    parser.add_argument(
        "--old_05l_detail_csv",
        type=Path,
        default=OLD_EXPERIMENT_ROOT
        / "05l_per_station_temporal_swr_anchor04l_field_patch_seed2026"
        / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
        help="Old-project 05l eval_detail CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PER_STATION_03_VS_05_EVAL_DIR / "old_05l_high_baseline_error",
    )
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--sample_count", type=int, default=40)
    parser.add_argument("--min_baseline_mae", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_audit(path: Path) -> dict:
    path = Path(path)
    info = {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": None,
        "mtime": None,
        "is_legacy_experiment_path": OLD_EXPERIMENT_ROOT == path or OLD_EXPERIMENT_ROOT in path.parents,
        "is_current_project_path": PROJECT_ROOT == path or PROJECT_ROOT in path.parents,
    }
    if path.exists():
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
        info["mtime"] = pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    return info


def load_nearby_run_meta(detail_csv: Path) -> dict:
    detail_csv = Path(detail_csv)
    candidates = [
        detail_csv.parent / "run_meta.json",
        detail_csv.parent / "field_patch_stats.json",
    ]
    out: dict[str, object] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            out[path.name] = payload
        except Exception as exc:
            out[path.name] = {"read_error": str(exc)}
    return out


def read_detail(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing old {label} eval detail: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = ["station_sn", "eval_split", "issue_time", "h", "pred_pv_norm", "true_pv_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].map(lambda value: str(pd.Timestamp(value)))
    df["h"] = pd.to_numeric(df["h"], errors="coerce").astype("Int64")
    df["horizon"] = "h" + df["h"].astype(str)
    df["pred_pv_norm"] = pd.to_numeric(df["pred_pv_norm"], errors="coerce")
    df["true_pv_norm"] = pd.to_numeric(df["true_pv_norm"], errors="coerce")
    df["abs_err_pv_norm"] = (df["pred_pv_norm"] - df["true_pv_norm"]).abs()
    df["sq_err_pv_norm"] = (df["pred_pv_norm"] - df["true_pv_norm"]) ** 2
    df = attach_capacity_from_station_meta(df, path)
    return df.dropna(subset=["h", "pred_pv_norm", "true_pv_norm"]).reset_index(drop=True)


def align_old_details(baseline: pd.DataFrame, old05l: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    key = ["station_sn", "eval_split", "issue_time", "h"]
    base_cols = key + ["horizon", "pred_pv_norm", "true_pv_norm", "abs_err_pv_norm", "sq_err_pv_norm"]
    base_cols += [c for c in ["capacity_raw", "capacity_scale"] if c in baseline.columns]
    hist_cols = sorted([c for c in baseline.columns if c.startswith("hist_pv_norm_")])
    future_cols = [c for c in baseline.columns if c.startswith("future_") and (c.endswith("_z") or c in {"future_solar_elevation", "future_tod_sin", "future_tod_cos"})]
    base_cols += hist_cols + future_cols
    base = baseline[baseline["eval_split"] == str(eval_split)][[c for c in base_cols if c in baseline.columns]].copy()
    base = base.rename(
        columns={
            "pred_pv_norm": "old03_pred_pv_norm",
            "abs_err_pv_norm": "old03_abs_err_pv_norm",
            "sq_err_pv_norm": "old03_sq_err_pv_norm",
        }
    )
    extra_cols = [
        "field_patch_center_pv_norm",
        "field_patch_mean_pv_norm",
        "field_patch_support_mean",
    ]
    old05_cols = key + ["pred_pv_norm", "true_pv_norm", "abs_err_pv_norm", "sq_err_pv_norm"] + [c for c in extra_cols if c in old05l.columns]
    cand = old05l[old05l["eval_split"] == str(eval_split)][old05_cols].copy()
    cand = cand.rename(
        columns={
            "pred_pv_norm": "old05l_pred_pv_norm",
            "true_pv_norm": "old05l_true_pv_norm",
            "abs_err_pv_norm": "old05l_abs_err_pv_norm",
            "sq_err_pv_norm": "old05l_sq_err_pv_norm",
        }
    )
    merged = base.merge(cand, on=key, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No overlapping old 03/05l rows after alignment.")
    truth_diff = (merged["true_pv_norm"] - merged["old05l_true_pv_norm"]).abs()
    if float(truth_diff.max()) > 1.0e-5:
        print(f"[warn] old truth mismatch max={float(truth_diff.max()):.6g}", flush=True)
    merged["old03_err_pv_norm"] = merged["old03_pred_pv_norm"] - merged["true_pv_norm"]
    merged["old05l_err_pv_norm"] = merged["old05l_pred_pv_norm"] - merged["true_pv_norm"]
    merged["old05l_gain_abs_err"] = merged["old03_abs_err_pv_norm"] - merged["old05l_abs_err_pv_norm"]
    return merged


def choose_high_baseline_samples(merged: pd.DataFrame, sample_count: int, min_baseline_mae: float, seed: int) -> pd.DataFrame:
    issue = (
        merged.groupby(["station_sn", "issue_time"], sort=False)
        .agg(
            old03_mae=("old03_abs_err_pv_norm", "mean"),
            old05l_mae=("old05l_abs_err_pv_norm", "mean"),
            old03_rmse=("old03_sq_err_pv_norm", lambda x: math.sqrt(float(np.mean(x)))),
            old05l_rmse=("old05l_sq_err_pv_norm", lambda x: math.sqrt(float(np.mean(x)))),
            truth_mean=("true_pv_norm", "mean"),
            truth_min=("true_pv_norm", "min"),
            truth_max=("true_pv_norm", "max"),
        )
        .reset_index()
    )
    issue["mae_gain"] = issue["old03_mae"] - issue["old05l_mae"]
    high = issue[issue["old03_mae"] >= float(min_baseline_mae)].copy()
    if high.empty:
        high = issue.copy()
    high = high.sort_values(["old03_mae", "old05l_mae"], ascending=False).reset_index(drop=True)
    if len(high) <= int(sample_count):
        return high
    top_n = max(int(sample_count) // 2, 1)
    top = high.head(top_n)
    pool = high.iloc[top_n:]
    rng = np.random.default_rng(int(seed))
    rand_n = int(sample_count) - len(top)
    rand_idx = rng.choice(pool.index.to_numpy(), size=rand_n, replace=False) if len(pool) > rand_n else pool.index.to_numpy()
    return pd.concat([top, pool.loc[rand_idx]], ignore_index=True)


def metric_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, g in list(merged.groupby("horizon", sort=True)) + [("all", merged)]:
        for model, pred_col, err_col in [
            ("old03_baseline", "old03_pred_pv_norm", "old03_err_pv_norm"),
            ("old05l_field_patch", "old05l_pred_pv_norm", "old05l_err_pv_norm"),
        ]:
            err = g[err_col].astype(float)
            row = {
                "model": model,
                "eval_split": str(g["eval_split"].iloc[0]) if not g.empty else "",
                "horizon": horizon,
                "n_points": int(len(g)),
                "mae_pv_norm": float(err.abs().mean()),
                "rmse_pv_norm": float(math.sqrt(float((err**2).mean()))),
                "bias_pv_norm": float(err.mean()),
            }
            row.update(regression_metric_extras(g, pred_col))
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(merged: pd.DataFrame, samples: pd.DataFrame, out_dir: Path, dpi: int) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(out_dir / "sample_figures")
    hist_cols = sorted([c for c in merged.columns if c.startswith("hist_pv_norm_")])
    rows = []
    for i, sample in samples.reset_index(drop=True).iterrows():
        station = str(sample["station_sn"])
        issue = str(pd.Timestamp(sample["issue_time"]))
        sub = merged[
            (merged["station_sn"].astype(str) == station)
            & (merged["issue_time"].map(lambda value: str(pd.Timestamp(value))) == issue)
        ].sort_values("h")
        if sub.empty:
            continue
        x = sub["h"].to_numpy(np.int64)
        fig, axes = plt.subplots(3, 1, figsize=(9.8, 8.8), sharex=False)
        if hist_cols:
            hist = sub.iloc[0][hist_cols].to_numpy(np.float64)
            axes[0].plot(np.arange(-len(hist) + 1, 1), hist, color="0.5", marker=".", linewidth=2.0, label="history PV_norm")
            axes[0].axvline(0, color="0.7", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.3, label="truth")
        axes[0].plot(x, sub["old03_pred_pv_norm"].to_numpy(np.float64), color="#1f77b4", marker="o", linewidth=2.0, label="old 03 baseline")
        axes[0].plot(x, sub["old05l_pred_pv_norm"].to_numpy(np.float64), color="#d62728", marker="o", linewidth=2.0, label="old 05l")
        if "field_patch_center_pv_norm" in sub.columns:
            axes[0].plot(
                x,
                sub["field_patch_center_pv_norm"].to_numpy(np.float64),
                color="#2ca02c",
                marker=".",
                linewidth=1.5,
                label="old 05l field center",
            )
        axes[0].set_title(f"Old project 05l on high-baseline-error sample | station={station} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", ncol=2)

        width = 0.36
        axes[1].bar(x - width / 2, sub["old03_err_pv_norm"], width=width, color="#1f77b4", alpha=0.85, label="old 03 error")
        axes[1].bar(x + width / 2, sub["old05l_err_pv_norm"], width=width, color="#d62728", alpha=0.85, label="old 05l error")
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("pred - truth")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        axes[2].plot(x, sub["old03_abs_err_pv_norm"], color="#1f77b4", marker="o", label="old 03 abs error")
        axes[2].plot(x, sub["old05l_abs_err_pv_norm"], color="#d62728", marker="o", label="old 05l abs error")
        if "field_patch_mean_pv_norm" in sub.columns:
            ax2 = axes[2].twinx()
            ax2.plot(x, sub["field_patch_mean_pv_norm"], color="#2ca02c", linestyle="--", marker=".", label="field patch mean")
            ax2.set_ylabel("field patch mean")
            h1, l1 = axes[2].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            axes[2].legend(h1 + h2, l1 + l2, loc="best")
        else:
            axes[2].legend(loc="best")
        axes[2].set_ylabel("absolute error")
        axes[2].set_xlabel("old forecast horizon, h1=T+15min")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"h{k}" for k in x])
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        safe_issue = issue.replace(":", "").replace(" ", "_")
        out_path = fig_dir / f"old05l_high_baseline_{i:03d}_{station}_{safe_issue}.png"
        fig.savefig(out_path, dpi=int(dpi))
        plt.close(fig)
        rows.append(
            {
                "sample_id": int(i),
                "station_sn": station,
                "issue_time": issue,
                "old03_mae": float(sample["old03_mae"]),
                "old05l_mae": float(sample["old05l_mae"]),
                "mae_gain": float(sample["mae_gain"]),
                "truth_min": float(sample["truth_min"]),
                "truth_max": float(sample["truth_max"]),
                "figure": str(out_path),
            }
        )
    return pd.DataFrame(rows)


def save_summary_plot(samples: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if samples.empty:
        return
    n = min(len(samples), 30)
    show = samples.head(n).copy()
    labels = [f"{i:02d}" for i in range(n)]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10.0, n * 0.36), 4.2))
    ax.bar(x - 0.18, show["old03_mae"], width=0.36, color="#1f77b4", label="old 03 baseline")
    ax.bar(x + 0.18, show["old05l_mae"], width=0.36, color="#d62728", label="old 05l")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("issue-level MAE PV_norm")
    ax.set_title("Old 05l behavior on high old-baseline-error samples")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "old05l_high_baseline_error_summary.png", dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_stage4_dirs()
    out_dir = ensure_dir(args.out_dir)

    baseline = read_detail(args.old_baseline_detail_csv, "old03")
    old05l = read_detail(args.old_05l_detail_csv, "old05l")
    merged = align_old_details(baseline, old05l, args.eval_split)
    samples = choose_high_baseline_samples(merged, args.sample_count, args.min_baseline_mae, args.seed)
    metrics = metric_rows(merged)
    fig_index = save_figures(merged, samples, out_dir, args.dpi)
    save_summary_plot(samples, out_dir, args.dpi)

    merged.to_csv(out_dir / "old05l_aligned_eval_detail.csv", index=False)
    samples.to_csv(out_dir / "old05l_high_baseline_error_samples.csv", index=False)
    metrics.to_csv(out_dir / "old05l_high_baseline_error_overall_metrics.csv", index=False)
    fig_index.to_csv(out_dir / "old05l_high_baseline_error_figure_index.csv", index=False)
    source_audit = {
        "old_baseline_detail_csv": file_audit(args.old_baseline_detail_csv),
        "old_05l_detail_csv": file_audit(args.old_05l_detail_csv),
        "old_baseline_nearby_meta": load_nearby_run_meta(args.old_baseline_detail_csv),
        "old_05l_nearby_meta": load_nearby_run_meta(args.old_05l_detail_csv),
    }
    write_json(out_dir / "source_audit.json", source_audit)
    write_json(
        out_dir / "run_meta.json",
        {
            "old_baseline_detail_csv": str(args.old_baseline_detail_csv),
            "old_05l_detail_csv": str(args.old_05l_detail_csv),
            "eval_split": str(args.eval_split),
            "sample_count": int(args.sample_count),
            "min_baseline_mae": float(args.min_baseline_mae),
            "note": "Old project issue_time has history ending at T and h1 target at T+15min.",
        },
    )

    print("==== Old-project 05l on high-baseline-error samples ====")
    print("==== Source audit ====")
    print(pd.DataFrame([source_audit["old_baseline_detail_csv"], source_audit["old_05l_detail_csv"]]).to_string(index=False))
    show = samples[["station_sn", "issue_time", "old03_mae", "old05l_mae", "mae_gain", "truth_min", "truth_max"]].head(30)
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    metric_show = metrics[metrics["horizon"].isin(["all", "h1", "h4", "h8"])]
    print("==== Overall old 03 vs old 05l ====")
    print(metric_show.to_string(index=False) if not metric_show.empty else "EMPTY")
    print(f"saved: {out_dir}")
    print(f"figures: {out_dir / 'sample_figures'}")


if __name__ == "__main__":
    main()
