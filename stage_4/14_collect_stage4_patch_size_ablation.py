#!/usr/bin/env python3
"""Collect Stage 4 PV-field patch-size ablation metrics.

This script is intentionally table-oriented: it discovers/evaluates available
patch-size runs, aligns all models on a common eval subset, computes publication
metrics, and prints a compact table for manuscript use.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras
from stage4_paths import (
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)


DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
CORE_COLS = [
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
CAPACITY_COLS = ["capacity_raw", "capacity_scale", "installed_capacity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root",
        type=Path,
        default=RUNS_DIR / "ablation_pv_field_patch_size",
        help="Root containing patch_1x1, patch_3x3, ... directories.",
    )
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--patch_5_dir",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
        help="Fallback standard 5x5 run when no local patch_5x5 run exists.",
    )
    parser.add_argument(
        "--patch_detail",
        action="append",
        default=[],
        metavar="SIZE=CSV_OR_DIR",
        help="Override/add a patch result. Can repeat, e.g. --patch_detail 9=/path/to/run_or_eval_detail.csv.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="auto",
        help="Comma-separated sizes to include, or auto to use discovered sizes.",
    )
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "14_stage4_patch_size_ablation_table")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--large_delta_threshold", type=float, default=0.10)
    parser.add_argument("--include_baseline", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def history_cols(df: pd.DataFrame) -> list[str]:
    def key(name: str) -> int:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 0

    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")], key=key)


def parse_sizes(text: str) -> list[int] | None:
    if str(text).strip().lower() == "auto":
        return None
    out = []
    for part in str(text).split(","):
        cleaned = part.strip().lower().replace("x", "")
        if cleaned:
            out.append(int(cleaned))
    return sorted(set(out))


def detail_from_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / DETAIL_NAME
    return path


def parse_patch_overrides(items: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"--patch_detail must be SIZE=CSV_OR_DIR, got {item!r}")
        size_text, path_text = str(item).split("=", 1)
        size = int(size_text.strip().lower().replace("x", ""))
        out[size] = detail_from_path(Path(path_text))
    return out


def discover_patch_details(args: argparse.Namespace) -> dict[int, Path]:
    details: dict[int, Path] = {}
    root = Path(args.run_root)
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = re.fullmatch(r"patch_(\d+)x\1", child.name)
            if not match:
                continue
            path = child / DETAIL_NAME
            if path.exists():
                details[int(match.group(1))] = path

    standard_5 = Path(args.patch_5_dir) / DETAIL_NAME
    if standard_5.exists():
        details.setdefault(5, standard_5)

    details.update(parse_patch_overrides(args.patch_detail))
    sizes = parse_sizes(args.sizes)
    if sizes is not None:
        details = {size: details.get(size, Path(args.run_root) / f"patch_{size}x{size}" / DETAIL_NAME) for size in sizes}
    return dict(sorted(details.items()))


def read_detail(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing eval-detail CSV for {label}: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = [c for c in CORE_COLS if c not in df.columns]
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


def patch_extra_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in [
            "field_patch_center_pv_norm",
            "field_patch_mean_pv_norm",
            "field_patch_support_mean",
        ]
        if c in df.columns
    ]


def align_details(
    baseline: pd.DataFrame | None,
    patch_details: dict[int, pd.DataFrame],
    large_delta_threshold: float,
) -> pd.DataFrame:
    if not patch_details:
        raise RuntimeError("No patch-size details were loaded.")
    ref_size = 5 if 5 in patch_details else sorted(patch_details)[0]
    ref = baseline if baseline is not None else patch_details[ref_size]

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
    ref_cols += [c for c in CAPACITY_COLS if c in ref.columns]
    if baseline is not None:
        ref_cols += ["pred_pv_norm"] + history_cols(baseline)
    else:
        ref_cols += history_cols(ref)
    ref_cols = list(dict.fromkeys([c for c in ref_cols if c in ref.columns]))

    merged = ref[ref_cols].copy()
    if baseline is not None and "pred_pv_norm" in merged.columns:
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
        ] + patch_extra_cols(df)
        right = df[[c for c in cols if c in df.columns]].copy().rename(
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
            print(f"[warn] align patch {size}x{size}: rows {before} -> {len(merged)}", flush=True)

        truth_diff = (merged["true_pv_norm"].astype(float) - merged[f"true_patch_{size}"].astype(float)).abs()
        max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
        if np.isfinite(max_diff) and max_diff > 1.0e-5:
            print(f"[warn] max truth mismatch for patch {size}x{size}: {max_diff:.6g}", flush=True)
        merged["large_change"] = merged["large_change"] | coerce_bool(merged[f"large_change_patch_{size}"])

    if merged["large_change"].isna().any():
        merged["large_change"] = merged["true_delta_pv_norm"].astype(float).abs() >= float(large_delta_threshold)

    for model, pred_col in prediction_columns(merged):
        err = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        tag = pred_col.replace("pred_", "")
        merged[f"err_{tag}"] = err
        merged[f"abs_err_{tag}"] = err.abs()
        merged[f"sq_err_{tag}"] = err**2
    return merged


def prediction_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = []
    if "pred_baseline" in df.columns:
        cols.append(("baseline", "pred_baseline"))
    patch_cols = sorted(
        [c for c in df.columns if c.startswith("pred_patch_")],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    for col in patch_cols:
        size = int(col.rsplit("_", 1)[-1])
        cols.append((f"patch_{size}x{size}", col))
    return cols


def metric_row(df: pd.DataFrame, model: str, pred_col: str, scope: dict[str, Any]) -> dict[str, Any]:
    err = df[pred_col].astype(float) - df["true_pv_norm"].astype(float)
    abs_err = err.abs()
    sq_err = err**2
    large = df["large_change"].astype(bool)

    hist = history_cols(df)
    if hist:
        hist_last = df[hist[-1]].astype(float)
        pred_delta = df[pred_col].astype(float) - hist_last
        true_delta = df["true_delta_pv_norm"].astype(float)
        valid_delta = np.isfinite(pred_delta) & np.isfinite(true_delta)
        sign_valid = valid_delta & (true_delta.abs() >= 1.0e-6)
        delta_corr = (
            float(np.corrcoef(pred_delta[valid_delta], true_delta[valid_delta])[0, 1])
            if int(valid_delta.sum()) >= 3
            else float("nan")
        )
        sign_acc = (
            float((np.sign(pred_delta[sign_valid]) == np.sign(true_delta[sign_valid])).mean())
            if bool(sign_valid.any())
            else float("nan")
        )
    else:
        delta_corr = float("nan")
        sign_acc = float("nan")

    row = {
        "model": model,
        "patch_size": patch_size_from_model(model),
        "n_points": int(len(df)),
        "mae_pv_norm": float(abs_err.mean()),
        "rmse_pv_norm": float(math.sqrt(float(sq_err.mean()))),
        "bias_pv_norm": float(err.mean()),
        "r2_pv_norm": float(regression_metric_extras(df, pred_col)["r2_pv_norm"]),
        "acc_installed_capacity": float(regression_metric_extras(df, pred_col)["acc_installed_capacity"]),
        "nrmse_installed_capacity": float(regression_metric_extras(df, pred_col)["nrmse_installed_capacity"]),
        "acc_capacity_source": regression_metric_extras(df, pred_col)["acc_capacity_source"],
        "p95_abs_err_pv_norm": float(abs_err.quantile(0.95)),
        "large_change_mae_pv_norm": float(abs_err[large].mean()) if bool(large.any()) else float("nan"),
        "large_change_n": int(large.sum()),
        "delta_corr": delta_corr,
        "delta_sign_accuracy": sign_acc,
    }
    row.update(scope)
    return row


def patch_size_from_model(model: str) -> int | float:
    match = re.fullmatch(r"patch_(\d+)x\1", str(model))
    return int(match.group(1)) if match else float("nan")


def build_summary(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for split, g in merged.groupby("eval_split", sort=True):
        for model, pred_col in prediction_columns(g):
            rows.append(metric_row(g, model, pred_col, {"eval_split": split, "horizon": "all"}))
    summary = pd.DataFrame(rows)

    horizon_rows = []
    for (split, horizon), g in merged.groupby(["eval_split", "horizon"], sort=True):
        for model, pred_col in prediction_columns(g):
            horizon_rows.append(metric_row(g, model, pred_col, {"eval_split": split, "horizon": horizon}))
    horizon = pd.DataFrame(horizon_rows)

    station_rows = []
    for (split, station), g in merged.groupby(["eval_split", "station_sn"], sort=True):
        for model, pred_col in prediction_columns(g):
            station_rows.append(metric_row(g, model, pred_col, {"eval_split": split, "station_sn": str(station)}))
    station = pd.DataFrame(station_rows)
    return summary, horizon, station


def build_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in summary.groupby("eval_split", sort=True):
        patch_rows = g[g["model"].astype(str).str.startswith("patch_")].copy()
        if patch_rows.empty:
            continue
        base = g[g["model"] == "baseline"]
        ref_5 = patch_rows[patch_rows["model"] == "patch_5x5"]
        ref_1 = patch_rows[patch_rows["model"] == "patch_1x1"]
        refs = []
        if not base.empty:
            refs.append(("baseline", base.iloc[0]))
        if not ref_1.empty:
            refs.append(("patch_1x1", ref_1.iloc[0]))
        if not ref_5.empty:
            refs.append(("patch_5x5", ref_5.iloc[0]))
        for _, cand in patch_rows.iterrows():
            for ref_name, ref in refs:
                rows.append(
                    {
                        "eval_split": split,
                        "candidate_model": cand["model"],
                        "reference_model": ref_name,
                        "candidate_patch_size": cand["patch_size"],
                        "mae_gain": float(ref["mae_pv_norm"] - cand["mae_pv_norm"]),
                        "rmse_gain": float(ref["rmse_pv_norm"] - cand["rmse_pv_norm"]),
                        "r2_gain": float(cand["r2_pv_norm"] - ref["r2_pv_norm"]),
                        "acc_gain": float(cand["acc_installed_capacity"] - ref["acc_installed_capacity"]),
                        "mae_gain_pct": (
                            float(100.0 * (ref["mae_pv_norm"] - cand["mae_pv_norm"]) / ref["mae_pv_norm"])
                            if float(ref["mae_pv_norm"]) > 0
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def round_for_print(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if np.issubdtype(out[col].dtype, np.number):
            out[col] = out[col].map(lambda x: round(float(x), int(digits)) if np.isfinite(float(x)) else np.nan)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "EMPTY"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_patch_details(args)
    if not discovered:
        raise RuntimeError(f"No patch-size eval-detail CSVs found under {args.run_root}")

    baseline = None
    if int(args.include_baseline) and Path(args.baseline_detail_csv).exists():
        baseline = read_detail(Path(args.baseline_detail_csv), "baseline")
        print(f"[read] baseline: {args.baseline_detail_csv} rows={len(baseline)}", flush=True)
    elif int(args.include_baseline):
        print(f"[warn] baseline missing: {args.baseline_detail_csv}", flush=True)

    patch_details: dict[int, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    for size, path in discovered.items():
        if not Path(path).exists():
            print(f"[skip] patch_{size}x{size}: missing {path}", flush=True)
            continue
        df = read_detail(Path(path), f"patch_{size}x{size}")
        patch_details[size] = df
        source_paths[f"patch_{size}x{size}"] = str(path)
        print(f"[read] patch_{size}x{size}: {path} rows={len(df)}", flush=True)

    merged = align_details(baseline, patch_details, float(args.large_delta_threshold))
    summary, horizon, station = build_summary(merged)
    improvements = build_improvements(summary)

    merged.to_csv(args.out_dir / "patch_size_aligned_detail.csv", index=False)
    summary.to_csv(args.out_dir / "patch_size_summary.csv", index=False)
    horizon.to_csv(args.out_dir / "patch_size_horizon_summary.csv", index=False)
    station.to_csv(args.out_dir / "patch_size_station_summary.csv", index=False)
    improvements.to_csv(args.out_dir / "patch_size_improvements.csv", index=False)
    write_json(
        args.out_dir / "patch_size_eval_meta.json",
        {
            "run_root": str(args.run_root),
            "baseline_detail_csv": str(args.baseline_detail_csv) if baseline is not None else None,
            "source_paths": source_paths,
            "sizes_loaded": sorted(patch_details),
            "n_aligned_rows": int(len(merged)),
            "splits": sorted(merged["eval_split"].astype(str).unique().tolist()),
        },
    )

    split = str(args.split)
    table_cols = [
        "model",
        "patch_size",
        "n_points",
        "mae_pv_norm",
        "rmse_pv_norm",
        "bias_pv_norm",
        "r2_pv_norm",
        "acc_installed_capacity",
        "large_change_mae_pv_norm",
        "delta_corr",
        "delta_sign_accuracy",
    ]
    main_table = summary[summary["eval_split"].astype(str) == split].copy()
    main_table = main_table.sort_values(["patch_size", "model"], na_position="first")
    main_table = main_table[[c for c in table_cols if c in main_table.columns]]
    print("\n==== Stage 4 Patch-Size Ablation Main Table ====")
    print(markdown_table(round_for_print(main_table, int(args.digits))))

    horizon_table = horizon[horizon["eval_split"].astype(str) == split].copy()
    horizon_table = horizon_table[horizon_table["model"].astype(str).str.startswith("patch_")]
    horizon_table = horizon_table.sort_values(["patch_size", "horizon"])
    horizon_cols = ["model", "horizon", "n_points", "mae_pv_norm", "rmse_pv_norm", "r2_pv_norm", "acc_installed_capacity"]
    print("\n==== Horizon-Wise Patch Metrics ====")
    print(markdown_table(round_for_print(horizon_table[[c for c in horizon_cols if c in horizon_table.columns]], int(args.digits))))

    imp = improvements[improvements["eval_split"].astype(str) == split].copy()
    if not imp.empty:
        imp = imp.sort_values(["reference_model", "candidate_patch_size"])
        imp_cols = ["candidate_model", "reference_model", "mae_gain", "mae_gain_pct", "rmse_gain", "r2_gain", "acc_gain"]
        print("\n==== Improvements ====")
        print(markdown_table(round_for_print(imp[[c for c in imp_cols if c in imp.columns]], int(args.digits))))

    print(f"\n[write] {args.out_dir}")


if __name__ == "__main__":
    main()
