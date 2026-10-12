#!/usr/bin/env python3
"""Analyze Stage 4 station forecast performance under future ramp regimes.

Ramp regimes are defined per target-station issue using the maximum adjacent
change inside the eight future ground-truth PV_norm values. The transition from
the latest historical frame to the first forecast step is intentionally not
included.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_paths import RUNS_DIR  # noqa: E402


def _load_weather_module():
    path = SCRIPT_DIR / "18_eval_stage4_weather_condition_analysis.py"
    spec = importlib.util.spec_from_file_location("stage4_weather_for_ramp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_weather_for_ramp"] = module
    spec.loader.exec_module(module)
    return module


w18 = _load_weather_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_detail_csv", type=Path, default=w18.PER_STATION_TEMPORAL_BASELINE_RUN_DIR / w18.BASELINE_DETAIL_NAME)
    parser.add_argument("--future_swr_patch_detail_csv", type=Path, default=w18.PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / w18.DETAIL_NAME)
    parser.add_argument("--historical_field_detail_csv", type=Path, default=w18.PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / w18.DETAIL_NAME)
    parser.add_argument("--future_field_detail_csv", type=Path, default=w18.PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / w18.DETAIL_NAME)
    parser.add_argument("--extra_detail_csv", action="append", default=[], help="Optional extra method as key:label:path")
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "19_eval_stage4_ramp_condition_analysis")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--align_mode", choices=["intersection", "reference", "none"], default="intersection")
    parser.add_argument("--min_group_points", type=int, default=100)
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_ramp_context(reference: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    for (sn, split, issue), g in reference.groupby(["station_sn", "eval_split", "issue_time"], sort=False):
        ordered = g.sort_values("h")
        truth = ordered["true_pv_norm"].astype(float).to_numpy(np.float64)
        finite = truth[np.isfinite(truth)]
        if truth.size < 2 or finite.size < 2:
            score = float("nan")
        else:
            score = float(np.nanmax(np.abs(np.diff(truth))))
        rows.append(
            {
                "station_sn": str(sn),
                "eval_split": str(split),
                "issue_time": str(issue),
                "ramp_score": score,
                "n_future_steps_for_ramp": int(truth.size),
            }
        )
    context = pd.DataFrame(rows)
    valid_scores = context["ramp_score"].astype(float)
    valid_scores = valid_scores[np.isfinite(valid_scores)]
    threshold = float(valid_scores.median()) if len(valid_scores) else float("nan")
    context["ramp_condition"] = np.where(context["ramp_score"].astype(float) <= threshold, "Low ramp", "High ramp")
    context.loc[~np.isfinite(context["ramp_score"].astype(float)), "ramp_condition"] = "Unknown"
    return context, threshold


def condition_counts(long_df: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    point_total = max(int(long_df[long_df["method_key"] == "temporal_baseline"].shape[0]), 1)
    rows = []
    for condition, g in context.groupby("ramp_condition", sort=False):
        issue_count = int(len(g))
        score = pd.to_numeric(g["ramp_score"], errors="coerce").astype(float)
        point_count = int(long_df[(long_df["method_key"] == "temporal_baseline") & (long_df["ramp_condition"] == condition)].shape[0])
        rows.append(
            {
                "ramp_condition": condition,
                "n_station_issues": issue_count,
                "n_points": point_count,
                "percentage": float(point_count / point_total * 100.0),
                "n_stations": int(g["station_sn"].nunique()),
                "mean_ramp_score": float(score.mean()),
                "median_ramp_score": float(score.median()),
            }
        )
    return pd.DataFrame(rows)


def summarize(long_df: pd.DataFrame, min_group_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (condition, method_key, method), g in long_df.groupby(["ramp_condition", "method_key", "method"], sort=False):
        if len(g) >= int(min_group_points):
            rows.append(w18.metric_row(g, {"ramp_condition": condition, "method_key": method_key, "method": method}))
    condition_summary = pd.DataFrame(rows)

    rows = []
    for (condition, horizon, method_key, method), g in long_df.groupby(["ramp_condition", "horizon", "method_key", "method"], sort=False):
        if len(g) >= int(min_group_points):
            rows.append(
                w18.metric_row(
                    g,
                    {
                        "ramp_condition": condition,
                        "horizon": horizon,
                        "method_key": method_key,
                        "method": method,
                    },
                )
            )
    return condition_summary, pd.DataFrame(rows)


def add_gain_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "temporal_baseline" not in set(summary["method_key"].astype(str)):
        return summary
    key_cols = [c for c in ["ramp_condition", "horizon"] if c in summary.columns]
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


def print_outputs(counts: pd.DataFrame, summary: pd.DataFrame, threshold: float, digits: int) -> None:
    print("==== Stage 4 ramp-condition analysis ====")
    print(f"ramp median threshold: {threshold:.6f}")
    print("\n=== Ramp condition counts ===")
    print(counts.to_string(index=False) if not counts.empty else "EMPTY")
    print("\n=== Ramp condition count LaTeX-ready rows ===")
    for _, row in counts.iterrows():
        print(
            f"{row['ramp_condition']} & {int(row['n_station_issues'])} & {int(row['n_points'])} & "
            f"{float(row['percentage']):.2f}\\% & {float(row['mean_ramp_score']):.4f} & "
            f"{float(row['median_ramp_score']):.4f} \\\\"
        )

    cols = ["ramp_condition", "method", "n_points", "mae_pv_norm", "rmse_pv_norm", "r2_pv_norm", "mae_gain_vs_temporal"]
    print("\n=== Ramp condition performance ===")
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False, float_format=lambda x: f"{x:.{digits}f}"))
    print("\n=== Ramp performance LaTeX-ready rows ===")
    for _, row in summary.iterrows():
        print(
            f"{row['ramp_condition']} & {row['method']} & {int(row['n_points'])} & "
            f"{float(row['mae_pv_norm']):.{digits}f} & {float(row['rmse_pv_norm']):.{digits}f} & "
            f"{float(row['r2_pv_norm']):.{digits}f} \\\\"
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = list(w18.METHOD_SPECS)
    for extra in w18.parse_extra_specs(list(args.extra_detail_csv)):
        specs.append({"key": extra["key"], "method": extra["method"], "path_arg": None, "path": extra["path"]})

    details = {}
    for spec in specs:
        path = Path(spec["path"]) if spec.get("path_arg") is None else Path(getattr(args, spec["path_arg"]))
        details[spec["key"]] = w18.read_detail(path, spec["key"], spec["method"], str(args.split))
    details = w18.align_details(details, str(args.align_mode))
    if not details:
        raise RuntimeError("No details loaded.")

    reference = details["temporal_baseline"] if "temporal_baseline" in details else next(iter(details.values()))
    ramp_context, threshold = build_ramp_context(reference)
    long_df = w18.to_long(details, specs)
    long_df = long_df.merge(
        ramp_context[["station_sn", "eval_split", "issue_time", "ramp_score", "ramp_condition"]],
        on=["station_sn", "eval_split", "issue_time"],
        how="left",
    )
    long_df = long_df[long_df["ramp_condition"].notna()].copy()

    counts = condition_counts(long_df, ramp_context)
    condition_summary, horizon_summary = summarize(long_df, int(args.min_group_points))
    condition_summary = add_gain_vs_baseline(condition_summary)
    horizon_summary = add_gain_vs_baseline(horizon_summary)

    ramp_context.to_csv(out_dir / "stage4_ramp_context.csv", index=False)
    long_df.to_csv(out_dir / "stage4_ramp_condition_aligned_detail_long.csv", index=False)
    counts.to_csv(out_dir / "stage4_ramp_condition_counts.csv", index=False)
    condition_summary.to_csv(out_dir / "stage4_ramp_condition_summary.csv", index=False)
    horizon_summary.to_csv(out_dir / "stage4_ramp_condition_horizon_summary.csv", index=False)
    write_json(
        out_dir / "stage4_ramp_condition_analysis_meta.json",
        {
            "split": str(args.split),
            "align_mode": str(args.align_mode),
            "ramp_score": "max absolute adjacent difference inside eight future ground-truth PV_norm steps",
            "excludes_latest_history_to_first_future": True,
            "threshold": "median over target-station issues",
            "median_threshold_value": float(threshold),
        },
    )

    print_outputs(counts, condition_summary, threshold, int(args.digits))
    print(f"\n[write] {out_dir / 'stage4_ramp_condition_summary.csv'}")
    print(f"[write] {out_dir / 'stage4_ramp_condition_horizon_summary.csv'}")


if __name__ == "__main__":
    main()
