#!/usr/bin/env python3
"""Collect the compact SAG-PVFF internal ablation table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage3_paths import RUNS_DIR, STAGE3_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SAG-PVFF internal ablation rows from existing eval CSVs.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--horizon", type=str, default="all")
    parser.add_argument("--out_dir", type=Path, default=STAGE3_DIR / "paper_tables")
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def value_or_nan(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def format_value(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def csv_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if any(ch in text for ch in [",", "\"", "\n"]):
        text = "\"" + text.replace("\"", "\"\"") + "\""
    return text


def read_filtered_row(path: Path, split: str, horizon: str, filters: dict[str, str]) -> tuple[pd.Series | None, str]:
    path = Path(path)
    if not path.exists():
        return None, f"missing_csv:{path}"
    df = pd.read_csv(path)
    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str).eq(str(split))]
    if "horizon" in df.columns:
        df = df[df["horizon"].astype(str).eq(str(horizon))]
    for col, value in filters.items():
        if col not in df.columns:
            return None, f"missing_filter_col:{col}"
        df = df[df[col].astype(str).eq(str(value))]
    if df.empty:
        return None, "missing_row"
    return df.iloc[0], "ok"


def compact_status(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    keys = [status.split(":", 1)[0] for status in statuses]
    if all(key == "missing_csv" for key in keys):
        return "missing_csv"
    if all(key in {"missing_csv", "missing_row"} for key in keys):
        return "missing_row"
    return keys[0]


def first_available(candidates: list[tuple[Path, dict[str, str]]], split: str, horizon: str) -> tuple[pd.Series | None, str, str, str]:
    diagnostics = []
    statuses = []
    for path, filters in candidates:
        row, status = read_filtered_row(path, split, horizon, filters)
        diagnostics.append(f"{status}:{path}")
        statuses.append(status)
        if row is not None:
            return row, str(path), "ok", "; ".join(diagnostics)
    return None, str(candidates[0][0]) if candidates else "", compact_status(statuses), "; ".join(diagnostics)


def metrics_from_row(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {key: float("nan") for key in ["n_points", "mae", "rmse", "r2", "ssim"]}
    return {
        "n_points": value_or_nan(row, ["n_points", "count", "pixels"]),
        "mae": value_or_nan(row, ["mae_pv_norm_field", "mae_field", "mae"]),
        "rmse": value_or_nan(row, ["rmse_pv_norm_field", "rmse_field", "rmse"]),
        "r2": value_or_nan(row, ["r2_pv_norm_field", "r2_field", "r2"]),
        "ssim": value_or_nan(row, ["ssim_pv_norm_field", "ssim_field", "ssim", "mean_ssim"]),
    }


def metric_status(item: dict[str, Any]) -> str:
    if item["status"] != "ok":
        return item["status"]
    missing = [name.upper() for name in ["mae", "rmse", "r2", "ssim"] if not np.isfinite(float(item[name]))]
    if missing:
        return "metric_missing:" + "/".join(missing)
    return "ok"


def build_specs() -> list[dict[str, Any]]:
    motion_last = RUNS_DIR / "ablation_motion_branch_last_only" / "eval_all" / "motion_branch_eval_summary.csv"
    swr_dep = RUNS_DIR / "ablation_swr_dependency" / "eval_all" / "swr_dependency_eval_summary.csv"
    fixed_fusion = RUNS_DIR / "ablation_motion_branch_last_only" / "eval_fixed_fusion_from_gate" / "motion_branch_eval_summary.csv"
    anchor_weights = RUNS_DIR / "ablation_anchor_weights" / "eval_all" / "anchor_weight_ablation_eval_summary.csv"
    eval_full_swr = RUNS_DIR / "ablation_swr_dependency" / "eval_full_swr_future" / "motion_branch_eval_summary.csv"
    eval_history_swr = RUNS_DIR / "ablation_swr_dependency" / "eval_history_swr_only" / "motion_branch_eval_summary.csv"

    return [
        {
            "setting": "Full SAG-PVFF",
            "future_swr": "yes",
            "f_tai": "yes",
            "motion_fusion": "warp+direct+adaptive gate",
            "note": "last_only + fixed physical temporal-anchor interpolation + full future SWR",
            "candidates": [
                (swr_dep, {"swr_dependency_variant": "full_swr_future", "motion_branch": "warp_direct_gate"}),
                (eval_full_swr, {"motion_branch": "warp_direct_gate"}),
                (motion_last, {"motion_branch": "warp_direct_gate"}),
            ],
        },
        {
            "setting": "w/o future SWR anchors",
            "future_swr": "no",
            "f_tai": "yes",
            "motion_fusion": "warp+direct+adaptive gate",
            "note": "future SWR slots replaced by latest observed SWR",
            "candidates": [
                (swr_dep, {"swr_dependency_variant": "history_swr_only", "motion_branch": "warp_direct_gate"}),
                (eval_history_swr, {"motion_branch": "warp_direct_gate"}),
            ],
        },
        {
            "setting": "w/o F-TAI",
            "future_swr": "yes",
            "f_tai": "no",
            "motion_fusion": "warp+direct+adaptive gate",
            "note": "learnable temporal-anchor weights from random init; no fixed physical interpolation prior",
            "candidates": [
                (anchor_weights, {"anchor_weight_variant": "learnable_random_init"}),
            ],
        },
        {
            "setting": "Direct-only prediction",
            "future_swr": "yes",
            "f_tai": "yes",
            "motion_fusion": "direct only",
            "note": "motion branch ablation with only direct decoder",
            "candidates": [(motion_last, {"motion_branch": "direct_only"})],
        },
        {
            "setting": "Warp-only prediction",
            "future_swr": "yes",
            "f_tai": "yes",
            "motion_fusion": "warp only",
            "note": "motion branch ablation with only flow warping",
            "candidates": [(motion_last, {"motion_branch": "warp_only"})],
        },
        {
            "setting": "w/o adaptive gate",
            "future_swr": "yes",
            "f_tai": "yes",
            "motion_fusion": "fixed 0.5 warp/direct",
            "note": "eval-only: full checkpoint with fusion gate replaced by fixed 0.5",
            "candidates": [(fixed_fusion, {"motion_branch": "fixed_fusion_from_gate"})],
        },
    ]


def build_table(split: str, horizon: str) -> pd.DataFrame:
    rows = []
    for spec in build_specs():
        row, source_csv, status, diagnostic = first_available(spec["candidates"], split, horizon)
        item = {
            "setting": spec["setting"],
            "future_swr": spec["future_swr"],
            "f_tai": spec["f_tai"],
            "motion_fusion": spec["motion_fusion"],
            **metrics_from_row(row),
            "source_csv": source_csv,
            "note": spec["note"],
            "status": status,
            "diagnostic": diagnostic,
        }
        item["status"] = metric_status(item)
        rows.append(item)
    return pd.DataFrame(rows)


def print_markdown(df: pd.DataFrame, digits: int) -> None:
    print("\n=== Stage 3 SAG-PVFF Ablation Table ===")
    cols = [
        ("Setting", "setting"),
        ("Future SWR", "future_swr"),
        ("F-TAI", "f_tai"),
        ("Motion Fusion", "motion_fusion"),
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("R2", "r2"),
        ("SSIM", "ssim"),
        ("Status", "status"),
    ]
    print("| " + " | ".join(label for label, _ in cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        cells = []
        for _label, key in cols:
            if key in {"mae", "rmse", "r2", "ssim"}:
                cells.append(format_value(row[key], digits))
            else:
                cells.append(str(row[key]))
        print("| " + " | ".join(cells) + " |")


def print_csv_block(df: pd.DataFrame) -> None:
    print("\n=== Stage 3 SAG-PVFF Ablation Table CSV ===")
    cols = list(df.columns)
    print(",".join(cols))
    for _, row in df.iterrows():
        print(",".join(csv_escape(row[col]) for col in cols))


def main() -> None:
    args = parse_args()
    df = build_table(str(args.split), str(args.horizon))
    print_markdown(df, int(args.digits))
    print_csv_block(df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stage3_sag_pvff_ablation_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[write] {out_csv}")


if __name__ == "__main__":
    main()
