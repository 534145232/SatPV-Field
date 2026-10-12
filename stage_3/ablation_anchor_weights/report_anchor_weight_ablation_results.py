#!/usr/bin/env python3
"""Print paper-ready Stage-3 F-TAI/anchor-weight ablation results.

This is a read-only reporter for already completed experiments. It first reads
the merged anchor-weight ablation CSV if available. If not, it reconstructs the
three-way comparison from the fixed F-TAI baseline and the two learnable-weight
evaluation CSVs without rerunning any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
RUNS_DIR = STAGE3_DIR / "runs"
DEFAULT_RUN_ROOT = RUNS_DIR / "ablation_anchor_weights"
DEFAULT_MERGED_CSV = DEFAULT_RUN_ROOT / "eval_all" / "anchor_weight_ablation_eval_summary.csv"
DEFAULT_FIXED_CSV = RUNS_DIR / "ablation_motion_branch" / "eval_all" / "motion_branch_eval_summary.csv"
DEFAULT_RANDOM_CSV = (
    DEFAULT_RUN_ROOT
    / "learnable_random_init"
    / "05_validation_visuals"
    / "swr_anchor_guided_validation_eval_summary.csv"
)
DEFAULT_PHYSICAL_CSV = (
    DEFAULT_RUN_ROOT
    / "learnable_physical_init"
    / "05_validation_visuals"
    / "swr_anchor_guided_validation_eval_summary.csv"
)
DEFAULT_OUT_CSV = DEFAULT_RUN_ROOT / "eval_all" / "anchor_weight_ablation_paper_table.csv"

VARIANT_LABELS = {
    "fixed_physical_reused_motion_warp_direct_gate": "Fixed F-TAI",
    "learnable_random_init": "Learnable weights (random init.)",
    "learnable_physical_init": "Learnable weights (F-TAI init.)",
}
VARIANT_ORDER = [
    "fixed_physical_reused_motion_warp_direct_gate",
    "learnable_random_init",
    "learnable_physical_init",
]
METRIC_COLUMNS = [
    ("MAE", "mae_pv_norm_field", "min"),
    ("RMSE", "rmse_pv_norm_field", "min"),
    ("R2", "r2_pv_norm_field", "max"),
    ("SSIM", "ssim_pv_norm_field", "max"),
    ("DeltaCorr", "delta_corr_median", "max"),
    ("SignAcc", "delta_sign_accuracy", "max"),
    ("LargeDeltaMAE", "large_delta_mae", "min"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--fixed_csv", type=Path, default=DEFAULT_FIXED_CSV)
    parser.add_argument("--random_csv", type=Path, default=DEFAULT_RANDOM_CSV)
    parser.add_argument("--physical_csv", type=Path, default=DEFAULT_PHYSICAL_CSV)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--horizon", type=str, default="all")
    parser.add_argument("--all_horizons", type=str, default="all,h1,h4,h8")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--save_csv", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} CSV: {path}")
    return pd.read_csv(path)


def attach_variant(df: pd.DataFrame, variant: str, source: Path) -> pd.DataFrame:
    out = df.copy()
    if "anchor_weight_variant" not in out.columns:
        out.insert(0, "anchor_weight_variant", variant)
    else:
        out["anchor_weight_variant"] = variant
    if "result_source" not in out.columns:
        out.insert(1, "result_source", str(source))
    return out


def load_combined(args: argparse.Namespace) -> pd.DataFrame:
    if Path(args.merged_csv).exists():
        return read_csv(Path(args.merged_csv), "merged anchor-weight ablation")

    fixed = read_csv(Path(args.fixed_csv), "fixed F-TAI baseline")
    if "motion_branch" in fixed.columns:
        fixed = fixed[fixed["motion_branch"].astype(str).eq("warp_direct_gate")].copy()
    if fixed.empty:
        raise RuntimeError(f"No fixed warp_direct_gate rows found in {args.fixed_csv}")

    random = read_csv(Path(args.random_csv), "learnable random-init")
    physical = read_csv(Path(args.physical_csv), "learnable F-TAI-init")
    return pd.concat(
        [
            attach_variant(fixed, "fixed_physical_reused_motion_warp_direct_gate", Path(args.fixed_csv)),
            attach_variant(random, "learnable_random_init", Path(args.random_csv)),
            attach_variant(physical, "learnable_physical_init", Path(args.physical_csv)),
        ],
        ignore_index=True,
        sort=False,
    )


def existing_metrics(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [(label, col, direction) for label, col, direction in METRIC_COLUMNS if col in df.columns]


def make_paper_table(df: pd.DataFrame, split: str, horizon: str) -> pd.DataFrame:
    metrics = existing_metrics(df)
    if not metrics:
        raise RuntimeError("No known metric columns found in anchor-weight ablation results.")
    work = df[
        df["eval_split"].astype(str).eq(str(split))
        & df["horizon"].astype(str).eq(str(horizon))
        & df["anchor_weight_variant"].astype(str).isin(VARIANT_ORDER)
    ].copy()
    if work.empty:
        available = df[["eval_split", "horizon", "anchor_weight_variant"]].drop_duplicates().to_string(index=False)
        raise RuntimeError(f"No rows for split={split}, horizon={horizon}. Available rows:\n{available}")
    work["variant_order"] = work["anchor_weight_variant"].map({key: i for i, key in enumerate(VARIANT_ORDER)})
    work = work.sort_values("variant_order")
    rows = []
    for _, row in work.iterrows():
        item = {
            "Setting": VARIANT_LABELS.get(str(row["anchor_weight_variant"]), str(row["anchor_weight_variant"])),
            "anchor_weight_variant": str(row["anchor_weight_variant"]),
            "eval_split": str(row["eval_split"]),
            "horizon": str(row["horizon"]),
        }
        for label, col, _direction in metrics:
            item[label] = float(row[col]) if pd.notna(row[col]) else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def best_mask(table: pd.DataFrame) -> dict[str, int]:
    masks: dict[str, int] = {}
    for label, _col, direction in existing_metrics_from_labels(table):
        values = pd.to_numeric(table[label], errors="coerce")
        if not values.notna().any():
            continue
        masks[label] = int(values.idxmin() if direction == "min" else values.idxmax())
    return masks


def existing_metrics_from_labels(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    by_label = {label: (col, direction) for label, col, direction in METRIC_COLUMNS}
    return [(label, by_label[label][0], by_label[label][1]) for label in table.columns if label in by_label]


def fmt_value(value: float, digits: int, bold: bool) -> str:
    if not np.isfinite(float(value)):
        text = "--"
    else:
        text = f"{float(value):.{int(digits)}f}"
    return rf"\textbf{{{text}}}" if bold and text != "--" else text


def print_table(table: pd.DataFrame, digits: int) -> None:
    display_cols = ["Setting"] + [label for label, _col, _direction in existing_metrics_from_labels(table)]
    print(table[display_cols].to_string(index=False, float_format=lambda value: f"{value:.{int(digits)}f}"))


def print_latex_rows(table: pd.DataFrame, digits: int) -> None:
    best = best_mask(table)
    metric_labels = [label for label, _col, _direction in existing_metrics_from_labels(table)]
    print("\n==== LaTeX-ready rows ====")
    for idx, row in table.iterrows():
        values = [str(row["Setting"])]
        for label in metric_labels:
            values.append(fmt_value(float(row[label]), int(digits), best.get(label) == idx))
        print(" & ".join(values) + r" \\")


def print_horizon_overview(df: pd.DataFrame, args: argparse.Namespace) -> None:
    horizons = [item.strip() for item in str(args.all_horizons).split(",") if item.strip()]
    rows = []
    for horizon in horizons:
        try:
            table = make_paper_table(df, str(args.split), horizon)
        except RuntimeError:
            continue
        for _, row in table.iterrows():
            item = {"horizon": horizon, "Setting": row["Setting"]}
            for label, _col, _direction in existing_metrics_from_labels(table):
                item[label] = row[label]
            rows.append(item)
    if rows:
        overview = pd.DataFrame(rows)
        print("\n==== Horizon overview ====")
        print(overview.to_string(index=False, float_format=lambda value: f"{value:.{int(args.digits)}f}"))


def main() -> None:
    args = parse_args()
    combined = load_combined(args)
    table = make_paper_table(combined, str(args.split), str(args.horizon))
    print("==== Stage 3 F-TAI / anchor-weight ablation ====")
    print(f"split={args.split}, horizon={args.horizon}")
    print_table(table, int(args.digits))
    print_latex_rows(table, int(args.digits))
    print_horizon_overview(combined, args)

    if int(args.save_csv):
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"\nsaved paper table: {args.out_csv}")
    print("\nresult source:")
    if Path(args.merged_csv).exists():
        print(f"merged_csv: {args.merged_csv}")
    else:
        print(f"fixed_csv: {args.fixed_csv}")
        print(f"random_csv: {args.random_csv}")
        print(f"physical_csv: {args.physical_csv}")


if __name__ == "__main__":
    main()
