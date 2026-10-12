#!/usr/bin/env python3
"""Compare no-anchor-mixer metrics with the saved current-best Stage 3 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
RUNS_DIR = STAGE3_DIR / "runs"
ABLATION_RUN_DIR = RUNS_DIR / "ablation_anchor_mixer" / "no_anchor_mixer"
BEST_RUN_DIR = RUNS_DIR / "ablation_motion_branch_last_only" / "warp_direct_gate"
OUT_DIR = RUNS_DIR / "ablation_anchor_mixer" / "eval_compare_best"
SUMMARY_NAME = "swr_anchor_guided_pvnorm_field_eval_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly read and compare saved Stage 3 best/no-anchor-mixer evaluation summaries."
    )
    parser.add_argument("--ablation_csv", type=Path, default=ABLATION_RUN_DIR / SUMMARY_NAME)
    parser.add_argument("--best_csv", type=Path, default=BEST_RUN_DIR / SUMMARY_NAME)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def read_summary(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} summary not found: {path}. "
            "Run the corresponding training first or pass its CSV explicitly."
        )
    frame = pd.read_csv(path)
    required = {"eval_split", "horizon"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")
    return frame.copy()


def metric_columns(best: pd.DataFrame, ablation: pd.DataFrame) -> list[str]:
    preferred = [
        "mae_pv_norm_field",
        "rmse_pv_norm_field",
        "delta_mae",
        "delta_corr_median",
        "delta_sign_accuracy",
        "large_delta_mae",
        "horizon_diversity_ratio",
    ]
    return [name for name in preferred if name in best.columns and name in ablation.columns]


def main() -> None:
    args = parse_args()
    best = read_summary(Path(args.best_csv), "current-best")
    ablation = read_summary(Path(args.ablation_csv), "no-anchor-mixer")
    metrics = metric_columns(best, ablation)
    if not metrics:
        raise ValueError("No common evaluation metrics were found in the two summaries.")

    keys = ["eval_split", "horizon"]
    best_small = best[keys + metrics].drop_duplicates(keys).copy()
    ablation_small = ablation[keys + metrics].drop_duplicates(keys).copy()
    compared = best_small.merge(
        ablation_small,
        on=keys,
        how="inner",
        suffixes=("_best", "_no_anchor_mixer"),
        validate="one_to_one",
    )
    if compared.empty:
        raise ValueError("The two summaries have no aligned eval_split/horizon rows.")

    lower_is_better = {
        "mae_pv_norm_field",
        "rmse_pv_norm_field",
        "delta_mae",
        "large_delta_mae",
    }
    for metric in metrics:
        best_col = f"{metric}_best"
        ablation_col = f"{metric}_no_anchor_mixer"
        compared[f"{metric}_no_anchor_mixer_minus_best"] = compared[ablation_col] - compared[best_col]
        if metric in lower_is_better:
            compared[f"{metric}_best_gain"] = compared[ablation_col] - compared[best_col]
        else:
            compared[f"{metric}_best_gain"] = compared[best_col] - compared[ablation_col]

    long_best = best.copy()
    long_best.insert(0, "anchor_mixer_variant", "current_best_with_anchor_mixer")
    long_ablation = ablation.copy()
    long_ablation.insert(0, "anchor_mixer_variant", "no_anchor_mixer")
    combined = pd.concat([long_best, long_ablation], ignore_index=True, sort=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "anchor_mixer_ablation_eval_summary.csv"
    comparison_path = out_dir / "anchor_mixer_vs_best_aligned_comparison.csv"
    combined.to_csv(combined_path, index=False)
    compared.to_csv(comparison_path, index=False)
    (out_dir / "anchor_mixer_eval_meta.json").write_text(
        json.dumps(
            {
                "purpose": "Measure the contribution of the learned all-anchor context.",
                "current_best_csv": str(args.best_csv),
                "no_anchor_mixer_csv": str(args.ablation_csv),
                "comparison_rule": (
                    "best_gain is positive when the saved current-best model is better; "
                    "error metrics use no_anchor_mixer - best, while correlation/accuracy metrics use best - no_anchor_mixer."
                ),
                "combined_summary": str(combined_path),
                "aligned_comparison": str(comparison_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    show = compared[
        (compared["eval_split"].astype(str) == "test")
        & compared["horizon"].astype(str).isin(["all", "h1", "h4", "h8"])
    ]
    display = ["eval_split", "horizon"]
    for metric in metrics:
        display.extend(
            [
                f"{metric}_best",
                f"{metric}_no_anchor_mixer",
                f"{metric}_best_gain",
            ]
        )
    print("==== Stage 3 anchor-mixer ablation vs current best ====")
    print(show[display].to_string(index=False) if not show.empty else compared[display].to_string(index=False))
    print("combined:", combined_path)
    print("comparison:", comparison_path)


if __name__ == "__main__":
    main()

