#!/usr/bin/env python3
"""Collect Stage 3 PV-field experiment tables from existing eval summaries.

This is a result-reader, not an evaluator. It assumes the training/evaluation
interfaces have already produced their standard CSV summaries and prints the
paper-table rows needed for the main text and appendix.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage3_paths import MOTION_FUSION_RUN_DIR, RUNS_DIR, STAGE3_DIR


SUMMARY_MAIN = "swr_anchor_guided_pvnorm_field_eval_summary.csv"
SUMMARY_BASE = "pvnorm_field_eval_summary.csv"


BASE_MODEL_LABELS = {
    "persistence_last": "Persistence",
    "past_mean": "Past mean",
    "linear_trend": "Linear trend",
    "pv_only": "PV-only U-Net",
    "pv_h8": "PV + historical H8/SWR U-Net",
    "pv_nwp": "PV + NWP U-Net",
    "pv_h8_nwp": "PV + H8/SWR + NWP U-Net",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read existing Stage 3 eval-summary CSVs and print experiment tables."
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--horizon", type=str, default="all")
    parser.add_argument(
        "--base_csv",
        type=Path,
        default=RUNS_DIR / "base_pvnorm_field_model" / SUMMARY_BASE,
        help="CSV from stage3_base_pvnorm_field_model.py, ideally with pv_only,pv_nwp,pv_h8_nwp rows.",
    )
    parser.add_argument(
        "--legacy_baseline_csv",
        type=Path,
        default=None,
        help="Optional old fullfield_temporal_eval_summary.csv with persistence_last/past_mean/linear_trend.",
    )
    parser.add_argument(
        "--proposed_csv",
        type=Path,
        default=None,
        help="Proposed-model summary. If omitted, common best/main paths are tried.",
    )
    parser.add_argument(
        "--swr_dependency_csv",
        type=Path,
        default=RUNS_DIR / "ablation_swr_dependency" / "eval_all" / "swr_dependency_eval_summary.csv",
    )
    parser.add_argument(
        "--motion_branch_csv",
        type=Path,
        default=RUNS_DIR / "ablation_motion_branch" / "eval_all" / "motion_branch_eval_summary.csv",
    )
    parser.add_argument(
        "--motion_branch_last_only_csv",
        type=Path,
        default=RUNS_DIR / "ablation_motion_branch_last_only" / "eval_all" / "motion_branch_eval_summary.csv",
    )
    parser.add_argument(
        "--temporal_collapse_csv",
        type=Path,
        default=RUNS_DIR / "ablation_temporal_collapse" / "eval_all" / "temporal_collapse_ablation_eval_summary.csv",
    )
    parser.add_argument(
        "--anchor_weight_csv",
        type=Path,
        default=RUNS_DIR / "ablation_anchor_weights" / "eval_all" / "anchor_weight_ablation_eval_summary.csv",
    )
    parser.add_argument(
        "--anchor_mixer_csv",
        type=Path,
        default=RUNS_DIR / "ablation_anchor_mixer" / "eval_compare_best" / "anchor_mixer_ablation_eval_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=STAGE3_DIR / "paper_tables",
        help="Directory for normalized CSV tables. Console output is always printed.",
    )
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def existing_proposed_candidates() -> list[Path]:
    return [
        RUNS_DIR / "ablation_swr_dependency" / "eval_full_swr_future" / "motion_branch_eval_summary.csv",
        RUNS_DIR / "ablation_motion_branch_last_only" / "warp_direct_gate" / SUMMARY_MAIN,
        MOTION_FUSION_RUN_DIR / SUMMARY_MAIN,
        RUNS_DIR / "ablation_motion_branch" / "warp_direct_gate" / SUMMARY_MAIN,
    ]


def read_csv_if_exists(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[skip] {label}: missing {path}", flush=True)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "source_csv", str(path))
    print(f"[read] {label}: {path} rows={len(df)}", flush=True)
    return df


def choose_proposed_csv(path: Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    for candidate in existing_proposed_candidates():
        if candidate.exists():
            return candidate
    return existing_proposed_candidates()[0]


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_summary(df: pd.DataFrame, source: str, split: str, horizon: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "eval_split" in work.columns:
        work = work[work["eval_split"].astype(str) == str(split)]
    if "horizon" in work.columns:
        work = work[work["horizon"].astype(str) == str(horizon)]
    if work.empty:
        return pd.DataFrame()

    model_col = first_col(
        work,
        [
            "swr_dependency_variant",
            "motion_branch",
            "temporal_collapse_variant",
            "anchor_weight_variant",
            "anchor_mixer_variant",
            "variant",
            "model",
        ],
    )
    mae_col = first_col(work, ["mae_pv_norm_field", "mae_field", "mae"])
    rmse_col = first_col(work, ["rmse_pv_norm_field", "rmse_field", "rmse"])
    ssim_col = first_col(work, ["ssim_pv_norm_field", "ssim_field", "ssim", "mean_ssim"])
    r2_col = first_col(work, ["r2_pv_norm_field", "r2_field", "r2"])
    bias_col = first_col(work, ["bias_pv_norm_field", "bias_field", "bias"])
    n_col = first_col(work, ["n_points", "count", "pixels"])

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        model = str(row[model_col]) if model_col is not None else source
        rows.append(
            {
                "source": source,
                "model": model,
                "raw_model": str(row["model"]) if "model" in row.index else model,
                "variant_note": str(row["variant_note"]) if "variant_note" in row.index else "",
                "eval_split": str(row["eval_split"]) if "eval_split" in work.columns else str(split),
                "horizon": str(row["horizon"]) if "horizon" in work.columns else str(horizon),
                "n_points": value_or_nan(row, n_col),
                "mae": value_or_nan(row, mae_col),
                "rmse": value_or_nan(row, rmse_col),
                "bias": value_or_nan(row, bias_col),
                "r2": value_or_nan(row, r2_col),
                "ssim": value_or_nan(row, ssim_col),
                "delta_mae": value_or_nan(row, "delta_mae"),
                "delta_corr_median": value_or_nan(row, "delta_corr_median"),
                "delta_sign_accuracy": value_or_nan(row, "delta_sign_accuracy"),
                "large_delta_mae": value_or_nan(row, "large_delta_mae"),
                "horizon_diversity_ratio": value_or_nan(row, "horizon_diversity_ratio"),
                "source_csv": str(row["source_csv"]) if "source_csv" in work.columns else "",
            }
        )
    return pd.DataFrame(rows)


def value_or_nan(row: pd.Series, col: str | None) -> float:
    if col is None or col not in row.index:
        return float("nan")
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return float("nan")


def format_value(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}", flush=True)


def print_markdown(title: str, df: pd.DataFrame, digits: int) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    columns = [
        ("Method", "method"),
        ("PV", "pv_history"),
        ("NWP", "nwp"),
        ("SWR", "swr"),
        ("Motion", "motion"),
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("Bias", "bias"),
        ("R2", "r2"),
        ("SSIM", "ssim"),
        ("DeltaCorr", "delta_corr_median"),
        ("SignAcc", "delta_sign_accuracy"),
        ("LargeDeltaMAE", "large_delta_mae"),
    ]
    columns = [(label, key) for label, key in columns if key in df.columns]
    print("| " + " | ".join(label for label, _ in columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in df.iterrows():
        cells = []
        for _label, key in columns:
            value = row[key]
            if key in {"mae", "rmse", "bias", "r2", "ssim", "delta_corr_median", "delta_sign_accuracy", "large_delta_mae"}:
                cells.append(format_value(value, digits))
            else:
                cells.append(str(value))
        print("| " + " | ".join(cells) + " |")


def warn_npoint_mismatch(df: pd.DataFrame) -> None:
    if df.empty or "n_points" not in df.columns:
        return
    vals = df["n_points"].dropna().astype(float)
    if vals.empty:
        return
    unique = sorted(set(int(v) for v in vals if np.isfinite(v)))
    if len(unique) <= 1:
        return
    print("\n[warn] Main-table rows have different n_points.")
    print("       This usually means the proposed SWR-anchor model was evaluated only on")
    print("       issue times with valid SWR-anchor cache coverage, while baselines used")
    print("       the full PV-field test split. For a strict paper table, evaluate all")
    print("       methods on the common issue subset or state the filtering explicitly.")
    print(f"       n_points values: {unique}")


def warn_missing_proposed_metrics(df: pd.DataFrame) -> None:
    if df.empty:
        return
    prop = df[df["source"].astype(str).eq("proposed")]
    if prop.empty:
        return
    missing = []
    for col, label in [("r2", "R2"), ("ssim", "SSIM")]:
        if col not in prop.columns or prop[col].isna().all():
            missing.append(label)
    if missing:
        print(
            "\n[warn] Proposed row is missing "
            + "/".join(missing)
            + ". Re-run the proposed eval summary with the current metric code before collecting the table.",
            flush=True,
        )


def print_csv_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} CSV ===")
    if df.empty:
        print("(empty)")
        return
    cols = list(df.columns)
    print(",".join(cols))
    for _, row in df.iterrows():
        print(",".join(csv_escape(row[col]) for col in cols))


def csv_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if any(ch in text for ch in [",", "\"", "\n"]):
        text = "\"" + text.replace("\"", "\"\"") + "\""
    return text


def add_main_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, row in df.iterrows():
        model = str(row["model"])
        method = BASE_MODEL_LABELS.get(model, model)
        pv = "yes"
        nwp = "no"
        swr = "no"
        motion = "no"
        group = str(row["source"])
        if model in {"pv_nwp", "pv_h8_nwp"}:
            nwp = "yes"
        if model in {"pv_h8", "pv_h8_nwp"}:
            swr = "history H8/SWR"
        if group == "proposed":
            method = "Proposed MFP"
            nwp = "yes"
            swr = "future SWR anchors"
            motion = "warp+direct+gate"
        rows.append(
            {
                "method": method,
                "model": model,
                "pv_history": pv,
                "nwp": nwp,
                "swr": swr,
                "motion": motion,
                **{k: row[k] for k in df.columns if k not in {"method", "pv_history", "nwp", "swr", "motion"}},
            }
        )
    out = pd.DataFrame(rows)
    order = {
        "persistence_last": 0,
        "past_mean": 1,
        "linear_trend": 2,
        "pv_only": 3,
        "pv_nwp": 4,
        "pv_h8_nwp": 5,
        "swr_anchor_motion_fusion_pv_nwp": 9,
    }
    fallback_order = pd.Series(50, index=out.index, dtype="float64")
    fallback_order.loc[out["source"].astype(str).eq("proposed")] = 99
    out["_order"] = out["model"].map(order).fillna(fallback_order)
    out = out.sort_values(["_order", "method"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "mae" not in df.columns:
        return pd.DataFrame()
    proposed = df[df["source"].astype(str) == "proposed"]
    if proposed.empty:
        return pd.DataFrame()
    prop = proposed.iloc[0]
    rows = []
    for _, row in df.iterrows():
        if row["source"] == "proposed":
            continue
        item = {
            "baseline": row.get("method", row.get("model", "")),
            "baseline_mae": row["mae"],
            "proposed_mae": prop["mae"],
            "mae_reduction": row["mae"] - prop["mae"],
            "mae_reduction_pct": 100.0 * (row["mae"] - prop["mae"]) / row["mae"] if np.isfinite(row["mae"]) and row["mae"] != 0 else np.nan,
            "baseline_rmse": row["rmse"],
            "proposed_rmse": prop["rmse"],
            "rmse_reduction": row["rmse"] - prop["rmse"],
        }
        rows.append(item)
    return pd.DataFrame(rows)


def simple_ablation_table(name: str, csv_path: Path, split: str, horizon: str) -> pd.DataFrame:
    raw = read_csv_if_exists(csv_path, name)
    return normalize_summary(raw, name, split, horizon)


def main() -> None:
    args = parse_args()
    split = str(args.split)
    horizon = str(args.horizon)

    frames = []
    base_raw = read_csv_if_exists(args.base_csv, "base_models")
    frames.append(normalize_summary(base_raw, "base_models", split, horizon))
    if args.legacy_baseline_csv is not None:
        legacy_raw = read_csv_if_exists(args.legacy_baseline_csv, "legacy_baselines")
        frames.append(normalize_summary(legacy_raw, "legacy_baselines", split, horizon))

    proposed_path = choose_proposed_csv(args.proposed_csv)
    proposed_raw = read_csv_if_exists(proposed_path, "proposed")
    frames.append(normalize_summary(proposed_raw, "proposed", split, horizon))
    main_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    main_df = add_main_metadata(main_df)

    gain_df = build_gain_table(main_df)
    warn_npoint_mismatch(main_df)
    warn_missing_proposed_metrics(main_df)
    print_markdown("Stage 3 Main Baseline Table", main_df, int(args.digits))
    print_csv_block("Stage 3 Main Baseline Table", main_df)
    print_csv_block("Stage 3 Gain vs Proposed", gain_df)

    out_dir = Path(args.out_dir)
    write_table(out_dir / "stage3_main_baseline_table.csv", main_df)
    write_table(out_dir / "stage3_gain_vs_proposed.csv", gain_df)

    ablations = {
        "swr_dependency": args.swr_dependency_csv,
        "motion_branch": args.motion_branch_csv,
        "motion_branch_last_only": args.motion_branch_last_only_csv,
        "temporal_collapse": args.temporal_collapse_csv,
        "anchor_weight": args.anchor_weight_csv,
        "anchor_mixer": args.anchor_mixer_csv,
    }
    combined = []
    for name, path in ablations.items():
        table = simple_ablation_table(name, Path(path), split, horizon)
        if table.empty:
            continue
        combined.append(table)
        print_csv_block(f"Stage 3 {name}", table)
        write_table(out_dir / f"stage3_{name}_table.csv", table)

    if combined:
        all_ablation = pd.concat(combined, ignore_index=True)
        write_table(out_dir / "stage3_all_ablation_tables_long.csv", all_ablation)


if __name__ == "__main__":
    main()
