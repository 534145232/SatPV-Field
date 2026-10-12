#!/usr/bin/env python3
"""Collect station-level density sensitivity results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
STAGE3_DENSITY_DIR = STAGE4_DIR.parent / "stage_3" / "source_fleet_density"
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))
if str(STAGE3_DENSITY_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DENSITY_DIR))

from density_common import (
    BEST_FULL_STAGE4_RUN_DIR,
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DENSITY_DATA_ROOT,
    DENSITY_STAGE4_RUN_ROOT,
    density_selection_dir,
    density_stage4_run_dir,
    parse_ratios,
    ratio_label,
)
from stage4_paths import PER_STATION_TEMPORAL_BASELINE_RUN_DIR, TARGET_SELECTION_DIR


BASELINE_DETAIL = "per_station_temporal_baseline_eval_detail.csv"
FIELD_DETAIL = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
FIELD_AGG = "per_station_temporal_swr_anchor04l_field_patch_aggregate.csv"
BASELINE_AGG = "per_station_temporal_baseline_aggregate.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect source-density station-level results.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default=",".join(str(r) for r in DEFAULT_RATIOS))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--baseline_run_dir", type=Path, default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR)
    parser.add_argument("--full_stage4_run_dir", type=Path, default=BEST_FULL_STAGE4_RUN_DIR)
    parser.add_argument("--out_dir", type=Path, default=DENSITY_STAGE4_RUN_ROOT)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"station_sn": str}, low_memory=False)
    if "station_sn" in df.columns:
        df["station_sn"] = df["station_sn"].astype(str).str.strip()
    return df


def _target_stations(seed: int, ratio: float) -> list[str]:
    candidates = [
        density_selection_dir(seed, ratio) / "target_stations.csv",
        Path(TARGET_SELECTION_DIR) / "target_stations.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, dtype={"station_sn": str})
            return sorted(df["station_sn"].dropna().astype(str).unique().tolist())
    return []


def _run_dir(seed: int, ratio: float, full_stage4_run_dir: Path) -> Path:
    if float(ratio) >= 1.0:
        return Path(full_stage4_run_dir)
    return density_stage4_run_dir(seed, ratio)


def _align_station_details(baseline: pd.DataFrame, proposed: pd.DataFrame, split: str) -> pd.DataFrame:
    base = baseline[baseline["eval_split"].astype(str) == str(split)].copy()
    prop = proposed[proposed["eval_split"].astype(str) == str(split)].copy()
    if base.empty or prop.empty:
        return pd.DataFrame()
    merged = base[KEY_COLS + ["horizon", "true_pv_norm", "pred_pv_norm"]].rename(
        columns={"pred_pv_norm": "pred_baseline", "true_pv_norm": "true_baseline"}
    )
    prop_cols = KEY_COLS + ["true_pv_norm", "pred_pv_norm"]
    merged = merged.merge(
        prop[prop_cols].rename(columns={"pred_pv_norm": "pred_proposed", "true_pv_norm": "true_proposed"}),
        on=KEY_COLS,
        how="inner",
    )
    merged["true_pv_norm"] = merged["true_proposed"].where(merged["true_proposed"].notna(), merged["true_baseline"])
    valid = (
        np.isfinite(merged["true_pv_norm"].astype(float))
        & np.isfinite(merged["pred_baseline"].astype(float))
        & np.isfinite(merged["pred_proposed"].astype(float))
    )
    return merged[valid].reset_index(drop=True)


def _r2_score(true: pd.Series, pred: pd.Series) -> float:
    y = true.astype(float).to_numpy(np.float64)
    p = pred.astype(float).to_numpy(np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.any(valid):
        return float("nan")
    y = y[valid]
    p = p[valid]
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= 1.0e-12:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def _pooled_metrics(aligned: pd.DataFrame) -> dict:
    true = aligned["true_pv_norm"].astype(float)
    base_err = aligned["pred_baseline"].astype(float) - true
    prop_err = aligned["pred_proposed"].astype(float) - true
    base_mae = float(base_err.abs().mean())
    prop_mae = float(prop_err.abs().mean())
    base_rmse = float(np.sqrt(np.mean(np.square(base_err))))
    prop_rmse = float(np.sqrt(np.mean(np.square(prop_err))))
    return {
        "pooled_n_points": int(len(aligned)),
        "pooled_baseline_mae": base_mae,
        "pooled_station_mae": prop_mae,
        "pooled_mae_gain_vs_baseline": base_mae - prop_mae,
        "pooled_baseline_rmse": base_rmse,
        "pooled_station_rmse": prop_rmse,
        "pooled_rmse_gain_vs_baseline": base_rmse - prop_rmse,
        "pooled_baseline_r2": _r2_score(true, aligned["pred_baseline"]),
        "pooled_station_r2": _r2_score(true, aligned["pred_proposed"]),
    }


def _station_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sn, g in aligned.groupby("station_sn", sort=True):
        true = g["true_pv_norm"].astype(float)
        base_err = g["pred_baseline"].astype(float) - true
        prop_err = g["pred_proposed"].astype(float) - true
        base_mae = float(base_err.abs().mean())
        prop_mae = float(prop_err.abs().mean())
        rows.append(
            {
                "station_sn": str(sn),
                "n_points": int(len(g)),
                "baseline_mae": base_mae,
                "proposed_mae": prop_mae,
                "mae_gain": base_mae - prop_mae,
                "baseline_rmse": float(np.sqrt(np.mean(np.square(base_err)))),
                "proposed_rmse": float(np.sqrt(np.mean(np.square(prop_err)))),
                "proposed_better_mae": bool(prop_mae < base_mae),
            }
        )
    return pd.DataFrame(rows)


def _station_accounting(target_sns: list[str], baseline_detail: pd.DataFrame, field_detail: pd.DataFrame, aligned: pd.DataFrame, split: str) -> pd.DataFrame:
    base_split = baseline_detail[baseline_detail["eval_split"].astype(str) == str(split)] if not baseline_detail.empty else pd.DataFrame()
    field_split = field_detail[field_detail["eval_split"].astype(str) == str(split)] if not field_detail.empty else pd.DataFrame()
    base_sns = set(base_split["station_sn"].astype(str).unique().tolist()) if "station_sn" in base_split.columns else set()
    field_sns = set(field_split["station_sn"].astype(str).unique().tolist()) if "station_sn" in field_split.columns else set()
    aligned_sns = set(aligned["station_sn"].astype(str).unique().tolist()) if not aligned.empty else set()
    rows = []
    for sn in target_sns:
        reasons = []
        if sn not in base_sns:
            reasons.append("missing_baseline_detail")
        if sn not in field_sns:
            reasons.append("missing_field_detail")
        if sn in base_sns and sn in field_sns and sn not in aligned_sns:
            reasons.append("no_common_eval_rows")
        rows.append(
            {
                "station_sn": str(sn),
                "configured_target": True,
                "in_baseline_detail": sn in base_sns,
                "in_field_detail": sn in field_sns,
                "in_aligned_eval": sn in aligned_sns,
                "missing_reason": "|".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _horizon_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, g in aligned.groupby("horizon", sort=True):
        true = g["true_pv_norm"].astype(float)
        base_err = g["pred_baseline"].astype(float) - true
        prop_err = g["pred_proposed"].astype(float) - true
        rows.append(
            {
                "horizon": horizon,
                "baseline_mae": float(base_err.abs().mean()),
                "proposed_mae": float(prop_err.abs().mean()),
                "mae_gain": float(base_err.abs().mean() - prop_err.abs().mean()),
                "baseline_rmse": float(np.sqrt(np.mean(np.square(base_err)))),
                "proposed_rmse": float(np.sqrt(np.mean(np.square(prop_err)))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    baseline_detail = _read(Path(args.baseline_run_dir) / BASELINE_DETAIL)
    if baseline_detail.empty:
        raise FileNotFoundError(f"Missing fixed station baseline detail: {Path(args.baseline_run_dir) / BASELINE_DETAIL}")

    obs = _read(DENSITY_DATA_ROOT / f"seed{int(args.seed)}" / "density_observability_summary.csv")
    summary_rows, station_rows, horizon_rows, accounting_rows = [], [], [], []
    full_station_mae = None
    for ratio in parse_ratios(args.ratios):
        run_dir = _run_dir(args.seed, ratio, Path(args.full_stage4_run_dir))
        target_sns = _target_stations(args.seed, ratio)
        field_detail = _read(run_dir / FIELD_DETAIL)
        field_agg = _read(run_dir / FIELD_AGG)
        base = {
            "seed": int(args.seed),
            "source_density_ratio": float(ratio),
            "source_density_label": ratio_label(ratio),
            "n_configured_target_stations": int(len(target_sns)),
            "n_target_stations": int(len(target_sns)),
        }
        if not obs.empty:
            hit = obs[obs["source_density_label"].astype(str) == ratio_label(ratio)]
            if not hit.empty:
                base.update(hit.iloc[0].to_dict())
        if not field_agg.empty:
            hit = field_agg[(field_agg["eval_split"].astype(str) == str(args.split)) & (field_agg["horizon"].astype(str) == "all")]
            if not hit.empty:
                row = hit.iloc[0]
                base["station_macro_mae_from_training_aggregate"] = row.get("station_mean_mae_pv_norm")
                base["station_macro_rmse_from_training_aggregate"] = row.get("station_mean_rmse_pv_norm")
                base["station_macro_r2_from_training_aggregate"] = row.get("station_mean_r2_pv_norm")
                base["field_patch_support_mean"] = row.get("field_patch_support_mean")
                base["mean_supported_pixel_ratio_target_patch"] = row.get("field_patch_support_mean")
        aligned = _align_station_details(baseline_detail, field_detail, str(args.split))
        accounting = _station_accounting(target_sns, baseline_detail, field_detail, aligned, str(args.split))
        accounting.insert(0, "source_density_label", ratio_label(ratio))
        accounting.insert(0, "source_density_ratio", float(ratio))
        accounting.insert(0, "seed", int(args.seed))
        accounting_rows.append(accounting)
        base["n_baseline_detail_stations"] = int(accounting["in_baseline_detail"].sum()) if not accounting.empty else 0
        base["n_field_detail_stations"] = int(accounting["in_field_detail"].sum()) if not accounting.empty else 0
        base["n_evaluated_target_stations"] = int(accounting["in_aligned_eval"].sum()) if not accounting.empty else 0
        base["n_missing_target_stations"] = int((~accounting["in_aligned_eval"]).sum()) if not accounting.empty else 0
        if not aligned.empty:
            pooled = _pooled_metrics(aligned)
            stations = _station_metrics(aligned)
            horizons = _horizon_metrics(aligned)
            stations.insert(0, "source_density_label", ratio_label(ratio))
            stations.insert(0, "source_density_ratio", float(ratio))
            stations.insert(0, "seed", int(args.seed))
            horizons.insert(0, "source_density_label", ratio_label(ratio))
            horizons.insert(0, "source_density_ratio", float(ratio))
            horizons.insert(0, "seed", int(args.seed))
            station_rows.append(stations)
            horizon_rows.append(horizons)
            base.update(pooled)
            base["station_mae"] = pooled["pooled_station_mae"]
            base["station_rmse"] = pooled["pooled_station_rmse"]
            base["station_r2"] = pooled["pooled_station_r2"]
            base["baseline_station_mae"] = pooled["pooled_baseline_mae"]
            base["baseline_station_rmse"] = pooled["pooled_baseline_rmse"]
            base["mae_gain_vs_baseline"] = pooled["pooled_mae_gain_vs_baseline"]
            base["rmse_gain_vs_baseline"] = pooled["pooled_rmse_gain_vs_baseline"]
            base["win_count_vs_baseline"] = int(stations["proposed_better_mae"].sum())
            base["win_denominator"] = int(len(stations))
            base["station_macro_mae_gain_vs_baseline"] = float(stations["mae_gain"].mean())
            base["station_macro_mae_from_detail"] = float(stations["proposed_mae"].mean())
            base["baseline_station_macro_mae_from_detail"] = float(stations["baseline_mae"].mean())
        if float(ratio) >= 1.0:
            full_station_mae = base.get("station_mae")
        summary_rows.append(base)

    summary = pd.DataFrame(summary_rows).sort_values("source_density_ratio").reset_index(drop=True)
    if full_station_mae is not None and np.isfinite(float(full_station_mae)):
        mae_col = "station_mae" if "station_mae" in summary.columns else "station_mae_from_detail"
        summary["station_mae_degradation_vs_r100"] = summary[mae_col].astype(float) - float(full_station_mae)
    out_dir = Path(args.out_dir) / f"seed{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "stage4_density_station_summary.csv", index=False)
    if station_rows:
        pd.concat(station_rows, ignore_index=True).to_csv(out_dir / "stage4_density_station_metrics_by_station.csv", index=False)
    if horizon_rows:
        pd.concat(horizon_rows, ignore_index=True).to_csv(out_dir / "stage4_density_horizon_metrics.csv", index=False)
    if accounting_rows:
        accounting = pd.concat(accounting_rows, ignore_index=True)
        accounting.to_csv(out_dir / "stage4_density_target_station_accounting.csv", index=False)
    else:
        accounting = pd.DataFrame()
    print(summary.to_string(index=False))
    if not accounting.empty:
        print("\n==== Target station accounting ====")
        account_cols = [
            "source_density_label",
            "configured_target",
            "in_baseline_detail",
            "in_field_detail",
            "in_aligned_eval",
        ]
        account_summary = (
            accounting.groupby("source_density_label", sort=True)[account_cols[1:]]
            .sum()
            .rename(
                columns={
                    "configured_target": "configured_targets",
                    "in_baseline_detail": "baseline_detail_stations",
                    "in_field_detail": "field_detail_stations",
                    "in_aligned_eval": "evaluated_targets",
                }
            )
            .reset_index()
        )
        account_summary["missing_targets"] = (
            account_summary["configured_targets"].astype(int)
            - account_summary["evaluated_targets"].astype(int)
        )
        print(account_summary.to_string(index=False))

        missing = accounting[~accounting["in_aligned_eval"].astype(bool)].copy()
        if not missing.empty:
            print("\n==== Missing / non-aligned target stations ====")
            cols = [
                "source_density_label",
                "station_sn",
                "in_baseline_detail",
                "in_field_detail",
                "in_aligned_eval",
                "missing_reason",
            ]
            print(missing[cols].sort_values(["source_density_label", "station_sn"]).to_string(index=False))
        else:
            print("\nAll configured target stations are included in aligned evaluation.")
    print("saved:", out_dir / "stage4_density_station_summary.csv")


if __name__ == "__main__":
    main()
