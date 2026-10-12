#!/usr/bin/env python3
"""Count stations where the proposed Stage 4 method beats the temporal baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_paths import (
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)


BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
FIELD_DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--proposed_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "15_stage4_station_win_rate")
    parser.add_argument("--digits", type=int, default=2)
    return parser.parse_args()


def read_detail(path: Path, label: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {label} eval-detail CSV: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = KEY_COLS + ["pred_pv_norm", "true_pv_norm", "horizon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df.copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["horizon"] = out["horizon"].astype(str)
    out["pred_pv_norm"] = pd.to_numeric(out["pred_pv_norm"], errors="coerce")
    out["true_pv_norm"] = pd.to_numeric(out["true_pv_norm"], errors="coerce")
    out["model_label"] = label
    return out


def align_details(baseline: pd.DataFrame, proposed: pd.DataFrame, split: str) -> pd.DataFrame:
    base = baseline[baseline["eval_split"].astype(str) == str(split)].copy()
    prop = proposed[proposed["eval_split"].astype(str) == str(split)].copy()
    base_cols = KEY_COLS + ["horizon", "true_pv_norm", "pred_pv_norm"]
    prop_cols = KEY_COLS + ["true_pv_norm", "pred_pv_norm"]
    merged = base[base_cols].rename(columns={"pred_pv_norm": "pred_baseline"})
    merged = merged.merge(
        prop[prop_cols].rename(
            columns={
                "pred_pv_norm": "pred_proposed",
                "true_pv_norm": "true_proposed",
            }
        ),
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
    )
    truth_diff = (merged["true_pv_norm"].astype(float) - merged["true_proposed"].astype(float)).abs()
    max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
    if np.isfinite(max_diff) and max_diff > 1.0e-5:
        print(f"[warn] max truth mismatch after alignment: {max_diff:.6g}", flush=True)
    merged = merged.rename(columns={"true_pv_norm": "true_pv_norm_baseline"})
    merged["true_pv_norm"] = merged["true_proposed"].where(
        merged["true_proposed"].notna(),
        merged["true_pv_norm_baseline"],
    )
    valid = (
        np.isfinite(merged["true_pv_norm"].astype(float))
        & np.isfinite(merged["pred_baseline"].astype(float))
        & np.isfinite(merged["pred_proposed"].astype(float))
    )
    return merged.loc[valid].copy()


def station_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, g in merged.groupby("station_sn", sort=True):
        true = g["true_pv_norm"].astype(float)
        base_err = g["pred_baseline"].astype(float) - true
        prop_err = g["pred_proposed"].astype(float) - true
        base_mae = float(base_err.abs().mean())
        prop_mae = float(prop_err.abs().mean())
        base_rmse = float(np.sqrt(np.mean(np.square(base_err))))
        prop_rmse = float(np.sqrt(np.mean(np.square(prop_err))))
        rows.append(
            {
                "station_sn": str(station),
                "n_points": int(len(g)),
                "baseline_mae": base_mae,
                "proposed_mae": prop_mae,
                "mae_gain": base_mae - prop_mae,
                "mae_gain_pct": 100.0 * (base_mae - prop_mae) / base_mae if base_mae > 0 else float("nan"),
                "proposed_better_mae": bool(prop_mae < base_mae),
                "baseline_rmse": base_rmse,
                "proposed_rmse": prop_rmse,
                "rmse_gain": base_rmse - prop_rmse,
                "rmse_gain_pct": 100.0 * (base_rmse - prop_rmse) / base_rmse if base_rmse > 0 else float("nan"),
                "proposed_better_rmse": bool(prop_rmse < base_rmse),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_gain", ascending=False).reset_index(drop=True)


def horizon_win_rates(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, hdf in merged.groupby("horizon", sort=True):
        station_rows = station_metrics(hdf)
        total = int(len(station_rows))
        wins = int(station_rows["proposed_better_mae"].sum()) if total else 0
        rows.append(
            {
                "horizon": str(horizon),
                "n_stations": total,
                "proposed_better_stations": wins,
                "win_rate_pct": 100.0 * wins / total if total else float("nan"),
                "mean_mae_gain": float(station_rows["mae_gain"].mean()) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def print_summary(stations: pd.DataFrame, horizons: pd.DataFrame, digits: int) -> None:
    total = int(len(stations))
    wins = int(stations["proposed_better_mae"].sum()) if total else 0
    losses = int((~stations["proposed_better_mae"]).sum()) if total else 0
    rate = 100.0 * wins / total if total else float("nan")
    mean_gain = float(stations["mae_gain"].mean()) if total else float("nan")
    median_gain = float(stations["mae_gain"].median()) if total else float("nan")
    print("==== Stage 4 station-wise win rate ====")
    print(f"Metric: station-level MAE over aligned test samples")
    print(f"Proposed better than temporal baseline: {wins}/{total} stations = {rate:.{digits}f}%")
    print(f"Not better or tied: {losses}/{total} stations = {(100.0 - rate):.{digits}f}%")
    print(f"Mean station MAE gain: {mean_gain:.6f}")
    print(f"Median station MAE gain: {median_gain:.6f}")
    print()
    print("==== Horizon-wise station win rate ====")
    if horizons.empty:
        print("EMPTY")
    else:
        show = horizons.copy()
        for col in ["win_rate_pct", "mean_mae_gain"]:
            show[col] = show[col].map(lambda x: f"{float(x):.{digits}f}" if np.isfinite(float(x)) else "")
        print(show.to_string(index=False))
    print()
    print("==== Per-station MAE gain, sorted descending ====")
    cols = ["station_sn", "n_points", "baseline_mae", "proposed_mae", "mae_gain", "mae_gain_pct", "proposed_better_mae"]
    show = stations[cols].copy()
    for col in ["baseline_mae", "proposed_mae", "mae_gain", "mae_gain_pct"]:
        show[col] = show[col].map(lambda x: f"{float(x):.{digits}f}" if np.isfinite(float(x)) else "")
    print(show.to_string(index=False))


def main() -> None:
    args = parse_args()
    baseline = read_detail(args.baseline_detail_csv, "temporal_baseline")
    proposed = read_detail(args.proposed_detail_csv, "proposed")
    merged = align_details(baseline, proposed, str(args.split))
    if merged.empty:
        raise RuntimeError(f"No aligned rows for split={args.split!r}")
    stations = station_metrics(merged)
    horizons = horizon_win_rates(merged)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "aligned_station_win_detail.csv", index=False)
    stations.to_csv(args.out_dir / "station_win_rate_by_station.csv", index=False)
    horizons.to_csv(args.out_dir / "station_win_rate_by_horizon.csv", index=False)

    total = int(len(stations))
    wins = int(stations["proposed_better_mae"].sum()) if total else 0
    summary = {
        "split": str(args.split),
        "baseline_detail_csv": str(args.baseline_detail_csv),
        "proposed_detail_csv": str(args.proposed_detail_csv),
        "n_aligned_rows": int(len(merged)),
        "n_stations": total,
        "proposed_better_stations_mae": wins,
        "proposed_better_station_rate_pct_mae": 100.0 * wins / total if total else float("nan"),
        "mean_station_mae_gain": float(stations["mae_gain"].mean()) if total else float("nan"),
        "median_station_mae_gain": float(stations["mae_gain"].median()) if total else float("nan"),
    }
    (args.out_dir / "station_win_rate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary(stations, horizons, int(args.digits))
    print(f"\n[write] {args.out_dir}")


if __name__ == "__main__":
    main()
