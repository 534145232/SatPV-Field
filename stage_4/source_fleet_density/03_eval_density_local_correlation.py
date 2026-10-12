#!/usr/bin/env python3
"""Correlate local source observability with station-level degradation."""

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

from density_common import DEFAULT_RATIOS, DEFAULT_SEED, DENSITY_STAGE4_RUN_ROOT, density_selection_dir, parse_ratios, ratio_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local density vs station degradation correlations.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--station_metrics_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DENSITY_STAGE4_RUN_ROOT)
    return parser.parse_args()


def _spearman(df: pd.DataFrame, x: str, y: str) -> float:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return float("nan")
    return float(sub[x].rank().corr(sub[y].rank(), method="pearson"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / f"seed{int(args.seed)}"
    station_path = args.station_metrics_csv or (out_dir / "stage4_density_station_metrics_by_station.csv")
    if not station_path.exists():
        raise FileNotFoundError(f"Station metrics not found. Run 02_collect_density_station_results.py first: {station_path}")
    stations = pd.read_csv(station_path, dtype={"station_sn": str})
    full = stations[stations["source_density_label"].astype(str) == "r100"][["station_sn", "proposed_mae"]].rename(
        columns={"proposed_mae": "r100_proposed_mae"}
    )
    rows = []
    merged_frames = []
    for ratio in parse_ratios(args.ratios):
        label = ratio_label(ratio)
        obs_path = density_selection_dir(args.seed, ratio) / "target_local_observability.csv"
        if not obs_path.exists():
            continue
        obs = pd.read_csv(obs_path, dtype={"station_sn": str})
        cur = stations[stations["source_density_label"].astype(str) == label].copy()
        if cur.empty:
            continue
        merged = cur.merge(full, on="station_sn", how="left").merge(obs, on="station_sn", how="left", suffixes=("", "_obs"))
        merged["mae_degradation_vs_r100"] = merged["proposed_mae"].astype(float) - merged["r100_proposed_mae"].astype(float)
        merged_frames.append(merged)
        for feature in (
            "nearest_source_distance_km",
            "source_count_within_10km",
            "target_patch_support_ratio",
            "target_patch_support_ratio_vs_full",
        ):
            rows.append(
                {
                    "seed": int(args.seed),
                    "source_density_ratio": float(ratio),
                    "source_density_label": label,
                    "feature": feature,
                    "target": "mae_degradation_vs_r100",
                    "spearman_rho": _spearman(merged, feature, "mae_degradation_vs_r100"),
                    "n_stations": int(merged[[feature, "mae_degradation_vs_r100"]].dropna().shape[0]),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "stage4_density_local_correlation.csv", index=False)
    if merged_frames:
        pd.concat(merged_frames, ignore_index=True).to_csv(out_dir / "stage4_density_station_metrics_with_local_observability.csv", index=False)
    print(corr.to_string(index=False) if not corr.empty else "EMPTY")
    print("saved:", out_dir / "stage4_density_local_correlation.csv")


if __name__ == "__main__":
    main()
