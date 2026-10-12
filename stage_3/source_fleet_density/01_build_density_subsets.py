#!/usr/bin/env python3
"""Build one nested spatially balanced source-fleet density split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from train_regional_dense_reconstruction import splat_values
from visualize_regional_split_frames import _bounds
from stage3_paths import TARGET_EXCLUDED_RUN_DIR, TARGET_SELECTION_DIR

from density_common import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DENSITY_DATA_ROOT,
    bounds_from_station_split,
    density_selection_dir,
    density_target_run_dir,
    ensure_dir,
    haversine_km,
    parse_ratios,
    patch_mean,
    ratio_label,
    station_sns_sha1,
    target_count_for_ratio,
    write_json,
    xy_to_ij,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nested source-station subsets for the density sensitivity experiment. "
            "The 50 target stations are reused unchanged and are never sampled into the source fleet."
        )
    )
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default=",".join(str(r) for r in DEFAULT_RATIOS))
    parser.add_argument("--grid_cells", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--gaussian_sigma_cells", type=float, default=1.0)
    parser.add_argument("--gaussian_radius_cells", type=int, default=2)
    parser.add_argument("--min_grid_weight", type=float, default=1.0e-6)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--local_radius_km", type=float, default=10.0)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"station_sn": str})
    df["station_sn"] = df["station_sn"].astype(str)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)


def _cell_ids(source: pd.DataFrame, grid_cells: int) -> pd.Series:
    lon = source["lon"].to_numpy(np.float64)
    lat = source["lat"].to_numpy(np.float64)
    lon_bin = np.floor((lon - lon.min()) / max(lon.max() - lon.min(), 1.0e-9) * int(grid_cells)).astype(int)
    lat_bin = np.floor((lat - lat.min()) / max(lat.max() - lat.min(), 1.0e-9) * int(grid_cells)).astype(int)
    lon_bin = np.clip(lon_bin, 0, int(grid_cells) - 1)
    lat_bin = np.clip(lat_bin, 0, int(grid_cells) - 1)
    return pd.Series([f"{x:02d}_{y:02d}" for x, y in zip(lon_bin, lat_bin)], index=source.index)


def spatially_balanced_order(source: pd.DataFrame, seed: int, grid_cells: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    work = source.copy().reset_index(drop=True)
    work["density_cell"] = _cell_ids(work, grid_cells)
    cell_order = sorted(work["density_cell"].unique().tolist())
    rng.shuffle(cell_order)
    grouped: dict[str, list[str]] = {}
    for cell in cell_order:
        sns = work.loc[work["density_cell"] == cell, "station_sn"].astype(str).tolist()
        rng.shuffle(sns)
        grouped[cell] = sns
    order = []
    while any(grouped[cell] for cell in cell_order):
        for cell in cell_order:
            if grouped[cell]:
                order.append(grouped[cell].pop(0))
    return order


def _static_support(source: pd.DataFrame, station_split: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    bounds = _bounds(station_split) if len(station_split) else bounds_from_station_split(source)
    vals = source[["station_sn", "lon", "lat"]].copy()
    vals["value"] = 1.0
    field, support = splat_values(vals, source, bounds, args)
    _ = field
    return (np.asarray(support) > float(args.min_grid_weight)), bounds


def observability_rows(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    full_support: np.ndarray,
    support: np.ndarray,
    bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    source_xy = source[["lon", "lat"]].to_numpy(np.float64)
    target_xy = targets[["lon", "lat"]].to_numpy(np.float64)
    per_target = []
    for _, row in targets.iterrows():
        dist = haversine_km(float(row["lon"]), float(row["lat"]), source_xy[:, 0], source_xy[:, 1])
        cy, cx = xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, int(args.grid_size))
        patch_support = patch_mean(support, cy, cx, int(args.patch_size))
        full_patch_support = patch_mean(full_support, cy, cx, int(args.patch_size))
        if full_patch_support <= 1.0e-9:
            patch_support_vs_full = 1.0 if patch_support <= 1.0e-9 else float("nan")
        else:
            patch_support_vs_full = patch_support / full_patch_support
        per_target.append(
            {
                "station_sn": str(row["station_sn"]),
                "nearest_source_distance_km": float(np.nanmin(dist)) if dist.size else np.nan,
                "source_count_within_10km": int(np.sum(dist <= float(args.local_radius_km))) if dist.size else 0,
                "target_patch_support_ratio": patch_support,
                "target_patch_support_ratio_vs_full": patch_support_vs_full,
            }
        )
    target_df = pd.DataFrame(per_target)
    global_support = float(np.asarray(support, dtype=np.float32).mean())
    full_global_support = float(np.asarray(full_support, dtype=np.float32).mean())
    summary = {
        "source_station_count": int(len(source)),
        "global_support_ratio": global_support,
        "global_support_ratio_vs_full": global_support / max(full_global_support, 1.0e-9),
        "target_patch_support_ratio_mean": float(target_df["target_patch_support_ratio"].mean()),
        "target_patch_support_ratio_vs_full_mean": float(target_df["target_patch_support_ratio_vs_full"].mean()),
        "nearest_source_distance_km_median": float(target_df["nearest_source_distance_km"].median()),
        "nearest_source_distance_km_p90": float(target_df["nearest_source_distance_km"].quantile(0.90)),
        "nearest_source_distance_km_max": float(target_df["nearest_source_distance_km"].max()),
        "source_count_within_10km_mean": float(target_df["source_count_within_10km"].mean()),
        "source_count_within_10km_median": float(target_df["source_count_within_10km"].median()),
    }
    return summary, target_df


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    targets = _read_csv(Path(args.selection_dir) / "target_stations.csv")
    full_source = _read_csv(Path(args.target_run_dir) / "target_excluded_station_split.csv")
    station_split_path = Path(args.target_run_dir) / "station_split.csv"
    station_split = _read_csv(station_split_path) if station_split_path.exists() else pd.concat([targets, full_source], ignore_index=True)
    issue_path = Path(args.selection_dir) / "issue_time_index.csv"
    if not issue_path.exists():
        raise FileNotFoundError(issue_path)

    overlap = sorted(set(targets["station_sn"].astype(str)) & set(full_source["station_sn"].astype(str)))
    if overlap:
        raise ValueError(f"Target stations are present in source fleet: {overlap[:10]}")

    ordered_sns = spatially_balanced_order(full_source, args.seed, args.grid_cells)
    full_support, bounds = _static_support(full_source, station_split, args)
    all_summary = []
    for ratio in ratios:
        n = target_count_for_ratio(len(full_source), ratio)
        selected = full_source[full_source["station_sn"].astype(str).isin(set(ordered_sns[:n]))].copy()
        selected["source_density_ratio"] = float(ratio)
        selected["source_density_label"] = ratio_label(ratio)
        selected = selected.sort_values("station_sn").reset_index(drop=True)

        sel_dir = ensure_dir(density_selection_dir(args.seed, ratio))
        run_dir = ensure_dir(density_target_run_dir(args.seed, ratio))
        targets.to_csv(sel_dir / "target_stations.csv", index=False)
        selected.to_csv(sel_dir / "target_excluded_station_split.csv", index=False)
        pd.read_csv(issue_path).to_csv(sel_dir / "issue_time_index.csv", index=False)
        station_split.to_csv(run_dir / "station_split.csv", index=False)
        selected.to_csv(run_dir / "target_excluded_station_split.csv", index=False)

        support, _ = _static_support(selected, station_split, args)
        summary, target_df = observability_rows(targets, selected, full_support, support, bounds, args)
        summary.update(
            {
                "seed": int(args.seed),
                "source_density_ratio": float(ratio),
                "source_density_label": ratio_label(ratio),
                "source_station_sns_sha1": station_sns_sha1(selected),
            }
        )
        target_df.insert(0, "seed", int(args.seed))
        target_df.insert(1, "source_density_ratio", float(ratio))
        target_df.insert(2, "source_density_label", ratio_label(ratio))
        target_df.to_csv(sel_dir / "target_local_observability.csv", index=False)
        write_json(sel_dir / "density_subset_meta.json", summary)
        all_summary.append(summary)

    out = pd.DataFrame(all_summary).sort_values("source_density_ratio").reset_index(drop=True)
    ensure_dir(DENSITY_DATA_ROOT / f"seed{int(args.seed)}")
    out.to_csv(DENSITY_DATA_ROOT / f"seed{int(args.seed)}" / "density_observability_summary.csv", index=False)
    write_json(
        DENSITY_DATA_ROOT / f"seed{int(args.seed)}" / "density_experiment_meta.json",
        {
            "purpose": "Nested spatially balanced source-fleet density sensitivity experiment.",
            "seed": int(args.seed),
            "ratios": ratios,
            "target_station_count": int(len(targets)),
            "full_source_station_count": int(len(full_source)),
            "sampling": "8x8-grid round-robin nested order; S25 subset S50 subset S75 subset S100",
        },
    )
    print(out.to_string(index=False))
    print("saved:", DENSITY_DATA_ROOT / f"seed{int(args.seed)}" / "density_observability_summary.csv")


if __name__ == "__main__":
    main()
