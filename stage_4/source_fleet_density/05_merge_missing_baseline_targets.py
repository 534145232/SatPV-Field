#!/usr/bin/env python3
"""Merge repaired missing-target Stage-4 baseline outputs into the fixed baseline run."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_paths import PER_STATION_TEMPORAL_BASELINE_RUN_DIR


FILES_WITH_STATION_SN = [
    "per_station_temporal_baseline_eval_detail.csv",
    "per_station_temporal_baseline_metrics.csv",
    "per_station_temporal_baseline_train_log.csv",
    "station_load_meta.csv",
]


def _load_baseline_module():
    path = STAGE4_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_baseline03", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge repaired missing baseline stations into the baseline run.")
    parser.add_argument("--baseline_run_dir", type=Path, default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR)
    parser.add_argument("--missing_baseline_run_dir", type=Path, required=True)
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--disable_backup", action="store_false", dest="backup")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"station_sn": str})


def _backup(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def _merge_file(base_dir: Path, patch_dir: Path, filename: str, backup_dir: Path | None) -> pd.DataFrame:
    base_path = base_dir / filename
    patch_path = patch_dir / filename
    base = _read(base_path)
    patch = _read(patch_path)
    if patch.empty:
        print(f"[skip] missing or empty repair file: {patch_path}")
        return base
    if "station_sn" not in patch.columns:
        raise ValueError(f"Repair file has no station_sn column: {patch_path}")
    patch_sns = set(patch["station_sn"].dropna().astype(str))
    kept = base[~base["station_sn"].astype(str).isin(patch_sns)].copy() if (not base.empty and "station_sn" in base.columns) else base
    merged = pd.concat([kept, patch], ignore_index=True)
    if backup_dir is not None:
        _backup(base_path, backup_dir)
    merged.to_csv(base_path, index=False)
    print(
        f"[merged] {filename}: base_rows={len(base)}, repair_rows={len(patch)}, "
        f"output_rows={len(merged)}, repaired_stations={len(patch_sns)}"
    )
    return merged


def main() -> None:
    args = parse_args()
    base_dir = Path(args.baseline_run_dir)
    patch_dir = Path(args.missing_baseline_run_dir)
    if not patch_dir.exists():
        raise FileNotFoundError(f"Missing repair run directory: {patch_dir}")
    backup_dir = None
    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = base_dir / f"_backup_before_missing_merge_{stamp}"

    merged = {}
    for filename in FILES_WITH_STATION_SN:
        merged[filename] = _merge_file(base_dir, patch_dir, filename, backup_dir)

    metrics = merged.get("per_station_temporal_baseline_metrics.csv", pd.DataFrame())
    if not metrics.empty:
        module = _load_baseline_module()
        aggregate = module.aggregate(metrics)
        agg_path = base_dir / "per_station_temporal_baseline_aggregate.csv"
        if backup_dir is not None:
            _backup(agg_path, backup_dir)
        aggregate.to_csv(agg_path, index=False)
        print("[recomputed] per_station_temporal_baseline_aggregate.csv")
        print(aggregate[aggregate["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False))

    detail = merged.get("per_station_temporal_baseline_eval_detail.csv", pd.DataFrame())
    if not detail.empty and "station_sn" in detail.columns:
        split_counts = (
            detail.groupby("eval_split")["station_sn"]
            .nunique()
            .reset_index(name="n_stations")
            .sort_values("eval_split")
        )
        print("\n==== Baseline detail station counts after merge ====")
        print(split_counts.to_string(index=False))
    if backup_dir is not None:
        print("backup:", backup_dir)
    print("merged baseline run:", base_dir)


if __name__ == "__main__":
    main()
