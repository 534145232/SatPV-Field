#!/usr/bin/env python3
"""Prepare a station subset for repairing missing Stage-4 baseline targets."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_paths import (
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    TARGET_ISSUE_TIME_INDEX,
    TARGET_SELECTION_DIR,
    TRAIN_PV_DIR,
    VAL_PV_DIR,
)


BASELINE_DETAIL = "per_station_temporal_baseline_eval_detail.csv"


def _load_baseline_module():
    path = STAGE4_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_baseline03", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and prepare missing target stations for baseline repair.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--time_index_csv", type=Path, default=TARGET_ISSUE_TIME_INDEX)
    parser.add_argument("--baseline_run_dir", type=Path, default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR)
    parser.add_argument("--train_pv_dir", type=Path, default=TRAIN_PV_DIR)
    parser.add_argument("--val_pv_dir", type=Path, default=VAL_PV_DIR)
    parser.add_argument("--out_selection_dir", type=Path, default=None)
    parser.add_argument("--missing_baseline_out_dir", type=Path, default=None)
    return parser.parse_args()


def _baseline_args(args: argparse.Namespace) -> SimpleNamespace:
    module = _load_baseline_module()
    values = dict(module.PYCHARM_DEFAULTS)
    values.update(
        {
            "selection_dir": Path(args.selection_dir),
            "time_index_csv": Path(args.time_index_csv),
            "out_dir": Path(args.baseline_run_dir),
            "train_pv_dir": Path(args.train_pv_dir),
            "val_pv_dir": Path(args.val_pv_dir),
            "seed": int(args.seed),
        }
    )
    return SimpleNamespace(**values)


def _sample_counts(args: argparse.Namespace, missing_targets: pd.DataFrame) -> pd.DataFrame:
    module = _load_baseline_module()
    bargs = _baseline_args(args)
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    rows = []
    for _, row in missing_targets.iterrows():
        sn = str(row["station_sn"])
        try:
            station_df, meta = module.load_station_table(sn, bargs)
            train_time = module._split_time_df(time_df, "train")
            stats = module.make_station_stats(station_df, train_time)
            counts = {}
            for split in ["train", "val", "test"]:
                dataset = module.StationTemporalDataset(
                    sn,
                    station_df,
                    row,
                    module._split_time_df(time_df, split),
                    stats,
                    bargs,
                )
                counts[f"{split}_samples"] = int(len(dataset))
            rows.append(
                {
                    "station_sn": sn,
                    "status": "ok",
                    "csv_path": meta.get("csv_path"),
                    "capacity_raw": meta.get("capacity_raw"),
                    "capacity_scale": meta.get("capacity_scale"),
                    "min_train_samples": int(bargs.min_train_samples),
                    **counts,
                    "would_skip_baseline": counts["train_samples"] < int(bargs.min_train_samples),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "station_sn": sn,
                    "status": "error",
                    "error": repr(exc),
                    "min_train_samples": int(bargs.min_train_samples),
                    "would_skip_baseline": True,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.out_selection_dir is None:
        args.out_selection_dir = Path(args.baseline_run_dir).parent / f"baseline_missing_targets_seed{int(args.seed)}_selection"
    if args.missing_baseline_out_dir is None:
        args.missing_baseline_out_dir = Path(args.baseline_run_dir).parent / f"03_per_station_temporal_baseline_missing_seed{int(args.seed)}"

    target_path = Path(args.selection_dir) / "target_stations.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"target_stations.csv not found: {target_path}")
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    configured = set(targets["station_sn"].dropna().astype(str))

    detail_path = Path(args.baseline_run_dir) / BASELINE_DETAIL
    if detail_path.exists():
        detail = pd.read_csv(detail_path, dtype={"station_sn": str})
        if "eval_split" in detail.columns:
            detail = detail[detail["eval_split"].astype(str) == str(args.split)].copy()
        present = set(detail["station_sn"].dropna().astype(str)) if "station_sn" in detail.columns else set()
    else:
        present = set()

    missing_sns = sorted(configured - present)
    missing_targets = targets[targets["station_sn"].astype(str).isin(missing_sns)].copy()
    args.out_selection_dir.mkdir(parents=True, exist_ok=True)
    missing_targets.to_csv(args.out_selection_dir / "target_stations.csv", index=False)
    shutil.copy2(args.time_index_csv, args.out_selection_dir / "issue_time_index.csv")
    counts = _sample_counts(args, missing_targets) if not missing_targets.empty else pd.DataFrame()
    counts.to_csv(args.out_selection_dir / "missing_baseline_sample_counts.csv", index=False)

    print("==== Missing baseline target diagnosis ====")
    print(f"configured_target_stations: {len(configured)}")
    print(f"baseline_detail_stations[{args.split}]: {len(present)}")
    print(f"missing_target_stations: {len(missing_sns)}")
    if missing_sns:
        print("missing_station_sn:", ", ".join(missing_sns))
    if not counts.empty:
        print("\n==== Missing station sample counts ====")
        print(counts.to_string(index=False))

    print("\n==== Repair command: retrain baseline only for missing targets ====")
    print(
        f"{sys.executable} {STAGE4_DIR / '03_train_per_station_temporal_baseline.py'} \\\n"
        f"  --selection_dir {args.out_selection_dir} \\\n"
        f"  --time_index_csv {args.out_selection_dir / 'issue_time_index.csv'} \\\n"
        f"  --out_dir {args.missing_baseline_out_dir} \\\n"
        f"  --seed {int(args.seed)} \\\n"
        f"  --device cuda"
    )
    print("\nAfter the repair run finishes, merge it with:")
    print(
        f"{sys.executable} {SCRIPT_DIR / '05_merge_missing_baseline_targets.py'} \\\n"
        f"  --baseline_run_dir {args.baseline_run_dir} \\\n"
        f"  --missing_baseline_run_dir {args.missing_baseline_out_dir}"
    )
    print("saved:", args.out_selection_dir / "target_stations.csv")
    print("saved:", args.out_selection_dir / "missing_baseline_sample_counts.csv")


if __name__ == "__main__":
    main()
