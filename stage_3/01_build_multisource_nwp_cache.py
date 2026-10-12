#!/usr/bin/env python3
"""Build the PV/NWP multisource cache with the leakage-safe Stage 3 clock."""

import json
import sys

import numpy as np
import pandas as pd

from stage3_paths import (
    LEGACY_CENTER_SWR_CACHE_DIR,
    MULTISOURCE_CACHE_DIR,
    NWP_FARM_INFO_XLSX,
    NWP_STATION_MAP_CSV,
    SOURCE_REGIONAL_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TIME_END,
    TIME_START,
    TRAIN_END,
    TRAIN_PV_DIR,
    VAL_END,
    VAL_PV_DIR,
    ensure_stage3_dirs,
)
from stage3_cli import inject_defaults
import stage3_build_multisource_cache_impl as module
from stage3_temporal import build_pv_issue_time_table, swr_history_times


def force_requested() -> bool:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--force_rebuild" and index + 2 <= len(sys.argv[1:]):
            return bool(int(sys.argv[1:][index + 1]))
        if value.startswith("--force_rebuild="):
            return bool(int(value.split("=", 1)[1]))
    return False


def validate_multisource_temporal_cache() -> None:
    time_path = MULTISOURCE_CACHE_DIR / "issue_time_index.csv"
    h8_sec_path = MULTISOURCE_CACHE_DIR / "h8_frame_sec.npy"
    if not time_path.exists() or not h8_sec_path.exists():
        return
    table = pd.read_csv(time_path)
    h8_sec = np.load(h8_sec_path, mmap_mode="r")
    if len(table) != len(h8_sec):
        raise AssertionError(f"Multisource cache row mismatch: {len(table)} != {len(h8_sec)}")
    for index, row in table.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        past_end = pd.Timestamp(row["past_end"])
        future_start = pd.Timestamp(row["future_start"])
        expected_h8 = swr_history_times(issue, int(h8_sec.shape[1]))
        actual_h8 = [pd.Timestamp(value, unit="s") for value in h8_sec[index]]
        if past_end != issue - pd.Timedelta(minutes=15):
            raise AssertionError(f"PV history does not end at T-15min: {issue}")
        if future_start != issue:
            raise AssertionError(f"PV future does not start at T: {issue}")
        if actual_h8 != expected_h8:
            raise AssertionError(f"Legacy H8 context is not T-4h..T-1h: {issue}")
    meta_path = MULTISOURCE_CACHE_DIR / "multisource_cache_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        info.update(
            {
                "stage3_temporal_protocol": (
                    "PV history=T-3h..T-15min; PV target=T..T+1h45min; "
                    "legacy H8 context=T-4h..T-1h."
                ),
                "h8_alignment": "PV issue T, observed hourly frames=[T-4h, T-3h, T-2h, T-1h]",
                "h8_age_minutes": "PV issue T - latest observed H8 label T-1h = 60 minutes.",
            }
        )
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
    print("Validated Stage 3 multisource temporal cache: PV target starts at T; observed H8 ends at T-1h")


def validate_multisource_h8_values() -> None:
    h8_path = MULTISOURCE_CACHE_DIR / "h8_swr_64_float16.npy"
    if not h8_path.exists():
        return
    h8 = np.load(h8_path, mmap_mode="r")
    finite = np.isfinite(h8)
    finite_ratio = float(finite.mean()) if h8.size else 0.0
    finite_values = np.asarray(h8[finite], dtype=np.float32)
    print(
        "Validated Stage 3 multisource H8 values: "
        f"shape={tuple(h8.shape)} finite_ratio={finite_ratio:.6f}",
        flush=True,
    )
    if finite_values.size:
        print(
            "  finite SWR range: "
            f"min={float(np.nanmin(finite_values)):.3f} "
            f"max={float(np.nanmax(finite_values)):.3f} "
            f"mean={float(np.nanmean(finite_values)):.3f}",
            flush=True,
        )
    if finite_ratio < 0.95:
        raise RuntimeError(
            "Stage 3 multisource H8 cache is mostly non-finite. "
            "Run 00_diagnose_stage3_h8_cache.py, then rebuild this cache with --force_rebuild 1."
        )


def main() -> None:
    ensure_stage3_dirs()
    try:
        validate_multisource_temporal_cache()
        validate_multisource_h8_values()
    except (AssertionError, RuntimeError) as exc:
        if not force_requested():
            raise RuntimeError(
                "Existing Stage 3 multisource cache is stale, misaligned or contains invalid H8 values. "
                "Rebuild with --force_rebuild 1."
            ) from exc
        print(
            "Existing multisource cache failed validation, but --force_rebuild 1 was requested. "
            f"Rebuilding now. Reason: {exc}",
            flush=True,
        )
    module.NEW_DEFAULTS.update(
        {
            "run_dir": str(SOURCE_REGIONAL_RUN_DIR),
            "out_dir": str(MULTISOURCE_CACHE_DIR),
            "pv_frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "h8_cache_dir": str(LEGACY_CENTER_SWR_CACHE_DIR),
            "train_pv_dir": str(TRAIN_PV_DIR),
            "val_pv_dir": str(VAL_PV_DIR),
            "nwp_farm_info_xlsx": str(NWP_FARM_INFO_XLSX),
            "nwp_station_map_csv": str(NWP_STATION_MAP_CSV),
            "time_start": TIME_START,
            "time_end": TIME_END,
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "issue_stride_minutes": 60,
            "past_frames": 12,
            "future_frames": 8,
            "force_rebuild": 0,
        }
    )
    module.base.build_multisource_issue_time_table = build_pv_issue_time_table
    module.base._select_issue_h8_times = swr_history_times
    inject_defaults(module.NEW_DEFAULTS)
    module.main()
    validate_multisource_temporal_cache()
    validate_multisource_h8_values()


if __name__ == "__main__":
    main()
