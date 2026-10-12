#!/usr/bin/env python3
"""Report how many stations are used to construct the Stage 3 PV field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage3_paths import (
    SOURCE_REGIONAL_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the original station count, selected 50 target stations, "
            "and the remaining source-station count actually used to build the Stage 3 PV field."
        )
    )
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--source_station_split", type=Path, default=SOURCE_REGIONAL_RUN_DIR / "station_split.csv")
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def read_station_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_sn": str})
    if "station_sn" not in df.columns:
        raise ValueError(f"{path} does not contain station_sn column.")
    df["station_sn"] = df["station_sn"].astype(str)
    return df


def resolve_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} not found. Tried: {[str(p) for p in candidates]}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()

    target_path = resolve_existing(
        [Path(args.selection_dir) / "target_stations.csv"],
        "target_stations.csv",
    )
    excluded_path = resolve_existing(
        [
            Path(args.target_run_dir) / "target_excluded_station_split.csv",
            Path(args.selection_dir) / "target_excluded_station_split.csv",
        ],
        "target_excluded_station_split.csv",
    )
    frame_meta_path = resolve_existing(
        [Path(args.frame_cache_dir) / "frame_cache_meta.json"],
        "frame_cache_meta.json",
    )

    targets = read_station_csv(target_path)
    excluded = read_station_csv(excluded_path)
    frame_meta = load_json(frame_meta_path)

    original_count = None
    original_path = Path(args.source_station_split)
    if original_path.exists():
        original_count = int(len(read_station_csv(original_path)))

    target_count = int(len(targets))
    pv_field_source_count = int(len(excluded))
    frame_cache_source_count = int(frame_meta.get("source_station_count", -1))
    overlap_count = int(
        len(set(targets["station_sn"].astype(str)) & set(excluded["station_sn"].astype(str)))
    )

    report = {
        "target_stations_csv": str(target_path),
        "target_excluded_station_split_csv": str(excluded_path),
        "frame_cache_meta_json": str(frame_meta_path),
        "source_station_split_csv": str(original_path) if original_path.exists() else None,
        "original_station_count": original_count,
        "target_station_count": target_count,
        "pv_field_source_station_count": pv_field_source_count,
        "frame_cache_source_station_count": frame_cache_source_count,
        "target_overlap_with_pv_field_sources": overlap_count,
        "frame_cache_source_station_policy": frame_meta.get("source_station_policy"),
        "counts_match": frame_cache_source_count == pv_field_source_count,
    }

    print("==== Stage 3 PV-field source station count ====")
    if original_count is not None:
        print(f"original stations: {original_count}")
    else:
        print("original stations: unavailable (source station split not found)")
    print(f"selected target stations: {target_count}")
    print(f"PV-field source stations after exclusion: {pv_field_source_count}")
    print(f"frame-cache metadata source_station_count: {frame_cache_source_count}")
    print(f"target/source overlap: {overlap_count}")
    print(f"counts match: {report['counts_match']}")
    print(f"policy: {report['frame_cache_source_station_policy']}")

    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json: {args.out_json}")


if __name__ == "__main__":
    main()
