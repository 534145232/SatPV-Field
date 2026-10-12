"""Build the single large monthly H8 source with bounded memory usage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h8_process_core as base

from stage1_low_memory_monthly import (  # noqa: E402
    is_complete_month,
    save_month_low_memory,
    save_stage1_metadata,
)
from stage1_paths import BIG_MONTHLY_DIR, ensure_stage1_dirs  # noqa: E402


base.DATASET_VARIANT = "stage1_big_region_single_source_low_memory"
base.OUT_DIR = BIG_MONTHLY_DIR
base.LAT_MIN = 18.0
base.LAT_MAX = 42.0
base.LON_MIN = 105.5
base.LON_MAX = 137.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def write_month(month_bjt: str, files: list[Path]) -> Path | None:
    return save_month_low_memory(base, month_bjt, files)


def main() -> None:
    args = parse_args()
    ensure_stage1_dirs()
    print("DATASET_VARIANT:", base.DATASET_VARIANT)
    print("RAW_ROOT:", base.RAW_ROOT)
    print("OUT_DIR:", base.OUT_DIR)
    print("Crop:")
    print(f"  lat: {base.LAT_MIN} to {base.LAT_MAX}")
    print(f"  lon: {base.LON_MIN} to {base.LON_MAX}")
    print("Channels:", base.SELECTED_VARS)
    print("Writer: low-memory two-pass monthly writer")
    print("Daytime filtering:")
    print(f"  keep Beijing start-hour {base.BJT_START_HOUR_KEEP}:00-{base.BJT_END_HOUR_KEEP}:00")

    groups = base.collect_files_by_bjt_month(base.RAW_ROOT)
    if args.months:
        keep_months = {str(month) for month in args.months}
        groups = {key: value for key, value in groups.items() if key in keep_months}
    if not groups:
        raise RuntimeError("No H8 files found")

    saved_files: list[Path] = []
    tasks: list[tuple[str, list[Path]]] = []
    print("\nFound months:")
    for month_bjt in sorted(groups):
        print(f"{month_bjt}: {len(groups[month_bjt])} files")
        expected_path = base.month_output_path(month_bjt)
        if args.skip_existing and is_complete_month(expected_path):
            print(f"[SKIP complete] {expected_path}")
            saved_files.append(expected_path)
        else:
            if expected_path.exists():
                print(f"[REBUILD incomplete or legacy file] {expected_path}")
            tasks.append((month_bjt, groups[month_bjt]))

    num_workers = max(1, args.num_workers)
    print("Month workers:", num_workers)
    if not tasks:
        save_stage1_metadata(base, saved_files)
        print("\nAll months already complete.")
        return
    if num_workers == 1:
        for month_bjt, files in tasks:
            out_path = write_month(month_bjt, files)
            if out_path is not None:
                saved_files.append(out_path)
    else:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as executor:
            futures = {
                executor.submit(write_month, month_bjt, files): month_bjt
                for month_bjt, files in tasks
            }
            for future in as_completed(futures):
                out_path = future.result()
                if out_path is not None:
                    saved_files.append(out_path)

    save_stage1_metadata(base, saved_files)
    print("\nDone.")


if __name__ == "__main__":
    main()
