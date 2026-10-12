#!/usr/bin/env python3
"""Check the public repository environment and configured input paths."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true", help="Skip all data-path checks.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when a required path is missing.")
    return parser.parse_args()


def check_imports() -> list[str]:
    required = [
        "numpy",
        "pandas",
        "scipy",
        "xarray",
        "netCDF4",
        "h5py",
        "matplotlib",
        "PIL",
        "torch",
        "openpyxl",
    ]
    optional = ["geopandas", "shapely", "contextily", "cairosvg", "fitz", "shapefile"]
    failures: list[str] = []
    print("Python:", sys.version.replace("\n", " "))
    for name in required:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] import {name}: {version}")
        except Exception as exc:  # import diagnostics should show the concrete failure
            print(f"[MISSING] import {name}: {exc}")
            failures.append(f"import:{name}")
    for name in optional:
        try:
            importlib.import_module(name)
            print(f"[OK] optional import {name}")
        except Exception as exc:
            print(f"[OPTIONAL MISSING] {name}: {exc}")

    try:
        import torch

        print("PyTorch CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    return failures


def configured_paths() -> dict[str, Path]:
    artifact_root = Path(
        os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
    ).expanduser().resolve()
    pv_root = Path(os.environ.get("PV_ROOT", artifact_root / "PV")).expanduser().resolve()
    nwp_root = Path(os.environ.get("NWP_ROOT", artifact_root / "NWP")).expanduser().resolve()
    return {
        "SATPV_PROJECT_ROOT": Path(os.environ.get("SATPV_PROJECT_ROOT", PROJECT_ROOT)).expanduser().resolve(),
        "SATPV_DATA_ROOT": artifact_root,
        "H8_RAW_ROOT": Path(os.environ.get("H8_RAW_ROOT", artifact_root / "raw" / "H8")).expanduser().resolve(),
        "PV train": Path(
            os.environ.get("TRAIN_PV_DIR", pv_root / "train_directory_cleaned_csv_same_day06_nwp")
        ).expanduser().resolve(),
        "PV validation/test": Path(
            os.environ.get("VAL_PV_DIR", pv_root / "val_directory_cleaned_csv_same_day06_nwp")
        ).expanduser().resolve(),
        "NWP farm metadata": Path(
            os.environ.get("NWP_FARM_INFO_XLSX", nwp_root / "old_NWP" / "farm_basic_infomation.xlsx")
        ).expanduser().resolve(),
        "station-to-NWP map": Path(
            os.environ.get("NWP_STATION_MAP_CSV", nwp_root / "old_NWP" / "basic_info_500_with_station.csv")
        ).expanduser().resolve(),
        "station split": Path(
            os.environ.get("STATION_SPLIT_CSV", artifact_root / "station_split.csv")
        ).expanduser().resolve(),
    }


def main() -> None:
    args = parse_args()
    failures = check_imports()
    print("Project root:", PROJECT_ROOT)
    if not args.code_only:
        for name, path in configured_paths().items():
            exists = path.exists()
            print(f"[{'OK' if exists else 'MISSING'}] {name}: {path}")
            if not exists:
                failures.append(f"path:{name}")
    if failures:
        print("\nUnresolved checks:")
        for item in failures:
            print(" -", item)
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()

