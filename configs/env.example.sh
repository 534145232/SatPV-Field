#!/usr/bin/env bash

# Repository root. `pwd` is correct when this file is sourced from the repo root.
export SATPV_PROJECT_ROOT="$(pwd)"

# External artifacts root. Keep data, caches, checkpoints, and result CSVs here.
export SATPV_DATA_ROOT="/path/to/satpv_artifacts"

# Raw inputs. These may live outside SATPV_DATA_ROOT.
export H8_RAW_ROOT="/path/to/raw/H8"
export PV_ROOT="/path/to/PV"
export NWP_ROOT="/path/to/NWP"
export STATION_SPLIT_CSV="/path/to/station_split.csv"

# Optional explicit overrides when directory names differ.
# export TRAIN_PV_DIR="${PV_ROOT}/train_directory_cleaned_csv_same_day06_nwp"
# export VAL_PV_DIR="${PV_ROOT}/val_directory_cleaned_csv_same_day06_nwp"
# export NWP_FARM_INFO_XLSX="${NWP_ROOT}/old_NWP/farm_basic_infomation.xlsx"
# export NWP_STATION_MAP_CSV="${NWP_ROOT}/old_NWP/basic_info_500_with_station.csv"

# Shell launchers use this executable.
export PYTHON="python"

