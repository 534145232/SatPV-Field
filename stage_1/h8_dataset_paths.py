#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", PROJECT_DIR.parent)).expanduser().resolve()
ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
).expanduser().resolve()
RAW_H8_ROOT = Path(
    os.environ.get("H8_RAW_ROOT", ARTIFACT_ROOT / "raw" / "H8")
).expanduser().resolve()
STAGE1_ROOT = Path(
    os.environ.get("SATPV_STAGE1_ROOT", ARTIFACT_ROOT / "stage1")
).expanduser().resolve()
DATA_DIR = STAGE1_ROOT / "data"
RUNS_DIR = STAGE1_ROOT / "runs"

CENTER_REGION_NAME = "center_lat26_34_lon116_126p5"
CENTER_PROCESSED_ROOT = DATA_DIR / "02_center_monthly_netcdf"
CENTER_INDEX_DIR = DATA_DIR / "03_indices" / "center"
CENTER_INDEX_4H3F_ALL = CENTER_INDEX_DIR / "h8_center_index_4hist_3future_06_17_all.csv"
CENTER_FRAME_CACHE_DIR = DATA_DIR / "04_cache" / "center_frame_cache_npy"
CENTER_SAMPLE_CACHE_4H3F_DIR = DATA_DIR / "04_cache" / "center_sample_cache_4hist_3future_float32"
CENTER_CLEAR_SKY_CACHE_4H3F_DIR = DATA_DIR / "04_cache" / "center_clear_sky_cache_4hist_3future"

BIG_PATCH_REGION_NAME = "big3x3_lat18_42_lon105p5_137"
BIG_PATCH_PROCESSED_ROOT = DATA_DIR / "01_big_monthly_netcdf"
BIG_PATCH_INDEX_DIR = DATA_DIR / "03_indices" / "big_patch"
BIG_PATCH_BANK_25_CSV = BIG_PATCH_INDEX_DIR / "patch_bank_25_overlap50.csv"
BIG_PATCH_INDEX_4H3F_TRAIN25_VALTEST_CENTER = (
    BIG_PATCH_INDEX_DIR / "h8_big_patch_index_4hist_3future_06_17_train25_valtest_center.csv"
)
BIG_PATCH_INDEX_4H3F_ALLPATCH_ALL_SPLITS = (
    BIG_PATCH_INDEX_DIR / "h8_big_patch_index_4hist_3future_06_17_allpatch_all_split.csv"
)
BIG_PATCH_FRAME_CACHE_DIR = DATA_DIR / "04_cache" / "big_frame_cache_npy"

CENTER_LAT_MIN = 26.0
CENTER_LAT_MAX = 34.0
CENTER_LON_MIN = 116.0
CENTER_LON_MAX = 126.5

BIG_PATCH_LAT_MIN = 18.0
BIG_PATCH_LAT_MAX = 42.0
BIG_PATCH_LON_MIN = 105.5
BIG_PATCH_LON_MAX = 137.0

PATCH_LAT_SIZE = 8.0
PATCH_LON_SIZE = 10.5
PATCH_LAT_STRIDE = 4.0
PATCH_LON_STRIDE = 5.25
PATCH_BANK_SIZE = 25
