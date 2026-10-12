#!/usr/bin/env python3
"""Shared helpers for the source-fleet density experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
PIPELINE_DIR = STAGE3_DIR.parent
STAGE4_DIR = PIPELINE_DIR / "stage_4"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

import stage3_paths as p3

DENSITY_DATA_ROOT = Path(
    os.environ.get(
        "DENSITY_DATA_ROOT",
        p3.DATA_DIR / "11_source_fleet_density",
    )
).expanduser().resolve()
DENSITY_STAGE3_RUN_ROOT = p3.RUNS_DIR / "source_fleet_density"
DENSITY_STAGE4_RUN_ROOT = p3.ARTIFACT_ROOT / "stage4" / "runs" / "source_fleet_density"

DEFAULT_SEED = 2026
DEFAULT_RATIOS = (0.25, 0.50, 0.75, 1.00)
RATIO_TO_COUNT = {0.25: 112, 0.50: 223, 0.75: 335, 1.00: 446}

BEST_FULL_STAGE3_RUN_DIR = p3.RUNS_DIR / "ablation_motion_branch_last_only" / "warp_direct_gate"
BEST_FULL_STAGE3_MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate"
BEST_FULL_STAGE3_CHECKPOINT = BEST_FULL_STAGE3_RUN_DIR / f"{BEST_FULL_STAGE3_MODEL_NAME}_best.pt"
BEST_FULL_STAGE4_RUN_DIR = p3.ARTIFACT_ROOT / "stage4" / "runs" / "05l_per_station_temporal_stage3_field_patch_seed2026"


def ratio_label(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):03d}"


def parse_ratios(text: str) -> list[float]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value > 1.0:
            value /= 100.0
        out.append(round(value, 4))
    return out


def density_dir(seed: int, ratio: float) -> Path:
    return DENSITY_DATA_ROOT / f"seed{int(seed)}" / ratio_label(ratio)


def density_selection_dir(seed: int, ratio: float) -> Path:
    return density_dir(seed, ratio) / "selection"


def density_target_run_dir(seed: int, ratio: float) -> Path:
    return density_dir(seed, ratio) / "target_run"


def density_frame_cache_dir(seed: int, ratio: float) -> Path:
    return density_dir(seed, ratio) / "pvnorm_frame_cache"


def density_stage3_run_dir(seed: int, ratio: float) -> Path:
    return DENSITY_STAGE3_RUN_ROOT / f"seed{int(seed)}" / ratio_label(ratio) / "sag_pvff"


def density_stage4_run_dir(seed: int, ratio: float) -> Path:
    return DENSITY_STAGE4_RUN_ROOT / f"seed{int(seed)}" / ratio_label(ratio) / "fgmsf"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def station_sns_sha1(df: pd.DataFrame) -> str:
    sns = sorted(str(v) for v in df["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(sns).encode("utf-8")).hexdigest()


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    r = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def xy_to_ij(lon: float, lat: float, bounds: tuple[float, float, float, float], grid_size: int) -> tuple[int, int]:
    lon_min, lon_max, lat_min, lat_max = bounds
    x = (float(lon) - lon_min) / max(lon_max - lon_min, 1.0e-9)
    y = (float(lat) - lat_min) / max(lat_max - lat_min, 1.0e-9)
    j = int(np.clip(round(x * (int(grid_size) - 1)), 0, int(grid_size) - 1))
    i = int(np.clip(round((1.0 - y) * (int(grid_size) - 1)), 0, int(grid_size) - 1))
    return i, j


def patch_mean(mask: np.ndarray, center_i: int, center_j: int, size: int) -> float:
    radius = int(size) // 2
    y0 = max(0, int(center_i) - radius)
    y1 = min(mask.shape[0], int(center_i) + radius + 1)
    x0 = max(0, int(center_j) - radius)
    x1 = min(mask.shape[1], int(center_j) + radius + 1)
    patch = np.asarray(mask[y0:y1, x0:x1], dtype=np.float32)
    return float(patch.mean()) if patch.size else 0.0


def bounds_from_station_split(station_split: pd.DataFrame) -> tuple[float, float, float, float]:
    lon = pd.to_numeric(station_split["lon"], errors="coerce").to_numpy(np.float64)
    lat = pd.to_numeric(station_split["lat"], errors="coerce").to_numpy(np.float64)
    pad_lon = max(0.01, float(np.nanmax(lon) - np.nanmin(lon)) * 0.02)
    pad_lat = max(0.01, float(np.nanmax(lat) - np.nanmin(lat)) * 0.02)
    return (
        float(np.nanmin(lon) - pad_lon),
        float(np.nanmax(lon) + pad_lon),
        float(np.nanmin(lat) - pad_lat),
        float(np.nanmax(lat) + pad_lat),
    )


def target_count_for_ratio(full_count: int, ratio: float) -> int:
    key = round(float(ratio), 2)
    if key in RATIO_TO_COUNT and int(full_count) == 446:
        return int(RATIO_TO_COUNT[key])
    return int(round(int(full_count) * float(ratio)))


def format_float(value: float) -> float:
    if value is None:
        return float("nan")
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return value
