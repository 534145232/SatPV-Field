#!/usr/bin/env python3
"""Preflight audit for the matched field-versus-graph representation experiment.

This script is intentionally read-only with respect to the Stage-3/Stage-4
artifacts.  It verifies the leakage boundary, the temporal protocol, cache
coverage, normalization provenance, and the exact spatial support available to
each held-out target.  It also exports a source-fleet geographic kNN graph for
the later graph branch.

It does *not* train a model and it never reads future target PV as an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_SUPPORT_DIR = SCRIPT_DIR.parent
if str(PAPER_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SUPPORT_DIR))

from paper_support_paths import resolve_old_data_root, resolve_stage3_data_dir  # noqa: E402


ID_ALIASES = ("station_sn", "station_id", "site_id", "id")
LON_ALIASES = ("lon", "longitude", "lng")
LAT_ALIASES = ("lat", "latitude")
ISSUE_COLUMNS = ("issue_sec", "time_split", "past_times", "future_times")
NWP_VARIABLES = ("radiation", "direction10", "speed10", "tcc")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    critical: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit inputs and build geometry for a matched field-vs-graph experiment."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=SCRIPT_DIR.parents[1],
        help="SatPV-Field repository root.",
    )
    parser.add_argument("--old-data-root", type=Path, default=None)
    parser.add_argument("--stage3-data-dir", type=Path, default=None)
    parser.add_argument(
        "--stage3-model-dir",
        type=Path,
        default=None,
        help="Optional trained Stage-3 run containing normalization JSON/checkpoint metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "audit",
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--gaussian-radius-cells", type=int, default=2)
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--expected-all-stations", type=int, default=496)
    parser.add_argument("--expected-source-stations", type=int, default=446)
    parser.add_argument("--expected-target-stations", type=int, default=50)
    parser.add_argument("--expected-train-issues", type=int, default=2268)
    parser.add_argument("--expected-val-issues", type=int, default=324)
    parser.add_argument("--expected-test-issues", type=int, default=648)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def choose_column(columns: Iterable[str], aliases: Iterable[str], purpose: str) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(f"No {purpose} column found. Available columns: {list(columns)}")


def load_station_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    raw = pd.read_csv(path, dtype=str)
    id_col = choose_column(raw.columns, ID_ALIASES, "station ID")
    lon_col = choose_column(raw.columns, LON_ALIASES, "longitude")
    lat_col = choose_column(raw.columns, LAT_ALIASES, "latitude")
    out = raw.loc[raw[id_col].notna()].copy()
    out["station_sn"] = out[id_col].astype(str).str.strip()
    out = out.loc[(out["station_sn"] != "") & (out["station_sn"].str.lower() != "nan")].copy()
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    return out.reset_index(drop=True)


def station_sha1(stations: pd.DataFrame) -> str:
    sns = sorted(stations["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(sns).encode("utf-8")).hexdigest()


def add_check(
    checks: list[Check],
    name: str,
    condition: bool | None,
    pass_detail: str,
    fail_detail: str,
    *,
    critical: bool = False,
    unavailable_detail: str | None = None,
) -> None:
    if condition is None:
        checks.append(Check(name, "PENDING", unavailable_detail or fail_detail, critical=False))
    elif bool(condition):
        checks.append(Check(name, "PASS", pass_detail, critical=critical))
    else:
        checks.append(Check(name, "FAIL" if critical else "WARN", fail_detail, critical=critical))


def bounds_from_all_stations(stations: pd.DataFrame) -> tuple[float, float, float, float]:
    """Match stage_3.visualize_regional_split_frames._bounds exactly."""

    lon_min = float(stations["lon"].min())
    lon_max = float(stations["lon"].max())
    lat_min = float(stations["lat"].min())
    lat_max = float(stations["lat"].max())
    lon_pad = max(1e-6, (lon_max - lon_min) * 0.04)
    lat_pad = max(1e-6, (lat_max - lat_min) * 0.04)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def xy_to_ij(
    lon: float,
    lat: float,
    bounds: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[int, int]:
    """Match Stage-3/Stage-4 round-and-clip grid assignment exactly."""

    lon_min, lon_max, lat_min, lat_max = bounds
    x = int(round((float(lon) - lon_min) / max(1e-12, lon_max - lon_min) * (grid_size - 1)))
    y = int(round((float(lat) - lat_min) / max(1e-12, lat_max - lat_min) * (grid_size - 1)))
    return int(np.clip(y, 0, grid_size - 1)), int(np.clip(x, 0, grid_size - 1))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def inspect_npy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return {
            "path": str(path),
            "exists": True,
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "bytes": int(path.stat().st_size),
        }
    except Exception as exc:  # keep audit report available even for a corrupt optional cache
        return {"path": str(path), "exists": True, "error": f"{type(exc).__name__}: {exc}"}


def resolve_swr_cache_dir(cache_root: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a copied active/versioned cache without trusting stale absolute paths."""

    manifest_path = cache_root / "active_cache.json"
    manifest = read_json(manifest_path)
    candidates: list[Path] = []
    recorded = manifest.get("active_cache_dir")
    if recorded:
        recorded_path = Path(str(recorded)).expanduser()
        candidates.append(recorded_path)
        # A cache copied from another server retains an obsolete absolute root,
        # but its version-directory basename remains portable.
        candidates.append(cache_root / recorded_path.name)
    candidates.append(cache_root)
    for child in sorted(cache_root.glob("*")) if cache_root.is_dir() else []:
        if child.is_dir():
            candidates.append(child)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "swr_anchor_64.npy").is_file() and (
            candidate / "center_h8_swr_anchor_input_cache_meta.csv"
        ).is_file():
            return candidate.resolve(), {
                "manifest_path": str(manifest_path),
                "manifest": manifest,
                "recorded_active_cache_dir": str(recorded) if recorded else None,
                "resolved_active_cache_dir": str(candidate.resolve()),
            }
    return cache_root, {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "recorded_active_cache_dir": str(recorded) if recorded else None,
        "resolved_active_cache_dir": None,
    }


def canonical_issue_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing issue-time index: {path}")
    table = pd.read_csv(path)
    if "issue_time" not in table.columns and "issue_sec" not in table.columns:
        raise ValueError(f"No issue_time or issue_sec in {path}")
    if "issue_time" in table.columns:
        table["issue_time"] = pd.to_datetime(table["issue_time"], errors="coerce")
    else:
        table["issue_time"] = pd.to_datetime(table["issue_sec"], unit="s", errors="coerce")
    if "issue_sec" not in table.columns:
        table["issue_sec"] = table["issue_time"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    table["issue_sec"] = pd.to_numeric(table["issue_sec"], errors="coerce").astype("Int64")
    return table


def split_times(text: Any) -> list[pd.Timestamp]:
    if pd.isna(text):
        return []
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def issue_signature(table: pd.DataFrame) -> str | None:
    if any(column not in table.columns for column in ISSUE_COLUMNS):
        return None
    canonical = table[list(ISSUE_COLUMNS)].fillna("").astype(str)
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def compare_issue_tables(reference: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    if any(column not in reference.columns or column not in candidate.columns for column in ISSUE_COLUMNS):
        return False
    left = reference[list(ISSUE_COLUMNS)].reset_index(drop=True).fillna("").astype(str)
    right = candidate[list(ISSUE_COLUMNS)].reset_index(drop=True).fillna("").astype(str)
    return len(left) == len(right) and left.equals(right)


def audit_temporal_protocol(
    time_df: pd.DataFrame,
    checks: list[Check],
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    required = {"issue_time", "issue_sec", "time_split", "past_times", "future_times"}
    missing = sorted(required - set(time_df.columns))
    add_check(
        checks,
        "canonical issue-time columns",
        not missing,
        "All canonical temporal columns are present.",
        f"Missing columns: {missing}",
        critical=True,
    )
    if missing:
        return {"rows": int(len(time_df)), "missing_columns": missing}

    frame_errors: list[str] = []
    hours: set[int] = set()
    for _, row in time_df.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        past = split_times(row["past_times"])
        future = split_times(row["future_times"])
        if pd.isna(issue):
            frame_errors.append("NaT issue_time")
            continue
        hours.add(int(issue.hour))
        if len(past) != 12 or len(future) != 8:
            frame_errors.append(f"{issue}: {len(past)} past / {len(future)} future")
            continue
        if past[-1] != issue - pd.Timedelta(minutes=15):
            frame_errors.append(f"{issue}: history does not end at T-15 min")
        if future[0] != issue:
            frame_errors.append(f"{issue}: future does not start at T")
        expected_past = [issue - pd.Timedelta(minutes=15 * (12 - index)) for index in range(12)]
        expected_future = [issue + pd.Timedelta(minutes=15 * index) for index in range(8)]
        if past != expected_past:
            frame_errors.append(f"{issue}: past timestamps are not the canonical 12-frame sequence")
        if future != expected_future:
            frame_errors.append(f"{issue}: future timestamps are not the canonical 8-frame sequence")

    add_check(
        checks,
        "12-to-8 temporal indexing",
        not frame_errors,
        "Every row uses T-3 h...T-15 min history and T...T+105 min targets (15-120 min leads).",
        "; ".join(frame_errors[:5]) + (f" (+{len(frame_errors) - 5} more)" if len(frame_errors) > 5 else ""),
        critical=True,
    )
    add_check(
        checks,
        "issue hours",
        hours == {10, 11, 12, 13, 14, 15},
        "Issue hours are exactly 10:00-15:00.",
        f"Observed issue hours: {sorted(hours)}",
        critical=True,
    )

    counts = {str(k): int(v) for k, v in time_df["time_split"].astype(str).value_counts().to_dict().items()}
    for split in ("train", "val", "test"):
        expected = int(expected_counts[split])
        observed = int(counts.get(split, 0))
        add_check(
            checks,
            f"{split} issue count",
            observed == expected if expected > 0 else None,
            f"{observed} rows match the expected protocol.",
            f"Observed {observed}, expected {expected}.",
            unavailable_detail=f"Observed {observed}; expected-count check disabled.",
        )
    return {
        "rows": int(len(time_df)),
        "split_counts": counts,
        "issue_hours": sorted(hours),
        "signature_sha1": issue_signature(time_df),
        "first_issue": json_ready(time_df["issue_time"].min()),
        "last_issue": json_ready(time_df["issue_time"].max()),
        "frame_error_count": int(len(frame_errors)),
    }


def build_matched_target_neighbors(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    grid_size: int,
    patch_size: int,
    gaussian_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("patch-size must be a positive odd integer")
    patch_radius = patch_size // 2
    source_records: list[dict[str, Any]] = []
    for source_index, row in sources.reset_index(drop=True).iterrows():
        sy, sx = xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, grid_size)
        support = {
            (sy + dy, sx + dx)
            for dy in range(-gaussian_radius, gaussian_radius + 1)
            for dx in range(-gaussian_radius, gaussian_radius + 1)
            if dy * dy + dx * dx <= gaussian_radius * gaussian_radius
            and 0 <= sy + dy < grid_size
            and 0 <= sx + dx < grid_size
        }
        source_records.append(
            {
                "source_index": int(source_index),
                "source_sn": str(row["station_sn"]),
                "source_lon": float(row["lon"]),
                "source_lat": float(row["lat"]),
                "source_row": int(sy),
                "source_col": int(sx),
                "support": support,
            }
        )

    neighbor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for target_index, row in targets.reset_index(drop=True).iterrows():
        target_sn = str(row["station_sn"])
        target_lon = float(row["lon"])
        target_lat = float(row["lat"])
        ty, tx = xy_to_ij(target_lon, target_lat, bounds, grid_size)
        patch = {
            (ty + dy, tx + dx)
            for dy in range(-patch_radius, patch_radius + 1)
            for dx in range(-patch_radius, patch_radius + 1)
            if 0 <= ty + dy < grid_size and 0 <= tx + dx < grid_size
        }
        local: list[dict[str, Any]] = []
        for source in source_records:
            overlap = patch & source["support"]
            if not overlap:
                continue
            mean_lat = math.radians((target_lat + float(source["source_lat"])) / 2.0)
            dx_km = (float(source["source_lon"]) - target_lon) * 111.32 * math.cos(mean_lat)
            dy_km = (float(source["source_lat"]) - target_lat) * 110.574
            distance = haversine_km(
                target_lon,
                target_lat,
                float(source["source_lon"]),
                float(source["source_lat"]),
            )
            local.append(
                {
                    "target_index": int(target_index),
                    "target_sn": target_sn,
                    "target_lon": target_lon,
                    "target_lat": target_lat,
                    "target_row": int(ty),
                    "target_col": int(tx),
                    "source_index": int(source["source_index"]),
                    "source_sn": str(source["source_sn"]),
                    "source_lon": float(source["source_lon"]),
                    "source_lat": float(source["source_lat"]),
                    "source_row": int(source["source_row"]),
                    "source_col": int(source["source_col"]),
                    "distance_km": float(distance),
                    "delta_x_km": float(dx_km),
                    "delta_y_km": float(dy_km),
                    "overlap_cell_count": int(len(overlap)),
                    "target_patch_valid_cell_count": int(len(patch)),
                }
            )
        local.sort(key=lambda item: (float(item["distance_km"]), str(item["source_sn"])))
        for rank, item in enumerate(local, start=1):
            item["rank_by_distance"] = int(rank)
            neighbor_rows.append(item)
        distances = np.asarray([item["distance_km"] for item in local], dtype=np.float64)
        summary_rows.append(
            {
                "target_index": int(target_index),
                "target_sn": target_sn,
                "target_lon": target_lon,
                "target_lat": target_lat,
                "target_row": int(ty),
                "target_col": int(tx),
                "target_patch_size": int(patch_size),
                "target_patch_valid_cell_count": int(len(patch)),
                "matched_source_count": int(len(local)),
                "min_distance_km": float(np.min(distances)) if distances.size else np.nan,
                "median_distance_km": float(np.median(distances)) if distances.size else np.nan,
                "max_distance_km": float(np.max(distances)) if distances.size else np.nan,
            }
        )
    return pd.DataFrame(neighbor_rows), pd.DataFrame(summary_rows)


def pairwise_haversine(stations: pd.DataFrame) -> np.ndarray:
    lon = np.radians(stations["lon"].to_numpy(np.float64))
    lat = np.radians(stations["lat"].to_numpy(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def build_source_knn_graph(sources: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = int(len(sources))
    if n < 2:
        raise ValueError("At least two source stations are required")
    k = max(1, min(int(k), n - 1))
    distances = pairwise_haversine(sources)
    np.fill_diagonal(distances, np.inf)
    directed: set[tuple[int, int]] = set()
    for index in range(n):
        nearest = np.argsort(distances[index], kind="stable")[:k]
        directed.update((index, int(other)) for other in nearest)
    undirected = sorted({(min(i, j), max(i, j)) for i, j in directed if i != j})
    edge_distances = np.asarray([distances[i, j] for i, j in undirected], dtype=np.float64)
    sigma = float(np.median(edge_distances)) if edge_distances.size else 1.0
    sigma = max(sigma, 1e-6)
    rows = []
    source_work = sources.reset_index(drop=True)
    for i, j in undirected:
        distance = float(distances[i, j])
        rows.append(
            {
                "source_u_index": int(i),
                "source_u_sn": str(source_work.iloc[i]["station_sn"]),
                "source_v_index": int(j),
                "source_v_sn": str(source_work.iloc[j]["station_sn"]),
                "distance_km": distance,
                "weight": float(math.exp(-(distance * distance) / (2.0 * sigma * sigma))),
                "selected_by_u": int((i, j) in directed),
                "selected_by_v": int((j, i) in directed),
            }
        )
    metadata = {
        "node_count": n,
        "requested_k": int(k),
        "undirected_edge_count": int(len(rows)),
        "distance_sigma_km": sigma,
        "weight_formula": "exp(-distance_km^2 / (2 * sigma_km^2))",
        "symmetrization": "union of directed geographic kNN edges",
        "self_loops_in_csv": False,
        "later_model_requirement": "Add self-loops and normalize adjacency inside the graph model.",
    }
    return pd.DataFrame(rows), metadata


def cache_coverage(
    data_dir: Path,
    reference_time: pd.DataFrame,
    checks: list[Check],
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame_dir = data_dir / "04_target_excluded_pvnorm_frame_cache"
    ms_dir = data_dir / "01_multisource_nwp_cache"
    swr_root = data_dir / "05_center_h8_swr_anchor_cache"
    swr_dir, swr_resolution = resolve_swr_cache_dir(swr_root)

    arrays = {
        "pv_fields": inspect_npy(frame_dir / "fields_float16.npy"),
        "pv_masks": inspect_npy(frame_dir / "masks_uint8.npy"),
        "pv_timestamp_sec": inspect_npy(frame_dir / "timestamp_sec.npy"),
        "h8_swr": inspect_npy(ms_dir / "h8_swr_64_float16.npy"),
        "h8_age_minutes": inspect_npy(ms_dir / "h8_age_minutes_float16.npy"),
        "h8_frame_sec": inspect_npy(ms_dir / "h8_frame_sec.npy"),
        "nwp_future_fields": inspect_npy(ms_dir / "nwp_future_fields_float16.npy"),
        "nwp_future_valid": inspect_npy(ms_dir / "nwp_future_valid_uint8.npy"),
        "nwp_nearest_distance_km": inspect_npy(ms_dir / "nwp_nearest_distance_km_float16.npy"),
        "swr_anchor": inspect_npy(swr_dir / "swr_anchor_64.npy"),
    }
    frame_meta = read_json(frame_dir / "frame_cache_meta.json")
    ms_meta = read_json(ms_dir / "multisource_cache_meta.json")
    swr_meta_json = read_json(swr_dir / "center_h8_swr_anchor_input_cache_meta.json")

    field_shape = arrays["pv_fields"].get("shape")
    mask_shape = arrays["pv_masks"].get("shape")
    add_check(
        checks,
        "PV field/mask cache shape",
        bool(field_shape) and field_shape == mask_shape and len(field_shape) == 3,
        f"Field and mask arrays share shape {field_shape}.",
        f"Field shape={field_shape}, mask shape={mask_shape}.",
        critical=True,
    )

    n_ref = int(len(reference_time))
    nwp_shape = arrays["nwp_future_fields"].get("shape")
    expected_nwp_tail = [8, 4, 16, 16]
    add_check(
        checks,
        "future NWP cache contract",
        bool(nwp_shape) and int(nwp_shape[0]) == n_ref and list(nwp_shape[1:]) == expected_nwp_tail,
        f"NWP cache is {nwp_shape}: N x 8 horizons x 4 variables x 16 x 16.",
        f"Observed NWP shape={nwp_shape}; expected [{n_ref}, 8, 4, 16, 16].",
    )
    add_check(
        checks,
        "NWP variable provenance",
        tuple(ms_meta.get("nwp_variables", [])) == NWP_VARIABLES if ms_meta else None,
        f"NWP variables are exactly {list(NWP_VARIABLES)}.",
        f"Metadata reports {ms_meta.get('nwp_variables')}.",
        unavailable_detail="multisource_cache_meta.json is unavailable; variable order cannot be certified.",
    )

    ms_time_path = ms_dir / "issue_time_index.csv"
    if ms_time_path.is_file():
        ms_time = canonical_issue_table(ms_time_path)
        exact = compare_issue_tables(reference_time, ms_time)
        add_check(
            checks,
            "multisource issue-time alignment",
            exact,
            "Multisource issue index exactly matches the canonical selection index.",
            "Multisource issue index differs from the canonical selection index.",
            critical=True,
        )
    else:
        ms_time = pd.DataFrame()
        add_check(
            checks,
            "multisource issue-time alignment",
            None,
            "",
            "",
            unavailable_detail=f"Missing optional cache index: {ms_time_path}",
        )

    swr_meta_path = swr_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    swr_times: set[pd.Timestamp] = set()
    if swr_meta_path.is_file():
        swr_meta = pd.read_csv(swr_meta_path)
        if "issue_time" in swr_meta.columns:
            swr_times = set(pd.to_datetime(swr_meta["issue_time"], errors="coerce").dropna().tolist())
        elif "issue_sec" in swr_meta.columns:
            swr_times = set(pd.to_datetime(swr_meta["issue_sec"], unit="s", errors="coerce").dropna().tolist())
        add_check(
            checks,
            "SWR-anchor metadata",
            bool(swr_times),
            f"SWR cache contains {len(swr_times)} valid issue times.",
            "SWR metadata exists but contains no readable issue timestamps.",
        )
    else:
        add_check(
            checks,
            "SWR-anchor metadata",
            None,
            "",
            "",
            unavailable_detail=f"Missing optional SWR metadata: {swr_meta_path}",
        )

    swr_shape = arrays["swr_anchor"].get("shape")
    add_check(
        checks,
        "SWR-anchor tensor contract",
        bool(swr_shape) and len(swr_shape) == 5 and list(swr_shape[1:]) == [7, 2, 64, 64],
        f"SWR anchor cache is {swr_shape}: 4 observed + 3 predicted anchors, 2 channels.",
        f"Observed SWR-anchor shape={swr_shape}; expected [N_valid, 7, 2, 64, 64].",
    )

    ref = reference_time.copy()
    ref["issue_time_key"] = pd.to_datetime(ref["issue_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if swr_times:
        swr_keys = {pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S") for value in swr_times}
        ref["has_swr_anchor"] = ref["issue_time_key"].isin(swr_keys)
    else:
        ref["has_swr_anchor"] = False
    if not ms_time.empty:
        ms_keys = set(pd.to_datetime(ms_time["issue_time"]).dt.strftime("%Y-%m-%d %H:%M:%S"))
        ref["has_multisource"] = ref["issue_time_key"].isin(ms_keys)
    else:
        ref["has_multisource"] = False
    coverage = (
        ref.groupby("time_split", dropna=False)
        .agg(
            canonical_issues=("issue_time_key", "size"),
            multisource_issues=("has_multisource", "sum"),
            swr_anchor_issues=("has_swr_anchor", "sum"),
        )
        .reset_index()
    )
    coverage["common_issues"] = ref.assign(common=ref["has_swr_anchor"] & ref["has_multisource"]).groupby("time_split")["common"].sum().reindex(coverage["time_split"]).to_numpy()

    return (
        {
            "directories": {
                "frame": str(frame_dir),
                "multisource": str(ms_dir),
                "swr_root": str(swr_root),
                "swr_resolved": str(swr_dir),
            },
            "swr_resolution": swr_resolution,
            "arrays": arrays,
            "frame_metadata": frame_meta,
            "multisource_metadata": ms_meta,
            "swr_metadata": swr_meta_json,
        },
        coverage,
    )


def normalization_audit(
    data_dir: Path,
    model_dir: Path | None,
    sources: pd.DataFrame,
    checks: list[Check],
) -> dict[str, Any]:
    selection_dir = data_dir / "02_target_selection_lhs50"
    q95_path = selection_dir / "target_excluded_station_pv_norm_q95.csv"
    selection_meta = read_json(selection_dir / "01_target_selection_meta.json")
    q95_summary: dict[str, Any] = {"path": str(q95_path), "exists": q95_path.is_file()}
    if q95_path.is_file():
        q95 = pd.read_csv(q95_path, dtype={"station_sn": str})
        if "station_sn" not in q95.columns or "pv_norm_q95" not in q95.columns:
            add_check(checks, "source q95 table", False, "", f"Unexpected q95 columns: {list(q95.columns)}", critical=True)
        else:
            q95["station_sn"] = q95["station_sn"].astype(str)
            q95["pv_norm_q95"] = pd.to_numeric(q95["pv_norm_q95"], errors="coerce")
            source_ids = set(sources["station_sn"].astype(str))
            q95_ids = set(q95["station_sn"].astype(str))
            values = q95.loc[q95["station_sn"].isin(source_ids), "pv_norm_q95"].to_numpy(np.float64)
            good = q95_ids == source_ids and values.size == len(source_ids) and np.all(np.isfinite(values)) and np.all(values > 0)
            add_check(
                checks,
                "source q95 table",
                good,
                f"All {len(source_ids)} sources have one finite positive q95 denominator.",
                f"q95 rows={len(q95)}, missing={len(source_ids - q95_ids)}, extra={len(q95_ids - source_ids)}, invalid={int(np.sum(~np.isfinite(values) | (values <= 0)))}.",
                critical=True,
            )
            q95_summary.update(
                {
                    "row_count": int(len(q95)),
                    "source_coverage_count": int(len(source_ids & q95_ids)),
                    "minimum": float(np.nanmin(values)) if values.size else None,
                    "median": float(np.nanmedian(values)) if values.size else None,
                    "maximum": float(np.nanmax(values)) if values.size else None,
                }
            )
    else:
        add_check(checks, "source q95 table", False, "", f"Missing {q95_path}", critical=True)

    # The current CSV stores only the final denominator; it does not retain a
    # station-wise train/all_fallback flag from estimate_station_pvnorm_peak.
    checks.append(
        Check(
            "q95 train-only provenance",
            "WARN",
            "The current q95 artifact has no per-station train/all_fallback column. Values are usable, but strict train-only provenance cannot be reconstructed from saved files.",
        )
    )

    stats: dict[str, Any] = {}
    if model_dir is not None:
        model_dir = Path(model_dir)
        for name in ("pv_field_norm_stats.json", "multisource_norm_stats.json"):
            stats[name] = read_json(model_dir / name)
        present = all(bool(stats[name]) for name in stats)
        add_check(
            checks,
            "Stage-3 normalization statistics",
            present,
            f"Both normalization files are present in {model_dir}.",
            f"One or more normalization files are missing in {model_dir}.",
        )
    else:
        add_check(
            checks,
            "Stage-3 normalization statistics",
            None,
            "",
            "",
            unavailable_detail="No --stage3-model-dir was supplied; run-specific z-score statistics were not audited.",
        )
    return {"q95": q95_summary, "selection_metadata": selection_meta, "stage3_model_stats": stats}


def markdown_report(report: dict[str, Any]) -> str:
    checks = report["checks"]
    counts = report["station_protocol"]
    temporal = report["temporal_protocol"]
    neighbor = report["matched_support"]
    lines = [
        "# Matched Representation Experiment: Phase-1 Audit",
        "",
        "This is a pre-training audit. It does not train either representation branch and does not use future target PV as an input.",
        "",
        "## Decision",
        "",
        f"**{report['decision']}** — {report['decision_detail']}",
        "",
        "## Leakage-safe station protocol",
        "",
        f"- Full fleet: **{counts['all_count']}** stations",
        f"- Source fleet: **{counts['source_count']}** stations",
        f"- Held-out targets: **{counts['target_count']}** stations",
        f"- Source/target overlap: **{counts['overlap_count']}**",
        f"- Grid bounds: `{counts['bounds']}`",
        "",
        "## Temporal protocol",
        "",
        f"- Canonical issue rows: **{temporal.get('rows')}**",
        f"- Split counts: `{temporal.get('split_counts')}`",
        f"- Issue hours: `{temporal.get('issue_hours')}`",
        f"- Index signature (SHA1): `{temporal.get('signature_sha1')}`",
        "- History/target convention: 12 historical frames ending at T-15 min; 8 targets from T through T+105 min (15-120 min ahead of the latest observation).",
        "",
        "## Strict target–source context matching",
        "",
        f"- Target patch: **{neighbor['patch_size']}×{neighbor['patch_size']}** cells",
        f"- Gaussian support radius: **{neighbor['gaussian_radius_cells']}** cells",
        "- A source station is included only if its actual circular Gaussian support intersects the valid in-grid cells of the target patch.",
        f"- Matched sources per target: min **{neighbor['min_sources']}**, median **{neighbor['median_sources']:.1f}**, max **{neighbor['max_sources']}**",
        "- This is not K=25 nearest-neighbor selection; 25 is the number of patch pixels, not the number of stations.",
        "",
        "## Audit checklist",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check['status']} | {check['name']} | {detail} |")
    lines.extend(
        [
            "",
            "## Generated contracts",
            "",
            "- `target_source_matched_neighbors.csv`: the exact source set available to each target's 5×5 field patch.",
            "- `target_neighbor_summary.csv`: matched-source coverage and distance statistics by target.",
            "- `source_graph_knn_edges.csv`: symmetric-union geographic kNN graph over source stations only.",
            "- `issue_time_coverage.csv`: canonical, multisource, SWR-anchor, and common issue coverage by split.",
            "",
            "## Fairness constraints for Phase 2",
            "",
            "The following items remain deliberately **PENDING** until the training script exists:",
            "",
            "1. field and graph branches must feed the same downstream target-history/future-condition backbone;",
            "2. parameter counts and tuning budgets must be reported;",
            "3. the main historical-context comparison must not consume future source PV;",
            "4. graph-to-target readout must use only the source rows listed in `target_source_matched_neighbors.csv`;",
            "5. all metrics must use the same held-out targets, issue times, horizons, normalization, and aggregation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_data_root = resolve_old_data_root(
        project_root,
        Path(args.old_data_root).expanduser() if args.old_data_root is not None else None,
        require_exists=False,
    )
    data_dir = resolve_stage3_data_dir(
        project_root,
        old_data_root=old_data_root,
        override=Path(args.stage3_data_dir).expanduser() if args.stage3_data_dir is not None else None,
    )
    selection_dir = data_dir / "02_target_selection_lhs50"
    run_dir = data_dir / "03_target_excluded_run"
    frame_dir = data_dir / "04_target_excluded_pvnorm_frame_cache"

    checks: list[Check] = []
    all_stations = load_station_table(run_dir / "station_split.csv", "full station split")
    sources = load_station_table(run_dir / "target_excluded_station_split.csv", "target-excluded source split")
    targets = load_station_table(selection_dir / "target_stations.csv", "held-out targets")

    for name, table in (("full", all_stations), ("source", sources), ("target", targets)):
        duplicates = int(table["station_sn"].duplicated().sum())
        coordinate_ok = bool(np.isfinite(table[["lon", "lat"]].to_numpy(np.float64)).all())
        add_check(
            checks,
            f"{name} station IDs unique",
            duplicates == 0,
            f"All {len(table)} IDs are unique.",
            f"Found {duplicates} duplicate station IDs.",
            critical=True,
        )
        add_check(
            checks,
            f"{name} station coordinates finite",
            coordinate_ok,
            "All longitudes and latitudes are finite.",
            "At least one longitude/latitude is missing or non-finite.",
            critical=True,
        )

    all_ids = set(all_stations["station_sn"].astype(str))
    source_ids = set(sources["station_sn"].astype(str))
    target_ids = set(targets["station_sn"].astype(str))
    overlap = source_ids & target_ids
    union_matches = source_ids | target_ids == all_ids
    add_check(checks, "source-target disjointness", not overlap, "Source and target sets are disjoint.", f"Overlap: {sorted(overlap)[:20]}", critical=True)
    add_check(checks, "source-target partition", union_matches, "Source ∪ target exactly equals the full fleet.", f"Missing from partition={len(all_ids - (source_ids | target_ids))}, extra={len((source_ids | target_ids) - all_ids)}.", critical=True)

    expected_triplets = (
        ("full fleet count", len(all_stations), int(args.expected_all_stations)),
        ("source fleet count", len(sources), int(args.expected_source_stations)),
        ("held-out target count", len(targets), int(args.expected_target_stations)),
    )
    for name, observed, expected in expected_triplets:
        add_check(
            checks,
            name,
            observed == expected if expected > 0 else None,
            f"Observed {observed}, matching the intended protocol.",
            f"Observed {observed}, expected {expected}.",
            unavailable_detail=f"Observed {observed}; expected-count check disabled.",
        )

    frame_meta = read_json(frame_dir / "frame_cache_meta.json")
    actual_sha1 = station_sha1(sources)
    hash_ok = bool(frame_meta) and str(frame_meta.get("source_station_sns_sha1", "")) == actual_sha1
    count_ok = bool(frame_meta) and int(frame_meta.get("source_station_count", -1)) == len(sources)
    add_check(
        checks,
        "field cache source-set hash",
        hash_ok and count_ok,
        "Frame-cache count and SHA1 exactly match the target-excluded source split.",
        f"CSV count/hash={len(sources)}/{actual_sha1}; cache={frame_meta.get('source_station_count')}/{frame_meta.get('source_station_sns_sha1')}.",
        critical=True,
    )

    issue_path = selection_dir / "issue_time_index.csv"
    time_df = canonical_issue_table(issue_path)
    temporal = audit_temporal_protocol(
        time_df,
        checks,
        {
            "train": int(args.expected_train_issues),
            "val": int(args.expected_val_issues),
            "test": int(args.expected_test_issues),
        },
    )
    cache_report, coverage = cache_coverage(data_dir, time_df, checks)
    coverage.to_csv(output_dir / "issue_time_coverage.csv", index=False)

    bounds = bounds_from_all_stations(all_stations)
    neighbors, neighbor_summary = build_matched_target_neighbors(
        sources,
        targets,
        bounds,
        int(args.grid_size),
        int(args.patch_size),
        int(args.gaussian_radius_cells),
    )
    neighbors.to_csv(output_dir / "target_source_matched_neighbors.csv", index=False)
    neighbor_summary.to_csv(output_dir / "target_neighbor_summary.csv", index=False)
    zero_neighbor_targets = neighbor_summary.loc[neighbor_summary["matched_source_count"] <= 0, "target_sn"].astype(str).tolist()
    add_check(
        checks,
        "matched source support for every target",
        not zero_neighbor_targets,
        "Every held-out target has at least one source whose Gaussian support intersects its 5x5 patch.",
        f"Targets with no matched source support: {zero_neighbor_targets}",
        critical=True,
    )

    graph_edges, graph_meta = build_source_knn_graph(sources, int(args.graph_k))
    graph_edges.to_csv(output_dir / "source_graph_knn_edges.csv", index=False)
    add_check(
        checks,
        "source-only graph nodes",
        not bool(set(graph_edges.get("source_u_sn", [])) & target_ids) and not bool(set(graph_edges.get("source_v_sn", [])) & target_ids),
        "The exported graph contains source stations only.",
        "At least one held-out target appears as a graph node.",
        critical=True,
    )

    norm_report = normalization_audit(data_dir, args.stage3_model_dir, sources, checks)
    checks.extend(
        [
            Check("matched downstream backbone", "PENDING", "Phase-2 training code has not yet been created."),
            Check("matched training/tuning budget", "PENDING", "Phase-2 training code has not yet been created."),
            Check("no future source PV in graph branch", "PENDING", "Must be enforced and tested in Phase-2 dataset code."),
            Check("matched evaluation aggregation", "PENDING", "Must be enforced and tested in Phase-2 evaluation code."),
        ]
    )

    failed_critical = [check for check in checks if check.status == "FAIL" and check.critical]
    decision = "STOP" if failed_critical else "READY FOR PHASE-2 IMPLEMENTATION"
    decision_detail = (
        f"{len(failed_critical)} critical preflight checks failed. Do not train until they are fixed."
        if failed_critical
        else "No critical leakage, split, cache-hash, temporal-index, or spatial-support failure was detected. Warnings and pending Phase-2 fairness checks remain."
    )

    matched_counts = neighbor_summary["matched_source_count"].to_numpy(np.int64)
    report = {
        "decision": decision,
        "decision_detail": decision_detail,
        "paths": {
            "project_root": project_root,
            "old_data_root": old_data_root,
            "stage3_data_dir": data_dir,
            "selection_dir": selection_dir,
            "target_run_dir": run_dir,
            "frame_cache_dir": frame_dir,
            "stage3_model_dir": args.stage3_model_dir,
            "output_dir": output_dir,
        },
        "station_protocol": {
            "all_count": int(len(all_stations)),
            "source_count": int(len(sources)),
            "target_count": int(len(targets)),
            "overlap_count": int(len(overlap)),
            "partition_matches": bool(union_matches),
            "source_station_sns_sha1": actual_sha1,
            "bounds": [float(v) for v in bounds],
            "grid_size": int(args.grid_size),
        },
        "temporal_protocol": temporal,
        "cache_audit": cache_report,
        "normalization_audit": norm_report,
        "matched_support": {
            "definition": "source circular Gaussian support intersects valid cells of the target-centered patch",
            "patch_size": int(args.patch_size),
            "gaussian_radius_cells": int(args.gaussian_radius_cells),
            "target_count": int(len(neighbor_summary)),
            "edge_count": int(len(neighbors)),
            "min_sources": int(np.min(matched_counts)) if matched_counts.size else 0,
            "median_sources": float(np.median(matched_counts)) if matched_counts.size else 0.0,
            "max_sources": int(np.max(matched_counts)) if matched_counts.size else 0,
            "zero_source_targets": zero_neighbor_targets,
        },
        "source_graph": graph_meta,
        "issue_time_coverage": coverage.to_dict(orient="records"),
        "checks": [asdict(check) for check in checks],
        "outputs": {
            "audit_json": output_dir / "matched_representation_audit.json",
            "audit_markdown": output_dir / "matched_representation_audit.md",
            "target_neighbors": output_dir / "target_source_matched_neighbors.csv",
            "target_neighbor_summary": output_dir / "target_neighbor_summary.csv",
            "source_graph_edges": output_dir / "source_graph_knn_edges.csv",
            "issue_coverage": output_dir / "issue_time_coverage.csv",
        },
    }
    report = json_ready(report)
    json_path = output_dir / "matched_representation_audit.json"
    md_path = output_dir / "matched_representation_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")

    print("==== Matched representation Phase-1 audit ====")
    print(f"decision: {decision}")
    print(f"stage3 data: {data_dir}")
    print(f"stations: all={len(all_stations)} source={len(sources)} target={len(targets)} overlap={len(overlap)}")
    print(
        "matched sources/target: "
        f"min={report['matched_support']['min_sources']} "
        f"median={report['matched_support']['median_sources']} "
        f"max={report['matched_support']['max_sources']}"
    )
    print(f"report: {md_path}")
    print(f"machine-readable report: {json_path}")
    if failed_critical:
        for check in failed_critical:
            print(f"CRITICAL FAIL: {check.name}: {check.detail}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
