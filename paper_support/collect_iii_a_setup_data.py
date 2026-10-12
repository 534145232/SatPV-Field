#!/usr/bin/env python3
"""Collect evidence needed to write III-A Experimental Setup and Evaluation Protocol.

The script is read-only with respect to experiment artifacts.  It inspects the
pipeline configuration, cached metadata and evaluation CSV files, then writes
one Markdown report.  Missing artifacts are reported explicitly rather than
replaced with assumptions.

Typical server usage:

    python /workspace/Final_3_stage_pipeline/paper_support/collect_iii_a_setup_data.py

The default report is written beside this script as
``iii_a_experimental_setup_data.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - server dependency check
    raise SystemExit(
        "This collector requires pandas, which is already used by the pipeline. "
        f"Import failed: {exc}"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_support_paths import (  # noqa: E402
    resolve_old_data_root,
    resolve_pv_directories,
    resolve_stage1_data_dir,
)


STATION_ID_ALIASES = ["station_sn", "SN", "sn", "station_id", "site_id", "id"]
LAT_ALIASES = ["lat", "latitude", "Latitude", "LAT"]
LON_ALIASES = ["lon", "longitude", "Longitude", "LON"]

SATELLITE_CHANNELS = ["SWR", "PAR", "UVA", "UVB", "TAOT_02", "TAAE"]
REGIONAL_NWP_VARIABLES = ["radiation", "direction10", "speed10", "tcc"]
STATION_NWP_VARIABLES = [
    "radiation",
    "pressure",
    "humidity",
    "temperature",
    "direction10",
    "speed10",
    "tcc",
    "solar_zenith",
    "clearsky_ghi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Final_3_stage_pipeline root. Default: inferred from this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "iii_a_experimental_setup_data.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--density-data-root",
        type=Path,
        default=Path("/data/Final_3_stage_pipeline/stage_3/data/11_source_fleet_density"),
        help="Optional external root used by the source-fleet-density experiment.",
    )
    parser.add_argument(
        "--old-data-root",
        type=Path,
        default=None,
        help="External cache root; defaults to the old_data directory beside the code directory.",
    )
    parser.add_argument(
        "--pv-root",
        type=Path,
        default=None,
        help="PV data root. Default: <old-data-root>/PV (or BIGPV_PV_ROOT).",
    )
    parser.add_argument(
        "--train-pv-dir",
        type=Path,
        default=None,
        help="Station CSV directory used for column/provenance inspection.",
    )
    parser.add_argument(
        "--val-pv-dir",
        type=Path,
        default=None,
        help="Validation/test station CSV directory.",
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=None,
        help=(
            "Optional external Stage-1 root or Stage-1 data directory; "
            "otherwise project-local and old_data locations are checked."
        ),
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return md(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def read_csv_safe(path: Path, **kwargs: Any) -> tuple[pd.DataFrame | None, str | None]:
    path = Path(path)
    if not path.exists():
        return None, "missing"
    if path.stat().st_size <= 2:
        return None, "empty placeholder"
    try:
        return pd.read_csv(path, **kwargs), None
    except Exception as exc:
        return None, f"read error: {type(exc).__name__}: {exc}"


def read_json_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path)
    if not path.exists():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"value": value}, None
    except Exception as exc:
        return None, f"read error: {type(exc).__name__}: {exc}"


def table(headers: list[str], rows: list[list[Any]], aligns: list[str] | None = None) -> list[str]:
    if aligns is None:
        aligns = ["---"] * len(headers)
    out = ["| " + " | ".join(md(x) for x in headers) + " |"]
    out.append("|" + "|".join(aligns) + "|")
    for row in rows:
        out.append("| " + " | ".join(md(x) for x in row) + " |")
    return out


def extract_python_string_constants(path: Path, names: Iterable[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for name in names:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text, flags=re.M)
        if match:
            out[name] = match.group(1)
    return out


def parse_timestamp_list(value: Any) -> list[pd.Timestamp]:
    return [pd.Timestamp(item) for item in str(value).split("|") if str(item).strip()]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return radius * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def station_summary(path: Path, root: Path, label: str) -> tuple[dict[str, Any] | None, str]:
    df, error = read_csv_safe(path, dtype=str)
    if df is None:
        return None, f"{label}: {error} ({rel(path, root)})"
    id_col = first_column(df, STATION_ID_ALIASES)
    lat_col = first_column(df, LAT_ALIASES)
    lon_col = first_column(df, LON_ALIASES)
    if id_col is None:
        return None, f"{label}: station-id column not found ({rel(path, root)})"
    result: dict[str, Any] = {
        "label": label,
        "path": rel(path, root),
        "count": int(df[id_col].astype(str).nunique()),
        "id_col": id_col,
    }
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        valid = lat.notna() & lon.notna()
        if valid.any():
            lat_min, lat_max = float(lat[valid].min()), float(lat[valid].max())
            lon_min, lon_max = float(lon[valid].min()), float(lon[valid].max())
            mid_lat = (lat_min + lat_max) / 2.0
            ns_km = haversine_km(lat_min, (lon_min + lon_max) / 2.0, lat_max, (lon_min + lon_max) / 2.0)
            ew_km = haversine_km(mid_lat, lon_min, mid_lat, lon_max)
            diagonal_km = haversine_km(lat_min, lon_min, lat_max, lon_max)
            result.update(
                {
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "ns_km": ns_km,
                    "ew_km": ew_km,
                    "diagonal_km": diagonal_km,
                }
            )
    return result, ""


def locate_station_files(root: Path) -> dict[str, Path]:
    stage3 = root / "stage_3"
    candidates = {
        "all": [
            stage3 / "data/00_source_regional_split/station_split.csv",
            stage3 / "data/03_target_excluded_run/station_split.csv",
        ],
        "target": [stage3 / "data/02_target_selection_lhs50/target_stations.csv"],
        "source": [
            stage3 / "data/03_target_excluded_run/target_excluded_station_split.csv",
            stage3 / "data/02_target_selection_lhs50/target_excluded_station_split.csv",
        ],
    }
    result: dict[str, Path] = {}
    for key, paths in candidates.items():
        result[key] = next((path for path in paths if path.exists()), paths[0])
    return result


def collect_time_protocol(root: Path) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    warnings: list[str] = []
    stage3 = root / "stage_3"
    constants = extract_python_string_constants(
        stage3 / "stage3_paths.py", ["TIME_START", "TIME_END", "TRAIN_END", "VAL_END"]
    )
    index_candidates = [
        stage3 / "data/02_target_selection_lhs50/issue_time_index.csv",
        stage3 / "data/01_multisource_nwp_cache/issue_time_index.csv",
    ]
    index_path = next((path for path in index_candidates if path.exists()), index_candidates[0])
    df, error = read_csv_safe(index_path)

    lines.append("## 1. Data period, split and forecasting task")
    lines.append("")
    lines.append("### Configured data boundaries")
    lines.append("")
    configured_duration = "not derived"
    try:
        start_ts = pd.Timestamp(constants["TIME_START"])
        end_ts = pd.Timestamp(constants["TIME_END"])
        elapsed_days = (end_ts.normalize() - start_ts.normalize()).days + 1
        configured_duration = f"{elapsed_days} calendar days inclusive ({start_ts.date()} to {end_ts.date()})"
    except Exception:
        pass
    configured_rows = [
        ["Data start", constants.get("TIME_START", "not parsed")],
        ["Data end", constants.get("TIME_END", "not parsed")],
        ["Calendar coverage", configured_duration],
        ["Training cutoff", constants.get("TRAIN_END", "not parsed")],
        ["Validation cutoff", constants.get("VAL_END", "not parsed")],
    ]
    lines.extend(table(["Item", "Configured value"], configured_rows))
    lines.append("")
    lines.append(f"Evidence: `{rel(stage3 / 'stage3_paths.py', root)}`")
    lines.append("")

    if df is None:
        warnings.append(f"Issue-time index is {error}: {rel(index_path, root)}")
        lines.append(f"Issue-time index: **{error}** at `{rel(index_path, root)}`.")
        lines.append("")
        lines.append("Configured protocol from source code: 15-min sampling, 12 history steps, 8 future steps, hourly forecast origins.")
        lines.append("")
        return lines, warnings

    if "issue_time" not in df.columns:
        warnings.append(f"issue_time column missing in {rel(index_path, root)}")
        lines.append(f"The index exists but has no `issue_time` column: `{rel(index_path, root)}`.")
        lines.append("")
        return lines, warnings

    df = df.copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"], errors="coerce")
    df = df[df["issue_time"].notna()].copy()
    split_col = "time_split" if "time_split" in df.columns else "split" if "split" in df.columns else None
    split_rows: list[list[Any]] = []
    if split_col:
        for split, group in df.groupby(split_col, sort=False):
            split_rows.append(
                [
                    split,
                    len(group),
                    group["issue_time"].min(),
                    group["issue_time"].max(),
                    int(group["issue_time"].dt.date.nunique()),
                ]
            )
    else:
        split_rows.append(
            ["all", len(df), df["issue_time"].min(), df["issue_time"].max(), int(df["issue_time"].dt.date.nunique())]
        )
    lines.append("### Actual forecast-origin index")
    lines.append("")
    lines.extend(
        table(
            ["Split", "Forecast origins", "First issue", "Last issue", "Distinct days"],
            split_rows,
            ["---", "---:", "---", "---", "---:"],
        )
    )
    lines.append("")
    lines.append(f"Evidence: `{rel(index_path, root)}`")
    lines.append("")

    issue_hours = sorted(int(value) for value in df["issue_time"].dt.hour.unique())
    protocol_rows: list[list[Any]] = [["Daily issue hours (BJT)", ", ".join(f"{hour:02d}:00" for hour in issue_hours)]]
    if "past_times" in df.columns and "future_times" in df.columns and not df.empty:
        sample = df.iloc[0]
        past = parse_timestamp_list(sample["past_times"])
        future = parse_timestamp_list(sample["future_times"])
        if past and future:
            sampling = int(round((future[1] - future[0]).total_seconds() / 60.0)) if len(future) > 1 else None
            ahead = [int(round((time - past[-1]).total_seconds() / 60.0)) for time in future]
            protocol_rows.extend(
                [
                    ["Historical PV sequence", f"{len(past)} steps: {past[0]} to {past[-1]}"],
                    ["Future PV targets", f"{len(future)} steps: {future[0]} to {future[-1]}"],
                    ["Sampling interval", f"{sampling} min" if sampling is not None else "not derived"],
                    ["Lead times from latest observation", ", ".join(f"{value} min" for value in ahead)],
                    ["Maximum lead time", f"{max(ahead)} min ({max(ahead) / 60.0:.2f} h)"],
                ]
            )
    lines.append("### Forecast protocol derived from timestamps")
    lines.append("")
    lines.extend(table(["Item", "Observed value"], protocol_rows))
    lines.append("")
    return lines, warnings


def collect_geography_and_split(root: Path, density_root: Path) -> tuple[list[str], list[str]]:
    lines = ["## 2. Geographic scope and source-target split", ""]
    warnings: list[str] = []
    files = locate_station_files(root)
    summaries: dict[str, dict[str, Any]] = {}
    for key, label in [("all", "All stations"), ("source", "Source stations"), ("target", "Held-out targets")]:
        summary, message = station_summary(files[key], root, label)
        if summary is None:
            warnings.append(message)
        else:
            summaries[key] = summary

    rows: list[list[Any]] = []
    for key in ["all", "source", "target"]:
        item = summaries.get(key)
        if item:
            rows.append([item["label"], item["count"], item["path"]])
        else:
            rows.append([{"all": "All stations", "source": "Source stations", "target": "Held-out targets"}[key], "not verified", rel(files[key], root)])
    lines.extend(table(["Role", "Unique stations", "Evidence"], rows, ["---", "---:", "---"]))
    lines.append("")

    geo = summaries.get("all") or summaries.get("source") or summaries.get("target")
    if geo and "lat_min" in geo:
        geo_rows = [
            ["Latitude range", f"{geo['lat_min']:.6f} to {geo['lat_max']:.6f} deg N"],
            ["Longitude range", f"{geo['lon_min']:.6f} to {geo['lon_max']:.6f} deg E"],
            ["North-south span", f"{geo['ns_km']:.2f} km"],
            ["East-west span", f"{geo['ew_km']:.2f} km"],
            ["Bounding-box diagonal", f"{geo['diagonal_km']:.2f} km"],
        ]
        lines.append("### Spatial extent derived from station coordinates")
        lines.append("")
        lines.extend(table(["Item", "Value"], geo_rows))
        lines.append("")
        lines.append(
            "The province/city name is not inferred from coordinates. The manuscript may state Ningbo, Zhejiang only if confirmed by dataset provenance."
        )
        lines.append("")
    else:
        lines.append("Spatial extent could not be calculated because station coordinate files were unavailable or lacked latitude/longitude columns.")
        lines.append("")

    density_candidates = [
        density_root / "seed2026/density_observability_summary.csv",
        root / "stage_3/runs/source_fleet_density/seed2026/density_observability_summary.csv",
    ]
    density_path = next((path for path in density_candidates if path.exists()), density_candidates[0])
    density_df, density_error = read_csv_safe(density_path)
    lines.append("### Source-fleet density settings")
    lines.append("")
    if density_df is not None:
        keep = [column for column in ["ratio", "ratio_tag", "n_sources", "source_count", "patch_coverage", "median_nearest_source_km"] if column in density_df.columns]
        show = density_df[keep].copy() if keep else density_df.head(20)
        lines.extend(table(list(show.columns), [[row[column] for column in show.columns] for _, row in show.iterrows()]))
        lines.append("")
        lines.append(f"Evidence: `{density_path}`")
    else:
        source_count = summaries.get("source", {}).get("count", "not verified")
        lines.append(f"Main experiment source count: **{source_count}**. Density-result metadata: **{density_error}** at `{density_path}`.")
        lines.append("")
        lines.append("Configured density design in the repository: 25%, 50%, 75%, and 100% of the fixed source fleet; the 50 held-out targets remain unchanged.")
        warnings.append(f"Density metadata is {density_error}: {density_path}")
    lines.append("")
    return lines, warnings


def inspect_netcdf(root: Path, stage1_data_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    monthly = stage1_data_dir / "02_center_monthly_netcdf"
    files = sorted(monthly.glob("*.nc")) if monthly.exists() else []
    if not files:
        return None, f"no NetCDF found under {monthly}"
    try:
        import numpy as np
        import xarray as xr
    except Exception as exc:
        return None, f"xarray/numpy unavailable: {exc}"
    path = files[0]
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            lat_name = next((name for name in ["lat", "latitude", "lat_values"] if name in ds.coords or name in ds.variables), None)
            lon_name = next((name for name in ["lon", "longitude", "lon_values"] if name in ds.coords or name in ds.variables), None)
            result: dict[str, Any] = {
                "path": rel(path, root),
                "variables": sorted(str(name) for name in ds.data_vars),
                "dimensions": {str(key): int(value) for key, value in ds.sizes.items()},
            }
            attrs = {}
            for key in ["title", "source", "platform", "product", "institution", "summary"]:
                if key in ds.attrs:
                    attrs[key] = str(ds.attrs[key])
            result["attrs"] = attrs
            if lat_name and lon_name:
                lat = np.asarray(ds[lat_name].values, dtype=float).reshape(-1)
                lon = np.asarray(ds[lon_name].values, dtype=float).reshape(-1)
                lat = lat[np.isfinite(lat)]
                lon = lon[np.isfinite(lon)]
                if lat.size and lon.size:
                    result.update(
                        {
                            "lat_min": float(lat.min()),
                            "lat_max": float(lat.max()),
                            "lon_min": float(lon.min()),
                            "lon_max": float(lon.max()),
                            "lat_step": float(np.median(np.abs(np.diff(np.unique(lat))))) if np.unique(lat).size > 1 else None,
                            "lon_step": float(np.median(np.abs(np.diff(np.unique(lon))))) if np.unique(lon).size > 1 else None,
                        }
                    )
            return result, None
    except Exception as exc:
        return None, f"NetCDF inspection failed for {path}: {type(exc).__name__}: {exc}"


def inspect_station_columns(paths: list[Path]) -> tuple[Path | None, list[str], str | None]:
    for directory in paths:
        if not directory.exists():
            continue
        sample = next(iter(sorted(directory.glob("*.csv"))), None)
        if sample is None:
            continue
        try:
            with sample.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                columns = next(reader)
            return sample, columns, None
        except Exception as exc:
            return sample, [], f"failed to inspect {sample}: {exc}"
    return None, [], "no station CSV found"


def collect_inputs(
    root: Path,
    train_pv_dir: Path,
    val_pv_dir: Path,
    stage1_data_dir: Path,
) -> tuple[list[str], list[str]]:
    lines = ["## 3. Satellite and meteorological inputs", ""]
    warnings: list[str] = []
    lines.append("### Satellite / LCIG")
    lines.append("")
    satellite_rows = [
        ["Configured channels", ", ".join(SATELLITE_CHANNELS)],
        ["Prediction target", "SWR / clear-sky SWR ratio, converted back to physical SWR"],
        ["Historical input", "4 hourly frames: T-4 h, T-3 h, T-2 h, T-1 h"],
        ["Future SWR anchors", "3 hourly predictions: T, T+1 h, T+2 h"],
        ["LCIG architecture", "3D-CNN base forecaster + residual ConvLSTM correction"],
        ["Training augmentation", "25 overlapping large-region patches; validation/test use the center region"],
    ]
    lines.extend(table(["Item", "Configured value"], satellite_rows))
    lines.append("")
    lines.append("Evidence: `stage_1/README.md`, `stage_2/README.md`, `stage_2/train_3dcnn_ratio_swr.py`, and `stage_3/README.md`.")
    lines.append("")

    netcdf, netcdf_error = inspect_netcdf(root, stage1_data_dir)
    if netcdf:
        lines.append("#### Spatial resolution observed in a processed H8 NetCDF")
        lines.append("")
        rows = [
            ["File", netcdf.get("path")],
            ["Dimensions", json.dumps(netcdf.get("dimensions", {}), ensure_ascii=False)],
            ["Latitude extent", f"{netcdf.get('lat_min')} to {netcdf.get('lat_max')}"],
            ["Longitude extent", f"{netcdf.get('lon_min')} to {netcdf.get('lon_max')}"],
            ["Median latitude spacing", netcdf.get("lat_step")],
            ["Median longitude spacing", netcdf.get("lon_step")],
            ["Dataset attributes", json.dumps(netcdf.get("attrs", {}), ensure_ascii=False)],
        ]
        lines.extend(table(["Item", "Observed value"], rows))
        lines.append("")
    else:
        lines.append(f"Processed H8 spatial resolution could not be inspected: **{netcdf_error}**.")
        lines.append("")
        warnings.append(str(netcdf_error))

    meta_path = root / "stage_3/data/01_multisource_nwp_cache/multisource_cache_meta.json"
    meta, meta_error = read_json_safe(meta_path)
    lines.append("### Regional gridded NWP used by SAG-PVFF")
    lines.append("")
    if meta:
        rows = [
            ["Variables", ", ".join(str(x) for x in meta.get("nwp_variables", []))],
            ["NWP grid", f"{meta.get('nwp_grid_size', 'unknown')} x {meta.get('nwp_grid_size', 'unknown')}"],
            ["Interpolation", meta.get("nwp_interpolation", "unknown")],
            ["Source/version tag", meta.get("nwp_source_version", "unknown")],
            ["H8 alignment", meta.get("h8_alignment", "unknown")],
        ]
        lines.extend(table(["Item", "Observed metadata"], rows))
        lines.append("")
        lines.append(f"Evidence: `{rel(meta_path, root)}`")
    else:
        lines.extend(
            table(
                ["Item", "Configured value"],
                [
                    ["Variables", ", ".join(REGIONAL_NWP_VARIABLES)],
                    ["Native interpolation grid", "16 x 16"],
                    ["Spatial interpolation", "IDW over 12 representative NWP points, then resized to 64 x 64"],
                ],
            )
        )
        lines.append("")
        lines.append(f"Runtime metadata is **{meta_error}**: `{rel(meta_path, root)}`.")
        warnings.append(f"Multisource cache metadata is {meta_error}: {rel(meta_path, root)}")
    lines.append("")

    sample_path, columns, sample_error = inspect_station_columns([train_pv_dir, val_pv_dir])
    lines.append("### Station-level meteorological inputs used by local baselines and FG-MSF")
    lines.append("")
    lower_columns = {column.lower() for column in columns}
    variable_rows = []
    for name in STATION_NWP_VARIABLES:
        aliases = {
            name,
            f"fore_{name}",
            f"nwp_{name}",
        }
        present = any(alias.lower() in lower_columns for alias in aliases)
        variable_rows.append([name, "present in sampled CSV" if present else "not confirmed from sampled header"])
    lines.extend(table(["Required code-level variable", "Runtime column check"], variable_rows))
    lines.append("")
    if sample_path:
        lines.append(f"Sampled header: `{sample_path}`")
    else:
        lines.append(f"Station CSV header check: **{sample_error}**.")
        warnings.append(str(sample_error))
    lines.append("")
    lines.append(
        "Provenance boundary: the code verifies variable names but generally cannot prove the provider identity. Confirm in the paper that historical station weather is ERA5/reanalysis and that future weather is the commercial forecast issued at 06:00 BJT on the preceding day."
    )
    lines.append("")
    return lines, warnings


def metric_columns(df: pd.DataFrame) -> list[str]:
    identity = [
        "model",
        "method",
        "setting",
        "motion_branch",
        "eval_split",
        "split",
        "horizon",
        "n_points",
        "n_frames",
    ]
    metric_pattern = re.compile(r"(mae|rmse|r2|ssim|corr|sign|bias)", flags=re.I)
    selected: list[str] = []
    for column in df.columns:
        if column in identity or metric_pattern.search(str(column)):
            selected.append(str(column))
    return selected


def filter_test_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["eval_split", "split"]:
        if column in out.columns and (out[column].astype(str).str.lower() == "test").any():
            out = out[out[column].astype(str).str.lower() == "test"].copy()
            break
    if "horizon" in out.columns:
        values = out["horizon"].astype(str).str.lower()
        if (values == "all").any():
            out = out[values == "all"].copy()
    return out


def collect_field_metrics(root: Path) -> tuple[list[str], list[str]]:
    lines = ["## 4. Field-level evaluation availability", ""]
    warnings: list[str] = []
    stage3 = root / "stage_3"
    candidates = [
        ("Regional field reconstruction", stage3 / "runs/regional_dense_reconstruction/regional_reconstruction_eval_summary.csv"),
        ("Field forecast baselines", stage3 / "runs/base_pvnorm_field_model/pvnorm_field_eval_summary.csv"),
        (
            "SAG-PVFF (full model used downstream)",
            stage3 / "runs/ablation_motion_branch_last_only/warp_direct_gate/swr_anchor_guided_pvnorm_field_eval_summary.csv",
        ),
        ("Frame-wise DeltaCorr", stage3 / "runs/framewise_delta_corr/stage3_framewise_delta_corr_summary.csv"),
        ("Source-density field evaluation", stage3 / "runs/source_fleet_density/stage3_density_summary.csv"),
    ]

    discovered: list[Path] = []
    runs_dir = stage3 / "runs"
    if runs_dir.exists():
        patterns = [
            "**/regional_reconstruction_eval_summary.csv",
            "**/pvnorm_field_eval_summary.csv",
            "**/swr_anchor_guided_pvnorm_field_eval_summary.csv",
            "**/stage3_framewise_delta_corr_summary.csv",
            "**/stage3_density_summary.csv",
        ]
        for pattern in patterns:
            discovered.extend(runs_dir.glob(pattern))
    seen = {path.resolve() for _, path in candidates if path.exists()}
    for path in sorted(set(discovered)):
        if path.resolve() not in seen:
            candidates.append(("Additional field metric file", path))
            seen.add(path.resolve())

    availability_rows: list[list[Any]] = []
    any_runtime_metrics = False
    for label, path in candidates:
        df, error = read_csv_safe(path)
        if df is None:
            availability_rows.append([label, error, rel(path, root), ""])
            continue
        subset = filter_test_all(df)
        cols = metric_columns(subset)
        numeric_found = any(
            pd.to_numeric(subset[column], errors="coerce").notna().any()
            for column in cols
            if re.search(r"(mae|rmse|r2|ssim|corr|sign|bias)", column, flags=re.I)
        )
        any_runtime_metrics = any_runtime_metrics or numeric_found
        availability_rows.append([label, "available" if numeric_found else "file present; no numeric metric found", rel(path, root), len(subset)])
        if not subset.empty and cols:
            lines.append(f"### {label}")
            lines.append("")
            show = subset[cols].head(30)
            rows = []
            for _, row in show.iterrows():
                values = []
                for column in cols:
                    value = row[column]
                    values.append(fmt_num(value) if re.search(r"(mae|rmse|r2|ssim|corr|sign|bias)", column, flags=re.I) else value)
                rows.append(values)
            lines.extend(table(cols, rows))
            lines.append("")
            lines.append(f"Evidence: `{rel(path, root)}`")
            lines.append("")

    lines.insert(2, "### Artifact availability")
    lines.insert(3, "")
    availability_table = table(
        ["Evaluation", "Status", "Evidence", "Selected rows"],
        availability_rows,
        ["---", "---", "---", "---:"],
    )
    lines[4:4] = availability_table + [""]

    if not any_runtime_metrics:
        warnings.append("No non-empty runtime field metric CSV was found. Field MAE/RMSE/SSIM/DeltaCorr cannot yet be independently verified.")
    lines.append("Metrics implemented by the current evaluation code: MAE, RMSE, R2, SSIM, frame-wise DeltaCorr, SignAcc, and large-change MAE over supported cells.")
    lines.append("")
    lines.append(
        "Interpretation: `regional_reconstruction_eval_summary.csv` addresses static field reconstruction fidelity; the baseline and SAG-PVFF summaries address future field forecasting. Keep these two tasks separate in the manuscript."
    )
    lines.append("")
    return lines, warnings


def artifact_status(detail_path: Path, checkpoint_patterns: list[str]) -> tuple[str, str, str]:
    df, error = read_csv_safe(detail_path, dtype={"station_sn": str})
    if df is not None:
        test = df
        if "eval_split" in test.columns:
            test = test[test["eval_split"].astype(str).str.lower() == "test"]
        stations = test["station_sn"].astype(str).nunique() if "station_sn" in test.columns else "unknown"
        return "evaluated", str(len(test)), str(stations)
    run_dir = detail_path.parent
    checkpoints: list[Path] = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(run_dir.glob(pattern))
    if checkpoints:
        return "checkpoint present; evaluation detail missing", "", ""
    return error or "missing", "", ""


def collect_baselines(root: Path) -> tuple[list[str], list[str]]:
    lines = ["## 5. Baseline and comparison-method status", ""]
    warnings: list[str] = []
    runs = root / "stage_4/runs"
    field_summary = root / "stage_3/runs/base_pvnorm_field_model/pvnorm_field_eval_summary.csv"
    field_df, field_error = read_csv_safe(field_summary)
    persistence_status = "missing"
    if field_df is not None and "model" in field_df.columns:
        mask = field_df["model"].astype(str).str.lower().str.contains("persist")
        persistence_status = "evaluated" if mask.any() else "summary present; persistence row not found"
    elif field_error:
        persistence_status = field_error

    specs = [
        ("Persistence", "Field", field_summary, persistence_status),
        ("LSTM+NWP", "Station", runs / "19_lstm_nwp_baseline_seed2026/lstm_nwp_eval_detail.csv", None),
        ("BiLSTM+NWP", "Station", runs / "03_per_station_temporal_baseline_seed2026/per_station_temporal_baseline_eval_detail.csv", None),
        ("TCN+NWP", "Station", runs / "19_tcn_nwp_baseline_seed2026/tcn_nwp_eval_detail.csv", None),
        ("Transformer+NWP", "Station", runs / "19_transformer_nwp_baseline_seed2026/transformer_nwp_eval_detail.csv", None),
        ("PatchTST+NWP", "Station", runs / "19_patchtst_nwp_baseline_seed2026/patchtst_nwp_eval_detail.csv", None),
        ("STGCN+NWP", "Station", runs / "20_stgcn_forecast_weather_baseline_seed2026/stgcn_forecast_weather_eval_detail.csv", None),
        ("AGCRN+NWP", "Station", runs / "23_agcrn_nwp_baseline_seed2026/agcrn_nwp_eval_detail.csv", None),
        ("3-NN PV guidance", "Station", runs / "21_nearest3_station_pv_baseline_seed2026/nearest3_station_pv_eval_detail.csv", None),
        ("Future SWR guidance", "Station", runs / "06l_per_station_temporal_stage3_swr_patch_seed2026/per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv", None),
        ("Historical PV-field guidance", "Station", runs / "05h_per_station_temporal_historical_field_patch_seed2026/per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv", None),
        ("FG-MSF", "Station", runs / "05l_per_station_temporal_stage3_field_patch_seed2026/per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv", None),
    ]
    rows: list[list[Any]] = []
    for method, level, path, fixed_status in specs:
        if fixed_status is not None:
            status, n_rows, stations = fixed_status, "", ""
        else:
            status, n_rows, stations = artifact_status(path, ["best.pt", "*.pt", "per_station/*.pt", "per_station/**/*.pt"])
        rows.append([method, level, status, n_rows, stations, rel(path, root)])
    lines.extend(
        table(
            ["Method", "Level", "Status", "Test rows", "Test stations", "Evidence expected"],
            rows,
            ["---", "---", "---", "---:", "---:", "---"],
        )
    )
    lines.append("")
    missing_methods = [row[0] for row in rows if row[2] not in {"evaluated", "checkpoint present; evaluation detail missing"}]
    if missing_methods:
        warnings.append("Methods without verified runtime artifacts: " + ", ".join(missing_methods))
    lines.append(
        "A method is marked **evaluated** only when a non-empty evaluation-detail CSV (or, for persistence, an explicit summary row) is present. Source code or a launcher alone is not counted as a completed experiment."
    )
    lines.append("")
    return lines, warnings


def collect_protocol_checks(root: Path) -> tuple[list[str], list[str]]:
    lines = ["## 6. Leakage-control and fairness evidence", ""]
    warnings: list[str] = []
    checks = [
        (
            "Target stations excluded from PV-field construction",
            root / "stage_3/data/07_target_exclusion_audit/target_exclusion_audit.json",
        ),
        (
            "Stage-3 training-input audit",
            root / "stage_3/data/09_training_input_audit/stage3_training_input_audit.json",
        ),
        (
            "Active SWR-anchor cache manifest",
            root / "stage_3/data/05_center_h8_swr_anchor_cache/active_cache.json",
        ),
        (
            "Multisource timing metadata",
            root / "stage_3/data/01_multisource_nwp_cache/multisource_cache_meta.json",
        ),
    ]
    rows = []
    for label, path in checks:
        info, error = read_json_safe(path)
        if info:
            status = "available"
            detail = json.dumps(info, ensure_ascii=False)
            if len(detail) > 500:
                detail = detail[:497] + "..."
        else:
            status = error or "missing"
            detail = ""
            warnings.append(f"{label}: {status} ({rel(path, root)})")
        rows.append([label, status, rel(path, root), detail])
    lines.extend(table(["Check", "Status", "Evidence", "Metadata excerpt"], rows))
    lines.append("")
    lines.append("Protocol statements supported by the current source code:")
    lines.append("")
    lines.append("- Target-site PV measurements are excluded from regional field construction and field-level supervision.")
    lines.append("- Target-site historical PV remains available to its station-level forecaster.")
    lines.append("- PV history ends at T-15 min; the first target is T.")
    lines.append("- Observed hourly H8/SWR labels end at T-1 h; predicted SWR anchors start at T.")
    lines.append("- Normalization/statistics are estimated from the training split and then fixed.")
    lines.append("- FG-MSF consumes fields predicted by a fixed SAG-PVFF model, not ground-truth future fields.")
    lines.append("")
    return lines, warnings


def final_answers(root: Path) -> list[str]:
    return [
        "## 7. Direct answers for the III-A writer",
        "",
        "Use the sections above as the evidence source. The intended concise answers are:",
        "",
        "1. **Data time:** report the configured raw coverage and the actual forecast-origin ranges/counts from the issue-time index; do not substitute a percentage split.",
        "2. **Geography:** report Ningbo, Zhejiang only with provenance confirmation, followed by the coordinate-derived latitude/longitude ranges and kilometre spans.",
        "3. **Forecast horizon:** eight 15-min steps, corresponding to 15–120 min ahead of the latest available PV observation.",
        "4. **Source-target count:** the main experiment should be 446 source stations and 50 held-out target stations if the runtime station CSVs verify those counts; density experiments change only the source fleet.",
        "5. **Field metrics:** distinguish static field reconstruction from future field forecasting; report only metrics backed by non-empty runtime CSVs.",
        "6. **Satellite/NWP:** state H8 product level and native resolution only when verified by metadata. Distinguish the four-variable regional NWP grid from the richer station-level weather vector.",
        "7. **Baselines:** call a baseline completed only when its evaluation artifact is present. Source files alone show implementation, not a finished run.",
        "",
        "Recommended wording boundary: use **held-out target sites**, not *new stations* or *unseen sites*.",
        "",
        f"Project inspected: `{root}`",
        "",
    ]


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if not root.exists():
        raise SystemExit(f"Project root does not exist: {root}")
    old_data_root = resolve_old_data_root(root, args.old_data_root)
    args.train_pv_dir, args.val_pv_dir = resolve_pv_directories(
        args.pv_root,
        args.train_pv_dir,
        args.val_pv_dir,
        project_root=root,
        old_data_root=old_data_root,
        require_exists=False,
    )
    stage1_data_dir = resolve_stage1_data_dir(
        args.stage1_root,
        project_root=root,
        old_data_root=old_data_root,
        require_exists=False,
    )

    sections: list[str] = [
        "# III-A Experimental Setup and Evaluation Protocol: evidence report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Project root: `{root}`",
        "",
        "> This is an evidence-collection report, not manuscript prose. Values marked missing or configured-only should not be presented as completed experimental facts.",
        "",
    ]
    all_warnings: list[str] = []

    collectors = [
        lambda: collect_time_protocol(root),
        lambda: collect_geography_and_split(root, args.density_data_root),
        lambda: collect_inputs(root, args.train_pv_dir, args.val_pv_dir, stage1_data_dir),
        lambda: collect_field_metrics(root),
        lambda: collect_baselines(root),
        lambda: collect_protocol_checks(root),
    ]
    for collector in collectors:
        block, warnings = collector()
        sections.extend(block)
        all_warnings.extend(warnings)

    sections.extend(final_answers(root))
    sections.extend(["## Warnings and missing evidence", ""])
    unique_warnings = list(dict.fromkeys(str(item) for item in all_warnings if str(item).strip()))
    if unique_warnings:
        sections.extend(f"- {warning}" for warning in unique_warnings)
    else:
        sections.append("- No missing evidence detected by this collector.")
    sections.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"[write] {output}")
    print(f"[warnings] {len(unique_warnings)}")


if __name__ == "__main__":
    main()
