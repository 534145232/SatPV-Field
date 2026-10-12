#!/usr/bin/env python3
"""Collect all Stage 4 station-level forecasting results into unified tables.

Outputs include:
  - overall method table
  - horizon-wise method table
  - station-wise method table
  - improvement against the temporal baseline
  - station win-rate against the temporal baseline

The script reads eval-detail CSVs, aligns them by station/split/issue/horizon,
and recomputes metrics from raw predictions so all methods use the same scope.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras  # noqa: E402
from stage4_paths import RUNS_DIR  # noqa: E402

KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
PATCH_DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"

DEFAULT_PAPER_KEYS = ",".join(
    [
        "lstm_nwp",
        "temporal_baseline",
        "tcn_nwp",
        "transformer_nwp",
        "patchtst_nwp",
        "stgcn_forecast_weather",
        "agcrn_nwp",
        "graph_wavenet_nwp",
        "nearest3_station_pv",
        "future_swr_patch",
        "historical_pv_field_patch",
        "fg_msf",
        "patch_1x1",
        "patch_3x3",
        "patch_5x5",
        "patch_7x7",
    ]
)

PAPER_LABELS = {
    "lstm_nwp": ("Station temporal", "LSTM + NWP"),
    "temporal_baseline": ("Station temporal", "BiLSTM + NWP"),
    "tcn_nwp": ("Station temporal", "TCN + NWP"),
    "transformer_nwp": ("Station temporal", "Transformer + NWP"),
    "patchtst_nwp": ("Station temporal", "PatchTST + NWP"),
    "stgcn_forecast_weather": ("Implicit spatio-temporal", "STGCN + NWP"),
    "agcrn_nwp": ("Implicit spatio-temporal", "AGCRN + NWP"),
    "graph_wavenet_nwp": ("Implicit spatio-temporal", "Graph WaveNet + NWP"),
    "agcrn_fgmsf_patch": ("Graph + field", "AGCRN + FG-MSF field"),
    "nearest3_station_pv": ("Neighbor PV", "Nearest-station PV guidance"),
    "future_swr_patch": ("Environmental field", "Future SWR-field guidance"),
    "historical_pv_field_patch": ("Generation field", "Historical PV-field guidance"),
    "fg_msf": ("Generation field", "Proposed FG-MSF"),
}


def _load_extended_table_module():
    path = SCRIPT_DIR / "17_eval_stage4_extended_baseline_table.py"
    spec = importlib.util.spec_from_file_location("stage4_extended_table_specs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_extended_table_specs"] = module
    spec.loader.exec_module(module)
    return module


ext17 = _load_extended_table_module()
METHOD_SPECS = ext17.METHOD_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        default_paths = ext17.parse_args()
    finally:
        sys.argv = original_argv
    seen_path_attrs = set()
    for spec in METHOD_SPECS:
        attr = spec["path_attr"]
        if attr in seen_path_attrs:
            continue
        seen_path_attrs.add(attr)
        parser.add_argument(
            f"--{attr}",
            type=Path,
            default=Path(getattr(default_paths, attr)),
            help=argparse.SUPPRESS,
        )
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "22_collect_stage4_all_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--patch_run_root",
        type=Path,
        default=RUNS_DIR / "ablation_pv_field_patch_size",
        help="Root containing patch_1x1, patch_3x3, patch_7x7 run directories.",
    )
    parser.add_argument(
        "--patch_sizes",
        type=str,
        default="1,3,5,7",
        help="Comma-separated patch sizes to include in the same alignment.",
    )
    parser.add_argument(
        "--patch_detail",
        action="append",
        default=[],
        metavar="SIZE=CSV_OR_DIR",
        help="Override a patch-size result. Can repeat, e.g. --patch_detail 3=/path/to/run_or_detail.csv.",
    )
    parser.add_argument(
        "--patch_5_detail_csv",
        type=Path,
        default=Path(getattr(default_paths, "future_field_detail_csv")),
        help="5x5 patch detail. Defaults to the main FG-MSF result so main and patch tables share the same 5x5 file.",
    )
    parser.add_argument(
        "--align_mode",
        choices=["intersection", "reference", "none"],
        default="intersection",
        help="intersection: all loaded methods share identical keys; reference: align to temporal baseline; none: evaluate as-is.",
    )
    parser.add_argument("--reference_key", type=str, default="temporal_baseline")
    parser.add_argument(
        "--include_keys",
        type=str,
        default=DEFAULT_PAPER_KEYS,
        help="Comma-separated method keys to include. Empty means all available.",
    )
    parser.add_argument(
        "--exclude_keys",
        type=str,
        default="",
        help="Comma-separated method keys to skip.",
    )
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--strict_missing", type=int, default=0)
    return parser.parse_args()


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def read_detail(path: Path, spec: dict[str, str], strict: bool) -> pd.DataFrame | None:
    path = Path(path)
    key = spec["key"]
    if not path.exists():
        message = f"[skip] {key}: missing {path}"
        if strict:
            raise FileNotFoundError(message)
        print(message, flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [*KEY_COLS, "horizon", "pred_pv_norm", "true_pv_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = attach_capacity_from_station_meta(df, path)
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["horizon"] = out["horizon"].astype(str)
    out["method_key"] = key
    out["Method"] = spec["method"]
    out["Spatial type"] = spec["spatial_type"]
    if "large_change" in out.columns:
        out["large_change"] = coerce_bool(out["large_change"])
    print(f"[read] {key}: {path} rows={len(out)}", flush=True)
    return out


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEY_COLS].drop_duplicates().copy()


def align_details(details: dict[str, pd.DataFrame], mode: str, reference_key: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if mode == "none" or not details:
        keys = pd.concat([key_frame(df) for df in details.values()], ignore_index=True).drop_duplicates(KEY_COLS)
        return details, keys
    if mode == "reference":
        if reference_key not in details:
            raise ValueError(f"--align_mode reference requires {reference_key!r} detail")
        keys = key_frame(details[reference_key])
    else:
        keys = None
        for df in details.values():
            k = key_frame(df)
            keys = k if keys is None else keys.merge(k, on=KEY_COLS, how="inner")
    assert keys is not None
    out: dict[str, pd.DataFrame] = {}
    for key, df in details.items():
        before = len(df)
        out[key] = df.merge(keys, on=KEY_COLS, how="inner")
        if len(out[key]) != before:
            print(f"[align] {key}: rows {before} -> {len(out[key])}", flush=True)
    return out, keys


def parse_key_set(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def parse_patch_sizes(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        cleaned = part.strip().lower().replace("x", "")
        if cleaned:
            out.append(int(cleaned))
    return sorted(set(out))


def detail_from_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / PATCH_DETAIL_NAME
    return path


def parse_patch_overrides(items: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"--patch_detail must be SIZE=CSV_OR_DIR, got {item!r}")
        size_text, path_text = str(item).split("=", 1)
        size = int(size_text.strip().lower().replace("x", ""))
        out[size] = detail_from_path(Path(path_text))
    return out


def patch_spec(size: int) -> dict[str, str]:
    key = f"patch_{size}x{size}"
    return {
        "key": key,
        "method": f"{size} x {size} patch",
        "spatial_type": "Patch-size ablation",
        "path_attr": f"{key}_detail_csv",
    }


def discover_patch_paths(args: argparse.Namespace) -> dict[int, Path]:
    overrides = parse_patch_overrides(list(args.patch_detail))
    sizes = parse_patch_sizes(str(args.patch_sizes))
    paths: dict[int, Path] = {}
    for size in sizes:
        if size in overrides:
            paths[size] = overrides[size]
        elif size == 5:
            paths[size] = Path(args.patch_5_detail_csv)
        else:
            paths[size] = Path(args.patch_run_root) / f"patch_{size}x{size}" / PATCH_DETAIL_NAME
    return paths


def apply_paper_labels(spec: dict[str, str]) -> dict[str, str]:
    out = dict(spec)
    if out["key"] in PAPER_LABELS:
        spatial_type, method = PAPER_LABELS[out["key"]]
        out["spatial_type"] = spatial_type
        out["method"] = method
    return out


def metric_row(df: pd.DataFrame, method: str, spatial_type: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    pred = df["pred_pv_norm"].astype(float)
    truth = df["true_pv_norm"].astype(float)
    err = pred - truth
    row: dict[str, Any] = {
        "Method": method,
        "Spatial type": spatial_type,
        "n_points": int(len(df)),
        "n_stations": int(df["station_sn"].nunique()),
        "MAE": float(err.abs().mean()),
        "RMSE": float(math.sqrt(float((err**2).mean()))),
        "Bias": float(err.mean()),
    }
    extras = regression_metric_extras(df, "pred_pv_norm")
    row["R2"] = float(extras["r2_pv_norm"])
    row["ACC"] = float(extras["acc_installed_capacity"])
    row["nRMSE_installed_capacity"] = float(extras["nrmse_installed_capacity"])
    row["ACC_capacity_source"] = extras["acc_capacity_source"]
    if "large_change" in df.columns:
        mask = coerce_bool(df["large_change"])
        row["LargeChangeMAE"] = float(err.abs()[mask].mean()) if bool(mask.any()) else float("nan")
        row["LargeChangeN"] = int(mask.sum())
    if extra:
        row.update(extra)
    return row


def build_method_tables(details: dict[str, pd.DataFrame], specs_by_key: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    for key, df in details.items():
        spec = specs_by_key[key]
        overall_rows.append(metric_row(df, spec["method"], spec["spatial_type"], {"method_key": key, "scope": "overall"}))
        for horizon, g in df.groupby("horizon", sort=True):
            horizon_rows.append(
                metric_row(g, spec["method"], spec["spatial_type"], {"method_key": key, "horizon": str(horizon)})
            )
        for station_sn, g in df.groupby("station_sn", sort=True):
            station_rows.append(
                metric_row(g, spec["method"], spec["spatial_type"], {"method_key": key, "station_sn": str(station_sn)})
            )
    overall = pd.DataFrame(overall_rows)
    horizon = pd.DataFrame(horizon_rows)
    station = pd.DataFrame(station_rows)
    if not horizon.empty:
        horizon["_h_order"] = horizon["horizon"].astype(str).str.extract(r"h(\d+)", expand=False).astype(float)
        horizon = horizon.sort_values(["_h_order", "method_key"]).drop(columns=["_h_order"]).reset_index(drop=True)
    return overall, horizon, station


def build_improvement_tables(
    overall: pd.DataFrame,
    horizon: pd.DataFrame,
    station: pd.DataFrame,
    reference_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def improve(table: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
        if table.empty or reference_key not in set(table["method_key"].astype(str)):
            return pd.DataFrame()
        ref_cols = [*join_cols, "MAE", "RMSE", "R2", "ACC", "n_points"]
        ref = table[table["method_key"].astype(str) == reference_key][ref_cols].copy()
        ref = ref.rename(
            columns={
                "MAE": "reference_MAE",
                "RMSE": "reference_RMSE",
                "R2": "reference_R2",
                "ACC": "reference_ACC",
                "n_points": "reference_n_points",
            }
        )
        cand = table[table["method_key"].astype(str) != reference_key].copy()
        if join_cols:
            out = cand.merge(ref, on=join_cols, how="inner")
        else:
            ref = ref.head(1).copy()
            ref["_tmp_key"] = 1
            cand["_tmp_key"] = 1
            out = cand.merge(ref, on="_tmp_key", how="inner").drop(columns=["_tmp_key"])
        out["MAE_gain"] = out["reference_MAE"] - out["MAE"]
        out["RMSE_gain"] = out["reference_RMSE"] - out["RMSE"]
        out["R2_gain"] = out["R2"] - out["reference_R2"]
        out["ACC_gain"] = out["ACC"] - out["reference_ACC"]
        out["MAE_gain_pct"] = np.where(out["reference_MAE"] > 0, 100.0 * out["MAE_gain"] / out["reference_MAE"], np.nan)
        return out

    return improve(overall, []), improve(horizon, ["horizon"]), improve(station, ["station_sn"])


def build_station_win_rate(station_improvement: pd.DataFrame) -> pd.DataFrame:
    if station_improvement.empty:
        return pd.DataFrame()
    rows = []
    for (key, method, spatial_type), g in station_improvement.groupby(["method_key", "Method", "Spatial type"], sort=False):
        rows.append(
            {
                "method_key": key,
                "Method": method,
                "Spatial type": spatial_type,
                "n_stations": int(g["station_sn"].nunique()),
                "n_win_mae": int((g["MAE_gain"] > 0).sum()),
                "win_rate_mae_pct": float(100.0 * (g["MAE_gain"] > 0).mean()),
                "mean_MAE_gain": float(g["MAE_gain"].mean()),
                "median_MAE_gain": float(g["MAE_gain"].median()),
                "mean_MAE_gain_pct": float(g["MAE_gain_pct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("win_rate_mae_pct", ascending=False).reset_index(drop=True)


def fmt(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def write_md_table(path: Path, table: pd.DataFrame, cols: list[str], digits: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in table.iterrows():
            cells = [fmt(row[col], digits) if col not in {"Method", "Spatial type", "method_key", "horizon", "station_sn"} else str(row[col]) for col in cols]
            f.write("| " + " | ".join(cells) + " |\n")


def print_table(title: str, table: pd.DataFrame, cols: list[str], digits: int) -> None:
    print(f"\n=== {title} ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        cells = [fmt(row[col], digits) if col not in {"Method", "Spatial type", "method_key", "horizon", "station_sn"} else str(row[col]) for col in cols]
        print("| " + " | ".join(cells) + " |")


def ordered_subset(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    out = table[table["method_key"].astype(str).isin(keys)].copy()
    order = {key: i for i, key in enumerate(keys)}
    out["_order"] = out["method_key"].astype(str).map(order)
    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def write_paper_tables(out_dir: Path, overall: pd.DataFrame, digits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_keys = [
        "lstm_nwp",
        "temporal_baseline",
        "tcn_nwp",
        "transformer_nwp",
        "patchtst_nwp",
        "stgcn_forecast_weather",
        "agcrn_nwp",
        "graph_wavenet_nwp",
        "nearest3_station_pv",
        "future_swr_patch",
        "historical_pv_field_patch",
        "fg_msf",
    ]
    patch_keys = ["temporal_baseline", "patch_1x1", "patch_3x3", "patch_5x5", "patch_7x7"]
    main = ordered_subset(overall, main_keys).copy()
    patch = ordered_subset(overall, patch_keys).copy()
    if not main.empty:
        main_out = main.rename(columns={"Spatial type": "Type"})
        main_cols = ["Type", "Method", "MAE", "RMSE", "R2", "n_points"]
        main_out[main_cols].to_csv(out_dir / "stage4_paper_main_table_common_subset.csv", index=False)
        write_md_table(out_dir / "stage4_paper_main_table_common_subset.md", main_out, main_cols, digits)
    if not patch.empty:
        patch_out = patch.copy()
        patch_out["Setting"] = patch_out["method_key"].map(
            {
                "temporal_baseline": "Temporal baseline",
                "patch_1x1": "1 x 1 patch",
                "patch_3x3": "3 x 3 patch",
                "patch_5x5": "5 x 5 patch",
                "patch_7x7": "7 x 7 patch",
            }
        )
        patch_cols = ["Setting", "MAE", "RMSE", "R2", "n_points"]
        patch_out[patch_cols].to_csv(out_dir / "stage4_patch_size_table_common_subset.csv", index=False)
        write_md_table(out_dir / "stage4_patch_size_table_common_subset.md", patch_out, patch_cols, digits)
    return main, patch


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = bool(int(args.strict_missing))
    include_keys = parse_key_set(str(args.include_keys))
    exclude_keys = parse_key_set(str(args.exclude_keys))

    standard_specs = [apply_paper_labels(spec) for spec in METHOD_SPECS]
    patch_paths = discover_patch_paths(args)
    patch_specs = [patch_spec(size) for size in sorted(patch_paths)]
    all_specs = standard_specs + patch_specs
    specs_by_key = {spec["key"]: spec for spec in all_specs}
    details: dict[str, pd.DataFrame] = {}
    path_map: dict[str, str] = {}
    count_rows: list[dict[str, Any]] = []
    for spec in standard_specs:
        if include_keys and spec["key"] not in include_keys:
            continue
        if spec["key"] in exclude_keys:
            continue
        path = Path(getattr(args, spec["path_attr"]))
        df = read_detail(path, spec, strict)
        if df is None:
            continue
        raw_rows = int(len(df))
        df = df[df["eval_split"].astype(str) == str(args.split)].copy()
        if df.empty:
            print(f"[skip] {spec['key']}: empty after split={args.split}", flush=True)
            continue
        details[spec["key"]] = df
        path_map[spec["key"]] = str(path)
        count_rows.append(
            {
                "method_key": spec["key"],
                "Method": spec["method"],
                "raw_rows": raw_rows,
                "split_rows_before_alignment": int(len(df)),
                "split_station_count_before_alignment": int(df["station_sn"].nunique()),
            }
        )
    for size, path in patch_paths.items():
        spec = patch_spec(size)
        if include_keys and spec["key"] not in include_keys:
            continue
        if spec["key"] in exclude_keys:
            continue
        df = read_detail(Path(path), spec, strict)
        if df is None:
            continue
        raw_rows = int(len(df))
        df = df[df["eval_split"].astype(str) == str(args.split)].copy()
        if df.empty:
            print(f"[skip] {spec['key']}: empty after split={args.split}", flush=True)
            continue
        details[spec["key"]] = df
        path_map[spec["key"]] = str(path)
        count_rows.append(
            {
                "method_key": spec["key"],
                "Method": spec["method"],
                "raw_rows": raw_rows,
                "split_rows_before_alignment": int(len(df)),
                "split_station_count_before_alignment": int(df["station_sn"].nunique()),
            }
        )

    details, aligned_keys = align_details(details, str(args.align_mode), str(args.reference_key))
    if not details:
        raise RuntimeError("No Stage 4 result files were loaded.")
    count_df = pd.DataFrame(count_rows)
    if not count_df.empty:
        aligned_counts = []
        for key, df in details.items():
            aligned_counts.append(
                {
                    "method_key": key,
                    "aligned_rows": int(len(df)),
                    "aligned_station_count": int(df["station_sn"].nunique()),
                    "aligned_issue_count": int(df[["station_sn", "issue_time"]].drop_duplicates().shape[0]),
                }
            )
        count_df = count_df.merge(pd.DataFrame(aligned_counts), on="method_key", how="left")

    overall, horizon, station = build_method_tables(details, specs_by_key)
    overall_imp, horizon_imp, station_imp = build_improvement_tables(overall, horizon, station, str(args.reference_key))
    win_rate = build_station_win_rate(station_imp)
    paper_main, paper_patch = write_paper_tables(out_dir, overall, int(args.digits))

    overall.to_csv(out_dir / "stage4_all_results_overall.csv", index=False)
    horizon.to_csv(out_dir / "stage4_all_results_by_horizon.csv", index=False)
    station.to_csv(out_dir / "stage4_all_results_by_station.csv", index=False)
    overall_imp.to_csv(out_dir / "stage4_all_results_improvement_overall_vs_reference.csv", index=False)
    horizon_imp.to_csv(out_dir / "stage4_all_results_improvement_horizon_vs_reference.csv", index=False)
    station_imp.to_csv(out_dir / "stage4_all_results_improvement_station_vs_reference.csv", index=False)
    win_rate.to_csv(out_dir / "stage4_all_results_station_win_rate_vs_reference.csv", index=False)
    aligned_keys.to_csv(out_dir / "stage4_all_results_aligned_keys.csv", index=False)
    count_df.to_csv(out_dir / "stage4_all_results_method_row_counts.csv", index=False)

    overall_cols = ["Method", "Spatial type", "MAE", "RMSE", "R2", "ACC", "Bias", "n_points", "n_stations"]
    win_cols = ["Method", "Spatial type", "n_stations", "n_win_mae", "win_rate_mae_pct", "mean_MAE_gain", "mean_MAE_gain_pct"]
    write_md_table(out_dir / "stage4_all_results_overall.md", overall, overall_cols, int(args.digits))
    if not win_rate.empty:
        write_md_table(out_dir / "stage4_all_results_station_win_rate_vs_reference.md", win_rate, win_cols, int(args.digits))

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "split": str(args.split),
                "align_mode": str(args.align_mode),
                "reference_key": str(args.reference_key),
                "include_keys": sorted(include_keys),
                "exclude_keys": sorted(exclude_keys),
                "patch_sizes": parse_patch_sizes(str(args.patch_sizes)),
                "patch_paths": {f"patch_{size}x{size}": str(path) for size, path in sorted(patch_paths.items())},
                "loaded_methods": list(details.keys()),
                "paths": path_map,
                "aligned_key_count": int(len(aligned_keys)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("==== Stage 4 all-result collection complete ====")
    print(f"split={args.split} align_mode={args.align_mode} reference={args.reference_key}")
    print(f"loaded methods: {', '.join(details.keys())}")
    print(f"aligned keys: {len(aligned_keys)}")
    if not count_df.empty:
        print("\n=== Row Counts ===")
        show_cols = ["method_key", "split_rows_before_alignment", "aligned_rows", "aligned_station_count"]
        print(count_df[show_cols].to_string(index=False))
    print_table("Overall", overall, overall_cols, int(args.digits))
    if not paper_main.empty:
        print_table("Paper Main Table Common Subset", paper_main.rename(columns={"Spatial type": "Type"}), ["Type", "Method", "MAE", "RMSE", "R2", "n_points"], int(args.digits))
    if not paper_patch.empty:
        patch_show = paper_patch.copy()
        patch_show["Setting"] = patch_show["method_key"].map(
            {
                "temporal_baseline": "Temporal baseline",
                "patch_1x1": "1 x 1 patch",
                "patch_3x3": "3 x 3 patch",
                "patch_5x5": "5 x 5 patch",
                "patch_7x7": "7 x 7 patch",
            }
        )
        print_table("Patch-Size Table Common Subset", patch_show, ["Setting", "MAE", "RMSE", "R2", "n_points"], int(args.digits))
    if not win_rate.empty:
        print_table("Station Win Rate vs Reference", win_rate, win_cols, int(args.digits))
    print(f"\n[write] {out_dir}")


if __name__ == "__main__":
    main()
