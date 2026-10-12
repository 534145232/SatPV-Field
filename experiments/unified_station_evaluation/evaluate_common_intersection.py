#!/usr/bin/env python3
"""Strict common-sample evaluation for the station-level paper tables.

This script does not train or rerun a model. It discovers the existing
per-prediction CSV files, canonicalizes them, intersects all methods on
(target_id, issue_time, horizon), verifies the common truth values, and then
recomputes every metric from that exact same sample set.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEYS = ["target_id", "issue_time", "horizon"]
EXPECTED_HORIZONS = tuple(range(1, 9))
EXPECTED_TARGETS = 50


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    category: str
    run_dir: str | None
    filename: str
    selector: str | None = None


METHODS = (
    MethodSpec("bilstm", "BiLSTM+NWP", "Temporal", "03_per_station_temporal_baseline_seed2026", "per_station_temporal_baseline_eval_detail.csv"),
    MethodSpec("nearest3", "3-NN PV guidance", "Spatial guidance", "21_nearest3_station_pv_baseline_seed2026", "nearest3_station_pv_eval_detail.csv"),
    MethodSpec("future_irradiance", "Future irradiance guidance", "Environmental guidance", "06l_per_station_temporal_stage3_swr_patch_seed2026", "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"),
    MethodSpec("historical_field", "Historical PV-field guidance", "Field guidance", "05h_per_station_temporal_historical_field_patch_seed2026", "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"),
    MethodSpec("predicted_field", "Predicted future PV-field guidance (SatPV-Field)", "Proposed", "05l_per_station_temporal_stage3_field_patch_seed2026", "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"),
    MethodSpec("historical_graph", "HistoricalGraph", "Spatio-temporal", None, "target_graph_eval_detail.csv", "HistoricalGraph"),
    MethodSpec("predicted_graph", "PredictedGraph", "Spatio-temporal", None, "target_graph_eval_detail.csv", "PredictedGraph"),
    MethodSpec("lstm", "LSTM+NWP", "Temporal", "19_lstm_nwp_baseline_seed2026", "lstm_nwp_eval_detail.csv"),
    MethodSpec("tcn", "TCN+NWP", "Temporal", "19_tcn_nwp_baseline_seed2026", "tcn_nwp_eval_detail.csv"),
    MethodSpec("transformer", "Transformer+NWP", "Temporal", "19_transformer_nwp_baseline_seed2026", "transformer_nwp_eval_detail.csv"),
    MethodSpec("patchtst", "PatchTST+NWP", "Temporal", "19_patchtst_nwp_baseline_seed2026", "patchtst_nwp_eval_detail.csv"),
    MethodSpec("stgcn", "STGCN+NWP", "Spatio-temporal", "20_stgcn_forecast_weather_baseline_seed2026", "stgcn_forecast_weather_eval_detail.csv"),
    MethodSpec("agcrn", "AGCRN+NWP", "Spatio-temporal", "23_agcrn_nwp_baseline_seed2026", "agcrn_nwp_eval_detail.csv"),
)

METHOD_BY_KEY = {item.key: item for item in METHODS}

TABLE_III_ORDER = (
    "bilstm", "nearest3", "future_irradiance", "historical_field",
    "predicted_field", "historical_graph", "predicted_graph",
)
TABLE_IV_ORDER = (
    "lstm", "bilstm", "tcn", "transformer", "patchtst", "stgcn",
    "agcrn", "historical_graph", "predicted_graph", "predicted_field",
)

PAIR_SPECS = (
    ("historical_graph", "historical_field"),
    ("predicted_graph", "predicted_field"),
    ("predicted_graph", "historical_graph"),
    ("predicted_field", "historical_field"),
)

# These numbers are only manuscript references for explaining changes. They
# are never used to fill a recomputed table or replace prediction records.
LEGACY_TABLE_IV = {
    "lstm": (0.0754, 0.1092, 0.7927),
    "bilstm": (0.0746, 0.1084, 0.7956),
    "tcn": (0.0753, 0.1086, 0.7949),
    "transformer": (0.0783, 0.1125, 0.7799),
    "patchtst": (0.0778, 0.1130, 0.7780),
    "stgcn": (0.0758, 0.1082, 0.7965),
    "agcrn": (0.0694, 0.1035, 0.8137),
    "predicted_field": (0.0672, 0.0999, 0.8266),
    "predicted_graph": (None, 0.0990, 0.7990),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument(
        "--old-data-root", type=Path,
        default=Path(
            os.environ.get(
                "SATPV_DATA_ROOT",
                os.environ.get("OLD_DATA_ROOT", project / "data"),
            )
        ),
        help="Root containing copied stage3/stage4 result trees.",
    )
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--target-stations-csv", type=Path, default=None)
    parser.add_argument(
        "--method-path", action="append", default=[], metavar="KEY=CSV",
        help="Override one input, e.g. predicted_graph=/path/target_graph_eval_detail.csv",
    )
    parser.add_argument("--truth-atol", type=float, default=1.0e-6)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-incomplete-common", action="store_true",
        help="Allow output when the common intersection does not contain all 50 targets and h1-h8.",
    )
    return parser.parse_args()


def parse_overrides(items: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method-path {item!r}; expected KEY=CSV")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in METHOD_BY_KEY:
            raise ValueError(f"Unknown method key {key!r}; choose from {sorted(METHOD_BY_KEY)}")
        result[key] = Path(value).expanduser()
    return result


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        token = str(path.resolve()) if path.exists() else str(path.absolute())
        if token not in seen:
            seen.add(token)
            result.append(path)
    return result


def stage4_roots(project: Path, old_data: Path) -> list[Path]:
    roots = [
        project / "stage_4" / "runs",
        project / "stage4" / "runs",
        old_data / "stage4" / "runs",
        old_data / "stage4",
        old_data / "stage_4" / "runs",
        old_data / "stage_4",
        old_data / "runs",
        old_data,
    ]
    return unique_paths(roots)


def graph_candidates(project: Path, old_data: Path) -> list[Path]:
    relative = Path("paper_support/matched_representation/outputs/full/target_graph/target_graph_eval_detail.csv")
    return unique_paths([
        project / relative,
        old_data / relative,
        old_data / "matched_representation/outputs/full/target_graph/target_graph_eval_detail.csv",
    ])


def resolve_method_path(
    spec: MethodSpec,
    project: Path,
    old_data: Path,
    override: Path | None,
) -> tuple[Path | None, list[Path]]:
    if override is not None:
        return (override if override.is_file() else None), [override]
    if spec.run_dir is None:
        tried = graph_candidates(project, old_data)
    else:
        tried = [root / spec.run_dir / spec.filename for root in stage4_roots(project, old_data)]
    hits = unique_paths(path for path in tried if path.is_file())
    if len(hits) > 1:
        options = "\n".join(f"  - {path}" for path in hits)
        raise RuntimeError(
            f"Ambiguous input for {spec.label}; multiple files exist. "
            f"Pass --method-path {spec.key}=CSV explicitly:\n{options}"
        )
    return (hits[0] if hits else None), tried


def target_candidates(project: Path, old_data: Path) -> list[Path]:
    relative = Path("data/02_target_selection_lhs50/target_stations.csv")
    return unique_paths([
        project / "stage_3" / relative,
        project / "stage3" / relative,
        old_data / "stage3" / relative,
        old_data / "stage_3" / relative,
        old_data.parent / "stage3" / relative,
        old_data.parent / "stage_3" / relative,
    ])


def load_target_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype=str)
    aliases = ("station_sn", "target_id", "station_id", "sn", "id")
    column = next((name for name in aliases if name in frame.columns), None)
    if column is None:
        raise ValueError(f"No station-id column in target list {path}; columns={list(frame.columns)}")
    values = frame[column].dropna().astype(str).str.strip()
    if values.duplicated().any():
        raise ValueError(f"Duplicate station IDs in target list {path}")
    result = set(values)
    if len(result) != EXPECTED_TARGETS:
        raise ValueError(f"Expected {EXPECTED_TARGETS} held-out targets, found {len(result)} in {path}")
    return result


def normalize_issue_time(series: pd.Series, path: Path) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        examples = series[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Unparseable issue_time in {path}: {examples}")
    try:
        timezone = parsed.dt.tz
    except AttributeError as exc:
        raise ValueError(f"Mixed or unsupported timezone values in {path}") from exc
    if timezone is not None:
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return parsed.dt.floor("s")


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y", "valid"})


def canonical_detail(
    path: Path,
    spec: MethodSpec,
    target_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    id_col = "station_sn" if "station_sn" in available else "target_id" if "target_id" in available else None
    h_col = "h" if "h" in available else "horizon" if "horizon" in available else None
    required = {"issue_time", "pred_pv_norm", "true_pv_norm"}
    missing = sorted(required - available)
    if id_col is None:
        missing.append("station_sn/target_id")
    if h_col is None:
        missing.append("h/horizon")
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")
    if "eval_split" not in available:
        raise ValueError(f"{path} has no eval_split column; test rows cannot be proven")
    method_col = None
    if spec.selector is not None:
        method_col = "method" if "method" in available else "model" if "model" in available else None
        if method_col is None:
            raise ValueError(f"{path} must contain method/model to select {spec.selector}")
    use_columns = {
        "eval_split", "issue_time", "pred_pv_norm", "true_pv_norm",
        str(id_col), str(h_col),
    }
    if "valid" in available:
        use_columns.add("valid")
    if method_col is not None:
        use_columns.add(method_col)
    dtype_map = {str(id_col): str, "issue_time": str, "eval_split": str}
    if method_col is not None:
        dtype_map[method_col] = str
    frame = pd.read_csv(path, usecols=sorted(use_columns), dtype=dtype_map)
    raw_rows = len(frame)
    frame = frame[frame["eval_split"].astype(str).str.lower().eq("test")].copy()
    test_rows = len(frame)
    if spec.selector is not None:
        assert method_col is not None
        frame = frame[frame[method_col].astype(str).eq(spec.selector)].copy()
    selected_rows = len(frame)
    frame["target_id"] = frame[id_col].astype(str).str.strip()
    frame = frame[frame["target_id"].isin(target_ids)].copy()
    target_rows = len(frame)
    horizon_text = frame[h_col].astype(str).str.lower().str.strip().str.replace("h", "", regex=False)
    frame["horizon"] = pd.to_numeric(horizon_text, errors="coerce")
    frame = frame[frame["horizon"].isin(EXPECTED_HORIZONS)].copy()
    frame["horizon"] = frame["horizon"].astype(int)
    horizon_rows = len(frame)
    frame["issue_time"] = normalize_issue_time(frame["issue_time"], path)
    frame["y_pred"] = pd.to_numeric(frame["pred_pv_norm"], errors="coerce")
    frame["y_true"] = pd.to_numeric(frame["true_pv_norm"], errors="coerce")
    if "valid" in frame.columns:
        explicit_valid = truthy(frame["valid"])
        valid_source = "explicit valid column + finite values"
    else:
        explicit_valid = pd.Series(True, index=frame.index)
        valid_source = "no valid column; finite values only"
    finite = np.isfinite(frame["y_pred"].to_numpy(float)) & np.isfinite(frame["y_true"].to_numpy(float))
    frame = frame[explicit_valid.to_numpy() & finite].copy()
    duplicate_mask = frame.duplicated(KEYS, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        examples = frame.loc[duplicate_mask, KEYS].head(10).to_dict("records")
        raise ValueError(f"Duplicate keys for {spec.label} in {path}: {examples}")
    result = frame[[*KEYS, "y_true", "y_pred"]].sort_values(KEYS).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"No valid test records remain for {spec.label} after target/horizon/finite filtering: {path}")
    original_pooled = metric_values(result["y_true"].to_numpy(), result["y_pred"].to_numpy())
    original_station_rows = [
        metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for _, group in result.groupby("target_id", sort=True)
    ]
    original_station = pd.DataFrame(original_station_rows)
    coverage = {
        "method_key": spec.key,
        "method": spec.label,
        "status": "loaded",
        "path": str(path.resolve()),
        "normalization": "capacity-normalized PV generation (p.u.); pred_pv_norm/true_pv_norm",
        "valid_rule": valid_source,
        "raw_rows": raw_rows,
        "test_rows": test_rows,
        "selector_rows": selected_rows,
        "target_rows": target_rows,
        "horizon_rows": horizon_rows,
        "valid_unique_rows": len(result),
        "duplicate_rows": duplicate_rows,
        "n_targets_before_intersection": result["target_id"].nunique(),
        "n_issue_times_before_intersection": result["issue_time"].nunique(),
        "horizons_before_intersection": ",".join(map(str, sorted(result["horizon"].unique()))),
        "original_pooled_mae": original_pooled["mae"],
        "original_pooled_rmse": original_pooled["rmse"],
        "original_pooled_r2": original_pooled["r2"],
        "original_macro_mae": float(original_station["mae"].mean()),
        "original_macro_rmse": float(original_station["rmse"].mean()),
        "original_macro_r2": float(original_station["r2"].mean()),
    }
    return result, coverage


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def key_hashes(frame: pd.DataFrame) -> np.ndarray:
    """Compact deterministic fingerprints used only to reduce peak memory.

    Exact key equality is verified again in the second pass, so a theoretical
    hash collision cannot silently enter the evaluation.
    """
    return pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy(dtype=np.uint64)


def common_hash_mask(frame: pd.DataFrame, sorted_common_hashes: np.ndarray) -> np.ndarray:
    hashes = key_hashes(frame)
    positions = np.searchsorted(sorted_common_hashes, hashes)
    mask = positions < len(sorted_common_hashes)
    safe_positions = np.minimum(positions, max(len(sorted_common_hashes) - 1, 0))
    if len(sorted_common_hashes):
        mask &= sorted_common_hashes[safe_positions] == hashes
    return mask


def evaluate_second_pass(
    resolved: dict[str, Path],
    target_ids: set[str],
    common_hashes: np.ndarray,
    truth_atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    common_keys: pd.DataFrame | None = None
    reference_truth: np.ndarray | None = None
    reference_method = ""
    max_truth_difference = 0.0
    metric_rows: list[dict] = []
    station_rows: list[dict] = []
    horizon_rows: list[dict] = []

    for index, spec in enumerate(METHODS, start=1):
        frame, _ = canonical_detail(resolved[spec.key], spec, target_ids)
        selected = frame.loc[common_hash_mask(frame, common_hashes)].copy()
        if len(selected) != len(common_hashes):
            raise AssertionError(
                f"Hash intersection count mismatch for {spec.label}: "
                f"selected={len(selected)}, expected={len(common_hashes)}"
            )
        if common_keys is None:
            common_keys = selected[KEYS].sort_values(KEYS).reset_index(drop=True)
        aligned = common_keys.merge(
            selected[[*KEYS, "y_true", "y_pred"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        if len(aligned) != len(common_keys) or aligned[["y_true", "y_pred"]].isna().any().any():
            raise AssertionError(
                f"Exact key verification failed for {spec.label}; a fingerprint collision or key mismatch exists"
            )
        truth = aligned["y_true"].to_numpy(np.float64)
        prediction = aligned["y_pred"].to_numpy(np.float64)
        if reference_truth is None:
            reference_truth = truth.copy()
            reference_method = spec.label
        else:
            difference = float(np.max(np.abs(reference_truth - truth)))
            max_truth_difference = max(max_truth_difference, difference)
            if not np.allclose(reference_truth, truth, rtol=0.0, atol=truth_atol):
                bad = np.flatnonzero(np.abs(reference_truth - truth) > truth_atol)[:10]
                examples = [
                    {**common_keys.iloc[int(i)].to_dict(), "reference": float(reference_truth[i]), "candidate": float(truth[i])}
                    for i in bad
                ]
                raise AssertionError(
                    f"y_true disagreement: {spec.label} versus {reference_method}; "
                    f"max_abs_diff={difference:.9g}; examples={examples}"
                )

        method_station_rows: list[dict] = []
        for target_id, group in aligned.groupby("target_id", sort=True):
            row = {
                "method_key": spec.key, "method": spec.label,
                "target_id": target_id, "n_samples": len(group),
            }
            row.update(metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
            method_station_rows.append(row)
        station_rows.extend(method_station_rows)
        station_frame = pd.DataFrame(method_station_rows)

        for horizon, group in aligned.groupby("horizon", sort=True):
            row = {
                "method_key": spec.key, "method": spec.label,
                "horizon": int(horizon), "horizon_minutes": int(horizon) * 15,
                "n_samples": len(group), "n_targets": group["target_id"].nunique(),
            }
            row.update(metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
            horizon_rows.append(row)

        pooled = metric_values(truth, prediction)
        metric_rows.append({
            "method_key": spec.key, "type": spec.category, "method": spec.label,
            "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"], "pooled_r2": pooled["r2"],
            "macro_mae": float(station_frame["mae"].mean()),
            "macro_rmse": float(station_frame["rmse"].mean()),
            "macro_r2": float(station_frame["r2"].mean()),
            "n_records": len(aligned), "n_targets": aligned["target_id"].nunique(),
            "n_origins": aligned["issue_time"].nunique(),
            "macro_r2_valid_targets": int(station_frame["r2"].notna().sum()),
        })
        print(f"[metrics {index:02d}/{len(METHODS)}] {spec.label}: common_rows={len(aligned):,}", flush=True)
        del frame, selected, aligned, station_frame
        gc.collect()

    assert common_keys is not None and reference_truth is not None
    manifest = common_keys.copy()
    manifest["issue_time"] = manifest["issue_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    manifest["horizon_minutes"] = manifest["horizon"] * 15
    manifest["y_true"] = reference_truth
    manifest = manifest[[*KEYS, "horizon_minutes", "y_true"]]
    metrics = pd.DataFrame(metric_rows)
    stationwise = pd.DataFrame(station_rows).sort_values(["method_key", "target_id"]).reset_index(drop=True)
    horizonwise = pd.DataFrame(horizon_rows).sort_values(["method_key", "horizon"]).reset_index(drop=True)
    return manifest, metrics, stationwise, horizonwise, max_truth_difference


def strict_intersection(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common: pd.DataFrame | None = None
    for spec in METHODS:
        keys = frames[spec.key][KEYS].copy()
        common = keys if common is None else common.merge(keys, on=KEYS, how="inner", validate="one_to_one")
    if common is None or common.empty:
        raise ValueError("The strict 13-method intersection is empty")
    return common.sort_values(KEYS).reset_index(drop=True)


def align_and_verify(
    frames: dict[str, pd.DataFrame], common: pd.DataFrame, truth_atol: float
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, float]:
    aligned: dict[str, pd.DataFrame] = {}
    reference: np.ndarray | None = None
    reference_method = ""
    max_truth_difference = 0.0
    manifest = common.copy()
    for spec in METHODS:
        sub = common.merge(frames[spec.key], on=KEYS, how="left", validate="one_to_one")
        if sub[["y_true", "y_pred"]].isna().any().any():
            raise AssertionError(f"Unexpected missing values after intersection for {spec.label}")
        truth = sub["y_true"].to_numpy(np.float64)
        if reference is None:
            reference = truth
            reference_method = spec.label
            manifest["y_true"] = truth
        else:
            difference = float(np.max(np.abs(reference - truth)))
            max_truth_difference = max(max_truth_difference, difference)
            if not np.allclose(reference, truth, rtol=0.0, atol=truth_atol):
                bad = np.flatnonzero(np.abs(reference - truth) > truth_atol)[:10]
                examples = [
                    {**common.iloc[int(i)].to_dict(), "reference": float(reference[i]), "candidate": float(truth[i])}
                    for i in bad
                ]
                raise AssertionError(
                    f"y_true disagreement: {spec.label} versus {reference_method}; "
                    f"max_abs_diff={difference:.9g}; examples={examples}"
                )
        aligned[spec.key] = sub[[*KEYS, "y_true", "y_pred"]].copy()
    manifest["issue_time"] = manifest["issue_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    manifest["horizon_minutes"] = manifest["horizon"] * 15
    return aligned, manifest[[*KEYS, "horizon_minutes", "y_true"]], max_truth_difference


def compute_stationwise(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in METHODS:
        for target_id, group in aligned[spec.key].groupby("target_id", sort=True):
            row = {"method_key": spec.key, "method": spec.label, "target_id": target_id, "n_samples": len(group)}
            row.update(metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_key", "target_id"]).reset_index(drop=True)


def compute_horizonwise(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in METHODS:
        for horizon, group in aligned[spec.key].groupby("horizon", sort=True):
            row = {
                "method_key": spec.key, "method": spec.label,
                "horizon": int(horizon), "horizon_minutes": int(horizon) * 15,
                "n_samples": len(group), "n_targets": group["target_id"].nunique(),
            }
            row.update(metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_key", "horizon"]).reset_index(drop=True)


def compute_all_metrics(
    aligned: dict[str, pd.DataFrame], stationwise: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in METHODS:
        frame = aligned[spec.key]
        per_station = stationwise[stationwise["method_key"].eq(spec.key)]
        pooled = metric_values(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        rows.append({
            "method_key": spec.key,
            "type": spec.category,
            "method": spec.label,
            "n_records": len(frame),
            "n_targets": frame["target_id"].nunique(),
            "n_origins": frame["issue_time"].nunique(),
            "pooled_mae": pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "pooled_r2": pooled["r2"],
            "macro_mae": float(per_station["mae"].mean()),
            "macro_rmse": float(per_station["rmse"].mean()),
            "macro_r2": float(per_station["r2"].mean()),
            "macro_r2_valid_targets": int(per_station["r2"].notna().sum()),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    stationwise: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for left_key, right_key in PAIR_SPECS:
        left = stationwise[stationwise["method_key"].eq(left_key)][["target_id", "mae"]].rename(columns={"mae": "mae_a"})
        right = stationwise[stationwise["method_key"].eq(right_key)][["target_id", "mae"]].rename(columns={"mae": "mae_b"})
        paired = left.merge(right, on="target_id", how="inner", validate="one_to_one").sort_values("target_id")
        if len(paired) != EXPECTED_TARGETS:
            raise ValueError(f"Paired comparison {left_key} vs {right_key} has {len(paired)} targets, expected 50")
        diff = paired["mae_a"].to_numpy(np.float64) - paired["mae_b"].to_numpy(np.float64)
        sample_indices = rng.integers(0, len(diff), size=(reps, len(diff)))
        bootstrap_means = diff[sample_indices].mean(axis=1)
        tolerance = 1.0e-12
        rows.append({
            "comparison": f"{METHOD_BY_KEY[left_key].label} vs {METHOD_BY_KEY[right_key].label}",
            "method_a": METHOD_BY_KEY[left_key].label,
            "method_b": METHOD_BY_KEY[right_key].label,
            "difference_definition": "station MAE(method_a) - station MAE(method_b); negative favors method_a",
            "n_targets": len(diff),
            "mean_mae_difference": float(np.mean(diff)),
            "median_mae_difference": float(np.median(diff)),
            "method_a_wins": int(np.sum(diff < -tolerance)),
            "method_b_wins": int(np.sum(diff > tolerance)),
            "ties": int(np.sum(np.abs(diff) <= tolerance)),
            "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
            "bootstrap_reps": reps,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def paired_conclusion_lines(frame: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for _, row in frame.iterrows():
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        mean = float(row["mean_mae_difference"])
        if high < 0:
            conclusion = f"{row['method_a']} 的 station-wise MAE 显著更低"
        elif low > 0:
            conclusion = f"{row['method_b']} 的 station-wise MAE 显著更低"
        else:
            conclusion = "95% bootstrap CI 跨过 0，当前样本下无明确优势"
        lines.append(
            f"{row['comparison']}：mean difference={mean:.6f}，"
            f"95% CI=[{low:.6f}, {high:.6f}]；{conclusion}。"
        )
    return lines


def table_from_metrics(metrics: pd.DataFrame, order: Iterable[str]) -> pd.DataFrame:
    indexed = metrics.set_index("method_key")
    result = indexed.loc[list(order)].reset_index()
    return result[[
        "type", "method", "n_records", "n_targets",
        "pooled_mae", "pooled_rmse", "pooled_r2",
        "macro_mae", "macro_rmse", "macro_r2",
    ]]


def write_empty_outputs(output_dir: Path) -> None:
    schemas = {
        "common_sample_manifest.csv": [*KEYS, "horizon_minutes", "y_true"],
        "all_methods_common_metrics.csv": ["method_key", "type", "method", "pooled_mae", "pooled_rmse", "pooled_r2", "macro_mae", "macro_rmse", "macro_r2", "n_records", "n_targets", "n_origins"],
        "stationwise_metrics.csv": ["method_key", "method", "target_id", "n_samples", "mae", "rmse", "r2"],
        "horizonwise_metrics.csv": ["method_key", "method", "horizon", "horizon_minutes", "n_samples", "mae", "rmse", "r2"],
        "paired_graph_field_comparison.csv": ["comparison", "method_a", "method_b", "mean_mae_difference", "bootstrap_ci95_low", "bootstrap_ci95_high"],
        "table_III_recomputed.csv": ["type", "method", "n_records", "n_targets", "pooled_mae", "pooled_rmse", "pooled_r2"],
        "table_IV_recomputed.csv": ["type", "method", "n_records", "n_targets", "pooled_mae", "pooled_rmse", "pooled_r2"],
    }
    for filename, columns in schemas.items():
        path = output_dir / filename
        # Clear stale tables before a new run. If validation later fails, no
        # previous metrics can be mistaken for the current audited result.
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> list[str]:
    if frame.empty:
        return ["（无可用结果。）"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_audit(
    output_dir: Path,
    coverage: pd.DataFrame,
    status: str,
    notes: list[str],
    metrics: pd.DataFrame | None = None,
    paired: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# III-D / III-E 统一测试样本评估审计",
        "",
        f"- 状态：**{status}**",
        "- 评价单位：每条 `(target_id, issue_time, horizon)` 预测记录。",
        "- 时间语义：输入中的无时区时间按北京时间语义原样匹配；有时区时间转换为 Asia/Shanghai。",
        "- 归一化：只读取 `pred_pv_norm` 与 `true_pv_norm`，即按站点装机容量归一化的 PV generation（p.u.）。",
        "- 主表指标：在严格公共交集上 pooled MAE、pooled RMSE、pooled R²；R² 由公共真值向量一次性计算，不做站点 R² 平均。",
        "- 补充指标：station-macro 指标、逐站指标、逐 horizon pooled 指标。",
        "",
        "## 输入与覆盖",
        "",
    ]
    show_columns = [column for column in [
        "method", "status", "path", "raw_rows", "test_rows", "valid_unique_rows",
        "common_rows", "excluded_rows", "n_targets_before_intersection",
    ] if column in coverage.columns]
    lines.extend(markdown_table(coverage.fillna(""), show_columns, digits=0))
    lines.extend(["", "## 审计说明", ""])
    lines.extend(f"- {note}" for note in notes)
    if metrics is not None and not metrics.empty:
        lines.extend(["", "## 严格公共交集上的主结果", ""])
        lines.extend(markdown_table(metrics, ["method", "n_records", "pooled_mae", "pooled_rmse", "pooled_r2"]))
        lines.extend(["", "## 与旧 Table IV 数值的变化（仅作溯源对照）", ""])
        for key, old in LEGACY_TABLE_IV.items():
            selected = metrics[metrics["method_key"].eq(key)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            old_text = "/".join("未提供" if value is None else f"{value:.4f}" for value in old)
            source = coverage[coverage["method_key"].eq(key)]
            diagnosis = ""
            if not source.empty:
                source_row = source.iloc[0]
                pooled_values = (source_row["original_pooled_mae"], source_row["original_pooled_rmse"], source_row["original_pooled_r2"])
                macro_values = (source_row["original_macro_mae"], source_row["original_macro_rmse"], source_row["original_macro_r2"])
                pooled_distance = sum(abs(float(value) - float(candidate)) for value, candidate in zip(old, pooled_values) if value is not None)
                macro_distance = sum(abs(float(value) - float(candidate)) for value, candidate in zip(old, macro_values) if value is not None)
                nearer = "原文件 pooled" if pooled_distance < macro_distance else "原文件 station-macro"
                diagnosis = f"；旧值在数值上更接近{nearer}口径（仅为诊断，不作为新表输入）"
            lines.append(
                f"- {row['method']}：旧值 MAE/RMSE/R²={old_text}；"
                f"统一交集重算={row['pooled_mae']:.4f}/{row['pooled_rmse']:.4f}/{row['pooled_r2']:.4f}{diagnosis}。"
            )
        lines.append("- 上述差异只可能来自样本集合、聚合方式或旧表统计口径；新表不复用任何旧指标。")
    if paired is not None and not paired.empty:
        lines.extend(["", "## 图表示与场表示的配对比较", ""])
        lines.extend(markdown_table(paired, [
            "comparison", "mean_mae_difference", "median_mae_difference",
            "method_a_wins", "method_b_wins", "ties", "bootstrap_ci95_low", "bootstrap_ci95_high",
        ]))
        lines.append("")
        lines.append("差值定义为 `station MAE(method_a) - station MAE(method_b)`；负值表示 method_a 更好。")
    (output_dir / "evaluation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    old_data = args.old_data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_empty_outputs(output_dir)
    overrides = parse_overrides(args.method_path)

    # Phase 1 is independent of data loading: always report both code-tree and
    # separately uploaded old_data candidates, even if the target list is absent.
    resolved: dict[str, Path] = {}
    coverage_rows: list[dict] = []
    preflight_rows: list[dict] = []
    missing: list[str] = []
    for spec in METHODS:
        path, tried = resolve_method_path(spec, project, old_data, overrides.get(spec.key))
        if path is None:
            missing.append(spec.key)
            preflight_rows.append({
                "method_key": spec.key, "method": spec.label, "status": "missing",
                "path": "", "tried_paths": " | ".join(str(item) for item in tried),
            })
        else:
            resolved[spec.key] = path
            preflight_rows.append({
                "method_key": spec.key, "method": spec.label, "status": "resolved_not_evaluated",
                "path": str(path.resolve()), "tried_paths": " | ".join(str(item) for item in tried),
            })

    target_tried = [args.target_stations_csv] if args.target_stations_csv else target_candidates(project, old_data)
    target_hits = [path for path in target_tried if path is not None and path.is_file()]
    if len(target_hits) != 1:
        status = "FAILED: held-out target list unresolved"
        notes = ["尝试的 target list 路径："] + [f"`{path}`" for path in target_tried if path is not None]
        if missing:
            notes.append("同时缺少预测明细：" + ", ".join(METHOD_BY_KEY[key].label for key in missing))
        coverage = pd.DataFrame(preflight_rows)
        coverage.to_csv(output_dir / "sample_coverage_report.csv", index=False)
        write_audit(output_dir, coverage, status, notes)
        raise SystemExit(status)
    target_path = target_hits[0]
    target_ids = load_target_ids(target_path)

    if missing:
        coverage = pd.DataFrame(preflight_rows)
        coverage.to_csv(output_dir / "sample_coverage_report.csv", index=False)
        notes = [
            f"权威 target list：`{target_path}`（{len(target_ids)} 站）。",
            "以下方法缺少逐预测明细，已停止，未用旧表数字替代：" + ", ".join(METHOD_BY_KEY[key].label for key in missing),
            "可通过 `--method-path KEY=/absolute/path.csv` 指定非标准位置。",
        ]
        write_audit(output_dir, coverage, "FAILED: one or more prediction files are missing", notes)
        print(coverage.to_string(index=False))
        raise SystemExit(2)

    try:
        common_hashes: np.ndarray | None = None
        for index, spec in enumerate(METHODS, start=1):
            frame, row = canonical_detail(resolved[spec.key], spec, target_ids)
            coverage_rows.append(row)
            hashes = np.unique(key_hashes(frame))
            if len(hashes) != len(frame):
                raise AssertionError(
                    f"Key fingerprint collision detected within {spec.label}; exact evaluation stopped"
                )
            common_hashes = hashes if common_hashes is None else np.intersect1d(
                common_hashes, hashes, assume_unique=True
            )
            print(
                f"[intersection {index:02d}/{len(METHODS)}] {spec.label}: "
                f"valid_rows={len(frame):,}, current_common={len(common_hashes):,}",
                flush=True,
            )
            del frame, hashes
            gc.collect()
        if common_hashes is None or len(common_hashes) == 0:
            raise ValueError("The strict 13-method intersection is empty")

        manifest, metrics, stationwise, horizonwise, max_truth_difference = evaluate_second_pass(
            resolved, target_ids, common_hashes, args.truth_atol
        )
        common_rows = len(manifest)
        common_targets = int(manifest["target_id"].nunique())
        common_horizons = tuple(int(value) for value in sorted(manifest["horizon"].unique()))
        if not args.allow_incomplete_common and (common_targets != EXPECTED_TARGETS or common_horizons != EXPECTED_HORIZONS):
            raise ValueError(
                f"Common intersection is incomplete: targets={common_targets}, horizons={common_horizons}; "
                "expected 50 targets and horizons 1..8"
            )
        coverage = pd.DataFrame(coverage_rows)
        coverage["common_rows"] = common_rows
        coverage["excluded_rows"] = coverage["valid_unique_rows"] - common_rows
        coverage["common_fraction"] = common_rows / coverage["valid_unique_rows"]
        coverage.to_csv(output_dir / "sample_coverage_report.csv", index=False)
        manifest.to_csv(output_dir / "common_sample_manifest.csv", index=False)

        metrics = metrics[[
            "method_key", "type", "method",
            "pooled_mae", "pooled_rmse", "pooled_r2",
            "macro_mae", "macro_rmse", "macro_r2",
            "n_records", "n_targets", "n_origins", "macro_r2_valid_targets",
        ]]
        paired = paired_bootstrap(stationwise, args.bootstrap_reps, args.seed)
        table_iii = table_from_metrics(metrics, TABLE_III_ORDER)
        table_iv = table_from_metrics(metrics, TABLE_IV_ORDER)

        metrics.to_csv(output_dir / "all_methods_common_metrics.csv", index=False)
        stationwise.to_csv(output_dir / "stationwise_metrics.csv", index=False)
        horizonwise.to_csv(output_dir / "horizonwise_metrics.csv", index=False)
        paired.to_csv(output_dir / "paired_graph_field_comparison.csv", index=False)
        table_iii.to_csv(output_dir / "table_III_recomputed.csv", index=False)
        table_iv.to_csv(output_dir / "table_IV_recomputed.csv", index=False)

        rmse_order = metrics.sort_values("pooled_rmse")["method_key"].tolist()
        r2_order = metrics.sort_values("pooled_r2", ascending=False)["method_key"].tolist()
        ranking_consistent = rmse_order == r2_order
        if not ranking_consistent:
            raise AssertionError("RMSE and R2 rankings disagree despite a shared y_true vector")
        notes = [
            f"权威 target list：`{target_path}`（{len(target_ids)} 站）。",
            f"严格公共交集：{common_rows:,} 条记录，{common_targets} 个 target，horizons={list(common_horizons)}。",
            f"各方法公共交集中的 y_true 最大绝对差：{max_truth_difference:.9g}（阈值 {args.truth_atol:g}）。",
            "每个方法在交集前后都通过 `(target_id, issue_time, horizon)` 唯一性检查。",
            (
                "各方法交集前的有效记录数范围为 "
                f"{int(coverage['valid_unique_rows'].min()):,}–{int(coverage['valid_unique_rows'].max()):,}；"
                "差异来自各原始 evaluation detail 的 test 记录覆盖和 finite/valid 状态，"
                "逐方法排除数见 sample_coverage_report.csv。"
            ),
            "所有方法使用同一 y_true 后，RMSE 升序与 R² 降序完全一致。",
            "因此旧结果中 PredictedGraph RMSE=0.0990、R²=0.7990 与 SatPV-Field RMSE=0.0999、R²=0.8266 的交叉排序不能来自同一真值样本上的 pooled 公式；它至少涉及样本集合或聚合口径差异。",
            "最终图表示/场表示结论请以 paired_graph_field_comparison.csv 的 station-paired 差值和 bootstrap CI 为准。",
        ]
        notes.extend(paired_conclusion_lines(paired))
        write_audit(output_dir, coverage, "PASS", notes, metrics, paired)

        print("==== Unified common-intersection evaluation ====")
        print(f"targets={common_targets} horizons={list(common_horizons)} common_rows={common_rows:,}")
        print(metrics[["method", "pooled_mae", "pooled_rmse", "pooled_r2"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("output:", output_dir)
    except Exception as exc:
        coverage = pd.DataFrame(coverage_rows)
        if not coverage.empty:
            coverage.to_csv(output_dir / "sample_coverage_report.csv", index=False)
        notes = [
            f"权威 target list：`{target_path}`（{len(target_ids)} 站）。",
            f"评估中止：`{type(exc).__name__}: {exc}`",
            "没有生成可用于论文替换的重算指标；请修复输入后重跑。",
        ]
        write_audit(output_dir, coverage, "FAILED DURING VALIDATION", notes)
        raise


if __name__ == "__main__":
    main()
