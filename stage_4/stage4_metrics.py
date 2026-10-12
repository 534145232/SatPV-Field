"""Shared Stage 4 PV forecast metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    return true[valid], pred[valid]


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination, computed over the provided evaluation group."""
    true, pred = _finite_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - float(np.mean(true))) ** 2))
    if ss_tot <= 1.0e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def acc_from_installed_capacity(
    y_true_norm,
    y_pred_norm,
    installed_capacity=None,
) -> tuple[float, float, str]:
    """Return ACC=1-nRMSE using installed capacity.

    The Stage 4 targets are PV_norm = raw_power / (Capacity * capacity_scale_factor).
    If per-row installed capacity is present, this function explicitly keeps it
    in the calculation so the reported ACC is tied to the real capacity column.
    Algebraically that equals 1 - RMSE(PV_norm), but keeping the capacity column
    in the computation documents the installed-capacity normalization.
    """
    true, pred = _finite_pair(y_true_norm, y_pred_norm)
    if true.size == 0:
        return float("nan"), float("nan"), "empty"
    err_norm = pred - true
    if installed_capacity is not None:
        cap = np.asarray(installed_capacity, dtype=np.float64)
        cap_valid = np.isfinite(cap) & (cap > 0)
        pair_valid = np.isfinite(np.asarray(y_true_norm, dtype=np.float64)) & np.isfinite(np.asarray(y_pred_norm, dtype=np.float64)) & cap_valid
        if bool(np.any(pair_valid)):
            err_all = np.asarray(y_pred_norm, dtype=np.float64) - np.asarray(y_true_norm, dtype=np.float64)
            err_power = err_all[pair_valid] * cap[pair_valid]
            normalized = err_power / cap[pair_valid]
            nrmse = float(math.sqrt(float(np.mean(normalized**2))))
            return float(1.0 - nrmse), nrmse, "installed_capacity"
    nrmse = float(math.sqrt(float(np.mean(err_norm**2))))
    return float(1.0 - nrmse), nrmse, "pv_norm_fallback"


def regression_metric_extras(df: pd.DataFrame, pred_col: str, truth_col: str = "true_pv_norm") -> dict:
    cap_col = None
    for candidate in ("installed_capacity", "capacity_raw", "capacity_scale"):
        if candidate in df.columns:
            cap_col = candidate
            break
    acc, nrmse, source = acc_from_installed_capacity(
        df[truth_col].astype(float),
        df[pred_col].astype(float),
        df[cap_col].astype(float) if cap_col is not None else None,
    )
    return {
        "r2_pv_norm": r2_score(df[truth_col].astype(float), df[pred_col].astype(float)),
        "acc_installed_capacity": acc,
        "nrmse_installed_capacity": nrmse,
        "acc_capacity_source": source,
    }


def attach_capacity_from_station_meta(df: pd.DataFrame, detail_csv_path: str | Path | None = None) -> pd.DataFrame:
    """Attach real station capacity columns to an eval-detail table when available."""
    out = df.copy()
    if "capacity_scale" in out.columns and "capacity_raw" in out.columns:
        return out
    if "station_sn" not in out.columns:
        return out
    if detail_csv_path is None:
        return out
    meta_path = Path(detail_csv_path).parent / "station_load_meta.csv"
    if not meta_path.exists():
        return out
    meta = pd.read_csv(meta_path, dtype={"station_sn": str})
    desired = ["station_sn", "capacity_raw", "capacity_scale", "capacity_col"]
    keep = [c for c in desired if c in meta.columns and (c == "station_sn" or c not in out.columns)]
    if "station_sn" not in keep:
        return out
    meta = meta[keep].drop_duplicates("station_sn")
    out["station_sn"] = out["station_sn"].astype(str)
    out = out.merge(meta, on="station_sn", how="left")
    return out
