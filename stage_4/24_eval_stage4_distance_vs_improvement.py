"""Analyze nearest-source-station distance versus Stage 4 guidance improvement.

This script reuses the common-subset station-level improvement table produced by
22_collect_stage4_all_results.py, so the analysis is aligned with the reported
Stage 4 main and patch-size tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_paths import (
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)


DEFAULT_COLLECT_DIR = RUNS_DIR / "22_collect_stage4_all_results"
DEFAULT_OUT_DIR = RUNS_DIR / "24_distance_vs_improvement"
DEFAULT_METHOD_KEYS = (
    "future_swr_patch,"
    "historical_pv_field_patch,"
    "nearest3_station_pv,"
    "fg_msf"
)
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
FIELD_PATCH_DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
NEAREST3_DETAIL_NAME = "nearest3_station_pv_eval_detail.csv"

METHOD_STYLE = {
    "fg_msf": {"label": "FG-MSF", "short_label": "FG-MSF", "color": "#d9483b", "marker": "s"},
    "future_swr_patch": {"label": "Future SWR-field guidance", "short_label": "SWR", "color": "#2a9d8f", "marker": "D"},
    "historical_pv_field_patch": {"label": "Historical PV-field guidance", "short_label": "Hist. PV", "color": "#7aa34d", "marker": "^"},
    "nearest3_station_pv": {"label": "3-NN PV guidance", "short_label": "3-NN PV", "color": "#8a5fbf", "marker": "o"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether target-station distance to source stations explains Stage 4 field-guidance gains."
    )
    parser.add_argument("--collect_dir", type=Path, default=DEFAULT_COLLECT_DIR)
    parser.add_argument("--station_improvement_csv", type=Path, default=None)
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--target_csv", type=Path, default=None)
    parser.add_argument("--source_csv", type=Path, default=None)
    parser.add_argument("--method_keys", type=str, default=DEFAULT_METHOD_KEYS)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
        help="Fallback reference detail CSV used to reconstruct missing guidance-variant station gains.",
    )
    parser.add_argument(
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / FIELD_PATCH_DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / FIELD_PATCH_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / FIELD_PATCH_DETAIL_NAME,
    )
    parser.add_argument(
        "--nearest3_station_pv_detail_csv",
        type=Path,
        default=RUNS_DIR / "21_nearest3_station_pv_baseline_seed2026" / NEAREST3_DETAIL_NAME,
    )
    parser.add_argument("--reference_label", type=str, default="Temporal baseline")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--metric", choices=["MAE_gain", "MAE_gain_pct", "RMSE_gain", "R2_gain"], default="MAE_gain")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}. Available columns: {list(df.columns)}")


def resolve_target_csv(args: argparse.Namespace) -> Path:
    candidates = []
    if args.target_csv is not None:
        candidates.append(args.target_csv)
    candidates.extend(
        [
            args.selection_dir / "target_stations.csv",
            args.selection_dir / "selected_target_stations.csv",
            args.target_run_dir / "target_stations.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find target station CSV. Tried:\n" + "\n".join(str(p) for p in candidates))


def resolve_source_csv(args: argparse.Namespace) -> Path:
    candidates = []
    if args.source_csv is not None:
        candidates.append(args.source_csv)
    candidates.extend(
        [
            args.target_run_dir / "target_excluded_station_split.csv",
            args.target_run_dir / "station_split.csv",
            args.selection_dir / "target_excluded_station_split.csv",
            args.selection_dir / "station_split.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find source station split CSV. Tried:\n" + "\n".join(str(p) for p in candidates))


def read_station_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["station_sn", "lat", "lon"], path)
    out = df[["station_sn", "lat", "lon"]].copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates("station_sn")
    if out.empty:
        raise ValueError(f"No valid station coordinates in {path}")
    return out


def read_source_station_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["station_sn", "lat", "lon"], path)
    if "role" in df.columns:
        role = df["role"].astype(str).str.lower()
        source = df[role.isin(["source", "train", "train_source", "field_source"])].copy()
        if source.empty:
            source = df[~role.isin(["target", "heldout", "held-out", "test_target"])].copy()
    elif "is_target" in df.columns:
        source = df[~df["is_target"].astype(bool)].copy()
    else:
        source = df.copy()
    out = source[["station_sn", "lat", "lon"]].copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates("station_sn")
    if out.empty:
        raise ValueError(f"No valid source-station coordinates in {path}")
    return out


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    phi1 = np.deg2rad(lat1)[:, None]
    phi2 = np.deg2rad(lat2)[None, :]
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2)[None, :] - np.deg2rad(lon1)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_source_distance(targets: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    dist = haversine_km(
        targets["lat"].to_numpy(float),
        targets["lon"].to_numpy(float),
        sources["lat"].to_numpy(float),
        sources["lon"].to_numpy(float),
    )
    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(dist.shape[0]), nearest_idx]
    out = targets.copy()
    out["nearest_source_station_sn"] = sources.iloc[nearest_idx]["station_sn"].to_numpy()
    out["nearest_source_distance_km"] = nearest_dist
    return out


def add_distance_bins(station_distance: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = station_distance.copy()
    n_unique = int(out["nearest_source_distance_km"].nunique())
    n_bins = max(1, min(int(n_bins), n_unique))
    if n_bins == 1:
        out["distance_bin_order"] = 1
        out["distance_bin"] = "all"
        return out

    ranks = out["nearest_source_distance_km"].rank(method="first")
    out["distance_bin_order"] = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop").astype(int) + 1
    labels = {}
    for order, g in out.groupby("distance_bin_order"):
        labels[int(order)] = f"Q{int(order)} ({g['nearest_source_distance_km'].min():.1f}-{g['nearest_source_distance_km'].max():.1f} km)"
    out["distance_bin"] = out["distance_bin_order"].map(labels)
    return out


def load_station_improvement(args: argparse.Namespace) -> pd.DataFrame:
    path = args.station_improvement_csv
    if path is None:
        path = args.collect_dir / "stage4_all_results_improvement_station_vs_reference.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing station improvement table: {path}\n"
            "Run 22_collect_stage4_all_results.py first, e.g.:\n"
            "bash stage_4/run_stage4_collect_all_results.sh"
        )
    df = pd.read_csv(path)
    require_columns(df, ["method_key", "station_sn", "Method", "MAE", "reference_MAE", "MAE_gain", "MAE_gain_pct"], path)
    df["method_key"] = df["method_key"].astype(str)
    df["station_sn"] = df["station_sn"].astype(str)
    available = sorted(df["method_key"].dropna().astype(str).unique().tolist())
    print(f"[read] station improvement table: {path}", flush=True)
    print(f"[read] available method_keys: {', '.join(available)}", flush=True)
    return df


def _read_detail(path: Path, split: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"station_sn": str}, low_memory=False)
    required = ["station_sn", "eval_split", "issue_time", "h", "pred_pv_norm", "true_pv_norm"]
    require_columns(df, required, path)
    out = df[df["eval_split"].astype(str) == str(split)].copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["pred_pv_norm"] = pd.to_numeric(out["pred_pv_norm"], errors="coerce")
    out["true_pv_norm"] = pd.to_numeric(out["true_pv_norm"], errors="coerce")
    return out


def _r2_score(true: pd.Series, pred: pd.Series) -> float:
    y = pd.to_numeric(true, errors="coerce").astype(float)
    p = pd.to_numeric(pred, errors="coerce").astype(float)
    valid = y.notna() & p.notna()
    y = y[valid].to_numpy(float)
    p = p[valid].to_numpy(float)
    if y.size == 0:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def _station_metric_rows(aligned: pd.DataFrame, method_key: str, method_label: str) -> pd.DataFrame:
    rows = []
    for sn, g in aligned.groupby("station_sn", sort=True):
        true = g["true_pv_norm"].astype(float)
        pred = g["pred_candidate"].astype(float)
        ref = g["pred_reference"].astype(float)
        cand_err = pred - true
        ref_err = ref - true
        mae = float(cand_err.abs().mean())
        ref_mae = float(ref_err.abs().mean())
        rmse = float(np.sqrt(np.mean(np.square(cand_err))))
        ref_rmse = float(np.sqrt(np.mean(np.square(ref_err))))
        r2 = _r2_score(true, pred)
        ref_r2 = _r2_score(true, ref)
        rows.append(
            {
                "method_key": str(method_key),
                "station_sn": str(sn),
                "Method": str(method_label),
                "MAE": mae,
                "reference_MAE": ref_mae,
                "MAE_gain": ref_mae - mae,
                "MAE_gain_pct": (ref_mae - mae) / ref_mae * 100.0 if ref_mae > 0 else float("nan"),
                "RMSE": rmse,
                "reference_RMSE": ref_rmse,
                "RMSE_gain": ref_rmse - rmse,
                "R2": r2,
                "reference_R2": ref_r2,
                "R2_gain": r2 - ref_r2,
                "n_points": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def build_missing_guidance_improvements(args: argparse.Namespace, missing_keys: list[str]) -> pd.DataFrame:
    detail_paths = {
        "fg_msf": args.future_field_detail_csv,
        "future_swr_patch": args.future_swr_patch_detail_csv,
        "historical_pv_field_patch": args.historical_field_detail_csv,
        "nearest3_station_pv": args.nearest3_station_pv_detail_csv,
    }
    wanted = [key for key in missing_keys if key in detail_paths]
    if not wanted:
        return pd.DataFrame()
    try:
        baseline = _read_detail(args.baseline_detail_csv, str(args.split))
    except FileNotFoundError as exc:
        print(f"[warn] cannot reconstruct missing guidance variants; missing baseline detail: {exc}", flush=True)
        return pd.DataFrame()

    base_cols = ["station_sn", "eval_split", "issue_time", "h", "true_pv_norm", "pred_pv_norm"]
    baseline = baseline[base_cols].rename(
        columns={
            "true_pv_norm": "true_reference",
            "pred_pv_norm": "pred_reference",
        }
    )
    frames = []
    for key in wanted:
        path = Path(detail_paths[key])
        try:
            candidate = _read_detail(path, str(args.split))
        except FileNotFoundError:
            print(f"[warn] cannot reconstruct {key}; missing detail CSV: {path}", flush=True)
            continue
        label = METHOD_STYLE.get(key, {}).get("label", key)
        candidate = candidate[base_cols].rename(
            columns={
                "true_pv_norm": "true_pv_norm",
                "pred_pv_norm": "pred_candidate",
            }
        )
        aligned = baseline.merge(candidate, on=["station_sn", "eval_split", "issue_time", "h"], how="inner")
        if aligned.empty:
            print(f"[warn] cannot reconstruct {key}; no aligned rows with baseline", flush=True)
            continue
        truth_diff = (aligned["true_reference"].astype(float) - aligned["true_pv_norm"].astype(float)).abs()
        max_truth_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
        if np.isfinite(max_truth_diff) and max_truth_diff > 1.0e-5:
            print(f"[warn] {key}: max truth mismatch after alignment = {max_truth_diff:.6g}", flush=True)
        rows = _station_metric_rows(aligned, key, label)
        print(f"[rebuild] {key}: {len(aligned)} aligned points, {len(rows)} stations from {path}", flush=True)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_relationship(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (key, method), g in df.groupby(["method_key", "Method"], sort=False):
        x = g["nearest_source_distance_km"].astype(float)
        y = g[metric].astype(float)
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        slope = np.nan
        intercept = np.nan
        if len(x) >= 2 and x.nunique() >= 2:
            slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), deg=1)
        far = g[g["distance_bin_order"].eq(g["distance_bin_order"].max())]
        near = g[g["distance_bin_order"].eq(g["distance_bin_order"].min())]
        rows.append(
            {
                "method_key": key,
                "Method": method,
                "n_stations": int(g["station_sn"].nunique()),
                "mean_nearest_source_distance_km": float(x.mean()) if len(x) else np.nan,
                f"mean_{metric}": float(y.mean()) if len(y) else np.nan,
                f"median_{metric}": float(y.median()) if len(y) else np.nan,
                "win_rate_pct": float(100.0 * (g["MAE_gain"] > 0).mean()),
                f"near_bin_mean_{metric}": float(near[metric].mean()) if not near.empty else np.nan,
                f"far_bin_mean_{metric}": float(far[metric].mean()) if not far.empty else np.nan,
                "pearson_corr_distance_vs_metric": float(x.corr(y, method="pearson")) if len(x) >= 2 else np.nan,
                "spearman_corr_distance_vs_metric": float(x.corr(y, method="spearman")) if len(x) >= 2 else np.nan,
                "linear_slope_per_km": float(slope),
                "linear_intercept": float(intercept),
                "far_bin_still_improves": bool(far["MAE_gain"].mean() > 0.0) if not far.empty else False,
            }
        )
    return pd.DataFrame(rows)


def build_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["method_key", "Method", "distance_bin_order", "distance_bin"]
    for cols, g in df.groupby(group_cols, sort=True):
        key, method, order, label = cols
        rows.append(
            {
                "method_key": key,
                "Method": method,
                "distance_bin_order": int(order),
                "distance_bin": label,
                "n_stations": int(g["station_sn"].nunique()),
                "mean_distance_km": float(g["nearest_source_distance_km"].mean()),
                "mean_MAE": float(g["MAE"].mean()),
                "mean_reference_MAE": float(g["reference_MAE"].mean()),
                "mean_MAE_gain": float(g["MAE_gain"].mean()),
                "mean_MAE_gain_pct": float(g["MAE_gain_pct"].mean()),
                "mean_RMSE_gain": float(g["RMSE_gain"].mean()) if "RMSE_gain" in g.columns else np.nan,
                "mean_R2_gain": float(g["R2_gain"].mean()) if "R2_gain" in g.columns else np.nan,
                "win_rate_pct": float(100.0 * (g["MAE_gain"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["method_key", "distance_bin_order"]).reset_index(drop=True)


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, ci: float, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or n_bootstrap <= 0:
        return float(values.mean()), float(values.mean())
    boot = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def paired_test_rows(df: pd.DataFrame, n_bootstrap: int, ci: float, seed: int) -> pd.DataFrame:
    try:
        from scipy import stats
    except Exception:
        stats = None

    rows: list[dict[str, Any]] = []
    for idx, ((key, method), g) in enumerate(df.groupby(["method_key", "Method"], sort=False)):
        ref = pd.to_numeric(g["reference_MAE"], errors="coerce").to_numpy(float)
        mae = pd.to_numeric(g["MAE"], errors="coerce").to_numpy(float)
        gain = ref - mae
        valid = np.isfinite(ref) & np.isfinite(mae) & np.isfinite(gain)
        ref = ref[valid]
        mae = mae[valid]
        gain = gain[valid]
        rng = np.random.default_rng(int(seed) + idx)
        ci_low, ci_high = bootstrap_mean_ci(gain, n_bootstrap, ci, rng)

        t_stat = np.nan
        t_p_one_sided = np.nan
        wilcoxon_stat = np.nan
        wilcoxon_p_one_sided = np.nan
        if stats is not None and gain.size >= 2:
            try:
                t_res = stats.ttest_rel(ref, mae, alternative="greater", nan_policy="omit")
            except TypeError:
                t_res = stats.ttest_rel(ref, mae, nan_policy="omit")
            t_stat = float(t_res.statistic)
            t_p_one_sided = float(t_res.pvalue)
            try:
                w_res = stats.wilcoxon(ref, mae, alternative="greater", zero_method="wilcox")
                wilcoxon_stat = float(w_res.statistic)
                wilcoxon_p_one_sided = float(w_res.pvalue)
            except ValueError:
                pass

        rows.append(
            {
                "method_key": key,
                "Method": method,
                "n_stations": int(gain.size),
                "n_improved_stations": int((gain > 0).sum()),
                "win_rate_pct": float(100.0 * (gain > 0).mean()) if gain.size else np.nan,
                "mean_MAE_gain": float(gain.mean()) if gain.size else np.nan,
                "median_MAE_gain": float(np.median(gain)) if gain.size else np.nan,
                f"mean_MAE_gain_ci{int(ci * 100)}_low": ci_low,
                f"mean_MAE_gain_ci{int(ci * 100)}_high": ci_high,
                "paired_t_stat_ref_gt_method": t_stat,
                "paired_t_p_one_sided": t_p_one_sided,
                "wilcoxon_stat_ref_gt_method": wilcoxon_stat,
                "wilcoxon_p_one_sided": wilcoxon_p_one_sided,
                "cohen_dz_gain": float(gain.mean() / gain.std(ddof=1)) if gain.size >= 2 and gain.std(ddof=1) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def y_label_for(metric: str, reference_label: str) -> str:
    if metric == "MAE_gain":
        return f"MAE gain vs {reference_label} (p.u.)"
    if metric == "MAE_gain_pct":
        return f"MAE gain vs {reference_label} (%)"
    if metric == "RMSE_gain":
        return f"RMSE gain vs {reference_label} (p.u.)"
    return f"R2 gain vs {reference_label}"


def plot_scatter(df: pd.DataFrame, metric: str, reference_label: str, out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, g in df.groupby("method_key", sort=False):
        style = METHOD_STYLE.get(key, {})
        label = style.get("label") or str(g["Method"].iloc[0])
        color = style.get("color", "#4d4d4d")
        marker = style.get("marker", "o")
        x = g["nearest_source_distance_km"].to_numpy(float)
        y = g[metric].to_numpy(float)
        ax.scatter(x, y, s=42, marker=marker, color=color, alpha=0.82, edgecolor="white", linewidth=0.45, label=label)
        if len(g) >= 3 and np.unique(x).size >= 2 and np.isfinite(y).all():
            slope, intercept = np.polyfit(x, y, deg=1)
            xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xx, slope * xx + intercept, color=color, linewidth=1.6, alpha=0.85)
    ax.axhline(0.0, color="#6f7b83", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Distance to nearest source station (km)", fontsize=11)
    ax.set_ylabel(y_label_for(metric, reference_label), fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, color="#d6dde2", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.8, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"distance_vs_{metric}_scatter.png", dpi=dpi)
    fig.savefig(out_dir / f"distance_vs_{metric}_scatter.pdf")
    plt.close(fig)


def plot_bin_summary(bin_summary: pd.DataFrame, metric: str, reference_label: str, out_dir: Path, dpi: int) -> None:
    if bin_summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_col = f"mean_{metric}"
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for key, g in bin_summary.groupby("method_key", sort=False):
        style = METHOD_STYLE.get(key, {})
        label = style.get("label") or str(g["Method"].iloc[0])
        color = style.get("color", "#4d4d4d")
        marker = style.get("marker", "o")
        ax.plot(
            g["distance_bin_order"],
            g[y_col],
            color=color,
            marker=marker,
            markersize=6.0,
            linewidth=2.0,
            label=label,
        )
    orders = sorted(bin_summary["distance_bin_order"].unique())
    labels = [
        str(bin_summary.loc[bin_summary["distance_bin_order"].eq(order), "distance_bin"].iloc[0]).replace(" ", "\n", 1)
        for order in orders
    ]
    ax.set_xticks(orders)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.0, color="#6f7b83", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Nearest-source-distance bin", fontsize=11)
    ax.set_ylabel(y_label_for(metric, reference_label), fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", color="#d6dde2", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.8, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"distance_bin_{metric}.png", dpi=dpi)
    fig.savefig(out_dir / f"distance_bin_{metric}.pdf")
    plt.close(fig)


def plot_station_gain_distribution(df: pd.DataFrame, reference_label: str, out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = [(key, g.copy()) for key, g in df.groupby("method_key", sort=False)]
    if not grouped:
        return
    labels = []
    values = []
    colors = []
    for key, g in grouped:
        style = METHOD_STYLE.get(key, {})
        labels.append(style.get("short_label") or style.get("label") or str(g["Method"].iloc[0]))
        values.append(pd.to_numeric(g["MAE_gain"], errors="coerce").dropna().to_numpy(float))
        colors.append(style.get("color", "#4d4d4d"))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)

    box = ax.boxplot(
        values,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#20272b", "linewidth": 1.5},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(2026)
    for i, (vals, color) in enumerate(zip(values, colors), start=1):
        if vals.size == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.035, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=18, color=color, alpha=0.72, edgecolor="white", linewidth=0.3)

    ax.axhline(0.0, color="#6f7b83", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("MAE gain (p.u.)", fontsize=11)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", color="#d6dde2", linewidth=0.8, alpha=0.75)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "station_wise_MAE_gain_distribution.png", dpi=dpi)
    fig.savefig(out_dir / "station_wise_MAE_gain_distribution.pdf")
    plt.close(fig)


def print_summary(summary: pd.DataFrame, metric: str) -> None:
    cols = [
        "Method",
        "n_stations",
        f"mean_{metric}",
        "win_rate_pct",
        f"near_bin_mean_{metric}",
        f"far_bin_mean_{metric}",
        "spearman_corr_distance_vs_metric",
        "linear_slope_per_km",
        "far_bin_still_improves",
    ]
    print("\n=== Distance vs improvement summary ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.4f}" if math.isfinite(float(val)) else "")
            else:
                cells.append(str(val))
        print("| " + " | ".join(cells) + " |")


def print_significance(table: pd.DataFrame, ci: float) -> None:
    cols = [
        "Method",
        "n_improved_stations",
        "win_rate_pct",
        "mean_MAE_gain",
        f"mean_MAE_gain_ci{int(ci * 100)}_low",
        f"mean_MAE_gain_ci{int(ci * 100)}_high",
        "paired_t_p_one_sided",
        "wilcoxon_p_one_sided",
        "cohen_dz_gain",
    ]
    print("\n=== Station-wise significance tests ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.4g}" if math.isfinite(float(val)) else "")
            else:
                cells.append(str(val))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    method_keys = [item.strip() for item in args.method_keys.split(",") if item.strip()]
    target_csv = resolve_target_csv(args)
    source_csv = resolve_source_csv(args)
    targets = read_station_table(target_csv)
    sources = read_source_station_table(source_csv)
    station_distance = nearest_source_distance(targets, sources)
    station_distance = add_distance_bins(station_distance, args.n_bins)

    improvement = load_station_improvement(args)
    available_keys = set(improvement["method_key"].dropna().astype(str).unique().tolist())
    missing_keys = [key for key in method_keys if key not in available_keys]
    if missing_keys:
        print(f"[warn] requested method_keys not found in station-improvement table: {', '.join(missing_keys)}", flush=True)
        rebuilt = build_missing_guidance_improvements(args, missing_keys)
        if not rebuilt.empty:
            improvement = pd.concat([improvement, rebuilt], ignore_index=True)
            rebuilt_keys = sorted(rebuilt["method_key"].dropna().astype(str).unique().tolist())
            print(f"[rebuild] appended method_keys: {', '.join(rebuilt_keys)}", flush=True)
        available_keys = set(improvement["method_key"].dropna().astype(str).unique().tolist())
        still_missing = [key for key in method_keys if key not in available_keys]
        if still_missing:
            print(f"[warn] requested method_keys still unavailable and will be skipped: {', '.join(still_missing)}", flush=True)
    improvement = improvement[improvement["method_key"].isin(method_keys)].copy()
    if improvement.empty:
        raise ValueError(f"No requested method_keys found in station improvement table: {method_keys}")
    improvement["method_key"] = pd.Categorical(improvement["method_key"], categories=method_keys, ordered=True)
    improvement = improvement.sort_values(["method_key", "station_sn"]).copy()
    improvement["method_key"] = improvement["method_key"].astype(str)

    merged = improvement.merge(
        station_distance[
            [
                "station_sn",
                "lat",
                "lon",
                "nearest_source_station_sn",
                "nearest_source_distance_km",
                "distance_bin_order",
                "distance_bin",
            ]
        ],
        on="station_sn",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No stations matched between improvement table and target-station coordinate table.")

    summary = summarize_relationship(merged, args.metric)
    bin_summary = build_bin_summary(merged)
    significance = paired_test_rows(merged, args.n_bootstrap, args.ci, args.seed)

    station_distance.to_csv(out_dir / "target_station_nearest_source_distance.csv", index=False)
    merged.to_csv(out_dir / "distance_vs_improvement_station.csv", index=False)
    summary.to_csv(out_dir / "distance_vs_improvement_summary.csv", index=False)
    bin_summary.to_csv(out_dir / "distance_bin_improvement_summary.csv", index=False)
    significance.to_csv(out_dir / "station_wise_significance_tests.csv", index=False)

    plot_scatter(merged, args.metric, args.reference_label, out_dir, args.dpi)
    plot_bin_summary(bin_summary, args.metric, args.reference_label, out_dir, args.dpi)
    plot_station_gain_distribution(merged, args.reference_label, out_dir, args.dpi)

    meta = {
        "target_csv": str(target_csv),
        "source_csv": str(source_csv),
        "collect_dir": str(args.collect_dir),
        "station_improvement_csv": str(args.station_improvement_csv or args.collect_dir / "stage4_all_results_improvement_station_vs_reference.csv"),
        "method_keys": method_keys,
        "metric": args.metric,
        "n_target_stations_with_distance": int(station_distance["station_sn"].nunique()),
        "n_rows": int(len(merged)),
        "n_stations": int(merged["station_sn"].nunique()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[read] target stations: {target_csv} rows={len(targets)}")
    print(f"[read] source stations: {source_csv} rows={len(sources)}")
    print(f"[write] {out_dir / 'distance_vs_improvement_station.csv'}")
    print(f"[write] {out_dir / 'distance_vs_improvement_summary.csv'}")
    print(f"[write] {out_dir / 'station_wise_significance_tests.csv'}")
    print(f"[write] {out_dir / f'distance_vs_{args.metric}_scatter.png'}")
    print(f"[write] {out_dir / 'station_wise_MAE_gain_distribution.png'}")
    print_summary(summary, args.metric)
    print_significance(significance, args.ci)


if __name__ == "__main__":
    main()
