#!/usr/bin/env python3
"""Report Stage-4 5x5 PV-field patch encoder ablation results.

This script is intentionally read-only with respect to model outputs. It scans
Stage-4 run directories for per-station field-patch prediction detail files and
reports whether the available 5x5 patch-guided station models use an MLP, CNN,
or another patch encoder. It is useful for checking whether an MLP-vs-CNN patch
encoder ablation has already been run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
AGG_NAME = "per_station_temporal_swr_anchor04l_field_patch_aggregate.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=RUNS_DIR / "report_stage4_patch_encoder_ablation",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--horizon",
        type=str,
        default="all",
        help="Use 'all' for all horizons or h1...h8 for one horizon.",
    )
    parser.add_argument(
        "--include_patch_size_runs",
        action="store_true",
        help="Also include patch-size ablation folders. By default they are skipped.",
    )
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(item, new_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            out.update(flatten_json(item, f"{prefix}[{idx}]"))
    else:
        out[prefix] = value
    return out


def infer_patch_size(path: Path, meta: dict[str, Any]) -> int | None:
    text = str(path).lower()
    for pattern in [r"patch[_-]?(\d)x\1", r"patch[_-]?(\d+)x\1", r"patch[_-]?(\d+)"]:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    flat = flatten_json(meta)
    for key, value in flat.items():
        if "patch_size" in key.lower():
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    if "05l_per_station_temporal_stage3_field_patch" in text:
        return 5
    return None


def infer_encoder(path: Path, meta: dict[str, Any]) -> str:
    text_items = [str(path).lower()]
    flat = flatten_json(meta)
    for key, value in flat.items():
        joined = f"{key}={value}".lower()
        if any(token in joined for token in ["encoder", "field_mlp", "field_cnn", "patch_mlp", "patch_cnn", "cnn", "mlp"]):
            text_items.append(joined)
    text = " ".join(text_items)
    if any(token in text for token in ["patch_cnn", "field_cnn", "cnn_patch", "conv_patch", "conv2d_patch"]):
        return "CNN"
    if any(token in text for token in ["patch_mlp", "field_mlp", "mlp_patch"]):
        return "MLP"
    if "05l_per_station_temporal_stage3_field_patch" in text:
        return "MLP"
    if "agcrn_fgmsf_patch" in text:
        return "MLP"
    if "cnn" in text:
        return "CNN"
    if "mlp" in text:
        return "MLP"
    return "unknown"


def find_meta(run_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for name in [
        "run_config.json",
        "config.json",
        "args.json",
        "field_patch_stats.json",
        "normalization_stats.json",
        "patch_size_eval_meta.json",
    ]:
        path = run_dir / name
        if path.exists():
            value = read_json(path)
            if value:
                meta[name] = value
    for path in list((run_dir / "per_station").glob("*/normalization_stats.json")):
        value = read_json(path)
        if value:
            meta[str(path.relative_to(run_dir))] = value
            break
    return meta


def is_patch_size_path(path: Path) -> bool:
    text = str(path).lower()
    return "ablation_pv_field_patch_size" in text or "patch_size" in text or re.search(r"patch_[137]x[137]", text) is not None


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if not args.runs_dir.exists():
        return []
    paths = sorted(list(args.runs_dir.rglob(DETAIL_NAME)) + list(args.runs_dir.rglob(AGG_NAME)))
    if args.include_patch_size_runs:
        return sorted({path.parent for path in paths})
    return sorted({path.parent for path in paths if not is_patch_size_path(path)})


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y_pred - y_true) ** 2)) / denom


def metrics_from_detail(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(path, dtype={"station_sn": str})
    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str) == str(args.split)].copy()
    if str(args.horizon) != "all" and "horizon" in df.columns:
        df = df[df["horizon"].astype(str) == str(args.horizon)].copy()
    if df.empty:
        raise ValueError(f"empty detail after filtering split={args.split}, horizon={args.horizon}")
    pred = df["pred_pv_norm"].to_numpy(np.float64)
    true = df["true_pv_norm"].to_numpy(np.float64)
    valid = np.isfinite(pred) & np.isfinite(true)
    df = df.loc[valid].copy()
    pred = pred[valid]
    true = true[valid]
    err = pred - true
    row: dict[str, Any] = {
        "n_points": int(len(df)),
        "n_stations": int(df["station_sn"].nunique()) if "station_sn" in df.columns else np.nan,
        "mae_pv_norm": float(np.mean(np.abs(err))),
        "rmse_pv_norm": float(math.sqrt(float(np.mean(err * err)))),
        "r2_pv_norm": r2_score(true, pred),
    }
    if "large_change" in df.columns:
        large = df["large_change"].astype(bool).to_numpy()
        row["large_change_mae_pv_norm"] = float(np.mean(np.abs(err[large]))) if bool(np.any(large)) else float("nan")
        row["large_change_n"] = int(np.sum(large))
    if "field_patch_support_mean" in df.columns:
        row["field_patch_support_mean"] = float(pd.to_numeric(df["field_patch_support_mean"], errors="coerce").mean())
    return row


def metrics_from_aggregate(path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    agg_path = path.parent / AGG_NAME
    if not agg_path.exists():
        return None
    df = pd.read_csv(agg_path)
    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str) == str(args.split)].copy()
    if "horizon" in df.columns:
        df = df[df["horizon"].astype(str) == str(args.horizon)].copy()
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {
        "n_points": np.nan,
        "n_stations": row.get("n_stations", np.nan),
        "mae_pv_norm": row.get("station_mean_mae_pv_norm", np.nan),
        "rmse_pv_norm": row.get("station_mean_rmse_pv_norm", np.nan),
        "r2_pv_norm": row.get("station_mean_r2_pv_norm", np.nan),
        "large_change_mae_pv_norm": row.get("station_mean_high_change_mae_pv_norm", np.nan),
        "field_patch_support_mean": row.get("field_patch_support_mean", np.nan),
        "metric_source": "aggregate_station_macro",
    }


def collect(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(args):
        detail_path = run_dir / DETAIL_NAME
        agg_path = run_dir / AGG_NAME
        meta = find_meta(run_dir)
        if detail_path.exists():
            try:
                metric = metrics_from_detail(detail_path, args)
                metric["metric_source"] = "eval_detail_pooled"
            except Exception as exc:
                fallback = metrics_from_aggregate(detail_path, args)
                if fallback is None:
                    print(f"[skip] {detail_path}: {exc}", flush=True)
                    continue
                metric = fallback
        else:
            fallback = metrics_from_aggregate(agg_path, args)
            if fallback is None:
                print(f"[skip] {run_dir}: no readable detail or aggregate metrics", flush=True)
                continue
            metric = fallback
        patch_size = infer_patch_size(run_dir, meta)
        encoder = infer_encoder(run_dir, meta)
        rows.append(
            {
                "run_name": run_dir.name,
                "patch_size": patch_size,
                "patch_encoder": encoder,
                **metric,
                "result_csv": str(detail_path if detail_path.exists() else agg_path),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    if abs(x - round(x)) < 1e-12 and abs(x) >= 1:
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def print_table(title: str, table: pd.DataFrame, digits: int) -> None:
    print(f"\n==== {title} ====")
    if table.empty:
        print("(empty)")
        return
    cols = [
        "run_name",
        "patch_size",
        "patch_encoder",
        "n_points",
        "n_stations",
        "mae_pv_norm",
        "rmse_pv_norm",
        "r2_pv_norm",
        "large_change_mae_pv_norm",
        "field_patch_support_mean",
        "metric_source",
    ]
    cols = [col for col in cols if col in table.columns]
    print(table[cols].to_string(index=False, formatters={col: (lambda x, d=digits: fmt(x, d)) for col in cols}))


def print_latex_rows(table: pd.DataFrame, digits: int) -> None:
    print("\n==== LaTeX-ready rows ====")
    if table.empty:
        print("(empty)")
        return
    for _, row in table.iterrows():
        label = str(row["patch_encoder"])
        if pd.notna(row.get("patch_size")):
            label = f"{label} ({int(row['patch_size'])} x {int(row['patch_size'])})"
        print(
            f"{label} & "
            f"{fmt(row.get('mae_pv_norm'), digits)} & "
            f"{fmt(row.get('rmse_pv_norm'), digits)} & "
            f"{fmt(row.get('r2_pv_norm'), digits)} & "
            f"{fmt(row.get('large_change_mae_pv_norm'), digits)} \\\\"
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args)
    if not summary.empty:
        summary = summary.sort_values(["patch_size", "patch_encoder", "run_name"], na_position="last").reset_index(drop=True)
    out_path = out_dir / "stage4_patch_encoder_ablation_summary.csv"
    summary.to_csv(out_path, index=False)

    print("==== Stage 4 patch-encoder ablation scan ====")
    print(f"runs_dir: {args.runs_dir}")
    print(f"split/horizon: {args.split}/{args.horizon}")
    print(f"found_detail_runs: {len(summary)}")
    print(f"saved: {out_path}")

    print_table("All discovered field-patch station runs", summary, int(args.digits))

    if summary.empty:
        print("\nNo field-patch eval-detail files were found under the scanned runs directory.")
        return

    focused = summary.copy()
    if "patch_size" in focused.columns:
        focused = focused[focused["patch_size"].fillna(5).astype(float) == 5].copy()
    print_table("5x5 patch-encoder candidates", focused, int(args.digits))
    print_latex_rows(focused, int(args.digits))

    encoders = sorted(set(focused["patch_encoder"].dropna().astype(str)))
    print("\n==== Interpretation hint ====")
    if "CNN" in encoders and "MLP" in encoders:
        print("Both MLP and CNN 5x5 patch-encoder results were found.")
    elif "MLP" in encoders and "CNN" not in encoders:
        print("Only the MLP/flatten 5x5 patch encoder was found. I did not find an existing CNN patch-encoder result in this runs directory.")
    elif "CNN" in encoders and "MLP" not in encoders:
        print("Only a CNN-like 5x5 patch encoder was found. Check whether the standard MLP run path is included in --runs_dir.")
    else:
        print("Patch encoder type could not be inferred from paths/configs. Inspect the detail_csv paths above.")


if __name__ == "__main__":
    main()
