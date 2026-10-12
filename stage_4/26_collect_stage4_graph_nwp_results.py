"""Collect Stage 4 AGCRN and Graph WaveNet NWP-baseline results.

This is a lightweight companion to 22_collect_stage4_all_results.py for quickly
checking the two stronger graph baselines after training finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4_metrics import attach_capacity_from_station_meta, regression_metric_extras
from stage4_paths import PER_STATION_TEMPORAL_BASELINE_RUN_DIR, RUNS_DIR


KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"

METHOD_SPECS = [
    {
        "key": "temporal_baseline",
        "method": "BiLSTM + NWP",
        "type": "Station temporal",
        "path_arg": "temporal_detail_csv",
    },
    {
        "key": "agcrn_nwp",
        "method": "AGCRN + NWP",
        "type": "Implicit spatio-temporal",
        "path_arg": "agcrn_detail_csv",
    },
    {
        "key": "graph_wavenet_nwp",
        "method": "Graph WaveNet + NWP",
        "type": "Implicit spatio-temporal",
        "path_arg": "graph_wavenet_detail_csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--agcrn_detail_csv",
        type=Path,
        default=RUNS_DIR / "23_agcrn_nwp_baseline_seed2026" / "agcrn_nwp_eval_detail.csv",
    )
    parser.add_argument(
        "--graph_wavenet_detail_csv",
        type=Path,
        default=RUNS_DIR / "23_graph_wavenet_nwp_baseline_seed2026" / "graph_wavenet_nwp_eval_detail.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "26_collect_stage4_graph_nwp_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--align_mode", choices=["intersection", "reference", "none"], default="intersection")
    parser.add_argument("--reference_key", type=str, default="temporal_baseline")
    parser.add_argument("--include_temporal", type=int, default=1)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--strict_missing", type=int, default=1)
    return parser.parse_args()


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def read_detail(path: Path, spec: dict[str, str], split: str, strict: bool) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        msg = f"[skip] {spec['key']}: missing {path}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg, flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [*KEY_COLS, "horizon", "pred_pv_norm", "true_pv_norm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = attach_capacity_from_station_meta(df, path)
    out = out[out["eval_split"].astype(str) == str(split)].copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["horizon"] = out["horizon"].astype(str)
    out["pred_pv_norm"] = pd.to_numeric(out["pred_pv_norm"], errors="coerce")
    out["true_pv_norm"] = pd.to_numeric(out["true_pv_norm"], errors="coerce")
    if "large_change" in out.columns:
        out["large_change"] = coerce_bool(out["large_change"])
    out["method_key"] = spec["key"]
    out["Method"] = spec["method"]
    out["Type"] = spec["type"]
    print(f"[read] {spec['key']}: {path} rows={len(out)}", flush=True)
    return out


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEY_COLS].drop_duplicates().copy()


def align_details(details: dict[str, pd.DataFrame], mode: str, reference_key: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if mode == "none":
        keys = pd.concat([key_frame(df) for df in details.values()], ignore_index=True).drop_duplicates(KEY_COLS)
        return details, keys
    if mode == "reference":
        if reference_key not in details:
            raise ValueError(f"--align_mode reference requires {reference_key!r}")
        keys = key_frame(details[reference_key])
    else:
        keys = None
        for df in details.values():
            k = key_frame(df)
            keys = k if keys is None else keys.merge(k, on=KEY_COLS, how="inner")
    assert keys is not None
    out = {}
    for key, df in details.items():
        before = len(df)
        out[key] = df.merge(keys, on=KEY_COLS, how="inner")
        if len(out[key]) != before:
            print(f"[align] {key}: rows {before} -> {len(out[key])}", flush=True)
    return out, keys


def metric_row(df: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    pred = df["pred_pv_norm"].astype(float)
    truth = df["true_pv_norm"].astype(float)
    err = pred - truth
    row: dict[str, Any] = {
        **extra,
        "n_points": int(len(df)),
        "n_stations": int(df["station_sn"].nunique()),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
    }
    extras = regression_metric_extras(df, "pred_pv_norm", "true_pv_norm")
    row["R2"] = float(extras["r2_pv_norm"])
    row["ACC"] = float(extras["acc_installed_capacity"])
    return row


def build_tables(details: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    horizon_rows = []
    station_rows = []
    for key, df in details.items():
        method = str(df["Method"].iloc[0])
        method_type = str(df["Type"].iloc[0])
        base = {"method_key": key, "Type": method_type, "Method": method}
        overall_rows.append(metric_row(df, {**base, "scope": "overall"}))
        for horizon, g in df.groupby("horizon", sort=True):
            horizon_rows.append(metric_row(g, {**base, "horizon": str(horizon)}))
        for station_sn, g in df.groupby("station_sn", sort=True):
            station_rows.append(metric_row(g, {**base, "station_sn": str(station_sn)}))
    overall = pd.DataFrame(overall_rows)
    horizon = pd.DataFrame(horizon_rows)
    station = pd.DataFrame(station_rows)
    if not horizon.empty:
        horizon["_h_order"] = horizon["horizon"].astype(str).str.extract(r"h(\d+)", expand=False).astype(float)
        horizon = horizon.sort_values(["_h_order", "method_key"]).drop(columns="_h_order").reset_index(drop=True)
    return overall, horizon, station


def build_improvement(overall: pd.DataFrame, reference_key: str) -> pd.DataFrame:
    if overall.empty or reference_key not in set(overall["method_key"].astype(str)):
        return pd.DataFrame()
    ref = overall[overall["method_key"].eq(reference_key)].iloc[0]
    out = overall[~overall["method_key"].eq(reference_key)].copy()
    out["reference_MAE"] = float(ref["MAE"])
    out["reference_RMSE"] = float(ref["RMSE"])
    out["reference_R2"] = float(ref["R2"])
    out["MAE_gain"] = out["reference_MAE"] - out["MAE"]
    out["RMSE_gain"] = out["reference_RMSE"] - out["RMSE"]
    out["R2_gain"] = out["R2"] - out["reference_R2"]
    out["MAE_gain_pct"] = np.where(out["reference_MAE"] > 0, 100.0 * out["MAE_gain"] / out["reference_MAE"], np.nan)
    out["RMSE_gain_pct"] = np.where(out["reference_RMSE"] > 0, 100.0 * out["RMSE_gain"] / out["reference_RMSE"], np.nan)
    return out


def fmt(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def write_md(path: Path, table: pd.DataFrame, cols: list[str], digits: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in table.iterrows():
            cells = [fmt(row[col], digits) if col not in {"method_key", "Type", "Method", "horizon", "station_sn"} else str(row[col]) for col in cols]
            f.write("| " + " | ".join(cells) + " |\n")


def print_table(title: str, table: pd.DataFrame, cols: list[str], digits: int) -> None:
    print(f"\n=== {title} ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        cells = [fmt(row[col], digits) if col not in {"method_key", "Type", "Method", "horizon", "station_sn"} else str(row[col]) for col in cols]
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for spec in METHOD_SPECS:
        if spec["key"] == "temporal_baseline" and not bool(args.include_temporal):
            continue
        specs.append(spec)

    details: dict[str, pd.DataFrame] = {}
    counts = []
    for spec in specs:
        path = getattr(args, spec["path_arg"])
        df = read_detail(path, spec, args.split, bool(args.strict_missing))
        if df is None:
            continue
        details[spec["key"]] = df
        counts.append(
            {
                "method_key": spec["key"],
                "Method": spec["method"],
                "path": str(path),
                "split_rows_before_alignment": int(len(df)),
                "split_station_count_before_alignment": int(df["station_sn"].nunique()),
            }
        )
    if not details:
        raise ValueError("No eval-detail CSVs were loaded.")

    details, aligned_keys = align_details(details, args.align_mode, args.reference_key)
    overall, horizon, station = build_tables(details)
    improvement = build_improvement(overall, args.reference_key)

    count_df = pd.DataFrame(counts)
    aligned_counts = [
        {
            "method_key": key,
            "aligned_rows": int(len(df)),
            "aligned_station_count": int(df["station_sn"].nunique()),
            "aligned_issue_count": int(df[["station_sn", "issue_time"]].drop_duplicates().shape[0]),
        }
        for key, df in details.items()
    ]
    count_df = count_df.merge(pd.DataFrame(aligned_counts), on="method_key", how="left")

    overall.to_csv(out_dir / "graph_nwp_overall.csv", index=False)
    horizon.to_csv(out_dir / "graph_nwp_by_horizon.csv", index=False)
    station.to_csv(out_dir / "graph_nwp_by_station.csv", index=False)
    improvement.to_csv(out_dir / "graph_nwp_improvement_vs_temporal.csv", index=False)
    count_df.to_csv(out_dir / "graph_nwp_row_counts.csv", index=False)
    write_md(out_dir / "graph_nwp_overall.md", overall, ["Type", "Method", "MAE", "RMSE", "R2", "ACC", "Bias", "n_points"], int(args.digits))
    if not improvement.empty:
        write_md(
            out_dir / "graph_nwp_improvement_vs_temporal.md",
            improvement,
            ["Method", "MAE_gain", "RMSE_gain", "R2_gain", "MAE_gain_pct", "RMSE_gain_pct"],
            int(args.digits),
        )
    meta = {
        "split": args.split,
        "align_mode": args.align_mode,
        "reference_key": args.reference_key,
        "loaded_methods": list(details),
        "aligned_key_count": int(len(aligned_keys)),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("==== Stage 4 graph/NWP baseline collection complete ====")
    print(f"split={args.split} align_mode={args.align_mode}")
    print(f"aligned keys: {len(aligned_keys)}")
    print("\n=== Row Counts ===")
    print(count_df[["method_key", "split_rows_before_alignment", "aligned_rows", "aligned_station_count"]].to_string(index=False))
    print_table("Graph/NWP Overall", overall, ["Type", "Method", "MAE", "RMSE", "R2", "ACC", "Bias", "n_points"], int(args.digits))
    if not improvement.empty:
        print_table("Improvement vs BiLSTM + NWP", improvement, ["Method", "MAE_gain", "RMSE_gain", "R2_gain", "MAE_gain_pct", "RMSE_gain_pct"], int(args.digits))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
