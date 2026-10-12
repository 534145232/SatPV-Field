"""Evaluate Stage 4 station forecasts under low- and high-ramp PV conditions.

The ramp condition is computed from the ground-truth future PV sequence of each
target-station issue. By default, the ramp score is max |PV(t+h)-PV(t+h-1)| and
the median score splits cases into Low-ramp and High-ramp groups.
"""

from __future__ import annotations

import argparse
import json
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
)


DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
SEQ_COLS = ["station_sn", "eval_split", "issue_time"]

METHOD_SPECS = [
    {
        "key": "temporal_baseline",
        "method": "BiLSTM + NWP",
        "path_arg": "baseline_detail_csv",
    },
    {
        "key": "fg_msf",
        "method": "FG-MSF",
        "path_arg": "future_field_detail_csv",
    },
    {
        "key": "future_swr_patch",
        "method": "Future SWR-field guidance",
        "path_arg": "future_swr_patch_detail_csv",
    },
    {
        "key": "historical_pv_field_patch",
        "method": "Historical PV-field guidance",
        "path_arg": "historical_field_detail_csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument("--extra_detail_csv", action="append", default=[], help="Optional extra method as key:label:path")
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "25_eval_stage4_ramp_condition_analysis")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--align_mode", choices=["intersection", "reference", "none"], default="intersection")
    parser.add_argument("--reference_key", type=str, default="temporal_baseline")
    parser.add_argument("--candidate_key", type=str, default="fg_msf")
    parser.add_argument("--include_keys", type=str, default="temporal_baseline,fg_msf")
    parser.add_argument("--ramp_metric", choices=["max_abs_delta", "mean_abs_delta", "std_delta"], default="max_abs_delta")
    parser.add_argument("--low_quantile", type=float, default=0.5)
    parser.add_argument("--high_quantile", type=float, default=0.5)
    parser.add_argument("--min_group_sequences", type=int, default=10)
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0.0
    return values.astype(str).str.lower().str.strip().isin(["true", "1", "yes", "y"])


def safe_label(label: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "method"


def parse_extra_specs(extra_args: list[str]) -> list[dict[str, str]]:
    specs = []
    for item in extra_args:
        parts = str(item).split(":", 2)
        if len(parts) != 3:
            raise ValueError("--extra_detail_csv must be key:label:path")
        key, label, path = parts
        specs.append({"key": safe_label(key), "method": label, "path": path})
    return specs


def parse_key_set(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def read_detail(path: Path, key: str, method: str, split: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing eval-detail CSV for {method}: {path}")
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = [*KEY_COLS, "horizon", "pred_pv_norm", "true_pv_norm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[df["eval_split"].astype(str) == str(split)].copy()
    out["station_sn"] = out["station_sn"].astype(str)
    out["eval_split"] = out["eval_split"].astype(str)
    out["issue_time"] = out["issue_time"].astype(str)
    out["h"] = out["h"].astype(int)
    out["horizon"] = out["horizon"].astype(str)
    out["pred_pv_norm"] = pd.to_numeric(out["pred_pv_norm"], errors="coerce")
    out["true_pv_norm"] = pd.to_numeric(out["true_pv_norm"], errors="coerce")
    if "large_change" in out.columns:
        out["large_change"] = coerce_bool(out["large_change"])
    out["method_key"] = key
    out["method"] = method
    print(f"[read] {key}: {path} rows={len(out)}", flush=True)
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


def load_method_details(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    include_keys = parse_key_set(args.include_keys)
    specs = []
    for spec in METHOD_SPECS:
        if include_keys and spec["key"] not in include_keys:
            continue
        path = getattr(args, spec["path_arg"])
        specs.append({"key": spec["key"], "method": spec["method"], "path": path})
    specs.extend(parse_extra_specs(args.extra_detail_csv))
    if not specs:
        raise ValueError("No methods selected.")
    return {spec["key"]: read_detail(Path(spec["path"]), spec["key"], spec["method"], args.split) for spec in specs}


def compute_ramp_scores(reference: pd.DataFrame, ramp_metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cols, g in reference.groupby(SEQ_COLS, sort=False):
        station_sn, eval_split, issue_time = cols
        seq = g.sort_values("h")["true_pv_norm"].to_numpy(float)
        seq = seq[np.isfinite(seq)]
        if seq.size < 2:
            score = np.nan
        else:
            delta = np.diff(seq)
            if ramp_metric == "max_abs_delta":
                score = float(np.max(np.abs(delta)))
            elif ramp_metric == "mean_abs_delta":
                score = float(np.mean(np.abs(delta)))
            else:
                score = float(np.std(delta))
        rows.append(
            {
                "station_sn": station_sn,
                "eval_split": eval_split,
                "issue_time": issue_time,
                "ramp_score": score,
            }
        )
    scores = pd.DataFrame(rows).dropna(subset=["ramp_score"])
    if scores.empty:
        raise ValueError("No valid ramp scores were computed.")
    return scores


def assign_ramp_condition(scores: pd.DataFrame, low_quantile: float, high_quantile: float) -> pd.DataFrame:
    if not (0.0 <= low_quantile <= 1.0 and 0.0 <= high_quantile <= 1.0):
        raise ValueError("Quantiles must be within [0, 1].")
    if low_quantile > high_quantile:
        raise ValueError("--low_quantile must be <= --high_quantile")
    out = scores.copy()
    low_thr = float(out["ramp_score"].quantile(low_quantile))
    high_thr = float(out["ramp_score"].quantile(high_quantile))
    condition = np.full(len(out), "Mid-ramp", dtype=object)
    condition[out["ramp_score"].to_numpy(float) <= low_thr] = "Low-ramp"
    condition[out["ramp_score"].to_numpy(float) > high_thr] = "High-ramp"
    if low_quantile == high_quantile:
        condition[out["ramp_score"].to_numpy(float) > high_thr] = "High-ramp"
    out["condition"] = condition
    out["ramp_low_threshold"] = low_thr
    out["ramp_high_threshold"] = high_thr
    return out[out["condition"].isin(["Low-ramp", "High-ramp"])].copy()


def metric_row(df: pd.DataFrame) -> dict[str, float | int]:
    y = pd.to_numeric(df["true_pv_norm"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["pred_pv_norm"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    err = p - y
    mae = float(np.mean(np.abs(err))) if err.size else np.nan
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else np.nan
    denom = float(np.sum((y - y.mean()) ** 2)) if y.size else np.nan
    r2 = float(1.0 - np.sum((y - p) ** 2) / denom) if y.size and denom > 0 else np.nan
    return {"N_points": int(err.size), "MAE": mae, "RMSE": rmse, "R2": r2}


def build_method_table(details: dict[str, pd.DataFrame], conditions: pd.DataFrame, min_group_sequences: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cond_cols = [*SEQ_COLS, "condition", "ramp_score"]
    for key, df in details.items():
        merged = df.merge(conditions[cond_cols], on=SEQ_COLS, how="inner")
        for condition, g in merged.groupby("condition", sort=False):
            n_seq = int(g[SEQ_COLS].drop_duplicates().shape[0])
            if n_seq < min_group_sequences:
                continue
            row = {
                "condition": condition,
                "method_key": key,
                "method": str(g["method"].iloc[0]),
                "N_seq": n_seq,
                "mean_ramp_score": float(g[SEQ_COLS + ["ramp_score"]].drop_duplicates()["ramp_score"].mean()),
            }
            row.update(metric_row(g))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"Low-ramp": 0, "High-ramp": 1}
        out["_order"] = out["condition"].map(order).fillna(99)
        out = out.sort_values(["_order", "method_key"]).drop(columns="_order").reset_index(drop=True)
    return out


def build_pair_table(method_table: pd.DataFrame, reference_key: str, candidate_key: str) -> pd.DataFrame:
    if method_table.empty:
        return method_table.copy()
    ref = method_table[method_table["method_key"].eq(reference_key)][["condition", "N_seq", "N_points", "RMSE", "MAE", "R2"]].copy()
    cand = method_table[method_table["method_key"].eq(candidate_key)][["condition", "RMSE", "MAE", "R2"]].copy()
    if ref.empty or cand.empty:
        return pd.DataFrame()
    ref = ref.rename(columns={"RMSE": "BiLSTM_RMSE", "MAE": "BiLSTM_MAE", "R2": "BiLSTM_R2"})
    cand = cand.rename(columns={"RMSE": "FG_MSF_RMSE", "MAE": "FG_MSF_MAE", "R2": "FG_MSF_R2"})
    out = ref.merge(cand, on="condition", how="inner")
    out["RMSE_gain"] = out["BiLSTM_RMSE"] - out["FG_MSF_RMSE"]
    out["RMSE_gain_pct"] = np.where(out["BiLSTM_RMSE"] > 0, 100.0 * out["RMSE_gain"] / out["BiLSTM_RMSE"], np.nan)
    out["MAE_gain"] = out["BiLSTM_MAE"] - out["FG_MSF_MAE"]
    out["R2_gain"] = out["FG_MSF_R2"] - out["BiLSTM_R2"]
    order = {"Low-ramp": 0, "High-ramp": 1}
    out["_order"] = out["condition"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


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
            f.write("| " + " | ".join(fmt(row[col], digits) if col != "condition" else str(row[col]) for col in cols) + " |\n")


def print_pair_table(table: pd.DataFrame, digits: int) -> None:
    print("\n=== Stage 4 ramp-condition evaluation ===")
    print("| Condition | N_seq | N_points | BiLSTM RMSE | FG-MSF RMSE | Gain | Gain (%) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in table.iterrows():
        print(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(int(row["N_seq"])),
                    str(int(row["N_points"])),
                    fmt(row["BiLSTM_RMSE"], digits),
                    fmt(row["FG_MSF_RMSE"], digits),
                    fmt(row["RMSE_gain"], digits),
                    fmt(row["RMSE_gain_pct"], 2),
                ]
            )
            + " |"
        )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    details = load_method_details(args)
    details, aligned_keys = align_details(details, args.align_mode, args.reference_key)
    if args.reference_key not in details:
        raise ValueError(f"Missing reference method {args.reference_key!r}.")
    if args.candidate_key not in details:
        raise ValueError(f"Missing candidate method {args.candidate_key!r}.")

    scores = compute_ramp_scores(details[args.reference_key], args.ramp_metric)
    conditions = assign_ramp_condition(scores, args.low_quantile, args.high_quantile)
    method_table = build_method_table(details, conditions, int(args.min_group_sequences))
    pair_table = build_pair_table(method_table, args.reference_key, args.candidate_key)
    if pair_table.empty:
        raise ValueError("Could not build the BiLSTM vs FG-MSF ramp-condition table.")

    scores.to_csv(out_dir / "ramp_scores_all_sequences.csv", index=False)
    conditions.to_csv(out_dir / "ramp_condition_sequences.csv", index=False)
    method_table.to_csv(out_dir / "ramp_condition_method_metrics.csv", index=False)
    pair_table.to_csv(out_dir / "ramp_condition_bilstm_vs_fgmsf.csv", index=False)
    write_md_table(
        out_dir / "ramp_condition_bilstm_vs_fgmsf.md",
        pair_table,
        ["condition", "N_seq", "N_points", "BiLSTM_RMSE", "FG_MSF_RMSE", "RMSE_gain", "RMSE_gain_pct"],
        int(args.digits),
    )
    meta = {
        "ramp_metric": args.ramp_metric,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "align_mode": args.align_mode,
        "split": args.split,
        "reference_key": args.reference_key,
        "candidate_key": args.candidate_key,
        "n_aligned_keys": int(len(aligned_keys)),
        "n_scored_sequences": int(len(scores)),
        "n_condition_sequences": int(len(conditions)),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[write] {out_dir / 'ramp_condition_bilstm_vs_fgmsf.csv'}")
    print(f"[write] {out_dir / 'ramp_condition_method_metrics.csv'}")
    print_pair_table(pair_table, int(args.digits))


if __name__ == "__main__":
    main()
