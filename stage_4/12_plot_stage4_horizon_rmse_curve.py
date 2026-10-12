#!/usr/bin/env python3
"""Plot horizon-wise Stage 4 RMSE for the main station baseline vs proposed field guidance."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_paths import (
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)


BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
FIELD_DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
BASELINE_MODEL = "temporal_baseline"
PROPOSED_MODEL = "future_pv_field_patch"
MODEL_SPECS = [
    {
        "model": BASELINE_MODEL,
        "label": "Temporal baseline",
        "color": "#526A7A",
        "marker": "o",
        "detail_arg": "baseline_detail_csv",
        "required": True,
    },
    {
        "model": "future_swr_patch",
        "label": "+ Future SWR patch",
        "color": "#2A9D8F",
        "marker": "D",
        "detail_arg": "future_swr_patch_detail_csv",
        "required": False,
    },
    {
        "model": "historical_pv_field_patch",
        "label": "+ Historical PV-field patch",
        "color": "#7FA650",
        "marker": "^",
        "detail_arg": "historical_field_detail_csv",
        "required": False,
    },
    {
        "model": PROPOSED_MODEL,
        "label": "FG-MSF",
        "color": "#D64A3A",
        "marker": "s",
        "detail_arg": "future_field_detail_csv",
        "required": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main_summary_csv",
        type=Path,
        default=RUNS_DIR / "11_eval_stage4_main_table" / "stage4_main_eval_summary.csv",
        help="Preferred summary produced by 11_eval_stage4_main_table.py.",
    )
    parser.add_argument(
        "--baseline_detail_csv",
        type=Path,
        default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / FIELD_DETAIL_NAME,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=RUNS_DIR / "12_stage4_horizon_rmse_curve",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_horizon_min", type=int, default=105)
    parser.add_argument("--future_step_min", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--fig_width", type=float, default=5.4)
    parser.add_argument("--fig_height", type=float, default=3.35)
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    parser.add_argument("--force_detail_recompute", action="store_true")
    return parser.parse_args()


def horizon_to_step(value: object) -> int | None:
    text = str(value)
    match = re.fullmatch(r"h(\d+)", text.strip().lower())
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number > 0 else None


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")


def load_curve_from_summary(args: argparse.Namespace) -> pd.DataFrame | None:
    path = Path(args.main_summary_csv)
    if args.force_detail_recompute or not path.exists():
        return None
    summary = pd.read_csv(path)
    require_columns(summary, ["model", "eval_split", "horizon", "dimension", "rmse_pv_norm", "n_points"], path)
    work = summary[
        (summary["eval_split"].astype(str) == str(args.split))
        & (summary["dimension"].astype(str) == "horizon")
        & (summary["model"].astype(str).isin([spec["model"] for spec in MODEL_SPECS]))
    ].copy()
    work["h_step"] = work["horizon"].map(horizon_to_step)
    work = work[work["h_step"].notna()].copy()
    if work.empty:
        return None
    work["h_step"] = work["h_step"].astype(int)
    work["horizon_min"] = work["h_step"] * int(args.future_step_min)
    work = work[work["horizon_min"] <= int(args.max_horizon_min)].copy()
    return work[["model", "horizon", "h_step", "horizon_min", "rmse_pv_norm", "n_points"]].sort_values(
        ["model", "h_step"]
    )


def read_detail(path: Path, label: str, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        message = f"Missing {label} detail CSV: {path}"
        if required:
            raise FileNotFoundError(message)
        print(f"[skip] {message}", flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    require_columns(df, KEY_COLS + ["horizon", "pred_pv_norm", "true_pv_norm"], path)
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].astype(str)
    df["h"] = df["h"].astype(int)
    return df


def load_curve_from_details(args: argparse.Namespace) -> pd.DataFrame:
    details = {}
    active_specs = []
    for spec in MODEL_SPECS:
        path = Path(getattr(args, spec["detail_arg"]))
        df = read_detail(path, str(spec["model"]), bool(spec["required"]))
        if df is None:
            continue
        details[str(spec["model"])] = df
        active_specs.append(spec)
    if BASELINE_MODEL not in details or PROPOSED_MODEL not in details:
        raise RuntimeError("Need at least baseline and proposed detail CSVs for detail recompute.")
    baseline = details[BASELINE_MODEL]
    base_cols = KEY_COLS + ["horizon", "pred_pv_norm", "true_pv_norm"]
    merged = baseline[base_cols].rename(columns={"pred_pv_norm": f"pred_{BASELINE_MODEL}"})
    cand_cols = KEY_COLS + ["pred_pv_norm", "true_pv_norm"]
    for spec in active_specs:
        model = str(spec["model"])
        if model == BASELINE_MODEL:
            continue
        right = details[model][cand_cols].rename(
            columns={"pred_pv_norm": f"pred_{model}", "true_pv_norm": f"true_{model}"}
        )
        before = len(merged)
        merged = merged.merge(right, on=KEY_COLS, how="inner", validate="one_to_one")
        if len(merged) != before:
            print(f"[warn] align {model}: rows {before} -> {len(merged)}", flush=True)
        truth_diff = (merged["true_pv_norm"].astype(float) - merged[f"true_{model}"].astype(float)).abs()
        max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
        if np.isfinite(max_diff) and max_diff > 1.0e-5:
            print(f"[warn] max truth mismatch for {model}: {max_diff:.6g}", flush=True)
    merged = merged[merged["eval_split"].astype(str) == str(args.split)].copy()
    merged["h_step"] = merged["h"].astype(int)
    merged["horizon_min"] = merged["h_step"] * int(args.future_step_min)
    merged = merged[merged["horizon_min"] <= int(args.max_horizon_min)].copy()
    rows = []
    for spec in active_specs:
        model = str(spec["model"])
        pred_col = f"pred_{model}"
        if pred_col not in merged.columns:
            continue
        for h_step, g in merged.groupby("h_step", sort=True):
            err = g[pred_col].astype(float) - g["true_pv_norm"].astype(float)
            rows.append(
                {
                    "model": model,
                    "horizon": f"h{int(h_step)}",
                    "h_step": int(h_step),
                    "horizon_min": int(h_step) * int(args.future_step_min),
                    "rmse_pv_norm": float(math.sqrt(float(np.mean(np.square(err))))),
                    "n_points": int(len(g)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "h_step"])


def add_improvement(curve: pd.DataFrame) -> pd.DataFrame:
    base = curve[curve["model"] == BASELINE_MODEL][["h_step", "rmse_pv_norm"]].rename(
        columns={"rmse_pv_norm": "baseline_rmse_pv_norm"}
    )
    prop = curve[curve["model"] == PROPOSED_MODEL][["h_step", "rmse_pv_norm"]].rename(
        columns={"rmse_pv_norm": "proposed_rmse_pv_norm"}
    )
    out = base.merge(prop, on="h_step", how="inner", validate="one_to_one")
    out["rmse_gain"] = out["baseline_rmse_pv_norm"] - out["proposed_rmse_pv_norm"]
    out["rmse_gain_pct"] = np.where(
        out["baseline_rmse_pv_norm"] > 0,
        100.0 * out["rmse_gain"] / out["baseline_rmse_pv_norm"],
        np.nan,
    )
    return out


def plot_curve(curve: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to draw the horizon RMSE curve. "
            "Install it in the active Python environment, e.g. `pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {str(spec["model"]): str(spec["label"]) for spec in MODEL_SPECS}
    colors = {str(spec["model"]): str(spec["color"]) for spec in MODEL_SPECS}
    markers = {str(spec["model"]): str(spec["marker"]) for spec in MODEL_SPECS}
    fig, ax = plt.subplots(figsize=(float(args.fig_width), float(args.fig_height)))
    for spec in MODEL_SPECS:
        model = str(spec["model"])
        g = curve[curve["model"] == model].sort_values("horizon_min")
        if g.empty:
            continue
        ax.plot(
            g["horizon_min"],
            g["rmse_pv_norm"],
            color=colors[model],
            marker=markers[model],
            markersize=4.8,
            linewidth=2.2,
            label=labels[model],
        )
    ticks = sorted(curve["horizon_min"].dropna().astype(int).unique().tolist())
    ax.set_xticks(ticks)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("RMSE (p.u.)")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#69737A")
    ax.spines["bottom"].set_color("#69737A")
    ax.tick_params(axis="both", colors="#263238", labelsize=9)
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=8.5,
        handlelength=2.6,
        ncol=2,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.8, rect=[0.0, 0.0, 1.0, 0.86])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in [x.strip().lower() for x in str(args.formats).split(",") if x.strip()]:
        path = out_dir / f"stage4_horizon_rmse_curve.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = int(args.dpi)
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    curve = load_curve_from_summary(args)
    source = "main summary"
    if curve is None:
        curve = load_curve_from_details(args)
        source = "two-way aligned details"
    if curve.empty:
        raise RuntimeError("No horizon-wise rows were available for plotting.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "stage4_horizon_rmse_curve_data.csv", index=False)
    improvement = add_improvement(curve)
    improvement.to_csv(out_dir / "stage4_horizon_rmse_improvement.csv", index=False)
    written = plot_curve(curve, args)

    print("==== Stage 4 horizon-wise RMSE curve ====")
    print(f"source : {source}")
    print(f"split  : {args.split}")
    print(curve.pivot_table(index="horizon_min", columns="model", values="rmse_pv_norm", aggfunc="first").to_string())
    print("\n==== RMSE improvement ====")
    print(improvement.to_string(index=False))
    for path in written:
        print(f"[write] {path}")
    print(f"[write] {out_dir / 'stage4_horizon_rmse_curve_data.csv'}")
    print(f"[write] {out_dir / 'stage4_horizon_rmse_improvement.csv'}")


if __name__ == "__main__":
    main()
