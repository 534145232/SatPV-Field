#!/usr/bin/env python3
"""Render Stage 4 station-level case comparisons with history, future truth, and baselines."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_metrics import attach_capacity_from_station_meta
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
CORE_DETAIL_COLS = [
    "station_sn",
    "eval_split",
    "issue_time",
    "h",
    "horizon",
    "pred_pv_norm",
    "true_pv_norm",
]
METHOD_SPECS = [
    {
        "model": "temporal_baseline",
        "label": "BiLSTM+NWP",
        "color": "#526A7A",
        "marker": "o",
        "path_arg": "baseline_detail_csv",
        "required": True,
    },
    {
        "model": "future_swr_patch",
        "label": "Future SWR guidance",
        "color": "#2A9D8F",
        "marker": "D",
        "path_arg": "future_swr_patch_detail_csv",
        "required": False,
    },
    {
        "model": "historical_pv_field_patch",
        "label": "Historical PV-field guidance",
        "color": "#7FA650",
        "marker": "^",
        "path_arg": "historical_field_detail_csv",
        "required": False,
    },
    {
        "model": "future_pv_field_patch",
        "label": "FG-MSF",
        "color": "#D64A3A",
        "marker": "s",
        "path_arg": "future_field_detail_csv",
        "required": True,
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
        "--future_swr_patch_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--historical_field_detail_csv",
        type=Path,
        default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--future_field_detail_csv",
        type=Path,
        default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / DETAIL_NAME,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=RUNS_DIR / "13_stage4_case_comparisons",
    )
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument(
        "--selection",
        choices=("random", "worst_baseline", "largest_gain", "largest_gap"),
        default="largest_gap",
        help="Case selection over station+issue sequences.",
    )
    parser.add_argument("--sample_count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--issue_times",
        type=str,
        default="",
        help="Optional comma-separated issue times, e.g. 20230608_120000. Overrides --selection.",
    )
    parser.add_argument(
        "--station_sns",
        type=str,
        default="",
        help="Optional comma-separated station ids. When --issue_times is set without stations, all stations at those issues are ranked by selection.",
    )
    parser.add_argument("--future_step_min", type=int, default=15)
    parser.add_argument("--hist_step_min", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--save_pdf", type=int, default=1)
    parser.add_argument("--save_svg", type=int, default=1)
    parser.add_argument("--y_min", type=float, default=-0.03)
    parser.add_argument("--y_max", type=float, default=1.20)
    parser.add_argument("--fig_width", type=float, default=6.2)
    parser.add_argument("--fig_height", type=float, default=3.2)
    return parser.parse_args()


def safe_label(label: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "model"


def history_cols(df: pd.DataFrame) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", name)
        return (int(match.group(1)) if match else 9999, name)

    return sorted([c for c in df.columns if c.startswith("hist_pv_norm_")], key=key)


def parse_issue_times(text: str) -> list[str]:
    out = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        if re.fullmatch(r"\d{8}_\d{6}", token):
            stamp = pd.to_datetime(token, format="%Y%m%d_%H%M%S")
        else:
            stamp = pd.Timestamp(token)
        out.append(str(stamp))
    return out


def parse_station_sns(text: str) -> list[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def read_detail(path: Path, label: str, required: bool) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        message = f"Missing eval-detail CSV for {label}: {path}"
        if required:
            raise FileNotFoundError(message)
        print(f"[skip] {message}", flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = [c for c in CORE_DETAIL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].map(lambda value: str(pd.Timestamp(value)))
    df["h"] = pd.to_numeric(df["h"], errors="coerce").astype("Int64")
    df["horizon"] = df["horizon"].astype(str)
    df["pred_pv_norm"] = pd.to_numeric(df["pred_pv_norm"], errors="coerce")
    df["true_pv_norm"] = pd.to_numeric(df["true_pv_norm"], errors="coerce")
    df = attach_capacity_from_station_meta(df, path)
    return df.dropna(subset=["h", "pred_pv_norm", "true_pv_norm"]).copy()


def load_details(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    details = {}
    active_specs = []
    for spec in METHOD_SPECS:
        path = getattr(args, spec["path_arg"])
        df = read_detail(path, spec["model"], bool(spec["required"]))
        if df is None:
            continue
        details[spec["model"]] = df
        active_specs.append(spec)
        print(f"[read] {spec['model']}: {path} rows={len(df)}", flush=True)
    if "temporal_baseline" not in details or "future_pv_field_patch" not in details:
        raise RuntimeError("Need at least temporal_baseline and future_pv_field_patch details for case visualization.")
    return details, active_specs


def align_details(details: dict[str, pd.DataFrame], active_specs: list[dict], eval_split: str) -> pd.DataFrame:
    baseline = details["temporal_baseline"]
    base_cols = KEY_COLS + ["horizon", "true_pv_norm", "pred_pv_norm"]
    base_cols += [c for c in ["capacity_raw", "capacity_scale"] if c in baseline.columns]
    base_cols += history_cols(baseline)
    merged = baseline[baseline["eval_split"] == str(eval_split)][[c for c in base_cols if c in baseline.columns]].copy()
    merged = merged.rename(columns={"pred_pv_norm": "pred_temporal_baseline"})
    for spec in active_specs:
        model = spec["model"]
        if model == "temporal_baseline":
            continue
        df = details[model]
        tag = safe_label(model)
        right_cols = KEY_COLS + ["pred_pv_norm", "true_pv_norm"]
        right = df[df["eval_split"] == str(eval_split)][right_cols].copy().rename(
            columns={"pred_pv_norm": f"pred_{tag}", "true_pv_norm": f"true_{tag}"}
        )
        before = len(merged)
        merged = merged.merge(right, on=KEY_COLS, how="inner", validate="one_to_one")
        if len(merged) != before:
            print(f"[warn] align {model}: rows {before} -> {len(merged)}", flush=True)
        diff = (merged["true_pv_norm"].astype(float) - merged[f"true_{tag}"].astype(float)).abs()
        max_diff = float(diff.max()) if len(diff) else float("nan")
        if np.isfinite(max_diff) and max_diff > 1.0e-5:
            print(f"[warn] max truth mismatch for {model}: {max_diff:.6g}", flush=True)
    for spec in active_specs:
        tag = safe_label(spec["model"])
        pred_col = f"pred_{tag}"
        if pred_col not in merged.columns:
            continue
        err = merged[pred_col].astype(float) - merged["true_pv_norm"].astype(float)
        merged[f"err_{tag}"] = err
        merged[f"abs_err_{tag}"] = err.abs()
    if merged.empty:
        raise RuntimeError(f"No aligned Stage4 rows for split={eval_split}.")
    return merged.sort_values(["station_sn", "issue_time", "h"]).reset_index(drop=True)


def build_case_scores(merged: pd.DataFrame, active_specs: list[dict]) -> pd.DataFrame:
    rows = []
    for (station, issue), g in merged.groupby(["station_sn", "issue_time"], sort=False):
        row = {
            "station_sn": str(station),
            "issue_time": str(issue),
            "n_horizons": int(g["h"].nunique()),
            "truth_mean": float(g["true_pv_norm"].astype(float).mean()),
            "truth_min": float(g["true_pv_norm"].astype(float).min()),
            "truth_max": float(g["true_pv_norm"].astype(float).max()),
            "truth_range": float(g["true_pv_norm"].astype(float).max() - g["true_pv_norm"].astype(float).min()),
        }
        for spec in active_specs:
            tag = safe_label(spec["model"])
            if f"abs_err_{tag}" in g.columns:
                row[f"{tag}_mae"] = float(g[f"abs_err_{tag}"].astype(float).mean())
                row[f"{tag}_rmse"] = float(math.sqrt(float(np.mean(np.square(g[f'err_{tag}'].astype(float))))))
        if "temporal_baseline_mae" in row and "future_pv_field_patch_mae" in row:
            row["proposed_gain_mae"] = row["temporal_baseline_mae"] - row["future_pv_field_patch_mae"]
            row["abs_gap_baseline_proposed"] = abs(row["proposed_gain_mae"])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_cases(args: argparse.Namespace, merged: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    issue_times = parse_issue_times(args.issue_times)
    station_sns = parse_station_sns(args.station_sns)
    work = scores.copy()
    if issue_times:
        work = work[work["issue_time"].isin(issue_times)].copy()
        if station_sns:
            work = work[work["station_sn"].astype(str).isin(station_sns)].copy()
        if work.empty:
            examples = ", ".join(scores["issue_time"].head(5).astype(str).tolist())
            raise ValueError(f"Requested issue/station cases were not found. First available issue examples: {examples}")
        sort_col = "abs_gap_baseline_proposed" if "abs_gap_baseline_proposed" in work.columns else "truth_range"
        return work.sort_values(sort_col, ascending=False).head(int(args.sample_count)).reset_index(drop=True)

    n = min(int(args.sample_count), len(work))
    if n <= 0:
        return work.head(0)
    if str(args.selection) == "random":
        rng = np.random.default_rng(int(args.seed))
        idx = np.sort(rng.choice(work.index.to_numpy(), size=n, replace=False))
        return work.loc[idx].reset_index(drop=True)
    if str(args.selection) == "worst_baseline":
        return work.sort_values("temporal_baseline_mae", ascending=False).head(n).reset_index(drop=True)
    if str(args.selection) == "largest_gain":
        return work.sort_values("proposed_gain_mae", ascending=False).head(n).reset_index(drop=True)
    return work.sort_values("abs_gap_baseline_proposed", ascending=False).head(n).reset_index(drop=True)


def plot_case(
    args: argparse.Namespace,
    merged: pd.DataFrame,
    sample: pd.Series,
    active_specs: list[dict],
    out_dir: Path,
    rank: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    station = str(sample["station_sn"])
    issue = str(sample["issue_time"])
    sub = merged[(merged["station_sn"].astype(str) == station) & (merged["issue_time"].astype(str) == issue)].sort_values("h")
    if sub.empty:
        raise RuntimeError(f"Internal error: empty case for station={station}, issue={issue}")

    hist_columns = history_cols(sub)
    future_x = sub["h"].to_numpy(dtype=float) * float(args.future_step_min)
    fig, ax = plt.subplots(1, 1, figsize=(float(args.fig_width), float(args.fig_height)))

    hist_min_x = -float(args.hist_step_min)
    if hist_columns:
        hist = sub.iloc[0][hist_columns].to_numpy(dtype=float)
        hist_x = (np.arange(len(hist), dtype=float) - (len(hist) - 1)) * float(args.hist_step_min)
        hist_min_x = float(hist_x.min())
        ax.plot(hist_x, hist, color="#7A8288", marker=".", markersize=7.0, linewidth=2.6, label="Historical PV")
    ax.axvline(0.0, color="#A7ADB2", linestyle="--", linewidth=1.5)
    ax.plot(
        future_x,
        sub["true_pv_norm"].to_numpy(dtype=float),
        color="#111111",
        marker="o",
        markersize=6.2,
        linewidth=3.0,
        label="Ground truth",
        zorder=5,
    )
    for spec in active_specs:
        tag = safe_label(spec["model"])
        pred_col = f"pred_{tag}"
        if pred_col not in sub.columns:
            continue
        ax.plot(
            future_x,
            sub[pred_col].to_numpy(dtype=float),
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            markersize=5.6,
            linewidth=2.4 if spec["model"] != "future_pv_field_patch" else 2.8,
            label=str(spec["label"]),
            alpha=0.95,
        )

    ax.set_ylabel("PV power (p.u.)", fontsize=14)
    ax.set_xlabel("Time relative to forecast issue (min)", fontsize=14)
    ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.set_xticks(np.arange(-150, int(future_x.max()) + 1, 30))
    ax.grid(axis="y", color="#D9DEE3", linewidth=1.0, alpha=0.78)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=12, handlelength=2.2, columnspacing=1.2)
    x_left = min(hist_min_x, float(future_x.min()) - float(args.future_step_min))
    x_right = float(future_x.max()) + float(args.future_step_min) * 0.4
    ax.set_xlim(x_left, x_right)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#69737A")
    ax.spines["bottom"].set_color("#69737A")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors="#263238", labelsize=13, width=1.1, length=4.5)

    fig.tight_layout(pad=0.35)
    safe_issue = pd.Timestamp(issue).strftime("%Y%m%d_%H%M%S")
    stem = f"stage4_case_{rank:02d}_{station}_{safe_issue}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    pdf_path = ""
    svg_path = ""
    if int(args.save_pdf):
        pdf_path = str(out_dir / f"{stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    if int(args.save_svg):
        svg_path = str(out_dir / f"{stem}.svg")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "rank": int(rank),
        "station_sn": station,
        "issue_time": issue,
        "png": str(png_path),
        "pdf": pdf_path,
        "svg": svg_path,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details, active_specs = load_details(args)
    merged = align_details(details, active_specs, str(args.eval_split))
    scores = build_case_scores(merged, active_specs)
    chosen = choose_cases(args, merged, scores)
    if chosen.empty:
        raise RuntimeError("No cases selected.")

    figure_rows = []
    for rank, sample in enumerate(chosen.itertuples(index=False), start=1):
        sample_series = pd.Series(sample._asdict())
        row = plot_case(args, merged, sample_series, active_specs, out_dir, rank)
        for key in sample_series.index:
            value = sample_series[key]
            row[key] = value.item() if hasattr(value, "item") else value
        figure_rows.append(row)
        print(f"[case {rank:02d}] station={row['station_sn']} issue={row['issue_time']} png={row['png']}", flush=True)

    merged.to_csv(out_dir / "stage4_case_aligned_detail.csv", index=False)
    scores.to_csv(out_dir / "stage4_case_scores.csv", index=False)
    chosen.to_csv(out_dir / "stage4_case_selected.csv", index=False)
    write_csv(out_dir / "stage4_case_figure_index.csv", figure_rows)
    print(f"[write] {out_dir / 'stage4_case_figure_index.csv'}", flush=True)
    print(f"[write] {out_dir / 'stage4_case_scores.csv'}", flush=True)
    print(f"[write] figures under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
