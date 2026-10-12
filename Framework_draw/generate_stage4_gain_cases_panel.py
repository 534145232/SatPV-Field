#!/usr/bin/env python3
"""Draw Stage-4 station-wise gain violin and two representative cases.

This is a redraw script, not an image stitcher. It reads Stage-4 evaluation
detail CSVs, recomputes station-wise gains against the BiLSTM/NWP temporal
baseline, and renders two specified station-level forecast cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE4_DIR = PIPELINE_DIR / "stage_4"
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_metrics import attach_capacity_from_station_meta  # noqa: E402
from stage4_paths import (  # noqa: E402
    PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    RUNS_DIR,
)


DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
BASELINE_DETAIL_NAME = "per_station_temporal_baseline_eval_detail.csv"
NEAREST3_DETAIL_NAME = "nearest3_station_pv_eval_detail.csv"
KEY_COLS = ["station_sn", "eval_split", "issue_time", "h"]
CORE_COLS = KEY_COLS + ["horizon", "pred_pv_norm", "true_pv_norm"]

METHODS = {
    "nearest3_station_pv": {
        "label": "3-NN PV",
        "case_label": "3-NN PV",
        "color": "#7E63B6",
        "marker": "v",
        "detail_arg": "nearest3_detail_csv",
    },
    "future_swr_patch": {
        "label": "Future SWR",
        "case_label": "Future SWR guidance",
        "color": "#2A9D8F",
        "marker": "D",
        "detail_arg": "future_swr_detail_csv",
    },
    "historical_pv_field_patch": {
        "label": "Hist. PV",
        "case_label": "Historical PV-field guidance",
        "color": "#7FA650",
        "marker": "^",
        "detail_arg": "historical_field_detail_csv",
    },
    "fg_msf": {
        "label": "FG-MSF",
        "case_label": "FG-MSF",
        "color": "#D64A3A",
        "marker": "s",
        "detail_arg": "future_field_detail_csv",
    },
}

VIOLIN_ORDER = ["nearest3_station_pv", "future_swr_patch", "historical_pv_field_patch", "fg_msf"]
CASE_METHOD_ORDER = ["temporal_baseline", "future_swr_patch", "historical_pv_field_patch", "fg_msf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_detail_csv", type=Path, default=PER_STATION_TEMPORAL_BASELINE_RUN_DIR / BASELINE_DETAIL_NAME)
    parser.add_argument("--future_swr_detail_csv", type=Path, default=PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / DETAIL_NAME)
    parser.add_argument("--historical_field_detail_csv", type=Path, default=PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / DETAIL_NAME)
    parser.add_argument("--future_field_detail_csv", type=Path, default=PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / DETAIL_NAME)
    parser.add_argument("--nearest3_detail_csv", type=Path, default=RUNS_DIR / "21_nearest3_station_pv_baseline_seed2026" / NEAREST3_DETAIL_NAME)
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "stage4_gain_cases_panel")
    parser.add_argument("--out_prefix", type=str, default="stage4_gain_cases_panel")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--rising_station", type=str, default="020460182020003")
    parser.add_argument("--rising_issue", type=str, default="20230604_100000")
    parser.add_argument("--down_station", type=str, default="020AA017C210027")
    parser.add_argument("--down_issue", type=str, default="20230320_120000")
    parser.add_argument("--include_3nn_in_cases", type=int, default=0)
    parser.add_argument("--future_step_min", type=int, default=15)
    parser.add_argument("--hist_step_min", type=int, default=15)
    parser.add_argument("--y_min", type=float, default=-0.03)
    parser.add_argument("--y_max", type=float, default=1.20)
    parser.add_argument("--fig_width", type=float, default=7.2)
    parser.add_argument("--fig_height", type=float, default=6.15)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--save_pdf", type=int, default=1)
    parser.add_argument("--save_png", type=int, default=1)
    parser.add_argument("--save_svg", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def safe_label(label: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip())
    return re.sub(r"_+", "_", label).strip("_") or "model"


def parse_issue(value: str) -> str:
    token = str(value).strip()
    if re.fullmatch(r"\d{8}_\d{6}", token):
        return str(pd.to_datetime(token, format="%Y%m%d_%H%M%S"))
    return str(pd.Timestamp(token))


def history_cols(df: pd.DataFrame) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", str(name))
        return (int(match.group(1)) if match else 9999, str(name))

    return sorted([c for c in df.columns if str(c).startswith("hist_pv_norm_")], key=key)


def read_detail(path: Path, split: str, label: str, required: bool = True) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing detail CSV for {label}: {path}")
        print(f"[skip] missing {label}: {path}", flush=True)
        return None
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    df = attach_capacity_from_station_meta(df, path)
    df = df[df["eval_split"].astype(str).eq(str(split))].copy()
    df["station_sn"] = df["station_sn"].astype(str)
    df["eval_split"] = df["eval_split"].astype(str)
    df["issue_time"] = df["issue_time"].map(lambda value: str(pd.Timestamp(value)))
    df["h"] = pd.to_numeric(df["h"], errors="coerce").astype("Int64")
    df["pred_pv_norm"] = pd.to_numeric(df["pred_pv_norm"], errors="coerce")
    df["true_pv_norm"] = pd.to_numeric(df["true_pv_norm"], errors="coerce")
    df = df.dropna(subset=["h", "pred_pv_norm", "true_pv_norm"]).copy()
    return df


def load_all_details(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    baseline = read_detail(args.baseline_detail_csv, str(args.eval_split), "temporal baseline", required=True)
    assert baseline is not None
    details: dict[str, pd.DataFrame] = {}
    for key, spec in METHODS.items():
        df = read_detail(getattr(args, spec["detail_arg"]), str(args.eval_split), spec["label"], required=(key != "nearest3_station_pv"))
        if df is not None:
            details[key] = df
            print(f"[read] {key}: rows={len(df)}", flush=True)
    return baseline, details


def align_method_with_baseline(baseline: pd.DataFrame, candidate: pd.DataFrame, method_key: str) -> pd.DataFrame:
    hist = history_cols(baseline)
    base_cols = CORE_COLS + [c for c in hist if c in baseline.columns]
    base = baseline[base_cols].rename(columns={"pred_pv_norm": "pred_temporal_baseline", "true_pv_norm": "true_baseline"})
    cand = candidate[CORE_COLS].rename(columns={"pred_pv_norm": f"pred_{method_key}", "true_pv_norm": f"true_{method_key}"})
    merged = base.merge(cand, on=KEY_COLS + ["horizon"], how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError(f"No aligned rows between baseline and {method_key}.")
    truth_diff = (merged["true_baseline"].astype(float) - merged[f"true_{method_key}"].astype(float)).abs()
    max_diff = float(truth_diff.max()) if len(truth_diff) else float("nan")
    if np.isfinite(max_diff) and max_diff > 1e-5:
        print(f"[warn] {method_key}: max truth mismatch after alignment={max_diff:.6g}", flush=True)
    merged["true_pv_norm"] = merged["true_baseline"].astype(float)
    return merged


def station_gain_table(baseline: pd.DataFrame, details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key in VIOLIN_ORDER:
        if key not in details:
            continue
        merged = align_method_with_baseline(baseline, details[key], key)
        pred_col = f"pred_{key}"
        for sn, g in merged.groupby("station_sn", sort=True):
            true = g["true_pv_norm"].to_numpy(float)
            ref = g["pred_temporal_baseline"].to_numpy(float)
            pred = g[pred_col].to_numpy(float)
            valid = np.isfinite(true) & np.isfinite(ref) & np.isfinite(pred)
            if not valid.any():
                continue
            ref_mae = float(np.mean(np.abs(ref[valid] - true[valid])))
            mae = float(np.mean(np.abs(pred[valid] - true[valid])))
            rows.append(
                {
                    "method_key": key,
                    "method": METHODS[key]["label"],
                    "station_sn": str(sn),
                    "mae": mae,
                    "baseline_mae": ref_mae,
                    "mae_gain": ref_mae - mae,
                    "n_points": int(valid.sum()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No station-wise gains were computed.")
    return out


def build_case_frame(baseline: pd.DataFrame, details: dict[str, pd.DataFrame], station: str, issue: str, include_3nn: bool) -> pd.DataFrame:
    issue_norm = parse_issue(issue)
    hist = history_cols(baseline)
    cols = CORE_COLS + [c for c in hist if c in baseline.columns]
    sub = baseline[
        baseline["station_sn"].astype(str).eq(str(station)) & baseline["issue_time"].astype(str).eq(issue_norm)
    ][cols].copy()
    if sub.empty:
        examples = baseline[["station_sn", "issue_time"]].drop_duplicates().head(8).to_string(index=False)
        raise ValueError(f"Case not found in baseline: station={station}, issue={issue_norm}\nExamples:\n{examples}")
    sub = sub.rename(columns={"pred_pv_norm": "pred_temporal_baseline"})
    method_order = ["future_swr_patch", "historical_pv_field_patch", "fg_msf"]
    if include_3nn and "nearest3_station_pv" in details:
        method_order.insert(0, "nearest3_station_pv")
    for key in method_order:
        if key not in details:
            continue
        cand = details[key]
        right = cand[
            cand["station_sn"].astype(str).eq(str(station)) & cand["issue_time"].astype(str).eq(issue_norm)
        ][CORE_COLS].rename(columns={"pred_pv_norm": f"pred_{key}", "true_pv_norm": f"true_{key}"})
        if right.empty:
            print(f"[warn] missing case rows for {key}: station={station}, issue={issue_norm}", flush=True)
            continue
        sub = sub.merge(right[KEY_COLS + ["horizon", f"pred_{key}", f"true_{key}"]], on=KEY_COLS + ["horizon"], how="inner")
    return sub.sort_values("h").reset_index(drop=True)


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#334155",
            "xtick.color": "#1f2937",
            "ytick.color": "#1f2937",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, text: str) -> None:
    ax.text(
        -0.052,
        1.045,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )


def draw_violin(ax, gains: pd.DataFrame, rng: np.random.Generator) -> None:
    import matplotlib.ticker as mticker

    positions = np.arange(1, len(VIOLIN_ORDER) + 1, dtype=float)
    data = []
    labels = []
    colors = []
    for key in VIOLIN_ORDER:
        g = gains[gains["method_key"].eq(key)]["mae_gain"].to_numpy(float)
        g = g[np.isfinite(g)]
        data.append(g)
        labels.append(METHODS[key]["label"])
        colors.append(METHODS[key]["color"])
    parts = ax.violinplot(data, positions=positions, widths=0.68, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(1.1)
    for x, values, color in zip(positions, data, colors):
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(x + jitter, values, s=9, color=color, alpha=0.72, edgecolors="white", linewidths=0.25, zorder=3)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        lo, hi = np.min(values), np.max(values)
        ax.vlines(x, lo, hi, color="#1f2937", linewidth=0.8, zorder=4)
        ax.vlines(x, q1, q3, color=color, linewidth=7.5, alpha=0.58, zorder=4)
        ax.hlines(med, x - 0.20, x + 0.20, color="#111827", linewidth=1.15, zorder=5)
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"MAE gain ($10^{-3}$ p.u.)")
    ax.set_xlim(0.45, len(VIOLIN_ORDER) + 0.55)
    ax.set_ylim(-0.006, 0.0205)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value * 1.0e3:g}"))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "(a)")


def draw_case(ax, case_df: pd.DataFrame, panel: str, args: argparse.Namespace, show_ylabel: bool) -> None:
    hist = history_cols(case_df)
    future_x = case_df["h"].to_numpy(dtype=float) * float(args.future_step_min)
    if hist:
        hist_y = case_df.iloc[0][hist].to_numpy(dtype=float)
        hist_x = (np.arange(len(hist_y), dtype=float) - (len(hist_y) - 1)) * float(args.hist_step_min)
        ax.plot(hist_x, hist_y, color="#7A8288", marker=".", markersize=4.2, linewidth=1.7, label="Historical PV", zorder=2)
    ax.axvline(0.0, color="#aeb6bd", linestyle="--", linewidth=0.85)
    ax.plot(
        future_x,
        case_df["true_pv_norm"].to_numpy(float),
        color="#111111",
        marker="o",
        markersize=3.8,
        linewidth=1.9,
        label="Ground truth",
        zorder=6,
    )
    style = {
        "pred_temporal_baseline": ("BiLSTM+NWP", "#526A7A", "o", 1.6),
        "pred_nearest3_station_pv": ("3-NN PV", METHODS["nearest3_station_pv"]["color"], "v", 1.45),
        "pred_future_swr_patch": ("Future SWR guidance", METHODS["future_swr_patch"]["color"], "D", 1.55),
        "pred_historical_pv_field_patch": ("Historical PV-field guidance", METHODS["historical_pv_field_patch"]["color"], "^", 1.55),
        "pred_fg_msf": ("FG-MSF", METHODS["fg_msf"]["color"], "s", 1.85),
    }
    for col, (label, color, marker, lw) in style.items():
        if col not in case_df.columns:
            continue
        ax.plot(future_x, case_df[col].to_numpy(float), color=color, marker=marker, markersize=3.4, linewidth=lw, label=label, alpha=0.97)
    panel_label(ax, panel)
    ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.set_xlim(-165, 126)
    ax.set_xticks(np.arange(-150, 121, 30))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.72)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Time relative to forecast issue (min)", labelpad=3)
    if show_ylabel:
        ax.set_ylabel("PV power (p.u.)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(args: argparse.Namespace, gains: pd.DataFrame, rising: pd.DataFrame, down: pd.DataFrame) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.95, 1.12],
        left=0.095,
        right=0.985,
        top=0.900,
        bottom=0.115,
        hspace=0.38,
        wspace=0.16,
    )
    ax_v = fig.add_subplot(gs[0, :])
    draw_violin(ax_v, gains, rng)
    ax_r = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_r)
    draw_case(ax_r, rising, "(b)", args, show_ylabel=True)
    draw_case(ax_d, down, "(c)", args, show_ylabel=False)

    handles, labels = ax_r.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.54, 0.985),
        fontsize=6.8,
        handlelength=1.55,
        columnspacing=0.85,
        borderaxespad=0.0,
    )

    stem = str(args.out_prefix).strip() or "stage4_gain_cases_panel"
    outputs = {}
    if int(args.save_png):
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
        outputs["png"] = str(png)
    if int(args.save_pdf):
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
        outputs["pdf"] = str(pdf)
    if int(args.save_svg):
        svg = out_dir / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
        outputs["svg"] = str(svg)
    plt.close(fig)

    gains.to_csv(out_dir / f"{stem}_station_gains.csv", index=False)
    meta = {
        "rising_case": {"station_sn": str(args.rising_station), "issue_time": parse_issue(args.rising_issue)},
        "down_case": {"station_sn": str(args.down_station), "issue_time": parse_issue(args.down_issue)},
        "violin_order": [METHODS[k]["label"] for k in VIOLIN_ORDER],
        "outputs": outputs,
    }
    meta_path = out_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["meta"] = str(meta_path)
    outputs["station_gains"] = str(out_dir / f"{stem}_station_gains.csv")
    return outputs


def main() -> None:
    args = parse_args()
    baseline, details = load_all_details(args)
    gains = station_gain_table(baseline, details)
    rising = build_case_frame(baseline, details, str(args.rising_station), str(args.rising_issue), bool(args.include_3nn_in_cases))
    down = build_case_frame(baseline, details, str(args.down_station), str(args.down_issue), bool(args.include_3nn_in_cases))
    outputs = render(args, gains, rising, down)
    print("==== Stage 4 gain distribution + cases panel ====", flush=True)
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
