#!/usr/bin/env python3
"""Generate clean multi-station Stage 4 PV prediction curves for framework figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE4_DIR = PIPELINE_DIR / "stage_4"
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_paths import TARGET_EXCLUDED_RUN_DIR, TARGET_SELECTION_DIR  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage4_station_prediction_curve_assets"
DEFAULT_SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"
DEFAULT_DETAIL_CSV = (
    PIPELINE_DIR
    / "stage_4"
    / "runs"
    / "05l_per_station_temporal_stage3_field_patch_seed2026"
    / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"
)

COLOR_SEQUENCE = [
    "#8b1e2d",
    "#c94c1a",
    "#b36a00",
    "#2a9d8f",
    "#356f9a",
    "#4f46e5",
    "#7c3aed",
    "#0f766e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail_csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--sampled_issue_times_csv", type=Path, default=DEFAULT_SAMPLED_ISSUE_TIMES_CSV)
    parser.add_argument("--issue_time_column", type=str, default="issue_time")
    parser.add_argument("--sample_rank", type=int, default=1)
    parser.add_argument(
        "--issue_scope",
        type=str,
        choices=["single_sample", "sampled_pool"],
        default="sampled_pool",
        help="Use only one sampled issue time, or search across the whole sampled issue pool for more diverse curve shapes.",
    )
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--station_count", type=int, default=5)
    parser.add_argument("--station_sn_list", type=str, default="")
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["diverse_trend", "volatile", "representative"],
        default="diverse_trend",
        help="How to choose stations when --station_sn_list is not provided.",
    )
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=float, default=4.6)
    parser.add_argument("--height", type=float, default=1.45)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--line_width", type=float, default=4.4)
    parser.add_argument("--axis_color", type=str, default="#0f172a")
    parser.add_argument("--axis_width", type=float, default=1.8)
    parser.add_argument("--y_pad_ratio", type=float, default=0.10)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_target_csv(selection_dir: Path, target_run_dir: Path) -> Path:
    candidates = [
        Path(selection_dir) / "target_stations.csv",
        Path(target_run_dir) / "target_stations.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"target_stations.csv not found. Checked: {[str(p) for p in candidates]}")


def load_requested_issue_time(sampled_issue_times_csv: Path, issue_time_column: str, sample_rank: int) -> pd.Timestamp:
    req = pd.read_csv(sampled_issue_times_csv)
    if issue_time_column not in req.columns:
        raise KeyError(
            f"Column {issue_time_column!r} not found in {sampled_issue_times_csv}. Available columns: {list(req.columns)}"
        )
    rank = max(1, min(int(sample_rank), len(req)))
    return pd.Timestamp(req.iloc[rank - 1][issue_time_column])


def load_requested_issue_times(sampled_issue_times_csv: Path, issue_time_column: str) -> list[pd.Timestamp]:
    req = pd.read_csv(sampled_issue_times_csv)
    if issue_time_column not in req.columns:
        raise KeyError(
            f"Column {issue_time_column!r} not found in {sampled_issue_times_csv}. Available columns: {list(req.columns)}"
        )
    series = pd.to_datetime(req[issue_time_column], errors="coerce").dropna()
    return [pd.Timestamp(ts) for ts in series.tolist()]


def choose_representative_stations(targets: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or targets.empty:
        return targets.iloc[0:0].copy()
    coords = targets[["lon", "lat"]].to_numpy(np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    scale = np.where((hi - lo) > 1.0e-9, hi - lo, 1.0)
    norm = (coords - lo) / scale
    centroid = norm.mean(axis=0)
    chosen = [int(np.argmin(np.sum((norm - centroid[None]) ** 2, axis=1)))]
    while len(chosen) < min(int(count), len(targets)):
        remaining = [i for i in range(len(targets)) if i not in chosen]
        d2 = []
        for idx in remaining:
            dist = min(float(np.sum((norm[idx] - norm[c]) ** 2)) for c in chosen)
            d2.append((dist, idx))
        d2.sort(reverse=True)
        chosen.append(int(d2[0][1]))
    return targets.iloc[chosen].reset_index(drop=True)


def parse_station_sn_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def select_stations(available_targets: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    requested = parse_station_sn_list(args.station_sn_list)
    if requested:
        selected = available_targets[available_targets["station_sn"].astype(str).isin(requested)].copy()
        if selected.empty:
            raise ValueError(f"None of the requested stations were available: {requested}")
        selected["requested_rank"] = selected["station_sn"].astype(str).map({sn: i for i, sn in enumerate(requested)})
        selected = selected.sort_values("requested_rank").drop(columns=["requested_rank"]).reset_index(drop=True)
        return selected
    if str(args.selection_strategy) == "representative":
        return choose_representative_stations(available_targets, int(args.station_count))
    return available_targets.head(int(args.station_count)).reset_index(drop=True)


def curve_volatility_score(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    diffs = np.diff(arr)
    amplitude = float(np.max(arr) - np.min(arr))
    std = float(np.std(arr))
    mean_abs_step = float(np.mean(np.abs(diffs)))
    max_abs_step = float(np.max(np.abs(diffs))) if diffs.size else 0.0
    return amplitude * 1.2 + std * 0.8 + mean_abs_step * 1.6 + max_abs_step * 0.6


def classify_curve_trend(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 2:
        return "flat"
    amplitude = float(np.max(arr) - np.min(arr))
    if amplitude < 1.0e-4:
        return "flat"
    diffs = np.diff(arr)
    pos_frac = float(np.mean(diffs > 0.0)) if diffs.size else 0.0
    neg_frac = float(np.mean(diffs < 0.0)) if diffs.size else 0.0
    slope = float(arr[-1] - arr[0])
    peak_idx = int(np.argmax(arr))
    valley_idx = int(np.argmin(arr))
    center_lo = max(1, int(0.20 * len(arr)))
    center_hi = min(len(arr) - 2, int(0.80 * len(arr)))
    first_half = arr[: len(arr) // 2 + 1]
    second_half = arr[len(arr) // 2 :]
    first_slope = float(first_half[-1] - first_half[0]) if first_half.size > 1 else 0.0
    second_slope = float(second_half[-1] - second_half[0]) if second_half.size > 1 else 0.0

    if slope >= 0.38 * amplitude and pos_frac >= 0.55:
        return "rising"
    if slope <= -0.38 * amplitude and neg_frac >= 0.55:
        return "falling"
    if center_lo <= peak_idx <= center_hi and first_slope > 0.18 * amplitude and second_slope < -0.18 * amplitude:
        return "peak"
    if center_lo <= valley_idx <= center_hi and first_slope < -0.18 * amplitude and second_slope > 0.18 * amplitude:
        return "valley"
    if amplitude > 0.0 and float(np.sum(np.abs(diffs))) >= 1.8 * amplitude:
        return "oscillating"
    return "mixed"


def compute_global_limits(curves: list[np.ndarray], pad_ratio: float) -> tuple[float, float]:
    valid_chunks = []
    for curve in curves:
        arr = np.asarray(curve, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            valid_chunks.append(arr)
    if not valid_chunks:
        return 0.0, 1.0
    merged = np.concatenate(valid_chunks)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    span = hi - lo
    if span < 1.0e-6:
        mid = 0.5 * (lo + hi)
        return mid - 0.5, mid + 0.5
    pad = span * float(pad_ratio)
    return lo - pad, hi + pad


def plot_curve(
    values: np.ndarray,
    out_path: Path,
    color: str,
    width: float,
    height: float,
    dpi: int,
    line_width: float,
    axis_color: str,
    axis_width: float,
    y_limits: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(values, dtype=np.float32)
    x = np.arange(1, len(y) + 1, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(float(width), float(height)), dpi=int(dpi))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.plot(x, y, color=str(color), linewidth=float(line_width), solid_capstyle="round", solid_joinstyle="round")

    ax.set_xlim(float(x.min()), float(x.max()) if len(x) > 1 else float(x.min()) + 1.0)
    ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(str(axis_color))
    ax.spines["bottom"].set_color(str(axis_color))
    ax.spines["left"].set_linewidth(float(axis_width))
    ax.spines["bottom"].set_linewidth(float(axis_width))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.08, top=0.995)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, format="svg", dpi=int(dpi), bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    anchor_issue_time = load_requested_issue_time(Path(args.sampled_issue_times_csv), str(args.issue_time_column), int(args.sample_rank))
    if str(args.issue_scope) == "sampled_pool":
        issue_times = load_requested_issue_times(Path(args.sampled_issue_times_csv), str(args.issue_time_column))
    else:
        issue_times = [anchor_issue_time]

    detail = pd.read_csv(Path(args.detail_csv), dtype={"station_sn": str})
    detail["issue_time"] = pd.to_datetime(detail["issue_time"])
    detail["eval_split"] = detail["eval_split"].astype(str)
    detail["h"] = pd.to_numeric(detail["h"], errors="coerce")
    detail["pred_pv_norm"] = pd.to_numeric(detail["pred_pv_norm"], errors="coerce")
    detail = detail.dropna(subset=["issue_time", "h", "pred_pv_norm"]).copy()

    target_csv = resolve_target_csv(Path(args.selection_dir), Path(args.target_run_dir))
    targets = pd.read_csv(target_csv, dtype={"station_sn": str})
    targets["lon"] = pd.to_numeric(targets["lon"], errors="coerce")
    targets["lat"] = pd.to_numeric(targets["lat"], errors="coerce")
    targets = targets.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)

    sub = detail[
        (detail["eval_split"] == str(args.eval_split))
        & (detail["issue_time"].isin(issue_times))
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No prediction rows found for requested issue scope split={args.eval_split} in {args.detail_csv}"
        )

    candidate_rows = []
    grouped = sub.groupby(["station_sn", "issue_time"], sort=False)
    for (station, issue_time), station_sub in grouped:
        station_sub = station_sub.sort_values("h")
        curve = station_sub["pred_pv_norm"].to_numpy(np.float32)
        if curve.size == 0:
            continue
        row_match = targets[targets["station_sn"].astype(str) == str(station)]
        if row_match.empty:
            continue
        target_row = row_match.iloc[0]
        candidate_rows.append(
            {
                "station_sn": str(station),
                "issue_time": pd.Timestamp(issue_time),
                "lon": float(target_row["lon"]),
                "lat": float(target_row["lat"]),
                "volatility_score": float(curve_volatility_score(curve)),
                "trend_type": str(classify_curve_trend(curve)),
                "curve": curve,
                "horizons": station_sub["h"].astype(int).tolist(),
            }
        )
    if not candidate_rows:
        raise ValueError("No station-issue prediction curves were available after filtering.")

    requested_stations = parse_station_sn_list(args.station_sn_list)
    if requested_stations:
        ordered = {sn: i for i, sn in enumerate(requested_stations)}
        filtered_candidates = [row for row in candidate_rows if row["station_sn"] in ordered]
        filtered_candidates.sort(key=lambda row: (ordered[row["station_sn"]], row["issue_time"]))
        candidate_rows = filtered_candidates
    elif str(args.selection_strategy) == "representative":
        available_targets = pd.DataFrame(
            [{"station_sn": row["station_sn"], "lon": row["lon"], "lat": row["lat"]} for row in candidate_rows]
        ).drop_duplicates("station_sn")
        selected_targets = select_stations(available_targets.reset_index(drop=True), args)
        selected_station_set = set(selected_targets["station_sn"].astype(str).tolist())
        candidate_rows = [row for row in candidate_rows if row["station_sn"] in selected_station_set]
        candidate_rows.sort(key=lambda row: (-row["volatility_score"], row["station_sn"], row["issue_time"]))
    elif str(args.selection_strategy) == "volatile":
        candidate_rows.sort(key=lambda row: (-row["volatility_score"], row["station_sn"], row["issue_time"]))
    else:
        trend_priority = ["rising", "peak", "valley", "oscillating", "mixed", "flat", "falling"]
        chosen = []
        used_station_issue = set()
        for trend in trend_priority:
            bucket = [row for row in candidate_rows if row["trend_type"] == trend]
            bucket.sort(key=lambda row: (-row["volatility_score"], row["station_sn"], row["issue_time"]))
            for row in bucket:
                key = (row["station_sn"], str(row["issue_time"]))
                if key in used_station_issue:
                    continue
                chosen.append(row)
                used_station_issue.add(key)
                break
        remainder = [row for row in candidate_rows if (row["station_sn"], str(row["issue_time"])) not in used_station_issue]
        remainder.sort(
            key=lambda row: (
                row["trend_type"] == "falling",
                -row["volatility_score"],
                row["station_sn"],
                row["issue_time"],
            )
        )
        candidate_rows = chosen + remainder

    sample_dir = ensure_dir(out_dir / f"sample_{int(args.sample_rank):02d}_{anchor_issue_time:%Y%m%d_%H%M%S}")
    records = []
    curves = []
    station_payloads = []
    for row in candidate_rows[: max(1, int(args.station_count))]:
        curve = np.asarray(row["curve"], dtype=np.float32)
        curves.append(curve)
        station_payloads.append(row)

    if not station_payloads:
        raise ValueError("Selected stations had no prediction curves after filtering.")

    y_limits = compute_global_limits(curves, float(args.y_pad_ratio))
    for idx, row in enumerate(station_payloads, start=1):
        station = str(row["station_sn"])
        issue_time = pd.Timestamp(row["issue_time"])
        curve = np.asarray(row["curve"], dtype=np.float32)
        color = COLOR_SEQUENCE[(idx - 1) % len(COLOR_SEQUENCE)]
        out_path = sample_dir / f"pred_curve_{idx:02d}_{issue_time:%Y%m%d_%H%M%S}_{station}.svg"
        plot_curve(
            curve,
            out_path,
            color,
            args.width,
            args.height,
            args.dpi,
            args.line_width,
            args.axis_color,
            args.axis_width,
            y_limits,
        )
        records.append(
            {
                "rank": int(idx),
                "station_sn": station,
                "issue_time": str(issue_time),
                "station_lon": float(row["lon"]),
                "station_lat": float(row["lat"]),
                "color": color,
                "volatility_score": float(row.get("volatility_score", float("nan"))),
                "trend_type": str(row.get("trend_type", "")),
                "curve_path": str(out_path),
                "horizons": list(row["horizons"]),
                "pred_pv_norm": [float(v) for v in curve.tolist()],
            }
        )

    meta = {
        "anchor_issue_time": str(anchor_issue_time),
        "candidate_issue_times": [str(pd.Timestamp(ts)) for ts in issue_times],
        "sample_rank": int(args.sample_rank),
        "eval_split": str(args.eval_split),
        "issue_scope": str(args.issue_scope),
        "selection_strategy": str(args.selection_strategy),
        "detail_csv": str(args.detail_csv),
        "target_csv": str(target_csv),
        "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
        "issue_time_column": str(args.issue_time_column),
        "shared_y_limits": [float(y_limits[0]), float(y_limits[1])],
        "model_source": "Stage 4 05l per_station_temporal_swr_anchor04l_field_patch eval detail",
        "stations": records,
    }
    write_json(sample_dir / "prediction_curve_meta.json", meta)

    print("==== Stage 4 station prediction curve assets complete ====")
    print(f"anchor issue: {anchor_issue_time}")
    print(f"stations: {len(records)}")
    print(f"output dir: {sample_dir}")


if __name__ == "__main__":
    main()
