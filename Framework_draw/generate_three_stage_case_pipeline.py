#!/usr/bin/env python3
"""Redraw a compact three-stage E2F2S case pipeline from model outputs.

Unlike compose_three_stage_case_pipeline.py, this script does not stitch three
completed figures. It reuses the case-loading and inference logic from the
existing Stage-2, Stage-3, and Stage-4 visualization scripts, then redraws a
single pipeline-oriented panel with simplified subplots.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE2_DIR = PIPELINE_DIR / "stage_2"
STAGE3_DIR = PIPELINE_DIR / "stage_3"
STAGE4_DIR = PIPELINE_DIR / "stage_4"
STAGE1_CENTER_INDEX_ALL = (
    PIPELINE_DIR
    / "stage_1"
    / "data"
    / "03_indices"
    / "center"
    / "h8_center_index_4hist_3future_06_17_all.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue_time", type=str, default="20230608_120000")
    parser.add_argument("--station_sn", type=str, default="", help="Optional Stage-4 station id. Empty selects the strongest available case at the issue time.")
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "three_stage_case_pipeline")
    parser.add_argument("--out_prefix", type=str, default="three_stage_case_pipeline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--save_pdf", type=int, default=1)
    parser.add_argument("--save_png", type=int, default=1)
    parser.add_argument("--fig_width", type=float, default=3.55)
    parser.add_argument("--fig_height", type=float, default=7.55)
    parser.add_argument("--representative_horizon", type=int, default=4, help="Stage-3/4 horizon id, 1-based. h4 corresponds to 60-min forecast lead.")
    parser.add_argument("--random_count", type=int, default=20, help="Additional random station-issue cases to render after the requested case.")
    parser.add_argument("--random_scan_limit", type=int, default=0, help="Optional cap on Stage-4 station-issue candidates before random sampling; 0 uses all.")
    parser.add_argument("--lcig_vmax", type=float, default=1000.0)
    parser.add_argument("--pv_vmin", type=float, default=0.0)
    parser.add_argument("--pv_vmax", type=float, default=1.2)
    parser.add_argument("--stage4_y_min", type=float, default=-0.03)
    parser.add_argument("--stage4_y_max", type=float, default=1.20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_module(name: str, path: Path, extra_paths: list[Path]) -> Any:
    for p in extra_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def issue_arg_to_timestamp(text: str) -> pd.Timestamp:
    token = str(text).strip()
    if token.isdigit() and len(token) >= 12:
        return pd.to_datetime(token[:14] if len(token) >= 14 else token[:12], format="%Y%m%d%H%M%S" if len(token) >= 14 else "%Y%m%d%H%M")
    if "_" in token and token.replace("_", "").isdigit():
        return pd.to_datetime(token, format="%Y%m%d_%H%M%S")
    return pd.Timestamp(token)


def normalize_issue_key(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y%m%d_%H%M%S")
    except Exception:
        token = str(value).strip()
        if "_" in token and token.replace("_", "").isdigit():
            return pd.to_datetime(token, format="%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S")
        if token.isdigit() and len(token) >= 12:
            fmt = "%Y%m%d%H%M%S" if len(token) >= 14 else "%Y%m%d%H%M"
            return pd.to_datetime(token[:14] if len(token) >= 14 else token[:12], format=fmt).strftime("%Y%m%d_%H%M%S")
        return token


def available_stage2_issue_keys(split: str = "test") -> set[str]:
    path = STAGE1_CENTER_INDEX_ALL
    try:
        s2 = load_module("pipeline_stage2_issue_key_source", STAGE2_DIR / "visualize_swr_forecast_case_study.py", [STAGE2_DIR])
        path = Path(s2.STAGE1_CENTER_INDEX_ALL)
    except Exception as exc:
        print(f"[random cases][warn] using fallback Stage-2 index path because Stage-2 module import failed: {exc}", flush=True)
    if not path.exists():
        print(f"[random cases][warn] Stage-2 index not found for issue filtering: {path}", flush=True)
        return set()
    df = pd.read_csv(path)
    split_cols = [c for c in ["split", "time_split", "eval_split"] if c in df.columns]
    if split_cols:
        df = df[df[split_cols[0]].astype(str).eq(str(split))].copy()
    issue_cols = [c for c in ["issue_time_bjt", "issue_time", "target_time_bjt", "target_time"] if c in df.columns]
    if not issue_cols:
        return set()
    return {normalize_issue_key(v) for v in df[issue_cols[0]].dropna().tolist()}


def parse_horizons(text: str) -> list[int]:
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if raw:
            out.append(max(0, int(raw) - 1))
    return out or [0, 3, 6]


def load_stage2_case(args: argparse.Namespace) -> dict:
    import torch

    s2 = load_module("pipeline_stage2_swr_case", STAGE2_DIR / "visualize_swr_forecast_case_study.py", [STAGE2_DIR])
    ns = SimpleNamespace(
        run_root=s2.DEFAULT_RUN_ROOT,
        loss_variant="v0_full",
        base_checkpoint=s2.DEFAULT_BASE_CKPT,
        index_csv=s2.STAGE1_CENTER_INDEX_ALL,
        frame_cache_dir=s2.STAGE1_CENTER_FRAME_CACHE_DIR,
        sample_cache_dir=s2.STAGE1_CENTER_SAMPLE_CACHE_DIR,
        clear_sky_dir=s2.STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
        split="test",
        out_dir=Path(args.out_dir) / "_stage2_cache",
        output_prefix="lcig_pipeline",
        selection="largest_gain",
        sample_count=1,
        issue_times=str(args.issue_time),
        scan_count=0,
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=2,
        max_stripe_quantile=0.85,
        vmax=float(args.lcig_vmax),
        lead_labels="T,T+1 h,T+2 h",
        dpi=int(args.dpi),
        save_pdf=0,
        save_svg=0,
    )
    device = torch.device(ns.device if torch.cuda.is_available() or str(ns.device) == "cpu" else "cpu")
    base_model, base_stats, _base_checkpoint, _temporal_head = s2.load_base_ratio_model(Path(ns.base_checkpoint), device, "conv3d_decoder")
    residual_item = s2.load_residual_model(ns, device)
    dataset = s2.make_dataset(ns, base_stats)
    try:
        selected = s2.find_issue_indices(dataset, str(ns.issue_times))
        if not selected:
            raise RuntimeError(f"Stage-2 issue time not found: {ns.issue_times}")
        item = selected[0]
        sample = dataset[int(item["idx"])]
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
            if key in {"x", "raw_y", "mask", "clear_y"}
        }
        pred = s2.predict_batch(batch, base_model, residual_item, device)
        truth = s2.to_masked_np(pred["truth"], pred["mask"], 0)
        final = s2.to_masked_np(pred["final"], pred["mask"], 0)
        lat, lon = s2.load_grid_axes(dataset, int(item["idx"]))
        final_large, lat_up, lon_up = s2.ascending_grid(final, lat, lon)
        truth_large, _, _ = s2.ascending_grid(truth, lat, lon)
        crops = []
        for h in range(final_large.shape[0]):
            gt_crop, crop_lat, crop_lon = s2.crop_by_lonlat(truth_large[h], lat_up, lon_up)
            pred_crop, _, _ = s2.crop_by_lonlat(final_large[h], lat_up, lon_up)
            crops.append({"truth": gt_crop, "pred": pred_crop, "error": np.abs(pred_crop - gt_crop), "lat": crop_lat, "lon": crop_lon})
        return {
            "idx": int(item["idx"]),
            "issue_time": str(sample.get("meta", {}).get("issue_time_bjt", args.issue_time)),
            "large_pred": final_large,
            "large_truth": truth_large,
            "large_lat": lat_up,
            "large_lon": lon_up,
            "crops": crops,
            "ningbo_box": (s2.NINGBO_LON_MIN, s2.NINGBO_LON_MAX, s2.NINGBO_LAT_MIN, s2.NINGBO_LAT_MAX),
            "palette": s2.palette_from_stops(s2.COLOR_STOPS),
            "error_palette": s2.palette_from_stops(s2.ERROR_STOPS),
        }
    finally:
        dataset.close()


def station_grid_location(station_sn: str, lon_grid: np.ndarray, lat_grid: np.ndarray) -> dict | None:
    if not station_sn:
        return None
    candidates = [
        STAGE3_DIR / "data" / "02_target_selection_lhs50" / "target_stations.csv",
        STAGE4_DIR / "runs" / "baseline_missing_targets_seed2026_selection" / "target_stations.csv",
    ]
    target_path = next((path for path in candidates if path.exists()), None)
    if target_path is None:
        return None
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    row = targets[targets["station_sn"].astype(str).eq(str(station_sn))]
    if row.empty or not {"lon", "lat"}.issubset(row.columns):
        return None
    lon = float(row.iloc[0]["lon"])
    lat = float(row.iloc[0]["lat"])
    dist2 = (np.asarray(lon_grid, dtype=float) - lon) ** 2 + (np.asarray(lat_grid, dtype=float) - lat) ** 2
    y, x = np.unravel_index(int(np.nanargmin(dist2)), dist2.shape)
    return {"station_sn": str(station_sn), "lon": lon, "lat": lat, "x": int(x), "y": int(y)}


def extract_patch(arr: np.ndarray, center: dict | None, size: int = 5) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    if center is None:
        return None, None
    half = int(size) // 2
    y = int(center["y"])
    x = int(center["x"])
    padded = np.pad(np.asarray(arr, dtype=float), ((half, half), (half, half)), mode="constant", constant_values=np.nan)
    patch = padded[y : y + size, x : x + size]
    y0 = max(0, y - half)
    y1 = min(arr.shape[0] - 1, y + half)
    x0 = max(0, x - half)
    x1 = min(arr.shape[1] - 1, x + half)
    return patch, (x0, x1, y0, y1)


def load_stage3_case(args: argparse.Namespace, station_sn: str = "") -> dict:
    s3 = load_module(
        "pipeline_stage3_pv_case",
        STAGE3_DIR / "visualize_pv_field_case_study.py",
        [STAGE3_DIR, STAGE3_DIR / "ablation_motion_branch"],
    )
    ns = SimpleNamespace(
        out_dir=Path(args.out_dir) / "_stage3_cache",
        baseline_run_dir=s3.RUNS_DIR / "base_pvnorm_field_model",
        baseline_model="pv_nwp",
        baseline_delta_scale=float(s3.f04.PYCHARM_DEFAULTS.get("delta_scale", 1.0)),
        proposed_run_root=s3.RUNS_DIR / "ablation_motion_branch_last_only",
        proposed_motion_branch="warp_direct_gate",
        temporal_collapse="last_only",
        swr_dependency="full_swr_future",
        swr_anchor_cache_dir=None,
        swr_checkpoint=s3.motion_eval.SWR_BASE_CHECKPOINT,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        scan_count=0,
        scan_mode="even",
        case_select_mode="random",
        seed=int(args.seed),
        num_cases=1,
        issue_times=str(issue_arg_to_timestamp(args.issue_time)),
        min_issue_gap_hours=0.0,
        horizons=str(int(args.representative_horizon)),
        pv_cmap="shared_swr",
        err_cmap="shared_flow",
        pv_vmin=float(args.pv_vmin),
        pv_vmax=float(args.pv_vmax),
        err_vmin=0.0,
        err_vmax=0.4,
        dpi=int(args.dpi),
        save_pdf=0,
    )
    dataset_args, dataset, baseline, proposed, pv_stats, device = s3.build_everything(ns)
    horizons = parse_horizons(str(int(args.representative_horizon)))
    selected = s3.requested_issue_time_candidates(ns, dataset)
    if not selected:
        raise RuntimeError(f"Stage-3 issue time not found: {args.issue_time}")
    item = selected[0]
    case = s3.infer_case(dataset, int(item["idx"]), baseline, proposed, pv_stats, device)
    station_loc = station_grid_location(station_sn, dataset.lon_grid, dataset.lat_grid)
    return {
        "idx": int(item["idx"]),
        "issue_time": str(pd.Timestamp(item["issue_time"])),
        "truth": case["truth"],
        "proposed": case["proposed"],
        "error": np.abs(case["proposed"] - case["truth"]),
        "mask": case["mask"],
        "horizons": [h for h in horizons if h < int(dataset_args.future_frames)],
        "station_loc": station_loc,
        "palette": s3.palette_from_stops(s3.COLOR_STOPS),
        "error_palette": s3.palette_from_stops(s3.FLOW_COLOR_STOPS),
    }


def load_stage4_case(args: argparse.Namespace) -> dict:
    s4 = load_module("pipeline_stage4_station_case", STAGE4_DIR / "13_visualize_stage4_case_comparisons.py", [STAGE4_DIR])
    ns = SimpleNamespace(
        baseline_detail_csv=s4.PER_STATION_TEMPORAL_BASELINE_RUN_DIR / s4.BASELINE_DETAIL_NAME,
        future_swr_patch_detail_csv=s4.PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / s4.DETAIL_NAME,
        historical_field_detail_csv=s4.PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / s4.DETAIL_NAME,
        future_field_detail_csv=s4.PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / s4.DETAIL_NAME,
        out_dir=Path(args.out_dir) / "_stage4_cache",
        eval_split="test",
        selection="largest_gap",
        sample_count=1,
        seed=int(args.seed),
        issue_times=str(issue_arg_to_timestamp(args.issue_time)),
        station_sns=str(args.station_sn),
        future_step_min=15,
        hist_step_min=15,
        dpi=int(args.dpi),
        save_pdf=0,
        save_svg=0,
        y_min=float(args.stage4_y_min),
        y_max=float(args.stage4_y_max),
        fig_width=4.6,
        fig_height=1.45,
    )
    details, active_specs = s4.load_details(ns)
    merged = s4.align_details(details, active_specs, str(ns.eval_split))
    scores = s4.build_case_scores(merged, active_specs)
    chosen = s4.choose_cases(ns, merged, scores)
    if chosen.empty:
        raise RuntimeError(f"Stage-4 issue/station case not found: issue={args.issue_time}, station={args.station_sn}")
    sample = chosen.iloc[0]
    station = str(sample["station_sn"])
    issue = str(sample["issue_time"])
    sub = merged[(merged["station_sn"].astype(str) == station) & (merged["issue_time"].astype(str) == issue)].sort_values("h")
    hist_cols = s4.history_cols(sub)
    hist = sub.iloc[0][hist_cols].to_numpy(dtype=float) if hist_cols else np.asarray([], dtype=float)
    hist_x = (np.arange(len(hist), dtype=float) - (len(hist) - 1)) * float(ns.hist_step_min) if hist.size else np.asarray([])
    future_x = sub["h"].to_numpy(dtype=float) * float(ns.future_step_min)
    series = {
        "Ground truth": {"x": future_x, "y": sub["true_pv_norm"].to_numpy(dtype=float), "color": "#111111", "marker": "o", "lw": 2.4},
    }
    for spec in active_specs:
        tag = s4.safe_label(spec["model"])
        pred_col = f"pred_{tag}"
        if pred_col in sub.columns:
            series[str(spec["label"])] = {
                "x": future_x,
                "y": sub[pred_col].to_numpy(dtype=float),
                "color": str(spec["color"]),
                "marker": str(spec["marker"]),
                "lw": 2.0 if spec["model"] != "future_pv_field_patch" else 2.35,
            }
    return {
        "station_sn": station,
        "issue_time": issue,
        "hist_x": hist_x,
        "hist_y": hist,
        "series": series,
    }


def select_random_stage4_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    if int(args.random_count) <= 0:
        return []
    s4 = load_module("pipeline_stage4_station_case_selector", STAGE4_DIR / "13_visualize_stage4_case_comparisons.py", [STAGE4_DIR])
    ns = SimpleNamespace(
        baseline_detail_csv=s4.PER_STATION_TEMPORAL_BASELINE_RUN_DIR / s4.BASELINE_DETAIL_NAME,
        future_swr_patch_detail_csv=s4.PER_STATION_STAGE3_SWR_PATCH_RUN_DIR / s4.DETAIL_NAME,
        historical_field_detail_csv=s4.PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR / s4.DETAIL_NAME,
        future_field_detail_csv=s4.PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR / s4.DETAIL_NAME,
        out_dir=Path(args.out_dir) / "_stage4_case_selector_cache",
        eval_split="test",
        selection="random",
        sample_count=int(args.random_count),
        seed=int(args.seed) + 904,
        issue_times="",
        station_sns="",
        future_step_min=15,
        hist_step_min=15,
        dpi=int(args.dpi),
        save_pdf=0,
        save_svg=0,
        y_min=float(args.stage4_y_min),
        y_max=float(args.stage4_y_max),
        fig_width=4.6,
        fig_height=1.45,
    )
    details, active_specs = s4.load_details(ns)
    merged = s4.align_details(details, active_specs, str(ns.eval_split))
    scores = s4.build_case_scores(merged, active_specs)
    if int(args.random_scan_limit) > 0 and len(scores) > int(args.random_scan_limit):
        scores = scores.sample(n=int(args.random_scan_limit), random_state=int(args.seed)).reset_index(drop=True)
    primary_issue = str(issue_arg_to_timestamp(args.issue_time))
    primary_station = str(args.station_sn).strip()
    candidates = scores.copy()
    if primary_station:
        same_primary = candidates["station_sn"].astype(str).eq(primary_station) & candidates["issue_time"].astype(str).eq(primary_issue)
    else:
        same_primary = candidates["issue_time"].astype(str).eq(primary_issue)
    candidates = candidates[~same_primary].copy()
    stage2_keys = available_stage2_issue_keys(str(ns.eval_split))
    if stage2_keys:
        before = len(candidates)
        candidates["_issue_key"] = candidates["issue_time"].map(normalize_issue_key)
        filtered = candidates[candidates["_issue_key"].isin(stage2_keys)].copy()
        print(f"[random cases] Stage-2 issue filter: {before} -> {len(filtered)}", flush=True)
        if not filtered.empty:
            candidates = filtered
        else:
            print(
                "[random cases][warn] no direct overlap between Stage-4 issue times and the Stage-2 index; "
                "falling back to unfiltered Stage-4 random candidates.",
                flush=True,
            )
    if candidates.empty:
        return []
    chosen = candidates.sample(n=min(int(args.random_count), len(candidates)), random_state=int(args.seed) + 17)
    return [
        {"issue_time": str(pd.Timestamp(row["issue_time"]).strftime("%Y%m%d_%H%M%S")), "station_sn": str(row["station_sn"])}
        for _, row in chosen.iterrows()
    ]


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~(mask > 0.5), arr)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.055, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold", color="#111827")


def draw_arrow_between(fig, ax0, ax1, text: str) -> None:
    import matplotlib.patches as patches

    b0 = ax0.get_position()
    b1 = ax1.get_position()
    y = (b0.y0 + b0.y1) / 2
    arrow = patches.FancyArrowPatch(
        (b0.x1 + 0.006, y),
        (b1.x0 - 0.006, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=13,
        lw=1.15,
        color="#334155",
    )
    fig.patches.append(arrow)
    fig.text((b0.x1 + b1.x0) / 2, y - 0.045, text, ha="center", va="top", fontsize=7.4, color="#475569")


def draw_vertical_arrow(fig, ax0, ax1, text: str) -> None:
    import matplotlib.patches as patches

    b0 = ax0.get_position()
    b1 = ax1.get_position()
    x = (b0.x0 + b0.x1) / 2
    arrow = patches.FancyArrowPatch(
        (x, b0.y0 - 0.008),
        (x, b1.y1 + 0.008),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        lw=0.9,
        color="#475569",
    )
    fig.patches.append(arrow)
    fig.text(x + 0.022, (b0.y0 + b1.y1) / 2, text, ha="left", va="center", fontsize=6.3, color="#64748b")


def render_pipeline(args: argparse.Namespace, s2_case: dict, s3_case: dict, s4_case: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    swr_cmap = ListedColormap(s2_case["palette"], name="pipeline_swr")
    swr_cmap.set_bad((1, 1, 1, 0))
    pv_cmap = ListedColormap(s3_case["palette"], name="pipeline_pv")
    pv_cmap.set_bad((1, 1, 1, 0))

    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.86, 1.18, 1.06], left=0.075, right=0.97, top=0.965, bottom=0.075, hspace=0.30)
    flow_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax in flow_axes:
        ax.set_axis_off()

    max_horizon = int(s3_case["truth"].shape[0])
    rep_h = max(0, min(int(args.representative_horizon) - 1, max_horizon - 1))
    lead_min = int((rep_h + 1) * 15)
    swr_h = min(1, s2_case["large_pred"].shape[0] - 1)

    # (a) LCIG: large context to target-region SWR guidance.
    s1_gs = gs[0, 0].subgridspec(1, 3, width_ratios=[1.18, 0.12, 1.0], wspace=0.02)
    ax_large = fig.add_subplot(s1_gs[0, 0])
    lon = s2_case["large_lon"]
    lat = s2_case["large_lat"]
    im_swr = ax_large.imshow(
        s2_case["large_pred"][swr_h],
        origin="lower",
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        cmap=swr_cmap,
        vmin=0,
        vmax=float(args.lcig_vmax),
        interpolation="nearest",
        aspect="auto",
    )
    lon0, lon1, lat0, lat1 = s2_case["ningbo_box"]
    ax_large.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, edgecolor="#e53935", linewidth=1.35, linestyle=(0, (2.0, 1.4))))
    ax_large.set_title("LCIG large-context forecast", fontsize=7.2, pad=2)
    ax_large.set_xticks([])
    ax_large.set_yticks([])
    for spine in ax_large.spines.values():
        spine.set_visible(False)
    add_panel_label(ax_large, "(a)")

    ax_arrow_a = fig.add_subplot(s1_gs[0, 1])
    ax_arrow_a.set_axis_off()
    ax_arrow_a.text(0.5, 0.50, ">", ha="center", va="center", fontsize=12, fontweight="bold", color="#475569")
    ax_crop = fig.add_subplot(s1_gs[0, 2])
    crop = s2_case["crops"][swr_h]["pred"]
    ax_crop.imshow(crop, origin="lower", cmap=swr_cmap, vmin=0, vmax=float(args.lcig_vmax), interpolation="nearest", aspect="auto")
    ax_crop.set_title("Target-region SWR guidance", fontsize=7.2, pad=2)
    ax_crop.set_xticks([])
    ax_crop.set_yticks([])
    for spine in ax_crop.spines.values():
        spine.set_visible(False)
    cax_swr = fig.add_axes([0.925, 0.746, 0.012, 0.12])
    cb = fig.colorbar(im_swr, cax=cax_swr)
    cb.ax.tick_params(labelsize=5.5, length=1.2)
    cb.set_label("SWR", fontsize=5.7, labelpad=1)

    # (b) PV field: reference, prediction used by FG-MSF, and target-centered patch.
    s2_gs = gs[1, 0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 0.13, 0.58], wspace=0.05)
    h = s3_case["horizons"][0] if s3_case["horizons"] else rep_h
    h = max(0, min(int(h), s3_case["truth"].shape[0] - 1))
    mask = s3_case["mask"][h]
    ax_gt = fig.add_subplot(s2_gs[0, 0])
    im_pv = ax_gt.imshow(masked(s3_case["truth"][h], mask), origin="lower", cmap=pv_cmap, vmin=float(args.pv_vmin), vmax=float(args.pv_vmax), interpolation="nearest")
    ax_gt.set_title(f"Reference GT, h={lead_min} min", fontsize=7.0, pad=2)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_visible(False)
    add_panel_label(ax_gt, "(b)")

    ax_pred = fig.add_subplot(s2_gs[0, 1])
    ax_pred.imshow(masked(s3_case["proposed"][h], mask), origin="lower", cmap=pv_cmap, vmin=float(args.pv_vmin), vmax=float(args.pv_vmax), interpolation="nearest")
    ax_pred.set_title("SAG-PVFF used by FG-MSF", fontsize=7.0, pad=2)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    loc = s3_case.get("station_loc")
    patch, rect = extract_patch(s3_case["proposed"][h], loc, 5)
    if loc is not None:
        ax_pred.scatter([loc["x"]], [loc["y"]], s=18, marker="x", color="#111827", linewidths=1.0, zorder=5)
        if rect is not None:
            x0, x1, y0, y1 = rect
            ax_pred.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, edgecolor="#111827", linewidth=0.85))
    for spine in ax_pred.spines.values():
        spine.set_visible(False)

    ax_arrow_b = fig.add_subplot(s2_gs[0, 2])
    ax_arrow_b.set_axis_off()
    ax_arrow_b.text(0.5, 0.50, ">", ha="center", va="center", fontsize=12, fontweight="bold", color="#475569")
    ax_patch = fig.add_subplot(s2_gs[0, 3])
    if patch is not None:
        ax_patch.imshow(patch, origin="lower", cmap=pv_cmap, vmin=float(args.pv_vmin), vmax=float(args.pv_vmax), interpolation="nearest")
    else:
        ax_patch.text(0.5, 0.5, "patch\nN/A", ha="center", va="center", fontsize=7, color="#64748b")
    ax_patch.set_title("5x5 patch", fontsize=7.0, pad=2)
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    for spine in ax_patch.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(0.6)
    cax_pv = fig.add_axes([0.925, 0.425, 0.012, 0.13])
    cb = fig.colorbar(im_pv, cax=cax_pv)
    cb.ax.tick_params(labelsize=5.5, length=1.2)
    cb.set_label("PV", fontsize=5.7, labelpad=1)

    # (c) Station prediction curve: final local baseline vs FG-MSF.
    ax_curve = fig.add_subplot(gs[2, 0])
    add_panel_label(ax_curve, "(c)")
    if len(s4_case["hist_x"]):
        ax_curve.plot(s4_case["hist_x"], s4_case["hist_y"], color="#7A8288", marker=".", markersize=4.8, linewidth=1.8, label="Historical PV")
    ax_curve.axvline(0, color="#A7ADB2", linestyle="--", linewidth=1.0)
    ax_curve.axvline(lead_min, color="#D97706", linestyle=":", linewidth=1.0)
    preferred = ["Ground truth", "BiLSTM+NWP", "Temporal baseline", "FG-MSF"]
    seen_labels: set[str] = set()
    for name in preferred:
        item = s4_case["series"].get(name)
        if item is None:
            continue
        label = "BiLSTM+NWP" if name == "Temporal baseline" else name
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ax_curve.plot(item["x"], item["y"], color=item["color"], marker=item["marker"], markersize=3.3, linewidth=item["lw"], label=label)
    ax_curve.set_ylim(float(args.stage4_y_min), float(args.stage4_y_max))
    ax_curve.set_xlim(-165, 125)
    ax_curve.set_xlabel("Time relative to forecast issue (min)", fontsize=7.5)
    ax_curve.set_ylabel("PV power (p.u.)", fontsize=7.5)
    ax_curve.grid(axis="y", color="#D9DEE3", linewidth=0.75, alpha=0.82)
    ax_curve.tick_params(axis="both", labelsize=7, length=2)
    ax_curve.legend(frameon=False, loc="upper left", fontsize=6.0, ncol=2, handlelength=1.6, columnspacing=0.8)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.set_title(f"Target station {s4_case['station_sn']}", fontsize=7.2, pad=2)

    issue = issue_arg_to_timestamp(args.issue_time)
    draw_vertical_arrow(fig, ax_crop, ax_gt, "SWR guidance")
    draw_vertical_arrow(fig, ax_patch, ax_curve, "field patch")
    fig.text(0.50, 0.025, f"{issue:%Y-%m-%d %H:%M} BJT, h={lead_min} min", ha="center", va="center", fontsize=6.8, color="#334155")

    stem = f"{args.out_prefix}_{issue:%Y%m%d_%H%M%S}"
    if s4_case.get("station_sn"):
        stem += f"_station{s4_case['station_sn']}"
    outputs = {}
    if int(args.save_png):
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.04)
        outputs["png"] = str(png)
    if int(args.save_pdf):
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        outputs["pdf"] = str(pdf)
    plt.close(fig)
    meta = {
        "issue_time": f"{issue:%Y-%m-%d %H:%M:%S}",
        "stage2_idx": s2_case.get("idx"),
        "stage3_idx": s3_case.get("idx"),
        "stage4_station_sn": s4_case.get("station_sn"),
        "outputs": outputs,
    }
    meta_path = out_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["meta"] = str(meta_path)
    return outputs


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("==== Generate three-stage case pipeline from model outputs ====", flush=True)
    cases = [{"issue_time": str(args.issue_time), "station_sn": str(args.station_sn), "case_role": "requested"}]
    print("[0/4] Selecting additional random Stage-4 station-issue cases", flush=True)
    for item in select_random_stage4_cases(args):
        item["case_role"] = "random"
        cases.append(item)

    records = []
    for rank, item in enumerate(cases, start=1):
        case_args = copy.copy(args)
        case_args.issue_time = str(item["issue_time"])
        case_args.station_sn = str(item.get("station_sn", ""))
        case_args.out_prefix = f"{args.out_prefix}_case{rank:02d}_{item.get('case_role', 'case')}"
        print(
            f"==== Case {rank}/{len(cases)} role={item.get('case_role')} issue={case_args.issue_time} station={case_args.station_sn or '[auto]'} ====",
            flush=True,
        )
        try:
            print("[1/4] Loading FG-MSF station outputs", flush=True)
            s4_case = load_stage4_case(case_args)
            print("[2/4] Loading LCIG case outputs", flush=True)
            s2_case = load_stage2_case(case_args)
            print("[3/4] Loading SAG-PVFF case outputs", flush=True)
            s3_case = load_stage3_case(case_args, station_sn=str(s4_case["station_sn"]))
            print("[4/4] Rendering end-to-end figure", flush=True)
            outputs = render_pipeline(case_args, s2_case, s3_case, s4_case)
            record = {
                "rank": rank,
                "case_role": item.get("case_role", ""),
                "issue_time": str(case_args.issue_time),
                "station_sn": str(s4_case.get("station_sn", "")),
                "status": "ok",
                **outputs,
            }
            for key, value in outputs.items():
                print(f"{key}: {value}", flush=True)
        except Exception as exc:
            record = {
                "rank": rank,
                "case_role": item.get("case_role", ""),
                "issue_time": str(case_args.issue_time),
                "station_sn": str(case_args.station_sn),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"[failed] {record['error']}", flush=True)
        records.append(record)

    manifest = Path(args.out_dir) / "three_stage_case_pipeline_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    ok = sum(1 for row in records if row.get("status") == "ok")
    print(f"==== Completed: {ok}/{len(records)} figures rendered ====", flush=True)
    print(f"manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()

