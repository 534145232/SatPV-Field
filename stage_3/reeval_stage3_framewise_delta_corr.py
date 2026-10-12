#!/usr/bin/env python3
"""Recompute Stage-3 DeltaCorr with a frame-wise, batch-independent definition.

No model is retrained. Existing checkpoints are loaded and evaluated on the same
common SWR-anchor issue subset used by the Stage-3 paper tables. For each test
sample and forecast horizon, the script computes Pearson correlation between
predicted and true PV-field increments over supported pixels only:

    corr(pred_h - latest_observed, truth_h - latest_observed).

Frames with near-zero predicted or true increment variance are skipped. The
reported DeltaCorr is the arithmetic mean over all valid sample-horizon frames.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MOTION_ABLATION_DIR = SCRIPT_DIR / "ablation_motion_branch"
TEMPORAL_COLLAPSE_DIR = SCRIPT_DIR / "ablation_temporal_collapse"
SWR_DEPENDENCY_DIR = SCRIPT_DIR / "ablation_swr_dependency"
for path in [MOTION_ABLATION_DIR, TEMPORAL_COLLAPSE_DIR, SWR_DEPENDENCY_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage3_base_pvnorm_field_model as base
from stage3_paths import RUNS_DIR, SWR_BASE_CHECKPOINT
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir


mfp = None
motion_eval = None
install_swr_dependency_dataset = None


BASE_MODEL_ORDER = ["persistence_last", "pv_only", "pv_h8", "pv_nwp", "pv_h8_nwp"]
LABELS = {
    "persistence_last": "Persistence",
    "pv_only": "PV-only U-Net",
    "pv_h8": "PV+Past SWR U-Net",
    "pv_nwp": "PV+NWP U-Net",
    "pv_h8_nwp": "PV+Past SWR+NWP U-Net",
    "sag_pvff": "SAG-PVFF",
}


def parse_args() -> argparse.Namespace:
    base_defaults = dict(base.PYCHARM_DEFAULTS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=Path(base_defaults["selection_dir"]))
    parser.add_argument("--target_run_dir", type=Path, default=Path(base_defaults["target_run_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(base_defaults["frame_cache_dir"]))
    parser.add_argument("--multisource_cache_dir", type=Path, default=Path(base_defaults["multisource_cache_dir"]))
    parser.add_argument("--base_run_dir", type=Path, default=RUNS_DIR / "base_pvnorm_field_model")
    parser.add_argument(
        "--sag_run_root",
        type=Path,
        default=RUNS_DIR / "ablation_motion_branch_last_only",
        help="Run root containing warp_direct_gate/SAG-PVFF checkpoint.",
    )
    parser.add_argument(
        "--sag_motion_branch",
        type=str,
        default="warp_direct_gate",
        choices=["warp_direct_gate", "direct_only", "warp_only", "fixed_average"],
    )
    parser.add_argument("--sag_temporal_collapse", type=str, default="last_only", choices=["last_mean", "last_only"])
    parser.add_argument("--sag_swr_dependency", type=str, default="full_swr_future")
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--swr_checkpoint", type=Path, default=SWR_BASE_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "framewise_delta_corr")
    parser.add_argument("--models", type=str, default=",".join([*BASE_MODEL_ORDER, "sag_pvff"]))
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--grid_size", type=int, default=int(base_defaults["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(base_defaults["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(base_defaults["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(base_defaults["min_grid_weight"]))
    parser.add_argument("--freq_minutes", type=int, default=int(base_defaults["freq_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(base_defaults["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(base_defaults["future_frames"]))
    parser.add_argument("--delta_scale", type=float, default=float(base_defaults["delta_scale"]))
    parser.add_argument("--batch_size", type=int, default=int(base_defaults["batch_size"]))
    parser.add_argument("--num_workers", type=int, default=int(base_defaults["num_workers"]))
    parser.add_argument("--device", type=str, default=str(base_defaults["device"]))
    parser.add_argument("--seed", type=int, default=int(base_defaults["seed"]))
    parser.add_argument("--corr_eps", type=float, default=1.0e-12)
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def load_motion_modules() -> None:
    global mfp, motion_eval, install_swr_dependency_dataset
    if mfp is not None and motion_eval is not None and install_swr_dependency_dataset is not None:
        return
    import stage3_motion_fusion_model_impl as _mfp
    import eval_all_motion_branches as _motion_eval
    from swr_dependency import install_swr_dependency_dataset as _install_swr_dependency_dataset

    mfp = _mfp
    motion_eval = _motion_eval
    install_swr_dependency_dataset = _install_swr_dependency_dataset


def requested_models(text: str) -> list[str]:
    models = [part.strip() for part in str(text).split(",") if part.strip()]
    valid = set(BASE_MODEL_ORDER) | {"sag_pvff"}
    bad = [model for model in models if model not in valid]
    if bad:
        raise ValueError(f"Unsupported models: {bad}; valid={sorted(valid)}")
    return models


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 2:
        return float("nan")
    a = a[ok] - float(np.mean(a[ok]))
    b = b[ok] - float(np.mean(b[ok]))
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom <= float(eps):
        return float("nan")
    return float(np.sum(a * b) / denom)


class DeltaCorrAccumulator:
    def __init__(self, eps: float):
        self.eps = float(eps)
        self.sum_by_h: dict[int, float] = defaultdict(float)
        self.count_by_h: dict[int, int] = defaultdict(int)
        self.skip_by_h: dict[int, int] = defaultdict(int)

    def update(self, pred_z: np.ndarray, y_z: np.ndarray, mask: np.ndarray, x_z: np.ndarray, pv_stats: dict) -> None:
        pred = base._denorm_field(np.asarray(pred_z), pv_stats)
        truth = base._denorm_field(np.asarray(y_z), pv_stats)
        latest = base._denorm_field(np.asarray(x_z)[:, 0, -1], pv_stats)
        latest_seq = np.repeat(latest[:, None], pred.shape[1], axis=1)
        pred_delta = pred - latest_seq
        true_delta = truth - latest_seq
        support = np.asarray(mask) > 0.5
        for b in range(pred.shape[0]):
            for h in range(pred.shape[1]):
                valid = support[b, h] & np.isfinite(pred_delta[b, h]) & np.isfinite(true_delta[b, h])
                corr = _pearson_corr(pred_delta[b, h][valid], true_delta[b, h][valid], self.eps)
                if np.isfinite(corr):
                    self.sum_by_h[h] += float(corr)
                    self.count_by_h[h] += 1
                else:
                    self.skip_by_h[h] += 1

    def rows(self, model: str, method: str, split: str, future_frames: int) -> list[dict[str, Any]]:
        rows = []
        total_sum = 0.0
        total_count = 0
        total_skip = 0
        for h in range(int(future_frames)):
            count = int(self.count_by_h.get(h, 0))
            skip = int(self.skip_by_h.get(h, 0))
            value = float(self.sum_by_h.get(h, 0.0) / count) if count else float("nan")
            rows.append(
                {
                    "model": model,
                    "Method": method,
                    "eval_split": split,
                    "horizon": f"h{h + 1}",
                    "lead_min": int((h + 1) * 15),
                    "delta_corr_framewise_mean": value,
                    "valid_frame_count": count,
                    "skipped_frame_count": skip,
                }
            )
            total_sum += float(self.sum_by_h.get(h, 0.0))
            total_count += count
            total_skip += skip
        rows.append(
            {
                "model": model,
                "Method": method,
                "eval_split": split,
                "horizon": "all",
                "lead_min": -1,
                "delta_corr_framewise_mean": float(total_sum / total_count) if total_count else float("nan"),
                "valid_frame_count": int(total_count),
                "skipped_frame_count": int(total_skip),
            }
        )
        return rows


def make_base_args(args: argparse.Namespace) -> SimpleNamespace:
    defaults = dict(base.PYCHARM_DEFAULTS)
    defaults.update(
        {
            "selection_dir": Path(args.selection_dir),
            "target_run_dir": Path(args.target_run_dir),
            "frame_cache_dir": Path(args.frame_cache_dir),
            "multisource_cache_dir": Path(args.multisource_cache_dir),
            "out_dir": Path(args.base_run_dir),
            "grid_size": int(args.grid_size),
            "gaussian_sigma_cells": float(args.gaussian_sigma_cells),
            "gaussian_radius_cells": int(args.gaussian_radius_cells),
            "min_grid_weight": float(args.min_grid_weight),
            "freq_minutes": int(args.freq_minutes),
            "past_frames": int(args.past_frames),
            "future_frames": int(args.future_frames),
            "delta_scale": float(args.delta_scale),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "seed": int(args.seed),
        }
    )
    return SimpleNamespace(**defaults)


def build_base_data(args: argparse.Namespace, swr_meta: pd.DataFrame):
    from torch.utils.data import DataLoader

    bargs = make_base_args(args)
    split_df = base._load_station_split(bargs)
    bounds = base._bounds(split_df)
    ms_cache = base._load_multisource_cache(Path(bargs.multisource_cache_dir))
    time_df = ms_cache["time_df"].copy()
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    base._assert_issue_time_alignment(bargs, time_df)
    base._assert_target_exclusion(bargs)
    tdf = time_df[time_df["time_split"].astype(str) == str(args.split)].reset_index(drop=True)
    original_len = len(tdf)
    tdf, _cache_indices, _sub_meta = mfp._subset_time_df_by_swr_meta(tdf, swr_meta, str(args.split))
    print(f"[base common subset][{args.split}] issue rows: {original_len} -> {len(tdf)}", flush=True)

    pv_cache = base.ReadOnlyDiskFrameCache(Path(bargs.frame_cache_dir), time_df, bargs)
    pv_stats = base._compute_or_load_pv_field_stats(pv_cache, time_df, Path(args.base_run_dir))
    ms_stats = base._compute_or_load_multisource_stats(ms_cache, Path(args.base_run_dir), time_df)
    datasets = {}
    loaders = {}
    for model_name in BASE_MODEL_ORDER:
        dataset_model_name = "pv_only" if model_name == "persistence_last" else model_name
        ds = base.TargetExcludedPVNormFieldDataset(tdf, pv_cache, ms_cache, bounds, pv_stats, ms_stats, dataset_model_name, bargs)
        datasets[model_name] = ds
        loaders[model_name] = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    return bargs, datasets, loaders, pv_stats


def eval_base_model(model_name: str, loader, dataset, pv_stats: dict, args: argparse.Namespace, device: str) -> list[dict[str, Any]]:
    import torch

    acc = DeltaCorrAccumulator(float(args.corr_eps))
    model = None
    if model_name != "persistence_last":
        in_channels, future_channels = dataset.source_dims()
        ckpt_path = Path(args.base_run_dir) / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing base checkpoint for {model_name}: {ckpt_path}")
        model = base.make_residual_3d_unet_z(
            int(args.future_frames),
            in_channels=int(in_channels),
            future_channels=int(future_channels),
            delta_scale=float(args.delta_scale),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[load base] {model_name}: {ckpt_path}", flush=True)
    else:
        print("[eval base] persistence_last: repeat latest historical PV field", flush=True)

    with torch.no_grad():
        for step, (x, future_feat, y, mask) in enumerate(loader, start=1):
            if model_name == "persistence_last":
                x_np = x.numpy()
                pred = x_np[:, 0, -1][:, None].repeat(y.shape[1], axis=1)
            else:
                pred = model(x.to(device), future_feat.to(device)).detach().cpu().numpy()
                x_np = x.numpy()
            acc.update(pred, y.numpy(), mask.numpy(), x_np, pv_stats)
            if step % 100 == 0:
                print(f"[framewise DeltaCorr][{model_name}] batches={step}/{len(loader)}", flush=True)
    return acc.rows(model_name, LABELS[model_name], str(args.split), int(args.future_frames))


def make_motion_eval_args(args: argparse.Namespace, anchor_cache_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=Path(args.sag_run_root),
        temporal_collapse=str(args.sag_temporal_collapse),
        swr_dependency=str(args.sag_swr_dependency),
        motion_branch=str(args.sag_motion_branch),
        swr_anchor_cache_dir=Path(anchor_cache_dir),
        swr_checkpoint=Path(args.swr_checkpoint),
        out_dir=Path(args.out_dir) / "_sag_loader_cache",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        sample_count=0,
        visual_horizons="1,4,8",
        dpi=120,
        seed=int(args.seed),
    )


def eval_sag_pvff(args: argparse.Namespace, anchor_cache_dir: Path, device: str) -> list[dict[str, Any]]:
    import torch

    install_swr_dependency_dataset(mfp, str(args.sag_swr_dependency))
    motion_args = make_motion_eval_args(args, anchor_cache_dir)
    motion_args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_args, datasets, loaders, pv_stats, model_kwargs = motion_eval.build_data(motion_args, anchor_cache_dir, motion_args.out_dir)
    models, _skipped = motion_eval.load_models(
        Path(args.sag_run_root),
        model_kwargs,
        device,
        str(args.sag_temporal_collapse),
        str(args.sag_motion_branch),
    )
    item = models[str(args.sag_motion_branch)]
    model = item["model"]
    acc = DeltaCorrAccumulator(float(args.corr_eps))
    with torch.no_grad():
        for step, (x, future_feat, swr, y, mask) in enumerate(loaders[str(args.split)], start=1):
            pred = model(x.to(device), future_feat.to(device), swr.to(device)).detach().cpu().numpy()
            acc.update(pred, y.numpy(), mask.numpy(), x.numpy(), pv_stats)
            if step % 100 == 0:
                print(f"[framewise DeltaCorr][sag_pvff] batches={step}/{len(loaders[str(args.split)])}", flush=True)
    rows = acc.rows("sag_pvff", LABELS["sag_pvff"], str(args.split), int(dataset_args.future_frames))
    for row in rows:
        row["checkpoint"] = str(item["checkpoint"])
        row["sag_motion_branch"] = str(args.sag_motion_branch)
        row["sag_temporal_collapse"] = str(args.sag_temporal_collapse)
        row["sag_swr_dependency"] = str(args.sag_swr_dependency)
    return rows


def fmt(value: Any, digits: int) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def print_outputs(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    all_rows = summary[summary["horizon"].astype(str).eq("all")].copy()
    order = {model: idx for idx, model in enumerate([*BASE_MODEL_ORDER, "sag_pvff"])}
    all_rows["_order"] = all_rows["model"].map(order)
    all_rows = all_rows.sort_values("_order").drop(columns=["_order"])
    cols = ["Method", "delta_corr_framewise_mean", "valid_frame_count", "skipped_frame_count"]
    print("\n==== Frame-wise DeltaCorr for Table II methods ====")
    print(all_rows[cols].to_string(index=False, formatters={"delta_corr_framewise_mean": lambda x: fmt(x, int(args.digits))}))
    print("\n==== LaTeX-ready rows ====")
    for _, row in all_rows.iterrows():
        print(f"{row['Method']} & {fmt(row['delta_corr_framewise_mean'], int(args.digits))} \\\\")

    horizon = summary[summary["horizon"].astype(str).ne("all")].copy()
    horizon["_order"] = horizon["model"].map(order)
    horizon = horizon.sort_values(["_order", "lead_min"]).drop(columns=["_order"])
    print("\n==== Horizon-wise frame-wise DeltaCorr ====")
    print(
        horizon[["Method", "lead_min", "delta_corr_framewise_mean", "valid_frame_count", "skipped_frame_count"]].to_string(
            index=False,
            formatters={"delta_corr_framewise_mean": lambda x: fmt(x, int(args.digits))},
        )
    )


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    load_motion_modules()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    anchor_cache_dir = Path(args.swr_anchor_cache_dir) if args.swr_anchor_cache_dir is not None else resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(anchor_cache_dir, Path(args.swr_checkpoint))
    _swr_arr, swr_meta, _swr_info = mfp._load_prebuilt_swr_anchor_cache(anchor_cache_dir)
    print(f"[common subset] SWR-anchor cache: {anchor_cache_dir} rows={len(swr_meta)}", flush=True)

    models = requested_models(str(args.models))
    rows: list[dict[str, Any]] = []
    base_models = [model for model in models if model in BASE_MODEL_ORDER]
    if base_models:
        _bargs, datasets, loaders, pv_stats = build_base_data(args, swr_meta)
        for model_name in base_models:
            rows.extend(eval_base_model(model_name, loaders[model_name], datasets[model_name], pv_stats, args, device))
    if "sag_pvff" in models:
        rows.extend(eval_sag_pvff(args, anchor_cache_dir, device))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        order = {model: idx for idx, model in enumerate([*BASE_MODEL_ORDER, "sag_pvff"])}
        summary["_order"] = summary["model"].map(order)
        summary["_horder"] = summary["horizon"].map({"all": 999}).fillna(summary["lead_min"]).astype(int)
        summary = summary.sort_values(["_order", "_horder"]).drop(columns=["_order", "_horder"]).reset_index(drop=True)
    out_csv = args.out_dir / "stage3_framewise_delta_corr_summary.csv"
    summary.to_csv(out_csv, index=False)
    print_outputs(summary, args)
    print(f"\nsaved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
