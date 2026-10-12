#!/usr/bin/env python3
"""Evaluate available Stage 3 motion-branch ablation checkpoints together."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))
TEMPORAL_COLLAPSE_DIR = STAGE3_DIR / "ablation_temporal_collapse"
if str(TEMPORAL_COLLAPSE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_COLLAPSE_DIR))
SWR_DEPENDENCY_DIR = STAGE3_DIR / "ablation_swr_dependency"
if str(SWR_DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(SWR_DEPENDENCY_DIR))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import patch_last_only_collapse
from motion_branch import MOTION_BRANCHES, patch_motion_branch
from swr_dependency import SWR_DEPENDENCY_MODES, install_swr_dependency_dataset
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir


f04 = module.f04
RUN_ROOT = STAGE3_DIR / "runs" / "ablation_motion_branch"
TEMPORAL_COLLAPSES = ("last_mean", "last_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 3 direct-only, warp-only and gated motion branches.")
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT)
    parser.add_argument("--temporal_collapse", choices=TEMPORAL_COLLAPSES, default="last_mean")
    parser.add_argument("--swr_dependency", choices=SWR_DEPENDENCY_MODES, default="full_swr_future")
    parser.add_argument("--motion_branch", choices=("all", *MOTION_BRANCHES), default="all")
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--swr_checkpoint", type=Path, default=SWR_BASE_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--visual_horizons", type=str, default="1,4,8")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_dataset_args(args: argparse.Namespace, anchor_cache_dir: Path, out_dir: Path) -> SimpleNamespace:
    defaults = dict(module.PYCHARM_DEFAULTS)
    defaults.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(out_dir),
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "h8_sample_cache_dir": "",
            "h8_clear_sky_dir": "",
            "swr_checkpoint": str(args.swr_checkpoint),
            "base_model_name": "pv_nwp",
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "sample_count": int(args.sample_count),
            "visual_horizons": str(args.visual_horizons),
            "dpi": int(args.dpi),
            "seed": int(args.seed),
        }
    )
    return SimpleNamespace(**defaults)


def build_data(args: argparse.Namespace, anchor_cache_dir: Path, out_dir: Path):
    from torch.utils.data import DataLoader

    dataset_args = make_dataset_args(args, anchor_cache_dir, out_dir)
    f04._assert_target_exclusion(dataset_args)
    station_split = f04._load_station_split(dataset_args)
    bounds = module._bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(dataset_args.multisource_cache_dir))
    f04._assert_issue_time_alignment(dataset_args, ms_cache["time_df"])
    time_df = pd.read_csv(Path(dataset_args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(f04._timestamp_key)
    pv_cache = f04.ReadOnlyDiskFrameCache(Path(dataset_args.frame_cache_dir), time_df, dataset_args)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, out_dir, time_df)
    swr_arr_all, swr_meta_all, _ = module._load_prebuilt_swr_anchor_cache(anchor_cache_dir)

    base_datasets, swr_arrays, datasets, loaders = {}, {}, {}, {}
    for split in ["train", "val", "test"]:
        sub, indices, _ = module._subset_time_df_by_swr_meta(time_df, swr_meta_all, split)
        base_datasets[split] = f04.TargetExcludedPVNormFieldDataset(
            sub,
            pv_cache,
            ms_cache,
            bounds,
            pv_stats,
            ms_stats,
            str(dataset_args.base_model_name),
            dataset_args,
        )
        swr_arrays[split] = np.asarray(swr_arr_all[indices], dtype=np.float32)
    swr_stats = module._compute_swr_anchor_stats(swr_arrays["train"], out_dir, force=False)
    for split in ["train", "val", "test"]:
        datasets[split] = module.SWRAnchorPVFieldDataset(
            base_datasets[split], swr_arrays[split], dataset_args, swr_stats
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
    base_in, future_channels = base_datasets["train"].source_dims()
    model_kwargs = {
        "future_frames": int(dataset_args.future_frames),
        "in_channels": int(base_in) + 2,
        "future_channels": int(future_channels) + 1,
        "anchor_channels": 2,
        "pv_hidden": int(dataset_args.pv_hidden_channels),
        "anchor_hidden": int(dataset_args.anchor_channels),
        "horizon_embed_channels": int(dataset_args.horizon_embed_channels),
        "dropout": float(dataset_args.dropout),
        "swr_history_frames": int(dataset_args.swr_history_frames),
        "swr_future_anchors": int(dataset_args.swr_future_anchors),
        "max_flow_pixels": float(dataset_args.max_flow_pixels),
    }
    return dataset_args, datasets, loaders, pv_stats, model_kwargs


def patch_temporal_collapse(model, temporal_collapse: str):
    if temporal_collapse == "last_mean":
        return model
    if temporal_collapse == "last_only":
        return patch_last_only_collapse(model)
    raise ValueError(f"Unknown temporal collapse: {temporal_collapse}")


def load_models(run_root: Path, model_kwargs: dict, device: str, temporal_collapse: str, motion_branch: str):
    import torch

    models, skipped = {}, {}
    modes = MOTION_BRANCHES if motion_branch == "all" else (motion_branch,)
    for mode in modes:
        model_name = f"swr_anchor_motion_fusion_pv_nwp_motion_{mode}"
        checkpoint = run_root / mode / f"{model_name}_best.pt"
        if not checkpoint.exists():
            skipped[mode] = str(checkpoint)
            print(f"[SKIP] {mode}: checkpoint not found: {checkpoint}", flush=True)
            continue
        model = patch_motion_branch(module.make_swr_anchor_attention_model(**model_kwargs), mode)
        model = patch_temporal_collapse(model, temporal_collapse)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        models[mode] = {"model": model, "checkpoint": checkpoint, "model_name": model_name}
        print(f"[LOAD] {mode}: {checkpoint}", flush=True)
    if not models:
        raise FileNotFoundError(f"No completed motion-branch checkpoints below {run_root}")
    return models, skipped


def parse_horizons(text: str, future_frames: int) -> list[int]:
    values = [int(value.strip()) - 1 for value in str(text).split(",") if value.strip()]
    return [value for value in values if 0 <= value < future_frames] or [0, 3, future_frames - 1]


def save_joint_samples(out_dir: Path, models: dict, dataset, args: argparse.Namespace, pv_stats: dict, device: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sample_dir = out_dir / "joint_sample_comparisons"
    sample_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(args.sample_count), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int) if n else []
    horizons = parse_horizons(args.visual_horizons, int(dataset.args.future_frames))
    rows = []
    diagnostic_rows = []
    for rank, idx in enumerate(indices, start=1):
        x, future_feat, swr, y, mask = dataset[int(idx)]
        truth = f04._denorm_field(y, pv_stats)
        predictions = {}
        with torch.no_grad():
            for mode, item in models.items():
                pred, _attention, _aligned, _warped, _direct, gate, flow = item["model"](
                    torch.from_numpy(x[None]).to(device),
                    torch.from_numpy(future_feat[None]).to(device),
                    torch.from_numpy(swr[None]).to(device),
                    return_attention=True,
                )
                predictions[mode] = f04._denorm_field(pred.detach().cpu().numpy()[0], pv_stats)
                flow_abs = torch.sqrt(torch.sum(flow.square(), dim=1))
                diagnostic_rows.append(
                    {
                        "rank": rank,
                        "dataset_idx": int(idx),
                        "motion_branch": mode,
                        "flow_abs_mean_pixels": float(flow_abs.mean().item()),
                        "flow_abs_p95_pixels": float(torch.quantile(flow_abs.flatten(), 0.95).item()),
                        "gate_mean": float(gate.mean().item()),
                        "gate_p05": float(torch.quantile(gate.flatten(), 0.05).item()),
                        "gate_p95": float(torch.quantile(gate.flatten(), 0.95).item()),
                    }
                )
        valid_values = [truth[mask > 0.5]]
        valid_values.extend(pred[mask > 0.5] for pred in predictions.values())
        scale_values = np.concatenate(valid_values)
        lo, hi = float(np.nanquantile(scale_values, 0.02)), float(np.nanquantile(scale_values, 0.98))
        errors = np.concatenate([(pred - truth)[mask > 0.5] for pred in predictions.values()])
        error_hi = max(0.05, float(np.nanquantile(np.abs(errors), 0.98)))
        fig, axes = plt.subplots(
            len(horizons),
            1 + 2 * len(predictions),
            figsize=(3.4 * (1 + 2 * len(predictions)), 3.3 * len(horizons)),
            squeeze=False,
        )
        for row, horizon in enumerate(horizons):
            panels = [("truth", truth[horizon], "viridis", lo, hi)]
            for mode, pred in predictions.items():
                panels.append((mode, pred[horizon], "viridis", lo, hi))
                panels.append((f"{mode} error", pred[horizon] - truth[horizon], "coolwarm", -error_hi, error_hi))
            for ax, (title, values, cmap, vmin, vmax) in zip(axes[row], panels):
                show = np.asarray(values, dtype=np.float32).copy()
                show[mask[horizon] <= 0.5] = np.nan
                image = ax.imshow(show, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title} h{horizon + 1}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        issue = str(dataset.time_df.iloc[int(idx)]["issue_time"])
        fig.suptitle(f"Stage 3 motion-branch ablation | issue={issue}", y=0.995)
        fig.tight_layout()
        path = sample_dir / f"motion_branch_case_{rank:03d}_idx_{idx:05d}.png"
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        rows.append({"rank": rank, "dataset_idx": int(idx), "issue_time": issue, "figure": str(path)})
    pd.DataFrame(rows).to_csv(out_dir / "joint_sample_comparisons.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        out_dir / "joint_sample_branch_diagnostics.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    anchor_cache_dir = args.swr_anchor_cache_dir or resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(anchor_cache_dir, args.swr_checkpoint)
    out_dir = args.out_dir or (args.run_root / "eval_all")
    out_dir.mkdir(parents=True, exist_ok=True)
    install_swr_dependency_dataset(module, str(args.swr_dependency))
    dataset_args, datasets, loaders, pv_stats, model_kwargs = build_data(args, anchor_cache_dir, out_dir)
    device = module._device(args.device)
    models, skipped = load_models(args.run_root, model_kwargs, device, str(args.temporal_collapse), str(args.motion_branch))

    frames = []
    for mode, item in models.items():
        for split in ["val", "test"]:
            frame = module.evaluate_model(
                item["model"], loaders[split], device, item["model_name"], split, pv_stats, float(dataset_args.large_delta_threshold)
            )
            frame.insert(0, "motion_branch", mode)
            frame.insert(1, "swr_dependency", str(args.swr_dependency))
            frames.append(frame)
        detailed_dir = out_dir / f"detailed_{mode}"
        detailed_dir.mkdir(parents=True, exist_ok=True)
        module.save_sample_comparisons(item["model"], datasets["test"], device, detailed_dir, dataset_args, pv_stats)
    summary = pd.concat(frames, ignore_index=True)
    summary_path = out_dir / "motion_branch_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_joint_samples(out_dir, models, datasets["test"], args, pv_stats, device)
    with (out_dir / "motion_branch_eval_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "purpose": "Stage 3 motion branch ablation: direct only, warp only, and full gated fusion.",
                "temporal_collapse": str(args.temporal_collapse),
                "swr_dependency": str(args.swr_dependency),
                "swr_anchor_cache_dir": str(anchor_cache_dir),
                "swr_checkpoint": str(args.swr_checkpoint),
                "models": {mode: str(item["checkpoint"]) for mode, item in models.items()},
                "skipped": skipped,
                "summary": str(summary_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("==== Stage 3 motion-branch ablation summary ====")
    show = summary[(summary["eval_split"] == "test") & summary["horizon"].isin(["all", "h1", "h4", "h8"])]
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    print("summary:", summary_path)
    print("figures:", out_dir)


if __name__ == "__main__":
    main()
