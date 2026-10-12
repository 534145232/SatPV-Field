#!/usr/bin/env python3
"""Evaluate the last-only full model with its adaptive gate replaced by fixed 0.5 fusion.

This is an eval-only ablation: it reuses the trained warp_direct_gate checkpoint
and disables the adaptive fusion gate at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
MOTION_DIR = STAGE3_DIR / "ablation_motion_branch"
SWR_DEPENDENCY_DIR = STAGE3_DIR / "ablation_swr_dependency"
TEMPORAL_COLLAPSE_DIR = STAGE3_DIR / "ablation_temporal_collapse"
for path in [STAGE3_DIR, MOTION_DIR, SWR_DEPENDENCY_DIR, TEMPORAL_COLLAPSE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_all_motion_branches as motion_eval
import stage3_motion_fusion_model_impl as module
from stage3_paths import RUNS_DIR, SWR_BASE_CHECKPOINT
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir
from swr_dependency import install_swr_dependency_dataset


RUN_ROOT = RUNS_DIR / "ablation_motion_branch_last_only"
OUT_DIR = RUN_ROOT / "eval_fixed_fusion_from_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed 0.5 warp/direct fusion from the full gated checkpoint.")
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--swr_checkpoint", type=Path, default=SWR_BASE_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--visual_horizons", type=str, default="1,4,8")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def replace_gate_with_fixed_half(model):
    import torch.nn as nn

    class FixedHalfGate(nn.Module):
        def forward(self, x):
            return x.new_full((x.shape[0], 1, x.shape[-2], x.shape[-1]), 0.5)

    model.fusion_gate = FixedHalfGate()
    model.motion_branch_mode = "fixed_fusion_from_gate"
    return model


def main() -> None:
    args = parse_args()
    anchor_cache_dir = args.swr_anchor_cache_dir or resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(anchor_cache_dir, args.swr_checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_args = argparse.Namespace(
        run_root=Path(args.run_root),
        temporal_collapse="last_only",
        swr_dependency="full_swr_future",
        motion_branch="warp_direct_gate",
        swr_anchor_cache_dir=anchor_cache_dir,
        swr_checkpoint=Path(args.swr_checkpoint),
        out_dir=out_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        sample_count=int(args.sample_count),
        visual_horizons=str(args.visual_horizons),
        dpi=int(args.dpi),
        seed=int(args.seed),
    )
    install_swr_dependency_dataset(module, "full_swr_future")
    dataset_args, datasets, loaders, pv_stats, model_kwargs = motion_eval.build_data(eval_args, anchor_cache_dir, out_dir)
    device = module._device(args.device)
    models, skipped = motion_eval.load_models(
        Path(args.run_root),
        model_kwargs,
        device,
        "last_only",
        "warp_direct_gate",
    )
    item = models["warp_direct_gate"]
    model = replace_gate_with_fixed_half(item["model"]).eval()

    frames = []
    model_name = "swr_anchor_motion_fusion_pv_nwp_motion_fixed_fusion_from_gate"
    for split in ["val", "test"]:
        frame = module.evaluate_model(
            model,
            loaders[split],
            device,
            model_name,
            split,
            pv_stats,
            float(dataset_args.large_delta_threshold),
        )
        frame.insert(0, "motion_branch", "fixed_fusion_from_gate")
        frame.insert(1, "swr_dependency", "full_swr_future")
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    summary_path = out_dir / "motion_branch_eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    with (out_dir / "fixed_fusion_eval_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "purpose": "Eval-only w/o adaptive gate ablation: replace learned fusion gate with fixed 0.5.",
                "source_checkpoint": str(item["checkpoint"]),
                "temporal_collapse": "last_only",
                "swr_dependency": "full_swr_future",
                "swr_anchor_cache_dir": str(anchor_cache_dir),
                "swr_checkpoint": str(args.swr_checkpoint),
                "summary": str(summary_path),
                "skipped": skipped,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    show = summary[(summary["eval_split"] == "test") & summary["horizon"].astype(str).isin(["all", "h1", "h4", "h8"])]
    print("==== Stage 3 fixed-fusion-from-gate ablation ====")
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()

