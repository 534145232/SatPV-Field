#!/usr/bin/env python3
"""Re-evaluate trained Stage 3 base PV_norm field models.

Use this after metric definitions change. It reloads existing *_best.pt
checkpoints and rewrites pvnorm_field_eval_summary.csv without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import stage3_base_pvnorm_field_model as base
from stage3_motion_fusion_model_impl import _load_prebuilt_swr_anchor_cache, _subset_time_df_by_swr_meta
from stage3_paths import RUNS_DIR
from stage3_swr_anchor_cache import resolve_active_cache_dir


def parse_args() -> argparse.Namespace:
    defaults = dict(base.PYCHARM_DEFAULTS)
    parser = argparse.ArgumentParser(description="Evaluate trained Stage 3 base PV_norm field checkpoints.")
    parser.add_argument("--out_dir", type=Path, default=RUNS_DIR / "base_pvnorm_field_model")
    parser.add_argument("--selection_dir", type=Path, default=Path(defaults["selection_dir"]))
    parser.add_argument("--target_run_dir", type=Path, default=Path(defaults["target_run_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(defaults["frame_cache_dir"]))
    parser.add_argument("--multisource_cache_dir", type=Path, default=Path(defaults["multisource_cache_dir"]))
    parser.add_argument("--models", type=str, default="persistence_last,pv_only,pv_h8,pv_nwp,pv_h8_nwp")
    parser.add_argument("--grid_size", type=int, default=int(defaults["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(defaults["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(defaults["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(defaults["min_grid_weight"]))
    parser.add_argument("--freq_minutes", type=int, default=int(defaults["freq_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(defaults["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(defaults["future_frames"]))
    parser.add_argument("--delta_scale", type=float, default=float(defaults["delta_scale"]))
    parser.add_argument("--batch_size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument("--num_workers", type=int, default=int(defaults["num_workers"]))
    parser.add_argument("--large_delta_threshold", type=float, default=float(defaults["large_delta_threshold"]))
    parser.add_argument("--device", type=str, default=str(defaults["device"]))
    parser.add_argument(
        "--common_swr_anchor_subset",
        type=int,
        default=1,
        help="When 1, evaluate baselines only on issue times retained by the active SWR-anchor cache.",
    )
    parser.add_argument(
        "--swr_anchor_cache_dir",
        type=Path,
        default=None,
        help="Optional SWR-anchor cache directory. Defaults to active_cache.json when common_swr_anchor_subset=1.",
    )
    return parser.parse_args()


def evaluate_persistence_last(loader, model_name: str, split: str, pv_stats: dict, threshold: float) -> pd.DataFrame:
    rows = []
    for x, _future_feat, y, m in loader:
        x_np = x.numpy()
        y_np = y.numpy()
        m_np = m.numpy()
        last = x_np[:, 0, -1]
        pred = last[:, None].repeat(y_np.shape[1], axis=1)
        rows.extend(base._metric_rows(model_name, split, pred, y_np, m_np, x_np, pv_stats, threshold))
    return base._combine_metrics(rows)


def main() -> None:
    args = parse_args()
    out_dir = base.ensure_dir(args.out_dir)

    import torch
    from torch.utils.data import DataLoader

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    requested = [x.strip() for x in str(args.models).split(",") if x.strip()]
    eval_only_models = {"persistence_last"}
    bad = [x for x in requested if x not in base.MODEL_SPECS and x not in eval_only_models]
    if bad:
        raise ValueError(f"unsupported models: {bad}; valid={sorted((*base.MODEL_SPECS, *eval_only_models))}")

    split_df = base._load_station_split(args)
    bounds = base._bounds(split_df)
    ms_cache = base._load_multisource_cache(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"].copy()
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    base._assert_issue_time_alignment(args, time_df)
    base._assert_target_exclusion(args)

    swr_meta = None
    if int(args.common_swr_anchor_subset):
        swr_cache_dir = Path(args.swr_anchor_cache_dir) if args.swr_anchor_cache_dir is not None else resolve_active_cache_dir(required=True)
        _swr_arr, swr_meta, _swr_info = _load_prebuilt_swr_anchor_cache(swr_cache_dir)
        print(f"[common_subset] using SWR-anchor valid issue subset from: {swr_cache_dir}", flush=True)

    pv_cache = base.ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    pv_stats = base._compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = base._compute_or_load_multisource_stats(ms_cache, out_dir, time_df)

    all_eval = []
    for model_name in requested:
        dataset_model_name = "pv_only" if model_name == "persistence_last" else model_name
        datasets, loaders = {}, {}
        for split in ["val", "test"]:
            tdf = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
            if swr_meta is not None:
                original_len = len(tdf)
                tdf, _cache_indices, _sub_meta = _subset_time_df_by_swr_meta(tdf, swr_meta, split)
                print(
                    f"[common_subset][{model_name}][{split}] issue rows: {original_len} -> {len(tdf)}",
                    flush=True,
                )
            ds = base.TargetExcludedPVNormFieldDataset(tdf, pv_cache, ms_cache, bounds, pv_stats, ms_stats, dataset_model_name, args)
            datasets[split] = ds
            loaders[split] = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
            )
        if model_name == "persistence_last":
            print("[eval] persistence_last: repeat latest historical PV field", flush=True)
            for split in ["val", "test"]:
                eval_df = evaluate_persistence_last(
                    loaders[split],
                    model_name,
                    split,
                    pv_stats,
                    threshold=float(args.large_delta_threshold),
                )
                all_eval.append(eval_df)
                show = eval_df[eval_df["horizon"].isin(["all", "h6", "h7", "h8"])]
                print(show.to_string(index=False) if not show.empty else f"[warn] empty eval for {model_name}/{split}", flush=True)
            continue
        in_channels, future_channels = datasets["val"].source_dims()
        ckpt_path = out_dir / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            print(f"[skip] missing checkpoint for {model_name}: {ckpt_path}", flush=True)
            continue
        model = base.make_residual_3d_unet_z(
            int(args.future_frames),
            in_channels=int(in_channels),
            future_channels=int(future_channels),
            delta_scale=float(args.delta_scale),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[eval] {model_name}: {ckpt_path}", flush=True)
        for split in ["val", "test"]:
            eval_df = base.evaluate_model(
                model,
                loaders[split],
                device,
                model_name,
                split,
                pv_stats,
                threshold=float(args.large_delta_threshold),
            )
            all_eval.append(eval_df)
            show = eval_df[eval_df["horizon"].isin(["all", "h6", "h7", "h8"])]
            print(show.to_string(index=False) if not show.empty else f"[warn] empty eval for {model_name}/{split}", flush=True)

    summary = (
        pd.concat(all_eval, ignore_index=True)
        .sort_values(["eval_split", "horizon", "mae_pv_norm_field", "model"])
        .reset_index(drop=True)
        if all_eval
        else pd.DataFrame()
    )
    summary.to_csv(out_dir / "pvnorm_field_eval_summary.csv", index=False)
    print(f"saved: {out_dir / 'pvnorm_field_eval_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
