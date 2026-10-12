#!/usr/bin/env python3
"""Evaluate sparse-density field prediction against native and full-density references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
for extra in (
    STAGE3_DIR,
    STAGE3_DIR / "ablation_motion_branch",
    STAGE3_DIR / "ablation_temporal_collapse",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import stage3_motion_fusion_model_impl as module
from last_only_collapse import patch_last_only_collapse
from motion_branch import patch_motion_branch
from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)
from visualize_regional_split_frames import _bounds

from density_common import (
    BEST_FULL_STAGE3_RUN_DIR,
    BEST_FULL_STAGE3_CHECKPOINT,
    BEST_FULL_STAGE3_MODEL_NAME,
    DEFAULT_SEED,
    density_frame_cache_dir,
    density_selection_dir,
    density_stage3_run_dir,
    density_target_run_dir,
    parse_ratios,
    ratio_label,
    write_json,
)


MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate source-fleet density field fidelity.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--splits", type=str, default="test,val")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--full_frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--full_selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--full_target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--full_stage3_run_dir", type=Path, default=BEST_FULL_STAGE3_RUN_DIR)
    parser.add_argument("--full_checkpoint", type=Path, default=BEST_FULL_STAGE3_CHECKPOINT)
    parser.add_argument("--multisource_cache_dir", type=Path, default=MULTISOURCE_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    import torch

    if str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={device}, but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return str(device)


def _field_args(args: argparse.Namespace) -> SimpleNamespace:
    if float(args.ratio) >= 1.0:
        selection_dir = Path(args.full_selection_dir)
        target_run_dir = Path(args.full_target_run_dir)
        frame_cache_dir = Path(args.full_frame_cache_dir)
        model_name = BEST_FULL_STAGE3_MODEL_NAME
    else:
        selection_dir = density_selection_dir(args.seed, args.ratio)
        target_run_dir = density_target_run_dir(args.seed, args.ratio)
        frame_cache_dir = density_frame_cache_dir(args.seed, args.ratio)
        model_name = MODEL_NAME
    return SimpleNamespace(
        selection_dir=selection_dir,
        target_run_dir=target_run_dir,
        frame_cache_dir=frame_cache_dir,
        multisource_cache_dir=Path(args.multisource_cache_dir),
        grid_size=64,
        gaussian_sigma_cells=1.0,
        gaussian_radius_cells=2,
        min_grid_weight=1.0e-6,
        freq_minutes=15,
        past_frames=12,
        future_frames=8,
        models="pv_nwp",
        model_name=model_name,
        base_model_name="pv_nwp",
        swr_anchor_cache_dir=Path(""),
        normalize_swr_anchor=1,
        swr_history_frames=4,
        swr_future_anchors=3,
        pv_hidden_channels=48,
        anchor_channels=48,
        horizon_embed_channels=16,
        delta_scale=1.0,
        batch_size=int(args.batch_size),
        num_workers=0,
    )


def _load_model(ckpt: dict, args: argparse.Namespace, device: str):
    import torch

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args"), dict) else {}
    model = module.make_swr_anchor_attention_model(
        future_frames=int(ckpt_args.get("future_frames", 8)),
        in_channels=int(ckpt["in_channels"]),
        future_channels=int(ckpt["future_channels"]),
        anchor_channels=int(ckpt.get("swr_anchor_channels", 2)),
        pv_hidden=int(ckpt_args.get("pv_hidden_channels", 48)),
        anchor_hidden=int(ckpt_args.get("anchor_channels", 48)),
        horizon_embed_channels=int(ckpt_args.get("horizon_embed_channels", 16)),
        dropout=float(ckpt_args.get("dropout", 0.05)),
        swr_history_frames=int(ckpt_args.get("swr_history_frames", 4)),
        swr_future_anchors=int(ckpt_args.get("swr_future_anchors", 3)),
        max_flow_pixels=float(ckpt_args.get("max_flow_pixels", 3.0)),
    )
    model = patch_motion_branch(model, "warp_direct_gate")
    model = patch_last_only_collapse(model)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _future_full_reference(full_cache, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    fields, masks = [], []
    for ts in module.f04._parse_time_list(row["future_times"]):
        field, mask = full_cache.get(pd.Timestamp(ts))
        fields.append(np.asarray(field, dtype=np.float32))
        masks.append(np.asarray(mask, dtype=np.float32))
    return np.stack(fields, axis=0), np.stack(masks, axis=0)


def _metric_accumulator() -> dict:
    return {"abs": 0.0, "sq": 0.0, "n": 0}


def _update(acc: dict, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    valid = np.asarray(mask) > 0.5
    if not np.any(valid):
        return
    diff = np.asarray(pred, dtype=np.float64)[valid] - np.asarray(target, dtype=np.float64)[valid]
    acc["abs"] += float(np.abs(diff).sum())
    acc["sq"] += float((diff * diff).sum())
    acc["n"] += int(diff.size)


def _finish(acc: dict) -> tuple[float, float, int]:
    n = max(int(acc["n"]), 1)
    return float(acc["abs"] / n), float(np.sqrt(acc["sq"] / n)), int(acc["n"])


def eval_split(args: argparse.Namespace, split: str, model, ckpt: dict, field_args: SimpleNamespace) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    time_df = pd.read_csv(Path(field_args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(module.f04._timestamp_key)
    swr_all, swr_meta, _ = module._load_prebuilt_swr_anchor_cache(Path(ckpt["args"]["swr_anchor_cache_dir"]))
    sub, cache_indices, _meta_df = module._subset_time_df_by_swr_meta(time_df, swr_meta, split)
    swr_arr = np.asarray(swr_all[cache_indices], dtype=np.float32)
    station_split = module.f04._load_station_split(field_args)
    bounds = _bounds(station_split)
    ms_cache = module.f04._load_multisource_cache(Path(args.multisource_cache_dir))
    sparse_cache = module.f04.ReadOnlyDiskFrameCache(Path(field_args.frame_cache_dir), time_df, field_args)
    full_cache = module.f04.ReadOnlyDiskFrameCache(Path(args.full_frame_cache_dir), time_df, field_args)
    pv_stats = ckpt["pv_field_norm_stats"]
    ms_stats = ckpt["multisource_norm_stats"]
    swr_stats = ckpt.get("swr_anchor_norm_stats")
    base = module.f04.TargetExcludedPVNormFieldDataset(sub, sparse_cache, ms_cache, bounds, pv_stats, ms_stats, "pv_nwp", field_args)
    dataset = module.SWRAnchorPVFieldDataset(base, swr_arr, field_args, swr_stats)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    native = [_metric_accumulator() for _ in range(8)]
    repr_loss = [_metric_accumulator() for _ in range(8)]
    total = [_metric_accumulator() for _ in range(8)]
    offset = 0
    with torch.no_grad():
        for x, future_feat, swr, y_z, m_sparse in loader:
            pred_z = model(x.to(args.device), future_feat.to(args.device), swr.to(args.device)).detach().cpu().numpy()
            pred = module.f04._denorm_field(pred_z, pv_stats)
            y_sparse = module.f04._denorm_field(y_z.numpy(), pv_stats)
            m_sparse_np = m_sparse.numpy().astype(np.float32)
            bsz = pred.shape[0]
            for bi in range(bsz):
                row = sub.iloc[offset + bi]
                full_y, full_m = _future_full_reference(full_cache, row)
                for h in range(pred.shape[1]):
                    _update(native[h], pred[bi, h], y_sparse[bi, h], m_sparse_np[bi, h])
                    _update(repr_loss[h], y_sparse[bi, h], full_y[h], full_m[h])
                    _update(total[h], pred[bi, h], full_y[h], full_m[h])
            offset += bsz

    rows = []
    groups = {"all": list(range(8)), "h1": [0], "h4": [3], "h8": [7]}
    for horizon, indices in groups.items():
        for metric_name, accs in (
            ("native", native),
            ("representation_vs_full", repr_loss),
            ("end_to_end_vs_full", total),
        ):
            merged = _metric_accumulator()
            for idx in indices:
                merged["abs"] += accs[idx]["abs"]
                merged["sq"] += accs[idx]["sq"]
                merged["n"] += accs[idx]["n"]
            mae, rmse, n = _finish(merged)
            rows.append(
                {
                    "eval_split": split,
                    "horizon": horizon,
                    "metric_scope": metric_name,
                    "mae_pv_norm_field": mae,
                    "rmse_pv_norm_field": rmse,
                    "pixel_count": n,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)
    if args.out_dir is None:
        args.out_dir = density_stage3_run_dir(args.seed, args.ratio) / "density_fidelity_eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if float(args.ratio) >= 1.0:
        ckpt_path = Path(args.full_checkpoint)
    else:
        ckpt_path = density_stage3_run_dir(args.seed, args.ratio) / f"{MODEL_NAME}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Density SAG-PVFF checkpoint not found: {ckpt_path}")
    import torch

    ckpt = torch.load(ckpt_path, map_location=args.device)
    field_args = _field_args(args)
    model = _load_model(ckpt, args, args.device)
    frames = []
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        frames.append(eval_split(args, split, model, ckpt, field_args))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.insert(0, "seed", int(args.seed))
    out.insert(1, "source_density_ratio", float(args.ratio))
    out.insert(2, "source_density_label", ratio_label(args.ratio))
    out.to_csv(args.out_dir / "density_field_fidelity_summary.csv", index=False)
    write_json(
        args.out_dir / "density_field_fidelity_meta.json",
        {
            "checkpoint": str(ckpt_path),
            "full_frame_cache_dir": str(args.full_frame_cache_dir),
            "selection_dir": str(field_args.selection_dir),
            "target_run_dir": str(field_args.target_run_dir),
            "frame_cache_dir": str(field_args.frame_cache_dir),
            "primary_full_reference_mask": "full-density support mask",
            "native_metric_mask": "current-density support mask",
        },
    )
    print(out.to_string(index=False))
    print("saved:", args.out_dir / "density_field_fidelity_summary.csv")


if __name__ == "__main__":
    main()
