#!/usr/bin/env python3
"""Evaluate all available residual ConvLSTM loss ablations on center-only test data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    write_json,
)
from train_residual_convlstm_correction_swr import (
    DEFAULT_BASE_CKPT,
    DEFAULT_RUN_ROOT,
    LOSS_VARIANTS,
    ResidualConvLSTMCorrection,
    build_residual_input,
    load_base_ratio_model,
    resolve_checkpoint,
    source_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare available residual ConvLSTM loss ablations on center-only evaluation data."
    )
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_vis", type=int, default=8)
    parser.add_argument("--vis_stride", type=int, default=20)
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    parser.add_argument(
        "--center_height_fraction",
        type=float,
        default=0.50,
        help="Fraction of image height included in the centered rectangular evaluation region.",
    )
    parser.add_argument(
        "--center_width_fraction",
        type=float,
        default=0.50,
        help="Fraction of image width included in the centered rectangular evaluation region.",
    )
    return parser.parse_args()


def make_loader(dataset: H8RatioDataset, args: argparse.Namespace) -> DataLoader:
    kwargs = {}
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


def load_residual_models(
    run_root: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    models = {}
    skipped = {}
    for variant in LOSS_VARIANTS:
        candidate = run_root / variant / "best.pt"
        try:
            checkpoint_path = resolve_checkpoint(candidate)
        except FileNotFoundError as exc:
            skipped[variant] = str(exc)
            print(f"[SKIP] {variant}: {exc}", flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        base_stats = checkpoint["base_stats"]
        model = ResidualConvLSTMCorrection(
            in_channels=int(checkpoint_args.get("input_channels", len(base_stats["channels"]) + 2)),
            future_frames=int(checkpoint_args.get("future_frames", 3)),
            width=int(checkpoint_args.get("width", 48)),
            depth=int(checkpoint_args.get("depth", 2)),
            dropout=float(checkpoint_args.get("dropout", 0.05)),
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        models[variant] = {
            "model": model,
            "checkpoint": checkpoint_path,
            "checkpoint_data": checkpoint,
            "args": checkpoint_args,
            "residual_stats": checkpoint["residual_stats"],
        }
        print(f"[LOAD] {variant}: {checkpoint_path}", flush=True)
    if not models:
        raise FileNotFoundError(f"No residual checkpoints found below {run_root}")
    return models, skipped


def init_accumulator(future_frames: int) -> dict[str, Any]:
    return {
        "count": 0,
        "truth_sum": 0.0,
        "truth_sq_sum": 0.0,
        "base_abs": 0.0,
        "base_sq": 0.0,
        "base_err": 0.0,
        "final_abs": 0.0,
        "final_sq": 0.0,
        "final_err": 0.0,
        "residual_abs": [],
        "horizons": {
            horizon: {"count": 0, "base_abs": 0.0, "final_abs": 0.0}
            for horizon in range(future_frames)
        },
    }


def update_accumulator(
    accumulator: dict[str, Any],
    base: torch.Tensor,
    residual: torch.Tensor,
    final: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = (
        (mask > 0.5)
        & torch.isfinite(base)
        & torch.isfinite(residual)
        & torch.isfinite(final)
        & torch.isfinite(truth)
    )
    if not torch.any(valid):
        return
    base_error = base - truth
    final_error = final - truth
    truth_values = truth[valid]
    base_values = base_error[valid]
    final_values = final_error[valid]
    accumulator["count"] += int(valid.sum().item())
    accumulator["truth_sum"] += float(truth_values.sum().item())
    accumulator["truth_sq_sum"] += float(truth_values.square().sum().item())
    accumulator["base_abs"] += float(base_values.abs().sum().item())
    accumulator["base_sq"] += float(base_values.square().sum().item())
    accumulator["base_err"] += float(base_values.sum().item())
    accumulator["final_abs"] += float(final_values.abs().sum().item())
    accumulator["final_sq"] += float(final_values.square().sum().item())
    accumulator["final_err"] += float(final_values.sum().item())
    residual_abs = residual[valid].abs().detach().cpu().numpy().astype(np.float32)
    if residual_abs.size:
        accumulator["residual_abs"].append(residual_abs)
    for horizon, item in accumulator["horizons"].items():
        horizon_valid = valid[:, horizon]
        if torch.any(horizon_valid):
            item["count"] += int(horizon_valid.sum().item())
            item["base_abs"] += float(base_error[:, horizon][horizon_valid].abs().sum().item())
            item["final_abs"] += float(final_error[:, horizon][horizon_valid].abs().sum().item())


def finish_accumulator(accumulator: dict[str, Any]) -> dict[str, float | int]:
    count = max(int(accumulator["count"]), 1)
    base_rmse = float(np.sqrt(accumulator["base_sq"] / count))
    final_rmse = float(np.sqrt(accumulator["final_sq"] / count))
    truth_mean = accumulator["truth_sum"] / count
    sst = accumulator["truth_sq_sum"] - count * truth_mean * truth_mean
    metrics: dict[str, float | int] = {
        "pixels": int(accumulator["count"]),
        "base_mae": accumulator["base_abs"] / count,
        "final_mae": accumulator["final_abs"] / count,
        "mae_improvement": (accumulator["base_abs"] - accumulator["final_abs"]) / count,
        "base_rmse": base_rmse,
        "final_rmse": final_rmse,
        "rmse_improvement": base_rmse - final_rmse,
        "base_bias": accumulator["base_err"] / count,
        "final_bias": accumulator["final_err"] / count,
        "base_r2": 1.0 - accumulator["base_sq"] / max(sst, 1e-6),
        "final_r2": 1.0 - accumulator["final_sq"] / max(sst, 1e-6),
    }
    for horizon, item in accumulator["horizons"].items():
        horizon_count = max(int(item["count"]), 1)
        metrics[f"base_mae_t{horizon + 1}"] = item["base_abs"] / horizon_count
        metrics[f"final_mae_t{horizon + 1}"] = item["final_abs"] / horizon_count
        metrics[f"mae_improvement_t{horizon + 1}"] = (
            item["base_abs"] - item["final_abs"]
        ) / horizon_count
    if accumulator["residual_abs"]:
        residual_abs = np.concatenate(accumulator["residual_abs"])
        metrics["residual_abs_mean"] = float(np.mean(residual_abs))
        metrics["residual_abs_p95"] = float(np.percentile(residual_abs, 95))
        metrics["residual_abs_max"] = float(np.max(residual_abs))
    return metrics


def prefix_metrics(metrics: dict[str, float | int], prefix: str) -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def center_region_mask(
    height: int,
    width: int,
    device: torch.device,
    height_fraction: float,
    width_fraction: float,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    for name, value in (
        ("center_height_fraction", height_fraction),
        ("center_width_fraction", width_fraction),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], found {value}")
    crop_height = max(1, min(height, int(round(height * float(height_fraction)))))
    crop_width = max(1, min(width, int(round(width * float(width_fraction)))))
    row_start = (height - crop_height) // 2
    col_start = (width - crop_width) // 2
    row_end = row_start + crop_height
    col_end = col_start + crop_width
    mask = torch.zeros((1, 1, height, width), device=device, dtype=torch.bool)
    mask[:, :, row_start:row_end, col_start:col_end] = True
    return mask, {
        "image_height": int(height),
        "image_width": int(width),
        "height_fraction": float(height_fraction),
        "width_fraction": float(width_fraction),
        "row_start_inclusive": int(row_start),
        "row_end_exclusive": int(row_end),
        "col_start_inclusive": int(col_start),
        "col_end_exclusive": int(col_end),
        "crop_height": int(crop_height),
        "crop_width": int(crop_width),
    }


def plot_sample(
    path: Path,
    truth: np.ndarray,
    base: np.ndarray,
    predictions: dict[str, np.ndarray],
    vmax: float,
    error_vmax: float,
    title: str,
    center_bounds: dict[str, int | float],
) -> None:
    variants = list(predictions)
    future_frames = int(truth.shape[0])
    columns = 2 + 2 * len(variants)
    fig, axes = plt.subplots(
        future_frames,
        columns,
        figsize=(3.2 * columns, 3.4 * future_frames),
        constrained_layout=True,
        squeeze=False,
    )
    for horizon in range(future_frames):
        panels = [
            ("truth", truth[horizon], 0.0, vmax, "turbo"),
            ("frozen base", base[horizon], 0.0, vmax, "turbo"),
        ]
        for variant in variants:
            prediction = predictions[variant][horizon]
            panels.extend(
                [
                    (variant, prediction, 0.0, vmax, "turbo"),
                    (f"{variant} error", prediction - truth[horizon], -error_vmax, error_vmax, "coolwarm"),
                ]
            )
        for ax, (name, values, vmin, vmax_i, cmap) in zip(axes[horizon], panels):
            image = ax.imshow(values, origin="upper", vmin=vmin, vmax=vmax_i, cmap=cmap)
            rectangle = plt.Rectangle(
                (
                    int(center_bounds["col_start_inclusive"]) - 0.5,
                    int(center_bounds["row_start_inclusive"]) - 0.5,
                ),
                int(center_bounds["crop_width"]),
                int(center_bounds["crop_height"]),
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                linestyle="--",
            )
            ax.add_patch(rectangle)
            ax.set_title(f"{name} +{horizon + 1}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    base_model: torch.nn.Module,
    models: dict[str, dict[str, Any]],
    dataset: H8RatioDataset,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, float | int]], dict[str, int | float]]:
    accumulators = {
        variant: init_accumulator(int(dataset.future_frames))
        for variant in models
    }
    center_accumulators = {
        variant: init_accumulator(int(dataset.future_frames))
        for variant in models
    }
    center_mask = None
    center_bounds = None
    vis_count = 0
    sample_offset = 0
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        truth = batch["raw_y"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        clear_y = batch["clear_y"].to(device, non_blocking=True).float()
        if center_mask is None:
            center_mask, center_bounds = center_region_mask(
                int(truth.shape[-2]),
                int(truth.shape[-1]),
                device,
                float(args.center_height_fraction),
                float(args.center_width_fraction),
            )
            print("Center evaluation region:", center_bounds, flush=True)
        base_ratio = base_model(x)
        first_args = next(iter(models.values()))["args"]
        base_phy = torch.clamp(base_ratio.float() * clear_y, 0.0, float(first_args.get("pred_clip", 1200.0)))
        final_predictions = {}
        for variant, item in models.items():
            residual_stats = item["residual_stats"]
            variant_args = item["args"]
            residual_input = build_residual_input(
                x,
                base_phy,
                float(residual_stats["swr_mean"]),
                float(residual_stats["swr_std"]),
            )
            residual_phy = torch.clamp(
                item["model"](residual_input).float() * float(variant_args.get("residual_scale", 100.0)),
                -float(variant_args.get("residual_clip", 200.0)),
                float(variant_args.get("residual_clip", 200.0)),
            )
            final_phy = torch.clamp(
                base_phy + residual_phy,
                0.0,
                float(variant_args.get("pred_clip", 1200.0)),
            )
            update_accumulator(
                accumulators[variant],
                base_phy,
                residual_phy,
                final_phy,
                truth,
                mask,
            )
            update_accumulator(
                center_accumulators[variant],
                base_phy,
                residual_phy,
                final_phy,
                truth,
                (mask > 0.5) & center_mask,
            )
            final_predictions[variant] = final_phy
        for local_index in range(x.shape[0]):
            global_index = sample_offset + local_index
            if vis_count >= int(args.num_vis) or global_index % max(1, int(args.vis_stride)) != 0:
                continue
            valid = mask[local_index].detach().cpu().numpy() > 0.5
            truth_np = np.where(valid, truth[local_index].detach().cpu().numpy(), np.nan)
            base_np = np.where(valid, base_phy[local_index].detach().cpu().numpy(), np.nan)
            prediction_np = {
                variant: np.where(valid, prediction[local_index].detach().cpu().numpy(), np.nan)
                for variant, prediction in final_predictions.items()
            }
            meta = dataset[global_index]["meta"]
            plot_sample(
                (args.out_dir / "samples" / f"{args.split}_sample_{global_index:05d}.png"),
                truth_np,
                base_np,
                prediction_np,
                float(args.vmax),
                float(args.error_vmax),
                f"issue {meta.get('issue_time_bjt', '')} | future {meta.get('future_times_bjt', '')}",
                center_bounds,
            )
            vis_count += 1
        sample_offset += int(x.shape[0])
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f"eval batch {batch_index}/{len(loader)}", flush=True)
    if center_bounds is None:
        raise RuntimeError("Evaluation loader did not yield any batches.")
    return (
        {
            variant: {
                **finish_accumulator(accumulator),
                **prefix_metrics(finish_accumulator(center_accumulators[variant]), "center_"),
            }
            for variant, accumulator in accumulators.items()
        },
        center_bounds,
    )


def print_summary(metrics: dict[str, dict[str, float | int]]) -> None:
    print("\n=== Residual ConvLSTM loss-ablation summary ===", flush=True)
    print(
        f"{'variant':24s} {'base_mae':>10s} {'final_mae':>10s} {'improve':>10s} "
        f"{'final_rmse':>11s} {'bias':>10s} {'res_abs':>10s} "
        f"{'ctr_final':>10s} {'ctr_gain':>10s}",
        flush=True,
    )
    for variant, values in metrics.items():
        print(
            f"{variant:24s} "
            f"{float(values['base_mae']):10.3f} "
            f"{float(values['final_mae']):10.3f} "
            f"{float(values['mae_improvement']):10.3f} "
            f"{float(values['final_rmse']):11.3f} "
            f"{float(values['final_bias']):10.3f} "
            f"{float(values.get('residual_abs_mean', float('nan'))):10.3f} "
            f"{float(values['center_final_mae']):10.3f} "
            f"{float(values['center_mae_improvement']):10.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir or (args.run_root / f"eval_all_{args.split}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models, skipped = load_residual_models(args.run_root, device)
    embedded_base_paths = {
        str(item["checkpoint_data"].get("base_checkpoint", ""))
        for item in models.values()
    }
    if len(embedded_base_paths) != 1:
        raise AssertionError(f"Residual checkpoints use different base checkpoints: {embedded_base_paths}")
    embedded_base_path = Path(next(iter(embedded_base_paths)))
    base_path = args.base_checkpoint if args.base_checkpoint != DEFAULT_BASE_CKPT else embedded_base_path
    base_model, base_stats, resolved_base, base_head = load_base_ratio_model(
        base_path,
        device,
        "conv3d_decoder",
    )
    print("Frozen base:", resolved_base, "| temporal_head:", base_head, flush=True)
    dataset_args = source_args(
        args.index_csv,
        args.frame_cache_dir,
        args.sample_cache_dir,
        args.clear_sky_dir,
    )
    dataset = H8RatioDataset(
        dataset_args,
        args.split,
        base_stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    loader = make_loader(dataset, args)
    try:
        metrics, center_bounds = evaluate(base_model, models, dataset, loader, args, device)
    finally:
        dataset.close()
    payload = {
        "split": args.split,
        "base_checkpoint": str(resolved_base),
        "base_temporal_head": base_head,
        "center_region": center_bounds,
        "models": {
            variant: {
                **values,
                "checkpoint": str(models[variant]["checkpoint"]),
            }
            for variant, values in metrics.items()
        },
        "skipped": skipped,
    }
    write_json(payload, args.out_dir / "metrics_all_residual_variants.json")
    rows = [
        {
            "loss_variant": variant,
            **values,
            "checkpoint": str(models[variant]["checkpoint"]),
        }
        for variant, values in metrics.items()
    ]
    with (args.out_dir / "metrics_all_residual_variants.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(metrics)
    print("saved:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
