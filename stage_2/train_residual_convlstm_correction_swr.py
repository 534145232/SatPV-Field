#!/usr/bin/env python3
"""Train residual ConvLSTM corrections on a frozen big-patch Ratio3DCNN base."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    Ratio3DCNN,
    STAGE1_BIG_FRAME_CACHE_DIR,
    STAGE1_BIG_INDEX_MAIN,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    STAGE2_RUNS_DIR,
    assert_center_only_eval_index,
    assert_train_mode_index,
    read_json,
    write_json,
)
from h8_3d_common import BasicConvLSTMForecaster
from train_3dcnn_swr import seed_everything, serializable_args, unwrap_model


LOSS_VARIANTS = {
    "v0_full": {
        "description": "Original residual loss: center weight + gradient loss + residual L1.",
        "use_center_weight": True,
        "grad_loss_weight": 0.1,
        "residual_l1_weight": 0.01,
    },
    "v1_no_gradient": {
        "description": "Ablation: remove gradient loss only.",
        "use_center_weight": True,
        "grad_loss_weight": 0.0,
        "residual_l1_weight": 0.01,
    },
    "v2_no_center_weight": {
        "description": "Ablation: remove center-aware spatial weighting only.",
        "use_center_weight": False,
        "grad_loss_weight": 0.1,
        "residual_l1_weight": 0.01,
    },
    "v3_no_residual_l1": {
        "description": "Ablation: remove residual L1 regularization only.",
        "use_center_weight": True,
        "grad_loss_weight": 0.1,
        "residual_l1_weight": 0.0,
    },
}

DEFAULT_BASE_CKPT = (
    STAGE2_RUNS_DIR
    / "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f"
    / "head_conv3d_decoder"
    / "best.pt"
)
DEFAULT_RUN_ROOT = (
    STAGE2_RUNS_DIR
    / "h8_swr_residual_convlstm_on_bigpatch_conv3d_decoder_4h3f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual ConvLSTM correction branch on a frozen Stage 2 "
            "big-patch Ratio3DCNN conv3d_decoder base."
        )
    )
    parser.add_argument("--loss_variant", choices=list(LOSS_VARIANTS), default="v0_full")
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--required_base_temporal_head", type=str, default="conv3d_decoder")
    parser.add_argument("--train_index_csv", type=Path, default=STAGE1_BIG_INDEX_MAIN)
    parser.add_argument("--train_frame_cache_dir", type=Path, default=STAGE1_BIG_FRAME_CACHE_DIR)
    parser.add_argument("--val_index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--val_frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--val_sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--val_clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--stats_json", type=Path, default=None)
    parser.add_argument("--force_recompute_stats", action="store_true")
    parser.add_argument("--stats_max_samples", type=int, default=768)
    parser.add_argument("--stats_pixels_per_sample", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual_scale", type=float, default=100.0)
    parser.add_argument("--residual_clip", type=float, default=200.0)
    parser.add_argument("--pred_clip", type=float, default=1200.0)
    parser.add_argument("--center_weight", type=float, default=2.5)
    parser.add_argument("--edge_weight", type=float, default=0.75)
    parser.add_argument("--center_sigma", type=float, default=0.45)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def configure_experiment(args: argparse.Namespace) -> argparse.Namespace:
    variant = LOSS_VARIANTS[str(args.loss_variant)]
    args.use_center_weight = bool(variant["use_center_weight"])
    args.grad_loss_weight = float(variant["grad_loss_weight"])
    args.residual_l1_weight = float(variant["residual_l1_weight"])
    args.loss_variant_description = str(variant["description"])
    args.out_dir = args.out_dir or (DEFAULT_RUN_ROOT / str(args.loss_variant))
    return args


def resolve_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "best.pt" and path.with_name("last.pt").exists():
        return path.with_name("last.pt")
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_base_ratio_model(
    path: Path,
    device: torch.device,
    required_temporal_head: str | None = "conv3d_decoder",
) -> tuple[nn.Module, dict[str, Any], Path, str]:
    path = resolve_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = checkpoint["stats"]
    checkpoint_args = checkpoint.get("args", {})
    temporal_head = str(
        checkpoint.get("temporal_head", checkpoint_args.get("temporal_head", "last_mean"))
    )
    if required_temporal_head and temporal_head != required_temporal_head:
        raise AssertionError(
            f"Base checkpoint temporal_head={temporal_head}, expected {required_temporal_head}: {path}"
        )
    model = Ratio3DCNN(
        in_channels=len(stats["channels"]),
        future_frames=int(checkpoint_args.get("future_frames", stats.get("future_frames", 3))),
        history_frames=int(checkpoint_args.get("history_frames", stats.get("history_frames", 4))),
        width=int(checkpoint_args.get("width", 48)),
        depth=int(checkpoint_args.get("depth", 4)),
        dropout=float(checkpoint_args.get("dropout", 0.05)),
        ratio_clip=float(stats["ratio_clip"]),
        temporal_head=temporal_head,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, stats, path, temporal_head


class ResidualConvLSTMCorrection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        future_frames: int,
        width: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = BasicConvLSTMForecaster(
            in_channels=in_channels,
            future_frames=future_frames,
            width=width,
            depth=depth,
            dropout=dropout,
        )
        last_layer = self.backbone.head[-1]
        if isinstance(last_layer, nn.Conv2d):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.backbone(sequence)


def source_args(
    index_csv: Path,
    frame_cache_dir: Path,
    sample_cache_dir: Path | None = None,
    clear_sky_dir: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index_csv=index_csv,
        frame_cache_dir=frame_cache_dir,
        sample_cache_dir=sample_cache_dir,
        clear_sky_dir=clear_sky_dir,
    )


def make_train_dataset(args: argparse.Namespace, stats: dict[str, Any]) -> H8RatioDataset:
    dataset_args = source_args(args.train_index_csv, args.train_frame_cache_dir)
    return H8RatioDataset(
        dataset_args,
        "train",
        stats,
        index_csv=args.train_index_csv,
        frame_cache_dir=args.train_frame_cache_dir,
        sample_cache_dir=None,
        clear_sky_dir=None,
    )


def make_val_dataset(args: argparse.Namespace, stats: dict[str, Any]) -> H8RatioDataset:
    dataset_args = source_args(
        args.val_index_csv,
        args.val_frame_cache_dir,
        args.val_sample_cache_dir,
        args.val_clear_sky_dir,
    )
    return H8RatioDataset(
        dataset_args,
        "val",
        stats,
        index_csv=args.val_index_csv,
        frame_cache_dir=args.val_frame_cache_dir,
        sample_cache_dir=args.val_sample_cache_dir,
        clear_sky_dir=args.val_clear_sky_dir,
    )


def make_loader(dataset: H8RatioDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    kwargs = {}
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=train,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


def default_stats_path(args: argparse.Namespace) -> Path:
    return DEFAULT_RUN_ROOT / "train_bigpatch_physical_swr_stats.json"


def estimate_physical_swr_stats(
    args: argparse.Namespace,
    base_stats: dict[str, Any],
) -> dict[str, Any]:
    dataset = make_train_dataset(args, base_stats)
    rng = np.random.default_rng(int(args.seed))
    indices = np.arange(len(dataset))
    if len(indices) > int(args.stats_max_samples):
        indices = rng.choice(indices, size=int(args.stats_max_samples), replace=False)
    values = []
    try:
        for index in indices:
            item = dataset[int(index)]
            raw_y = item["raw_y"].numpy()
            valid = item["mask"].numpy() > 0.5
            good = raw_y[valid]
            good = good[np.isfinite(good)]
            if good.size == 0:
                continue
            take = min(int(args.stats_pixels_per_sample), good.size)
            values.append(rng.choice(good.reshape(-1), size=take, replace=False).astype(np.float32))
    finally:
        dataset.close()
    all_values = np.concatenate(values) if values else np.asarray([350.0], dtype=np.float32)
    return {
        "swr_mean": float(np.mean(all_values)),
        "swr_std": float(max(np.std(all_values), 1.0)),
        "num_values": int(all_values.size),
        "source": str(args.train_index_csv),
        "note": "Physical SWR mean/std estimated from Stage 1 big-patch train split only.",
    }


def center_weight_map(
    height: int,
    width: int,
    device: torch.device,
    center_weight: float,
    edge_weight: float,
    sigma: float,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return torch.ones((1, 1, height, width), device=device)
    yy = torch.linspace(-1.0, 1.0, height, device=device)
    xx = torch.linspace(-1.0, 1.0, width, device=device)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    gaussian = torch.exp(-(x.square() + y.square()) / (2.0 * float(sigma) ** 2))
    weights = float(edge_weight) + (float(center_weight) - float(edge_weight)) * gaussian
    return weights.view(1, 1, height, width)


def build_residual_input(
    x_norm: torch.Tensor,
    base_pred_phy: torch.Tensor,
    swr_mean: float,
    swr_std: float,
) -> torch.Tensor:
    batch, history, channels, height, width = x_norm.shape
    future = base_pred_phy.shape[1]
    future_features = x_norm.new_zeros((batch, future, channels, height, width))
    base_norm = (base_pred_phy - float(swr_mean)) / max(float(swr_std), 1.0)
    base_history = x_norm.new_zeros((batch, history, 1, height, width))
    base_future = base_norm.unsqueeze(2)
    flag_history = x_norm.new_zeros((batch, history, 1, height, width))
    flag_future = x_norm.new_ones((batch, future, 1, height, width))
    history_sequence = torch.cat([x_norm, base_history, flag_history], dim=2)
    future_sequence = torch.cat([future_features, base_future, flag_future], dim=2)
    return torch.cat([history_sequence, future_sequence], dim=1)


def weighted_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5) & torch.isfinite(prediction) & torch.isfinite(target)
    diff = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    weights = weight * valid.float()
    return (diff.abs() * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5) & torch.isfinite(prediction) & torch.isfinite(target)
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    dx_valid = valid[..., 1:] & valid[..., :-1]
    dy_valid = valid[..., 1:, :] & valid[..., :-1, :]
    dx_error = (
        safe_prediction[..., 1:] - safe_prediction[..., :-1]
        - safe_target[..., 1:] + safe_target[..., :-1]
    )
    dy_error = (
        safe_prediction[..., 1:, :] - safe_prediction[..., :-1, :]
        - safe_target[..., 1:, :] + safe_target[..., :-1, :]
    )
    dx_weight = 0.5 * (weight[..., 1:] + weight[..., :-1]) * dx_valid.float()
    dy_weight = 0.5 * (weight[..., 1:, :] + weight[..., :-1, :]) * dy_valid.float()
    dx = (dx_error.abs() * dx_weight).sum() / dx_weight.sum().clamp_min(1.0)
    dy = (dy_error.abs() * dy_weight).sum() / dy_weight.sum().clamp_min(1.0)
    return 0.5 * (dx + dy)


def residual_loss(
    final_prediction: torch.Tensor,
    truth: torch.Tensor,
    residual_prediction: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: torch.Tensor,
    grad_loss_weight: float,
    residual_l1_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mae = weighted_mae(final_prediction, truth, mask, spatial_weight)
    gradient = (
        weighted_gradient_loss(final_prediction, truth, mask, spatial_weight)
        if float(grad_loss_weight) != 0.0
        else final_prediction.new_zeros(())
    )
    valid = (mask > 0.5) & torch.isfinite(residual_prediction)
    residual_l1 = (
        torch.where(valid, residual_prediction.abs(), torch.zeros_like(residual_prediction)).sum()
        / valid.float().sum().clamp_min(1.0)
    )
    loss = mae + float(grad_loss_weight) * gradient + float(residual_l1_weight) * residual_l1
    return loss, {
        "weighted_mae": float(mae.detach().item()),
        "gradient_loss": float(gradient.detach().item()),
        "residual_l1": float(residual_l1.detach().item()),
    }


@torch.no_grad()
def batch_metrics(
    base: torch.Tensor,
    final: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    valid = (mask > 0.5) & torch.isfinite(base) & torch.isfinite(final) & torch.isfinite(truth)
    if not torch.any(valid):
        return {
            "base_mae": float("nan"),
            "final_mae": float("nan"),
            "base_rmse": float("nan"),
            "final_rmse": float("nan"),
            "final_bias": float("nan"),
        }
    base_error = base[valid] - truth[valid]
    final_error = final[valid] - truth[valid]
    return {
        "base_mae": float(base_error.abs().mean().item()),
        "final_mae": float(final_error.abs().mean().item()),
        "base_rmse": float(torch.sqrt(torch.mean(base_error.square())).item()),
        "final_rmse": float(torch.sqrt(torch.mean(final_error.square())).item()),
        "final_bias": float(final_error.mean().item()),
    }


def run_epoch(
    residual_model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    residual_stats: dict[str, Any],
    use_amp: bool,
    epoch_name: str,
) -> dict[str, float]:
    train = optimizer is not None
    residual_model.train(train)
    totals = {
        "loss": [],
        "weighted_mae": [],
        "gradient_loss": [],
        "residual_l1": [],
        "base_mae": [],
        "final_mae": [],
        "base_rmse": [],
        "final_rmse": [],
        "final_bias": [],
    }
    spatial_weight = None
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        truth = batch["raw_y"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        clear_y = batch["clear_y"].to(device, non_blocking=True).float()
        if spatial_weight is None or spatial_weight.shape[-2:] != truth.shape[-2:]:
            spatial_weight = center_weight_map(
                truth.shape[-2],
                truth.shape[-1],
                device,
                float(args.center_weight),
                float(args.edge_weight),
                float(args.center_sigma),
                bool(args.use_center_weight),
            )
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                base_ratio = base_model(x)
            base_phy = torch.clamp(base_ratio.float() * clear_y, 0.0, float(args.pred_clip))
        residual_input = build_residual_input(
            x,
            base_phy,
            float(residual_stats["swr_mean"]),
            float(residual_stats["swr_std"]),
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                residual_norm = residual_model(residual_input)
                residual_phy = torch.clamp(
                    residual_norm.float() * float(args.residual_scale),
                    -float(args.residual_clip),
                    float(args.residual_clip),
                )
                final_phy = torch.clamp(base_phy + residual_phy, 0.0, float(args.pred_clip))
                loss, parts = residual_loss(
                    final_phy,
                    truth,
                    residual_phy,
                    mask,
                    spatial_weight,
                    float(args.grad_loss_weight),
                    float(args.residual_l1_weight),
                )
            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), float(args.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), float(args.grad_clip))
                    optimizer.step()
        metrics = batch_metrics(base_phy.detach(), final_phy.detach(), truth, mask)
        totals["loss"].append(float(loss.detach().item()))
        for key, value in parts.items():
            totals[key].append(value)
        for key, value in metrics.items():
            if math.isfinite(value):
                totals[key].append(value)
        if (
            int(args.log_interval) > 0
            and (
                batch_index == 1
                or batch_index % int(args.log_interval) == 0
                or batch_index == len(loader)
            )
        ):
            print(
                f"{epoch_name} batch {batch_index}/{len(loader)} | "
                f"loss {float(np.mean(totals['loss'])):.3f} | "
                f"base_mae {float(np.mean(totals['base_mae'])):.2f} "
                f"final_mae {float(np.mean(totals['final_mae'])):.2f} "
                f"res_l1 {float(np.mean(totals['residual_l1'])):.2f}",
                flush=True,
            )
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in totals.items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    base_stats: dict[str, Any],
    residual_stats: dict[str, Any],
    base_checkpoint: Path,
    base_temporal_head: str,
    epoch: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "base_stats": base_stats,
            "residual_stats": residual_stats,
            "base_checkpoint": str(base_checkpoint),
            "base_temporal_head": base_temporal_head,
            "loss_variant": str(args.loss_variant),
            "epoch": int(epoch),
            "best_val_final_mae": float(best_val),
            "model_name": "stage2_residual_convlstm_correction",
        },
        path,
    )


def main() -> None:
    args = configure_experiment(parse_args())
    seed_everything(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert_train_mode_index(args.train_index_csv, "big_patch_5x5")
    assert_center_only_eval_index(args.val_index_csv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, base_stats, base_checkpoint, base_temporal_head = load_base_ratio_model(
        args.base_checkpoint,
        device,
        args.required_base_temporal_head,
    )
    stats_path = args.stats_json or default_stats_path(args)
    if stats_path.exists() and not bool(args.force_recompute_stats):
        residual_stats = read_json(stats_path)
        print("Loading residual physical stats:", stats_path)
    else:
        print("Estimating residual physical stats ->", stats_path)
        residual_stats = estimate_physical_swr_stats(args, base_stats)
        write_json(residual_stats, stats_path)

    train_dataset = make_train_dataset(args, base_stats)
    val_dataset = make_val_dataset(args, base_stats)
    future_frames = int(train_dataset.future_frames)
    args.future_frames = future_frames
    args.history_frames = int(train_dataset.history_frames)
    args.input_channels = len(base_stats["channels"]) + 2
    print(
        f"Data: train={len(train_dataset)} val={len(val_dataset)} | "
        f"forecast={args.history_frames}h->{future_frames}f | "
        "train_source=stage1_big_frame_cache_dynamic_crop | val_source=stage1_center_sample_cache"
    )
    print("Frozen base:", base_checkpoint)
    print("Base temporal head:", base_temporal_head)
    print("Loss variant:", args.loss_variant, "|", args.loss_variant_description)
    print(
        "Effective loss: "
        f"center_weight={args.use_center_weight} "
        f"gradient_weight={args.grad_loss_weight} "
        f"residual_l1_weight={args.residual_l1_weight}"
    )
    print("Physical SWR context stats:", residual_stats)

    train_loader = make_loader(train_dataset, args, True)
    val_loader = make_loader(val_dataset, args, False)
    residual_model = ResidualConvLSTMCorrection(
        in_channels=int(args.input_channels),
        future_frames=future_frames,
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    if bool(args.compile) and hasattr(torch, "compile"):
        residual_model = torch.compile(residual_model)
    optimizer = torch.optim.AdamW(
        residual_model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    use_amp = bool(args.amp) and device.type == "cuda"
    write_json(
        {
            "args": serializable_args(args),
            "base_checkpoint": str(base_checkpoint),
            "base_temporal_head": base_temporal_head,
            "base_stats": base_stats,
            "residual_stats": residual_stats,
        },
        args.out_dir / "config.json",
    )

    fields = [
        "epoch",
        "train_loss",
        "train_base_mae",
        "train_final_mae",
        "train_gradient_loss",
        "train_residual_l1",
        "val_loss",
        "val_base_mae",
        "val_final_mae",
        "val_base_rmse",
        "val_final_rmse",
        "val_final_bias",
        "val_gradient_loss",
        "val_residual_l1",
    ]
    history_path = args.out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_val = float("inf")
    bad_epochs = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = run_epoch(
                residual_model,
                base_model,
                train_loader,
                optimizer,
                scaler,
                device,
                args,
                residual_stats,
                use_amp,
                f"epoch {epoch:03d} train",
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    residual_model,
                    base_model,
                    val_loader,
                    None,
                    None,
                    device,
                    args,
                    residual_stats,
                    False,
                    f"epoch {epoch:03d} val",
                )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_base_mae": train_metrics["base_mae"],
                "train_final_mae": train_metrics["final_mae"],
                "train_gradient_loss": train_metrics["gradient_loss"],
                "train_residual_l1": train_metrics["residual_l1"],
                "val_loss": val_metrics["loss"],
                "val_base_mae": val_metrics["base_mae"],
                "val_final_mae": val_metrics["final_mae"],
                "val_base_rmse": val_metrics["base_rmse"],
                "val_final_rmse": val_metrics["final_rmse"],
                "val_final_bias": val_metrics["final_bias"],
                "val_gradient_loss": val_metrics["gradient_loss"],
                "val_residual_l1": val_metrics["residual_l1"],
            }
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)
            print(
                f"res-convlstm {args.loss_variant} epoch {epoch:03d} | "
                f"train final_mae {train_metrics['final_mae']:.2f} base {train_metrics['base_mae']:.2f} | "
                f"val final_mae {val_metrics['final_mae']:.2f} base {val_metrics['base_mae']:.2f} "
                f"rmse {val_metrics['final_rmse']:.2f} res_l1 {val_metrics['residual_l1']:.2f}",
                flush=True,
            )
            save_checkpoint(
                args.out_dir / "last.pt",
                residual_model,
                optimizer,
                args,
                base_stats,
                residual_stats,
                base_checkpoint,
                base_temporal_head,
                epoch,
                best_val,
            )
            if val_metrics["final_mae"] < best_val:
                best_val = val_metrics["final_mae"]
                bad_epochs = 0
                save_checkpoint(
                    args.out_dir / "best.pt",
                    residual_model,
                    optimizer,
                    args,
                    base_stats,
                    residual_stats,
                    base_checkpoint,
                    base_temporal_head,
                    epoch,
                    best_val,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.patience):
                    print(f"Early stopping at epoch {epoch}, best val final MAE={best_val:.2f}")
                    break
        write_json(
            {
                "best_val_final_mae": best_val,
                "loss_variant": str(args.loss_variant),
                "base_checkpoint": str(base_checkpoint),
                "base_temporal_head": base_temporal_head,
            },
            args.out_dir / "metrics.json",
        )
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    main()
