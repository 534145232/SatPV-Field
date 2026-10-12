#!/usr/bin/env python3
"""Collect the main Stage 2 SWR table from trained checkpoints.

The script evaluates three main-text candidates on the same center-region
split:

1. Center-only Ratio3DCNN
2. Big-patch Ratio3DCNN
3. Big-patch Ratio3DCNN + residual ConvLSTM correction

It prints a compact Markdown table and CSV rows for paper/table assembly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_3dcnn_ratio_center_only import fallback_checkpoint, load_model
from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    STAGE2_RUNS_DIR,
)
from train_residual_convlstm_correction_swr import (
    DEFAULT_RUN_ROOT as DEFAULT_RESIDUAL_RUN_ROOT,
    ResidualConvLSTMCorrection,
    build_residual_input,
)


DEFAULT_CENTER_CHECKPOINT = (
    STAGE2_RUNS_DIR
    / "h8_swr_3dcnn_ratio_stage1_center_only_4h3f"
    / "head_last_mean"
    / "best.pt"
)
DEFAULT_BIG_PATCH_CHECKPOINT = (
    STAGE2_RUNS_DIR
    / "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f"
    / "head_conv3d_decoder"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate main Stage 2 SWR models and print paper-table metrics."
    )
    parser.add_argument("--center_checkpoint", type=Path, default=DEFAULT_CENTER_CHECKPOINT)
    parser.add_argument("--big_patch_checkpoint", type=Path, default=DEFAULT_BIG_PATCH_CHECKPOINT)
    parser.add_argument("--residual_run_root", type=Path, default=DEFAULT_RESIDUAL_RUN_ROOT)
    parser.add_argument("--residual_variant", type=str, default="v0_full")
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pred_clip", type=float, default=1200.0)
    parser.add_argument("--ssim_data_range", type=float, default=1000.0)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=0, help="Debug limit; 0 means full split.")
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


def masked_global_ssim(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    data_range: float,
) -> float:
    """Compute a lightweight masked SSIM over one 2-D field.

    This uses the standard SSIM luminance/contrast/structure formula with
    statistics computed over valid pixels in the field. It is stable for the
    Stage 2 center-patch evaluation masks and avoids an extra dependency.
    """
    m = valid & np.isfinite(pred) & np.isfinite(truth)
    if int(m.sum()) < 2:
        return float("nan")
    x = np.asarray(pred[m], dtype=np.float64)
    y = np.asarray(truth[m], dtype=np.float64)
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(np.mean((x - mu_x) ** 2))
    var_y = float(np.mean((y - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator <= 0.0:
        return float("nan")
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denominator)


class SwrAccumulator:
    def __init__(self, future_frames: int, data_range: float):
        self.future_frames = int(future_frames)
        self.data_range = float(data_range)
        self.count = 0
        self.abs_err = 0.0
        self.sq_err = 0.0
        self.err = 0.0
        self.truth_sum = 0.0
        self.truth_sq_sum = 0.0
        self.ssim_sum = 0.0
        self.ssim_count = 0
        self.horizon = [
            {
                "count": 0,
                "abs_err": 0.0,
                "sq_err": 0.0,
                "err": 0.0,
                "truth_sum": 0.0,
                "truth_sq_sum": 0.0,
                "ssim_sum": 0.0,
                "ssim_count": 0,
            }
            for _ in range(self.future_frames)
        ]

    def update(self, pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> None:
        valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(truth)
        if torch.any(valid):
            error = pred - truth
            e = error[valid]
            y = truth[valid]
            self.count += int(e.numel())
            self.abs_err += float(e.abs().sum().item())
            self.sq_err += float(e.square().sum().item())
            self.err += float(e.sum().item())
            self.truth_sum += float(y.sum().item())
            self.truth_sq_sum += float(y.square().sum().item())

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        truth_np = truth.detach().cpu().numpy().astype(np.float32)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        for batch_index in range(pred_np.shape[0]):
            for horizon in range(self.future_frames):
                ssim = masked_global_ssim(
                    pred_np[batch_index, horizon],
                    truth_np[batch_index, horizon],
                    valid_np[batch_index, horizon],
                    self.data_range,
                )
                if np.isfinite(ssim):
                    self.ssim_sum += float(ssim)
                    self.ssim_count += 1
                    item = self.horizon[horizon]
                    item["ssim_sum"] += float(ssim)
                    item["ssim_count"] += 1

        for horizon in range(self.future_frames):
            horizon_valid = valid[:, horizon]
            if not torch.any(horizon_valid):
                continue
            error = pred[:, horizon] - truth[:, horizon]
            e = error[horizon_valid]
            y = truth[:, horizon][horizon_valid]
            item = self.horizon[horizon]
            item["count"] += int(e.numel())
            item["abs_err"] += float(e.abs().sum().item())
            item["sq_err"] += float(e.square().sum().item())
            item["err"] += float(e.sum().item())
            item["truth_sum"] += float(y.sum().item())
            item["truth_sq_sum"] += float(y.square().sum().item())

    @staticmethod
    def _finish_item(item: dict[str, float | int]) -> dict[str, float | int]:
        count = int(item["count"])
        denom = max(count, 1)
        sst = float(item["truth_sq_sum"]) - float(item["truth_sum"]) ** 2 / denom
        return {
            "count": count,
            "swr_mae": float(item["abs_err"]) / denom,
            "swr_rmse": math.sqrt(float(item["sq_err"]) / denom),
            "swr_bias": float(item["err"]) / denom,
            "swr_r2": 1.0 - float(item["sq_err"]) / sst if sst > 1.0e-12 else float("nan"),
            "swr_ssim": float(item["ssim_sum"]) / max(int(item["ssim_count"]), 1),
            "ssim_frames": int(item["ssim_count"]),
        }

    def summary(self) -> dict[str, float | int]:
        return self._finish_item(
            {
                "count": self.count,
                "abs_err": self.abs_err,
                "sq_err": self.sq_err,
                "err": self.err,
                "truth_sum": self.truth_sum,
                "truth_sq_sum": self.truth_sq_sum,
                "ssim_sum": self.ssim_sum,
                "ssim_count": self.ssim_count,
            }
        )

    def by_horizon(self) -> dict[str, dict[str, float | int]]:
        return {f"+{i + 1}": self._finish_item(item) for i, item in enumerate(self.horizon)}


def load_residual_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Path]:
    checkpoint_path = fallback_checkpoint(path)
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
    return model, checkpoint_args, checkpoint["residual_stats"], checkpoint_path


def format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("Model", "model"),
        ("Training", "training_setting"),
        ("Correction", "correction"),
        ("MAE down", "swr_mae"),
        ("RMSE down", "swr_rmse"),
        ("Bias", "swr_bias"),
        ("R2 up", "swr_r2"),
        ("SSIM up", "swr_ssim"),
    ]
    print("\n=== Main Stage 2 SWR Table (Markdown) ===")
    print("| " + " | ".join(name for name, _ in columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        cells = []
        for _name, key in columns:
            value = row.get(key, "")
            cells.append(format_float(value) if isinstance(value, (float, int, np.floating)) else str(value))
        print("| " + " | ".join(cells) + " |")


def print_csv(rows: list[dict[str, Any]]) -> None:
    print("\n=== Main Stage 2 SWR Table (CSV) ===")
    fields = [
        "model",
        "training_setting",
        "correction",
        "checkpoint",
        "count",
        "swr_mae",
        "swr_rmse",
        "swr_bias",
        "swr_r2",
        "swr_ssim",
        "ssim_frames",
    ]
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    center_model, center_stats, center_ckpt, center_head = load_model(args.center_checkpoint, device)
    big_model, big_stats, big_ckpt, big_head = load_model(args.big_patch_checkpoint, device)
    residual_ckpt = args.residual_run_root / str(args.residual_variant) / "best.pt"
    residual_model, residual_args, residual_stats, residual_ckpt = load_residual_model(residual_ckpt, device)

    dataset_args = argparse.Namespace(
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    center_dataset = H8RatioDataset(
        dataset_args,
        args.split,
        center_stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    big_dataset = H8RatioDataset(
        dataset_args,
        args.split,
        big_stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    if len(center_dataset) != len(big_dataset):
        raise RuntimeError(f"Dataset length mismatch: center={len(center_dataset)} big={len(big_dataset)}")
    if int(center_dataset.future_frames) != int(big_dataset.future_frames):
        raise RuntimeError(
            f"future_frames mismatch: center={center_dataset.future_frames} big={big_dataset.future_frames}"
        )
    center_loader = make_loader(center_dataset, args)
    big_loader = make_loader(big_dataset, args)
    future_frames = int(big_dataset.future_frames)
    accumulators = {
        "center_only": SwrAccumulator(future_frames, args.ssim_data_range),
        "big_patch": SwrAccumulator(future_frames, args.ssim_data_range),
        "big_patch_residual": SwrAccumulator(future_frames, args.ssim_data_range),
    }

    print("=== Stage 2 main SWR table evaluation ===", flush=True)
    print(f"split={args.split} device={device} samples={len(big_dataset)} future_frames={future_frames}", flush=True)
    print(f"center checkpoint: {center_ckpt} ({center_head})", flush=True)
    print(f"big-patch checkpoint: {big_ckpt} ({big_head})", flush=True)
    print(f"residual checkpoint: {residual_ckpt} ({args.residual_variant})", flush=True)

    total_batches = min(len(center_loader), len(big_loader))
    for batch_index, (center_batch, big_batch) in enumerate(zip(center_loader, big_loader), start=1):
        center_x = center_batch["x"].to(device, non_blocking=True)
        big_x = big_batch["x"].to(device, non_blocking=True)
        truth = big_batch["raw_y"].to(device, non_blocking=True).float()
        mask = big_batch["mask"].to(device, non_blocking=True).float()
        clear_y = big_batch["clear_y"].to(device, non_blocking=True).float()

        center_pred = torch.clamp(center_model(center_x).float() * clear_y, 0.0, float(args.pred_clip))
        big_pred = torch.clamp(big_model(big_x).float() * clear_y, 0.0, float(args.pred_clip))
        residual_input = build_residual_input(
            big_x,
            big_pred,
            float(residual_stats["swr_mean"]),
            float(residual_stats["swr_std"]),
        )
        residual = torch.clamp(
            residual_model(residual_input).float() * float(residual_args.get("residual_scale", 100.0)),
            -float(residual_args.get("residual_clip", 200.0)),
            float(residual_args.get("residual_clip", 200.0)),
        )
        residual_pred = torch.clamp(
            big_pred + residual,
            0.0,
            float(residual_args.get("pred_clip", args.pred_clip)),
        )

        accumulators["center_only"].update(center_pred, truth, mask)
        accumulators["big_patch"].update(big_pred, truth, mask)
        accumulators["big_patch_residual"].update(residual_pred, truth, mask)

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == total_batches:
            print(f"eval batch {batch_index}/{total_batches}", flush=True)
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            print(f"[debug] stopped at max_batches={args.max_batches}", flush=True)
            break

    row_specs = [
        (
            "Center-only 3D CNN",
            "Center region only",
            "-",
            str(center_ckpt),
            accumulators["center_only"],
        ),
        (
            "Big-patch 3D CNN",
            "5x5 overlapping patches",
            "-",
            str(big_ckpt),
            accumulators["big_patch"],
        ),
        (
            "Big-patch 3D CNN + ConvLSTM Res.",
            "5x5 overlapping patches",
            f"Residual ConvLSTM ({args.residual_variant})",
            str(residual_ckpt),
            accumulators["big_patch_residual"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, training, correction, checkpoint, accumulator in row_specs:
        row = {
            "model": model_name,
            "training_setting": training,
            "correction": correction,
            "checkpoint": checkpoint,
            **accumulator.summary(),
        }
        rows.append(row)

    print_markdown(rows)
    print_csv(rows)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
