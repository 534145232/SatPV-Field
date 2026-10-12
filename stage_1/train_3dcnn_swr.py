#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from h8_3d_common import (
    DEFAULT_INDEX_CSV,
    DEFAULT_FRAME_CACHE_DIR,
    DEFAULT_SAMPLE_CACHE_DIR,
    BasicSWR3DBackbone,
    H8SWRForecastDataset,
    estimate_normalization,
    masked_metrics,
    masked_smooth_l1,
    parse_channels,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight 3D CNN baseline for H8 SWR 4-history to 3-future forecasting."
    )
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--out_dir", type=Path, default=Path("/code/Final_3_stage_pipeline/stage_2/runs/h8_swr_3dcnn_center_4h3f"))
    parser.add_argument("--frame_cache_dir", type=Path, default=DEFAULT_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=DEFAULT_SAMPLE_CACHE_DIR)
    parser.add_argument("--channels", type=str, default=",".join(parse_channels(None)))
    parser.add_argument("--stats_json", type=Path, default=None)
    parser.add_argument("--stats_max_samples", type=int, default=2000)
    parser.add_argument("--force_recompute_stats", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def serializable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def default_stats_path(index_csv: Path, channels: list[str]) -> Path:
    signature = hashlib.sha1(",".join(channels).encode("utf-8")).hexdigest()[:10]
    return index_csv.parent / f"{index_csv.stem}_h8_swr_3dcnn_normalization_{signature}.json"


def make_loader(ds: H8SWRForecastDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    num_workers = int(args.num_workers)
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    y_mean: float,
    y_std: float,
    grad_clip: float,
    use_amp: bool,
    epoch_name: str = "",
    log_interval: int = 0,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_items = 0
    maes = []
    rmses = []
    biases = []

    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = masked_smooth_l1(pred, y, mask)

            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        bs = int(x.shape[0])
        total_loss += float(loss.item()) * bs
        total_items += bs
        mt = masked_metrics(pred.detach(), y, mask, y_mean, y_std)
        if math.isfinite(mt["mae"]):
            maes.append(mt["mae"])
            rmses.append(mt["rmse"])
            biases.append(mt["bias"])
        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            phase = epoch_name or ("train" if train else "val")
            print(
                f"{phase} batch {batch_idx}/{total_batches} | "
                f"loss {total_loss / max(total_items, 1):.4f} | "
                f"mae {float(np.mean(maes)) if maes else float('nan'):.2f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(total_items, 1),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "bias": float(np.mean(biases)) if biases else float("nan"),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stats: dict,
    channels: list[str],
    epoch: int,
    best_val_mae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "stats": stats,
            "channels": channels,
            "epoch": epoch,
            "best_val_mae": best_val_mae,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    channels = parse_channels(args.channels)
    stats_path = args.stats_json or default_stats_path(args.index_csv, channels)
    if stats_path.exists() and not args.force_recompute_stats:
        print(f"Loading normalization: {stats_path}")
        stats = read_json(stats_path)
    else:
        print(f"Estimating normalization from train split -> {stats_path}")
        stats = estimate_normalization(
            args.index_csv,
            channels,
            max_samples=int(args.stats_max_samples),
            seed=int(args.seed),
        )
        write_json(stats, stats_path)
    if list(stats.get("channels", [])) != channels:
        raise ValueError(
            "Normalization channels do not match requested channels: "
            f"stats={stats.get('channels')} requested={channels}"
        )

    write_json(
        {
            "index_csv": str(args.index_csv),
            "channels": channels,
            "stats_json": str(stats_path),
            "args": serializable_args(args),
        },
        args.out_dir / "config.json",
    )
    if stats_path != args.out_dir / "normalization.json":
        write_json(stats, args.out_dir / "normalization.json")

    frame_cache_dir = args.frame_cache_dir
    sample_cache_dir = args.sample_cache_dir
    if sample_cache_dir is not None:
        print(f"Using sample cache: {sample_cache_dir}")
        if not (sample_cache_dir / "sample_cache_meta.json").exists():
            print(f"[WARN] sample cache metadata not found, fallback may use frame cache: {sample_cache_dir}")
    if frame_cache_dir is not None:
        print(f"Using frame cache: {frame_cache_dir}")
    train_ds = H8SWRForecastDataset(args.index_csv, "train", channels, stats, frame_cache_dir, sample_cache_dir)
    val_ds = H8SWRForecastDataset(args.index_csv, "val", channels, stats, frame_cache_dir, sample_cache_dir)
    future_frames = int(train_ds.df["future_frames"].iloc[0]) if "future_frames" in train_ds.df.columns else 2
    args.future_frames = future_frames
    print(f"Forecast setting: 4 history -> {future_frames} future frames")
    write_json(
        {
            "index_csv": str(args.index_csv),
            "channels": channels,
            "stats_json": str(stats_path),
            "args": serializable_args(args),
        },
        args.out_dir / "config.json",
    )
    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)
    print(
        f"Data: train={len(train_ds)} val={len(val_ds)} | "
        f"batch_size={args.batch_size} workers={args.num_workers} prefetch={args.prefetch_factor}",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = BasicSWR3DBackbone(
        in_channels=len(channels),
        future_frames=future_frames,
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    if bool(args.compile) and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    use_amp = bool(args.amp) and device.type == "cuda"

    history_path = args.out_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mae",
                "train_rmse",
                "train_bias",
                "val_loss",
                "val_mae",
                "val_rmse",
                "val_bias",
            ],
        )
        writer.writeheader()

    best_val_mae = float("inf")
    bad_epochs = 0
    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])

    try:
        for epoch in range(1, int(args.epochs) + 1):
            train_m = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                y_mean,
                y_std,
                float(args.grad_clip),
                use_amp,
                epoch_name=f"epoch {epoch:03d} train",
                log_interval=int(args.log_interval),
            )
            with torch.no_grad():
                val_m = run_epoch(
                    model,
                    val_loader,
                    None,
                    None,
                    device,
                    y_mean,
                    y_std,
                    float(args.grad_clip),
                    False,
                    epoch_name=f"epoch {epoch:03d} val",
                    log_interval=max(int(args.log_interval), 100),
                )

            row = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_mae": train_m["mae"],
                "train_rmse": train_m["rmse"],
                "train_bias": train_m["bias"],
                "val_loss": val_m["loss"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "val_bias": val_m["bias"],
            }
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_m['loss']:.4f} mae {train_m['mae']:.2f} | "
                f"val loss {val_m['loss']:.4f} mae {val_m['mae']:.2f} rmse {val_m['rmse']:.2f}"
            )

            save_checkpoint(
                args.out_dir / "last.pt",
                model,
                optimizer,
                args,
                stats,
                channels,
                epoch,
                best_val_mae,
            )

            if val_m["mae"] < best_val_mae:
                best_val_mae = val_m["mae"]
                bad_epochs = 0
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    stats,
                    channels,
                    epoch,
                    best_val_mae,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.patience):
                    print(f"Early stopping at epoch {epoch}, best val MAE={best_val_mae:.2f}")
                    break

        write_json({"best_val_mae": best_val_mae}, args.out_dir / "metrics.json")
    finally:
        train_ds.close()
        val_ds.close()


if __name__ == "__main__":
    main()
