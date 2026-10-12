"""Evaluate and visualize the Stage 2 center-only ratio 3D CNN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    Ratio3DCNN,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    STAGE2_RUNS_DIR,
    write_json,
)


DEFAULT_CHECKPOINT = (
    STAGE2_RUNS_DIR
    / "h8_swr_3dcnn_ratio_stage1_center_only_4h3f"
    / "head_last_mean"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the center-only Stage 2 ratio 3D CNN and draw samples."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_vis", type=int, default=8)
    parser.add_argument("--vis_stride", type=int, default=20)
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    return parser.parse_args()


def fallback_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "best.pt":
        fallback = path.with_name("last.pt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_model(path: Path, device: torch.device):
    checkpoint_path = fallback_checkpoint(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = checkpoint["stats"]
    train_args = checkpoint.get("args", {})
    temporal_head = str(
        checkpoint.get("temporal_head", train_args.get("temporal_head", "last_mean"))
    )
    history_frames = int(train_args.get("history_frames", stats.get("history_frames", 4)))
    future_frames = int(train_args.get("future_frames", stats.get("future_frames", 3)))
    model = Ratio3DCNN(
        in_channels=len(stats["channels"]),
        future_frames=future_frames,
        history_frames=history_frames,
        width=int(train_args.get("width", 48)),
        depth=int(train_args.get("depth", 4)),
        dropout=float(train_args.get("dropout", 0.05)),
        ratio_clip=float(stats["ratio_clip"]),
        temporal_head=temporal_head,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, stats, checkpoint_path, temporal_head


def make_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )


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


class MetricAccumulator:
    def __init__(self, future_frames: int):
        self.future_frames = future_frames
        self.count = np.zeros(future_frames, dtype=np.float64)
        self.abs_err = np.zeros(future_frames, dtype=np.float64)
        self.sq_err = np.zeros(future_frames, dtype=np.float64)
        self.err = np.zeros(future_frames, dtype=np.float64)
        self.ratio_abs_err = np.zeros(future_frames, dtype=np.float64)
        self.y_sum = np.zeros(future_frames, dtype=np.float64)
        self.y_sq_sum = np.zeros(future_frames, dtype=np.float64)

    def update(
        self,
        pred_ratio: torch.Tensor,
        true_ratio: torch.Tensor,
        clear_y: torch.Tensor,
        raw_y: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        pred_swr = pred_ratio * clear_y
        valid = (mask > 0.5) & torch.isfinite(raw_y) & torch.isfinite(pred_swr)
        for horizon in range(self.future_frames):
            horizon_valid = valid[:, horizon]
            if not torch.any(horizon_valid):
                continue
            pred_values = pred_swr[:, horizon][horizon_valid]
            true_values = raw_y[:, horizon][horizon_valid]
            ratio_error = (
                pred_ratio[:, horizon][horizon_valid]
                - true_ratio[:, horizon][horizon_valid]
            )
            error = pred_values - true_values
            self.count[horizon] += float(error.numel())
            self.abs_err[horizon] += float(error.abs().sum().item())
            self.sq_err[horizon] += float((error * error).sum().item())
            self.err[horizon] += float(error.sum().item())
            self.ratio_abs_err[horizon] += float(ratio_error.abs().sum().item())
            self.y_sum[horizon] += float(true_values.sum().item())
            self.y_sq_sum[horizon] += float((true_values * true_values).sum().item())

    def one(self, horizon: int) -> dict[str, float]:
        count = max(float(self.count[horizon]), 1.0)
        mae = float(self.abs_err[horizon] / count)
        rmse = float(np.sqrt(self.sq_err[horizon] / count))
        bias = float(self.err[horizon] / count)
        ratio_mae = float(self.ratio_abs_err[horizon] / count)
        sst = float(self.y_sq_sum[horizon] - self.y_sum[horizon] ** 2 / count)
        r2 = float(1.0 - self.sq_err[horizon] / sst) if sst > 1e-12 else float("nan")
        return {
            "count": int(self.count[horizon]),
            "ratio_mae": ratio_mae,
            "swr_mae": mae,
            "swr_rmse": rmse,
            "swr_bias": bias,
            "swr_r2": r2,
        }

    def summary(self) -> dict[str, float]:
        count = max(float(self.count.sum()), 1.0)
        sst = float(self.y_sq_sum.sum() - self.y_sum.sum() ** 2 / count)
        return {
            "count": int(self.count.sum()),
            "ratio_mae": float(self.ratio_abs_err.sum() / count),
            "swr_mae": float(self.abs_err.sum() / count),
            "swr_rmse": float(np.sqrt(self.sq_err.sum() / count)),
            "swr_bias": float(self.err.sum() / count),
            "swr_r2": float(1.0 - self.sq_err.sum() / sst) if sst > 1e-12 else float("nan"),
        }


@torch.no_grad()
def evaluate(model, loader, device: torch.device, future_frames: int) -> dict:
    accumulator = MetricAccumulator(future_frames)
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        clear_y = batch["clear_y"].to(device, non_blocking=True)
        raw_y = batch["raw_y"].to(device, non_blocking=True)
        pred_ratio = model(x)
        accumulator.update(pred_ratio, y, clear_y, raw_y, mask)
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f"eval batch {batch_index}/{len(loader)}", flush=True)
    return {
        "overall": accumulator.summary(),
        "by_horizon": {
            f"+{horizon + 1}": accumulator.one(horizon)
            for horizon in range(future_frames)
        },
    }


def plot_sample(
    path: Path,
    true_swr: np.ndarray,
    pred_swr: np.ndarray,
    clear_swr: np.ndarray,
    pred_ratio: np.ndarray,
    meta: dict,
    vmax: float,
    error_vmax: float,
) -> None:
    future_frames = int(true_swr.shape[0])
    error = pred_swr - true_swr
    fig, axes = plt.subplots(
        future_frames,
        5,
        figsize=(19, 3.5 * future_frames),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for horizon in range(future_frames):
        panels = [
            (f"clear SWR +{horizon + 1}", clear_swr[horizon], 0.0, vmax, "turbo"),
            (f"truth SWR +{horizon + 1}", true_swr[horizon], 0.0, vmax, "turbo"),
            (f"pred SWR +{horizon + 1}", pred_swr[horizon], 0.0, vmax, "turbo"),
            (f"pred ratio +{horizon + 1}", pred_ratio[horizon], 0.0, 1.6, "viridis"),
            (
                f"error +{horizon + 1}",
                error[horizon],
                -error_vmax,
                error_vmax,
                "coolwarm",
            ),
        ]
        for ax, (title, values, vmin, vmax_i, cmap) in zip(axes[horizon], panels):
            image = ax.imshow(values, origin="upper", vmin=vmin, vmax=vmax_i, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"issue {meta['issue_time_bjt']} | future {meta.get('future_times_bjt', '')}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def visualize_samples(
    model,
    dataset: H8RatioDataset,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    made = 0
    for index in range(0, len(dataset), max(1, int(args.vis_stride))):
        if made >= int(args.num_vis):
            break
        sample = dataset[index]
        pred_ratio = model(sample["x"].unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        clear_y = sample["clear_y"].numpy()
        true_swr = sample["raw_y"].numpy()
        finite = np.isfinite(true_swr)
        pred_swr = np.where(finite, pred_ratio * clear_y, np.nan)
        true_swr = np.where(finite, true_swr, np.nan)
        pred_ratio = np.where(finite, pred_ratio, np.nan)
        path = out_dir / f"{args.split}_sample_{index:05d}_ratio_eval.png"
        plot_sample(
            path,
            true_swr,
            pred_swr,
            clear_y,
            pred_ratio,
            sample["meta"],
            float(args.vmax),
            float(args.error_vmax),
        )
        print("saved:", path)
        made += 1


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, stats, checkpoint, temporal_head = load_model(args.checkpoint, device)
    out_dir = args.out_dir or (checkpoint.parent / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = make_dataset_args(args)
    dataset = H8RatioDataset(
        dataset_args,
        args.split,
        stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    loader = make_loader(dataset, args)
    try:
        result = evaluate(model, loader, device, int(dataset.future_frames))
        result["checkpoint"] = str(checkpoint)
        result["split"] = args.split
        result["temporal_head"] = temporal_head
        write_json(result, out_dir / "metrics.json")
        rows = [{"scope": "overall", **result["overall"]}]
        rows.extend(
            {"scope": horizon, **metrics}
            for horizon, metrics in result["by_horizon"].items()
        )
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("metrics:", result)
        visualize_samples(model, dataset, out_dir, args, device)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
