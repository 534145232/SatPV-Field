"""Compare all available center-only temporal-head ablation checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_3dcnn_ratio_center_only import MetricAccumulator, fallback_checkpoint, load_model
from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    STAGE2_RUNS_DIR,
    write_json,
)


HEADS = ("last_mean", "last_only", "flatten", "conv3d_decoder")
DEFAULT_RUN_ROOT = (
    STAGE2_RUNS_DIR / "h8_swr_3dcnn_ratio_stage1_center_only_4h3f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all available center-only temporal heads together."
    )
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split. Default: test.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_vis", type=int, default=8)
    parser.add_argument("--vis_stride", type=int, default=20)
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    return parser.parse_args()


def checkpoint_candidates(run_root: Path) -> dict[str, Path]:
    return {
        head: run_root / f"head_{head}" / "best.pt"
        for head in HEADS
    }


def load_available_models(args: argparse.Namespace, device: torch.device):
    loaded = {}
    skipped = {}
    for requested_head, candidate in checkpoint_candidates(args.run_root).items():
        try:
            model, stats, checkpoint, actual_head = load_model(candidate, device)
        except FileNotFoundError as exc:
            skipped[requested_head] = str(exc)
            print(f"[SKIP] {requested_head}: {exc}", flush=True)
            continue
        if actual_head != requested_head:
            raise AssertionError(
                f"Checkpoint {checkpoint} reports temporal_head={actual_head}, "
                f"expected {requested_head}."
            )
        loaded[requested_head] = {
            "model": model,
            "stats": stats,
            "checkpoint": checkpoint,
        }
        print(f"[LOAD] {requested_head}: {checkpoint}", flush=True)
    if not loaded:
        raise FileNotFoundError(f"No temporal-head checkpoints found below {args.run_root}")
    return loaded, skipped


def make_dataset(args: argparse.Namespace, stats: dict) -> H8RatioDataset:
    dataset_args = argparse.Namespace(
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )
    return H8RatioDataset(
        dataset_args,
        args.split,
        stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )


def check_stats_compatible(models: dict[str, dict]) -> None:
    reference_head = next(iter(models))
    reference = models[reference_head]["stats"]
    keys = ("channels", "min_clear_sky", "ratio_clip", "x_mean", "x_std")
    for head, item in models.items():
        stats = item["stats"]
        for key in keys:
            if stats.get(key) != reference.get(key):
                raise AssertionError(
                    f"Stats differ between {reference_head} and {head}: key={key}"
                )


@torch.no_grad()
def evaluate_models(
    models: dict[str, dict],
    dataset: H8RatioDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict]:
    future_frames = int(dataset.future_frames)
    accumulators = {
        head: MetricAccumulator(future_frames)
        for head in models
    }
    total_batches = (len(dataset) + int(args.batch_size) - 1) // int(args.batch_size)
    for batch_number, start in enumerate(range(0, len(dataset), int(args.batch_size)), start=1):
        samples = [
            dataset[index]
            for index in range(start, min(start + int(args.batch_size), len(dataset)))
        ]
        x = torch.stack([sample["x"] for sample in samples]).to(device, non_blocking=True)
        y = torch.stack([sample["y"] for sample in samples]).to(device, non_blocking=True)
        mask = torch.stack([sample["mask"] for sample in samples]).to(device, non_blocking=True)
        clear_y = torch.stack([sample["clear_y"] for sample in samples]).to(device, non_blocking=True)
        raw_y = torch.stack([sample["raw_y"] for sample in samples]).to(device, non_blocking=True)
        for head, item in models.items():
            pred_ratio = item["model"](x)
            accumulators[head].update(pred_ratio, y, clear_y, raw_y, mask)
        if batch_number == 1 or batch_number % 20 == 0 or batch_number == total_batches:
            print(f"eval batch {batch_number}/{total_batches}", flush=True)

    return {
        head: {
            "overall": accumulator.summary(),
            "by_horizon": {
                f"+{horizon + 1}": accumulator.one(horizon)
                for horizon in range(future_frames)
            },
            "checkpoint": str(models[head]["checkpoint"]),
        }
        for head, accumulator in accumulators.items()
    }


def save_metrics(
    out_dir: Path,
    split: str,
    metrics_by_head: dict[str, dict],
    skipped: dict[str, str],
) -> None:
    payload = {
        "split": split,
        "models": metrics_by_head,
        "skipped": skipped,
    }
    write_json(payload, out_dir / "metrics_all_heads.json")

    rows = []
    for head, result in metrics_by_head.items():
        rows.append({"temporal_head": head, "scope": "overall", **result["overall"]})
        rows.extend(
            {"temporal_head": head, "scope": horizon, **metrics}
            for horizon, metrics in result["by_horizon"].items()
        )
    with (out_dir / "metrics_all_heads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_bar(
    out_dir: Path,
    metrics_by_head: dict[str, dict],
) -> None:
    heads = list(metrics_by_head)
    first_head = heads[0]
    horizons = ("overall", *metrics_by_head[first_head]["by_horizon"].keys())
    x = np.arange(len(heads))
    width = min(0.8 / max(len(horizons), 1), 0.18)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, horizon in enumerate(horizons):
        values = [
            metrics_by_head[head]["overall"]["swr_mae"]
            if horizon == "overall"
            else metrics_by_head[head]["by_horizon"][horizon]["swr_mae"]
            for head in heads
        ]
        offset = index - (len(horizons) - 1) / 2
        ax.bar(x + offset * width, values, width=width, label=horizon)
    ax.set_xticks(x, heads, rotation=15)
    ax.set_ylabel("SWR MAE (W/m2)")
    ax.set_title("Center-only temporal-head ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "temporal_head_mae_comparison.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def predict_sample(
    models: dict[str, dict],
    sample: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x = sample["x"].unsqueeze(0).to(device)
    clear_y = sample["clear_y"].numpy()
    predictions = {}
    for head, item in models.items():
        pred_ratio = item["model"](x).squeeze(0).cpu().numpy()
        predictions[head] = pred_ratio * clear_y
    return predictions


def plot_sample_comparison(
    path: Path,
    sample: dict,
    predictions: dict[str, np.ndarray],
    vmax: float,
    error_vmax: float,
) -> None:
    true_swr = sample["raw_y"].numpy()
    heads = list(predictions)
    future_frames = int(true_swr.shape[0])
    columns = 1 + 2 * len(heads)
    fig, axes = plt.subplots(
        future_frames,
        columns,
        figsize=(3.2 * columns, 3.35 * future_frames),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for horizon in range(future_frames):
        truth_ax = axes[horizon, 0]
        image = truth_ax.imshow(true_swr[horizon], origin="upper", vmin=0.0, vmax=vmax, cmap="turbo")
        truth_ax.set_title(f"truth SWR +{horizon + 1}", fontsize=9)
        truth_ax.set_xticks([])
        truth_ax.set_yticks([])
        fig.colorbar(image, ax=truth_ax, fraction=0.046, pad=0.02)
        for index, head in enumerate(heads):
            pred = predictions[head][horizon]
            pred_ax = axes[horizon, 1 + 2 * index]
            error_ax = axes[horizon, 2 + 2 * index]
            pred_image = pred_ax.imshow(pred, origin="upper", vmin=0.0, vmax=vmax, cmap="turbo")
            err_image = error_ax.imshow(
                pred - true_swr[horizon],
                origin="upper",
                vmin=-error_vmax,
                vmax=error_vmax,
                cmap="coolwarm",
            )
            pred_ax.set_title(f"{head} pred +{horizon + 1}", fontsize=9)
            error_ax.set_title(f"{head} error +{horizon + 1}", fontsize=9)
            for ax in (pred_ax, error_ax):
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(pred_image, ax=pred_ax, fraction=0.046, pad=0.02)
            fig.colorbar(err_image, ax=error_ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"issue {sample['meta']['issue_time_bjt']} | "
        f"future {sample['meta'].get('future_times_bjt', '')}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_samples(
    models: dict[str, dict],
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
        predictions = predict_sample(models, sample, device)
        path = out_dir / f"{args.split}_sample_{index:05d}_all_temporal_heads.png"
        plot_sample_comparison(
            path,
            sample,
            predictions,
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
    models, skipped = load_available_models(args, device)
    check_stats_compatible(models)
    stats = next(iter(models.values()))["stats"]
    out_dir = args.out_dir or (args.run_root / f"eval_all_heads_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Evaluating split: {args.split} ===", flush=True)
    dataset = make_dataset(args, stats)
    try:
        metrics_by_head = evaluate_models(models, dataset, args, device)
        save_metrics(out_dir, args.split, metrics_by_head, skipped)
        plot_metric_bar(out_dir, metrics_by_head)
        visualize_samples(models, dataset, out_dir, args, device)
        print("saved comparison:", out_dir)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
