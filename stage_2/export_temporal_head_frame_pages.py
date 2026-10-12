"""Export ordered SWR frame pages for temporal-head prediction comparisons."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_all_temporal_heads import (
    DEFAULT_RUN_ROOT,
    check_stats_compatible,
    load_available_models,
    make_dataset,
)
from train_3dcnn_ratio_swr import (
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save ordered history and future SWR pages for easy frame-by-frame review."
    )
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=20)
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices, for example: 0,20,100.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    return parser.parse_args()


def selected_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    if args.sample_indices.strip():
        indices = [int(value.strip()) for value in args.sample_indices.split(",") if value.strip()]
    else:
        indices = list(
            range(0, dataset_size, max(1, int(args.sample_stride)))
        )[: max(0, int(args.num_samples))]
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise IndexError(f"Sample indices out of range for dataset size {dataset_size}: {invalid}")
    return indices


def safe_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-")


def raw_history_swr(dataset, index: int) -> tuple[np.ndarray, np.ndarray]:
    row = dataset.df.iloc[index]
    global_index = int(row["_index_row"])
    if dataset.use_sample_cache:
        x_raw = np.asarray(dataset.x_arr[global_index], dtype=np.float32)
        clear = np.asarray(dataset.clear_arr[global_index], dtype=np.float32)
    else:
        x_raw, _, _, clear = dataset.reader.read(row)
    swr_index = dataset.cache_channels.index("SWR")
    history_swr = np.asarray(x_raw[:, swr_index], dtype=np.float32).copy()
    clear_x = np.asarray(clear[: dataset.history_frames], dtype=np.float32).copy()
    return history_swr, clear_x


def save_field(
    path: Path,
    values: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    image = ax.imshow(values, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_future_comparison_page(
    path: Path,
    true_swr: np.ndarray,
    clear_swr: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    vmax: float,
    error_vmax: float,
) -> None:
    heads = list(predictions)
    columns = 2 + len(heads)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(3.6 * columns, 7.0),
        constrained_layout=True,
    )
    panels = [
        ("clear SWR", clear_swr, 0.0, vmax, "turbo"),
        ("truth SWR", true_swr, 0.0, vmax, "turbo"),
    ]
    panels.extend(
        (f"{head} pred", predictions[head], 0.0, vmax, "turbo")
        for head in heads
    )
    for ax, (panel_title, values, vmin, vmax_i, cmap) in zip(axes[0], panels):
        image = ax.imshow(values, origin="upper", vmin=vmin, vmax=vmax_i, cmap=cmap)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    for index, ax in enumerate(axes[1]):
        ax.set_xticks([])
        ax.set_yticks([])
        if index < 2:
            ax.axis("off")
            continue
        head = heads[index - 2]
        error = predictions[head] - true_swr
        image = ax.imshow(
            error,
            origin="upper",
            vmin=-error_vmax,
            vmax=error_vmax,
            cmap="coolwarm",
        )
        ax.set_title(f"{head} error", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def export_sample(
    dataset,
    index: int,
    models: dict[str, dict],
    skipped: dict[str, str],
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    sample = dataset[index]
    row = dataset.df.iloc[index]
    history_swr, clear_x = raw_history_swr(dataset, index)
    clear_y = sample["clear_y"].numpy()
    true_swr = sample["raw_y"].numpy()
    valid = sample["mask"].numpy() > 0.5
    model_predictions = {}
    model_ratios = {}
    x = sample["x"].unsqueeze(0).to(device)
    for head, item in models.items():
        pred_ratio = item["model"](x).squeeze(0).cpu().numpy()
        pred_swr = pred_ratio * clear_y
        model_ratios[head] = pred_ratio
        model_predictions[head] = np.where(valid, pred_swr, np.nan)

    issue = str(sample["meta"]["issue_time_bjt"])
    sample_dir = out_dir / f"sample_{index:05d}_issue_{safe_token(issue)}"
    field_dir = sample_dir / "fields"
    sample_dir.mkdir(parents=True, exist_ok=True)

    history_times = [
        str(row.get(f"x_time_{frame}_bjt", f"history_{frame}"))
        for frame in range(dataset.history_frames)
    ]
    future_times = [
        str(row.get(f"y_time_{frame}_bjt", f"future_{frame}"))
        for frame in range(dataset.future_frames)
    ]
    metadata = {
        "split": args.split,
        "dataset_index": int(index),
        "global_index": int(row["_index_row"]),
        "issue_time_bjt": issue,
        "history_times_bjt": history_times,
        "future_times_bjt": future_times,
        "models": {
            head: str(item["checkpoint"])
            for head, item in models.items()
        },
        "skipped": skipped,
        "page_order": [
            *(f"{frame:02d}_history_swr.png" for frame in range(dataset.history_frames)),
            *(
                f"{dataset.history_frames + frame:02d}_future_comparison.png"
                for frame in range(dataset.future_frames)
            ),
        ],
    }
    write_json(metadata, sample_dir / "metadata.json")

    arrays = {
        "history_swr": history_swr,
        "history_clear_swr": clear_x,
        "truth_swr": true_swr,
        "future_clear_swr": clear_y,
        "future_valid_mask": valid.astype(np.uint8),
    }
    for head in models:
        arrays[f"{head}_pred_ratio"] = model_ratios[head]
        arrays[f"{head}_pred_swr"] = model_predictions[head]
    np.savez_compressed(sample_dir / "arrays.npz", **arrays)

    for frame in range(dataset.history_frames):
        label = f"history {frame - dataset.history_frames + 1:+d} | {history_times[frame]}"
        save_field(
            sample_dir / f"{frame:02d}_history_swr.png",
            history_swr[frame],
            label,
            0.0,
            float(args.vmax),
            "turbo",
        )
        save_field(
            field_dir / f"{frame:02d}_history_swr.png",
            history_swr[frame],
            label,
            0.0,
            float(args.vmax),
            "turbo",
        )

    for horizon in range(dataset.future_frames):
        frame_number = dataset.history_frames + horizon
        title = f"future +{horizon + 1} | {future_times[horizon]}"
        predictions = {
            head: model_predictions[head][horizon]
            for head in models
        }
        save_future_comparison_page(
            sample_dir / f"{frame_number:02d}_future_comparison.png",
            np.where(valid[horizon], true_swr[horizon], np.nan),
            clear_y[horizon],
            predictions,
            title,
            float(args.vmax),
            float(args.error_vmax),
        )
        save_field(
            field_dir / f"{frame_number:02d}_future_truth_swr.png",
            np.where(valid[horizon], true_swr[horizon], np.nan),
            f"{title} | truth SWR",
            0.0,
            float(args.vmax),
            "turbo",
        )
        for head, pred_swr in predictions.items():
            save_field(
                field_dir / f"{frame_number:02d}_future_{head}_pred_swr.png",
                pred_swr,
                f"{title} | {head} pred SWR",
                0.0,
                float(args.vmax),
                "turbo",
            )
            save_field(
                field_dir / f"{frame_number:02d}_future_{head}_error.png",
                pred_swr - true_swr[horizon],
                f"{title} | {head} error",
                -float(args.error_vmax),
                float(args.error_vmax),
                "coolwarm",
            )
    print("saved frame pages:", sample_dir, flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    models, skipped = load_available_models(args, device)
    check_stats_compatible(models)
    stats = next(iter(models.values()))["stats"]
    dataset = make_dataset(args, stats)
    out_dir = args.out_dir or (args.run_root / f"frame_pages_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        indices = selected_indices(args, len(dataset))
        print(f"Exporting {len(indices)} samples from {args.split}: {indices}", flush=True)
        for index in indices:
            export_sample(dataset, index, models, skipped, out_dir, args, device)
    finally:
        dataset.close()
    print("saved frame-page root:", out_dir, flush=True)


if __name__ == "__main__":
    main()
