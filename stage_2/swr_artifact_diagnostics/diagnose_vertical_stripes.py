#!/usr/bin/env python3
"""Diagnose vertical stripe artifacts in Stage 2 SWR base predictions.

The script compares raw truth, clear-sky SWR, target ratio, base predicted
ratio, base physical SWR, and optionally residual-corrected outputs.  It writes
case figures plus a CSV with simple anisotropy/column-artifact scores.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from train_3dcnn_ratio_swr import (  # noqa: E402
    H8RatioDataset,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
)
from train_residual_convlstm_correction_swr import (  # noqa: E402
    DEFAULT_BASE_CKPT,
    DEFAULT_RUN_ROOT,
    LOSS_VARIANTS,
    ResidualConvLSTMCorrection,
    build_residual_input,
    load_base_ratio_model,
    resolve_checkpoint,
    source_args,
)


COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]

RESIDUAL_STOPS = [
    (0.00, (8, 48, 107)),
    (0.30, (33, 113, 181)),
    (0.50, (255, 255, 217)),
    (0.70, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--loss_variant", choices=list(LOSS_VARIANTS), default="v0_full")
    parser.add_argument("--include_residual", type=int, default=1)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument(
        "--selection",
        choices=["worst_base_ratio_stripe", "worst_base_swr_stripe", "random"],
        default="worst_base_ratio_stripe",
    )
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--also_render_low_stripe", type=int, default=1)
    parser.add_argument("--scan_count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--swr_vmax", type=float, default=1000.0)
    parser.add_argument("--ratio_vmax", type=float, default=1.35)
    parser.add_argument("--residual_vmax", type=float, default=180.0)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def palette_from_stops(stops_in: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    stops = np.asarray([rgb for _, rgb in stops_in], dtype=np.float32)
    x = np.asarray([pos for pos, _ in stops_in], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.float32) / 255.0


def make_loader(dataset: H8RatioDataset | Subset, args: argparse.Namespace) -> DataLoader:
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


def make_dataset(args: argparse.Namespace, stats: dict[str, Any]) -> H8RatioDataset:
    dataset_args = source_args(args.index_csv, args.frame_cache_dir, args.sample_cache_dir, args.clear_sky_dir)
    return H8RatioDataset(
        dataset_args,
        str(args.split),
        stats,
        index_csv=args.index_csv,
        frame_cache_dir=args.frame_cache_dir,
        sample_cache_dir=args.sample_cache_dir,
        clear_sky_dir=args.clear_sky_dir,
    )


def load_residual_model(args: argparse.Namespace, device: torch.device, base_stats: dict[str, Any]) -> dict[str, Any] | None:
    if not int(args.include_residual):
        return None
    checkpoint_path = resolve_checkpoint(Path(args.run_root) / str(args.loss_variant) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = ResidualConvLSTMCorrection(
        in_channels=int(checkpoint_args.get("input_channels", len(base_stats["channels"]) + 2)),
        future_frames=int(checkpoint_args.get("future_frames", 3)),
        width=int(checkpoint_args.get("width", 48)),
        depth=int(checkpoint_args.get("depth", 2)),
        dropout=float(checkpoint_args.get("dropout", 0.05)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"[LOAD] residual {args.loss_variant}: {checkpoint_path}", flush=True)
    return {
        "model": model,
        "args": checkpoint_args,
        "residual_stats": checkpoint["residual_stats"],
        "checkpoint": checkpoint_path,
    }


def candidate_indices(n: int, scan_count: int, seed: int) -> np.ndarray:
    if scan_count <= 0 or scan_count >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(scan_count), replace=False)).astype(int)


@torch.no_grad()
def predict_batch(batch: dict[str, Any], base_model: torch.nn.Module, residual_item: dict[str, Any] | None, device: torch.device) -> dict[str, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    truth = batch["raw_y"].to(device, non_blocking=True).float()
    mask = batch["mask"].to(device, non_blocking=True).float()
    clear_y = batch["clear_y"].to(device, non_blocking=True).float()
    base_ratio = base_model(x).float()
    base_phy = torch.clamp(base_ratio * clear_y, 0.0, 1200.0)
    out = {
        "x": x,
        "truth": truth,
        "mask": mask,
        "clear_y": clear_y,
        "truth_ratio": torch.clamp(truth / clear_y.clamp_min(1.0), 0.0, 2.0),
        "base_ratio": base_ratio,
        "base_phy": base_phy,
    }
    if residual_item is not None:
        variant_args = residual_item["args"]
        residual_stats = residual_item["residual_stats"]
        residual_input = build_residual_input(
            x,
            base_phy,
            float(residual_stats["swr_mean"]),
            float(residual_stats["swr_std"]),
        )
        residual_phy = torch.clamp(
            residual_item["model"](residual_input).float() * float(variant_args.get("residual_scale", 100.0)),
            -float(variant_args.get("residual_clip", 200.0)),
            float(variant_args.get("residual_clip", 200.0)),
        )
        final_phy = torch.clamp(base_phy + residual_phy, 0.0, float(variant_args.get("pred_clip", 1200.0)))
        out["residual_phy"] = residual_phy
        out["final_phy"] = final_phy
    return out


def masked_numpy(tensor: torch.Tensor, mask: torch.Tensor, batch_index: int) -> np.ndarray:
    arr = tensor[batch_index].detach().cpu().numpy().astype(np.float32)
    valid = mask[batch_index].detach().cpu().numpy() > 0.5
    return np.where(valid, arr, np.nan)


def moving_average_reflect(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def moving_average_axis(arr: np.ndarray, window: int, axis: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    pad = window // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), csum], axis=axis)
    hi = np.arange(window, window + arr.shape[axis])
    lo = np.arange(0, arr.shape[axis])
    return (np.take(csum, hi, axis=axis) - np.take(csum, lo, axis=axis)) / float(window)


def smooth2d_reflect(arr: np.ndarray, window: int = 15) -> np.ndarray:
    smoothed = moving_average_axis(arr, window=window, axis=0)
    smoothed = moving_average_axis(smoothed, window=window, axis=1)
    return smoothed.astype(np.float32)


def profile_mean(arr: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    counts = valid.sum(axis=axis)
    sums = np.where(valid, arr, 0.0).sum(axis=axis)
    profile = np.full_like(sums, np.nan, dtype=np.float32)
    ok = counts > max(3, 0.05 * valid.shape[axis])
    profile[ok] = sums[ok] / np.maximum(counts[ok], 1)
    return profile


def artifact_scores(arr: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(arr)
    if int(valid.sum()) < 16:
        return {
            "field_std": float("nan"),
            "grad_x": float("nan"),
            "grad_y": float("nan"),
            "grad_x_over_y": float("nan"),
            "column_profile_std": float("nan"),
            "row_profile_std": float("nan"),
            "column_over_row_profile": float("nan"),
            "column_hf_score": float("nan"),
            "highpass_std": float("nan"),
            "highpass_grad_x_over_y": float("nan"),
            "vertical_stripe_score": float("nan"),
        }
    median = float(np.nanmedian(arr))
    safe = np.where(valid, arr, median).astype(np.float32)
    vals = safe[valid]
    field_std = float(np.nanstd(vals))
    dx_valid = valid[:, 1:] & valid[:, :-1]
    dy_valid = valid[1:, :] & valid[:-1, :]
    grad_x = float(np.mean(np.abs(safe[:, 1:] - safe[:, :-1])[dx_valid])) if np.any(dx_valid) else float("nan")
    grad_y = float(np.mean(np.abs(safe[1:, :] - safe[:-1, :])[dy_valid])) if np.any(dy_valid) else float("nan")
    col_profile = profile_mean(safe, valid, axis=0)
    row_profile = profile_mean(safe, valid, axis=1)
    col_vals = col_profile[np.isfinite(col_profile)]
    row_vals = row_profile[np.isfinite(row_profile)]
    col_std = float(np.std(col_vals)) if col_vals.size else float("nan")
    row_std = float(np.std(row_vals)) if row_vals.size else float("nan")
    if col_vals.size >= 9:
        col_hf = col_vals - moving_average_reflect(col_vals, window=9)
        column_hf_score = float(np.std(col_hf) / max(field_std, 1e-6))
    else:
        column_hf_score = float("nan")
    background = smooth2d_reflect(safe, window=15)
    highpass = np.where(valid, safe - background, 0.0).astype(np.float32)
    hp_vals = highpass[valid]
    highpass_std = float(np.std(hp_vals)) if hp_vals.size else float("nan")
    hp_dx = float(np.mean(np.abs(highpass[:, 1:] - highpass[:, :-1])[dx_valid])) if np.any(dx_valid) else float("nan")
    hp_dy = float(np.mean(np.abs(highpass[1:, :] - highpass[:-1, :])[dy_valid])) if np.any(dy_valid) else float("nan")
    hp_col_profile = profile_mean(highpass, valid, axis=0)
    hp_col_vals = hp_col_profile[np.isfinite(hp_col_profile)]
    if hp_col_vals.size >= 9 and np.isfinite(highpass_std):
        hp_col_hf = hp_col_vals - moving_average_reflect(hp_col_vals, window=9)
        vertical_stripe_score = float(np.std(hp_col_hf) / max(highpass_std, 1e-6))
    else:
        vertical_stripe_score = float("nan")
    return {
        "field_std": field_std,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "grad_x_over_y": grad_x / max(grad_y, 1e-6) if np.isfinite(grad_x) and np.isfinite(grad_y) else float("nan"),
        "column_profile_std": col_std,
        "row_profile_std": row_std,
        "column_over_row_profile": col_std / max(row_std, 1e-6) if np.isfinite(col_std) and np.isfinite(row_std) else float("nan"),
        "column_hf_score": column_hf_score,
        "highpass_std": highpass_std,
        "highpass_grad_x_over_y": hp_dx / max(hp_dy, 1e-6) if np.isfinite(hp_dx) and np.isfinite(hp_dy) else float("nan"),
        "vertical_stripe_score": vertical_stripe_score,
    }


def batch_rows(
    batch: dict[str, Any],
    pred: dict[str, torch.Tensor],
    base_stats: dict[str, Any],
    global_indices: np.ndarray,
    offset: int,
) -> list[dict[str, Any]]:
    rows = []
    mask = pred["mask"]
    swr_pos = list(base_stats["channels"]).index("SWR")
    x_mean = float(np.asarray(base_stats["x_mean"], dtype=np.float32)[swr_pos])
    x_std = float(np.asarray(base_stats["x_std"], dtype=np.float32)[swr_pos])
    input_swr_ratio = pred["x"][:, :, swr_pos].float() * x_std + x_mean
    field_tensors = {
        "truth_swr": pred["truth"],
        "clear_sky": pred["clear_y"],
        "truth_ratio": pred["truth_ratio"],
        "base_ratio": pred["base_ratio"],
        "base_swr": pred["base_phy"],
    }
    if "residual_phy" in pred:
        field_tensors["residual_swr"] = pred["residual_phy"]
        field_tensors["final_swr"] = pred["final_phy"]
    batch_size = pred["truth"].shape[0]
    future_frames = pred["truth"].shape[1]
    history_frames = input_swr_ratio.shape[1]
    for local in range(batch_size):
        meta = batch["meta"]
        issue = ""
        if isinstance(meta, dict) and "issue_time_bjt" in meta:
            value = meta["issue_time_bjt"]
            issue = str(value[local] if isinstance(value, list) else value)
        for h in range(future_frames):
            for field_name, tensor in field_tensors.items():
                arr = masked_numpy(tensor[:, h], mask[:, h], local)
                scores = artifact_scores(arr)
                rows.append(
                    {
                        "dataset_idx": int(global_indices[offset + local]),
                        "issue_time_bjt": issue,
                        "horizon": int(h + 1),
                        "field": field_name,
                        **scores,
                    }
                )
        hist_mask = mask[:, 0]
        for h in range(history_frames):
            arr = masked_numpy(input_swr_ratio[:, h], hist_mask, local)
            scores = artifact_scores(arr)
            rows.append(
                {
                    "dataset_idx": int(global_indices[offset + local]),
                    "issue_time_bjt": issue,
                    "horizon": int(h - history_frames + 1),
                    "field": "input_history_swr_ratio",
                    **scores,
                }
            )
    return rows


def choose_cases(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    by_index: dict[int, dict[str, float]] = {}
    target_field = "base_ratio" if args.selection == "worst_base_ratio_stripe" else "base_swr"
    for row in rows:
        idx = int(row["dataset_idx"])
        by_index.setdefault(idx, {"score": 0.0, "count": 0.0})
        if args.selection == "random":
            continue
        if row["field"] == target_field and int(row["horizon"]) > 0:
            score = float(row["vertical_stripe_score"])
            if np.isfinite(score):
                by_index[idx]["score"] += score
                by_index[idx]["count"] += 1.0
    indices = sorted(by_index)
    n = min(int(args.sample_count), len(indices))
    if args.selection == "random":
        rng = np.random.default_rng(int(args.seed))
        return [int(x) for x in rng.choice(indices, size=n, replace=False)]
    ranked = []
    for idx, item in by_index.items():
        denom = max(float(item["count"]), 1.0)
        ranked.append((float(item["score"]) / denom, idx))
    return [idx for _, idx in sorted(ranked, reverse=True)[:n]]


def choose_low_stripe_cases(rows: list[dict[str, Any]], args: argparse.Namespace, excluded: set[int]) -> list[int]:
    target_field = "base_ratio" if args.selection != "worst_base_swr_stripe" else "base_swr"
    by_index: dict[int, dict[str, float]] = {}
    for row in rows:
        idx = int(row["dataset_idx"])
        if idx in excluded:
            continue
        by_index.setdefault(idx, {"score": 0.0, "count": 0.0})
        if row["field"] == target_field and int(row["horizon"]) > 0:
            score = float(row["vertical_stripe_score"])
            if np.isfinite(score):
                by_index[idx]["score"] += score
                by_index[idx]["count"] += 1.0
    ranked = []
    for idx, item in by_index.items():
        if item["count"] > 0:
            ranked.append((float(item["score"]) / float(item["count"]), idx))
    n = min(max(1, int(args.sample_count) // 2), len(ranked))
    return [idx for _, idx in sorted(ranked)[:n]]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["field"]), int(row["horizon"])), []).append(row)
    summary = []
    keys = [
        "grad_x_over_y",
        "column_hf_score",
        "highpass_grad_x_over_y",
        "vertical_stripe_score",
    ]
    for (field, horizon), items in sorted(groups.items()):
        out: dict[str, Any] = {"field": field, "horizon": horizon, "n": len(items)}
        for key in keys:
            vals = np.asarray([float(item[key]) for item in items if np.isfinite(float(item[key]))], dtype=np.float32)
            if vals.size:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_median"] = float(np.median(vals))
                out[f"{key}_p90"] = float(np.percentile(vals, 90))
            else:
                out[f"{key}_mean"] = float("nan")
                out[f"{key}_median"] = float("nan")
                out[f"{key}_p90"] = float("nan")
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_case(
    path_stem: Path,
    dataset: H8RatioDataset,
    idx: int,
    base_model: torch.nn.Module,
    residual_item: dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    swr_cmap = ListedColormap(palette_from_stops(COLOR_STOPS), name="shared_swr")
    ratio_cmap = ListedColormap(palette_from_stops(COLOR_STOPS), name="shared_ratio")
    res_cmap = ListedColormap(palette_from_stops(RESIDUAL_STOPS), name="shared_residual")
    for cmap in [swr_cmap, ratio_cmap, res_cmap]:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    sample = dataset[idx]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in sample.items()
        if key in {"x", "raw_y", "mask", "clear_y"}
    }
    pred = predict_batch(batch, base_model, residual_item, device)
    mask = pred["mask"]
    panels = [
        ("Truth SWR", pred["truth"], swr_cmap, 0.0, float(args.swr_vmax)),
        ("Clear-sky", pred["clear_y"], swr_cmap, 0.0, float(args.swr_vmax)),
        ("Truth ratio", pred["truth_ratio"], ratio_cmap, 0.0, float(args.ratio_vmax)),
        ("Base ratio", pred["base_ratio"], ratio_cmap, 0.0, float(args.ratio_vmax)),
        ("Base SWR", pred["base_phy"], swr_cmap, 0.0, float(args.swr_vmax)),
    ]
    if "residual_phy" in pred:
        panels.extend(
            [
                ("Residual", pred["residual_phy"], res_cmap, -float(args.residual_vmax), float(args.residual_vmax)),
                ("Final SWR", pred["final_phy"], swr_cmap, 0.0, float(args.swr_vmax)),
            ]
        )
    future_frames = pred["truth"].shape[1]
    fig, axes = plt.subplots(future_frames, len(panels), figsize=(1.85 * len(panels), 1.85 * future_frames), squeeze=False)
    for col, (title, _, _, _, _) in enumerate(panels):
        axes[0, col].set_title(title, fontsize=8, pad=3)
    for h in range(future_frames):
        for col, (_, tensor, cmap, vmin, vmax) in enumerate(panels):
            arr = masked_numpy(tensor[:, h], mask[:, h], 0)
            axes[h, col].imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[h, col].set_xticks([])
            axes[h, col].set_yticks([])
            for spine in axes[h, col].spines.values():
                spine.set_visible(False)
            if col == 0:
                axes[h, col].set_ylabel(f"T+{h + 1}h", fontsize=8)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.03, wspace=0.03, hspace=0.06)
    fig.savefig(path_stem.with_suffix(".png"), dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(future_frames, 1, figsize=(5.6, 1.65 * future_frames), squeeze=False)
    profile_fields = [
        ("truth_ratio", pred["truth_ratio"], "#111111"),
        ("base_ratio", pred["base_ratio"], "#d94b3d"),
        ("clear_sky", pred["clear_y"] / max(float(args.swr_vmax), 1.0), "#577082"),
    ]
    for h in range(future_frames):
        ax = axes[h, 0]
        for label, tensor, color in profile_fields:
            arr = masked_numpy(tensor[:, h], mask[:, h], 0)
            valid = np.isfinite(arr)
            prof = profile_mean(np.where(valid, arr, 0.0), valid, axis=0)
            prof = prof[np.isfinite(prof)]
            if prof.size:
                prof = (prof - np.nanmean(prof)) / max(float(np.nanstd(prof)), 1e-6)
                ax.plot(prof, lw=1.2, color=color, label=label if h == 0 else None)
        ax.set_ylabel(f"T+{h + 1}h", fontsize=8)
        ax.tick_params(labelsize=7, length=2)
        ax.grid(True, alpha=0.18, lw=0.5)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    fig.savefig(path_stem.with_name(path_stem.name + "_column_profiles").with_suffix(".png"), dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(path_stem.with_name(path_stem.name + "_column_profiles").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    base_model, base_stats, base_checkpoint, temporal_head = load_base_ratio_model(Path(args.base_checkpoint), device, "conv3d_decoder")
    print(f"[LOAD] base {temporal_head}: {base_checkpoint}", flush=True)
    residual_item = load_residual_model(args, device, base_stats)
    dataset = make_dataset(args, base_stats)
    try:
        indices = candidate_indices(len(dataset), int(args.scan_count), int(args.seed))
        subset = Subset(dataset, indices.tolist())
        loader = make_loader(subset, args)
        rows: list[dict[str, Any]] = []
        offset = 0
        for batch_id, batch in enumerate(loader, start=1):
            pred = predict_batch(batch, base_model, residual_item, device)
            rows.extend(batch_rows(batch, pred, base_stats, indices, offset))
            offset += pred["truth"].shape[0]
            if batch_id == 1 or batch_id % 10 == 0 or batch_id == len(loader):
                print(f"[scan] batch {batch_id}/{len(loader)}", flush=True)
        metrics_csv = Path(args.out_dir) / "vertical_stripe_metrics.csv"
        write_csv(metrics_csv, rows)
        summary_csv = Path(args.out_dir) / "vertical_stripe_field_summary.csv"
        write_csv(summary_csv, summarize_rows(rows))
        selected = choose_cases(rows, args)
        low_selected = choose_low_stripe_cases(rows, args, set(selected)) if int(args.also_render_low_stripe) else []
        selected_rows = (
            [{"dataset_idx": idx, "stripe_group": "high"} for idx in selected]
            + [{"dataset_idx": idx, "stripe_group": "low"} for idx in low_selected]
        )
        write_csv(Path(args.out_dir) / "selected_cases.csv", selected_rows)
        for rank, idx in enumerate(selected, start=1):
            sample = dataset[idx]
            meta = sample.get("meta", {})
            issue = str(meta.get("issue_time_bjt", f"idx{idx}")).replace(":", "").replace(" ", "_")
            stem = Path(args.out_dir) / f"stripe_high_{rank:02d}_idx{idx:05d}_{issue}"
            render_case(stem, dataset, idx, base_model, residual_item, args, device)
            print(f"[case {rank:02d}] idx={idx} figure={stem.with_suffix('.png')}", flush=True)
        for rank, idx in enumerate(low_selected, start=1):
            sample = dataset[idx]
            meta = sample.get("meta", {})
            issue = str(meta.get("issue_time_bjt", f"idx{idx}")).replace(":", "").replace(" ", "_")
            stem = Path(args.out_dir) / f"stripe_low_{rank:02d}_idx{idx:05d}_{issue}"
            render_case(stem, dataset, idx, base_model, residual_item, args, device)
            print(f"[low case {rank:02d}] idx={idx} figure={stem.with_suffix('.png')}", flush=True)
        print(f"[write] metrics: {metrics_csv}", flush=True)
        print(f"[write] summary: {summary_csv}", flush=True)
        print(f"[write] figures under {args.out_dir}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
