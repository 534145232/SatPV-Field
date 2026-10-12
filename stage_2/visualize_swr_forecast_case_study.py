#!/usr/bin/env python3
"""Render qualitative SWR forecasting cases for the Stage 2 residual ConvLSTM module."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import xarray as xr

from train_3dcnn_ratio_swr import (
    H8RatioDataset,
    STAGE1_CENTER_CLEAR_SKY_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_SAMPLE_CACHE_DIR,
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


# Same SWR palette as Framework_draw/generate_real_h8_swr_svg_assets.py.
COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]
ERROR_STOPS = [
    (0.00, (255, 255, 217)),
    (0.30, (222, 235, 247)),
    (0.55, (254, 204, 92)),
    (0.78, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]

NINGBO_LON_MIN, NINGBO_LON_MAX = 121.0, 122.0
NINGBO_LAT_MIN, NINGBO_LAT_MAX = 29.1, 30.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--loss_variant", choices=list(LOSS_VARIANTS), default="v0_full")
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--index_csv", type=Path, default=STAGE1_CENTER_INDEX_ALL)
    parser.add_argument("--frame_cache_dir", type=Path, default=STAGE1_CENTER_FRAME_CACHE_DIR)
    parser.add_argument("--sample_cache_dir", type=Path, default=STAGE1_CENTER_SAMPLE_CACHE_DIR)
    parser.add_argument("--clear_sky_dir", type=Path, default=STAGE1_CENTER_CLEAR_SKY_CACHE_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "paper_figures" / "swr_forecast_cases")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="lcig_swr_anchor",
        help="Output filename prefix used to avoid confusion with PV-field qualitative cases.",
    )
    parser.add_argument("--selection", choices=["largest_gain", "random", "worst_base"], default="largest_gain")
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument(
        "--issue_times",
        type=str,
        default="20230608_120000",
        help="Comma-separated issue times to force-render. Matching ignores seconds, e.g. 20230608_120000.",
    )
    parser.add_argument("--scan_count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_stripe_quantile", type=float, default=0.65)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--residual_vmax", type=float, default=180.0)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--save_pdf", type=int, default=1)
    parser.add_argument("--save_svg", type=int, default=1)
    parser.add_argument(
        "--lead_labels",
        type=str,
        default="T,T+1 h,T+2 h",
        help=(
            "Comma-separated row labels for the future SWR frames. "
            "Default matches the LCIG anchors used by SAG-PVFF."
        ),
    )
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


def make_dataset(args: argparse.Namespace, stats: dict) -> H8RatioDataset:
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


def load_residual_model(args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint_path = resolve_checkpoint(Path(args.run_root) / str(args.loss_variant) / "best.pt")
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
    print(f"[LOAD] residual {args.loss_variant}: {checkpoint_path}", flush=True)
    return {
        "model": model,
        "checkpoint": checkpoint_path,
        "args": checkpoint_args,
        "residual_stats": checkpoint["residual_stats"],
        "checkpoint_data": checkpoint,
    }


def candidate_indices(n: int, scan_count: int, seed: int) -> np.ndarray:
    if scan_count <= 0 or scan_count >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(scan_count), replace=False)).astype(int)


def normalize_issue_key(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[:12]


def parse_issue_times(value: str) -> list[str]:
    return [normalize_issue_key(item) for item in str(value).split(",") if normalize_issue_key(item)]


def find_issue_indices(dataset: H8RatioDataset, issue_times: str) -> list[dict]:
    wanted = parse_issue_times(issue_times)
    if not wanted:
        return []
    rows = []
    seen = set()
    wanted_set = set(wanted)
    for idx, row in dataset.df.reset_index(drop=True).iterrows():
        issue = str(row.get("issue_time_bjt", ""))
        key = normalize_issue_key(issue)
        if key in wanted_set and idx not in seen:
            rows.append(
                {
                    "idx": int(idx),
                    "issue_time_bjt": issue,
                    "future_times_bjt": "",
                    "base_mae": float("nan"),
                    "final_mae": float("nan"),
                    "mae_gain": float("nan"),
                    "truth_std": float("nan"),
                    "base_ratio_vertical_stripe_score": float("nan"),
                    "selection_reason": "forced_issue_time",
                }
            )
            seen.add(idx)
    found_keys = {normalize_issue_key(row["issue_time_bjt"]) for row in rows}
    missing = [key for key in wanted if key not in found_keys]
    if missing:
        print(f"[warn] forced issue times not found in {dataset.index_csv}: {', '.join(missing)}", flush=True)
    return rows


@torch.no_grad()
def predict_batch(batch: dict, base_model, residual_item: dict, device: torch.device) -> dict[str, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    truth = batch["raw_y"].to(device, non_blocking=True).float()
    mask = batch["mask"].to(device, non_blocking=True).float()
    clear_y = batch["clear_y"].to(device, non_blocking=True).float()
    base_ratio = base_model(x)
    variant_args = residual_item["args"]
    residual_stats = residual_item["residual_stats"]
    base_phy = torch.clamp(base_ratio.float() * clear_y, 0.0, float(variant_args.get("pred_clip", 1200.0)))
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
    return {"truth": truth, "mask": mask, "base_ratio": base_ratio.float(), "base": base_phy, "residual": residual_phy, "final": final_phy}


def masked_mae(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(truth)
    diff = torch.where(valid, (pred - truth).abs(), torch.zeros_like(pred))
    denom = valid.float().flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff.flatten(1).sum(dim=1) / denom).detach().cpu().numpy()


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


def moving_average_reflect(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def profile_mean(arr: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    counts = valid.sum(axis=axis)
    sums = np.where(valid, arr, 0.0).sum(axis=axis)
    profile = np.full_like(sums, np.nan, dtype=np.float32)
    ok = counts > max(3, 0.05 * valid.shape[axis])
    profile[ok] = sums[ok] / np.maximum(counts[ok], 1)
    return profile


def vertical_stripe_score(field: np.ndarray) -> float:
    valid = np.isfinite(field)
    if int(valid.sum()) < 16:
        return float("nan")
    median = float(np.nanmedian(field))
    safe = np.where(valid, field, median).astype(np.float32)
    background = smooth2d_reflect(safe, window=15)
    highpass = np.where(valid, safe - background, 0.0).astype(np.float32)
    hp_vals = highpass[valid]
    hp_std = float(np.std(hp_vals)) if hp_vals.size else float("nan")
    col_profile = profile_mean(highpass, valid, axis=0)
    col_vals = col_profile[np.isfinite(col_profile)]
    if col_vals.size < 9 or not np.isfinite(hp_std):
        return float("nan")
    col_hf = col_vals - moving_average_reflect(col_vals, window=9)
    return float(np.std(col_hf) / max(hp_std, 1e-6))


def batch_base_ratio_stripe_scores(pred: dict[str, torch.Tensor]) -> list[float]:
    base_ratio = pred["base_ratio"].detach().cpu().numpy().astype(np.float32)
    mask = pred["mask"].detach().cpu().numpy() > 0.5
    scores = []
    for i in range(base_ratio.shape[0]):
        horizon_scores = []
        for h in range(base_ratio.shape[1]):
            arr = np.where(mask[i, h], base_ratio[i, h], np.nan)
            score = vertical_stripe_score(arr)
            if np.isfinite(score):
                horizon_scores.append(score)
        scores.append(float(np.mean(horizon_scores)) if horizon_scores else float("nan"))
    return scores


def score_candidates(dataset: H8RatioDataset, base_model, residual_item: dict, args: argparse.Namespace, device: torch.device) -> list[dict]:
    indices = candidate_indices(len(dataset), int(args.scan_count), int(args.seed))
    subset = Subset(dataset, indices.tolist())
    loader = make_loader(subset, args)
    rows = []
    offset = 0
    for batch_id, batch in enumerate(loader, start=1):
        pred = predict_batch(batch, base_model, residual_item, device)
        base_mae = masked_mae(pred["base"], pred["truth"], pred["mask"])
        final_mae = masked_mae(pred["final"], pred["truth"], pred["mask"])
        stripe_scores = batch_base_ratio_stripe_scores(pred)
        truth_std = []
        truth_np = pred["truth"].detach().cpu().numpy()
        mask_np = pred["mask"].detach().cpu().numpy() > 0.5
        for i in range(truth_np.shape[0]):
            vals = truth_np[i][mask_np[i]]
            truth_std.append(float(np.nanstd(vals)) if vals.size else float("nan"))
        for local in range(len(base_mae)):
            idx = int(indices[offset + local])
            item = dataset[idx]
            meta = item.get("meta", {})
            rows.append(
                {
                    "idx": idx,
                    "issue_time_bjt": str(meta.get("issue_time_bjt", "")),
                    "future_times_bjt": str(meta.get("future_times_bjt", "")),
                    "base_mae": float(base_mae[local]),
                    "final_mae": float(final_mae[local]),
                    "mae_gain": float(base_mae[local] - final_mae[local]),
                    "truth_std": float(truth_std[local]),
                    "base_ratio_vertical_stripe_score": float(stripe_scores[local]),
                    "selection_reason": str(args.selection),
                }
            )
        offset += len(base_mae)
        if batch_id == 1 or batch_id % 10 == 0 or batch_id == len(loader):
            print(f"[scan] batch {batch_id}/{len(loader)}", flush=True)
    return rows


def choose_cases(scores: list[dict], args: argparse.Namespace) -> list[dict]:
    if not scores:
        return []
    rows = list(scores)
    n = min(int(args.sample_count), len(rows))
    if str(args.selection) == "random":
        rng = np.random.default_rng(int(args.seed))
        selected = rng.choice(np.arange(len(rows)), size=n, replace=False)
        return [rows[int(i)] for i in selected]
    if str(args.selection) == "worst_base":
        return sorted(rows, key=lambda r: r["base_mae"], reverse=True)[:n]
    stripe_values = np.asarray(
        [
            float(row["base_ratio_vertical_stripe_score"])
            for row in rows
            if np.isfinite(float(row["base_ratio_vertical_stripe_score"]))
        ],
        dtype=np.float32,
    )
    if stripe_values.size:
        threshold = float(np.quantile(stripe_values, float(args.max_stripe_quantile)))
        filtered = [
            row
            for row in rows
            if not np.isfinite(float(row["base_ratio_vertical_stripe_score"]))
            or float(row["base_ratio_vertical_stripe_score"]) <= threshold
        ]
        if len(filtered) >= n:
            rows = filtered
            print(
                f"[select] filtered high-stripe candidates: keep={len(rows)} threshold={threshold:.4f}",
                flush=True,
            )
    return sorted(rows, key=lambda r: (r["mae_gain"], r["truth_std"]), reverse=True)[:n]


def merge_forced_and_selected(forced: list[dict], selected: list[dict], sample_count: int) -> list[dict]:
    merged = []
    seen = set()
    for row in forced + selected:
        idx = int(row["idx"])
        if idx in seen:
            continue
        merged.append(row)
        seen.add(idx)
        if len(merged) >= max(1, int(sample_count)):
            break
    return merged


def to_masked_np(tensor: torch.Tensor, mask: torch.Tensor, batch_index: int) -> np.ndarray:
    arr = tensor[batch_index].detach().cpu().numpy().astype(np.float32)
    valid = mask[batch_index].detach().cpu().numpy() > 0.5
    return np.where(valid, arr, np.nan)


def load_grid_axes(dataset: H8RatioDataset, idx: int) -> tuple[np.ndarray, np.ndarray]:
    row = dataset.df.iloc[int(idx)]
    nc_path = Path(str(row["nc_path"]))
    with xr.open_dataset(nc_path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    return lat, lon


def ascending_grid(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.asarray(arr, dtype=np.float32)
    lat_out = np.asarray(lat, dtype=np.float32).copy()
    lon_out = np.asarray(lon, dtype=np.float32).copy()
    if lat_out[0] > lat_out[-1]:
        lat_out = lat_out[::-1]
        out = out[..., ::-1, :]
    if lon_out[0] > lon_out[-1]:
        lon_out = lon_out[::-1]
        out = out[..., ::-1]
    return out, lat_out, lon_out


def crop_by_lonlat(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr2, lat2, lon2 = ascending_grid(arr, lat, lon)
    lat_mask = (lat2 >= NINGBO_LAT_MIN) & (lat2 <= NINGBO_LAT_MAX)
    lon_mask = (lon2 >= NINGBO_LON_MIN) & (lon2 <= NINGBO_LON_MAX)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            "Ningbo region does not overlap Stage 1 center SWR grid: "
            f"lon={float(lon2.min()):.3f}-{float(lon2.max()):.3f}, "
            f"lat={float(lat2.min()):.3f}-{float(lat2.max()):.3f}"
        )
    return arr2[np.ix_(lat_mask, lon_mask)], lat2[lat_mask], lon2[lon_mask]


def lonlat_extent(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    return [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]


def future_lead_labels(args: argparse.Namespace, future_frames: int) -> list[str]:
    labels = [item.strip() for item in str(args.lead_labels).split(",") if item.strip()]
    if len(labels) >= int(future_frames):
        return labels[: int(future_frames)]
    fallback = labels[:]
    for h in range(len(fallback), int(future_frames)):
        fallback.append("T" if h == 0 else f"T+{h} h")
    return fallback


def render_case(path_stem: Path, case: dict, pred: dict[str, torch.Tensor], meta: dict, args: argparse.Namespace) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    swr_cmap = ListedColormap(palette_from_stops(COLOR_STOPS), name="shared_swr")
    err_cmap = ListedColormap(palette_from_stops(ERROR_STOPS), name="shared_error")
    for cmap in [swr_cmap, err_cmap]:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    truth = to_masked_np(pred["truth"], pred["mask"], 0)
    final = to_masked_np(pred["final"], pred["mask"], 0)
    lat = np.asarray(case["lat"], dtype=np.float32)
    lon = np.asarray(case["lon"], dtype=np.float32)
    final_large, lat_up, lon_up = ascending_grid(final, lat, lon)
    truth_large, _, _ = ascending_grid(truth, lat, lon)
    large_extent = lonlat_extent(lat_up, lon_up)
    future_frames = int(truth.shape[0])
    lead_labels = future_lead_labels(args, future_frames)
    fig, axes = plt.subplots(future_frames, 4, figsize=(8.8, 5.5), squeeze=False)
    first_swr = None
    first_error = None
    col_titles = [
        "Large-context forecast",
        "Target-region GT",
        "Target-region forecast",
        "Absolute error",
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, pad=5)
    for h in range(future_frames):
        gt_crop, crop_lat, crop_lon = crop_by_lonlat(truth_large[h], lat_up, lon_up)
        final_crop, _, _ = crop_by_lonlat(final_large[h], lat_up, lon_up)
        err_crop = np.abs(final_crop - gt_crop)
        crop_extent = lonlat_extent(crop_lat, crop_lon)
        panels = [
            (final_large[h], swr_cmap, 0.0, float(args.vmax), large_extent, "large"),
            (gt_crop, swr_cmap, 0.0, float(args.vmax), crop_extent, "crop"),
            (final_crop, swr_cmap, 0.0, float(args.vmax), crop_extent, "crop"),
            (err_crop, err_cmap, 0.0, float(args.error_vmax), crop_extent, "crop"),
        ]
        for c, (values, cmap, vmin, vmax, extent, panel_kind) in enumerate(panels):
            im = axes[h, c].imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            axes[h, c].set_xticks([])
            axes[h, c].set_yticks([])
            for spine in axes[h, c].spines.values():
                spine.set_visible(False)
            if panel_kind == "large":
                axes[h, c].add_patch(
                    Rectangle(
                        (NINGBO_LON_MIN, NINGBO_LAT_MIN),
                        NINGBO_LON_MAX - NINGBO_LON_MIN,
                        NINGBO_LAT_MAX - NINGBO_LAT_MIN,
                        fill=False,
                        edgecolor="#e53935",
                        linewidth=1.7,
                        linestyle=(0, (2.2, 1.4)),
                        alpha=0.95,
                    )
                )
                axes[h, c].set_xlim(large_extent[0], large_extent[1])
                axes[h, c].set_ylim(large_extent[2], large_extent[3])
            else:
                axes[h, c].set_xlim(NINGBO_LON_MIN, NINGBO_LON_MAX)
                axes[h, c].set_ylim(NINGBO_LAT_MIN, NINGBO_LAT_MAX)
            if c == 0:
                axes[h, c].set_ylabel(lead_labels[h], fontsize=10, rotation=90, labelpad=8)
                if first_swr is None:
                    first_swr = im
            elif c == 3 and first_error is None:
                first_error = im

    fig.subplots_adjust(left=0.055, right=0.90, top=0.88, bottom=0.06, wspace=0.045, hspace=0.08)
    if first_swr is not None:
        cax = fig.add_axes([0.915, 0.56, 0.014, 0.31])
        cb = fig.colorbar(first_swr, cax=cax)
        cb.set_label("SWR (W m$^{-2}$)", fontsize=8)
        cb.ax.tick_params(labelsize=7, length=2)
    if first_error is not None:
        cax = fig.add_axes([0.915, 0.16, 0.014, 0.31])
        cb = fig.colorbar(first_error, cax=cax)
        cb.set_label("Absolute error", fontsize=8)
        cb.ax.tick_params(labelsize=7, length=2)

    png = path_stem.with_suffix(".png")
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
    pdf = ""
    svg = ""
    if int(args.save_pdf):
        pdf = str(path_stem.with_suffix(".pdf"))
        fig.savefig(pdf, bbox_inches="tight")
    if int(args.save_svg):
        svg = str(path_stem.with_suffix(".svg"))
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": pdf, "svg": svg, "issue_time_bjt": str(meta.get("issue_time_bjt", ""))}


@torch.no_grad()
def render_selected(dataset: H8RatioDataset, selected: list[dict], base_model, residual_item: dict, args: argparse.Namespace, device: torch.device) -> list[dict]:
    rows = []
    for rank, item in enumerate(selected, start=1):
        sample = dataset[int(item["idx"])]
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
            if key in {"x", "raw_y", "mask", "clear_y"}
        }
        pred = predict_batch(batch, base_model, residual_item, device)
        meta = sample.get("meta", {})
        lat, lon = load_grid_axes(dataset, int(item["idx"]))
        render_item = {**item, "lat": lat, "lon": lon}
        safe_issue = str(meta.get("issue_time_bjt", f"idx{item['idx']}")).replace(":", "").replace(" ", "_")
        stem = Path(args.out_dir) / f"{args.output_prefix}_{rank:02d}_idx{int(item['idx']):05d}_{safe_issue}"
        rendered = render_case(stem, render_item, pred, meta, args)
        row = {**item, **rendered}
        rows.append(row)
        print(
            f"[case {rank:02d}] idx={item['idx']} gain={item['mae_gain']:.2f} "
            f"base={item['base_mae']:.2f} final={item['final_mae']:.2f} png={rendered['png']}",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    base_model, base_stats, base_checkpoint, temporal_head = load_base_ratio_model(Path(args.base_checkpoint), device, "conv3d_decoder")
    print(f"[LOAD] base {temporal_head}: {base_checkpoint}", flush=True)
    residual_item = load_residual_model(args, device)
    dataset = make_dataset(args, base_stats)
    try:
        scores = score_candidates(dataset, base_model, residual_item, args, device)
        forced = find_issue_indices(dataset, str(args.issue_times))
        selected = merge_forced_and_selected(forced, choose_cases(scores, args), int(args.sample_count))
        if not selected:
            raise RuntimeError("No samples were selected.")
        rows = render_selected(dataset, selected, base_model, residual_item, args, device)
        write_csv(Path(args.out_dir) / "swr_case_scores.csv", scores)
        write_csv(Path(args.out_dir) / "swr_case_selected.csv", rows)
        print(f"[write] {Path(args.out_dir) / 'swr_case_selected.csv'}", flush=True)
        print(f"[write] figures under {args.out_dir}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
