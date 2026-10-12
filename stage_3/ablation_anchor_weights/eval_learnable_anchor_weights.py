#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

import stage3_motion_fusion_model_impl as m04
import stage3_build_swr_anchor_cache_impl as m00
from learnable_anchor_weights import install_learnable_anchor_weight_factory
from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS
from stage3_paths import (
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    RUNS_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir

f04 = m04.f04


USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "anchor_weight_init": "physical",
    "anchor_weight_random_std": 0.10,
    "run_dir": str(RUNS_DIR / "ablation_anchor_weights" / "learnable_physical_init"),
    "selection_dir": str(TARGET_SELECTION_DIR),
    "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
    "swr_anchor_cache_dir": "",
    "h8_center_index_csv": str(STAGE1_CENTER_INDEX_ALL),
    "h8_center_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
    "h8_center_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
    "h8_netcdf_glob": "H8_region_multivar_*_CENTER_BJT_*.nc",
    "swr_checkpoint": str(SWR_BASE_CHECKPOINT),
    "hour_round_tolerance_minutes": 10.0,
    "show_true_future_h8_swr": 1,
    "out_dir": "",
    "checkpoint": "",
    "model_name": "swr_anchor_motion_fusion_pv_nwp_learnable_physical_init",
    "base_model_name": "pv_nwp",
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1.0e-6,
    "freq_minutes": 15,
    "past_frames": 12,
    "future_frames": 8,
    "swr_history_frames": 4,
    "swr_future_anchors": 3,
    "normalize_swr_anchor": 1,
    "pv_hidden_channels": 48,
    "anchor_channels": 48,
    "horizon_embed_channels": 8,
    "dropout": 0.05,
    "large_delta_threshold": 0.10,
    "max_flow_pixels": 3.0,
    "delta_loss_weight": 0.2,
    "flow_smooth_loss_weight": 0.0,
    "batch_size": 8,
    "num_workers": 0,
    "device": "cuda",
    "sample_split": "test",
    "sample_count": 5,
    "sample_mode": "high_change",
    "sample_modes": "high_change,sunny",
    "indices": "",
    "visual_horizons": "1,4,8",
    "save_gif": 1,
    "dpi": 150,
    "seed": 2026,
}


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate motion-fusion SWR-anchor guided PV_norm field model and save per-horizon frame visualizations."
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if key in {"out_dir", "checkpoint"}:
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key.endswith("_root") or key == "run_dir" or key == "swr_checkpoint":
            parser.add_argument(f"--{key}", type=Path, default=Path(value) if value else Path(""))
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def _parse_indices(text: str, n: int) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += n
        if 0 <= idx < n:
            out.append(idx)
    return sorted(set(out))


def _parse_modes(text: str, fallback: str) -> list[str]:
    modes = [p.strip() for p in str(text).split(",") if p.strip()]
    return modes or [str(fallback)]


def _build_validation_state(args: argparse.Namespace, out_dir: Path):
    import torch
    from torch.utils.data import DataLoader

    station_split = f04._load_station_split(args)
    bounds = m04._bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(args.multisource_cache_dir))
    f04._assert_issue_time_alignment(args, ms_cache["time_df"])
    time_df = pd.read_csv(Path(args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(f04._timestamp_key)

    pv_cache = f04.ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, out_dir, time_df)

    swr_arr_all, swr_meta_all, swr_info = m04._load_prebuilt_swr_anchor_cache(Path(args.swr_anchor_cache_dir))
    base_datasets, swr_arrays, swr_meta, datasets, loaders = {}, {}, {}, {}, {}
    for split in ["train", "val", "test"]:
        sub, cache_indices, meta_df = m04._subset_time_df_by_swr_meta(time_df, swr_meta_all, split)
        base = f04.TargetExcludedPVNormFieldDataset(sub, pv_cache, ms_cache, bounds, pv_stats, ms_stats, str(args.base_model_name), args)
        base_datasets[split] = base
        swr_arrays[split] = np.asarray(swr_arr_all[cache_indices], dtype=np.float32)
        swr_meta[split] = meta_df

    swr_anchor_stats = m04._compute_swr_anchor_stats(swr_arrays["train"], Path(args.run_dir), force=False) if int(args.normalize_swr_anchor) else None
    for split in ["train", "val", "test"]:
        datasets[split] = m04.SWRAnchorPVFieldDataset(base_datasets[split], swr_arrays[split], args, swr_anchor_stats)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )

    base_in, future_channels = base_datasets["train"].source_dims()
    in_channels = int(base_in) + 2
    future_channels = int(future_channels) + 1
    model = m04.make_swr_anchor_attention_model(
        future_frames=int(args.future_frames),
        in_channels=in_channels,
        future_channels=future_channels,
        anchor_channels=2,
        pv_hidden=int(args.pv_hidden_channels),
        anchor_hidden=int(args.anchor_channels),
        horizon_embed_channels=int(args.horizon_embed_channels),
        dropout=float(args.dropout),
        swr_history_frames=int(args.swr_history_frames),
        swr_future_anchors=int(args.swr_future_anchors),
        max_flow_pixels=float(args.max_flow_pixels),
    )
    device = m04._device(args.device)
    ckpt_path = Path(args.checkpoint) if str(args.checkpoint).strip() else Path(args.run_dir) / f"{args.model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return {
        "model": model,
        "device": device,
        "datasets": datasets,
        "loaders": loaders,
        "pv_stats": pv_stats,
        "swr_anchor_stats": swr_anchor_stats,
        "swr_info": swr_info,
        "bounds": bounds,
        "checkpoint": ckpt_path,
    }


class TrueFutureSWRProvider:
    def __init__(self, args: argparse.Namespace, bounds: dict):
        import torch

        self.args = args
        self.pv_lon, self.pv_lat = f04._grid_lonlat(bounds, int(args.grid_size))
        ckpt = torch.load(Path(args.swr_checkpoint), map_location="cpu")
        stats = ckpt.get("stats", ckpt.get("base_stats"))
        if stats is None:
            raise KeyError(
                "SWR checkpoint contains neither stats nor base_stats: "
                f"{args.swr_checkpoint}"
            )
        self.channels = [str(x) for x in stats["channels"]]
        self.swr_pos = self.channels.index("SWR")
        self.channel_idx = list(range(len(self.channels)))
        self.min_clear_sky = float(stats["min_clear_sky"])
        self.ratio_clip = float(stats["ratio_clip"])
        reader_args = SimpleNamespace(
            h8_center_frame_cache_dir=Path(args.h8_center_frame_cache_dir),
            h8_center_processed_root=Path(args.h8_center_processed_root),
            h8_center_index_csv=Path(args.h8_center_index_csv),
            h8_netcdf_glob=str(args.h8_netcdf_glob),
            hour_round_tolerance_minutes=float(args.hour_round_tolerance_minutes),
        )
        self.reader = m00.CenterH8Reader(reader_args, self.channels)

    def close(self) -> None:
        self.reader.close()

    def true_future_ratio_context(
        self,
        issue: pd.Timestamp,
        attn_np: np.ndarray,
        future_count: int,
        observed_anchor_ratio: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, str]:
        issue = pd.Timestamp(issue).floor("h")
        y_times = [issue + pd.Timedelta(hours=j) for j in range(int(future_count))]
        hits = []
        missing = []
        for ts in y_times:
            key = m00._hour_key(ts, float(self.args.hour_round_tolerance_minutes))
            hit = self.reader.time_lookup.get(key)
            if hit is None:
                missing.append(m00._ts_text(ts))
            else:
                hits.append(hit)
        if missing:
            return None, "missing true future H8/SWR labels: " + "|".join(missing)
        raw = np.stack([self.reader.read_frame(hit) for hit in hits], axis=0).astype(np.float32)
        clear = self.reader.clear_sky(y_times)
        feat = m00.make_features(
            raw,
            clear,
            self.channel_idx,
            self.swr_pos,
            self.min_clear_sky,
            self.ratio_clip,
        )
        ratio = np.asarray(feat[:, self.swr_pos], dtype=np.float32)
        ratio64 = m00._resample_to_grid(ratio, self.reader.lon, self.reader.lat, self.pv_lon, self.pv_lat)
        full = np.zeros(
            (
                int(self.args.swr_history_frames) + int(self.args.swr_future_anchors),
                int(self.args.grid_size),
                int(self.args.grid_size),
            ),
            dtype=np.float32,
        )
        if observed_anchor_ratio is not None:
            observed = np.asarray(observed_anchor_ratio, dtype=np.float32)
            n = min(full.shape[0], observed.shape[0])
            full[:n] = observed[:n]
        full[int(self.args.swr_history_frames): int(self.args.swr_history_frames) + int(future_count)] = ratio64
        context = np.asarray(
            [(full * attn_np[h].mean(axis=(1, 2))[:, None, None]).sum(axis=0) for h in range(attn_np.shape[0])],
            dtype=np.float32,
        )
        return context, "ok"


def _select_samples(dataset, args: argparse.Namespace, pv_stats: dict) -> list[int]:
    if str(args.indices).strip():
        return _parse_indices(str(args.indices), len(dataset))
    n = min(int(args.sample_count), len(dataset))
    if n <= 0:
        return []
    mode = str(args.sample_mode).lower()
    if mode not in {"high_change", "sunny"}:
        rng = np.random.default_rng(int(args.seed))
        return sorted(rng.choice(len(dataset), size=n, replace=False).astype(int).tolist())
    scores = []
    mean = float(pv_stats["pv_field_mean"])
    std = max(float(pv_stats["pv_field_std"]), 1e-6)
    for idx in range(len(dataset)):
        x, _future_feat, swr, y, m = dataset[int(idx)]
        last = x[0, -1] * std + mean
        truth = y * std + mean
        mask = m > 0.5
        if not mask.any():
            score = -1.0
        elif mode == "sunny":
            future_mean = float(np.nanmean(truth[mask]))
            ramp_mean = float(np.nanmean(np.abs((truth - last[None])[mask])))
            temporal_std = float(np.nanmean(np.nanstd(np.where(mask, truth, np.nan), axis=0)))
            swr_ratio = dataset.denorm_swr(swr)[:, 0]
            future_swr_mean = float(np.nanmean(swr_ratio[-int(args.swr_future_anchors):]))
            # Prefer bright, stable, high-radiation cases. Penalize ramp/change so
            # "sunny" does not collapse into high-change partially cloudy samples.
            score = future_mean + 0.25 * future_swr_mean - 0.80 * ramp_mean - 0.35 * temporal_std
        else:
            score = float(np.nanmean(np.abs((truth - last[None])[mask])))
        scores.append((score, idx))
    scores.sort(reverse=True)
    return [idx for _score, idx in scores[:n]]


def _masked(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(np.asarray(mask) <= 0.5, np.asarray(arr, dtype=np.float32))


def _denorm_swr_ratio_feature(dataset, arr: np.ndarray) -> np.ndarray:
    """Map model-internal normalized SWR-ratio feature back to ratio-like scale."""
    stats = getattr(dataset, "swr_anchor_stats", None)
    if not int(getattr(dataset.args, "normalize_swr_anchor", 1)) or stats is None:
        return np.asarray(arr, dtype=np.float32)
    mean = float(np.asarray(stats["mean"], dtype=np.float32).reshape(-1)[0])
    std = max(float(np.asarray(stats["std"], dtype=np.float32).reshape(-1)[0]), 1.0e-6)
    return np.clip(np.asarray(arr, dtype=np.float32) * std + mean, 0.0, None).astype(np.float32)


def _learned_weight_matrix(model) -> np.ndarray:
    return model.current_future_anchor_weights().detach().cpu().numpy().astype(np.float32)


def _export_learned_weights(model, out_dir: Path, init_mode: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = _learned_weight_matrix(model)
    anchor_labels = ["A0_T-1h", "A1_T", "A2_T+1h", "A3_T+2h"]
    frame = pd.DataFrame(weights, columns=anchor_labels)
    frame.insert(0, "pv_horizon", [f"h{index + 1}" for index in range(len(frame))])
    frame["row_sum"] = weights.sum(axis=1)
    csv_path = out_dir / "learned_anchor_weights.csv"
    json_path = out_dir / "learned_anchor_weights.json"
    figure_path = out_dir / "learned_anchor_weights_heatmap.png"
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "anchor_weight_init": str(init_mode),
                "constraint": "softmax per PV horizon: non-negative weights with row sum 1",
                "anchors": anchor_labels,
                "weights": weights.tolist(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    image = ax.imshow(weights, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(anchor_labels)))
    ax.set_xticklabels(anchor_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(weights)))
    ax.set_yticklabels([f"h{index + 1}" for index in range(len(weights))])
    ax.set_xlabel("Hourly SWR anchor")
    ax.set_ylabel("15-minute PV horizon")
    ax.set_title(f"Learned temporal-anchor weights | init={init_mode}")
    for row in range(weights.shape[0]):
        for col in range(weights.shape[1]):
            ax.text(col, row, f"{weights[row, col]:.3f}", ha="center", va="center", color="white" if weights[row, col] > 0.45 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="softmax weight")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print("==== Learned Stage 3 anchor weights ====")
    print(frame.to_string(index=False))
    print("weights csv:", csv_path)
    return {
        "weights_csv": str(csv_path),
        "weights_json": str(json_path),
        "weights_figure": str(figure_path),
        "weights": weights.tolist(),
    }


def _save_anchor_fusion_plot(
    path: Path,
    issue: pd.Timestamp,
    swr_show: np.ndarray,
    attn_np: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = attn_np.mean(axis=(2, 3))
    anchor_indices = [3, 4, 5, 6]
    anchor_labels = ["A0 T-1h observed", "A1 T predicted", "A2 T+1h predicted", "A3 T+2h predicted"]
    ratio = np.asarray(swr_show[:, 0], dtype=np.float32)
    clear_wm2 = np.asarray(swr_show[:, 1], dtype=np.float32) * 1000.0
    hourly_swr = ratio * clear_wm2
    weighted_ratio = np.asarray(
        [(ratio * weights[h, :, None, None]).sum(axis=0) for h in range(weights.shape[0])],
        dtype=np.float32,
    )
    weighted_clear = np.asarray(
        [(clear_wm2 * weights[h, :, None, None]).sum(axis=0) for h in range(weights.shape[0])],
        dtype=np.float32,
    )
    fused_swr = weighted_ratio * weighted_clear
    vmax = max(100.0, float(np.nanquantile(np.concatenate([hourly_swr[anchor_indices].reshape(-1), fused_swr.reshape(-1)]), 0.99)))
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), squeeze=False)
    for column, (index, label) in enumerate(zip(anchor_indices, anchor_labels)):
        image = axes[0, column].imshow(hourly_swr[index], origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        axes[0, column].set_title(label)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        fig.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.02)
    for horizon in range(8):
        row = 1 + horizon // 4
        column = horizon % 4
        local = weights[horizon, anchor_indices]
        image = axes[row, column].imshow(fused_swr[horizon], origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        axes[row, column].set_title(
            f"h{horizon + 1} learned fusion\n"
            + " ".join(f"{value:.2f}" for value in local),
            fontsize=9,
        )
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
        fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Learned SWR-anchor fusion | issue={issue:%Y-%m-%d %H:%M:%S}\n"
        "top: 4 hourly physical SWR anchors | lower rows: 8 learned 15-minute SWR contexts",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _save_frame_plot(
    path: Path,
    issue: pd.Timestamp,
    h: int,
    latest: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    warped: np.ndarray,
    direct: np.ndarray,
    gate: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    swr_future: np.ndarray,
    true_swr_future: np.ndarray | None,
    aligned_swr_feature: np.ndarray,
    swr_attn: np.ndarray,
    display_ranges: dict,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(display_ranges["pv_vmin"])
    vmax = float(display_ranges["pv_vmax"])
    dmax = float(display_ranges["delta_abs"])
    err = np.abs(pred - truth)
    err_hi = float(display_ranges["err_hi"])
    swr_hi = float(display_ranges["swr_hi"])
    true_swr_hi = float(display_ranges["true_swr_hi"])
    flow_abs = float(display_ranges["flow_abs"])
    flow_mag_hi = float(display_ranges["flow_mag_hi"])
    flow_u = flow[0]
    flow_v = flow[1]
    flow_mag = np.sqrt(flow_u ** 2 + flow_v ** 2)

    panels = [
        ("latest", latest, "viridis", vmin, vmax),
        (f"truth h{h + 1}", truth, "viridis", vmin, vmax),
        (f"pred h{h + 1}", pred, "viridis", vmin, vmax),
        (f"warped h{h + 1}", warped, "viridis", vmin, vmax),
        (f"direct h{h + 1}", direct, "viridis", vmin, vmax),
        (f"truth-latest h{h + 1}", truth - latest, "coolwarm", -dmax, dmax),
        (f"pred-latest h{h + 1}", pred - latest, "coolwarm", -dmax, dmax),
        (f"warped-latest h{h + 1}", warped - latest, "coolwarm", -dmax, dmax),
        (f"abs error h{h + 1}", err, "magma", 0.0, err_hi),
        (f"fusion gate h{h + 1}", gate, "viridis", 0.0, 1.0),
        (f"flow u h{h + 1}", flow_u, "coolwarm", -flow_abs, flow_abs),
        (f"flow v h{h + 1}", flow_v, "coolwarm", -flow_abs, flow_abs),
        (f"flow mag h{h + 1}", flow_mag, "magma", 0.0, flow_mag_hi),
        ("learned-weight SWR context", swr_future, "viridis", 0.0, swr_hi),
        (
            "true future H8/SWR context",
            true_swr_future if true_swr_future is not None else np.full_like(swr_future, np.nan),
            "viridis",
            0.0,
            true_swr_hi,
        ),
        ("learned anchor weights", swr_attn, "bar", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(18.5, 14.4), squeeze=False)
    for ax, (title, arr, cmap, lo, hi) in zip(axes.reshape(-1), panels):
        if cmap == "bar":
            weights = np.asarray(arr, dtype=np.float32)
            if weights.ndim == 3:
                weights = weights.mean(axis=(1, 2))
            elif weights.ndim == 2:
                weights = np.asarray([weights.mean()], dtype=np.float32)
            labels = ["x T-4h", "x T-3h", "x T-2h", "A0 T-1h", "A1 T", "A2 T+1h", "A3 T+2h"][: len(weights)]
            colors = ["#c9ced6", "#c9ced6", "#c9ced6", "#4c78a8", "#f58518", "#54a24b", "#e45756"][: len(weights)]
            ax.bar(np.arange(len(weights)), weights, color=colors)
            ax.set_ylim(float(lo), float(hi))
            ax.set_xticks(np.arange(len(weights)))
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("weight")
            ax.grid(axis="y", alpha=0.25)
        else:
            show = np.asarray(arr, dtype=np.float32).copy()
            if "SWR" not in title and "attention" not in title and "weight" not in title and "flow" not in title and "gate" not in title:
                show = _masked(show, mask)
            im = ax.imshow(show, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
    for ax in axes.reshape(-1)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"SWR-anchor PV field frame | issue={issue:%Y-%m-%d %H:%M:%S} | h{h + 1}", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _save_case_visuals(
    case_dir: Path,
    case_rank: int,
    idx: int,
    dataset,
    model,
    device: str,
    pv_stats: dict,
    args: argparse.Namespace,
    true_swr_provider: TrueFutureSWRProvider | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    ensure_dir(case_dir)
    x, future_feat, swr, y, m = dataset[int(idx)]
    with torch.no_grad():
        pred_z, attn, aligned_swr, warped_z, direct_z, gate_t, flow_t = model(
            torch.from_numpy(x[None]).to(device),
            torch.from_numpy(future_feat[None]).to(device),
            torch.from_numpy(swr[None]).to(device),
            return_attention=True,
        )
    pred = f04._denorm_field(pred_z.detach().cpu().numpy()[0], pv_stats)
    warped = f04._denorm_field(warped_z.detach().cpu().numpy()[0], pv_stats)
    direct = f04._denorm_field(direct_z.detach().cpu().numpy()[0], pv_stats)
    gate_np = gate_t.detach().cpu().numpy()[0]
    flow_np = flow_t.detach().cpu().numpy()[0]
    truth = f04._denorm_field(y, pv_stats)
    latest = f04._denorm_field(x[0, -1], pv_stats)
    swr_show = dataset.denorm_swr(swr)
    attn_np = attn.detach().cpu().numpy()[0]  # [future, anchor, H, W]
    aligned_swr_np = _denorm_swr_ratio_feature(dataset, aligned_swr.detach().cpu().numpy()[0, :, 0])
    swr_context = np.asarray(
        [(swr_show[:, 0] * attn_np[h].mean(axis=(1, 2))[:, None, None]).sum(axis=0) for h in range(pred.shape[0])],
        dtype=np.float32,
    )
    issue = pd.Timestamp(dataset.time_df.iloc[int(idx)]["issue_time"])
    true_swr_context = None
    true_swr_status = "disabled"
    if true_swr_provider is not None:
        true_swr_context, true_swr_status = true_swr_provider.true_future_ratio_context(
            issue,
            attn_np,
            int(args.swr_future_anchors),
            swr_show[:, 0],
        )
    mask = m

    # Overview grid for quick inspection.
    horizons = m04._parse_visual_horizons(str(args.visual_horizons), int(args.future_frames))
    vals = np.concatenate([truth[mask > 0.5], pred[mask > 0.5], warped[mask > 0.5], direct[mask > 0.5]]) if np.any(mask > 0.5) else np.asarray([0.0, 1.0])
    vmin, vmax = float(np.nanquantile(vals, 0.02)), float(np.nanquantile(vals, 0.98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    dvals = np.concatenate(
        [
            (truth - latest[None])[mask > 0.5],
            (pred - latest[None])[mask > 0.5],
            (warped - latest[None])[mask > 0.5],
        ]
    ) if np.any(mask > 0.5) else np.asarray([0.0])
    dmax = max(0.05, float(np.nanquantile(np.abs(dvals), 0.98)))
    err = np.abs(pred - truth)
    err_hi = max(0.05, float(np.nanquantile(err[mask > 0.5], 0.98))) if np.any(mask > 0.5) else 0.2
    swr_scale_vals = [swr_context.reshape(-1), aligned_swr_np.reshape(-1)]
    if true_swr_context is not None:
        swr_scale_vals.append(true_swr_context.reshape(-1))
    swr_hi = max(1.2, float(np.nanquantile(np.concatenate(swr_scale_vals), 0.98)))
    true_swr_hi = swr_hi
    align_lo = 0.0
    align_hi = swr_hi
    flow_abs = max(float(args.max_flow_pixels), float(np.nanquantile(np.abs(flow_np), 0.98)))
    flow_mag_hi = max(0.25, float(np.nanquantile(np.sqrt(flow_np[:, 0] ** 2 + flow_np[:, 1] ** 2), 0.98)))
    display_ranges = {
        "pv_vmin": vmin,
        "pv_vmax": vmax,
        "delta_abs": dmax,
        "err_hi": err_hi,
        "swr_hi": swr_hi,
        "true_swr_hi": true_swr_hi,
        "align_lo": align_lo,
        "align_hi": align_hi,
        "flow_abs": flow_abs,
        "flow_mag_hi": flow_mag_hi,
    }
    fig, axes = plt.subplots(13, len(horizons), figsize=(4.1 * len(horizons), 34.0), squeeze=False)
    for c, h in enumerate(horizons):
        true_panel = (
            true_swr_context[h]
            if true_swr_context is not None
            else np.full_like(swr_context[h], np.nan, dtype=np.float32)
        )
        flow_u = flow_np[h, 0]
        flow_v = flow_np[h, 1]
        flow_mag = np.sqrt(flow_u ** 2 + flow_v ** 2)
        panels = [
            (f"truth h{h + 1}", truth[h], "viridis", vmin, vmax),
            (f"pred h{h + 1}", pred[h], "viridis", vmin, vmax),
            (f"warped h{h + 1}", warped[h], "viridis", vmin, vmax),
            (f"direct h{h + 1}", direct[h], "viridis", vmin, vmax),
            (f"truth-latest h{h + 1}", truth[h] - latest, "coolwarm", -dmax, dmax),
            (f"pred-latest h{h + 1}", pred[h] - latest, "coolwarm", -dmax, dmax),
            (f"warped-latest h{h + 1}", warped[h] - latest, "coolwarm", -dmax, dmax),
            (f"gate h{h + 1}", gate_np[h], "viridis", 0.0, 1.0),
            (f"flow u h{h + 1}", flow_u, "coolwarm", -flow_abs, flow_abs),
            (f"flow v h{h + 1}", flow_v, "coolwarm", -flow_abs, flow_abs),
            (f"flow mag h{h + 1}", flow_mag, "magma", 0.0, flow_mag_hi),
            (f"learned-weight SWR context h{h + 1}", swr_context[h], "viridis", 0.0, swr_hi),
            (f"true future H8/SWR context h{h + 1}", true_panel, "viridis", 0.0, true_swr_hi),
        ]
        for r, (title, arr, cmap, lo, hi) in enumerate(panels):
            show = np.asarray(arr, dtype=np.float32).copy()
            if r < 7:
                show = _masked(show, mask[h])
            im = axes[r, c].imshow(show, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            axes[r, c].set_title(title)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
    fig.suptitle(f"Overview | case={case_rank:03d} idx={idx} issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
    fig.tight_layout()
    overview_path = case_dir / f"case_{case_rank:03d}_idx_{idx:05d}_overview.png"
    fig.savefig(overview_path, dpi=int(args.dpi))
    plt.close(fig)

    fusion_path = case_dir / f"case_{case_rank:03d}_idx_{idx:05d}_learned_anchor_fusion.png"
    _save_anchor_fusion_plot(fusion_path, issue, swr_show, attn_np, int(args.dpi))

    frame_dir = ensure_dir(case_dir / "frames_h01_h08")
    frame_paths = []
    for h in range(pred.shape[0]):
        frame_path = frame_dir / f"case_{case_rank:03d}_idx_{idx:05d}_h{h + 1:02d}.png"
        _save_frame_plot(
            frame_path,
            issue,
            h,
            latest,
            truth[h],
            pred[h],
            warped[h],
            direct[h],
            gate_np[h],
            flow_np[h],
            mask[h],
            swr_context[h],
            true_swr_context[h] if true_swr_context is not None else None,
            aligned_swr_np[h],
            attn_np[h],
            display_ranges,
            int(args.dpi),
        )
        frame_paths.append(str(frame_path))

    gif_path = ""
    if int(args.save_gif):
        try:
            import imageio.v2 as imageio

            images = [imageio.imread(p) for p in frame_paths]
            gif = case_dir / f"case_{case_rank:03d}_idx_{idx:05d}_horizon_animation.gif"
            imageio.mimsave(gif, images, duration=0.65)
            gif_path = str(gif)
        except Exception as exc:
            print(f"[validate_swr_anchor] gif skipped for idx={idx}: {exc}", flush=True)

    return {
        "case_rank": int(case_rank),
        "dataset_idx": int(idx),
        "issue_time": str(issue),
        "overview": str(overview_path),
        "learned_anchor_fusion": str(fusion_path),
        "frame_dir": str(frame_dir),
        "gif": gif_path,
        "true_future_h8_swr_status": true_swr_status,
        "pv_vmin": vmin,
        "pv_vmax": vmax,
        "delta_abs": dmax,
        "err_hi": err_hi,
        "swr_hi": swr_hi,
        "true_swr_hi": true_swr_hi,
        "align_lo": align_lo,
        "align_hi": align_hi,
        "flow_abs": flow_abs,
        "flow_mag_hi": flow_mag_hi,
        "gate_mean": float(np.nanmean(gate_np)),
        "flow_mag_mean": float(np.nanmean(np.sqrt(flow_np[:, 0] ** 2 + flow_np[:, 1] ** 2))),
    }


def _plot_metric_summary(summary: pd.DataFrame, out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = summary[(summary["eval_split"].astype(str) == "test") & (summary["horizon"].astype(str).str.match(r"h\d+"))].copy()
    if sub.empty:
        return ""
    sub["h"] = sub["horizon"].str.replace("h", "", regex=False).astype(int)
    sub = sub.sort_values("h")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sub["h"], sub["mae_pv_norm_field"], marker="o", label="MAE")
    ax.plot(sub["h"], sub["rmse_pv_norm_field"], marker="s", label="RMSE")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("PV_norm field error")
    ax.set_title("SWR-anchor guided PV field validation | test")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "test_horizon_metric_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if not str(args.swr_anchor_cache_dir).strip() or str(args.swr_anchor_cache_dir) == ".":
        args.swr_anchor_cache_dir = resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(Path(args.swr_anchor_cache_dir), Path(args.swr_checkpoint))
    install_learnable_anchor_weight_factory(
        m04,
        fixed_weights=STAGE3_FUTURE_ANCHOR_WEIGHTS,
        init_mode=str(args.anchor_weight_init),
        seed=int(args.seed),
        random_std=float(args.anchor_weight_random_std),
    )

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(Path(args.out_dir) if str(args.out_dir).strip() else run_dir / "05_validation_visuals")
    state = _build_validation_state(args, out_dir)
    model = state["model"]
    device = state["device"]
    datasets = state["datasets"]
    loaders = state["loaders"]
    pv_stats = state["pv_stats"]
    learned_weight_artifacts = _export_learned_weights(model, out_dir, str(args.anchor_weight_init))

    metric_frames = []
    for split in ["val", "test"]:
        metric_frames.append(m04.evaluate_model(model, loaders[split], device, str(args.model_name), split, pv_stats, float(args.large_delta_threshold)))
    summary = pd.concat(metric_frames, ignore_index=True)
    summary_path = out_dir / "swr_anchor_guided_validation_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    metric_fig = _plot_metric_summary(summary, out_dir)

    # Basic overview figures using the existing training-script visualizer.
    basic_dir = ensure_dir(out_dir / "basic_sample_comparisons")
    m04.save_sample_comparisons(model, datasets[str(args.sample_split)], device, basic_dir, args, pv_stats)

    sample_split = str(args.sample_split)
    dataset = datasets[sample_split]
    true_swr_provider = None
    if int(args.show_true_future_h8_swr):
        try:
            true_swr_provider = TrueFutureSWRProvider(args, state["bounds"])
        except Exception as exc:
            print(f"[validate_swr_anchor] true future H8/SWR visual disabled: {exc}", flush=True)
            true_swr_provider = None
    modes = _parse_modes(getattr(args, "sample_modes", ""), str(args.sample_mode))
    all_rows = []
    frame_index_paths = {}
    try:
        for mode in modes:
            mode_args = SimpleNamespace(**vars(args))
            mode_args.sample_mode = mode
            indices = _select_samples(dataset, mode_args, pv_stats)
            frame_root = ensure_dir(out_dir / f"per_frame_{sample_split}_{mode}")
            rows = []
            for rank, idx in enumerate(indices, start=1):
                case_dir = ensure_dir(frame_root / f"case_{rank:03d}_idx_{idx:05d}")
                print(
                    f"[validate_swr_anchor] saving {mode} per-frame case {rank}/{len(indices)} idx={idx}",
                    flush=True,
                )
                item = _save_case_visuals(case_dir, rank, idx, dataset, model, device, pv_stats, mode_args, true_swr_provider)
                item["sample_mode"] = mode
                rows.append(item)
                all_rows.append(item)
            mode_frame_index = pd.DataFrame(rows)
            mode_frame_index_path = out_dir / f"per_frame_case_index_{mode}.csv"
            mode_frame_index.to_csv(mode_frame_index_path, index=False)
            frame_index_paths[mode] = str(mode_frame_index_path)
    finally:
        if true_swr_provider is not None:
            true_swr_provider.close()
    frame_index = pd.DataFrame(all_rows)
    frame_index_path = out_dir / "per_frame_case_index.csv"
    frame_index.to_csv(frame_index_path, index=False)

    with open(out_dir / "05_validation_visuals_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "purpose": "Validation and per-horizon visualization for motion-fusion SWR-anchor guided PV_norm field model.",
                "checkpoint": str(state["checkpoint"]),
                "run_dir": str(run_dir),
                "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
                "sample_split": sample_split,
                "sample_modes": modes,
                "sample_count_per_mode": int(args.sample_count),
                "metric_summary": str(summary_path),
                "metric_figure": metric_fig,
                "per_frame_case_index": str(frame_index_path),
                "per_mode_case_indices": frame_index_paths,
                "true_future_h8_swr_visual": bool(true_swr_provider is not None),
                "learned_anchor_weights": learned_weight_artifacts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("==== SWR-anchor guided validation summary ====")
    show = summary[summary["horizon"].isin(["all", "h1", "h4", "h8", "horizon_diversity"])].copy()
    print(show.to_string(index=False) if not show.empty else "EMPTY")
    print(f"summary: {summary_path}")
    print(f"metric figure: {metric_fig}")
    print(f"per-frame cases: {frame_index_path}")
    print(f"figures: {out_dir}")


if __name__ == "__main__":
    main()
