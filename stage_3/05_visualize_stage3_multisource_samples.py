#!/usr/bin/env python3
"""Visualize Stage 3 PV, H8, NWP and optional SWR-anchor cache samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_paths import (
    MULTISOURCE_CACHE_DIR,
    MULTISOURCE_VIS_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    ensure_stage3_dirs,
)
from stage3_cli import inject_defaults
import stage3_visualize_multisource_impl as module
from stage3_swr_anchor_cache import resolve_active_cache_dir


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def timestamp_key(value: pd.Timestamp) -> int:
    return int(pd.Timestamp(value).timestamp())


def robust_limits(array: np.ndarray, low: float = 0.02, high: float = 0.98) -> tuple[float, float]:
    finite = np.asarray(array, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanquantile(finite, low))
    hi = float(np.nanquantile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.nanmin(finite)), float(np.nanmax(finite) + 1.0e-6)
    return lo, hi


def render_anchor_package(module, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchor_cache_dir = resolve_active_cache_dir(required=False)
    anchor_path = anchor_cache_dir / "swr_anchor_64.npy"
    feature_path = anchor_cache_dir / "h8_history_features_64.npy"
    meta_path = anchor_cache_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    info_path = anchor_cache_dir / "center_h8_swr_anchor_input_cache_meta.json"
    if not all(path.exists() for path in (anchor_path, feature_path, meta_path, info_path)):
        print(
            "[visualize_stage3] optional future SWR-anchor cache not found; "
            "PV/H8/NWP static visualizations were saved. Run 03_build_center_h8_swr_anchor_cache.py "
            "and rerun this script to add predicted-SWR panels.",
            flush=True,
        )
        return
    print("[visualize_stage3] active future-SWR cache:", anchor_cache_dir, flush=True)

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    channels = [str(value) for value in info["channels"]]
    anchor = np.load(anchor_path, mmap_mode="r")
    h8_features = np.load(feature_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    meta["issue_sec"] = meta["issue_time"].map(timestamp_key)
    by_sec = {int(row["issue_sec"]): row for _, row in meta.iterrows()}

    sampled = pd.read_csv(Path(args.out_dir) / "sampled_issue_times.csv")
    sampled["issue_time"] = pd.to_datetime(sampled["issue_time"])
    split_df = module._load_split_df(args)
    bounds = module._bounds(split_df)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    figure_dir = Path(args.out_dir) / "sample_feature_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    for sample_id, sample_row in sampled.iterrows():
        issue = pd.Timestamp(sample_row["issue_time"])
        row = by_sec.get(timestamp_key(issue))
        if row is None:
            print(f"[visualize_stage3] no SWR-anchor cache row for issue={issue}; skip optional panels")
            continue
        cache_idx = int(row["cache_idx"])
        anchor_item = np.asarray(anchor[cache_idx], dtype=np.float32)
        ratio = anchor_item[:, 0]
        clear_wm2 = anchor_item[:, 1] * 1000.0
        swr_wm2 = ratio * clear_wm2
        labels = split_times(row["history_label_times"]) + split_times(row["future_anchor_label_times"])
        labels = [
            f"{'observed' if index < 4 else 'predicted'}\n{timestamp:%m-%d %H:%M}"
            for index, timestamp in enumerate(labels)
        ]

        fig, axes = plt.subplots(3, 7, figsize=(25, 10), squeeze=False)
        rows = [
            ("SWR ratio", ratio, "viridis", 0.0, max(1.2, robust_limits(ratio)[1])),
            ("clear sky SWR W/m2", clear_wm2, "inferno", 0.0, max(100.0, robust_limits(clear_wm2)[1])),
            ("restored SWR W/m2", swr_wm2, "turbo", 0.0, max(100.0, robust_limits(swr_wm2)[1])),
        ]
        for row_index, (row_name, values, cmap, lo, hi) in enumerate(rows):
            for col_index in range(7):
                ax = axes[row_index, col_index]
                image = ax.imshow(
                    values[col_index],
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=lo,
                    vmax=hi,
                    interpolation="nearest",
                    aspect="auto",
                )
                ax.set_title(f"{labels[col_index]}\n{row_name}", fontsize=8)
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(
            f"Stage 3 SWR anchors | sample={sample_id:03d} | PV issue T={issue:%Y-%m-%d %H:%M:%S}",
            y=0.998,
        )
        fig.tight_layout()
        anchor_figure = figure_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_swr_anchors.png"
        fig.savefig(anchor_figure, dpi=int(args.dpi))
        plt.close(fig)

        features = np.asarray(h8_features[cache_idx], dtype=np.float32)
        fig, axes = plt.subplots(len(channels), features.shape[0], figsize=(15, 3.0 * len(channels)), squeeze=False)
        history_labels = split_times(row["history_label_times"])
        for channel_index, channel in enumerate(channels):
            values = features[:, channel_index]
            lo, hi = robust_limits(values)
            if channel in {"SWR", "SWR_ratio"}:
                lo, hi = 0.0, max(1.2, hi)
            for time_index in range(features.shape[0]):
                ax = axes[channel_index, time_index]
                image = ax.imshow(
                    values[time_index],
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=lo,
                    vmax=hi,
                    interpolation="nearest",
                    aspect="auto",
                )
                ax.set_title(f"{channel}\n{history_labels[time_index]:%m-%d %H:%M}", fontsize=8)
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(
            f"Stage 3 cropped H8 model-input features | sample={sample_id:03d} | issue={issue:%Y-%m-%d %H:%M:%S}",
            y=0.998,
        )
        fig.tight_layout()
        feature_figure = figure_dir / f"sample_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}_h8_input_features.png"
        fig.savefig(feature_figure, dpi=int(args.dpi))
        plt.close(fig)
        output_rows.append(
            {
                "sample_id": int(sample_id),
                "issue_time": str(issue),
                "cache_idx": cache_idx,
                "swr_anchor_figure": str(anchor_figure),
                "h8_input_feature_figure": str(feature_figure),
            }
        )

    pd.DataFrame(output_rows).to_csv(Path(args.out_dir) / "optional_swr_anchor_figure_index.csv", index=False)
    print(f"[visualize_stage3] optional SWR-anchor figures: {len(output_rows)}", flush=True)


def main() -> None:
    ensure_stage3_dirs()
    module.PYCHARM_DEFAULTS.update(
        {
            "run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(MULTISOURCE_VIS_DIR),
            "sample_split": "test",
            "sample_count": 5,
            "nwp_horizons": "1,4,8",
        }
    )
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()
    render_anchor_package(module, module.parse_args())


if __name__ == "__main__":
    main()
