#!/usr/bin/env python3
"""Visualize active Stage 3 cached future-SWR predictions against future truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_paths import (
    LEGACY_CENTER_SWR_CACHE_DIR,
    SWR_PREDICTION_VIS_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    ensure_stage3_dirs,
)
from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS
from stage3_swr_anchor_cache import resolve_active_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cached Stage 3 SWR forecasts.")
    parser.add_argument("--anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=SWR_PREDICTION_VIS_DIR)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--swr_vmax", type=float, default=1000.0)
    parser.add_argument("--ratio_vmax", type=float, default=1.6)
    parser.add_argument("--error_vmax", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def station_bounds() -> tuple[float, float, float, float]:
    candidates = [
        TARGET_EXCLUDED_RUN_DIR / "station_split.csv",
        TARGET_SELECTION_DIR / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            table = pd.read_csv(path)
            lon = pd.to_numeric(table["lon"], errors="coerce")
            lat = pd.to_numeric(table["lat"], errors="coerce")
            return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    raise FileNotFoundError("Station split not found. Run Stage 3 static cache construction first.")


def resample_to_grid(
    array: np.ndarray,
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    lon = np.asarray(source_lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(source_lat, dtype=np.float64).reshape(-1)
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        values = values[..., ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[..., ::-1, :]
    x = np.interp(np.asarray(target_lon, dtype=np.float64), lon, np.arange(len(lon), dtype=np.float64))
    y = np.interp(np.asarray(target_lat, dtype=np.float64), lat, np.arange(len(lat), dtype=np.float64))
    x0 = np.clip(np.floor(x).astype(np.int64), 0, len(lon) - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, len(lat) - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    y0_grid, y1_grid = y0[:, None], y1[:, None]
    x0_grid, x1_grid = x0[None, :], x1[None, :]
    wx_grid, wy_grid = wx[None, :], wy[:, None]
    v00 = values[..., y0_grid, x0_grid]
    v01 = values[..., y0_grid, x1_grid]
    v10 = values[..., y1_grid, x0_grid]
    v11 = values[..., y1_grid, x1_grid]
    return ((1.0 - wy_grid) * ((1.0 - wx_grid) * v00 + wx_grid * v01) + wy_grid * ((1.0 - wx_grid) * v10 + wx_grid * v11)).astype(np.float32)


def load_truth_adapter() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[pd.Timestamp, int]]:
    swr_path = LEGACY_CENTER_SWR_CACHE_DIR / "swr_3d.npy"
    lat_path = LEGACY_CENTER_SWR_CACHE_DIR / "lat_values.npy"
    lon_path = LEGACY_CENTER_SWR_CACHE_DIR / "lon_values.npy"
    time_path = LEGACY_CENTER_SWR_CACHE_DIR / "timestamps.csv"
    for path in (swr_path, lat_path, lon_path, time_path):
        if not path.exists():
            raise FileNotFoundError(f"Stage 1 center SWR adapter is incomplete: {path}")
    times = pd.read_csv(time_path)
    times["timestamp"] = pd.to_datetime(times["timestamp"])
    by_time = {pd.Timestamp(row["timestamp"]): int(row["index"]) for _, row in times.iterrows()}
    return (
        np.load(swr_path, mmap_mode="r"),
        np.load(lat_path),
        np.load(lon_path),
        by_time,
    )


def choose_rows(meta: pd.DataFrame, split: str, count: int, seed: int) -> pd.DataFrame:
    candidates = meta if split == "all" else meta[meta["time_split"].astype(str) == split]
    if candidates.empty:
        raise RuntimeError(f"No cached future-SWR rows for split={split!r}")
    count = min(int(count), len(candidates))
    return candidates.sample(n=count, random_state=int(seed)).sort_values("issue_time").reset_index(drop=True)


def future_truth(
    times: list[pd.Timestamp],
    truth_array: np.ndarray,
    by_time: dict[pd.Timestamp, int],
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    output = []
    for timestamp in times:
        index = by_time.get(pd.Timestamp(timestamp))
        if index is None:
            output.append(np.full((len(target_lat), len(target_lon)), np.nan, dtype=np.float32))
        else:
            output.append(
                resample_to_grid(
                    np.asarray(truth_array[index], dtype=np.float32),
                    source_lon,
                    source_lat,
                    target_lon,
                    target_lat,
                )
            )
    return np.stack(output, axis=0)


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return {"valid_pixels": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    error = np.asarray(prediction[valid] - truth[valid], dtype=np.float64)
    return {
        "valid_pixels": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def weighted_15min_anchor_context(anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match Stage 3 motion-fusion context: interpolate [T-1h, T, T+1h, T+2h]."""
    anchor = np.asarray(anchor, dtype=np.float32)
    if anchor.shape[0] != 7 or anchor.shape[1] != 2:
        raise ValueError(f"Expected seven two-channel SWR anchors, found {anchor.shape}")
    temporal_anchor = anchor[3:7]
    weighted = np.einsum(
        "ha,acxy->hcxy",
        STAGE3_FUTURE_ANCHOR_WEIGHTS,
        temporal_anchor,
        optimize=True,
    ).astype(np.float32)
    ratio = weighted[:, 0]
    clear_swr = weighted[:, 1] * 1000.0
    restored_swr = ratio * clear_swr
    return ratio, clear_swr, restored_swr


def pv_horizon_times(issue: pd.Timestamp) -> list[pd.Timestamp]:
    return [pd.Timestamp(issue) + pd.Timedelta(minutes=15 * horizon) for horizon in range(8)]


def save_weighted_15min_context(
    path: Path,
    issue: pd.Timestamp,
    ratio: np.ndarray,
    clear_swr: np.ndarray,
    restored_swr: np.ndarray,
    extent: list[float],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = pv_horizon_times(issue)
    fig, axes = plt.subplots(3, 8, figsize=(27, 10), squeeze=False)
    rows = [
        ("weighted SWR context W/m2", restored_swr, "turbo", 0.0, float(args.swr_vmax)),
        ("weighted SWR ratio", ratio, "viridis", 0.0, float(args.ratio_vmax)),
        ("weighted clear sky W/m2", clear_swr, "inferno", 0.0, float(args.swr_vmax)),
    ]
    for row_index, (label, values, cmap, vmin, vmax) in enumerate(rows):
        for horizon in range(8):
            ax = axes[row_index, horizon]
            image = ax.imshow(
                values[horizon],
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"h{horizon + 1} {times[horizon]:%H:%M}\n{label}", fontsize=8)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Stage 3 fixed-weight 15-minute SWR context "
        f"| PV issue T={issue:%Y-%m-%d %H:%M:%S}",
        y=0.998,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def save_weighted_15min_frame_pages(
    out_dir: Path,
    issue: pd.Timestamp,
    ratio: np.ndarray,
    clear_swr: np.ndarray,
    restored_swr: np.ndarray,
    extent: list[float],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    times = pv_horizon_times(issue)
    source_labels = ["A0 obs T-1h", "A1 pred T", "A2 pred T+1h", "A3 pred T+2h"]
    for horizon in range(8):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
        panels = [
            ("weighted SWR context W/m2", restored_swr[horizon], "turbo", 0.0, float(args.swr_vmax)),
            ("weighted SWR ratio", ratio[horizon], "viridis", 0.0, float(args.ratio_vmax)),
            ("weighted clear sky W/m2", clear_swr[horizon], "inferno", 0.0, float(args.swr_vmax)),
        ]
        for column, (label, values, cmap, vmin, vmax) in enumerate(panels):
            image = axes[0, column].imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            axes[0, column].set_title(label, fontsize=9)
            axes[0, column].set_xlabel("lon")
            axes[0, column].set_ylabel("lat")
            fig.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.02)
        weights = STAGE3_FUTURE_ANCHOR_WEIGHTS[horizon]
        axes[0, 3].bar(np.arange(4), weights, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        axes[0, 3].set_xticks(np.arange(4), source_labels, rotation=25, ha="right", fontsize=8)
        axes[0, 3].set_ylim(0.0, 1.05)
        axes[0, 3].set_ylabel("fixed interpolation weight")
        axes[0, 3].set_title("Stage 3 fixed anchor weights", fontsize=9)
        axes[0, 3].grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"issue={issue:%Y-%m-%d %H:%M} | PV horizon h{horizon + 1}={times[horizon]:%H:%M}"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"horizon_{horizon + 1:02d}_{times[horizon]:%Y%m%d_%H%M}.png",
            dpi=int(args.dpi),
            bbox_inches="tight",
        )
        plt.close(fig)


def save_overview(
    path: Path,
    issue: pd.Timestamp,
    times: list[pd.Timestamp],
    ratios: np.ndarray,
    restored_swr: np.ndarray,
    extent: list[float],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 7, figsize=(24, 7), squeeze=False)
    for column in range(7):
        phase = "observed" if column < 4 else "predicted"
        image = axes[0, column].imshow(
            restored_swr[column],
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=0.0,
            vmax=float(args.swr_vmax),
            aspect="auto",
        )
        axes[0, column].set_title(f"{phase} SWR\n{times[column]:%m-%d %H:%M}", fontsize=9)
        fig.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.02)
        image = axes[1, column].imshow(
            ratios[column],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=float(args.ratio_vmax),
            aspect="auto",
        )
        axes[1, column].set_title(f"{phase} ratio\n{times[column]:%m-%d %H:%M}", fontsize=9)
        fig.colorbar(image, ax=axes[1, column], fraction=0.046, pad=0.02)
        for row in range(2):
            axes[row, column].set_xlabel("lon")
            axes[row, column].set_ylabel("lat")
    fig.suptitle(f"Cached Stage 3 SWR motion sequence | PV issue T={issue:%Y-%m-%d %H:%M:%S}")
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def save_future_comparison(
    path: Path,
    issue: pd.Timestamp,
    times: list[pd.Timestamp],
    truth: np.ndarray,
    prediction: np.ndarray,
    ratios: np.ndarray,
    extent: list[float],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(15, 11), squeeze=False)
    for horizon in range(3):
        error = prediction[horizon] - truth[horizon]
        panels = [
            ("truth SWR", truth[horizon], "turbo", 0.0, float(args.swr_vmax)),
            ("cached prediction", prediction[horizon], "turbo", 0.0, float(args.swr_vmax)),
            ("prediction - truth", error, "coolwarm", -float(args.error_vmax), float(args.error_vmax)),
            ("cached ratio", ratios[horizon], "viridis", 0.0, float(args.ratio_vmax)),
        ]
        for column, (label, values, cmap, vmin, vmax) in enumerate(panels):
            image = axes[horizon, column].imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            axes[horizon, column].set_title(f"+{horizon + 1}h {times[horizon]:%m-%d %H:%M}\n{label}", fontsize=9)
            axes[horizon, column].set_xlabel("lon")
            axes[horizon, column].set_ylabel("lat")
            fig.colorbar(image, ax=axes[horizon, column], fraction=0.046, pad=0.02)
    fig.suptitle(f"Cached future SWR validation | PV issue T={issue:%Y-%m-%d %H:%M:%S}")
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def save_frame_pages(
    out_dir: Path,
    issue: pd.Timestamp,
    times: list[pd.Timestamp],
    truth: np.ndarray,
    prediction: np.ndarray,
    extent: list[float],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for horizon in range(3):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        error = prediction[horizon] - truth[horizon]
        panels = [
            ("truth SWR", truth[horizon], "turbo", 0.0, float(args.swr_vmax)),
            ("cached prediction", prediction[horizon], "turbo", 0.0, float(args.swr_vmax)),
            ("prediction - truth", error, "coolwarm", -float(args.error_vmax), float(args.error_vmax)),
        ]
        for column, (label, values, cmap, vmin, vmax) in enumerate(panels):
            image = axes[0, column].imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            axes[0, column].set_title(label)
            axes[0, column].set_xlabel("lon")
            axes[0, column].set_ylabel("lat")
            fig.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.02)
        fig.suptitle(f"issue={issue:%Y-%m-%d %H:%M} | forecast +{horizon + 1}h={times[horizon]:%H:%M}")
        fig.tight_layout()
        fig.savefig(out_dir / f"future_{horizon + 1:02d}_{times[horizon]:%Y%m%d_%H%M}.png", dpi=int(args.dpi))
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.anchor_cache_dir or resolve_active_cache_dir(required=True)
    anchor_path = cache_dir / "swr_anchor_64.npy"
    meta_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    info_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.json"
    for path in (anchor_path, meta_path, info_path):
        if not path.exists():
            raise FileNotFoundError(f"Active future-SWR cache is incomplete: {path}")

    info = read_json(info_path)
    anchors = np.load(anchor_path, mmap_mode="r")
    if anchors.ndim != 5 or anchors.shape[1:3] != (7, 2):
        raise ValueError(
            "Unexpected Stage 3 SWR-anchor array shape. Expected [N, 7, 2, H, W], "
            f"found {anchors.shape}."
        )
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    selected = choose_rows(meta, args.split, args.sample_count, args.seed)
    truth_array, source_lat, source_lon, by_time = load_truth_adapter()
    lon_min, lon_max, lat_min, lat_max = station_bounds()
    target_lon = np.linspace(lon_min, lon_max, 64, dtype=np.float32)
    target_lat = np.linspace(lat_min, lat_max, 64, dtype=np.float32)
    extent = [lon_min, lon_max, lat_min, lat_max]

    rows = []
    all_prediction = []
    all_truth = []
    figure_dir = args.out_dir / "samples"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for sample_index, row in selected.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        cache_index = int(row["cache_idx"])
        anchor = np.asarray(anchors[cache_index], dtype=np.float32)
        ratios = anchor[:, 0]
        clear_swr = anchor[:, 1] * 1000.0
        restored_swr = ratios * clear_swr
        history_times = split_times(row["history_label_times"])
        future_times = split_times(row["future_anchor_label_times"])
        times = history_times + future_times
        truth = future_truth(
            future_times,
            truth_array,
            by_time,
            source_lon,
            source_lat,
            target_lon,
            target_lat,
        )
        prediction = restored_swr[4:7]
        weighted_ratio, weighted_clear_swr, weighted_swr = weighted_15min_anchor_context(anchor)
        sample_metrics = metrics(prediction, truth)
        sample_dir = figure_dir / f"sample_{sample_index:03d}_{issue:%Y%m%d_%H%M%S}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        overview_path = sample_dir / "00_motion_sequence_4obs_3pred.png"
        comparison_path = sample_dir / "01_future_truth_prediction_error.png"
        weighted_context_path = sample_dir / "02_weighted_15min_anchor_context.png"
        save_overview(overview_path, issue, times, ratios, restored_swr, extent, args)
        save_future_comparison(comparison_path, issue, future_times, truth, prediction, ratios[4:7], extent, args)
        save_frame_pages(sample_dir / "frame_pages", issue, future_times, truth, prediction, extent, args)
        save_weighted_15min_context(
            weighted_context_path,
            issue,
            weighted_ratio,
            weighted_clear_swr,
            weighted_swr,
            extent,
            args,
        )
        save_weighted_15min_frame_pages(
            sample_dir / "weighted_15min_frame_pages",
            issue,
            weighted_ratio,
            weighted_clear_swr,
            weighted_swr,
            extent,
            args,
        )
        rows.append(
            {
                "sample_index": int(sample_index),
                "cache_idx": cache_index,
                "issue_time": str(issue),
                **sample_metrics,
                "motion_sequence_figure": str(overview_path),
                "future_comparison_figure": str(comparison_path),
                "weighted_15min_anchor_context_figure": str(weighted_context_path),
            }
        )
        all_prediction.append(prediction)
        all_truth.append(truth)
        print(
            f"[{sample_index + 1}/{len(selected)}] issue={issue} "
            f"MAE={sample_metrics['mae']:.2f} RMSE={sample_metrics['rmse']:.2f} "
            f"bias={sample_metrics['bias']:.2f}",
            flush=True,
        )

    overall = metrics(np.stack(all_prediction), np.stack(all_truth))
    if int(overall["valid_pixels"]) == 0:
        raise RuntimeError(
            "No finite future SWR truth pixels matched the sampled cache rows. "
            "Check the Stage 1 center SWR adapter timestamps before interpreting visualizations."
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "sample_metrics.csv", index=False)
    pd.DataFrame(
        STAGE3_FUTURE_ANCHOR_WEIGHTS,
        columns=["A0_observed_T_minus_1h", "A1_predicted_T", "A2_predicted_T_plus_1h", "A3_predicted_T_plus_2h"],
        index=[f"h{horizon + 1}_{15 * horizon:03d}min" for horizon in range(8)],
    ).to_csv(args.out_dir / "stage3_fixed_15min_anchor_weights.csv", index_label="pv_horizon")
    cache_version = info.get("stage3_swr_cache_version", {})
    payload = {
        "purpose": "Visual validation of the active Stage 3 cached future-SWR prediction package.",
        "active_anchor_cache_dir": str(cache_dir),
        "checkpoint": cache_version.get("checkpoint_path", info.get("swr_checkpoint", "")),
        "checkpoint_model_name": cache_version.get("model_name", ""),
        "loss_variant": cache_version.get("loss_variant", ""),
        "base_checkpoint": cache_version.get("base_checkpoint_path", ""),
        "base_temporal_head": cache_version.get("base_temporal_head", ""),
        "split": str(args.split),
        "sample_count": int(len(rows)),
        "sampled_metrics": overall,
        "outputs": {
            "sample_metrics": str(args.out_dir / "sample_metrics.csv"),
            "sample_figure_dir": str(figure_dir),
            "fixed_15min_anchor_weights": str(args.out_dir / "stage3_fixed_15min_anchor_weights.csv"),
        },
    }
    with (args.out_dir / "visualization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("==== Cached future-SWR visualization ====")
    print("active cache:", cache_dir)
    print("checkpoint:", payload["checkpoint"])
    print("model:", payload["checkpoint_model_name"])
    print("loss variant:", payload["loss_variant"])
    print("frozen base:", payload["base_checkpoint"])
    print("base temporal head:", payload["base_temporal_head"])
    print("sampled metrics:", overall)
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
