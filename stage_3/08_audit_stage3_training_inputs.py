#!/usr/bin/env python3
"""Audit the exact Stage 3 Dataset tensors before PV-field model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS
from stage3_paths import (
    MOTION_FUSION_RUN_DIR,
    MULTISOURCE_CACHE_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    SWR_BASE_CHECKPOINT,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    TRAINING_INPUT_AUDIT_DIR,
    ensure_stage3_dirs,
)
import stage3_motion_fusion_model_impl as module
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir
from stage3_temporal import assert_swr_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit actual Stage 3 PV-field training Dataset tensors.")
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=TRAINING_INPUT_AUDIT_DIR)
    parser.add_argument("--samples_per_split", type=int, default=4)
    parser.add_argument(
        "--min_split_coverage",
        type=float,
        default=0.95,
        help="Minimum retained SWR-anchor coverage required independently for train, val and test.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def update_check(checks: dict[str, bool], name: str, passed: bool) -> None:
    checks[name] = bool(checks.get(name, True) and bool(passed))


def expected_shapes(base_dataset) -> dict[str, tuple[int, ...]]:
    base_history_channels, base_future_channels = base_dataset.source_dims()
    return {
        "x_aug": (int(base_history_channels) + 2, 12, 64, 64),
        "future_aug": (8, int(base_future_channels) + 1, 64, 64),
        "swr_anchor": (7, 2, 64, 64),
        "y": (8, 64, 64),
        "mask": (8, 64, 64),
    }


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


class IndexedArray:
    """Expose selected cache rows without materializing a complete split."""

    def __init__(self, values: np.ndarray, indices: np.ndarray):
        self.values = values
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[int(self.indices[int(index)])]


def tensor_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    out: dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "finite_ratio": float(finite.mean()) if array.size else 0.0,
    }
    if finite.any():
        selected = np.asarray(array[finite], dtype=np.float64)
        out.update(
            {
                "min": float(selected.min()),
                "p01": float(np.quantile(selected, 0.01)),
                "mean": float(selected.mean()),
                "p99": float(np.quantile(selected, 0.99)),
                "max": float(selected.max()),
            }
        )
    return out


def anchor_stats(anchor_array: np.ndarray, cache_indices: np.ndarray, chunk_size: int = 128) -> dict:
    count = np.zeros(2, dtype=np.float64)
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    for start in range(0, len(cache_indices), int(chunk_size)):
        values = np.asarray(anchor_array[cache_indices[start : start + chunk_size]], dtype=np.float64)
        for channel in range(2):
            selected = values[:, :, channel]
            selected = selected[np.isfinite(selected)]
            count[channel] += selected.size
            total[channel] += selected.sum()
            total_sq[channel] += np.square(selected).sum()
    mean = total / np.maximum(count, 1.0)
    variance = np.maximum(total_sq / np.maximum(count, 1.0) - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-6)
    return {
        "anchor_channel_names": ["swr_ratio", "clear_sky_swr_kwm2"],
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "note": "Exact train-split anchor stats computed in chunks for Dataset audit.",
    }


def temporal_checks(issue_table: pd.DataFrame, swr_meta: pd.DataFrame) -> dict[str, bool]:
    pv_ok = True
    for _, row in issue_table.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        past = split_times(row["past_times"])
        future = split_times(row["future_times"])
        pv_ok = pv_ok and len(past) == 12 and len(future) == 8
        pv_ok = pv_ok and past[-1] == issue - pd.Timedelta(minutes=15)
        pv_ok = pv_ok and future[0] == issue and future[-1] == issue + pd.Timedelta(minutes=105)
    swr_ok = True
    for _, row in swr_meta.iterrows():
        try:
            assert_swr_alignment(
                pd.Timestamp(row["issue_time"]),
                split_times(row["history_label_times"]),
                split_times(row["future_anchor_label_times"]),
            )
        except AssertionError:
            swr_ok = False
            break
    return {
        "PV Dataset temporal protocol is T-3h..T-15min -> T..T+1h45min": bool(pv_ok),
        "SWR Dataset temporal protocol is T-4h..T-1h -> T..T+2h": bool(swr_ok),
    }


def weighted_swr_physical(swr_denorm: np.ndarray) -> np.ndarray:
    temporal = np.asarray(swr_denorm[3:7], dtype=np.float32)
    weighted = np.einsum("ha,acxy->hcxy", STAGE3_FUTURE_ANCHOR_WEIGHTS, temporal, optimize=True)
    return weighted[:, 0] * weighted[:, 1] * 1000.0


def plot_dataset_item(
    path: Path,
    split: str,
    index: int,
    issue: pd.Timestamp,
    item: tuple[np.ndarray, ...],
    dataset,
    pv_stats: dict,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_aug, _future_aug, swr_anchor, y, mask = item
    swr_denorm = dataset.denorm_swr(swr_anchor)
    weighted_swr = weighted_swr_physical(swr_denorm)
    pv_mean = float(pv_stats["pv_field_mean"])
    pv_std = float(pv_stats["pv_field_std"])
    history_pv = x_aug[0] * pv_std + pv_mean
    truth_pv = y * pv_std + pv_mean
    fig, axes = plt.subplots(4, 12, figsize=(30, 10), squeeze=False)
    for column in range(12):
        panels = [
            ("past PV field", history_pv[column], "viridis", 0.0, 1.5),
            ("past PV mask", x_aug[1, column], "gray", 0.0, 1.0),
        ]
        if column < 8:
            panels.extend(
                [
                    ("future PV truth", truth_pv[column], "viridis", 0.0, 1.5),
                    ("weighted 15min SWR", weighted_swr[column], "turbo", 0.0, 1000.0),
                ]
            )
        else:
            panels.extend([("", np.full_like(history_pv[0], np.nan), "viridis", 0.0, 1.0)] * 2)
        for row, (label, values, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label}\n{'t' + str(column + 1) if label else ''}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Actual Stage 3 Dataset item | split={split} index={index} issue={issue}")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_training_args(module, anchor_cache_dir: Path, out_dir: Path) -> SimpleNamespace:
    defaults = dict(module.PYCHARM_DEFAULTS)
    defaults.update(
        {
            "selection_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "out_dir": str(out_dir),
            "swr_anchor_cache_dir": str(anchor_cache_dir),
            "h8_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "h8_sample_cache_dir": "",
            "h8_clear_sky_dir": "",
            "swr_checkpoint": str(SWR_BASE_CHECKPOINT),
            "base_model_name": "pv_nwp",
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    return SimpleNamespace(**defaults)


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchor_cache_dir = args.swr_anchor_cache_dir or resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(anchor_cache_dir, SWR_BASE_CHECKPOINT)
    info_path = anchor_cache_dir / "center_h8_swr_anchor_input_cache_meta.json"
    anchor_info = json.loads(info_path.read_text(encoding="utf-8"))
    anchor_schema = str(anchor_info.get("stage3_swr_cache_version", {}).get("spatial_resample_schema", ""))

    train_args = build_training_args(module, anchor_cache_dir, args.out_dir / "train_only_stats")
    f04 = module.f04
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        f04._assert_target_exclusion(train_args)
        checks["selected PV targets are excluded from the actual PV-field cache provenance"] = True
    except Exception as exc:
        checks["selected PV targets are excluded from the actual PV-field cache provenance"] = False
        errors.append(f"target exclusion: {exc}")
    checks["active SWR cache uses corrected 2D bilinear mesh resampling"] = anchor_schema == "bilinear_mesh_v3"

    station_split = f04._load_station_split(train_args)
    bounds = module._bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(train_args.multisource_cache_dir))
    f04._assert_issue_time_alignment(train_args, ms_cache["time_df"])
    checks["multisource issue table exactly matches selection issue table"] = True
    issue_table = pd.read_csv(Path(train_args.selection_dir) / "issue_time_index.csv")
    issue_table["issue_time"] = pd.to_datetime(issue_table["issue_time"])
    pv_cache = f04.ReadOnlyDiskFrameCache(Path(train_args.frame_cache_dir), issue_table, train_args)
    stats_dir = Path(train_args.out_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, issue_table, stats_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, stats_dir, issue_table)
    anchor_array, swr_meta, _ = module._load_prebuilt_swr_anchor_cache(anchor_cache_dir)
    checks.update(temporal_checks(issue_table, swr_meta))
    checks["PV issue times are unique"] = not issue_table["issue_time"].duplicated().any()
    checks["SWR anchor issue times are unique"] = not swr_meta["issue_time"].duplicated().any()

    split_payload: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    datasets = {}
    train_indices = None
    for split in ["train", "val", "test"]:
        split_df, cache_indices, _ = module._subset_time_df_by_swr_meta(issue_table, swr_meta, split)
        source_rows = int((issue_table["time_split"].astype(str) == split).sum())
        retained_rows = int(len(split_df))
        coverage = retained_rows / max(source_rows, 1)
        split_payload[split] = {
            "source_issue_rows": source_rows,
            "dataset_rows": retained_rows,
            "missing_swr_anchor_rows": source_rows - retained_rows,
            "coverage": float(coverage),
        }
        checks[f"{split} SWR-anchor coverage >= {float(args.min_split_coverage):.2%}"] = (
            coverage >= float(args.min_split_coverage)
        )
        if split == "train":
            train_indices = cache_indices
        datasets[split] = (split_df, cache_indices)
    if train_indices is None:
        raise RuntimeError("No train cache indices found.")
    swr_stats = anchor_stats(anchor_array, train_indices)

    rng = np.random.default_rng(int(args.seed))
    lat_gradients, lon_gradients = [], []
    sample_dir = args.out_dir / "sample_figures"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for split, (split_df, cache_indices) in datasets.items():
        base = f04.TargetExcludedPVNormFieldDataset(
            split_df,
            pv_cache,
            ms_cache,
            bounds,
            pv_stats,
            ms_stats,
            str(train_args.base_model_name),
            train_args,
        )
        dataset = module.SWRAnchorPVFieldDataset(
            base,
            IndexedArray(anchor_array, cache_indices),
            train_args,
            swr_stats,
        )
        contract = expected_shapes(base)
        selected_indices = sorted(
            rng.choice(len(dataset), size=min(int(args.samples_per_split), len(dataset)), replace=False).tolist()
        )
        split_payload[split]["sampled_indices"] = selected_indices
        split_payload[split]["expected_tensor_shapes"] = {
            name: list(shape) for name, shape in contract.items()
        }
        for index in selected_indices:
            item = dataset[int(index)]
            x_aug, future_aug, swr_anchor, y, mask = item
            actual = {
                "x_aug": x_aug,
                "future_aug": future_aug,
                "swr_anchor": swr_anchor,
                "y": y,
                "mask": mask,
            }
            swr_denorm = dataset.denorm_swr(swr_anchor)
            swr_phy = swr_denorm[:, 0] * swr_denorm[:, 1] * 1000.0
            lat_gradients.append(float(np.nanmean(np.abs(np.diff(swr_phy, axis=-2)))))
            lon_gradients.append(float(np.nanmean(np.abs(np.diff(swr_phy, axis=-1)))))
            issue = pd.Timestamp(split_df.iloc[int(index)]["issue_time"])
            item_stats = {name: tensor_stats(value) for name, value in actual.items()}
            sample_rows.append(
                {
                    "split": split,
                    "index": int(index),
                    "issue_time": str(issue),
                    "stats": item_stats,
                }
            )
            for name, values in actual.items():
                update_check(
                    checks,
                    f"{split} sampled {name} shapes match training contract",
                    tuple(values.shape) == contract[name],
                )
                update_check(
                    checks,
                    f"{split} sampled {name} contains only finite values",
                    bool(np.isfinite(values).all()),
                )
            update_check(checks, f"{split} sampled mask is binary", bool(np.isin(mask, [0.0, 1.0]).all()))
            update_check(checks, f"{split} sampled mask contains valid PV-field pixels", bool(np.any(mask > 0.5)))
            plot_dataset_item(
                sample_dir / f"{split}_idx_{index:05d}_{issue:%Y%m%d_%H%M%S}.png",
                split,
                int(index),
                issue,
                item,
                dataset,
                pv_stats,
                args.dpi,
            )

    checks["sampled SWR fields vary along latitude axis"] = bool(np.nanmean(lat_gradients) > 1.0e-4)
    checks["sampled SWR fields vary along longitude axis"] = bool(np.nanmean(lon_gradients) > 1.0e-4)
    checks["all train/val/test datasets contain rows"] = all(
        int(split_payload[split]["dataset_rows"]) > 0 for split in ["train", "val", "test"]
    )
    status = "PASS" if all(checks.values()) and not errors else "FAIL"
    report = {
        "purpose": "Audit the exact Stage 3 Dataset tensors consumed by PV-field training.",
        "status": status,
        "checks": checks,
        "errors": errors,
        "active_swr_anchor_cache": str(anchor_cache_dir),
        "swr_anchor_cache_spatial_schema": anchor_schema,
        "dataset_splits": split_payload,
        "train_only_normalization": {
            "pv_field_stats": pv_stats,
            "multisource_stats": ms_stats,
            "swr_anchor_stats": swr_stats,
        },
        "sampled_swr_spatial_gradient": {
            "latitude_mean_abs_gradient": float(np.nanmean(lat_gradients)),
            "longitude_mean_abs_gradient": float(np.nanmean(lon_gradients)),
        },
        "sampled_items": sample_rows,
    }
    report_path = args.out_dir / "stage3_training_input_audit.json"
    write_json(report_path, report)
    print("==== Stage 3 actual training-input audit ====")
    print("status:", status)
    print("active SWR anchor cache:", anchor_cache_dir)
    print(
        "dataset coverage:",
        {
            split: (
                f"{payload['dataset_rows']}/{payload['source_issue_rows']} "
                f"({payload['coverage']:.2%}), missing={payload['missing_swr_anchor_rows']}"
            )
            for split, payload in split_payload.items()
        },
    )
    print("sampled SWR spatial gradients:", report["sampled_swr_spatial_gradient"])
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if errors:
        print("errors:")
        for error in errors:
            print(" -", error)
    print("report:", report_path)
    print("sample figures:", sample_dir)
    if status != "PASS":
        raise RuntimeError("Stage 3 training-input audit failed. Inspect the saved JSON and sample figures.")


if __name__ == "__main__":
    main()
