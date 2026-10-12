#!/usr/bin/env python3
"""Historical-PV-field patch ablation for Stage 4 station forecasting.

This is the fairness baseline for the 05l Stage3 future-field-patch model.
The station model, station/NWP/time inputs, patch tensor shape, optimizer,
loss, splits, and evaluation code are inherited from
05l_train_per_station_temporal_with_stage3_field_patch.py.

Only the field-patch source changes:

    Full 05l:
        future predicted PV field patch for h1-h8

    This ablation:
        latest observed target-excluded PV field patch at issue T-15min,
        repeated to h1-h8

Thus the model sees [future_frames, 2, patch, patch] in both cases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_base_module():
    path = SCRIPT_DIR / "05l_train_per_station_temporal_with_stage3_field_patch.py"
    spec = importlib.util.spec_from_file_location("stage4_05l_field_patch_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import 05l base module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_05l_field_patch_base"] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

from stage4_paths import PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR


class HistoricalPVFieldPatchProvider:
    """Use the latest observed target-excluded PV field patch, repeated over h1-h8."""

    def __init__(self, targets: pd.DataFrame, time_df: pd.DataFrame, args, device: str):
        self.targets = targets.copy().reset_index(drop=True)
        self.time_df = time_df.reset_index(drop=True).copy()
        self.args = args
        self.device = device
        self.patch_size = int(args.field_patch_size)
        if self.patch_size % 2 != 1:
            raise ValueError("--field_patch_size must be odd, e.g. 5")
        self.issue_to_index = {int(v): i for i, v in enumerate(self.time_df["issue_sec"].astype(np.int64).tolist())}
        self.patch_raw: dict[str, np.ndarray] = {}
        self.patch_mask: dict[str, np.ndarray] = {}
        self.patch_mean = 0.0
        self.patch_std = 1.0
        self.bounds: tuple[float, float, float, float] | None = None
        self._build()

    def _field_args(self):
        return SimpleNamespace(
            selection_dir=Path(self.args.selection_dir),
            target_run_dir=Path(self.args.target_run_dir),
            frame_cache_dir=Path(self.args.frame_cache_dir),
            multisource_cache_dir=Path(self.args.multisource_cache_dir),
            grid_size=int(self.args.grid_size),
            gaussian_sigma_cells=float(self.args.gaussian_sigma_cells),
            gaussian_radius_cells=int(self.args.gaussian_radius_cells),
            min_grid_weight=float(self.args.min_grid_weight),
            freq_minutes=15,
            past_frames=int(self.args.past_frames),
            future_frames=int(self.args.future_frames),
        )

    def _target_cells(self, bounds: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
        cells = {}
        for _, row in self.targets.iterrows():
            sn = str(row["station_sn"])
            cells[sn] = base._xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, int(self.args.grid_size))
        return cells

    def _build(self) -> None:
        field_args = self._field_args()
        split_df = base.f04._load_station_split(field_args)
        bounds = base._bounds(split_df)
        self.bounds = bounds
        cells = self._target_cells(bounds)
        pv_cache = base.f04.ReadOnlyDiskFrameCache(Path(self.args.frame_cache_dir), self.time_df, field_args)

        n = len(self.time_df)
        future_frames = int(self.args.future_frames)
        for sn in self.targets["station_sn"].astype(str):
            self.patch_raw[sn] = np.zeros((n, future_frames, self.patch_size, self.patch_size), dtype=np.float32)
            self.patch_mask[sn] = np.zeros((n, future_frames, self.patch_size, self.patch_size), dtype=np.float32)

        for row_index, row in self.time_df.iterrows():
            past_times = base.b03._parse_time_list(row["past_times"])
            if not past_times:
                raise ValueError(f"empty past_times for issue_time={row['issue_time']}")
            latest_time = pd.Timestamp(past_times[-1])
            field, mask = pv_cache.get(latest_time)
            field = np.asarray(field, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            for sn, (cy, cx) in cells.items():
                patch = base._extract_patch(field, cy, cx, self.patch_size)
                mask_patch = base._extract_patch(mask, cy, cx, self.patch_size)
                for h in range(future_frames):
                    self.patch_raw[sn][row_index, h] = patch
                    self.patch_mask[sn][row_index, h] = mask_patch
            if (row_index + 1) % 500 == 0:
                print(
                    f"[historical_field_patch] cached latest-observed patches for {row_index + 1}/{n} issue times",
                    flush=True,
                )

        train_idx = self.time_df.index[self.time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
        chunks = [self.patch_raw[str(sn)][train_idx].reshape(-1) for sn in self.targets["station_sn"].astype(str)]
        vals = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        self.patch_mean = float(np.mean(vals)) if vals.size else 0.0
        self.patch_std = float(np.std(vals)) if vals.size else 1.0
        if not np.isfinite(self.patch_std) or self.patch_std < 1e-6:
            self.patch_std = 1.0
        print(
            f"[historical_field_patch] ready: stations={len(self.targets)} issues={len(self.time_df)} "
            f"patch={self.patch_size} mean={self.patch_mean:.5f} std={self.patch_std:.5f}",
            flush=True,
        )

    def get(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        sec = base._timestamp_key(pd.Timestamp(issue_time))
        idx = self.issue_to_index[int(sec)]
        raw = self.patch_raw[str(station_sn)][idx]
        mask = self.patch_mask[str(station_sn)][idx]
        z = (raw - self.patch_mean) / self.patch_std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return z, mask.astype(np.float32)

    def raw_center_series(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sec = base._timestamp_key(pd.Timestamp(issue_time))
        idx = self.issue_to_index[int(sec)]
        raw = self.patch_raw[str(station_sn)][idx]
        mask = self.patch_mask[str(station_sn)][idx]
        c = self.patch_size // 2
        return raw[:, c, c].astype(np.float32), raw.mean(axis=(1, 2)).astype(np.float32), mask.mean(axis=(1, 2)).astype(np.float32)

    def stats(self) -> dict:
        return {
            "field_patch_mean": float(self.patch_mean),
            "field_patch_std": float(self.patch_std),
            "field_patch_size": int(self.patch_size),
            "field_patch_source": "latest_observed_target_excluded_pv_field_patch_repeated_to_all_future_horizons",
            "latest_observed_time": "issue_time - 15min, i.e. final PV history frame",
            "channels": ["historical_pvnorm_field_z", "latest_support_mask"],
            "mask_mode": "latest_input_support_mask_repeated_for_all_future_horizons",
        }


def _patch_base_defaults() -> None:
    base.PYCHARM_DEFAULTS["out_dir"] = str(PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR)
    base.PYCHARM_DEFAULTS["sample_count"] = 20
    base.PredictedFieldPatchProvider = HistoricalPVFieldPatchProvider


def _patch_run_meta() -> None:
    args = base.parse_args()
    out_dir = Path(args.out_dir)
    for name in (
        "per_station_temporal_swr_anchor04l_field_patch_metrics.csv",
        "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv",
    ):
        csv_path = out_dir / name
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"station_sn": str})
            if "model" in df.columns:
                df["model"] = "per_station_temporal_historical_pv_field_patch"
                df.to_csv(csv_path, index=False)
    path = out_dir / "run_meta.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["purpose"] = (
        "Historical-PV-field patch ablation for Stage 4: same station temporal "
        "model as 05l, but the field patch is the latest observed target-excluded "
        "PV field patch repeated over all future horizons."
    )
    meta["baseline_alignment"] = (
        "Strictly aligned with 05l full model except replacing future predicted "
        "PV field patches with the final historical PV field patch."
    )
    meta["field_patch"] = {
        "source": "target-excluded observed PV_norm field cache",
        "variant": "historical_latest_observed_field_patch_repeated",
        "patch_size": int(args.field_patch_size),
        "shape": [int(args.future_frames), 2, int(args.field_patch_size), int(args.field_patch_size)],
        "channels": ["historical_pvnorm_field_z", "latest_support_mask"],
        "temporal_protocol": "h1-h8 all receive the same issue-time latest historical PV field patch at T-15min",
        "fairness_note": "Network input shape, parameter count, station/NWP/time inputs, optimizer, loss, and splits match 05l.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    _patch_base_defaults()
    base.main()
    _patch_run_meta()


if __name__ == "__main__":
    main()
