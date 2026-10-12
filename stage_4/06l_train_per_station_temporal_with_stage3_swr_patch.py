#!/usr/bin/env python3
"""Stage 4 ablation: station temporal model guided directly by Stage 3 SWR patches.

This keeps the station-level architecture from 05l, but replaces the predicted
PV-field patch with a 5x5 future SWR patch. The future SWR patch is built by
applying the exact Stage 3 fixed temporal-anchor interpolation weights to the
active Stage 3 SWR-anchor cache.
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
STAGE3_DIR = SCRIPT_DIR.parent / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_swr_anchor_cache import resolve_active_cache_dir  # noqa: E402
from stage4_paths import (  # noqa: E402
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
    PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
    ensure_stage4_dirs,
)


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m05 = _load_local_module(
    "stage4_05l_per_station_stage3_field_patch_for_swr_patch",
    SCRIPT_DIR / "05l_train_per_station_temporal_with_stage3_field_patch.py",
)


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _stage3_fixed_anchor_weights(future_frames: int = 8, anchor_count: int = 4) -> np.ndarray:
    """Match Stage 3's current fixed temporal-anchor weights exactly."""
    weights = np.zeros((int(future_frames), int(anchor_count)), dtype=np.float32)
    if int(future_frames) == 8 and int(anchor_count) >= 4:
        weights[:, :4] = np.asarray(
            [
                [0.25, 0.75, 0.00, 0.00],
                [0.00, 1.00, 0.00, 0.00],
                [0.00, 0.75, 0.25, 0.00],
                [0.00, 0.50, 0.50, 0.00],
                [0.00, 0.25, 0.75, 0.00],
                [0.00, 0.00, 1.00, 0.00],
                [0.00, 0.00, 0.75, 0.25],
                [0.00, 0.00, 0.50, 0.50],
            ],
            dtype=np.float32,
        )
        return weights
    positions = np.arange(int(anchor_count), dtype=np.float32)
    for h in range(int(future_frames)):
        pos = h / max(int(future_frames) - 1, 1) * max(int(anchor_count) - 1, 0)
        lo = int(np.floor(pos))
        hi = min(lo + 1, int(anchor_count) - 1)
        if lo == hi:
            weights[h, lo] = 1.0
        else:
            weights[h, lo] = float(hi - pos)
            weights[h, hi] = float(pos - lo)
    return weights


class Stage3SWRPatchProvider:
    """Provide 5x5 future SWR patches using Stage 3's SWR-anchor interpolation."""

    def __init__(self, targets: pd.DataFrame, time_df: pd.DataFrame, args, device: str):
        self.targets = targets.copy().reset_index(drop=True)
        self.time_df = time_df.reset_index(drop=True).copy()
        self.args = args
        self.device = device
        self.patch_size = int(args.field_patch_size)
        if self.patch_size % 2 != 1:
            raise ValueError("--field_patch_size must be odd, e.g. 5")
        self.issue_to_index = {
            int(value): idx for idx, value in enumerate(self.time_df["issue_sec"].astype(np.int64).tolist())
        }
        self.patch_raw: dict[str, np.ndarray] = {}
        self.patch_mask: dict[str, np.ndarray] = {}
        self.patch_mean = 0.0
        self.patch_std = 1.0
        self.bounds: tuple[float, float, float, float] | None = None
        self.cache_dir = Path(args.swr_anchor_cache_dir)
        self._build()

    def _field_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            selection_dir=Path(self.args.selection_dir),
            target_run_dir=Path(self.args.target_run_dir),
            grid_size=int(self.args.grid_size),
        )

    def _swr_anchor_array_for_time_df(self, swr_arr_all: np.ndarray, swr_meta: pd.DataFrame) -> np.ndarray:
        meta = swr_meta.copy()
        meta["issue_time"] = pd.to_datetime(meta["issue_time"])
        if "issue_sec" not in meta.columns:
            meta["issue_sec"] = meta["issue_time"].map(_timestamp_key)
        if "cache_idx" not in meta.columns:
            meta["cache_idx"] = np.arange(len(meta), dtype=np.int64)
        by_sec = {int(row["issue_sec"]): int(row["cache_idx"]) for _, row in meta.iterrows()}
        indices, missing = [], []
        for sec in self.time_df["issue_sec"].astype(np.int64).tolist():
            idx = by_sec.get(int(sec))
            if idx is None:
                missing.append(int(sec))
            else:
                indices.append(idx)
        if missing:
            raise KeyError(
                f"{len(missing)} issue times are missing from Stage 3 SWR-anchor cache; "
                f"first missing issue_sec={missing[:5]}"
            )
        return np.asarray(swr_arr_all[np.asarray(indices, dtype=np.int64)], dtype=np.float32)

    def _target_cells(self, bounds: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
        cells = {}
        for _, row in self.targets.iterrows():
            cells[str(row["station_sn"])] = m05._xy_to_ij(
                float(row["lon"]), float(row["lat"]), bounds, int(self.args.grid_size)
            )
        return cells

    @staticmethod
    def _future_swr_15min(anchor: np.ndarray, future_frames: int) -> np.ndarray:
        # anchor shape: [7, 2, H, W]. Channel 0 is SWR ratio, channel 1 is clear-sky SWR in kW/m2.
        temporal = np.asarray(anchor[3:7], dtype=np.float32)
        if temporal.shape[0] != 4:
            raise ValueError(f"Expected 4 Stage 3 temporal SWR anchors [x3,y0,y1,y2], got {temporal.shape}")
        swr_physical = temporal[:, 0] * temporal[:, 1]
        weights = _stage3_fixed_anchor_weights(int(future_frames), 4)
        return np.einsum("fa,ahw->fhw", weights, swr_physical).astype(np.float32)

    def _build(self) -> None:
        field_args = self._field_args()
        split_df = m05.f04._load_station_split(field_args)
        bounds = m05._bounds(split_df)
        self.bounds = bounds
        cells = self._target_cells(bounds)
        swr_arr_all, swr_meta, _swr_info = m05._load_swr04l()._load_prebuilt_swr_anchor_cache(self.cache_dir)
        swr_arr = self._swr_anchor_array_for_time_df(swr_arr_all, swr_meta)
        n = len(self.time_df)
        for sn in self.targets["station_sn"].astype(str):
            self.patch_raw[sn] = np.zeros((n, int(self.args.future_frames), self.patch_size, self.patch_size), dtype=np.float32)
            self.patch_mask[sn] = np.ones((n, int(self.args.future_frames), self.patch_size, self.patch_size), dtype=np.float32)
        for idx in range(n):
            future_swr = self._future_swr_15min(swr_arr[idx], int(self.args.future_frames))
            for sn, (cy, cx) in cells.items():
                for h in range(int(self.args.future_frames)):
                    self.patch_raw[sn][idx, h] = m05._extract_patch(future_swr[h], cy, cx, self.patch_size)
            if (idx + 1) % 500 == 0 or idx + 1 == n:
                print(f"[stage4_swr_patch] built SWR patches for {idx + 1}/{n} issue times", flush=True)
        train_idx = self.time_df.index[self.time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
        chunks = [self.patch_raw[str(sn)][train_idx].reshape(-1) for sn in self.targets["station_sn"].astype(str)]
        vals = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        self.patch_mean = float(np.mean(vals)) if vals.size else 0.0
        self.patch_std = float(np.std(vals)) if vals.size else 1.0
        if not np.isfinite(self.patch_std) or self.patch_std < 1e-6:
            self.patch_std = 1.0
        print(
            f"[stage4_swr_patch] ready: stations={len(self.targets)} issues={n} "
            f"patch={self.patch_size} mean={self.patch_mean:.5f} std={self.patch_std:.5f}",
            flush=True,
        )

    def get(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        sec = _timestamp_key(pd.Timestamp(issue_time))
        idx = self.issue_to_index[int(sec)]
        raw = self.patch_raw[str(station_sn)][idx]
        z = (raw - self.patch_mean) / self.patch_std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return z, self.patch_mask[str(station_sn)][idx].astype(np.float32)

    def raw_center_series(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sec = _timestamp_key(pd.Timestamp(issue_time))
        idx = self.issue_to_index[int(sec)]
        raw = self.patch_raw[str(station_sn)][idx]
        mask = self.patch_mask[str(station_sn)][idx]
        c = self.patch_size // 2
        return raw[:, c, c].astype(np.float32), raw.mean(axis=(1, 2)).astype(np.float32), mask.mean(axis=(1, 2)).astype(np.float32)

    def stats(self) -> dict:
        return {
            "patch_type": "stage3_interpolated_future_swr_patch",
            "swr_units": "kW/m2, computed as ratio * clear_sky",
            "stage3_temporal_anchor_weights": _stage3_fixed_anchor_weights(int(self.args.future_frames), 4).astype(float).tolist(),
            "field_patch_mean": float(self.patch_mean),
            "field_patch_std": float(self.patch_std),
            "field_patch_size": int(self.patch_size),
            "mask_mode": "constant_one_support_mask",
            "swr_anchor_cache_dir": str(self.cache_dir),
        }


def _resolve_swr_cache_arg(args) -> None:
    cache_dir = Path(args.swr_anchor_cache_dir)
    if not (cache_dir / "center_h8_swr_anchor_input_cache_meta.csv").exists():
        cache_dir = resolve_active_cache_dir(required=True)
    args.swr_anchor_cache_dir = cache_dir
    print(f"[stage4_swr_patch] active Stage 3 SWR-anchor cache: {cache_dir}", flush=True)


def _skip_field_run_alignment(time_df: pd.DataFrame, args) -> pd.DataFrame:
    print("[stage4_swr_patch] using Stage 3 SWR-anchor cache directly; skip PV-field run index alignment", flush=True)
    return time_df.reset_index(drop=True)


def _rewrite_run_meta(out_dir: Path) -> None:
    meta_path = out_dir / "run_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["purpose"] = (
        "Per-target-station temporal model guided directly by 5x5 future SWR patches "
        "interpolated from the Stage 3 SWR-anchor cache."
    )
    meta["swr_patch_ablation"] = {
        "source": "Stage 3 active SWR-anchor cache",
        "interpolation": "Stage 3 fixed temporal-anchor weights over [observed T-1h, predicted T, predicted T+1h, predicted T+2h]",
        "patch_channels": ["future_swr_patch_z", "constant_one_support_mask"],
        "comparison_target": "05l Stage 4 model guided by predicted PV-field patches",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _out_dir_from_argv() -> Path:
    argv = sys.argv[1:]
    for index, value in enumerate(argv):
        if value == "--out_dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
    return Path(m05.PYCHARM_DEFAULTS["out_dir"])


def main() -> None:
    ensure_stage4_dirs()
    m05.PYCHARM_DEFAULTS.update(
        {
            "out_dir": str(PER_STATION_STAGE3_SWR_PATCH_RUN_DIR),
            "swr_anchor_cache_dir": str(CENTER_H8_SWR_ANCHOR_CACHE_ROOT),
            "field_model_name": "stage3_interpolated_future_swr_patch",
            "field_checkpoint": "",
        }
    )
    m05.PredictedFieldPatchProvider = Stage3SWRPatchProvider
    m05.resolve_stage3_swr_anchor_cache_arg = _resolve_swr_cache_arg
    m05.assert_time_index_matches_field_run = _skip_field_run_alignment
    m05.main()
    _rewrite_run_meta(_out_dir_from_argv())


if __name__ == "__main__":
    main()
