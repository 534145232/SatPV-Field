#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))
MOTION_BRANCH_DIR = STAGE3_DIR / "ablation_motion_branch"
TEMPORAL_COLLAPSE_DIR = STAGE3_DIR / "ablation_temporal_collapse"
for extra_dir in (MOTION_BRANCH_DIR, TEMPORAL_COLLAPSE_DIR):
    if str(extra_dir) not in sys.path:
        sys.path.insert(0, str(extra_dir))

from visualize_regional_split_frames import _bounds, _xy_to_ij, ensure_dir, write_json
import stage3_motion_fusion_model_impl as stage3_field_module
from last_only_collapse import patch_last_only_collapse
from motion_branch import patch_motion_branch
from stage3_swr_anchor_cache import resolve_active_cache_dir
from stage4_paths import (
    BEST_STAGE3_FIELD_CHECKPOINT,
    BEST_STAGE3_FIELD_MODEL_NAME,
    BEST_STAGE3_FIELD_RUN_DIR,
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
    MULTISOURCE_CACHE_DIR,
    PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_ISSUE_TIME_INDEX,
    TARGET_SELECTION_DIR,
    TRAIN_PV_DIR,
    VAL_PV_DIR,
    ensure_stage4_dirs,
)
from stage4_metrics import regression_metric_extras


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b03 = _load_local_module("stage4_03_per_station_temporal_baseline", SCRIPT_DIR / "03_train_per_station_temporal_baseline.py")
f04 = stage3_field_module.f04
_SWR04L = stage3_field_module


def _load_swr04l():
    return _SWR04L

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "selection_dir": str(TARGET_SELECTION_DIR),
    "time_index_csv": str(TARGET_ISSUE_TIME_INDEX),
    "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
    "field_model_dir": str(BEST_STAGE3_FIELD_RUN_DIR),
    "field_model_name": str(BEST_STAGE3_FIELD_MODEL_NAME),
    "field_checkpoint": str(BEST_STAGE3_FIELD_CHECKPOINT),
    "swr_anchor_cache_dir": str(CENTER_H8_SWR_ANCHOR_CACHE_ROOT),
    "base_model_name": "pv_nwp",
    "normalize_swr_anchor": 1,
    "swr_history_frames": 4,
    "swr_future_anchors": 3,
    "pv_hidden_channels": 48,
    "anchor_channels": 48,
    "horizon_embed_channels": 8,
    "field_model_dropout": 0.05,
    "max_flow_pixels": 3.0,
    "out_dir": str(PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR),
    "train_pv_dir": str(TRAIN_PV_DIR),
    "val_pv_dir": str(VAL_PV_DIR),
    "capacity_scale": 1000.0,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1.0e-6,
    "field_patch_size": 5,
    "field_batch_size": 8,
    "hidden_dim": 96,
    "future_hidden_dim": 64,
    "field_hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.10,
    "batch_size": 128,
    "epochs": 60,
    "patience": 10,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "grad_clip": 1.0,
    "min_train_samples": 150,
    "large_delta_threshold": 0.10,
    "big_change_loss_weight": 0.5,
    "sample_count": 20,
    "seed": 2026,
    "device": "cuda",
    "disable_cudnn_rnn": True,
    "num_workers": 0,
    "dpi": 150,
    "health_filter": True,
    "pv_health_max": 1.35,
    "pv_health_active_threshold": 0.05,
    "pv_health_flatline_window": 8,
    "pv_health_flatline_range": 1e-5,
}


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        if value == "":
            continue
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-target-station temporal models with the same baseline inputs as step 03, "
            "plus a future PV_norm-field patch predicted by the 04l SWR-anchor motion-fusion field model."
        )
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=bool(value))
            parser.add_argument(f"--disable_{key}", action="store_false", dest=key)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        elif key == "field_checkpoint":
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir"}:
            parser.add_argument(f"--{key}", type=Path, default=Path(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _safe_read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_time_index_matches_field_run(time_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.field_model_dir) / "pvnorm_field_issue_time_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"field-model issue index not found: {path}")
    field_time = pd.read_csv(path)
    field_time["issue_time"] = pd.to_datetime(field_time["issue_time"])
    if "issue_sec" not in field_time.columns:
        field_time["issue_sec"] = field_time["issue_time"].map(_timestamp_key)
    required = ["issue_sec", "time_split", "past_times", "future_times"]
    missing = [c for c in required if c not in time_df.columns or c not in field_time.columns]
    if missing:
        raise ValueError(f"Cannot verify station/field split alignment; missing columns: {missing}")
    left = time_df[required].reset_index(drop=True).astype(str)
    right = field_time[required].reset_index(drop=True).astype(str)
    if len(left) == len(right) and left.equals(right):
        return time_df.reset_index(drop=True)
    by_sec = {int(row["issue_sec"]): row for _, row in time_df.iterrows()}
    rows = []
    mismatches = []
    missing_secs = []
    for _, row in field_time.iterrows():
        sec = int(row["issue_sec"])
        src = by_sec.get(sec)
        if src is None:
            missing_secs.append(sec)
            continue
        for col in ["time_split", "past_times", "future_times"]:
            if str(src[col]) != str(row[col]):
                mismatches.append({"issue_time": str(row["issue_time"]), "column": col, "station_time": str(src[col]), "field_time": str(row[col])})
                break
        item = src.to_dict()
        item["issue_time"] = pd.Timestamp(row["issue_time"])
        item["issue_sec"] = sec
        rows.append(item)
    if missing_secs or mismatches:
        raise ValueError(
            "Station issue_time_index cannot be aligned to the 04l PV-field model issue index. "
            f"missing_secs={missing_secs[:5]} mismatches={mismatches[:3]}"
        )
    aligned = pd.DataFrame(rows).reset_index(drop=True)
    dropped = len(time_df) - len(aligned)
    if dropped:
        print(
            f"[swr_anchor04l_field_patch] station issue index aligned to field run: "
            f"{len(time_df)} -> {len(aligned)} rows, dropped={dropped}",
            flush=True,
        )
    return aligned


def filter_time_index_to_swr_anchor_cache(time_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    meta_path = Path(args.swr_anchor_cache_dir) / "center_h8_swr_anchor_input_cache_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"SWR-anchor cache metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    if "issue_sec" not in meta.columns:
        meta["issue_sec"] = meta["issue_time"].map(_timestamp_key)
    valid_secs = set(meta["issue_sec"].astype(np.int64).tolist())
    before = len(time_df)
    out = time_df[time_df["issue_sec"].astype(np.int64).isin(valid_secs)].copy().reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        by_split = (
            time_df.loc[~time_df["issue_sec"].astype(np.int64).isin(valid_secs), "time_split"]
            .astype(str)
            .value_counts()
            .to_dict()
        )
        print(
            f"[swr_anchor04l_field_patch] filtered station/field issue index by valid SWR-anchor cache: "
            f"{before} -> {len(out)} rows, dropped={dropped}, dropped_by_split={by_split}",
            flush=True,
        )
    return out


def resolve_stage3_swr_anchor_cache_arg(args: argparse.Namespace) -> None:
    cache_dir = Path(args.swr_anchor_cache_dir)
    if not (cache_dir / "center_h8_swr_anchor_input_cache_meta.csv").exists():
        cache_dir = resolve_active_cache_dir(required=True)
    args.swr_anchor_cache_dir = cache_dir
    print(f"[stage4_field_patch] active Stage 3 SWR-anchor cache: {cache_dir}", flush=True)


def _extract_patch(arr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    radius = int(size) // 2
    arr = np.asarray(arr, dtype=np.float32)
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="constant", constant_values=0.0)
    py = int(cy) + radius
    px = int(cx) + radius
    return padded[py - radius : py + radius + 1, px - radius : px + radius + 1].astype(np.float32)


class PredictedFieldPatchProvider:
    def __init__(self, targets: pd.DataFrame, time_df: pd.DataFrame, args: argparse.Namespace, device: str):
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

    def _field_args(self) -> argparse.Namespace:
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
            models=str(self.args.base_model_name),
            model_name=str(self.args.field_model_name),
            base_model_name=str(self.args.base_model_name),
            swr_anchor_cache_dir=Path(self.args.swr_anchor_cache_dir),
            normalize_swr_anchor=int(self.args.normalize_swr_anchor),
            swr_history_frames=int(self.args.swr_history_frames),
            swr_future_anchors=int(self.args.swr_future_anchors),
            pv_hidden_channels=int(self.args.pv_hidden_channels),
            anchor_channels=int(self.args.anchor_channels),
            horizon_embed_channels=int(self.args.horizon_embed_channels),
            delta_scale=1.0,
            batch_size=int(self.args.field_batch_size),
            num_workers=0,
        )

    def _load_field_model(self, ckpt: dict):
        swr04l = _load_swr04l()
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args"), dict) else {}
        model = swr04l.make_swr_anchor_attention_model(
            future_frames=int(self.args.future_frames),
            in_channels=int(ckpt["in_channels"]),
            future_channels=int(ckpt["future_channels"]),
            anchor_channels=int(ckpt.get("swr_anchor_channels", 2)),
            pv_hidden=int(ckpt_args.get("pv_hidden_channels", self.args.pv_hidden_channels)),
            anchor_hidden=int(ckpt_args.get("anchor_channels", self.args.anchor_channels)),
            horizon_embed_channels=int(ckpt_args.get("horizon_embed_channels", self.args.horizon_embed_channels)),
            dropout=float(ckpt_args.get("dropout", self.args.field_model_dropout)),
            swr_history_frames=int(ckpt_args.get("swr_history_frames", self.args.swr_history_frames)),
            swr_future_anchors=int(ckpt_args.get("swr_future_anchors", self.args.swr_future_anchors)),
            max_flow_pixels=float(ckpt_args.get("max_flow_pixels", self.args.max_flow_pixels)),
        )
        model = patch_motion_branch(model, "warp_direct_gate")
        model = patch_last_only_collapse(model)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model

    def _swr_anchor_array_for_time_df(self, swr_arr_all: np.ndarray, swr_meta: pd.DataFrame) -> np.ndarray:
        meta = swr_meta.copy()
        meta["issue_time"] = pd.to_datetime(meta["issue_time"])
        if "issue_sec" not in meta.columns:
            meta["issue_sec"] = meta["issue_time"].map(_timestamp_key)
        if "cache_idx" not in meta.columns:
            meta["cache_idx"] = np.arange(len(meta), dtype=np.int64)
        by_sec = {int(row["issue_sec"]): int(row["cache_idx"]) for _, row in meta.iterrows()}
        indices = []
        missing = []
        for sec in self.time_df["issue_sec"].astype(np.int64).tolist():
            idx = by_sec.get(int(sec))
            if idx is None:
                missing.append(int(sec))
            else:
                indices.append(idx)
        if missing:
            raise KeyError(
                f"{len(missing)} issue times in the aligned station index are missing from SWR-anchor cache; "
                f"first missing issue_sec={missing[:5]}"
            )
        return np.asarray(swr_arr_all[np.asarray(indices, dtype=np.int64)], dtype=np.float32)

    def _target_cells(self, bounds: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
        cells = {}
        for _, row in self.targets.iterrows():
            sn = str(row["station_sn"])
            cells[sn] = _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, int(self.args.grid_size))
        return cells

    def _build(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        ckpt_arg = str(getattr(self.args, "field_checkpoint", "")).strip()
        ckpt_path = Path(ckpt_arg) if ckpt_arg else Path("")
        if not ckpt_arg:
            ckpt_path = Path(self.args.field_model_dir) / f"{self.args.field_model_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"field model checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        field_args = self._field_args()
        split_df = f04._load_station_split(field_args)
        bounds = _bounds(split_df)
        self.bounds = bounds
        cells = self._target_cells(bounds)
        ms_cache = f04._load_multisource_cache(Path(self.args.multisource_cache_dir))
        pv_cache = f04.ReadOnlyDiskFrameCache(Path(self.args.frame_cache_dir), self.time_df, field_args)
        pv_stats = ckpt.get("pv_field_norm_stats") or _safe_read_json(Path(self.args.field_model_dir) / "pv_field_norm_stats.json")
        ms_stats = ckpt.get("multisource_norm_stats") or _safe_read_json(Path(self.args.field_model_dir) / "multisource_norm_stats.json")
        swr04l = _load_swr04l()
        base_dataset = f04.TargetExcludedPVNormFieldDataset(
            self.time_df,
            pv_cache,
            ms_cache,
            bounds,
            pv_stats,
            ms_stats,
            str(self.args.base_model_name),
            field_args,
        )
        swr_arr_all, swr_meta, _swr_info = swr04l._load_prebuilt_swr_anchor_cache(Path(self.args.swr_anchor_cache_dir))
        swr_arr = self._swr_anchor_array_for_time_df(swr_arr_all, swr_meta)
        swr_stats = ckpt.get("swr_anchor_norm_stats")
        if swr_stats is None:
            stats_path = Path(self.args.field_model_dir) / "swr_anchor_norm_stats.json"
            if stats_path.exists():
                swr_stats = _safe_read_json(stats_path)
            else:
                train_idx = self.time_df.index[self.time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
                swr_stats = swr04l._compute_swr_anchor_stats(swr_arr[train_idx], Path(self.args.field_model_dir), force=False)
        dataset = swr04l.SWRAnchorPVFieldDataset(base_dataset, swr_arr, field_args, swr_stats)
        loader = DataLoader(dataset, batch_size=int(self.args.field_batch_size), shuffle=False, num_workers=0)
        model = self._load_field_model(ckpt)

        for sn in self.targets["station_sn"].astype(str):
            n = len(self.time_df)
            self.patch_raw[sn] = np.zeros((n, int(self.args.future_frames), self.patch_size, self.patch_size), dtype=np.float32)
            self.patch_mask[sn] = np.zeros((n, int(self.args.future_frames), self.patch_size, self.patch_size), dtype=np.float32)

        offset = 0
        with torch.no_grad():
            for x, future_feat, swr_anchor, _y, _m in loader:
                pred_z = model(
                    x.to(self.device),
                    future_feat.to(self.device),
                    swr_anchor.to(self.device),
                ).detach().cpu().numpy()
                pred = f04._denorm_field(pred_z, pv_stats)
                latest_mask = x[:, 1, -1].numpy().astype(np.float32)
                bsz = pred.shape[0]
                for bi in range(bsz):
                    row_index = offset + bi
                    for sn, (cy, cx) in cells.items():
                        mask_patch = _extract_patch(latest_mask[bi], cy, cx, self.patch_size)
                        for h in range(int(self.args.future_frames)):
                            self.patch_raw[sn][row_index, h] = _extract_patch(pred[bi, h], cy, cx, self.patch_size)
                            self.patch_mask[sn][row_index, h] = mask_patch
                offset += bsz
                if offset % 500 == 0:
                    print(f"[swr_anchor04l_field_patch] predicted patches for {offset}/{len(self.time_df)} issue times", flush=True)

        train_idx = self.time_df.index[self.time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
        chunks = [self.patch_raw[str(sn)][train_idx].reshape(-1) for sn in self.targets["station_sn"].astype(str)]
        vals = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        self.patch_mean = float(np.mean(vals)) if vals.size else 0.0
        self.patch_std = float(np.std(vals)) if vals.size else 1.0
        if not np.isfinite(self.patch_std) or self.patch_std < 1e-6:
            self.patch_std = 1.0
        print(
            f"[swr_anchor04l_field_patch] ready: stations={len(self.targets)} issues={len(self.time_df)} "
            f"patch={self.patch_size} mean={self.patch_mean:.5f} std={self.patch_std:.5f}",
            flush=True,
        )

    def get(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        sec = _timestamp_key(pd.Timestamp(issue_time))
        idx = self.issue_to_index[int(sec)]
        raw = self.patch_raw[str(station_sn)][idx]
        mask = self.patch_mask[str(station_sn)][idx]
        z = (raw - self.patch_mean) / self.patch_std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return z, mask.astype(np.float32)

    def raw_center_series(self, station_sn: str, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sec = _timestamp_key(pd.Timestamp(issue_time))
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
            "mask_mode": "latest_input_support_mask_repeated_for_all_future_horizons",
        }


class StationTemporalFieldPatchDataset:
    def __init__(
        self,
        station_sn: str,
        station_df: pd.DataFrame,
        station_row: pd.Series,
        time_df: pd.DataFrame,
        stats: dict,
        patch_provider: PredictedFieldPatchProvider,
        args: argparse.Namespace,
    ):
        self.station_sn = str(station_sn)
        self.df = station_df
        self.row = station_row
        self.time_df = time_df.reset_index(drop=True)
        self.stats = stats
        self.patch_provider = patch_provider
        self.args = args
        self.samples: list[int] = []
        self._build_samples()

    def _has_all(self, times: list[pd.Timestamp]) -> bool:
        return all(pd.Timestamp(t) in self.df.index for t in times)

    def _build_samples(self) -> None:
        for i, row in self.time_df.iterrows():
            past = b03._parse_time_list(row["past_times"])
            future = b03._parse_time_list(row["future_times"])
            if self._has_all(past) and self._has_all(future):
                self.samples.append(int(i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        issue = pd.Timestamp(row["issue_time"])
        past_times = b03._parse_time_list(row["past_times"])
        future_times = b03._parse_time_list(row["future_times"])
        hist_raw = self.df.loc[past_times, ["pv_norm", *b03.FUTURE_NWP_NAMES]].to_numpy(np.float32)
        fut_raw = self.df.loc[future_times, b03.FUTURE_NWP_NAMES].to_numpy(np.float32)
        y_raw = self.df.loc[future_times, "pv_norm"].to_numpy(np.float32)
        hist_z = (hist_raw - self.stats["hist_mean"]) / self.stats["hist_std"]
        fut_z = (fut_raw - self.stats["future_mean"]) / self.stats["future_std"]
        time_solar = np.stack([b03._time_solar_point(ts, float(self.row["lon"]), float(self.row["lat"])) for ts in future_times], axis=0)
        future_feat = np.concatenate([fut_z, time_solar], axis=1).astype(np.float32)
        y_z = (y_raw - float(self.stats["pv_mean"])) / float(self.stats["pv_std"])
        hist_pv = hist_raw[:, 0].astype(np.float32)
        field_z, field_mask = self.patch_provider.get(self.station_sn, issue)
        field_patch = np.stack([field_z, field_mask], axis=1).astype(np.float32)
        return {
            "hist": hist_z.astype(np.float32),
            "future": future_feat.astype(np.float32),
            "field_patch": field_patch,
            "y_z": y_z.astype(np.float32),
            "y": y_raw.astype(np.float32),
            "hist_pv": hist_pv,
            "issue_time": str(issue),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "field_patch": np.stack([b["field_patch"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


def _configure_torch_runtime(torch, args: argparse.Namespace) -> None:
    if bool(getattr(args, "disable_cudnn_rnn", True)) and torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print(
            "[station_field_patch] disabled torch.backends.cudnn for small LSTM stability",
            flush=True,
        )


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def make_model(hist_dim: int, future_dim: int, field_dim: int, future_frames: int, args: argparse.Namespace):
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(args.hidden_dim)
            self.future_frames = int(future_frames)
            self.hist_encoder = nn.LSTM(
                input_size=int(hist_dim),
                hidden_size=hidden,
                num_layers=int(args.num_layers),
                batch_first=True,
                dropout=float(args.dropout) if int(args.num_layers) > 1 else 0.0,
                bidirectional=True,
            )
            self.hist_proj = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
            self.future_mlp = nn.Sequential(
                nn.Linear(int(future_dim), int(args.future_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(args.future_hidden_dim)),
                nn.Linear(int(args.future_hidden_dim), hidden),
                nn.GELU(),
            )
            self.field_mlp = nn.Sequential(
                nn.Linear(int(field_dim), int(args.field_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(args.field_hidden_dim)),
                nn.Linear(int(args.field_hidden_dim), hidden),
                nn.GELU(),
            )
            self.horizon_embed = nn.Embedding(int(future_frames), hidden // 2)
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + hidden + hidden // 2, hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, hist, future, field_patch):
            enc, _ = self.hist_encoder(hist)
            hist_ctx = self.hist_proj(enc[:, -1])
            fut_ctx = self.future_mlp(future)
            patch_flat = field_patch.flatten(start_dim=2)
            field_ctx = self.field_mlp(patch_flat)
            h_ids = torch.arange(self.future_frames, device=future.device)
            h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
            hist_rep = hist_ctx[:, None].expand(-1, self.future_frames, -1)
            return self.decoder(torch.cat([hist_rep, fut_ctx, field_ctx, h_emb], dim=-1)).squeeze(-1)

    return Model()


def weighted_loss(pred_z, target_z, target_raw, hist_pv_raw, args):
    import torch
    import torch.nn.functional as F

    last = hist_pv_raw[:, -1][:, None]
    true_delta = torch.abs(target_raw - last)
    weights = 1.0 + float(args.big_change_loss_weight) * (true_delta >= float(args.large_delta_threshold)).float()
    loss = F.smooth_l1_loss(pred_z, target_z, reduction="none") * weights
    return loss.sum() / torch.clamp(weights.sum(), min=1.0)


def train_one(model, loaders, args, device: str, ckpt_path: Path) -> pd.DataFrame:
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = float("inf")
    bad = 0
    logs = []
    model.to(device)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            patch = _tensor(batch, "field_patch", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            opt.zero_grad(set_to_none=True)
            loss = weighted_loss(model(hist, fut, patch), y, y_raw, hist_pv, args)
            loss.backward()
            if float(getattr(args, "grad_clip", 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, loaders["val"], args, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[station_field_patch] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[station_field_patch] early stop at epoch={epoch}", flush=True)
                break
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(logs)


def evaluate_loss(model, loader, args, device: str) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            patch = _tensor(batch, "field_patch", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            losses.append(float(weighted_loss(model(hist, fut, patch), y, y_raw, hist_pv, args).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(
    model,
    dataset: StationTemporalFieldPatchDataset,
    loader,
    stats: dict,
    patch_provider: PredictedFieldPatchProvider,
    split: str,
    args,
    device: str,
    capacity_raw: float | None = None,
    capacity_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    model.eval()
    details = []
    pv_mean = float(stats["pv_mean"])
    pv_std = float(stats["pv_std"])
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            patch = _tensor(batch, "field_patch", device, torch)
            pred_z = model(hist, fut, patch).detach().cpu().numpy()
            pred = pred_z * pv_std + pv_mean
            y = np.asarray(batch["y"], dtype=np.float32)
            hist_pv = np.asarray(batch["hist_pv"], dtype=np.float32)
            issues = batch["issue_time"]
            for i in range(pred.shape[0]):
                field_center, field_mean, field_support = patch_provider.raw_center_series(dataset.station_sn, issues[i])
                last = float(hist_pv[i, -1])
                for h in range(pred.shape[1]):
                    e = float(pred[i, h] - y[i, h])
                    delta = float(y[i, h] - last)
                    row = {
                        "station_sn": dataset.station_sn,
                        "model": "per_station_temporal_swr_anchor04l_field_patch",
                        "eval_split": split,
                        "issue_time": str(issues[i]),
                        "h": int(h + 1),
                        "horizon": f"h{h + 1}",
                        "pred_pv_norm": float(pred[i, h]),
                        "true_pv_norm": float(y[i, h]),
                        "capacity_raw": float(capacity_raw) if capacity_raw is not None else float("nan"),
                        "capacity_scale": float(capacity_scale) if capacity_scale is not None else float("nan"),
                        "abs_err_pv_norm": abs(e),
                        "sq_err_pv_norm": e * e,
                        "true_delta_pv_norm": delta,
                        "large_change": abs(delta) >= float(args.large_delta_threshold),
                        "field_patch_center_pv_norm": float(field_center[h]),
                        "field_patch_mean_pv_norm": float(field_mean[h]),
                        "field_patch_support_mean": float(field_support[h]),
                    }
                    for k, value in enumerate(hist_pv[i], start=1):
                        row[f"hist_pv_norm_{k:02d}"] = float(value)
                    details.append(row)
    detail_df = pd.DataFrame(details)
    metric_rows = []
    if not detail_df.empty:
        for horizon, g in list(detail_df.groupby("horizon")) + [("all", detail_df)]:
            row = {
                "station_sn": dataset.station_sn,
                "model": "per_station_temporal_swr_anchor04l_field_patch",
                "eval_split": split,
                "horizon": horizon,
                "n_samples": int(len(g)),
                "mae_pv_norm": float(g["abs_err_pv_norm"].mean()),
                "rmse_pv_norm": float(math.sqrt(g["sq_err_pv_norm"].mean())),
                "high_change_mae_pv_norm": float(g.loc[g["large_change"], "abs_err_pv_norm"].mean()) if bool(g["large_change"].any()) else float("nan"),
                "high_change_n": int(g["large_change"].sum()),
                "field_patch_support_mean": float(g["field_patch_support_mean"].mean()),
            }
            row.update(regression_metric_extras(g, "pred_pv_norm"))
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), detail_df


def plot_samples(detail_df: pd.DataFrame, station_sn: str, out_dir: Path, args) -> None:
    if detail_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(out_dir / "figures" / str(station_sn))
    test_df = detail_df[detail_df["eval_split"] == "test"].copy()
    issues = test_df["issue_time"].drop_duplicates().astype(str).to_numpy()
    if issues.size == 0:
        return
    rng = np.random.default_rng(int(args.seed) + sum(ord(ch) for ch in str(station_sn)))
    if issues.size > int(args.sample_count):
        issues = rng.choice(issues, size=int(args.sample_count), replace=False)
    hist_cols = [f"hist_pv_norm_{k:02d}" for k in range(1, int(args.past_frames) + 1)]
    for i, issue in enumerate(issues):
        sub = test_df[test_df["issue_time"].astype(str) == str(issue)].sort_values("h")
        if len(sub) != int(args.future_frames):
            continue
        x = np.arange(1, int(args.future_frames) + 1)
        hist_x = np.arange(-int(args.past_frames) + 1, 1)
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.4), sharex=False)
        hist = sub.iloc[0][hist_cols].to_numpy(np.float64)
        axes[0].plot(hist_x, hist, color="#7f7f7f", marker=".", linewidth=2.0, label="history PV_norm")
        axes[0].axvline(0, color="0.75", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.2, label="truth")
        axes[0].plot(x, sub["pred_pv_norm"].to_numpy(np.float64), color="#d62728", marker="o", linewidth=2.0, label="+ predicted field patch")
        axes[0].plot(x, sub["field_patch_center_pv_norm"].to_numpy(np.float64), color="#2ca02c", marker=".", linewidth=1.5, label="field patch center")
        axes[0].set_title(f"Per-station temporal + field patch | {station_sn} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        err = sub["pred_pv_norm"].to_numpy(np.float64) - sub["true_pv_norm"].to_numpy(np.float64)
        axes[1].bar(x, err, color=np.where(err >= 0, "#d62728", "#1f77b4"), alpha=0.85)
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("pred - truth")
        axes[1].grid(alpha=0.25)

        axes[2].plot(x, sub["field_patch_mean_pv_norm"].to_numpy(np.float64), color="#2ca02c", marker="o", label="field patch mean")
        axes[2].plot(x, sub["field_patch_support_mean"].to_numpy(np.float64), color="#9467bd", marker=".", label="patch support mean")
        axes[2].set_ylabel("field patch")
        axes[2].set_xlabel("forecast horizon")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"h{k}" for k in x])
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best")
        fig.tight_layout()
        safe_issue = str(issue).replace(":", "").replace(" ", "_")
        fig.savefig(fig_dir / f"field_patch_curve_{i:03d}_{safe_issue}.png", dpi=int(args.dpi))
        plt.close(fig)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in metrics.groupby(["model", "eval_split", "horizon"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "eval_split": keys[1],
                "horizon": keys[2],
                "n_stations": int(g["station_sn"].nunique()),
                "station_mean_mae_pv_norm": float(g["mae_pv_norm"].mean()),
                "station_median_mae_pv_norm": float(g["mae_pv_norm"].median()),
                "station_mean_rmse_pv_norm": float(g["rmse_pv_norm"].mean()),
                "station_mean_r2_pv_norm": float(g["r2_pv_norm"].mean()) if "r2_pv_norm" in g.columns else float("nan"),
                "station_mean_acc_installed_capacity": float(g["acc_installed_capacity"].mean()) if "acc_installed_capacity" in g.columns else float("nan"),
                "station_mean_nrmse_installed_capacity": float(g["nrmse_installed_capacity"].mean()) if "nrmse_installed_capacity" in g.columns else float("nan"),
                "station_mean_high_change_mae_pv_norm": float(g["high_change_mae_pv_norm"].mean()),
                "station_sum_high_change_n": int(g["high_change_n"].sum()),
                "field_patch_support_mean": float(g["field_patch_support_mean"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["eval_split", "horizon"]).reset_index(drop=True)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for station field-patch training.") from exc

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    _configure_torch_runtime(torch, args)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"target_stations.csv not found: {target_path}")
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(_timestamp_key)
    resolve_stage3_swr_anchor_cache_arg(args)
    time_df = assert_time_index_matches_field_run(time_df, args)
    time_df = filter_time_index_to_swr_anchor_cache(time_df, args)
    time_df.to_csv(out_dir / "station_issue_time_index_aligned_to_swr_anchor04l_field_run.csv", index=False)

    patch_provider = PredictedFieldPatchProvider(targets, time_df, args, device)
    write_json(out_dir / "field_patch_stats.json", patch_provider.stats())

    all_metrics, all_details, all_logs, all_meta = [], [], [], []
    for idx, row in targets.reset_index(drop=True).iterrows():
        sn = str(row["station_sn"])
        print(f"==== per-station temporal + field patch {idx + 1}/{len(targets)}: {sn} ====", flush=True)
        station_dir = ensure_dir(out_dir / "per_station" / sn)
        station_df, meta = b03.load_station_table(sn, args)
        all_meta.append(meta)
        train_time = b03._split_time_df(time_df, "train")
        stats = b03.make_station_stats(station_df, train_time)
        datasets, loaders = {}, {}
        for split in ["train", "val", "test"]:
            datasets[split] = StationTemporalFieldPatchDataset(
                sn,
                station_df,
                row,
                b03._split_time_df(time_df, split),
                stats,
                patch_provider,
                args,
            )
            print(f"[station_swr_anchor04l_field_patch] {sn} split={split} samples={len(datasets[split])}", flush=True)
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
                collate_fn=collate,
            )
        if len(datasets["train"]) < int(args.min_train_samples):
            print(f"[station_swr_anchor04l_field_patch] skip {sn}: train samples={len(datasets['train'])} < {int(args.min_train_samples)}", flush=True)
            continue
        field_dim = 2 * int(args.field_patch_size) * int(args.field_patch_size)
        model = make_model(
            1 + len(b03.FUTURE_NWP_NAMES),
            len(b03.FUTURE_NWP_NAMES) + 3,
            field_dim,
            int(args.future_frames),
            args,
        )
        log_df = train_one(model, loaders, args, device, station_dir / "best.pt")
        log_df.insert(0, "station_sn", sn)
        all_logs.append(log_df)
        write_json(
            station_dir / "normalization_stats.json",
            {
                "pv_mean": float(stats["pv_mean"]),
                "pv_std": float(stats["pv_std"]),
                "hist_mean": np.asarray(stats["hist_mean"]).astype(float).tolist(),
                "hist_std": np.asarray(stats["hist_std"]).astype(float).tolist(),
                "future_mean": np.asarray(stats["future_mean"]).astype(float).tolist(),
                "future_std": np.asarray(stats["future_std"]).astype(float).tolist(),
                "field_patch": patch_provider.stats(),
                "hist_features": ["pv_norm", *b03.FUTURE_NWP_NAMES],
                "future_features": [*b03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
            },
        )
        station_details = []
        capacity_raw = float(meta.get("capacity_raw", np.nan))
        capacity_scale = float(meta.get("capacity_scale", np.nan))
        for split in ["val", "test"]:
            metrics, detail = evaluate(
                model,
                datasets[split],
                loaders[split],
                stats,
                patch_provider,
                split,
                args,
                device,
                capacity_raw=capacity_raw,
                capacity_scale=capacity_scale,
            )
            print(f"==== {sn} {split} ====")
            print(metrics[metrics["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False))
            all_metrics.append(metrics)
            station_details.append(detail)
        station_detail = pd.concat(station_details, ignore_index=True) if station_details else pd.DataFrame()
        station_detail.to_csv(station_dir / "eval_detail.csv", index=False)
        plot_samples(station_detail, sn, out_dir, args)
        all_details.append(station_detail)

    pd.DataFrame(all_meta).to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    metrics_df.to_csv(out_dir / "per_station_temporal_swr_anchor04l_field_patch_metrics.csv", index=False)
    detail_df.to_csv(out_dir / "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / "per_station_temporal_swr_anchor04l_field_patch_train_log.csv", index=False)
    agg = aggregate(metrics_df)
    agg.to_csv(out_dir / "per_station_temporal_swr_anchor04l_field_patch_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Per-target-station temporal model with a future PV_norm-field patch predicted by the best Stage 3 SWR-anchor motion-fusion field model.",
            "baseline_alignment": "Same station inputs, label definition, splits, optimizer, and temporal architecture style as step 03.",
            "field_patch": {
                "source": str(Path(args.field_checkpoint)) if str(args.field_checkpoint).strip() else str(Path(args.field_model_dir) / f"{args.field_model_name}_best.pt"),
                "field_model_variant": "stage3_best_full_swr_future_last_only_warp_direct_gate",
                "stage3_field_model_dir": str(args.field_model_dir),
                "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
                "base_field_model_name": str(args.base_model_name),
                "patch_size": int(args.field_patch_size),
                "channels": ["predicted_future_pvnorm_field_z", "latest_support_mask"],
                "mask_mode": "latest input support mask, repeated across future horizons to avoid future-data leakage",
            },
            "target_stations": targets["station_sn"].astype(str).tolist(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print("==== Per-station temporal + field patch aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"saved: {out_dir / 'per_station_temporal_swr_anchor04l_field_patch_aggregate.csv'}")
    print(f"figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
