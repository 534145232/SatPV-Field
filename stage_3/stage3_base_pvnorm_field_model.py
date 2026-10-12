#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPT_DIR))

from build_fullfield_multisource_cache import _resize_bilinear
from train_fullfield_dense_temporal_forecast import _parse_time_list
from train_fullfield_dense_temporal_residual_unet import _grid_lonlat, _time_solar_features
from train_fullfield_dense_temporal_residual_unet_v2 import ReadOnlyDiskFrameCache, _masked_for_plot
from train_fullfield_multisource_residual_unet import MODEL_SPECS, NWP_NAMES, _load_multisource_cache, _normalize_h8, _normalize_nwp
from visualize_regional_split_frames import _bounds, ensure_dir, write_json
import stage3_paths as p3

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "selection_dir": str(p3.TARGET_SELECTION_DIR),
    "target_run_dir": str(p3.TARGET_EXCLUDED_RUN_DIR),
    "frame_cache_dir": str(p3.TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "multisource_cache_dir": str(p3.MULTISOURCE_CACHE_DIR),
    "out_dir": str(p3.RUNS_DIR / "base_pvnorm_field_model"),
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1.0e-6,
    "freq_minutes": 15,
    "past_frames": 12,
    "future_frames": 8,
    "models": "pv_h8_nwp",
    "delta_scale": 1.0,
    "batch_size": 8,
    "epochs": 60,
    "patience": 12,
    "early_stop_min_delta": 5.0e-5,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "lr_patience": 4,
    "lr_factor": 0.5,
    "large_delta_threshold": 0.10,
    "big_change_loss_weight": 0.5,
    "seed": 2026,
    "device": "cuda",
    "num_workers": 0,
    "sample_count": 8,
    "visual_horizons": "1,4,8",
    "dpi": 150,
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
        description=(
            "Train target-excluded PV_norm-field forecasting models. The PV field value is the target-excluded "
            "DensePV frame cache value, and both PV inputs and labels are z-scored using train split masked pixels."
        )
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if key.endswith("_dir"):
            parser.add_argument(f"--{key}", type=Path, default=Path(value))
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _parse_visual_horizons(text: str, future_frames: int) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if 1 <= h <= int(future_frames):
            out.append(h - 1)
    return out or [0, int(future_frames) - 1]


def _safe_read_json(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _station_sns_sha1(split_df: pd.DataFrame) -> str:
    sns = sorted(str(v) for v in split_df["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(sns).encode("utf-8")).hexdigest()


def _assert_issue_time_alignment(args: argparse.Namespace, ms_time_df: pd.DataFrame) -> None:
    path = Path(args.selection_dir) / "issue_time_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"selection issue_time_index.csv not found: {path}")
    ref = pd.read_csv(path)
    required = ["issue_sec", "time_split", "past_times", "future_times"]
    missing = [c for c in required if c not in ref.columns or c not in ms_time_df.columns]
    if missing:
        raise ValueError(f"Cannot verify issue-time alignment; missing columns: {missing}")
    left = ref[required].reset_index(drop=True).astype(str)
    right = ms_time_df[required].reset_index(drop=True).astype(str)
    if len(left) != len(right) or not left.equals(right):
        raise ValueError(
            "Multisource cache issue_time_index does not exactly match 01 selection issue_time_index.csv. "
            "Rebuild 00 and 01 with the same time_start/time_end/train_end/val_end/past/future settings before training 04."
        )


def _assert_target_exclusion(args: argparse.Namespace) -> None:
    target_path = Path(args.selection_dir) / "target_stations.csv"
    excluded_path = Path(args.target_run_dir) / "target_excluded_station_split.csv"
    if not excluded_path.exists():
        excluded_path = Path(args.selection_dir) / "target_excluded_station_split.csv"
    if not target_path.exists() or not excluded_path.exists():
        raise FileNotFoundError(f"Cannot verify target exclusion; need {target_path} and {excluded_path}.")
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    excluded = pd.read_csv(excluded_path, dtype={"station_sn": str})
    overlap = sorted(set(targets["station_sn"].astype(str)) & set(excluded["station_sn"].astype(str)))
    if overlap:
        raise ValueError(f"Target PV stations still present in target-excluded station split: {overlap[:20]}")

    meta_path = Path(args.frame_cache_dir) / "frame_cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"PV field frame cache metadata not found: {meta_path}")
    meta = _safe_read_json(meta_path)
    expected_count = int(len(excluded))
    expected_sha1 = _station_sns_sha1(excluded)
    if "source_station_count" not in meta or "source_station_sns_sha1" not in meta:
        raise ValueError(
            "PV field frame cache metadata does not record the source station set. "
            "Re-run 01_select_targets_build_pvnorm_cache_new.py so cached frames can be audited for target exclusion."
        )
    if int(meta["source_station_count"]) != expected_count or str(meta["source_station_sns_sha1"]) != expected_sha1:
        raise ValueError(
            "PV field frame cache source station set does not match target_excluded_station_split.csv. "
            "Delete/rebuild the frame cache to avoid target-station leakage from a reused cache."
        )


def _load_station_split(args: argparse.Namespace) -> pd.DataFrame:
    candidates = [
        Path(args.target_run_dir) / "station_split.csv",
        Path(args.selection_dir) / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, dtype={"station_sn": str})
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            return df.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)
    raise FileNotFoundError("station split not found. Run 01_select_targets_build_pvnorm_cache_new.py first.")


def _compute_or_load_multisource_stats(ms_cache: dict, out_dir: Path, time_df: pd.DataFrame) -> dict:
    path = out_dir / "multisource_norm_stats.json"
    if path.exists():
        return _safe_read_json(path)
    idx = time_df.index[time_df["time_split"].astype(str) == "train"].to_numpy(np.int64)
    if idx.size == 0:
        raise ValueError("No train rows for multisource normalization.")
    h8 = np.asarray(ms_cache["h8"][idx], dtype=np.float32)
    h8_vals = h8[np.isfinite(h8)]
    h8_mean = float(np.nanmean(h8_vals)) if h8_vals.size else 0.0
    h8_std = float(np.nanstd(h8_vals)) if h8_vals.size else 1.0
    if not np.isfinite(h8_std) or h8_std < 1e-6:
        h8_std = 1.0

    nwp = np.asarray(ms_cache["nwp"][idx], dtype=np.float32)
    nwp_mean, nwp_std = [], []
    for c in range(nwp.shape[2]):
        vals = nwp[:, :, c]
        vals = vals[np.isfinite(vals)]
        mean = float(np.nanmean(vals)) if vals.size else 0.0
        std = float(np.nanstd(vals)) if vals.size else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        nwp_mean.append(mean)
        nwp_std.append(std)
    stats = {
        "h8_mean": h8_mean,
        "h8_std": h8_std,
        "nwp_names": NWP_NAMES,
        "nwp_mean": nwp_mean,
        "nwp_std": nwp_std,
        "source": str(ms_cache["dir"]),
    }
    write_json(path, stats)
    return stats


def _compute_or_load_pv_field_stats(pv_cache: ReadOnlyDiskFrameCache, time_df: pd.DataFrame, out_dir: Path) -> dict:
    path = out_dir / "pv_field_norm_stats.json"
    if path.exists():
        return _safe_read_json(path)
    secs: set[int] = set()
    train_df = time_df[time_df["time_split"].astype(str) == "train"]
    for _, row in train_df.iterrows():
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            secs.add(_timestamp_key(ts))
    if not secs:
        raise ValueError("No train timestamps available for PV field normalization.")
    count = 0
    s1 = 0.0
    s2 = 0.0
    for sec in sorted(secs):
        field, mask = pv_cache.get(pd.Timestamp(sec, unit="s"))
        vals = np.asarray(field, dtype=np.float64)[np.asarray(mask) > 0.5]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        count += int(vals.size)
        s1 += float(vals.sum())
        s2 += float((vals * vals).sum())
    if count <= 0:
        raise ValueError("No valid masked PV field pixels found for normalization.")
    mean = s1 / count
    var = max(0.0, s2 / count - mean * mean)
    std = math.sqrt(var)
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    stats = {
        "pv_field_mean": float(mean),
        "pv_field_std": float(std),
        "n_pixels": int(count),
        "source": str(pv_cache.cache_dir),
        "definition": "target-excluded PV_norm field cache value; z-scored on train split masked pixels",
    }
    write_json(path, stats)
    return stats


def _normalize_field(field: np.ndarray, mask: np.ndarray, pv_stats: dict) -> np.ndarray:
    mean = float(pv_stats["pv_field_mean"])
    std = max(float(pv_stats["pv_field_std"]), 1e-6)
    out = (np.asarray(field, dtype=np.float32) - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[np.asarray(mask) <= 0.5] = 0.0
    return out.astype(np.float32)


def _denorm_field(field_z: np.ndarray, pv_stats: dict) -> np.ndarray:
    return np.asarray(field_z, dtype=np.float32) * float(pv_stats["pv_field_std"]) + float(pv_stats["pv_field_mean"])


def make_residual_3d_unet_z(future_frames: int, in_channels: int, future_channels: int, delta_scale: float):
    import torch
    import torch.nn as nn

    class Block3D(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class Block2D(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.future_frames = int(future_frames)
            self.delta_scale = float(delta_scale)
            self.enc1 = Block3D(in_channels, 32)
            self.down1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
            self.enc2 = Block3D(64, 64)
            self.down2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
            self.enc3 = Block3D(128, 128)
            self.down3 = nn.Conv3d(128, 160, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
            self.mid = Block3D(160, 160)
            self.proj1 = nn.Conv2d(32, 32, 1)
            self.proj2 = nn.Conv2d(64, 64, 1)
            self.proj3 = nn.Conv2d(128, 128, 1)
            self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
            self.dec3 = Block2D(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = Block2D(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = Block2D(64, 32)
            self.delta_head = nn.Sequential(
                nn.Conv2d(32 + future_channels, 48, 3, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
                nn.Conv2d(48, 1, 1),
                nn.Tanh(),
            )

        @staticmethod
        def collapse(x):
            return 0.7 * x[:, :, -1] + 0.3 * x.mean(dim=2)

        @staticmethod
        def _match(x, ref):
            if x.shape[-2:] != ref.shape[-2:]:
                return torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            return x

        def forward(self, x, future_feat):
            # x[:, 0] is z-scored PV_norm field. Do not clamp in z-space.
            last = x[:, 0, -1]
            e1_3d = self.enc1(x)
            e2_3d = self.enc2(self.down1(e1_3d))
            e3_3d = self.enc3(self.down2(e2_3d))
            mid_3d = self.mid(self.down3(e3_3d))
            e1 = self.proj1(self.collapse(e1_3d))
            e2 = self.proj2(self.collapse(e2_3d))
            e3 = self.proj3(self.collapse(e3_3d))
            mid = self.collapse(mid_3d)
            d3 = self._match(self.up3(mid), e3)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self._match(self.up2(d3), e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self._match(self.up1(d2), e1)
            feat = self.dec1(torch.cat([d1, e1], dim=1))
            preds = []
            for h in range(self.future_frames):
                delta = self.delta_head(torch.cat([feat, future_feat[:, h]], dim=1)).squeeze(1)
                preds.append(last + self.delta_scale * delta)
            return torch.stack(preds, dim=1)

    return Model()


class TargetExcludedPVNormFieldDataset:
    def __init__(
        self,
        time_df: pd.DataFrame,
        pv_cache: ReadOnlyDiskFrameCache,
        ms_cache: dict,
        bounds: tuple[float, float, float, float],
        pv_stats: dict,
        ms_stats: dict,
        model_name: str,
        args: argparse.Namespace,
    ):
        self.time_df = time_df.reset_index(drop=True)
        self.pv_cache = pv_cache
        self.ms_cache = ms_cache
        self.pv_stats = pv_stats
        self.ms_stats = ms_stats
        self.model_name = str(model_name)
        self.spec = MODEL_SPECS[model_name]
        self.args = args
        self.global_issue_index = {int(v): i for i, v in enumerate(ms_cache["time_df"]["issue_sec"].astype(np.int64).tolist())}
        lon_grid, lat_grid = _grid_lonlat(bounds, int(args.grid_size))
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.feature_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.time_df)

    def _features(self, ts: pd.Timestamp) -> np.ndarray:
        key = _timestamp_key(ts)
        cached = self.feature_cache.get(key)
        if cached is None:
            cached = _time_solar_features(ts, self.lon_grid, self.lat_grid)
            self.feature_cache[key] = cached
        return cached

    def source_dims(self) -> tuple[int, int]:
        in_channels = 5 + (4 if self.spec["use_h8"] else 0)
        future_channels = 3 + (4 if self.spec["use_nwp"] else 0)
        return in_channels, future_channels

    def __getitem__(self, idx: int):
        row = self.time_df.iloc[idx]
        issue_idx = self.global_issue_index[int(row["issue_sec"])]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])

        h8_extra = None
        if self.spec["use_h8"]:
            h8_extra = _normalize_h8(np.asarray(self.ms_cache["h8"][issue_idx], dtype=np.float32), self.ms_stats)

        nwp_extra = None
        if self.spec["use_nwp"]:
            nwp_extra = _normalize_nwp(
                np.asarray(self.ms_cache["nwp"][issue_idx], dtype=np.float32),
                np.asarray(self.ms_cache["nwp_valid"][issue_idx], dtype=np.uint8),
                self.ms_stats,
                int(self.args.grid_size),
            )

        x_list = []
        for ts in past_times:
            field, mask = self.pv_cache.get(ts)
            field_z = _normalize_field(field, mask, self.pv_stats)
            channels = [field_z[None], mask[None], self._features(ts)]
            if h8_extra is not None:
                channels.append(h8_extra)
            x_list.append(np.concatenate(channels, axis=0))

        y_list, m_list, future_list = [], [], []
        for h, ts in enumerate(future_times):
            field, mask = self.pv_cache.get(ts)
            field_z = _normalize_field(field, mask, self.pv_stats)
            future_channels = [self._features(ts)]
            if nwp_extra is not None:
                future_channels.append(nwp_extra[h])
            y_list.append(field_z)
            m_list.append(mask)
            future_list.append(np.concatenate(future_channels, axis=0))

        return (
            np.stack(x_list, axis=1).astype(np.float32),
            np.stack(future_list, axis=0).astype(np.float32),
            np.stack(y_list, axis=0).astype(np.float32),
            np.stack(m_list, axis=0).astype(np.float32),
        )


def masked_huber(pred, target, mask) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    loss = F.smooth_l1_loss(pred, target, reduction="none")
    valid = mask.float()
    return (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)


def weighted_masked_huber_z(pred, target, mask, x, pv_stats: dict, threshold: float, big_weight: float):
    import torch
    import torch.nn.functional as F

    mean = float(pv_stats["pv_field_mean"])
    std = max(float(pv_stats["pv_field_std"]), 1e-6)
    last_raw = x[:, 0, -1] * std + mean
    target_raw = target * std + mean
    true_delta = torch.abs(target_raw - last_raw[:, None])
    weights = 1.0 + float(big_weight) * (true_delta >= float(threshold)).float()
    valid = mask.float()
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights * valid
    return loss.sum() / torch.clamp((weights * valid).sum(), min=1.0)


def run_epoch(model, loader, optimizer, device: str, train: bool, args: argparse.Namespace, pv_stats: dict) -> float:
    import torch

    model.train(train)
    losses = []
    for step, (x, future_feat, y, m) in enumerate(loader, start=1):
        x = x.to(device)
        future_feat = future_feat.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x, future_feat)
            if train and float(args.big_change_loss_weight) > 0:
                loss = weighted_masked_huber_z(
                    pred,
                    y,
                    m,
                    x,
                    pv_stats,
                    threshold=float(args.large_delta_threshold),
                    big_weight=float(args.big_change_loss_weight),
                )
            else:
                loss = masked_huber(pred, y, m)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[pvnorm_field][{mode}] batches={step}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _masked_ssim_2d(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(truth)
    if int(valid.sum()) < 2:
        return float("nan")
    x = np.asarray(pred[valid], dtype=np.float64)
    y = np.asarray(truth[valid], dtype=np.float64)
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(np.mean((x - mu_x) ** 2))
    var_y = float(np.mean((y - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    denom = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denom <= 0.0:
        return float("nan")
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denom)


def _masked_ssim_stack(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, horizon: int | None = None) -> tuple[float, int]:
    vals = []
    if horizon is None:
        horizons = range(pred.shape[1])
    else:
        horizons = [int(horizon)]
    for b in range(pred.shape[0]):
        for h in horizons:
            ssim = _masked_ssim_2d(pred[b, h], truth[b, h], mask[b, h], data_range=1.0)
            if np.isfinite(ssim):
                vals.append(ssim)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), int(len(vals))


def _metric_rows(model_name: str, split: str, pred_z: np.ndarray, y_z: np.ndarray, m: np.ndarray, x_z: np.ndarray, pv_stats: dict, threshold: float) -> list[dict]:
    pred = _denorm_field(pred_z, pv_stats)
    y = _denorm_field(y_z, pv_stats)
    last = _denorm_field(x_z[:, 0, -1], pv_stats)
    mask = m > 0.5
    if not mask.any():
        return []
    last_seq = np.repeat(last[:, None], pred.shape[1], axis=1)
    err = pred - y
    pred_delta = pred - last_seq
    true_delta = y - last_seq
    rows = []
    for label, h in [("all", None)] + [(f"h{i + 1}", i) for i in range(pred.shape[1])]:
        mh = mask if h is None else mask[:, h]
        e = err[mask] if h is None else err[:, h][mh]
        yy = y[mask] if h is None else y[:, h][mh]
        pdlt = pred_delta[mask] if h is None else pred_delta[:, h][mh]
        tdlt = true_delta[mask] if h is None else true_delta[:, h][mh]
        if e.size == 0:
            continue
        ssim, ssim_count = _masked_ssim_stack(pred, y, mask, h)
        large = np.abs(tdlt) >= float(threshold)
        sign_mask = np.abs(tdlt) > 1e-6
        rows.append(
            {
                "model": model_name,
                "eval_split": split,
                "horizon": label,
                "n_points": int(e.size),
                "sum_err_pv_norm_field": float(np.sum(e, dtype=np.float64)),
                "sum_sq_err_pv_norm_field": float(np.sum(e * e, dtype=np.float64)),
                "sum_true_pv_norm_field": float(np.sum(yy, dtype=np.float64)),
                "sum_sq_true_pv_norm_field": float(np.sum(yy * yy, dtype=np.float64)),
                "mae_pv_norm_field": float(np.mean(np.abs(e))),
                "rmse_pv_norm_field": float(np.sqrt(np.mean(e * e))),
                "bias_pv_norm_field": float(np.mean(e)),
                "ssim_pv_norm_field": ssim,
                "ssim_frame_count": ssim_count,
                "delta_mae": float(np.mean(np.abs(pdlt - tdlt))),
                "delta_corr": _corrcoef(pdlt, tdlt),
                "delta_sign_accuracy": float(np.mean(np.sign(pdlt[sign_mask]) == np.sign(tdlt[sign_mask]))) if sign_mask.any() else float("nan"),
                "large_delta_threshold": float(threshold),
                "large_delta_n_points": int(large.sum()),
                "large_delta_mae": float(np.mean(np.abs(e[large]))) if large.any() else float("nan"),
            }
        )
    step_mask = mask[:, 1:] & mask[:, :-1]
    if step_mask.any():
        pred_step = np.abs(pred[:, 1:] - pred[:, :-1])[step_mask]
        true_step = np.abs(y[:, 1:] - y[:, :-1])[step_mask]
        rows.append(
            {
                "model": model_name,
                "eval_split": split,
                "horizon": "horizon_diversity",
                "n_points": int(step_mask.sum()),
                "mae_pv_norm_field": float("nan"),
                "rmse_pv_norm_field": float("nan"),
                "bias_pv_norm_field": float("nan"),
                "r2_pv_norm_field": float("nan"),
                "ssim_pv_norm_field": float("nan"),
                "ssim_frame_count": 0,
                "delta_mae": float("nan"),
                "delta_corr": float("nan"),
                "delta_sign_accuracy": float("nan"),
                "large_delta_threshold": float(threshold),
                "large_delta_n_points": 0,
                "large_delta_mae": float("nan"),
                "pred_horizon_diversity": float(np.mean(pred_step)),
                "truth_horizon_diversity": float(np.mean(true_step)),
                "horizon_diversity_ratio": float(np.mean(pred_step) / max(float(np.mean(true_step)), 1e-8)),
            }
        )
    return rows


def _combine_metrics(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    weighted_cols = ["mae_pv_norm_field", "rmse_pv_norm_field", "bias_pv_norm_field", "delta_mae", "delta_sign_accuracy", "large_delta_mae"]
    for keys, g in df.groupby(["model", "eval_split", "horizon"], sort=True):
        n = g["n_points"].to_numpy(np.float64)
        item = {"model": keys[0], "eval_split": keys[1], "horizon": keys[2], "n_points": int(n.sum())}
        for col in weighted_cols:
            vals = g[col].to_numpy(np.float64)
            ok = np.isfinite(vals)
            if not ok.any():
                item[col] = float("nan")
            elif col == "rmse_pv_norm_field":
                item[col] = float(math.sqrt(np.sum((vals[ok] ** 2) * n[ok]) / max(float(n[ok].sum()), 1.0)))
            elif col == "large_delta_mae":
                large_n = g["large_delta_n_points"].to_numpy(np.float64)
                item[col] = float(np.sum(vals[ok] * large_n[ok]) / max(float(large_n[ok].sum()), 1.0))
            else:
                item[col] = float(np.sum(vals[ok] * n[ok]) / max(float(n[ok].sum()), 1.0))
        corr = g["delta_corr"].to_numpy(np.float64)
        item["delta_corr_median"] = float(np.nanmedian(corr)) if np.isfinite(corr).any() else float("nan")
        if {"sum_sq_err_pv_norm_field", "sum_true_pv_norm_field", "sum_sq_true_pv_norm_field"}.issubset(g.columns):
            total_n = float(n.sum())
            ss_res = float(np.nansum(g["sum_sq_err_pv_norm_field"].to_numpy(np.float64)))
            true_sum = float(np.nansum(g["sum_true_pv_norm_field"].to_numpy(np.float64)))
            true_sq_sum = float(np.nansum(g["sum_sq_true_pv_norm_field"].to_numpy(np.float64)))
            ss_tot = true_sq_sum - true_sum * true_sum / max(total_n, 1.0)
            item["r2_pv_norm_field"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-12 else float("nan")
        if "ssim_pv_norm_field" in g.columns:
            ssim_vals = g["ssim_pv_norm_field"].to_numpy(np.float64)
            ssim_n = g.get("ssim_frame_count", pd.Series(np.ones(len(g), dtype=np.float64))).to_numpy(np.float64)
            ok = np.isfinite(ssim_vals) & np.isfinite(ssim_n) & (ssim_n > 0)
            item["ssim_pv_norm_field"] = float(np.sum(ssim_vals[ok] * ssim_n[ok]) / max(float(ssim_n[ok].sum()), 1.0)) if ok.any() else float("nan")
            item["ssim_frame_count"] = int(np.nansum(ssim_n[ok])) if ok.any() else 0
        item["large_delta_threshold"] = float(g["large_delta_threshold"].iloc[0])
        item["large_delta_n_points"] = int(np.nansum(g["large_delta_n_points"].to_numpy(np.float64)))
        for col in ["pred_horizon_diversity", "truth_horizon_diversity", "horizon_diversity_ratio"]:
            if col in g.columns:
                vals = g[col].to_numpy(np.float64)
                ok = np.isfinite(vals)
                item[col] = float(np.sum(vals[ok] * n[ok]) / max(float(n[ok].sum()), 1.0)) if ok.any() else float("nan")
        out.append(item)
    return pd.DataFrame(out).sort_values(["eval_split", "horizon", "mae_pv_norm_field", "model"]).reset_index(drop=True)


def evaluate_model(model, loader, device: str, model_name: str, split: str, pv_stats: dict, threshold: float) -> pd.DataFrame:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for step, (x, future_feat, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device), future_feat.to(device)).detach().cpu().numpy()
            rows.extend(_metric_rows(model_name, split, pred, y.numpy(), m.numpy(), x.numpy(), pv_stats, threshold))
            if step % 100 == 0:
                print(f"[pvnorm_field][{model_name}][{split}] eval batches={step}/{len(loader)}", flush=True)
    return _combine_metrics(rows)


def _robust_limits(values: list[np.ndarray], masks: list[np.ndarray]) -> tuple[float, float]:
    chunks = []
    for val, mask in zip(values, masks):
        val = np.asarray(val)
        mask = np.asarray(mask)
        if np.any(mask > 0.5):
            chunks.append(val[mask > 0.5])
    if not chunks:
        return 0.0, 1.0
    arr = np.concatenate(chunks)
    lo = float(np.nanquantile(arr, 0.02))
    hi = float(np.nanquantile(arr, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def save_sample_comparisons(models: dict, datasets: dict, bounds: tuple[float, float, float, float], device: str, args: argparse.Namespace, out_dir: Path, pv_stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sample_dir = ensure_dir(out_dir / "sample_pvnorm_field_comparison")
    if not datasets:
        return
    ref_name = next(iter(datasets))
    test_ds = datasets[ref_name]["test"]
    n = min(int(args.sample_count), len(test_ds))
    if n <= 0:
        return
    horizons = _parse_visual_horizons(str(args.visual_horizons), int(args.future_frames))
    indices = np.linspace(0, len(test_ds) - 1, n, dtype=int)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    rows = []
    for sample_id, idx in enumerate(indices):
        row = test_ds.time_df.iloc[int(idx)]
        issue = pd.Timestamp(row["issue_time"])
        preds: dict[str, np.ndarray] = {}
        truth = None
        mask = None
        latest = None
        latest_mask = None
        for name, model in models.items():
            ds = datasets[name]["test"]
            x, fut, y, m = ds[int(idx)]
            if truth is None:
                truth = _denorm_field(y, pv_stats)
                mask = m
                latest = _denorm_field(x[0, -1], pv_stats)
                latest_mask = x[1, -1]
            with torch.no_grad():
                pred_z = model(torch.from_numpy(x[None]).to(device), torch.from_numpy(fut[None]).to(device)).detach().cpu().numpy()[0]
            preds[name] = _denorm_field(pred_z, pv_stats)
        values = [truth[h] for h in horizons] + [preds[name][h] for name in models for h in horizons]
        masks = [mask[h] for _ in range(1 + len(models)) for h in horizons]
        vmin, vmax = _robust_limits(values, masks)
        fig, axes = plt.subplots(len(horizons), 2 + len(models), figsize=(3.25 * (2 + len(models)), 3.0 * len(horizons)), squeeze=False)
        for r, h in enumerate(horizons):
            future_time = issue + pd.Timedelta(minutes=int(args.freq_minutes) * (h + 1))
            panels = [("latest", latest, latest_mask), (f"truth h{h + 1}", truth[h], mask[h])]
            panels += [(MODEL_SPECS[name]["label"], preds[name][h], mask[h]) for name in models]
            for c, (title, arr, mm) in enumerate(panels):
                shown = _masked_for_plot(arr, mm)
                im = axes[r, c].imshow(shown, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
                axes[r, c].set_title(f"{title}\n{future_time:%Y-%m-%d %H:%M}" if c > 0 else f"{title}\nissue {issue:%H:%M}", fontsize=9)
                axes[r, c].set_xlabel("lon")
                axes[r, c].set_ylabel("lat")
                fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
        fig.suptitle(f"Target-excluded PV_norm field forecast | issue={issue:%Y-%m-%d %H:%M:%S}", y=0.995)
        fig.tight_layout()
        path = sample_dir / f"pvnorm_field_compare_{sample_id:03d}_{issue:%Y%m%d_%H%M%S}.png"
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        rows.append({"sample_id": int(sample_id), "dataset_index": int(idx), "issue_time": str(issue), "path": str(path)})
        print(f"[pvnorm_field] saved sample {path}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "sample_pvnorm_field_comparison.csv", index=False)


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
        raise RuntimeError("PyTorch is required for PV_norm field model training.") from exc

    requested = [x.strip() for x in str(args.models).split(",") if x.strip()]
    bad = [x for x in requested if x not in MODEL_SPECS]
    if bad:
        raise ValueError(f"unsupported models: {bad}; valid={sorted(MODEL_SPECS)}")

    split_df = _load_station_split(args)
    bounds = _bounds(split_df)
    ms_cache = _load_multisource_cache(Path(args.multisource_cache_dir))
    time_df = ms_cache["time_df"].copy()
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    _assert_issue_time_alignment(args, time_df)
    _assert_target_exclusion(args)
    time_df.to_csv(out_dir / "pvnorm_field_issue_time_index.csv", index=False)

    pv_cache = ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    pv_stats = _compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = _compute_or_load_multisource_stats(ms_cache, out_dir, time_df)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    target_stations = pd.read_csv(target_path, dtype={"station_sn": str}) if target_path.exists() else pd.DataFrame()
    if not target_stations.empty:
        target_stations.to_csv(out_dir / "target_stations_used_for_exclusion.csv", index=False)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    datasets, loaders, models = {}, {}, {}
    train_logs, all_eval = [], []
    for model_name in requested:
        datasets[model_name] = {}
        loaders[model_name] = {}
        for split in ["train", "val", "test"]:
            tdf = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
            ds = TargetExcludedPVNormFieldDataset(tdf, pv_cache, ms_cache, bounds, pv_stats, ms_stats, model_name, args)
            datasets[model_name][split] = ds
            loaders[model_name][split] = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
            )
        in_channels, future_channels = datasets[model_name]["train"].source_dims()
        print(
            f"==== PV_norm field model: {model_name} ({MODEL_SPECS[model_name]['label']}) "
            f"in_channels={in_channels} future_channels={future_channels} ====",
            flush=True,
        )
        model = make_residual_3d_unet_z(
            int(args.future_frames),
            in_channels=in_channels,
            future_channels=future_channels,
            delta_scale=float(args.delta_scale),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_factor),
            patience=int(args.lr_patience),
            min_lr=1e-5,
        )
        best_val = float("inf")
        bad_count = 0
        ckpt_path = out_dir / f"{model_name}_best.pt"
        for epoch in range(1, int(args.epochs) + 1):
            tr = run_epoch(model, loaders[model_name]["train"], optimizer, device, train=True, args=args, pv_stats=pv_stats)
            va = run_epoch(model, loaders[model_name]["val"], optimizer, device, train=False, args=args, pv_stats=pv_stats)
            scheduler.step(va)
            lr_now = float(optimizer.param_groups[0]["lr"])
            train_logs.append({"model": model_name, "epoch": int(epoch), "train_loss": tr, "val_loss": va, "lr": lr_now})
            print(f"[pvnorm_field][{model_name}] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f} lr={lr_now:.2e}", flush=True)
            if va < best_val - float(args.early_stop_min_delta):
                best_val = va
                bad_count = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "model_name": model_name,
                        "in_channels": in_channels,
                        "future_channels": future_channels,
                        "pv_field_norm_stats": pv_stats,
                        "multisource_norm_stats": ms_stats,
                    },
                    ckpt_path,
                )
            else:
                bad_count += 1
                if bad_count >= int(args.patience):
                    print(f"[pvnorm_field][{model_name}] early stop at epoch={epoch} best_val={best_val:.6f}", flush=True)
                    break
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        models[model_name] = model.eval()
        for split in ["val", "test"]:
            eval_df = evaluate_model(
                model,
                loaders[model_name][split],
                device,
                model_name,
                split,
                pv_stats,
                threshold=float(args.large_delta_threshold),
            )
            all_eval.append(eval_df)
            print(f"==== {model_name} {split} ====")
            if not eval_df.empty:
                print(eval_df[eval_df["horizon"].isin(["all", "h6", "h7", "h8"])].to_string(index=False))

    pd.DataFrame(train_logs).to_csv(out_dir / "pvnorm_field_train_log.csv", index=False)
    summary = pd.concat(all_eval, ignore_index=True).sort_values(["eval_split", "horizon", "mae_pv_norm_field", "model"]).reset_index(drop=True) if all_eval else pd.DataFrame()
    summary.to_csv(out_dir / "pvnorm_field_eval_summary.csv", index=False)
    save_sample_comparisons(models, datasets, bounds, device, args, out_dir, pv_stats)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Target-excluded PV_norm field forecasting for station field-gain experiments.",
            "field_value": "Read from target-excluded frame cache; interpreted as PV_norm field value and z-scored during training.",
            "pv_field_standardization": pv_stats,
            "target_stations_excluded": target_stations["station_sn"].astype(str).tolist() if not target_stations.empty else [],
            "model_framework": (
                "Residual 3D U-Net V1 style encoder over past 12 PV_norm field frames, support masks, time/solar, "
                "and optional geo-resampled H8; future time/solar and optional NWP condition each horizon. "
                "Output is z-scored future PV_norm field without z-space clipping."
            ),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print("==== Target-excluded PV_norm field aggregate summary ====")
    print(summary[summary["horizon"].isin(["all", "h6", "h7", "h8", "horizon_diversity"])].to_string(index=False) if not summary.empty else "EMPTY")
    print(f"saved: {out_dir / 'pvnorm_field_eval_summary.csv'}")
    print(f"figures: {out_dir / 'sample_pvnorm_field_comparison'}")


if __name__ == "__main__":
    main()
