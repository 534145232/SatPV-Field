#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_fullfield_dense_temporal_forecast import (
    baseline_predictions,
    build_issue_time_table,
    collect_all_values,
    _combine_metric_rows,
    _load_split_df,
    _parse_time_list,
)
from train_regional_dense_reconstruction import masked_huber, splat_values
from visualize_regional_split_frames import _bounds, _geo_solar_elevation_scale, ensure_dir, estimate_kpi, load_station_series, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults",
    "train_pv_dir": "/data/DayinChen/NWP/PV/train_directory_cleaned_csv",
    "val_pv_dir": "/data/DayinChen/NWP/PV/val_directory_cleaned_csv",
    "out_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_dense_temporal_residual_unet_seed2026",
    "frame_cache_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_frame_cache_64_seed2026",
    "time_start": "2022-01-01 07:00:00",
    "time_end": "2023-06-24 16:45:00",
    "train_end": "2023-01-13 23:59:59",
    "val_end": "2023-03-08 23:59:59",
    "freq_minutes": 15,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "gaussian_sigma_cells": 1.0,
    "gaussian_radius_cells": 2,
    "min_grid_weight": 1e-6,
    "clearsky_eps": 0.02,
    "min_clear_sky": 0.03,
    "csi_clip": 2.0,
    "capacity_scale": 1000.0,
    "kpi_quantile": 0.95,
    "kpi_min_samples": 40,
    "kpi_max_samples_per_station": 2000,
    "delta_scale": 0.6,
    "batch_size": 10,
    "epochs": 35,
    "patience": 7,
    "lr": 1.2e-3,
    "weight_decay": 1e-4,
    "seed": 2026,
    "device": "cuda",
    "num_workers": 0,
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
            "Upgraded full-field DensePV temporal model: disk frame cache, support/time/solar channels, "
            "lightweight 3D U-Net encoder-decoder, and residual future prediction."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--frame_cache_dir", type=Path, default=Path(PYCHARM_DEFAULTS["frame_cache_dir"]))
    parser.add_argument("--time_start", type=str, default=str(PYCHARM_DEFAULTS["time_start"]))
    parser.add_argument("--time_end", type=str, default=str(PYCHARM_DEFAULTS["time_end"]))
    parser.add_argument("--train_end", type=str, default=str(PYCHARM_DEFAULTS["train_end"]))
    parser.add_argument("--val_end", type=str, default=str(PYCHARM_DEFAULTS["val_end"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
    parser.add_argument("--past_frames", type=int, default=int(PYCHARM_DEFAULTS["past_frames"]))
    parser.add_argument("--future_frames", type=int, default=int(PYCHARM_DEFAULTS["future_frames"]))
    parser.add_argument("--grid_size", type=int, default=int(PYCHARM_DEFAULTS["grid_size"]))
    parser.add_argument("--gaussian_sigma_cells", type=float, default=float(PYCHARM_DEFAULTS["gaussian_sigma_cells"]))
    parser.add_argument("--gaussian_radius_cells", type=int, default=int(PYCHARM_DEFAULTS["gaussian_radius_cells"]))
    parser.add_argument("--min_grid_weight", type=float, default=float(PYCHARM_DEFAULTS["min_grid_weight"]))
    parser.add_argument("--clearsky_eps", type=float, default=float(PYCHARM_DEFAULTS["clearsky_eps"]))
    parser.add_argument("--min_clear_sky", type=float, default=float(PYCHARM_DEFAULTS["min_clear_sky"]))
    parser.add_argument("--csi_clip", type=float, default=float(PYCHARM_DEFAULTS["csi_clip"]))
    parser.add_argument("--capacity_scale", type=float, default=float(PYCHARM_DEFAULTS["capacity_scale"]))
    parser.add_argument("--kpi_quantile", type=float, default=float(PYCHARM_DEFAULTS["kpi_quantile"]))
    parser.add_argument("--kpi_min_samples", type=int, default=int(PYCHARM_DEFAULTS["kpi_min_samples"]))
    parser.add_argument("--kpi_max_samples_per_station", type=int, default=int(PYCHARM_DEFAULTS["kpi_max_samples_per_station"]))
    parser.add_argument("--delta_scale", type=float, default=float(PYCHARM_DEFAULTS["delta_scale"]))
    parser.add_argument("--batch_size", type=int, default=int(PYCHARM_DEFAULTS["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(PYCHARM_DEFAULTS["epochs"]))
    parser.add_argument("--patience", type=int, default=int(PYCHARM_DEFAULTS["patience"]))
    parser.add_argument("--lr", type=float, default=float(PYCHARM_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(PYCHARM_DEFAULTS["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--device", type=str, default=str(PYCHARM_DEFAULTS["device"]))
    parser.add_argument("--num_workers", type=int, default=int(PYCHARM_DEFAULTS["num_workers"]))
    return parser.parse_args()


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _unique_timestamps(time_df: pd.DataFrame) -> list[pd.Timestamp]:
    vals: set[int] = set()
    for _, row in time_df.iterrows():
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            vals.add(_timestamp_key(ts))
    return [pd.Timestamp(v, unit="s") for v in sorted(vals)]


def _grid_lonlat(bounds: tuple[float, float, float, float], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bounds
    xs = lon_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lon_max - lon_min)
    ys = lat_min + (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * (lat_max - lat_min)
    return np.meshgrid(xs, ys)


def _time_solar_features(ts: pd.Timestamp, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    ts = pd.Timestamp(ts)
    minute = float(ts.hour * 60 + ts.minute)
    phase = 2.0 * math.pi * minute / (24.0 * 60.0)
    tod_sin = np.full(lon_grid.shape, math.sin(phase), dtype=np.float32)
    tod_cos = np.full(lon_grid.shape, math.cos(phase), dtype=np.float32)
    day = int(ts.dayofyear)
    hour = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    time_offset = eq_time + 4.0 * lon_grid - 60.0 * 8.0
    true_solar_min = hour * 60.0 + time_offset
    hour_angle = np.deg2rad(true_solar_min / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat_grid)
    sin_elev = np.sin(lat_rad) * math.sin(decl) + np.cos(lat_rad) * math.cos(decl) * np.cos(hour_angle)
    solar = np.clip(sin_elev, 0.0, 1.2).astype(np.float32)
    return np.stack([tod_sin, tod_cos, solar], axis=0).astype(np.float32)


class FullFieldDiskFrameCache:
    def __init__(
        self,
        cache_dir: Path,
        split_df: pd.DataFrame,
        series: dict[str, pd.DataFrame],
        kpi: dict[str, float],
        bounds: tuple[float, float, float, float],
        time_df: pd.DataFrame,
        args: argparse.Namespace,
    ):
        self.cache_dir = ensure_dir(cache_dir)
        self.split_df = split_df
        self.series = series
        self.kpi = kpi
        self.bounds = bounds
        self.args = args
        self.grid_size = int(args.grid_size)
        self.times = _unique_timestamps(time_df)
        self.sec = np.asarray([_timestamp_key(t) for t in self.times], dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.sec)}
        self.field_path = self.cache_dir / "fields_float16.npy"
        self.mask_path = self.cache_dir / "masks_uint8.npy"
        self.sec_path = self.cache_dir / "timestamp_sec.npy"
        self.meta_path = self.cache_dir / "frame_cache_meta.json"
        self.fields = None
        self.masks = None

    def ready(self) -> bool:
        if not (self.field_path.exists() and self.mask_path.exists() and self.sec_path.exists()):
            return False
        if not self.meta_path.exists():
            return False
        try:
            import json

            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        expected = {
            "grid_size": int(self.grid_size),
            "gaussian_sigma_cells": float(self.args.gaussian_sigma_cells),
            "gaussian_radius_cells": int(self.args.gaussian_radius_cells),
            "min_grid_weight": float(self.args.min_grid_weight),
        }
        for key, value in expected.items():
            if key not in meta:
                return False
            if isinstance(value, float):
                if abs(float(meta[key]) - value) > 1e-9:
                    return False
            elif int(meta[key]) != int(value):
                return False
        old_sec = np.load(self.sec_path, mmap_mode="r")
        return old_sec.shape == self.sec.shape and np.array_equal(np.asarray(old_sec), self.sec)

    def build_or_load(self) -> None:
        if self.ready():
            print(f"[residual_unet] reuse frame cache: {self.cache_dir}", flush=True)
        else:
            print(f"[residual_unet] building frame cache: {self.cache_dir}", flush=True)
            fields = np.lib.format.open_memmap(self.field_path, mode="w+", dtype=np.float16, shape=(len(self.times), self.grid_size, self.grid_size))
            masks = np.lib.format.open_memmap(self.mask_path, mode="w+", dtype=np.uint8, shape=(len(self.times), self.grid_size, self.grid_size))
            for i, ts in enumerate(self.times):
                vals = collect_all_values(self.split_df, ts, self.series, self.kpi, self.args)
                field, support = splat_values(vals, self.split_df, self.bounds, self.args)
                mask = support > float(self.args.min_grid_weight)
                fields[i] = field.astype(np.float16)
                masks[i] = mask.astype(np.uint8)
                if (i + 1) % 500 == 0:
                    print(f"[residual_unet] cached {i + 1}/{len(self.times)} frames latest={ts}", flush=True)
            fields.flush()
            masks.flush()
            np.save(self.sec_path, self.sec)
            write_json(
                self.meta_path,
                {
                    "n_frames": int(len(self.times)),
                    "grid_size": int(self.grid_size),
                    "gaussian_sigma_cells": float(self.args.gaussian_sigma_cells),
                    "gaussian_radius_cells": int(self.args.gaussian_radius_cells),
                    "min_grid_weight": float(self.args.min_grid_weight),
                    "dtype_field": "float16",
                    "dtype_mask": "uint8",
                    "value": "KPI-adjusted CSI field and valid support mask",
                },
            )
        self.fields = np.load(self.field_path, mmap_mode="r")
        self.masks = np.load(self.mask_path, mmap_mode="r")

    def get(self, ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        idx = self.index[_timestamp_key(ts)]
        field = np.asarray(self.fields[idx], dtype=np.float32)
        mask = np.asarray(self.masks[idx], dtype=np.float32)
        return field, mask


class FullFieldResidualDataset:
    def __init__(
        self,
        time_df: pd.DataFrame,
        frame_cache: FullFieldDiskFrameCache,
        bounds: tuple[float, float, float, float],
        args: argparse.Namespace,
    ):
        self.time_df = time_df.reset_index(drop=True)
        self.cache = frame_cache
        self.args = args
        lon_grid, lat_grid = _grid_lonlat(bounds, int(args.grid_size))
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.feature_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.time_df)

    def _features(self, ts: pd.Timestamp) -> np.ndarray:
        key = _timestamp_key(ts)
        item = self.feature_cache.get(key)
        if item is None:
            item = _time_solar_features(ts, self.lon_grid, self.lat_grid)
            self.feature_cache[key] = item
        return item

    def __getitem__(self, idx: int):
        row = self.time_df.iloc[idx]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])
        x_list = []
        for ts in past_times:
            field, mask = self.cache.get(ts)
            feat = self._features(ts)
            x_list.append(np.concatenate([field[None], mask[None], feat], axis=0))
        y_list, m_list, fut_feat = [], [], []
        for ts in future_times:
            field, mask = self.cache.get(ts)
            y_list.append(field)
            m_list.append(mask)
            fut_feat.append(self._features(ts))
        x = np.stack(x_list, axis=1).astype(np.float32)  # [C=5, T, H, W]
        y = np.stack(y_list, axis=0).astype(np.float32)
        m = np.stack(m_list, axis=0).astype(np.float32)
        fut = np.stack(fut_feat, axis=0).astype(np.float32)  # [Hf, 3, H, W]
        return x, fut, y, m


class Residual3DUNet:
    def __new__(cls, future_frames: int, in_channels: int = 5, future_channels: int = 3, delta_scale: float = 0.6):
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

            def _match(self, x, ref):
                if x.shape[-2:] != ref.shape[-2:]:
                    return torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
                return x

            def forward(self, x, future_feat):
                # x: [B, 5, T, H, W], future_feat: [B, Hf, 3, H, W]
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
                    preds.append(torch.clamp(last + self.delta_scale * delta, 0.0, 2.0))
                return torch.stack(preds, dim=1)

        return Model()


def _accumulate(rows: list[dict], model_name: str, split: str, pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> None:
    mask = m > 0.5
    if not mask.any():
        return
    diff = pred - y
    rows.append(
        {
            "model": model_name,
            "eval_split": split,
            "horizon": "all",
            "n_points": int(mask.sum()),
            "mae_field": float(np.abs(diff[mask]).mean()),
            "rmse_field": float(np.sqrt((diff[mask] ** 2).mean())),
        }
    )
    for h in range(pred.shape[1]):
        mh = mask[:, h]
        if not mh.any():
            continue
        dh = diff[:, h][mh]
        rows.append(
            {
                "model": model_name,
                "eval_split": split,
                "horizon": f"h{h + 1}",
                "n_points": int(mh.sum()),
                "mae_field": float(np.abs(dh).mean()),
                "rmse_field": float(np.sqrt((dh**2).mean())),
            }
        )


def _combine(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    out = []
    for keys, g in work.groupby(["model", "eval_split", "horizon"], sort=True):
        n = g["n_points"].to_numpy(np.float64)
        total = float(n.sum())
        out.append(
            {
                "model": keys[0],
                "eval_split": keys[1],
                "horizon": keys[2],
                "n_points": int(total),
                "mae_field": float((g["mae_field"].to_numpy(np.float64) * n).sum() / max(1.0, total)),
                "rmse_field": float(math.sqrt(((g["rmse_field"].to_numpy(np.float64) ** 2) * n).sum() / max(1.0, total))),
            }
        )
    return pd.DataFrame(out).sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)


def run_epoch(model, loader, optimizer, device: str, train: bool) -> float:
    import torch

    losses = []
    model.train(train)
    for step, (x, fut, y, m) in enumerate(loader, start=1):
        x = x.to(device)
        fut = fut.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x, fut)
            loss = masked_huber(pred, y, m)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[residual_unet][{mode}] batches={step}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model, loader, device: str, split: str, model_name: str) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for step, (x, fut, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device), fut.to(device)).detach().cpu().numpy()
            _accumulate(rows, model_name, split, pred, y.numpy(), m.numpy())
            if step % 100 == 0:
                print(f"[residual_unet][{split}] eval batches={step}/{len(loader)}", flush=True)
    return _combine(rows)


def evaluate_baselines(loader, split: str, future_frames: int) -> pd.DataFrame:
    rows = []
    for step, (x, _fut, y, m) in enumerate(loader, start=1):
        x_np = x.numpy()
        y_np = y.numpy()
        m_np = m.numpy()
        # baseline_predictions expects [B, 2, T, H, W]; extra channels are ignored by [:, 0].
        for name, pred in baseline_predictions(x_np[:, :2], future_frames).items():
            _accumulate(rows, name, split, pred, y_np, m_np)
        if step % 100 == 0:
            print(f"[residual_unet][baseline][{split}] batches={step}/{len(loader)}", flush=True)
    return _combine(rows)


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
        raise RuntimeError("PyTorch is required for residual 3D U-Net training") from exc

    split_df = _load_split_df(args)
    split_df["split"] = "all"
    series, load_status = load_station_series(split_df, args)
    load_status.to_csv(out_dir / "station_csv_load_status.csv", index=False)
    kpi = estimate_kpi(split_df, series, args)
    pd.DataFrame([{"station_sn": k, "kpi": v} for k, v in sorted(kpi.items())]).to_csv(out_dir / "station_kpi_geo_baselines.csv", index=False)
    bounds = _bounds(split_df)
    time_df = build_issue_time_table(args)
    time_df.to_csv(out_dir / "fullfield_residual_unet_issue_time_index.csv", index=False)

    cache = FullFieldDiskFrameCache(Path(args.frame_cache_dir), split_df, series, kpi, bounds, time_df, args)
    cache.build_or_load()

    datasets = {}
    loaders = {}
    for split in ["train", "val", "test"]:
        tdf = time_df[time_df["time_split"] == split].reset_index(drop=True)
        datasets[split] = FullFieldResidualDataset(tdf, cache, bounds, args)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
        )
        print(f"[residual_unet] {split} samples={len(datasets[split])}", flush=True)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    all_eval = []
    for split in ["val", "test"]:
        base_df = evaluate_baselines(loaders[split], split, int(args.future_frames))
        all_eval.append(base_df)
        print(f"==== baselines {split} ====")
        print(base_df[base_df["horizon"] == "all"].to_string(index=False))

    model = Residual3DUNet(int(args.future_frames), in_channels=5, future_channels=3, delta_scale=float(args.delta_scale)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    bad = 0
    logs = []
    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(model, loaders["train"], optimizer, device, train=True)
        va = run_epoch(model, loaders["val"], optimizer, device, train=False)
        logs.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"[residual_unet] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)
        if va < best_val:
            best_val = va
            bad = 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "model_name": "residual_3d_unet"}, out_dir / "residual_3d_unet_best.pt")
        else:
            bad += 1
            if bad >= int(args.patience):
                break
    pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)

    ckpt = torch.load(out_dir / "residual_3d_unet_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    for split in ["val", "test"]:
        eval_df = evaluate(model, loaders[split], device, split, "residual_3d_unet")
        all_eval.append(eval_df)
        print(f"==== residual_3d_unet {split} ====")
        print(eval_df[eval_df["horizon"] == "all"].to_string(index=False))

    summary = pd.concat(all_eval, ignore_index=True).sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)
    summary.to_csv(out_dir / "fullfield_residual_unet_eval_summary.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "task": "full-field DensePV residual temporal forecasting",
            "model": "lightweight 3D U-Net encoder-decoder with residual future = last_frame + delta",
            "input_channels": ["CSI field", "support mask", "time-of-day sin", "time-of-day cos", "geo solar elevation"],
            "future_condition_channels": ["future time-of-day sin", "future time-of-day cos", "future geo solar elevation"],
            "frame_cache_dir": str(args.frame_cache_dir),
            "frame_cache_n_frames": int(len(cache.times)),
            "window_rule": "all 12 history frames and all 8 future frames must be between 07:00 and 16:45 and within the same calendar day",
        },
    )
    print("==== Full-field residual 3D U-Net aggregate summary ====")
    print(summary[summary["horizon"] == "all"].to_string(index=False))
    print(f"saved: {out_dir / 'fullfield_residual_unet_eval_summary.csv'}")


if __name__ == "__main__":
    main()

