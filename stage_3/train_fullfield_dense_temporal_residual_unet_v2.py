#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stage3_paths as p3

from train_fullfield_dense_temporal_forecast import build_issue_time_table, _load_split_df, _parse_time_list
from train_fullfield_dense_temporal_residual_unet import (
    FullFieldResidualDataset,
    _accumulate,
    _combine,
    _grid_lonlat,
    _time_solar_features,
    _timestamp_key,
)
from visualize_regional_split_frames import _bounds, ensure_dir, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": str(p3.DATA_DIR / "_local_core_defaults"),
    "out_dir": str(p3.DATA_DIR / "_local_core_defaults" / "fullfield_dense_temporal_residual_unet_v2_seed2026"),
    "frame_cache_dir": str(p3.DATA_DIR / "_local_core_defaults" / "fullfield_frame_cache_64_seed2026"),
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
    "delta_scale": 0.6,
    "horizon_embed_dim": 8,
    "grad_loss_weight": 0.1,
    "temporal_loss_weight": 0.1,
    "batch_size": 8,
    "epochs": 35,
    "patience": 7,
    "lr": 1.0e-3,
    "weight_decay": 1e-4,
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
            "Full-field DensePV residual 3D U-Net v2. This version reads the existing disk frame cache, "
            "adds horizon-aware residual decoding, uses temporal attention collapse, and trains with "
            "Huber + spatial-gradient + temporal-difference losses."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
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
    parser.add_argument("--delta_scale", type=float, default=float(PYCHARM_DEFAULTS["delta_scale"]))
    parser.add_argument("--horizon_embed_dim", type=int, default=int(PYCHARM_DEFAULTS["horizon_embed_dim"]))
    parser.add_argument("--grad_loss_weight", type=float, default=float(PYCHARM_DEFAULTS["grad_loss_weight"]))
    parser.add_argument("--temporal_loss_weight", type=float, default=float(PYCHARM_DEFAULTS["temporal_loss_weight"]))
    parser.add_argument("--batch_size", type=int, default=int(PYCHARM_DEFAULTS["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(PYCHARM_DEFAULTS["epochs"]))
    parser.add_argument("--patience", type=int, default=int(PYCHARM_DEFAULTS["patience"]))
    parser.add_argument("--lr", type=float, default=float(PYCHARM_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(PYCHARM_DEFAULTS["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--device", type=str, default=str(PYCHARM_DEFAULTS["device"]))
    parser.add_argument("--num_workers", type=int, default=int(PYCHARM_DEFAULTS["num_workers"]))
    parser.add_argument("--sample_count", type=int, default=int(PYCHARM_DEFAULTS["sample_count"]))
    parser.add_argument("--visual_horizons", type=str, default=str(PYCHARM_DEFAULTS["visual_horizons"]))
    parser.add_argument("--dpi", type=int, default=int(PYCHARM_DEFAULTS["dpi"]))
    return parser.parse_args()


def _unique_timestamps(time_df: pd.DataFrame) -> list[pd.Timestamp]:
    vals: set[int] = set()
    for _, row in time_df.iterrows():
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            vals.add(_timestamp_key(ts))
    return [pd.Timestamp(v, unit="s") for v in sorted(vals)]


class ReadOnlyDiskFrameCache:
    def __init__(self, cache_dir: Path, time_df: pd.DataFrame, args: argparse.Namespace):
        self.cache_dir = Path(cache_dir)
        self.grid_size = int(args.grid_size)
        self.times = _unique_timestamps(time_df)
        self.sec = np.asarray([_timestamp_key(t) for t in self.times], dtype=np.int64)
        self.field_path = self.cache_dir / "fields_float16.npy"
        self.mask_path = self.cache_dir / "masks_uint8.npy"
        self.sec_path = self.cache_dir / "timestamp_sec.npy"
        self.meta_path = self.cache_dir / "frame_cache_meta.json"
        self.fields = None
        self.masks = None
        self.index: dict[int, int] = {}
        self._validate_paths()
        self._validate_meta(args)
        self._load_arrays()
        self._validate_time_coverage()

    def _validate_paths(self) -> None:
        missing = [p for p in [self.field_path, self.mask_path, self.sec_path, self.meta_path] if not p.exists()]
        if missing:
            msg = "\n".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Frame cache is incomplete. Run train_fullfield_dense_temporal_residual_unet.py once "
                f"to build the shared cache first. Missing:\n{msg}"
            )

    def _validate_meta(self, args: argparse.Namespace) -> None:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected = {
            "grid_size": int(args.grid_size),
            "gaussian_sigma_cells": float(args.gaussian_sigma_cells),
            "gaussian_radius_cells": int(args.gaussian_radius_cells),
            "min_grid_weight": float(args.min_grid_weight),
        }
        for key, value in expected.items():
            if key not in meta:
                raise ValueError(f"Frame cache metadata misses key: {key}")
            if isinstance(value, float):
                if abs(float(meta[key]) - value) > 1e-9:
                    raise ValueError(f"Frame cache {key}={meta[key]} does not match requested {value}")
            elif int(meta[key]) != int(value):
                raise ValueError(f"Frame cache {key}={meta[key]} does not match requested {value}")

    def _load_arrays(self) -> None:
        cached_sec = np.load(self.sec_path, mmap_mode="r")
        self.index = {int(s): i for i, s in enumerate(np.asarray(cached_sec))}
        self.fields = np.load(self.field_path, mmap_mode="r")
        self.masks = np.load(self.mask_path, mmap_mode="r")
        if self.fields.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError(f"cache field shape {self.fields.shape} is incompatible with grid_size={self.grid_size}")
        if self.masks.shape != self.fields.shape:
            raise ValueError(f"cache mask shape {self.masks.shape} does not match field shape {self.fields.shape}")

    def _validate_time_coverage(self) -> None:
        missing = [int(s) for s in self.sec if int(s) not in self.index]
        if missing:
            examples = ", ".join(str(pd.Timestamp(v, unit="s")) for v in missing[:5])
            raise ValueError(f"Frame cache misses {len(missing)} required timestamps. Examples: {examples}")

    def get(self, ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        idx = self.index[_timestamp_key(ts)]
        field = np.asarray(self.fields[idx], dtype=np.float32)
        mask = np.asarray(self.masks[idx], dtype=np.float32)
        return field, mask


def make_model(future_frames: int, in_channels: int, future_channels: int, delta_scale: float, horizon_embed_dim: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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

    class TemporalAttentionCollapse(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.score = nn.Conv3d(channels, 1, kernel_size=1)

        def forward(self, x):
            weight = torch.softmax(self.score(x), dim=2)
            return (x * weight).sum(dim=2)

    class Residual3DUNetV2(nn.Module):
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

            self.t1 = TemporalAttentionCollapse(32)
            self.t2 = TemporalAttentionCollapse(64)
            self.t3 = TemporalAttentionCollapse(128)
            self.tm = TemporalAttentionCollapse(160)

            self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
            self.dec3 = Block2D(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = Block2D(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = Block2D(64, 32)

            self.horizon_embed = nn.Embedding(self.future_frames, int(horizon_embed_dim))
            head_in = 32 + future_channels + int(horizon_embed_dim)
            self.delta_head = nn.Sequential(
                nn.Conv2d(head_in, 64, 3, padding=1),
                nn.GroupNorm(4, 64),
                nn.GELU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
                nn.Conv2d(32, 1, 1),
                nn.Tanh(),
            )

        @staticmethod
        def _match(x, ref):
            if x.shape[-2:] != ref.shape[-2:]:
                return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            return x

        def forward(self, x, future_feat):
            # x: [B, 5, T, H, W], future_feat: [B, Hf, 3, H, W]
            b, _c, _t, height, width = x.shape
            last = x[:, 0, -1]

            e1_3d = self.enc1(x)
            e2_3d = self.enc2(self.down1(e1_3d))
            e3_3d = self.enc3(self.down2(e2_3d))
            mid_3d = self.mid(self.down3(e3_3d))

            e1 = self.t1(e1_3d)
            e2 = self.t2(e2_3d)
            e3 = self.t3(e3_3d)
            mid = self.tm(mid_3d)

            d3 = self._match(self.up3(mid), e3)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self._match(self.up2(d3), e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self._match(self.up1(d2), e1)
            feat = self.dec1(torch.cat([d1, e1], dim=1))

            feat_seq = feat[:, None].expand(-1, self.future_frames, -1, -1, -1)
            ids = torch.arange(self.future_frames, device=x.device)
            emb = self.horizon_embed(ids).view(1, self.future_frames, -1, 1, 1)
            emb = emb.expand(b, -1, -1, height, width)
            head_in = torch.cat([feat_seq, future_feat, emb], dim=2)
            head_in = head_in.reshape(b * self.future_frames, head_in.shape[2], height, width)
            delta = self.delta_head(head_in).reshape(b, self.future_frames, height, width)
            return torch.clamp(last[:, None] + self.delta_scale * delta, 0.0, 2.0)

    return Residual3DUNetV2()


def _masked_smooth_l1(pred, target, mask):
    import torch.nn.functional as F

    valid = mask > 0.5
    if not bool(valid.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid])


def forecast_loss(pred, y, m, grad_weight: float, temporal_weight: float):
    base = _masked_smooth_l1(pred, y, m)

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]
    dx_m = m[:, :, :, 1:] * m[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    dy_m = m[:, :, 1:, :] * m[:, :, :-1, :]
    grad = 0.5 * (_masked_smooth_l1(dx_pred, dx_y, dx_m) + _masked_smooth_l1(dy_pred, dy_y, dy_m))

    dt_pred = pred[:, 1:] - pred[:, :-1]
    dt_y = y[:, 1:] - y[:, :-1]
    dt_m = m[:, 1:] * m[:, :-1]
    temporal = _masked_smooth_l1(dt_pred, dt_y, dt_m)

    total = base + float(grad_weight) * grad + float(temporal_weight) * temporal
    return total, base.detach(), grad.detach(), temporal.detach()


def run_epoch(model, loader, optimizer, device: str, train: bool, args: argparse.Namespace) -> dict[str, float]:
    import torch

    model.train(train)
    totals, bases, grads, temps = [], [], [], []
    for step, (x, fut, y, m) in enumerate(loader, start=1):
        x = x.to(device)
        fut = fut.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x, fut)
            loss, base, grad, temporal = forecast_loss(
                pred,
                y,
                m,
                grad_weight=float(args.grad_loss_weight),
                temporal_weight=float(args.temporal_loss_weight),
            )
            if train:
                loss.backward()
                optimizer.step()
        totals.append(float(loss.detach().cpu().item()))
        bases.append(float(base.cpu().item()))
        grads.append(float(grad.cpu().item()))
        temps.append(float(temporal.cpu().item()))
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[residual_unet_v2][{mode}] batches={step}/{len(loader)} loss={totals[-1]:.6f}", flush=True)
    return {
        "loss": float(np.mean(totals)) if totals else float("nan"),
        "base": float(np.mean(bases)) if bases else float("nan"),
        "grad": float(np.mean(grads)) if grads else float("nan"),
        "temporal": float(np.mean(temps)) if temps else float("nan"),
    }


def evaluate(model, loader, device: str, split: str, model_name: str) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for step, (x, fut, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device), fut.to(device)).detach().cpu().numpy()
            _accumulate(rows, model_name, split, pred, y.numpy(), m.numpy())
            if step % 100 == 0:
                print(f"[residual_unet_v2][{split}] eval batches={step}/{len(loader)}", flush=True)
    return _combine(rows)


def _parse_visual_horizons(text: str, future_frames: int) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if 1 <= h <= future_frames:
            out.append(h - 1)
    return sorted(set(out)) or [0, future_frames - 1]


def _masked_for_plot(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(mask <= 0.5, arr)


def _robust_limits(values: list[np.ndarray], masks: list[np.ndarray], fallback: tuple[float, float]) -> tuple[float, float]:
    chunks = []
    for val, mask in zip(values, masks):
        if np.any(mask > 0.5):
            chunks.append(np.asarray(val)[mask > 0.5])
    if not chunks:
        return fallback
    data = np.concatenate(chunks)
    lo = float(np.quantile(data, 0.02))
    hi = float(np.quantile(data, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return fallback
    pad = 0.05 * (hi - lo)
    return max(0.0, lo - pad), min(2.0, hi + pad)


def save_visual_samples(model, dataset, bounds, device: str, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    sample_dir = ensure_dir(out_dir / "sample_forecasts_v2")
    horizons = _parse_visual_horizons(str(args.visual_horizons), int(args.future_frames))
    n = min(int(args.sample_count), len(dataset))
    if n <= 0:
        return pd.DataFrame()
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    lon_min, lon_max, lat_min, lat_max = bounds
    extent = [lon_min, lon_max, lat_min, lat_max]
    rows = []
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad(color="white")

    model.eval()
    with torch.no_grad():
        for sample_id, idx in enumerate(indices):
            x, fut, y, m = dataset[int(idx)]
            pred = model(
                torch.from_numpy(x[None]).to(device),
                torch.from_numpy(fut[None]).to(device),
            ).detach().cpu().numpy()[0]
            issue_time = pd.Timestamp(dataset.time_df.iloc[int(idx)]["issue_time"])
            last = x[0, -1]
            last_mask = x[1, -1]

            values = [y[h] for h in horizons] + [pred[h] for h in horizons] + [last]
            masks = [m[h] for h in horizons] + [m[h] for h in horizons] + [last_mask]
            vmin, vmax = _robust_limits(values, masks, fallback=(0.0, 1.2))
            err_values = [np.abs(pred[h] - y[h]) for h in horizons]
            err_masks = [m[h] for h in horizons]
            _emin, emax = _robust_limits(err_values, err_masks, fallback=(0.0, 0.4))
            emax = max(0.05, min(1.0, emax))

            fig, axes = plt.subplots(len(horizons), 4, figsize=(13, 3.0 * len(horizons)), squeeze=False)
            for r, h in enumerate(horizons):
                future_time = issue_time + pd.Timedelta(minutes=int(args.freq_minutes) * (h + 1))
                panels = [
                    (last, last_mask, "latest input", cmap, vmin, vmax),
                    (y[h], m[h], f"h{h + 1} truth\n{future_time:%Y-%m-%d %H:%M}", cmap, vmin, vmax),
                    (pred[h], m[h], f"h{h + 1} prediction", cmap, vmin, vmax),
                    (np.abs(pred[h] - y[h]), m[h], f"h{h + 1} |pred-truth|", err_cmap, 0.0, emax),
                ]
                for c, (arr, mask, title, cm, lo, hi) in enumerate(panels):
                    ax = axes[r, c]
                    im = ax.imshow(
                        _masked_for_plot(arr, mask),
                        origin="lower",
                        extent=extent,
                        vmin=lo,
                        vmax=hi,
                        cmap=cm,
                        interpolation="nearest",
                        aspect="auto",
                    )
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("lon")
                    ax.set_ylabel("lat")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.suptitle(f"Residual 3D U-Net v2 sample | issue={issue_time:%Y-%m-%d %H:%M:%S}", y=0.995)
            fig.tight_layout()
            path = sample_dir / f"sample_{sample_id:03d}_{issue_time:%Y%m%d_%H%M%S}.png"
            fig.savefig(path, dpi=int(args.dpi))
            plt.close(fig)
            rows.append({"sample_id": sample_id, "dataset_index": int(idx), "issue_time": str(issue_time), "path": str(path)})
            print(f"[residual_unet_v2][sample] saved {path}", flush=True)
    return pd.DataFrame(rows)


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
        raise RuntimeError("PyTorch is required for residual 3D U-Net v2 training") from exc

    split_df = _load_split_df(args)
    split_df["split"] = "all"
    bounds = _bounds(split_df)
    time_df = build_issue_time_table(args)
    time_df.to_csv(out_dir / "fullfield_residual_unet_v2_issue_time_index.csv", index=False)

    cache = ReadOnlyDiskFrameCache(Path(args.frame_cache_dir), time_df, args)
    print(f"[residual_unet_v2] reuse frame cache: {args.frame_cache_dir}", flush=True)
    print(f"[residual_unet_v2] cache frames={len(cache.index)} required_frames={len(cache.sec)}", flush=True)

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
        print(f"[residual_unet_v2] {split} samples={len(datasets[split])}", flush=True)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = make_model(
        future_frames=int(args.future_frames),
        in_channels=5,
        future_channels=3,
        delta_scale=float(args.delta_scale),
        horizon_embed_dim=int(args.horizon_embed_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    bad = 0
    logs = []
    ckpt_path = out_dir / "residual_3d_unet_v2_best.pt"

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(model, loaders["train"], optimizer, device, train=True, args=args)
        va = run_epoch(model, loaders["val"], optimizer, device, train=False, args=args)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_base": tr["base"],
            "train_grad": tr["grad"],
            "train_temporal": tr["temporal"],
            "val_loss": va["loss"],
            "val_base": va["base"],
            "val_grad": va["grad"],
            "val_temporal": va["temporal"],
        }
        logs.append(row)
        print(
            "[residual_unet_v2] "
            f"epoch={epoch:03d} train_loss={tr['loss']:.6f} val_loss={va['loss']:.6f} "
            f"val_base={va['base']:.6f} val_grad={va['grad']:.6f} val_temporal={va['temporal']:.6f}",
            flush=True,
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "model_name": "residual_3d_unet_v2",
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= int(args.patience):
                break
    pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    all_eval = []
    for split in ["val", "test"]:
        eval_df = evaluate(model, loaders[split], device, split, "residual_3d_unet_v2")
        all_eval.append(eval_df)
        print(f"==== residual_3d_unet_v2 {split} ====")
        print(eval_df[eval_df["horizon"] == "all"].to_string(index=False))

    summary = pd.concat(all_eval, ignore_index=True).sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)
    summary.to_csv(out_dir / "fullfield_residual_unet_v2_eval_summary.csv", index=False)
    sample_df = save_visual_samples(model, datasets["test"], bounds, device, args, out_dir)
    sample_df.to_csv(out_dir / "sample_forecasts_v2.csv", index=False)

    write_json(
        out_dir / "run_meta.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "task": "full-field DensePV residual temporal forecasting v2",
            "model": "horizon-aware residual 3D U-Net with temporal-attention feature collapse",
            "loss": {
                "base": "masked SmoothL1 on future fields",
                "spatial_gradient_weight": float(args.grad_loss_weight),
                "temporal_difference_weight": float(args.temporal_loss_weight),
            },
            "frame_cache_dir": str(args.frame_cache_dir),
            "frame_cache_mode": "read_only",
            "frame_cache_required_frames": int(len(cache.sec)),
            "window_rule": "all 12 history frames and all 8 future frames must be between 07:00 and 16:45 and within the same calendar day",
        },
    )
    print("==== Full-field residual 3D U-Net v2 aggregate summary ====")
    print(summary[summary["horizon"] == "all"].to_string(index=False))
    print(f"saved: {out_dir / 'fullfield_residual_unet_v2_eval_summary.csv'}")
    print(f"figures: {out_dir / 'sample_forecasts_v2'}")


if __name__ == "__main__":
    main()

