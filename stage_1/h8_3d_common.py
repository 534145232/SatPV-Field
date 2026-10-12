#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

from h8_dataset_paths import (
    BIG_PATCH_FRAME_CACHE_DIR,
    BIG_PATCH_INDEX_4H3F_ALLPATCH_ALL_SPLITS,
    BIG_PATCH_INDEX_4H3F_TRAIN25_VALTEST_CENTER,
    BIG_PATCH_INDEX_DIR,
    BIG_PATCH_PROCESSED_ROOT,
    CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_4H3F_ALL,
    CENTER_INDEX_DIR,
    CENTER_PROCESSED_ROOT,
    CENTER_SAMPLE_CACHE_4H3F_DIR,
)

PROCESSED_ROOT = CENTER_PROCESSED_ROOT
INDEX_DIR = CENTER_INDEX_DIR
DEFAULT_INDEX_CSV = CENTER_INDEX_4H3F_ALL
DEFAULT_FRAME_CACHE_DIR = CENTER_FRAME_CACHE_DIR
DEFAULT_SAMPLE_CACHE_DIR = CENTER_SAMPLE_CACHE_4H3F_DIR

BIG_PROCESSED_ROOT = BIG_PATCH_PROCESSED_ROOT
BIG_INDEX_DIR = BIG_PATCH_INDEX_DIR
DEFAULT_AUGMENTED_INDEX_CSV = BIG_PATCH_INDEX_4H3F_TRAIN25_VALTEST_CENTER
DEFAULT_AUGMENTED_ALLPATCH_INDEX_CSV = BIG_PATCH_INDEX_4H3F_ALLPATCH_ALL_SPLITS
DEFAULT_AUGMENTED_FRAME_CACHE_DIR = BIG_PATCH_FRAME_CACHE_DIR

DEFAULT_CHANNELS = [
    "SWR",
    "PAR",
    "UVA",
    "UVB",
    "TAOT_02",
    "TAAE",
    "Sample_number",
]
TARGET_CHANNEL = "SWR"


def parse_channels(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(DEFAULT_CHANNELS)
    return [x.strip() for x in str(text).split(",") if x.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_index(index_csv: Path, split: str | None = None) -> pd.DataFrame:
    if not Path(index_csv).exists():
        hint = "Please generate the sample index first."
        if "h8_big_patch_index" in str(index_csv):
            hint = "Please run from the repository root: bash stage_1/run_stage1_build_indices_and_cache.sh"
        elif "4hist_3future" in str(index_csv):
            hint = "Please run from the repository root: bash stage_1/run_stage1_build_indices_and_cache.sh"
        raise FileNotFoundError(f"Index CSV not found: {index_csv}\n{hint}")
    df = pd.read_csv(index_csv)
    df["_index_row"] = np.arange(len(df), dtype=np.int64)
    if split is not None:
        df = df[df["split"] == split].copy()
    if len(df) == 0:
        raise ValueError(f"No rows found in {index_csv} for split={split}")
    return df.reset_index(drop=True)


class RunningMoments:
    def __init__(self, size: int):
        self.sum = np.zeros(size, dtype=np.float64)
        self.sumsq = np.zeros(size, dtype=np.float64)
        self.count = np.zeros(size, dtype=np.float64)

    def update_channels(self, arr: np.ndarray) -> None:
        for i in range(arr.shape[1]):
            v = arr[:, i]
            m = np.isfinite(v)
            if not np.any(m):
                continue
            vv = v[m].astype(np.float64)
            self.sum[i] += vv.sum()
            self.sumsq[i] += np.square(vv).sum()
            self.count[i] += vv.size

    def update_scalar(self, arr: np.ndarray) -> None:
        m = np.isfinite(arr)
        if not np.any(m):
            return
        vv = arr[m].astype(np.float64)
        self.sum[0] += vv.sum()
        self.sumsq[0] += np.square(vv).sum()
        self.count[0] += vv.size

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.sum / np.maximum(self.count, 1.0)
        var = self.sumsq / np.maximum(self.count, 1.0) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-6))
        std = np.where(self.count > 0, std, 1.0)
        mean = np.where(self.count > 0, mean, 0.0)
        return mean.astype(np.float32), std.astype(np.float32)


def _channel_lookup(ds: xr.Dataset) -> dict[str, int]:
    names = [str(x) for x in ds["channel"].values.tolist()]
    return {name: i for i, name in enumerate(names)}


def cache_key_for_nc(nc_path: str | Path) -> str:
    path = Path(nc_path)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def cache_meta_path(cache_dir: Path, nc_path: str | Path) -> Path:
    return cache_dir / f"{cache_key_for_nc(nc_path)}.json"


def estimate_normalization(
    index_csv: Path,
    channels: list[str],
    max_samples: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    df = load_index(index_csv, split="train")
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    x_mom = RunningMoments(len(channels))
    y_mom = RunningMoments(1)
    ds_cache: dict[str, xr.Dataset] = {}

    try:
        for _, row in df.iterrows():
            nc_path = str(row["nc_path"])
            ds = ds_cache.get(nc_path)
            if ds is None:
                ds = xr.open_dataset(nc_path)
                ds_cache[nc_path] = ds
            lookup = _channel_lookup(ds)
            missing = [c for c in channels + [TARGET_CHANNEL] if c not in lookup]
            if missing:
                raise KeyError(f"Missing channels in {nc_path}: {missing}")

            x_idx = [lookup[c] for c in channels]
            y_idx = lookup[TARGET_CHANNEL]
            xs = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
            ys = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)
            if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index):
                lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
                lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
            else:
                lat_slice = slice(None)
                lon_slice = slice(None)

            x = ds["data"].isel(time=xs, channel=x_idx, latitude=lat_slice, longitude=lon_slice).values
            y = ds["data"].isel(time=ys, channel=y_idx, latitude=lat_slice, longitude=lon_slice).values
            x_mom.update_channels(np.asarray(x, dtype=np.float32))
            y_mom.update_scalar(np.asarray(y, dtype=np.float32))
    finally:
        for ds in ds_cache.values():
            ds.close()

    x_mean, x_std = x_mom.finish()
    y_mean, y_std = y_mom.finish()
    return {
        "channels": channels,
        "target_channel": TARGET_CHANNEL,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y_mean[0]),
        "y_std": float(y_std[0]),
        "num_train_rows_used": int(len(df)),
    }


class H8SWRForecastDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        split: str,
        channels: list[str],
        stats: dict[str, Any],
        frame_cache_dir: Path | None = None,
        sample_cache_dir: Path | None = None,
    ):
        self.df = load_index(index_csv, split=split)
        self.split = split
        self.channels = channels
        self.stats = stats
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir is not None else None
        self.sample_cache_dir = Path(sample_cache_dir) if sample_cache_dir is not None else None
        self.x_mean = np.asarray(stats["x_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.x_std = np.asarray(stats["x_std"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.y_mean = np.float32(stats["y_mean"])
        self.y_std = np.float32(stats["y_std"])
        self.ds_cache: dict[str, xr.Dataset] = {}
        self.lookup_cache: dict[str, dict[str, int]] = {}
        self.array_cache: dict[str, dict[str, Any]] = {}
        self.sample_cache: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.df)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["ds_cache"] = {}
        state["lookup_cache"] = {}
        state["array_cache"] = {}
        state["sample_cache"] = None
        return state

    def close(self) -> None:
        for ds in self.ds_cache.values():
            ds.close()
        self.ds_cache = {}
        self.array_cache = {}
        self.sample_cache = None

    def _open(self, path: str) -> xr.Dataset:
        ds = self.ds_cache.get(path)
        if ds is None:
            ds = xr.open_dataset(path)
            self.ds_cache[path] = ds
        return ds

    def _lookup(self, path: str, ds: xr.Dataset) -> dict[str, int]:
        lookup = self.lookup_cache.get(path)
        if lookup is None:
            lookup = _channel_lookup(ds)
            missing = [c for c in self.channels + [TARGET_CHANNEL] if c not in lookup]
            if missing:
                raise KeyError(f"Missing channels in {path}: {missing}")
            self.lookup_cache[path] = lookup
        return lookup

    def _open_frame_cache(self, path: str) -> dict[str, Any] | None:
        if self.frame_cache_dir is None:
            return None
        cached = self.array_cache.get(path)
        if cached is not None:
            return cached

        meta_path = cache_meta_path(self.frame_cache_dir, path)
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        data_path = Path(meta["data_path"])
        if not data_path.is_absolute():
            data_path = meta_path.parent / data_path
        data = np.load(data_path, mmap_mode="r")

        mask = None
        mask_path_text = meta.get("valid_mask_path")
        if mask_path_text:
            mask_path = Path(mask_path_text)
            if not mask_path.is_absolute():
                mask_path = meta_path.parent / mask_path
            if mask_path.exists():
                mask = np.load(mask_path, mmap_mode="r")

        lookup = {str(name): i for i, name in enumerate(meta["channels"])}
        missing = [c for c in self.channels + [TARGET_CHANNEL] if c not in lookup]
        if missing:
            raise KeyError(f"Missing channels in frame cache {meta_path}: {missing}")

        cached = {"data": data, "valid_mask": mask, "lookup": lookup}
        self.array_cache[path] = cached
        return cached

    def _open_sample_cache(self) -> dict[str, Any] | None:
        if self.sample_cache_dir is None:
            return None
        if {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(self.df.columns):
            # Existing sample caches are center-window caches. Patch-augmented
            # indexes must read from big frame cache or NetCDF and crop spatially.
            return None
        if self.sample_cache is not None:
            return self.sample_cache

        meta_path = self.sample_cache_dir / "sample_cache_meta.json"
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        cache_channels = [str(x) for x in meta["channels"]]
        if cache_channels != self.channels:
            raise ValueError(
                "Sample cache channels do not match requested channels: "
                f"cache={cache_channels}, requested={self.channels}"
            )
        if str(meta.get("target_channel", TARGET_CHANNEL)) != TARGET_CHANNEL:
            raise ValueError(f"Unexpected sample cache target: {meta.get('target_channel')}")

        x = np.load(self.sample_cache_dir / "x_float32.npy", mmap_mode="r")
        y = np.load(self.sample_cache_dir / "y_float32.npy", mmap_mode="r")
        mask = np.load(self.sample_cache_dir / "mask_uint8.npy", mmap_mode="r")
        self.sample_cache = {"x": x, "y": y, "mask": mask, "meta": meta}
        return self.sample_cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        nc_path = str(row["nc_path"])

        xs = slice(int(row["x_start_idx"]), int(row["x_end_idx"]) + 1)
        ys = slice(int(row["y_start_idx"]), int(row["y_end_idx"]) + 1)
        has_patch_crop = {"lat_start_idx", "lat_end_idx", "lon_start_idx", "lon_end_idx"}.issubset(row.index)
        if has_patch_crop:
            lat_slice = slice(int(row["lat_start_idx"]), int(row["lat_end_idx"]) + 1)
            lon_slice = slice(int(row["lon_start_idx"]), int(row["lon_end_idx"]) + 1)
        else:
            lat_slice = slice(None)
            lon_slice = slice(None)

        sample_cache = self._open_sample_cache()
        if sample_cache is not None:
            sample_idx = int(row["_index_row"])
            x = np.asarray(sample_cache["x"][sample_idx], dtype=np.float32)
            y = np.asarray(sample_cache["y"][sample_idx], dtype=np.float32)
            mask = np.asarray(sample_cache["mask"][sample_idx], dtype=bool)
        else:
            frame_cache = self._open_frame_cache(nc_path)
            if frame_cache is not None:
                lookup = frame_cache["lookup"]
                x_idx = [lookup[c] for c in self.channels]
                y_idx = lookup[TARGET_CHANNEL]
                data = frame_cache["data"]
                x = np.asarray(data[xs, :, lat_slice, lon_slice][:, x_idx, :, :], dtype=np.float32)
                y = np.asarray(data[ys, y_idx, lat_slice, lon_slice], dtype=np.float32)
                mask = np.isfinite(y)
                if frame_cache["valid_mask"] is not None:
                    vm = np.asarray(frame_cache["valid_mask"][ys, lat_slice, lon_slice], dtype=np.float32)
                    mask &= vm > 0
            else:
                ds = self._open(nc_path)
                lookup = self._lookup(nc_path, ds)
                x_idx = [lookup[c] for c in self.channels]
                y_idx = lookup[TARGET_CHANNEL]

                x = np.asarray(
                    ds["data"].isel(time=xs, channel=x_idx, latitude=lat_slice, longitude=lon_slice).values,
                    dtype=np.float32,
                )
                y = np.asarray(
                    ds["data"].isel(time=ys, channel=y_idx, latitude=lat_slice, longitude=lon_slice).values,
                    dtype=np.float32,
                )

                mask = np.isfinite(y)
                if "valid_mask" in ds:
                    vm = np.asarray(
                        ds["valid_mask"].isel(time=ys, latitude=lat_slice, longitude=lon_slice).values,
                        dtype=np.float32,
                    )
                    mask &= vm > 0

        x = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        y = (y - self.y_mean) / max(float(self.y_std), 1e-6)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        future_frames = int(row["future_frames"]) if "future_frames" in row.index else int(y.shape[0])
        future_times = [
            str(row[f"y_time_{j}_bjt"])
            for j in range(future_frames)
            if f"y_time_{j}_bjt" in row.index
        ]
        meta = {
            "nc_path": nc_path,
            "issue_time_bjt": str(row["issue_time_bjt"]),
            "future_times_bjt": " -> ".join(future_times),
            "patch_id": str(row["patch_id"]) if "patch_id" in row.index else "",
            "is_center": str(row["is_center"]) if "is_center" in row.index else "",
        }
        for j, ts in enumerate(future_times):
            meta[f"y_time_{j}_bjt"] = ts

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "meta": meta,
        }


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = max(g for g in range(1, min(8, channels) + 1) if channels % g == 0)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BasicSWR3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        future_frames: int = 2,
        width: int = 48,
        depth: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.future_frames = future_frames
        self.stem = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[Residual3DBlock(width, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, future_frames, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.blocks(self.stem(x))
        last = h[:, :, -1]
        mean = h.mean(dim=2)
        return self.head(torch.cat([last, mean], dim=1))


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            b, _, h, w = x.shape
            h_prev = x.new_zeros(b, self.hidden_channels, h, w)
            c_prev = x.new_zeros(b, self.hidden_channels, h, w)
        else:
            h_prev, c_prev = state

        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class BasicConvLSTMForecaster(nn.Module):
    def __init__(
        self,
        in_channels: int,
        future_frames: int = 2,
        width: int = 48,
        depth: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.future_frames = future_frames
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(max(g for g in range(1, min(8, width) + 1) if width % g == 0), width),
            nn.SiLU(inplace=True),
        )
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(width if i == 0 else width, width)
                for i in range(depth)
            ]
        )
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, future_frames, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        outputs = []
        for t in range(x.shape[1]):
            h = self.stem(x[:, t])
            for i, cell in enumerate(self.cells):
                states[i] = cell(h, states[i])
                h = states[i][0]
            outputs.append(h)
        last = self.dropout(outputs[-1])
        mean = torch.stack(outputs, dim=2).mean(dim=2)
        return self.head(torch.cat([last, mean], dim=1))


class MultiScaleConvLSTMForecaster(nn.Module):
    def __init__(
        self,
        in_channels: int,
        future_frames: int = 2,
        width: int = 40,
        depth: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.future_frames = future_frames
        coarse_width = max(width, 16)
        fine_width = max(width // 2, 16)
        groups_c = max(g for g in range(1, min(8, coarse_width) + 1) if coarse_width % g == 0)
        groups_f = max(g for g in range(1, min(8, fine_width) + 1) if fine_width % g == 0)

        self.coarse_stem = nn.Sequential(
            nn.Conv2d(in_channels, coarse_width, kernel_size=3, padding=1),
            nn.GroupNorm(groups_c, coarse_width),
            nn.SiLU(inplace=True),
        )
        self.fine_stem = nn.Sequential(
            nn.Conv2d(in_channels, fine_width, kernel_size=3, padding=1),
            nn.GroupNorm(groups_f, fine_width),
            nn.SiLU(inplace=True),
        )
        self.coarse_cells = nn.ModuleList(
            [ConvLSTMCell(coarse_width, coarse_width) for _ in range(max(1, depth))]
        )
        self.fine_cells = nn.ModuleList(
            [ConvLSTMCell(fine_width, fine_width) for _ in range(max(1, depth))]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(coarse_width + fine_width, width, kernel_size=3, padding=1),
            nn.GroupNorm(max(g for g in range(1, min(8, width) + 1) if width % g == 0), width),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, future_frames, kernel_size=1),
        )

    def _run_cells(
        self,
        h: torch.Tensor,
        cells: nn.ModuleList,
        states: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> torch.Tensor:
        for i, cell in enumerate(cells):
            states[i] = cell(h, states[i])
            h = states[i][0]
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        coarse_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.coarse_cells)
        fine_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.fine_cells)
        out_h, out_w = x.shape[-2], x.shape[-1]

        fine_last = None
        coarse_last = None
        for t in range(x.shape[1]):
            xt = x[:, t]
            fine = self.fine_stem(xt)
            coarse_input = F.interpolate(
                xt,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            coarse = self.coarse_stem(coarse_input)
            coarse_last = self._run_cells(coarse, self.coarse_cells, coarse_states)
            fine_last = self._run_cells(fine, self.fine_cells, fine_states)

        assert fine_last is not None and coarse_last is not None
        coarse_up = F.interpolate(coarse_last, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([fine_last, coarse_up], dim=1))


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


@torch.no_grad()
def masked_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    pred = pred_norm * y_std + y_mean
    target = target_norm * y_std + y_mean
    m = mask > 0.5
    if not torch.any(m):
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    err = pred[m] - target[m]
    return {
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(err * err)).item()),
        "bias": float(err.mean().item()),
    }
