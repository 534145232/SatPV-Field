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

from visualize_regional_split_frames import (
    _bounds,
    _xy_to_ij,
    ensure_dir,
    estimate_kpi,
    load_station_series,
    splat_train_field,
    sparse_truth,
    station_value,
    write_json,
)

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": str(p3.DATA_DIR / "_local_core_defaults"),
    "train_pv_dir": str(p3.TRAIN_PV_DIR),
    "val_pv_dir": str(p3.VAL_PV_DIR),
    "out_dir": str(p3.DATA_DIR / "_local_core_defaults" / "regional_dense_reconstruction_seed2026"),
    "time_start": "2023-03-09 07:00:00",
    "time_end": "2023-06-24 16:45:00",
    "train_end": "2023-05-31 23:59:59",
    "val_end": "2023-06-07 23:59:59",
    "freq_minutes": 15,
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
    "mask_grid_size": 8,
    "mask_block_w_min": 1,
    "mask_block_w_max": 3,
    "mask_block_h_min": 1,
    "mask_block_h_max": 3,
    "min_train_values": 80,
    "min_holdout_values": 8,
    "batch_size": 32,
    "epochs": 50,
    "patience": 8,
    "lr": 2e-3,
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
            "Train a regional DensePV reconstruction model with dynamic regional masks on train stations "
            "and fixed geographic val/test holdout regions."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=Path(PYCHARM_DEFAULTS["run_dir"]))
    parser.add_argument("--train_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["train_pv_dir"]))
    parser.add_argument("--val_pv_dir", type=Path, default=Path(PYCHARM_DEFAULTS["val_pv_dir"]))
    parser.add_argument("--out_dir", type=Path, default=Path(PYCHARM_DEFAULTS["out_dir"]))
    parser.add_argument("--time_start", type=str, default=str(PYCHARM_DEFAULTS["time_start"]))
    parser.add_argument("--time_end", type=str, default=str(PYCHARM_DEFAULTS["time_end"]))
    parser.add_argument("--train_end", type=str, default=str(PYCHARM_DEFAULTS["train_end"]))
    parser.add_argument("--val_end", type=str, default=str(PYCHARM_DEFAULTS["val_end"]))
    parser.add_argument("--freq_minutes", type=int, default=int(PYCHARM_DEFAULTS["freq_minutes"]))
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
    parser.add_argument("--mask_grid_size", type=int, default=int(PYCHARM_DEFAULTS["mask_grid_size"]))
    parser.add_argument("--mask_block_w_min", type=int, default=int(PYCHARM_DEFAULTS["mask_block_w_min"]))
    parser.add_argument("--mask_block_w_max", type=int, default=int(PYCHARM_DEFAULTS["mask_block_w_max"]))
    parser.add_argument("--mask_block_h_min", type=int, default=int(PYCHARM_DEFAULTS["mask_block_h_min"]))
    parser.add_argument("--mask_block_h_max", type=int, default=int(PYCHARM_DEFAULTS["mask_block_h_max"]))
    parser.add_argument("--min_train_values", type=int, default=int(PYCHARM_DEFAULTS["min_train_values"]))
    parser.add_argument("--min_holdout_values", type=int, default=int(PYCHARM_DEFAULTS["min_holdout_values"]))
    parser.add_argument("--batch_size", type=int, default=int(PYCHARM_DEFAULTS["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(PYCHARM_DEFAULTS["epochs"]))
    parser.add_argument("--patience", type=int, default=int(PYCHARM_DEFAULTS["patience"]))
    parser.add_argument("--lr", type=float, default=float(PYCHARM_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(PYCHARM_DEFAULTS["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--device", type=str, default=str(PYCHARM_DEFAULTS["device"]))
    parser.add_argument("--num_workers", type=int, default=int(PYCHARM_DEFAULTS["num_workers"]))
    return parser.parse_args()


def build_time_table(args: argparse.Namespace) -> pd.DataFrame:
    times = pd.date_range(pd.Timestamp(args.time_start), pd.Timestamp(args.time_end), freq=f"{int(args.freq_minutes)}min")
    rows = []
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    for ts in times:
        minute_of_day = int(ts.hour) * 60 + int(ts.minute)
        if minute_of_day < 7 * 60 or minute_of_day > 16 * 60 + 45:
            continue
        if ts <= train_end:
            split = "train"
        elif ts <= val_end:
            split = "val"
        else:
            split = "test"
        rows.append({"timestamp": ts, "timestamp_sec": int(ts.timestamp()), "time_split": split})
    return pd.DataFrame(rows)


def collect_split_values(
    split_df: pd.DataFrame,
    station_split: str,
    ts: pd.Timestamp,
    series: dict[str, pd.DataFrame],
    kpi: dict[str, float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for _, row in split_df[split_df["split"] == station_split].iterrows():
        val = station_value(row, ts, series, kpi, args)
        if val is None:
            continue
        rows.append(
            {
                "station_sn": str(row["station_sn"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "value": float(val),
            }
        )
    return pd.DataFrame(rows)


def _rectangle_mask(values: pd.DataFrame, bounds: tuple[float, float, float, float], args: argparse.Namespace, rng: np.random.Generator) -> pd.Series:
    g = int(args.mask_grid_size)
    lon_min, lon_max, lat_min, lat_max = bounds
    bx = np.floor((values["lon"].to_numpy(np.float64) - lon_min) / max(1e-12, lon_max - lon_min) * g).astype(int)
    by = np.floor((values["lat"].to_numpy(np.float64) - lat_min) / max(1e-12, lat_max - lat_min) * g).astype(int)
    bx = np.clip(bx, 0, g - 1)
    by = np.clip(by, 0, g - 1)
    for _ in range(80):
        w = int(rng.integers(int(args.mask_block_w_min), int(args.mask_block_w_max) + 1))
        h = int(rng.integers(int(args.mask_block_h_min), int(args.mask_block_h_max) + 1))
        x0 = int(rng.integers(0, max(1, g - w + 1)))
        y0 = int(rng.integers(0, max(1, g - h + 1)))
        mask = (bx >= x0) & (bx < x0 + w) & (by >= y0) & (by < y0 + h)
        if int(mask.sum()) >= int(args.min_holdout_values):
            return pd.Series(mask, index=values.index)
    # Fallback: pick a random anchor station and mask its coarse cell.
    if values.empty:
        return pd.Series([], dtype=bool)
    idx = int(rng.integers(0, len(values)))
    return pd.Series((bx == bx[idx]) & (by == by[idx]), index=values.index)


def splat_values(
    values: pd.DataFrame,
    split_df_like: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    # Reuse the visualization helper by temporarily shaping values as train stations.
    # The helper expects station rows plus per-station series, so this local version keeps it explicit.
    g = int(args.grid_size)
    field_sum = np.zeros((g, g), dtype=np.float32)
    weight_sum = np.zeros((g, g), dtype=np.float32)
    sigma = max(1e-6, float(args.gaussian_sigma_cells))
    radius = max(0, int(args.gaussian_radius_cells))
    for _, row in values.iterrows():
        cy, cx = _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, g)
        val = float(row["value"])
        for yy in range(max(0, cy - radius), min(g, cy + radius + 1)):
            for xx in range(max(0, cx - radius), min(g, cx + radius + 1)):
                dist2 = float((yy - cy) ** 2 + (xx - cx) ** 2)
                if dist2 > float(radius * radius):
                    continue
                w = math.exp(-dist2 / (2.0 * sigma * sigma))
                field_sum[yy, xx] += val * w
                weight_sum[yy, xx] += w
    field = np.zeros_like(field_sum)
    mask = weight_sum > float(args.min_grid_weight)
    field[mask] = field_sum[mask] / weight_sum[mask]
    return field, np.clip(weight_sum, 0.0, 1.0)


def sparse_grid(values: pd.DataFrame, bounds: tuple[float, float, float, float], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    g = int(args.grid_size)
    target = np.zeros((g, g), dtype=np.float32)
    mask = np.zeros((g, g), dtype=np.float32)
    count = np.zeros((g, g), dtype=np.float32)
    for _, row in values.iterrows():
        y, x = _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, g)
        target[y, x] += float(row["value"])
        count[y, x] += 1.0
    valid = count > 0
    target[valid] = target[valid] / count[valid]
    mask[valid] = 1.0
    return target, mask


def target_gaussian_grid(values: pd.DataFrame, bounds: tuple[float, float, float, float], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    target, support = splat_values(values, pd.DataFrame(), bounds, args)
    mask = (support > float(args.min_grid_weight)).astype(np.float32)
    return target.astype(np.float32), mask


class RegionalDenseDataset:
    def __init__(
        self,
        time_df: pd.DataFrame,
        split_df: pd.DataFrame,
        series: dict[str, pd.DataFrame],
        kpi: dict[str, float],
        bounds: tuple[float, float, float, float],
        args: argparse.Namespace,
        mode: str,
    ):
        self.time_df = time_df.reset_index(drop=True)
        self.split_df = split_df
        self.series = series
        self.kpi = kpi
        self.bounds = bounds
        self.args = args
        self.mode = mode
        self.rng = np.random.default_rng(int(args.seed) + (0 if mode == "train" else 10_000))

    def __len__(self) -> int:
        return len(self.time_df)

    def __getitem__(self, idx: int):
        ts = pd.Timestamp(self.time_df.iloc[idx]["timestamp"])
        train_values = collect_split_values(self.split_df, "train", ts, self.series, self.kpi, self.args)
        if self.mode == "train":
            if len(train_values) < int(self.args.min_train_values):
                raise RuntimeError(f"too few train values at {ts}: {len(train_values)}")
            held_mask = _rectangle_mask(train_values, self.bounds, self.args, self.rng)
            held = train_values[held_mask].copy()
            observed = train_values[~held_mask].copy()
            if len(held) < 1 or len(observed) < 1:
                held = train_values.sample(min(len(train_values), int(self.args.min_holdout_values)), random_state=int(idx))
                observed = train_values.drop(index=held.index)
        else:
            target_split = "val" if self.mode == "val" else "test"
            observed = train_values
            held = collect_split_values(self.split_df, target_split, ts, self.series, self.kpi, self.args)
            if len(held) < 1:
                held = train_values.iloc[:0].copy()
        x_field, x_support = splat_values(observed, self.split_df, self.bounds, self.args)
        y, y_mask = target_gaussian_grid(held, self.bounds, self.args)
        x = np.stack([x_field, x_support], axis=0).astype(np.float32)
        return x, y[None].astype(np.float32), y_mask[None].astype(np.float32)


class TinyUNet:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class Block(nn.Module):
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
                self.enc1 = Block(2, 32)
                self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
                self.enc2 = Block(64, 64)
                self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
                self.mid = Block(128, 128)
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = Block(128, 64)
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = Block(64, 32)
                self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.down1(e1))
                m = self.mid(self.down2(e2))
                d2 = self.up2(m)
                if d2.shape[-2:] != e2.shape[-2:]:
                    d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                if d1.shape[-2:] != e1.shape[-2:]:
                    d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
                return self.out(self.dec1(torch.cat([d1, e1], dim=1))) * 2.0

        return Model()


def masked_huber(pred, target, mask):
    import torch
    import torch.nn.functional as F

    valid = mask > 0.5
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid])


def run_epoch(model, loader, optimizer, device: str, train: bool) -> float:
    losses = []
    model.train(train)
    import torch

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = masked_huber(pred, y, m)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model, loader, device: str, split: str) -> dict:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for x, y, m in loader:
            pred = model(x.to(device)).cpu().numpy()
            target = y.numpy()
            mask = m.numpy() > 0.5
            if not mask.any():
                continue
            diff = pred[mask] - target[mask]
            rows.append(
                {
                    "eval_split": split,
                    "n_points": int(mask.sum()),
                    "mae_field": float(np.mean(np.abs(diff))),
                    "rmse_field": float(np.sqrt(np.mean(diff**2))),
                }
            )
    if not rows:
        return {"eval_split": split, "n_points": 0, "mae_field": np.nan, "rmse_field": np.nan}
    total = sum(r["n_points"] for r in rows)
    return {
        "eval_split": split,
        "n_points": int(total),
        "mae_field": float(sum(r["mae_field"] * r["n_points"] for r in rows) / total),
        "rmse_field": float(math.sqrt(sum((r["rmse_field"] ** 2) * r["n_points"] for r in rows) / total)),
    }


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
        raise RuntimeError("PyTorch is required for regional dense reconstruction") from exc

    split_path = Path(args.run_dir) / "station_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"station split not found: {split_path}")
    split_df = pd.read_csv(split_path, dtype={"station_sn": str})
    split_df["lon"] = pd.to_numeric(split_df["lon"], errors="coerce")
    split_df["lat"] = pd.to_numeric(split_df["lat"], errors="coerce")
    split_df = split_df.dropna(subset=["station_sn", "lon", "lat", "split"]).reset_index(drop=True)

    series, load_status = load_station_series(split_df, args)
    load_status.to_csv(out_dir / "station_csv_load_status.csv", index=False)
    kpi = estimate_kpi(split_df, series, args)
    pd.DataFrame([{"station_sn": k, "kpi": v} for k, v in sorted(kpi.items())]).to_csv(out_dir / "station_kpi_geo_baselines.csv", index=False)
    bounds = _bounds(split_df)
    time_df = build_time_table(args)
    time_df.to_csv(out_dir / "regional_reconstruction_time_index.csv", index=False)

    ds_train = RegionalDenseDataset(time_df[time_df["time_split"] == "train"], split_df, series, kpi, bounds, args, mode="train")
    ds_val = RegionalDenseDataset(time_df[time_df["time_split"] == "val"], split_df, series, kpi, bounds, args, mode="val")
    ds_test = RegionalDenseDataset(time_df[time_df["time_split"] == "test"], split_df, series, kpi, bounds, args, mode="test")
    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    bad = 0
    log_rows = []
    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(model, dl_train, optimizer, device, train=True)
        va = run_epoch(model, dl_val, optimizer, device, train=False)
        log_rows.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"[regional_dense] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)
        if va < best_val:
            best_val = va
            bad = 0
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(args.patience):
                break
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    eval_rows = [evaluate(model, dl_val, device, "val_region"), evaluate(model, dl_test, device, "test_region")]
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "regional_reconstruction_eval_summary.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "station_split_csv": str(split_path),
            "bounds": {"lon_min": bounds[0], "lon_max": bounds[1], "lat_min": bounds[2], "lat_max": bounds[3]},
            "n_train_stations": int((split_df["split"] == "train").sum()),
            "n_val_stations": int((split_df["split"] == "val").sum()),
            "n_test_stations": int((split_df["split"] == "test").sum()),
            "input_channels": ["observed_train_kpi_adjusted_csi_field", "observed_train_support_mask"],
            "training_target": "dynamically masked contiguous train-region stations",
            "validation_target": "fixed geographic val-region stations",
            "test_target": "fixed geographic test-region stations",
        },
    )
    print("==== Regional dense reconstruction summary ====")
    print(eval_df.to_string(index=False))
    print(f"saved: {out_dir / 'best.pt'}")
    print(f"saved: {out_dir / 'regional_reconstruction_eval_summary.csv'}")


if __name__ == "__main__":
    main()

