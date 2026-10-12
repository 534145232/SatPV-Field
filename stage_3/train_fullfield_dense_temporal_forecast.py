#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_regional_dense_reconstruction import masked_huber, splat_values
from visualize_regional_split_frames import _bounds, ensure_dir, estimate_kpi, load_station_series, write_json

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "run_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults",
    "train_pv_dir": "/data/DayinChen/NWP/PV/train_directory_cleaned_csv",
    "val_pv_dir": "/data/DayinChen/NWP/PV/val_directory_cleaned_csv",
    "out_dir": "/code/Final_3_stage_pipeline/stage_3/data/_local_core_defaults/fullfield_dense_temporal_seed2026",
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
    "models": "3d_cnn,convlstm",
    "batch_size": 12,
    "epochs": 30,
    "patience": 6,
    "lr": 1.5e-3,
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
            "Full-field DensePV temporal forecasting sanity check. "
            "No unseen-region mask: past all-station KPI-adjusted CSI fields predict future all-station fields."
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
    parser.add_argument("--models", type=str, default=str(PYCHARM_DEFAULTS["models"]))
    parser.add_argument("--batch_size", type=int, default=int(PYCHARM_DEFAULTS["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(PYCHARM_DEFAULTS["epochs"]))
    parser.add_argument("--patience", type=int, default=int(PYCHARM_DEFAULTS["patience"]))
    parser.add_argument("--lr", type=float, default=float(PYCHARM_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(PYCHARM_DEFAULTS["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(PYCHARM_DEFAULTS["seed"]))
    parser.add_argument("--device", type=str, default=str(PYCHARM_DEFAULTS["device"]))
    parser.add_argument("--num_workers", type=int, default=int(PYCHARM_DEFAULTS["num_workers"]))
    return parser.parse_args()


def _daytime(ts: pd.Timestamp) -> bool:
    minute = int(ts.hour) * 60 + int(ts.minute)
    return 7 * 60 <= minute <= 16 * 60 + 45


def build_issue_time_table(args: argparse.Namespace) -> pd.DataFrame:
    # Strictly follow the original PV forecasting temporal protocol:
    # 12 daytime history frames + 8 daytime future frames, all within one
    # calendar day. With 07:00-16:45 source data, this prevents cross-day
    # windows and makes the dense-field task comparable to the prior
    # multimodal 3h -> 2h setup.
    step = pd.Timedelta(minutes=int(args.freq_minutes))
    start = pd.Timestamp(args.time_start)
    end = pd.Timestamp(args.time_end)
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    rows = []
    for issue in pd.date_range(start, end, freq=f"{int(args.freq_minutes)}min"):
        past = [issue - step * (int(args.past_frames) - 1 - i) for i in range(int(args.past_frames))]
        future = [issue + step * (i + 1) for i in range(int(args.future_frames))]
        all_times = past + future
        if not all(_daytime(t) for t in all_times):
            continue
        if past[0].date() != issue.date() or future[-1].date() != issue.date():
            continue
        if issue <= train_end:
            split = "train"
        elif issue <= val_end:
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "issue_time": issue,
                "issue_sec": int(issue.timestamp()),
                "time_split": split,
                "past_start": past[0].strftime("%Y-%m-%d %H:%M:%S"),
                "past_end": past[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "future_start": future[0].strftime("%Y-%m-%d %H:%M:%S"),
                "future_end": future[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "past_times": "|".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in past),
                "future_times": "|".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in future),
            }
        )
    return pd.DataFrame(rows)


def _parse_time_list(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(x) for x in str(text).split("|") if str(x).strip()]


def collect_all_values(
    split_df: pd.DataFrame,
    ts: pd.Timestamp,
    series: dict[str, pd.DataFrame],
    kpi: dict[str, float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    from visualize_regional_split_frames import station_value

    rows = []
    for _, row in split_df.iterrows():
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


class FullFieldTemporalDataset:
    def __init__(
        self,
        time_df: pd.DataFrame,
        split_df: pd.DataFrame,
        series: dict[str, pd.DataFrame],
        kpi: dict[str, float],
        bounds: tuple[float, float, float, float],
        args: argparse.Namespace,
        frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.time_df = time_df.reset_index(drop=True)
        self.split_df = split_df
        self.series = series
        self.kpi = kpi
        self.bounds = bounds
        self.args = args
        self.frame_cache = frame_cache if frame_cache is not None else {}

    def __len__(self) -> int:
        return len(self.time_df)

    def frame(self, ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        key = int(pd.Timestamp(ts).timestamp())
        item = self.frame_cache.get(key)
        if item is not None:
            return item
        vals = collect_all_values(self.split_df, ts, self.series, self.kpi, self.args)
        field, support = splat_values(vals, self.split_df, self.bounds, self.args)
        mask = (support > float(self.args.min_grid_weight)).astype(np.float32)
        item = (field.astype(np.float32), mask.astype(np.float32))
        self.frame_cache[key] = item
        if len(self.frame_cache) % 500 == 0:
            print(f"[fullfield] cached frames={len(self.frame_cache)} latest={ts}", flush=True)
        return item

    def __getitem__(self, idx: int):
        row = self.time_df.iloc[idx]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])
        x_fields, x_masks = [], []
        for ts in past_times:
            field, support = self.frame(ts)
            x_fields.append(field)
            x_masks.append(support)
        y_list, m_list = [], []
        for ts in future_times:
            field, mask = self.frame(ts)
            y_list.append(field)
            m_list.append(mask)
        x = np.stack([np.stack(x_fields, axis=0), np.stack(x_masks, axis=0)], axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        m = np.stack(m_list, axis=0).astype(np.float32)
        return x, y, m


class Conv3DForecaster:
    def __new__(cls, future_frames: int):
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
            def __init__(self, h: int):
                super().__init__()
                self.enc1 = Block3D(2, 32)
                self.down1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
                self.enc2 = Block3D(64, 64)
                self.down2 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
                self.mid = Block3D(96, 96)
                self.proj1 = nn.Conv2d(32, 32, 1)
                self.proj2 = nn.Conv2d(64, 64, 1)
                self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
                self.dec2 = Block2D(128, 64)
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = Block2D(64, 32)
                self.out = nn.Sequential(nn.Conv2d(32, h, 1), nn.Sigmoid())

            @staticmethod
            def collapse(x):
                return 0.7 * x[:, :, -1] + 0.3 * x.mean(dim=2)

            def forward(self, x):
                e1_3d = self.enc1(x)
                e2_3d = self.enc2(self.down1(e1_3d))
                mid_3d = self.mid(self.down2(e2_3d))
                e1 = self.proj1(self.collapse(e1_3d))
                e2 = self.proj2(self.collapse(e2_3d))
                mid = self.collapse(mid_3d)
                d2 = self.up2(mid)
                if d2.shape[-2:] != e2.shape[-2:]:
                    d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                if d1.shape[-2:] != e1.shape[-2:]:
                    d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
                return self.out(self.dec1(torch.cat([d1, e1], dim=1))) * 2.0

        return Model(int(future_frames))


class ConvLSTMForecaster:
    def __new__(cls, future_frames: int):
        import torch
        import torch.nn as nn

        class ConvLSTMCell(nn.Module):
            def __init__(self, cin: int, hidden: int):
                super().__init__()
                self.hidden = hidden
                self.conv = nn.Conv2d(cin + hidden, 4 * hidden, 3, padding=1)

            def forward(self, x, h, c):
                gates = self.conv(torch.cat([x, h], dim=1))
                i, f, o, g = torch.chunk(gates, 4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                g = torch.tanh(g)
                c = f * c + i * g
                h = o * torch.tanh(c)
                return h, c

        class Model(nn.Module):
            def __init__(self, h_out: int):
                super().__init__()
                self.h_out = int(h_out)
                self.in_proj = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU())
                self.cell1 = ConvLSTMCell(32, 48)
                self.cell2 = ConvLSTMCell(48, 48)
                self.readout = nn.Sequential(
                    nn.Conv2d(48, 48, 3, padding=1),
                    nn.GroupNorm(4, 48),
                    nn.GELU(),
                    nn.Conv2d(48, h_out, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, _, t, h, w = x.shape
                device = x.device
                h1 = torch.zeros(b, 48, h, w, device=device)
                c1 = torch.zeros_like(h1)
                h2 = torch.zeros_like(h1)
                c2 = torch.zeros_like(h1)
                for step in range(t):
                    z = self.in_proj(x[:, :, step])
                    h1, c1 = self.cell1(z, h1, c1)
                    h2, c2 = self.cell2(h1, h2, c2)
                return self.readout(h2) * 2.0

        return Model(int(future_frames))


def build_model(name: str, future_frames: int):
    if name == "3d_cnn":
        return Conv3DForecaster(future_frames)
    if name == "convlstm":
        return ConvLSTMForecaster(future_frames)
    raise ValueError(f"unsupported model: {name}")


def baseline_predictions(x: np.ndarray, future_frames: int) -> dict[str, np.ndarray]:
    field = x[:, 0]  # [B, T, H, W]
    last = field[:, -1]
    prev = field[:, -2] if field.shape[1] >= 2 else last
    mean = field.mean(axis=1)
    trend = last - prev
    out = {}
    out["persistence_last"] = np.repeat(last[:, None], future_frames, axis=1)
    out["past_mean"] = np.repeat(mean[:, None], future_frames, axis=1)
    linear = []
    for h in range(1, future_frames + 1):
        linear.append(np.clip(last + trend * h, 0.0, 2.0))
    out["linear_trend"] = np.stack(linear, axis=1)
    return out


def _accumulate_metrics(rows: list[dict], model_name: str, split: str, pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> None:
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


def _combine_metric_rows(rows: list[dict]) -> pd.DataFrame:
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
    for step, (x, y, m) in enumerate(loader, start=1):
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
        if step % 100 == 0:
            mode = "train" if train else "eval"
            print(f"[fullfield][{mode}] batches={step}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_model(model, loader, device: str, split: str, name: str) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for step, (x, y, m) in enumerate(loader, start=1):
            pred = model(x.to(device)).detach().cpu().numpy()
            _accumulate_metrics(rows, name, split, pred, y.numpy(), m.numpy())
            if step % 100 == 0:
                print(f"[fullfield][{name}][{split}] eval batches={step}/{len(loader)}", flush=True)
    return _combine_metric_rows(rows)


def evaluate_baselines(loader, split: str, future_frames: int) -> pd.DataFrame:
    rows = []
    for step, (x, y, m) in enumerate(loader, start=1):
        x_np = x.numpy()
        y_np = y.numpy()
        m_np = m.numpy()
        for name, pred in baseline_predictions(x_np, future_frames).items():
            _accumulate_metrics(rows, name, split, pred, y_np, m_np)
        if step % 100 == 0:
            print(f"[fullfield][baseline][{split}] batches={step}/{len(loader)}", flush=True)
    return _combine_metric_rows(rows)


def _load_split_df(args: argparse.Namespace) -> pd.DataFrame:
    split_path = Path(args.run_dir) / "station_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"station split not found: {split_path}")
    split_df = pd.read_csv(split_path, dtype={"station_sn": str})
    split_df["lon"] = pd.to_numeric(split_df["lon"], errors="coerce")
    split_df["lat"] = pd.to_numeric(split_df["lat"], errors="coerce")
    split_df = split_df.dropna(subset=["station_sn", "lon", "lat"]).drop_duplicates("station_sn").reset_index(drop=True)
    return split_df


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
        raise RuntimeError("PyTorch is required for full-field DensePV temporal forecasting") from exc

    split_df = _load_split_df(args)
    # Full-field sanity check: all stations are visible in every split.
    split_df["split"] = "all"
    series, load_status = load_station_series(split_df, args)
    load_status.to_csv(out_dir / "station_csv_load_status.csv", index=False)
    kpi = estimate_kpi(split_df, series, args)
    pd.DataFrame([{"station_sn": k, "kpi": v} for k, v in sorted(kpi.items())]).to_csv(out_dir / "station_kpi_geo_baselines.csv", index=False)
    bounds = _bounds(split_df)

    time_df = build_issue_time_table(args)
    time_df.to_csv(out_dir / "fullfield_temporal_issue_time_index.csv", index=False)
    ds = {}
    dl = {}
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        ds[split] = FullFieldTemporalDataset(time_df[time_df["time_split"] == split], split_df, series, kpi, bounds, args, frame_cache)
        dl[split] = DataLoader(
            ds[split],
            batch_size=int(args.batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
        )
        print(f"[fullfield] {split} samples={len(ds[split])}", flush=True)

    all_eval = []
    for split in ["val", "test"]:
        base_df = evaluate_baselines(dl[split], split, int(args.future_frames))
        all_eval.append(base_df)
        print(f"==== baselines {split} ====")
        print(base_df[base_df["horizon"] == "all"].to_string(index=False))

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model_names = [x.strip() for x in str(args.models).split(",") if x.strip()]
    train_logs = []
    for model_name in model_names:
        print(f"==== full-field temporal model: {model_name} ====", flush=True)
        model = build_model(model_name, int(args.future_frames)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        best_val = float("inf")
        bad = 0
        for epoch in range(1, int(args.epochs) + 1):
            tr = run_epoch(model, dl["train"], optimizer, device, train=True)
            va = run_epoch(model, dl["val"], optimizer, device, train=False)
            train_logs.append({"model": model_name, "epoch": epoch, "train_loss": tr, "val_loss": va})
            print(f"[fullfield][{model_name}] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)
            if va < best_val:
                best_val = va
                bad = 0
                torch.save({"model_state": model.state_dict(), "args": vars(args), "model_name": model_name}, out_dir / f"{model_name}_best.pt")
            else:
                bad += 1
                if bad >= int(args.patience):
                    break
        ckpt = torch.load(out_dir / f"{model_name}_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        for split in ["val", "test"]:
            eval_df = evaluate_model(model, dl[split], device, split, model_name)
            all_eval.append(eval_df)
            print(f"==== {model_name} {split} ====")
            print(eval_df[eval_df["horizon"] == "all"].to_string(index=False))

    pd.DataFrame(train_logs).to_csv(out_dir / "train_log.csv", index=False)
    summary = pd.concat(all_eval, ignore_index=True)
    summary = summary.sort_values(["eval_split", "horizon", "mae_field", "model"]).reset_index(drop=True)
    summary.to_csv(out_dir / "fullfield_temporal_eval_summary.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "task": "full-station DensePV field forecasting, no spatial holdout",
            "input_shape": "[2, past_frames, grid, grid] = [KPI-adjusted CSI field/support, time, y, x]",
            "target_shape": "[future_frames, grid, grid]",
            "temporal_split": {
                "train_start": str(args.time_start),
                "train_end": str(args.train_end),
                "val_start": "2023-01-14 00:00:00",
                "val_end": str(args.val_end),
                "test_start": "2023-03-09 00:00:00",
                "test_end": str(args.time_end),
            },
            "window_rule": "all 12 history frames and all 8 future frames must be between 07:00 and 16:45 and within the same calendar day",
            "n_stations": int(len(split_df)),
            "frame_cache_unique_frames": int(len(frame_cache)),
            "bounds": {"lon_min": bounds[0], "lon_max": bounds[1], "lat_min": bounds[2], "lat_max": bounds[3]},
        },
    )
    print("==== Full-field DensePV temporal aggregate summary ====")
    print(summary[summary["horizon"] == "all"].to_string(index=False))
    print(f"saved: {out_dir / 'fullfield_temporal_eval_summary.csv'}")


if __name__ == "__main__":
    main()

