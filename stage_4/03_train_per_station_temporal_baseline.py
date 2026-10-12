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

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent / "stage_3"
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from train_fullfield_dense_temporal_forecast import _parse_time_list
from visualize_regional_split_frames import ensure_dir, write_json
from stage4_paths import (
    PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
    TARGET_ISSUE_TIME_INDEX,
    TARGET_SELECTION_DIR,
    TRAIN_PV_DIR,
    VAL_PV_DIR,
    ensure_stage4_dirs,
)
from stage4_metrics import regression_metric_extras

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "selection_dir": str(TARGET_SELECTION_DIR),
    "time_index_csv": str(TARGET_ISSUE_TIME_INDEX),
    "out_dir": str(PER_STATION_TEMPORAL_BASELINE_RUN_DIR),
    "train_pv_dir": str(TRAIN_PV_DIR),
    "val_pv_dir": str(VAL_PV_DIR),
    "capacity_scale": 1000.0,
    "past_frames": 12,
    "future_frames": 8,
    "hidden_dim": 96,
    "future_hidden_dim": 64,
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

TIME_CANDIDATES = ["date", "datetime", "timestamp", "time", "ds", "采集时间", "时间", "日期"]
PV_CANDIDATES = ["elec", "pv", "power", "active_power", "发电功率", "功率", "出力"]
CAPACITY_CANDIDATES = ["Capacity", "capacity", "cap", "装机容量", "容量"]

HIST_WEATHER_ALIASES = {
    "radiation": ["radiation", "fore_radiation", "nwp_radiation", "SSRD", "ssrd"],
    "pressure": ["surface_pressure", "fore_pressure", "fore_pressure2", "pressure", "nwp_pressure"],
    "humidity": ["humidity2", "fore_humidity", "fore_humidity2", "humidity", "nwp_humidity"],
    "temperature": ["temperature10", "fore_temperature", "fore_temperature10", "temperature", "nwp_temperature"],
    "direction10": ["direction10", "fore_direction10", "nwp_direction10", "fore_wind_direction10"],
    "speed10": ["speed10", "fore_speed10", "nwp_speed10", "fore_wind_speed10"],
    "tcc": ["tcc", "fore_tcc", "nwp_tcc", "TCC"],
    "solar_zenith": ["solar_zenith", "nwp_solar_zenith"],
    "clearsky_ghi": ["clearsky_ghi", "nwp_clearsky_ghi"],
}

FUTURE_NWP_NAMES = ["radiation", "pressure", "humidity", "temperature", "direction10", "speed10", "tcc", "solar_zenith", "clearsky_ghi"]


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent per-station temporal baselines on the new corrected station time series. "
            "Inputs: historical PV_norm + historical weather/NWP sequence, plus future own-station NWP/time/solar. "
            "Target: future PV_norm."
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
        elif key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir"}:
            parser.add_argument(f"--{key}", type=Path, default=Path(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in columns:
        low = str(col).strip().lower()
        if any(cand.lower() in low for cand in candidates):
            return str(col)
    return None


def _time_solar_point(ts: pd.Timestamp, lon: float, lat: float) -> np.ndarray:
    ts = pd.Timestamp(ts)
    minute = float(ts.hour * 60 + ts.minute)
    phase = 2.0 * math.pi * minute / (24.0 * 60.0)
    tod_sin = math.sin(phase)
    tod_cos = math.cos(phase)
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
    time_offset = eq_time + 4.0 * float(lon) - 60.0 * 8.0
    true_solar_min = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_min / 4.0 - 180.0)
    lat_rad = math.radians(float(lat))
    sin_elev = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    solar = min(max(float(sin_elev), 0.0), 1.2)
    return np.asarray([tod_sin, tod_cos, solar], dtype=np.float32)


def _station_path(sn: str, train_dir: Path, val_dir: Path) -> Path | None:
    p = train_dir / f"{sn}.csv"
    if p.exists():
        return p
    p = val_dir / f"{sn}.csv"
    if p.exists():
        return p
    return None


def load_station_table(sn: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    path = _station_path(sn, Path(args.train_pv_dir), Path(args.val_pv_dir))
    if path is None:
        raise FileNotFoundError(f"station CSV not found for {sn}")
    raw = pd.read_csv(path)
    columns = [str(c) for c in raw.columns]
    time_col = "date" if "date" in raw.columns else _find_col(columns, TIME_CANDIDATES)
    pv_col = "elec" if "elec" in raw.columns else _find_col(columns, PV_CANDIDATES)
    cap_col = "Capacity" if "Capacity" in raw.columns else _find_col(columns, CAPACITY_CANDIDATES)
    if time_col is None or pv_col is None or cap_col is None:
        raise ValueError(f"{path} misses time/pv/capacity columns; columns={list(raw.columns)}")
    cap = pd.to_numeric(raw[cap_col], errors="coerce").astype(float)
    cap_value = float(cap.dropna().median()) if cap.dropna().size else float("nan")
    if not np.isfinite(cap_value) or cap_value <= 0:
        raise ValueError(f"{path} has invalid capacity column {cap_col}")
    cap_scale = cap_value * float(args.capacity_scale)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[time_col], errors="coerce"),
            "pv_norm": pd.to_numeric(raw[pv_col], errors="coerce").astype(float) / max(cap_scale, 1e-6),
        }
    )
    feature_cols = {}
    for name, aliases in HIST_WEATHER_ALIASES.items():
        col = _find_col(columns, aliases)
        if col is not None:
            df[name] = pd.to_numeric(raw[col], errors="coerce").astype(float)
            feature_cols[name] = col
    missing = [name for name in FUTURE_NWP_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"{path} misses required corrected weather/NWP columns: {missing}; mapped={feature_cols}")
    df = df.dropna(subset=["timestamp", "pv_norm"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    before = len(df)
    df = df[(df["pv_norm"] >= 0.0) & (df["pv_norm"] <= float(args.pv_health_max))]
    if bool(args.health_filter):
        pv = pd.to_numeric(df["pv_norm"], errors="coerce").astype(float)
        active = pv >= float(args.pv_health_active_threshold)
        win = max(2, int(args.pv_health_flatline_window))
        rr = pv.rolling(win, center=True, min_periods=win).max() - pv.rolling(win, center=True, min_periods=win).min()
        flat_bad = active & (rr <= float(args.pv_health_flatline_range))
        df = df.loc[~flat_bad.fillna(False)].copy()
    meta = {
        "station_sn": sn,
        "csv_path": str(path),
        "time_col": str(time_col),
        "pv_col": str(pv_col),
        "capacity_col": str(cap_col),
        "capacity_raw": float(cap_value),
        "capacity_scale": float(cap_scale),
        "rows_before_health": int(before),
        "rows_after_health": int(len(df)),
        "feature_columns": feature_cols,
    }
    return df, meta


def _safe_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _split_time_df(time_df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = time_df[time_df["time_split"].astype(str) == str(split)].copy().reset_index(drop=True)
    out["issue_time"] = pd.to_datetime(out["issue_time"])
    return out


class StationTemporalDataset:
    def __init__(self, station_sn: str, station_df: pd.DataFrame, station_row: pd.Series, time_df: pd.DataFrame, stats: dict, args: argparse.Namespace):
        self.station_sn = str(station_sn)
        self.df = station_df
        self.row = station_row
        self.time_df = time_df.reset_index(drop=True)
        self.stats = stats
        self.args = args
        self.samples = []
        self._build_samples()

    def _has_all(self, times: list[pd.Timestamp]) -> bool:
        return all(pd.Timestamp(t) in self.df.index for t in times)

    def _build_samples(self) -> None:
        for i, row in self.time_df.iterrows():
            past = _parse_time_list(row["past_times"])
            future = _parse_time_list(row["future_times"])
            if self._has_all(past) and self._has_all(future):
                self.samples.append(int(i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])
        hist_raw = self.df.loc[past_times, ["pv_norm", *FUTURE_NWP_NAMES]].to_numpy(np.float32)
        fut_raw = self.df.loc[future_times, FUTURE_NWP_NAMES].to_numpy(np.float32)
        y_raw = self.df.loc[future_times, "pv_norm"].to_numpy(np.float32)
        hist_z = (hist_raw - self.stats["hist_mean"]) / self.stats["hist_std"]
        fut_z = (fut_raw - self.stats["future_mean"]) / self.stats["future_std"]
        time_solar = np.stack([_time_solar_point(ts, float(self.row["lon"]), float(self.row["lat"])) for ts in future_times], axis=0)
        future_feat = np.concatenate([fut_z, time_solar], axis=1).astype(np.float32)
        y_z = (y_raw - float(self.stats["pv_mean"])) / float(self.stats["pv_std"])
        hist_pv = hist_raw[:, 0].astype(np.float32)
        return {
            "hist": hist_z.astype(np.float32),
            "future": future_feat.astype(np.float32),
            "y_z": y_z.astype(np.float32),
            "y": y_raw.astype(np.float32),
            "hist_pv": hist_pv,
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def make_station_stats(df: pd.DataFrame, train_time_df: pd.DataFrame) -> dict:
    secs = set()
    for _, row in train_time_df.iterrows():
        for ts in _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"]):
            secs.add(pd.Timestamp(ts))
    train_rows = df.loc[df.index.isin(secs)].copy()
    if train_rows.empty:
        train_rows = df.copy()
    hist_arr = train_rows[["pv_norm", *FUTURE_NWP_NAMES]].to_numpy(np.float64)
    future_arr = train_rows[FUTURE_NWP_NAMES].to_numpy(np.float64)
    hist_mean, hist_std = _safe_stats(hist_arr)
    future_mean, future_std = _safe_stats(future_arr)
    pv = train_rows["pv_norm"].to_numpy(np.float64)
    pv = pv[np.isfinite(pv)]
    pv_mean = float(np.nanmean(pv)) if pv.size else 0.0
    pv_std = float(np.nanstd(pv)) if pv.size else 1.0
    if not np.isfinite(pv_std) or pv_std < 1e-6:
        pv_std = 1.0
    return {
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "future_mean": future_mean,
        "future_std": future_std,
        "pv_mean": pv_mean,
        "pv_std": pv_std,
    }


def make_model(hist_dim: int, future_dim: int, future_frames: int, args: argparse.Namespace):
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.future_frames = int(future_frames)
            hidden = int(args.hidden_dim)
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
            self.horizon_embed = nn.Embedding(int(future_frames), hidden // 2)
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + hidden // 2, hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, hist, future):
            enc, _ = self.hist_encoder(hist)
            hist_ctx = self.hist_proj(enc[:, -1])
            fut_ctx = self.future_mlp(future)
            h_ids = torch.arange(self.future_frames, device=future.device)
            h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
            hist_rep = hist_ctx[:, None].expand(-1, self.future_frames, -1)
            out = self.decoder(torch.cat([hist_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)
            return out

    return Model()


def weighted_loss(pred_z, target_z, target_raw, hist_pv_raw, args):
    import torch
    import torch.nn.functional as F

    last = hist_pv_raw[:, -1][:, None]
    true_delta = torch.abs(target_raw - last)
    weights = 1.0 + float(args.big_change_loss_weight) * (true_delta >= float(args.large_delta_threshold)).float()
    loss = F.smooth_l1_loss(pred_z, target_z, reduction="none") * weights
    return loss.sum() / torch.clamp(weights.sum(), min=1.0)


def collate(batch: list[dict]) -> dict:
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


def _configure_torch_runtime(torch, args: argparse.Namespace) -> None:
    if bool(getattr(args, "disable_cudnn_rnn", True)) and torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print(
            "[station_temporal] disabled torch.backends.cudnn for small LSTM stability",
            flush=True,
        )


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def train_one(model, loaders, args, device: str, ckpt_path: Path) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

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
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            opt.zero_grad(set_to_none=True)
            pred = model(hist, fut)
            loss = weighted_loss(pred, y, y_raw, hist_pv, args)
            loss.backward()
            if float(getattr(args, "grad_clip", 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, loaders["val"], args, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[station_temporal] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[station_temporal] early stop at epoch={epoch}", flush=True)
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
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            losses.append(float(weighted_loss(model(hist, fut), y, y_raw, hist_pv, args).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(
    model,
    dataset: StationTemporalDataset,
    loader,
    stats: dict,
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
            pred_z = model(hist, fut).detach().cpu().numpy()
            pred = pred_z * pv_std + pv_mean
            y = np.asarray(batch["y"], dtype=np.float32)
            hist_pv = np.asarray(batch["hist_pv"], dtype=np.float32)
            issue_times = batch["issue_time"]
            for i in range(pred.shape[0]):
                last = float(hist_pv[i, -1])
                for h in range(pred.shape[1]):
                    e = float(pred[i, h] - y[i, h])
                    delta = float(y[i, h] - last)
                    row = {
                        "station_sn": dataset.station_sn,
                        "eval_split": split,
                        "issue_time": str(issue_times[i]),
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
                    }
                    for k, value in enumerate(hist_pv[i], start=1):
                        row[f"hist_pv_norm_{k:02d}"] = float(value)
                    fut_raw_z = np.asarray(batch["future"][i, h], dtype=np.float32)
                    for j, name in enumerate(FUTURE_NWP_NAMES):
                        row[f"future_{name}_z"] = float(fut_raw_z[j])
                    row["future_tod_sin"] = float(fut_raw_z[len(FUTURE_NWP_NAMES)])
                    row["future_tod_cos"] = float(fut_raw_z[len(FUTURE_NWP_NAMES) + 1])
                    row["future_solar_elevation"] = float(fut_raw_z[len(FUTURE_NWP_NAMES) + 2])
                    details.append(row)
    detail_df = pd.DataFrame(details)
    metric_rows = []
    if not detail_df.empty:
        for horizon, g in list(detail_df.groupby("horizon")) + [("all", detail_df)]:
            row = {
                "station_sn": dataset.station_sn,
                "model": "per_station_temporal_baseline",
                "eval_split": split,
                "horizon": horizon,
                "n_samples": int(len(g)),
                "mae_pv_norm": float(g["abs_err_pv_norm"].mean()),
                "rmse_pv_norm": float(math.sqrt(g["sq_err_pv_norm"].mean())),
                "high_change_mae_pv_norm": float(g.loc[g["large_change"], "abs_err_pv_norm"].mean()) if bool(g["large_change"].any()) else float("nan"),
                "high_change_n": int(g["large_change"].sum()),
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
    labels = [f"h{i}" for i in range(1, int(args.future_frames) + 1)]
    hist_cols = [f"hist_pv_norm_{k:02d}" for k in range(1, int(args.past_frames) + 1)]
    nwp_cols = [f"future_{name}_z" for name in FUTURE_NWP_NAMES]
    for i, issue in enumerate(issues):
        sub = test_df[test_df["issue_time"].astype(str) == str(issue)].sort_values("h")
        if len(sub) != int(args.future_frames):
            continue
        x = np.arange(1, int(args.future_frames) + 1)
        hist_x = np.arange(-int(args.past_frames) + 1, 1)
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=False)
        hist = sub.iloc[0][hist_cols].to_numpy(np.float64)
        axes[0].plot(hist_x, hist, color="#7f7f7f", marker=".", linewidth=2.0, label="history PV_norm")
        axes[0].axvline(0, color="0.75", linestyle="--", linewidth=1.0)
        axes[0].plot(x, sub["true_pv_norm"].to_numpy(np.float64), color="black", marker="o", linewidth=2.2, label="truth")
        axes[0].plot(x, sub["pred_pv_norm"].to_numpy(np.float64), color="#1f77b4", marker="o", linewidth=2.0, label="baseline")
        axes[0].set_title(f"Per-station temporal baseline | {station_sn} | issue={issue}")
        axes[0].set_ylabel("PV_norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        err = sub["pred_pv_norm"].to_numpy(np.float64) - sub["true_pv_norm"].to_numpy(np.float64)
        axes[1].bar(x, err, color=np.where(err >= 0, "#d62728", "#1f77b4"), alpha=0.85)
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("pred - truth")
        axes[1].grid(alpha=0.25)

        for col in nwp_cols:
            if col in sub.columns:
                axes[2].plot(x, sub[col].to_numpy(np.float64), marker=".", linewidth=1.4, label=col.replace("future_", "").replace("_z", " z"))
        axes[2].plot(x, sub["future_solar_elevation"].to_numpy(np.float64), color="goldenrod", linewidth=1.8, marker=".", label="solar elev")
        axes[2].set_ylabel("future NWP z / solar")
        axes[2].set_xlabel("forecast horizon")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best", ncol=3, fontsize=7)
        fig.tight_layout()
        safe_issue = str(issue).replace(":", "").replace(" ", "_")
        fig.savefig(fig_dir / f"temporal_baseline_curve_{i:03d}_{safe_issue}.png", dpi=int(args.dpi))
        plt.close(fig)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
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
            }
        )
    return pd.DataFrame(rows).sort_values(["eval_split", "horizon"]).reset_index(drop=True)


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    ensure_stage4_dirs()
    out_dir = ensure_dir(args.out_dir)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for station temporal baseline training.") from exc

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

    all_metrics, all_details, all_logs, all_meta = [], [], [], []
    for idx, row in targets.reset_index(drop=True).iterrows():
        sn = str(row["station_sn"])
        print(f"==== per-station temporal baseline {idx + 1}/{len(targets)}: {sn} ====", flush=True)
        station_dir = ensure_dir(out_dir / "per_station" / sn)
        station_df, meta = load_station_table(sn, args)
        all_meta.append(meta)
        train_time = _split_time_df(time_df, "train")
        stats = make_station_stats(station_df, train_time)
        datasets, loaders = {}, {}
        for split in ["train", "val", "test"]:
            datasets[split] = StationTemporalDataset(sn, station_df, row, _split_time_df(time_df, split), stats, args)
            print(f"[station_temporal] {sn} split={split} samples={len(datasets[split])}", flush=True)
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
                collate_fn=collate,
            )
        if len(datasets["train"]) < int(args.min_train_samples):
            print(f"[station_temporal] skip {sn}: train samples={len(datasets['train'])} < {int(args.min_train_samples)}", flush=True)
            continue
        model = make_model(1 + len(FUTURE_NWP_NAMES), len(FUTURE_NWP_NAMES) + 3, int(args.future_frames), args)
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
                "hist_features": ["pv_norm", *FUTURE_NWP_NAMES],
                "future_features": [*FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
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

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    metrics_df.to_csv(out_dir / "per_station_temporal_baseline_metrics.csv", index=False)
    detail_df.to_csv(out_dir / "per_station_temporal_baseline_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / "per_station_temporal_baseline_train_log.csv", index=False)
    agg = aggregate(metrics_df)
    agg.to_csv(out_dir / "per_station_temporal_baseline_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Per-target-station baseline using only single-point historical weather/PV and future own-station NWP.",
            "target_definition": "PV_norm = raw PV / (Capacity * capacity_scale); labels and PV history use z-scored PV_norm.",
            "architecture": "LSTM historical sequence encoder + future NWP/time-solar MLP + horizon embedding decoder.",
            "target_stations": targets["station_sn"].astype(str).tolist(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print("==== Per-station temporal baseline aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"saved: {out_dir / 'per_station_temporal_baseline_aggregate.csv'}")
    print(f"figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
