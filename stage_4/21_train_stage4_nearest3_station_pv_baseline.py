#!/usr/bin/env python3
"""Train a Stage 4 nearest-neighbor station-PV replacement baseline.

This baseline keeps the per-target station temporal inputs unchanged:

Past stream:
    target-station historical PV_norm + historical weather columns

Future stream:
    target-station forecast-weather columns + deterministic time/solar features

Replacement signal:
    PV_norm from the geographically nearest source stations. By default this
    uses each neighbor's latest available historical PV value repeated across
    future horizons, so it stays operationally fair. The optional future_truth
    mode can be used as an upper-bound neighbor-station integration check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_metrics import regression_metric_extras  # noqa: E402
from stage4_paths import RUNS_DIR, TARGET_EXCLUDED_RUN_DIR  # noqa: E402


def _load_stage4_temporal_module():
    path = SCRIPT_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_temporal_for_nearest3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_temporal_for_nearest3"] = module
    spec.loader.exec_module(module)
    return module


f03 = _load_stage4_temporal_module()

MODEL_NAME = "nearest3_station_pv"

DEFAULTS = dict(f03.PYCHARM_DEFAULTS)
DEFAULTS.update(
    {
        "out_dir": "",
        "epochs": 60,
        "patience": 10,
        "hidden_dim": 96,
        "future_hidden_dim": 64,
        "field_hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "batch_size": 128,
        "neighbor_count": 3,
        "neighbor_pool_size": 24,
        "neighbor_source": "last_observed",
        "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
        "sample_count": 0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULTS.items():
        if key == "neighbor_source":
            parser.add_argument("--neighbor_source", choices=["future_truth", "last_observed"], default=str(value))
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=bool(value))
            if not str(key).startswith("disable_"):
                parser.add_argument(f"--disable_{key}", action="store_false", dest=key)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir", "target_run_dir"}:
            parser.add_argument(f"--{key}", type=Path, default=Path(value) if str(value) else Path(""))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_out_dir(args: argparse.Namespace) -> Path:
    out_arg = str(args.out_dir)
    if out_arg and out_arg not in {".", "./"}:
        return Path(args.out_dir)
    return RUNS_DIR / f"21_nearest3_station_pv_baseline_seed{int(args.seed)}"


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.maximum(a, 0.0)))


def read_station_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"station_sn": str})
    required = {"station_sn", "lon", "lat"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    return df.dropna(subset=["station_sn", "lon", "lat"]).drop_duplicates("station_sn").reset_index(drop=True)


def load_source_station_table(args: argparse.Namespace, targets: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        Path(args.target_run_dir) / "target_excluded_station_split.csv",
        Path(args.target_run_dir) / "station_split.csv",
        Path(args.selection_dir) / "target_excluded_station_split.csv",
        Path(args.selection_dir) / "station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            df = read_station_csv(path)
            target_sns = set(targets["station_sn"].astype(str))
            out = df[~df["station_sn"].astype(str).isin(target_sns)].copy()
            if len(out) >= int(args.neighbor_count):
                print(f"[nearest3] source stations: {len(out)} from {path}", flush=True)
                return out.reset_index(drop=True)
    raise FileNotFoundError(f"Could not find a source station split. Checked: {[str(p) for p in candidates]}")


def ranked_neighbors(targets: pd.DataFrame, sources: pd.DataFrame, args: argparse.Namespace) -> dict[str, list[dict]]:
    src_lon = sources["lon"].astype(float).to_numpy()
    src_lat = sources["lat"].astype(float).to_numpy()
    pool_size = max(int(args.neighbor_count), int(args.neighbor_pool_size))
    out: dict[str, list[dict]] = {}
    for _, row in targets.iterrows():
        dist = haversine_km(float(row["lon"]), float(row["lat"]), src_lon, src_lat)
        order = np.argsort(dist)[:pool_size]
        out[str(row["station_sn"])] = [
            {
                "station_sn": str(sources.iloc[int(i)]["station_sn"]),
                "lon": float(sources.iloc[int(i)]["lon"]),
                "lat": float(sources.iloc[int(i)]["lat"]),
                "distance_km": float(dist[int(i)]),
            }
            for i in order
        ]
    return out


def load_neighbor_data_for_target(
    target_sn: str,
    candidates: list[dict],
    cache: dict[str, tuple[pd.DataFrame, dict, dict]],
    train_time: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict]:
    chosen: list[dict] = []
    for rec in candidates:
        sn = str(rec["station_sn"])
        if sn not in cache:
            try:
                df, meta = f03.load_station_table(sn, args)
                stats = f03.make_station_stats(df, train_time)
                cache[sn] = (df, stats, meta)
            except Exception as exc:
                print(f"[nearest3] skip neighbor {sn} for {target_sn}: {exc}", flush=True)
                cache[sn] = (pd.DataFrame(), {}, {})
        df, stats, meta = cache[sn]
        if df.empty:
            continue
        row = dict(rec)
        row["meta"] = meta
        row["stats"] = stats
        chosen.append(row)
        if len(chosen) >= int(args.neighbor_count):
            break
    if len(chosen) < int(args.neighbor_count):
        raise RuntimeError(f"Target {target_sn} has only {len(chosen)} loadable neighbors.")
    return chosen


class StationTemporalNearestPVDataset:
    def __init__(
        self,
        station_sn: str,
        station_df: pd.DataFrame,
        station_row: pd.Series,
        time_df: pd.DataFrame,
        stats: dict,
        neighbor_records: list[dict],
        neighbor_cache: dict[str, tuple[pd.DataFrame, dict, dict]],
        args: argparse.Namespace,
    ):
        self.station_sn = str(station_sn)
        self.df = station_df
        self.row = station_row
        self.time_df = time_df.reset_index(drop=True)
        self.stats = stats
        self.neighbor_records = neighbor_records
        self.neighbor_cache = neighbor_cache
        self.args = args
        self.samples: list[int] = []
        self._build_samples()

    def _has_all_target(self, times: list[pd.Timestamp]) -> bool:
        return all(pd.Timestamp(t) in self.df.index for t in times)

    def _build_samples(self) -> None:
        for i, row in self.time_df.iterrows():
            past = f03._parse_time_list(row["past_times"])
            future = f03._parse_time_list(row["future_times"])
            if self._has_all_target(past + future):
                self.samples.append(int(i))

    def __len__(self) -> int:
        return len(self.samples)

    def _neighbor_values(self, past_times: list[pd.Timestamp], future_times: list[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((int(self.args.future_frames), int(self.args.neighbor_count)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.float32)
        for j, rec in enumerate(self.neighbor_records[: int(self.args.neighbor_count)]):
            sn = str(rec["station_sn"])
            ndf, nstats, _ = self.neighbor_cache[sn]
            if str(self.args.neighbor_source) == "last_observed":
                valid_past = [ts for ts in past_times if pd.Timestamp(ts) in ndf.index]
                if not valid_past:
                    continue
                raw = float(ndf.loc[valid_past[-1], "pv_norm"])
                z = (raw - float(nstats["pv_mean"])) / float(nstats["pv_std"])
                values[:, j] = float(z)
                mask[:, j] = 1.0
            else:
                for h, ts in enumerate(future_times):
                    if pd.Timestamp(ts) in ndf.index:
                        raw = float(ndf.loc[pd.Timestamp(ts), "pv_norm"])
                        values[h, j] = float((raw - float(nstats["pv_mean"])) / float(nstats["pv_std"]))
                        mask[h, j] = 1.0
        return values, mask

    def __getitem__(self, idx: int) -> dict:
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        past_times = f03._parse_time_list(row["past_times"])
        future_times = f03._parse_time_list(row["future_times"])
        hist_raw = self.df.loc[past_times, ["pv_norm", *f03.FUTURE_NWP_NAMES]].to_numpy(np.float32)
        fut_raw = self.df.loc[future_times, f03.FUTURE_NWP_NAMES].to_numpy(np.float32)
        y_raw = self.df.loc[future_times, "pv_norm"].to_numpy(np.float32)
        hist_z = (hist_raw - self.stats["hist_mean"]) / self.stats["hist_std"]
        fut_z = (fut_raw - self.stats["future_mean"]) / self.stats["future_std"]
        time_solar = np.stack(
            [f03._time_solar_point(ts, float(self.row["lon"]), float(self.row["lat"])) for ts in future_times],
            axis=0,
        )
        future_feat = np.concatenate([fut_z, time_solar], axis=1).astype(np.float32)
        neighbor_z, neighbor_mask = self._neighbor_values(past_times, future_times)
        neighbor_feat = np.concatenate([neighbor_z, neighbor_mask], axis=1).astype(np.float32)
        y_z = (y_raw - float(self.stats["pv_mean"])) / float(self.stats["pv_std"])
        return {
            "hist": hist_z.astype(np.float32),
            "future": future_feat,
            "neighbor": neighbor_feat,
            "neighbor_raw_z": neighbor_z.astype(np.float32),
            "neighbor_mask": neighbor_mask.astype(np.float32),
            "y_z": y_z.astype(np.float32),
            "y": y_raw.astype(np.float32),
            "hist_pv": hist_raw[:, 0].astype(np.float32),
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "neighbor": np.stack([b["neighbor"] for b in batch]).astype(np.float32),
        "neighbor_raw_z": np.stack([b["neighbor_raw_z"] for b in batch]).astype(np.float32),
        "neighbor_mask": np.stack([b["neighbor_mask"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


def make_model(hist_dim: int, future_dim: int, neighbor_dim: int, future_frames: int, args: argparse.Namespace):
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
            self.neighbor_mlp = nn.Sequential(
                nn.Linear(int(neighbor_dim), int(args.field_hidden_dim)),
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

        def forward(self, hist, future, neighbor):
            enc, _ = self.hist_encoder(hist)
            hist_ctx = self.hist_proj(enc[:, -1])
            fut_ctx = self.future_mlp(future)
            neighbor_ctx = self.neighbor_mlp(neighbor)
            h_ids = torch.arange(self.future_frames, device=future.device)
            h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
            hist_rep = hist_ctx[:, None].expand(-1, self.future_frames, -1)
            return self.decoder(torch.cat([hist_rep, fut_ctx, neighbor_ctx, h_emb], dim=-1)).squeeze(-1)

    return Model()


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_loss(pred_z, target_z, target_raw, hist_pv_raw, args):
    import torch
    import torch.nn.functional as F

    last = hist_pv_raw[:, -1][:, None]
    true_delta = torch.abs(target_raw - last)
    weights = 1.0 + float(args.big_change_loss_weight) * (true_delta >= float(args.large_delta_threshold)).float()
    return (F.smooth_l1_loss(pred_z, target_z, reduction="none") * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def eval_loss(model, loader, args, device: str) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            neighbor = _tensor(batch, "neighbor", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            losses.append(float(weighted_loss(model(hist, fut, neighbor), y, y_raw, hist_pv, args).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


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
            neighbor = _tensor(batch, "neighbor", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            opt.zero_grad(set_to_none=True)
            loss = weighted_loss(model(hist, fut, neighbor), y, y_raw, hist_pv, args)
            loss.backward()
            if float(getattr(args, "grad_clip", 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = eval_loss(model, loaders["val"], args, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{MODEL_NAME}] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[{MODEL_NAME}] early stop at epoch={epoch}", flush=True)
                break
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(logs)


def evaluate(model, dataset, loader, stats: dict, split: str, args, device: str, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    model.eval()
    details = []
    pv_mean = float(stats["pv_mean"])
    pv_std = float(stats["pv_std"])
    neighbor_sns = [str(x["station_sn"]) for x in dataset.neighbor_records[: int(args.neighbor_count)]]
    neighbor_dist = [float(x["distance_km"]) for x in dataset.neighbor_records[: int(args.neighbor_count)]]
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            neighbor = _tensor(batch, "neighbor", device, torch)
            pred_z = model(hist, fut, neighbor).detach().cpu().numpy()
            pred = pred_z * pv_std + pv_mean
            y = np.asarray(batch["y"], dtype=np.float32)
            hist_pv = np.asarray(batch["hist_pv"], dtype=np.float32)
            neighbor_raw_z = np.asarray(batch["neighbor_raw_z"], dtype=np.float32)
            neighbor_mask = np.asarray(batch["neighbor_mask"], dtype=np.float32)
            issue_times = batch["issue_time"]
            for i in range(pred.shape[0]):
                last = float(hist_pv[i, -1])
                for h in range(pred.shape[1]):
                    e = float(pred[i, h] - y[i, h])
                    delta = float(y[i, h] - last)
                    row = {
                        "station_sn": dataset.station_sn,
                        "model": MODEL_NAME,
                        "eval_split": split,
                        "issue_time": str(issue_times[i]),
                        "h": int(h + 1),
                        "horizon": f"h{h + 1}",
                        "pred_pv_norm": float(pred[i, h]),
                        "true_pv_norm": float(y[i, h]),
                        "capacity_raw": float(meta.get("capacity_raw", np.nan)),
                        "capacity_scale": float(meta.get("capacity_scale", np.nan)),
                        "abs_err_pv_norm": abs(e),
                        "sq_err_pv_norm": e * e,
                        "true_delta_pv_norm": delta,
                        "large_change": abs(delta) >= float(args.large_delta_threshold),
                    }
                    for k, value in enumerate(hist_pv[i], start=1):
                        row[f"hist_pv_norm_{k:02d}"] = float(value)
                    for j, sn in enumerate(neighbor_sns, start=1):
                        row[f"neighbor{j}_station_sn"] = sn
                        row[f"neighbor{j}_distance_km"] = float(neighbor_dist[j - 1])
                        row[f"neighbor{j}_pv_norm_z"] = float(neighbor_raw_z[i, h, j - 1])
                        row[f"neighbor{j}_mask"] = float(neighbor_mask[i, h, j - 1])
                    details.append(row)
    detail_df = pd.DataFrame(details)
    metric_rows = []
    if not detail_df.empty:
        for horizon, g in list(detail_df.groupby("horizon")) + [("all", detail_df)]:
            row = {
                "station_sn": dataset.station_sn,
                "model": MODEL_NAME,
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


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(default_out_dir(args))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Stage 4 nearest-neighbor PV baseline training.") from exc

    f03.ensure_stage4_dirs()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    f03._configure_torch_runtime(torch, args)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    targets = pd.read_csv(target_path, dtype={"station_sn": str}).reset_index(drop=True)
    sources = load_source_station_table(args, targets)
    candidate_map = ranked_neighbors(targets, sources, args)

    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    train_time = f03._split_time_df(time_df, "train")

    neighbor_cache: dict[str, tuple[pd.DataFrame, dict, dict]] = {}
    neighbor_map: dict[str, list[dict]] = {}
    all_metrics, all_details, all_logs, all_meta = [], [], [], []
    neighbor_meta_rows = []

    hist_dim = 1 + len(f03.FUTURE_NWP_NAMES)
    future_dim = len(f03.FUTURE_NWP_NAMES) + 3
    neighbor_dim = int(args.neighbor_count) * 2

    for idx, row in targets.reset_index(drop=True).iterrows():
        sn = str(row["station_sn"])
        print(f"==== Stage4 {MODEL_NAME} {idx + 1}/{len(targets)}: {sn} ====", flush=True)
        station_dir = ensure_dir(out_dir / "per_station" / sn)
        station_df, meta = f03.load_station_table(sn, args)
        all_meta.append(meta)
        stats = f03.make_station_stats(station_df, train_time)
        neighbors = load_neighbor_data_for_target(sn, candidate_map[sn], neighbor_cache, train_time, args)
        neighbor_map[sn] = neighbors
        for rank, rec in enumerate(neighbors, start=1):
            neighbor_meta_rows.append(
                {
                    "target_station_sn": sn,
                    "rank": rank,
                    "neighbor_station_sn": str(rec["station_sn"]),
                    "distance_km": float(rec["distance_km"]),
                    "neighbor_lon": float(rec["lon"]),
                    "neighbor_lat": float(rec["lat"]),
                }
            )
        datasets, loaders = {}, {}
        for split in ["train", "val", "test"]:
            datasets[split] = StationTemporalNearestPVDataset(
                sn,
                station_df,
                row,
                f03._split_time_df(time_df, split),
                stats,
                neighbors,
                neighbor_cache,
                args,
            )
            print(f"[{MODEL_NAME}] {sn} split={split} samples={len(datasets[split])}", flush=True)
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
                collate_fn=collate,
            )
        if len(datasets["train"]) < int(args.min_train_samples):
            print(f"[{MODEL_NAME}] skip {sn}: train samples={len(datasets['train'])}", flush=True)
            continue

        model = make_model(hist_dim, future_dim, neighbor_dim, int(args.future_frames), args)
        log_df = train_one(model, loaders, args, device, station_dir / "best.pt")
        log_df.insert(0, "station_sn", sn)
        log_df.insert(1, "model", MODEL_NAME)
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
                "hist_features": ["pv_norm", *f03.FUTURE_NWP_NAMES],
                "future_features": [*f03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
                "neighbor_features": ["nearest_source_station_pv_norm_z", "availability_mask"],
            },
        )

        station_details = []
        for split in ["val", "test"]:
            metrics, detail = evaluate(model, datasets[split], loaders[split], stats, split, args, device, meta)
            all_metrics.append(metrics)
            station_details.append(detail)
        station_detail = pd.concat(station_details, ignore_index=True) if station_details else pd.DataFrame()
        station_detail.to_csv(station_dir / "eval_detail.csv", index=False)
        all_details.append(station_detail)

    meta_df = pd.DataFrame(all_meta)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    agg = f03.aggregate(metrics_df) if not metrics_df.empty else pd.DataFrame()

    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    pd.DataFrame(neighbor_meta_rows).to_csv(out_dir / "nearest_neighbor_map.csv", index=False)
    metrics_df.to_csv(out_dir / f"{MODEL_NAME}_metrics.csv", index=False)
    detail_df.to_csv(out_dir / f"{MODEL_NAME}_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / f"{MODEL_NAME}_train_log.csv", index=False)
    agg.to_csv(out_dir / f"{MODEL_NAME}_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Stage 4 replacement baseline: nearest source-station PV sequence instead of PV-field patch.",
            "model": MODEL_NAME,
            "inputs": {
                "target_past_stream": ["PV_norm", *f03.FUTURE_NWP_NAMES],
                "target_future_stream": [*f03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
                "replacement_signal": f"{int(args.neighbor_count)} nearest source-station PV_norm sequences",
                "neighbor_source": str(args.neighbor_source),
            },
            "notes": (
                "Default neighbor_source=last_observed uses only the nearest source stations' latest "
                "historical PV. Optional neighbor_source=future_truth uses neighbor PV at forecast "
                "horizons and should be treated as an upper-bound neighbor-integration check."
            ),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"==== Stage 4 {MODEL_NAME} aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"detail: {out_dir / f'{MODEL_NAME}_eval_detail.csv'}")


if __name__ == "__main__":
    main()
