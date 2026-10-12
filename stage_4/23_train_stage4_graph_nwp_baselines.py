#!/usr/bin/env python3
"""Train stronger graph/NWP Stage 4 baselines: AGCRN and Graph WaveNet.

Both baselines use exactly the same graph-weather input protocol as the STGCN
forecast-weather baseline:

Past stream for each target station node:
    historical PV_norm + historical weather columns

Future stream for each target station node and horizon:
    forecast-weather columns + deterministic time/solar features

No field patch, SWR patch, or neighboring-station future PV is used.
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
from stage4_paths import RUNS_DIR  # noqa: E402


def _load_stage4_temporal_module():
    path = SCRIPT_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_temporal_for_graph_nwp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_temporal_for_graph_nwp"] = module
    spec.loader.exec_module(module)
    return module


f03 = _load_stage4_temporal_module()

MODEL_CHOICES = ["agcrn", "graph_wavenet"]
MODEL_NAMES = {"agcrn": "agcrn_nwp", "graph_wavenet": "graph_wavenet_nwp"}

DEFAULTS = dict(f03.PYCHARM_DEFAULTS)
DEFAULTS.update(
    {
        "model": "agcrn",
        "out_dir": "",
        "epochs": int(f03.PYCHARM_DEFAULTS["epochs"]),
        "patience": int(f03.PYCHARM_DEFAULTS["patience"]),
        "hidden_dim": 96,
        "future_hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "graph_batch_size": 32,
        "graph_k": 6,
        "graph_sigma_km": 18.0,
        "node_emb_dim": 16,
        "gwnet_layers": 4,
        "gwnet_kernel_size": 2,
        "sample_count": 0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULTS.items():
        if key == "model":
            parser.add_argument("--model", choices=MODEL_CHOICES, default=str(value))
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=bool(value))
            if not str(key).startswith("disable_"):
                parser.add_argument(f"--disable_{key}", action="store_false", dest=key)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir"}:
            parser.add_argument(f"--{key}", type=Path, default=Path(value) if str(value) else Path(""))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    parser.add_argument("legacy_model_arg", nargs="?", choices=MODEL_CHOICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if getattr(args, "legacy_model_arg", None):
        args.model = str(args.legacy_model_arg)
    return args


def model_name(args: argparse.Namespace) -> str:
    return MODEL_NAMES[str(args.model)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_out_dir(args: argparse.Namespace) -> Path:
    out_arg = str(args.out_dir)
    if out_arg and out_arg not in {".", "./"}:
        return Path(args.out_dir)
    return RUNS_DIR / f"23_{model_name(args)}_baseline_seed{int(args.seed)}"


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.maximum(a, 0.0)))


def build_adjacency(targets: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    lon = targets["lon"].astype(float).to_numpy()
    lat = targets["lat"].astype(float).to_numpy()
    n = len(targets)
    dist = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    sigma = max(float(args.graph_sigma_km), 1.0)
    adj = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    k = min(max(1, int(args.graph_k)), max(1, n - 1))
    mask = np.zeros_like(adj, dtype=bool)
    for i in range(n):
        nn = np.argsort(dist[i])[1 : k + 1]
        mask[i, nn] = True
    mask = mask | mask.T | np.eye(n, dtype=bool)
    adj = np.where(mask, adj, 0.0).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    deg = np.sum(adj, axis=1)
    deg = np.where(deg > 1e-6, deg, 1.0)
    d_inv = 1.0 / np.sqrt(deg)
    return (d_inv[:, None] * adj * d_inv[None, :]).astype(np.float32)


class GraphWeatherDataset:
    def __init__(self, targets, station_dfs, stats_by_sn, time_df, args):
        self.targets = targets.reset_index(drop=True)
        self.station_sns = self.targets["station_sn"].astype(str).tolist()
        self.station_dfs = station_dfs
        self.stats_by_sn = stats_by_sn
        self.time_df = time_df.reset_index(drop=True)
        self.args = args
        self.samples: list[int] = []
        self._build_samples()

    def _has_all(self, sn: str, times: list[pd.Timestamp]) -> bool:
        return all(pd.Timestamp(t) in self.station_dfs[sn].index for t in times)

    def _build_samples(self) -> None:
        for i, row in self.time_df.iterrows():
            past = f03._parse_time_list(row["past_times"])
            future = f03._parse_time_list(row["future_times"])
            if all(self._has_all(sn, past + future) for sn in self.station_sns):
                self.samples.append(int(i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        past_times = f03._parse_time_list(row["past_times"])
        future_times = f03._parse_time_list(row["future_times"])
        hist_z, hist_raw, fut_feat, y_z, y_raw = [], [], [], [], []
        for sidx, sn in enumerate(self.station_sns):
            df = self.station_dfs[sn]
            stats = self.stats_by_sn[sn]
            hist_raw_arr = df.loc[past_times, ["pv_norm", *f03.FUTURE_NWP_NAMES]].to_numpy(np.float32)
            fut_raw_arr = df.loc[future_times, f03.FUTURE_NWP_NAMES].to_numpy(np.float32)
            pv_future = df.loc[future_times, "pv_norm"].to_numpy(np.float32)
            hist_z.append((hist_raw_arr - stats["hist_mean"]) / stats["hist_std"])
            fut_z = (fut_raw_arr - stats["future_mean"]) / stats["future_std"]
            st = self.targets.iloc[sidx]
            time_solar = np.stack(
                [f03._time_solar_point(ts, float(st["lon"]), float(st["lat"])) for ts in future_times],
                axis=0,
            )
            fut_feat.append(np.concatenate([fut_z, time_solar], axis=1))
            hist_raw.append(hist_raw_arr[:, 0].astype(np.float32))
            y_z.append((pv_future - float(stats["pv_mean"])) / float(stats["pv_std"]))
            y_raw.append(pv_future)
        return {
            "hist": np.stack(hist_z, axis=0).astype(np.float32),
            "future": np.stack(fut_feat, axis=0).astype(np.float32),
            "hist_pv": np.stack(hist_raw, axis=0).astype(np.float32),
            "y_z": np.stack(y_z, axis=0).astype(np.float32),
            "y": np.stack(y_raw, axis=0).astype(np.float32),
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def collate_graph(batch: list[dict]) -> dict:
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def make_adaptive_adj(nodevec1, nodevec2):
    import torch
    import torch.nn.functional as F

    return F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)


def make_model(args: argparse.Namespace, adj: np.ndarray, hist_dim: int, future_dim: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AdaptiveGraphConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim)

        def forward(self, x, adaptive_adj):
            ax = torch.einsum("ij,bjf->bif", adaptive_adj, x)
            return self.linear(torch.cat([x, ax], dim=-1))

    class AGCRNCell(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.gate = AdaptiveGraphConv(input_dim + hidden_dim, 2 * hidden_dim)
            self.update = AdaptiveGraphConv(input_dim + hidden_dim, hidden_dim)

        def forward(self, x, state, adaptive_adj):
            gates = torch.sigmoid(self.gate(torch.cat([x, state], dim=-1), adaptive_adj))
            z, r = torch.chunk(gates, 2, dim=-1)
            candidate = torch.tanh(self.update(torch.cat([x, r * state], dim=-1), adaptive_adj))
            return z * state + (1.0 - z) * candidate

    class AGCRN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(args.hidden_dim)
            emb_dim = int(args.node_emb_dim)
            self.future_frames = int(args.future_frames)
            self.n_nodes = int(adj.shape[0])
            self.nodevec1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(emb_dim, self.n_nodes) * 0.1)
            self.cells = nn.ModuleList()
            for layer in range(max(1, int(args.num_layers))):
                self.cells.append(AGCRNCell(int(hist_dim) if layer == 0 else hidden, hidden))
            self.future_mlp = nn.Sequential(
                nn.Linear(int(future_dim), int(args.future_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(args.future_hidden_dim)),
                nn.Linear(int(args.future_hidden_dim), hidden),
                nn.GELU(),
            )
            self.horizon_embed = nn.Embedding(int(args.future_frames), max(8, hidden // 2))
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, hist, future):
            b, n, t, _ = hist.shape
            adaptive_adj = make_adaptive_adj(self.nodevec1, self.nodevec2)
            x_seq = hist
            last_state = None
            for cell in self.cells:
                state = hist.new_zeros(b, n, int(args.hidden_dim))
                outputs = []
                for ti in range(t):
                    state = cell(x_seq[:, :, ti, :], state, adaptive_adj)
                    outputs.append(state)
                x_seq = torch.stack(outputs, dim=2)
                last_state = state
            ctx = last_state
            fut = self.future_mlp(future)
            h_ids = torch.arange(self.future_frames, device=hist.device)
            h_emb = self.horizon_embed(h_ids)[None, None].expand(b, n, -1, -1)
            ctx_rep = ctx[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([ctx_rep, fut, h_emb], dim=-1)).squeeze(-1)

    class GraphWaveNet(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(args.hidden_dim)
            emb_dim = int(args.node_emb_dim)
            self.future_frames = int(args.future_frames)
            self.n_nodes = int(adj.shape[0])
            self.register_buffer("fixed_adj", torch.as_tensor(adj, dtype=torch.float32))
            self.nodevec1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(emb_dim, self.n_nodes) * 0.1)
            self.start = nn.Conv2d(int(hist_dim), hidden, kernel_size=(1, 1))
            self.filter_convs = nn.ModuleList()
            self.gate_convs = nn.ModuleList()
            self.residual_convs = nn.ModuleList()
            self.graph_convs = nn.ModuleList()
            kernel = int(args.gwnet_kernel_size)
            for layer in range(max(1, int(args.gwnet_layers))):
                dilation = 2 ** (layer % max(1, int(args.num_layers)))
                pad = (kernel - 1) * dilation
                self.filter_convs.append(nn.Conv2d(hidden, hidden, kernel_size=(1, kernel), dilation=(1, dilation), padding=(0, pad)))
                self.gate_convs.append(nn.Conv2d(hidden, hidden, kernel_size=(1, kernel), dilation=(1, dilation), padding=(0, pad)))
                self.residual_convs.append(nn.Conv2d(hidden, hidden, kernel_size=(1, 1)))
                self.graph_convs.append(nn.Conv2d(hidden * 3, hidden, kernel_size=(1, 1)))
            self.norm = nn.LayerNorm(hidden)
            self.future_mlp = nn.Sequential(
                nn.Linear(int(future_dim), int(args.future_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(args.future_hidden_dim)),
                nn.Linear(int(args.future_hidden_dim), hidden),
                nn.GELU(),
            )
            self.horizon_embed = nn.Embedding(int(args.future_frames), max(8, hidden // 2))
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, 1),
            )

        @staticmethod
        def nconv(x, adj_t):
            return torch.einsum("bcnt,nm->bcmt", x, adj_t)

        def forward(self, hist, future):
            b, n, _t, _c = hist.shape
            adaptive_adj = make_adaptive_adj(self.nodevec1, self.nodevec2)
            x = hist.permute(0, 3, 1, 2)
            x = self.start(x)
            for fconv, gconv, rconv, graph_conv in zip(self.filter_convs, self.gate_convs, self.residual_convs, self.graph_convs):
                residual = x
                filt = torch.tanh(fconv(x))
                gate = torch.sigmoid(gconv(x))
                x = filt * gate
                if x.shape[-1] != residual.shape[-1]:
                    x = x[..., : residual.shape[-1]]
                gx_fixed = self.nconv(x, self.fixed_adj)
                gx_adapt = self.nconv(x, adaptive_adj)
                x = F.gelu(graph_conv(torch.cat([x, gx_fixed, gx_adapt], dim=1)))
                x = rconv(x) + residual[..., -x.shape[-1] :]
            ctx = x[..., -1].permute(0, 2, 1)
            ctx = self.norm(ctx)
            fut = self.future_mlp(future)
            h_ids = torch.arange(self.future_frames, device=hist.device)
            h_emb = self.horizon_embed(h_ids)[None, None].expand(b, n, -1, -1)
            ctx_rep = ctx[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([ctx_rep, fut, h_emb], dim=-1)).squeeze(-1)

    if str(args.model) == "agcrn":
        return AGCRN()
    if str(args.model) == "graph_wavenet":
        return GraphWaveNet()
    raise ValueError(f"Unknown model: {args.model}")


def graph_weighted_loss(pred_z, target_z, target_raw, hist_pv_raw, args):
    import torch
    import torch.nn.functional as F

    last = hist_pv_raw[:, :, -1][:, :, None]
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
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            losses.append(float(graph_weighted_loss(model(hist, fut), y, y_raw, hist_pv, args).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(model, loaders, args, device: str, ckpt_path: Path) -> pd.DataFrame:
    import torch

    name = model_name(args)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model.to(device)
    best = float("inf")
    bad = 0
    logs = []
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
            loss = graph_weighted_loss(model(hist, fut), y, y_raw, hist_pv, args)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = eval_loss(model, loaders["val"], args, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{name}] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[{name}] early stop at epoch={epoch}", flush=True)
                break
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(logs)


def evaluate_graph(model, dataset, loader, stats_by_sn, meta_by_sn, split: str, args, device: str):
    import torch

    name = model_name(args)
    model.eval()
    details = []
    sns = dataset.station_sns
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            pred_z = model(hist, fut).detach().cpu().numpy()
            y = np.asarray(batch["y"], dtype=np.float32)
            hist_pv = np.asarray(batch["hist_pv"], dtype=np.float32)
            issue_times = batch["issue_time"]
            for bi in range(pred_z.shape[0]):
                for si, sn in enumerate(sns):
                    stats = stats_by_sn[sn]
                    pred = pred_z[bi, si] * float(stats["pv_std"]) + float(stats["pv_mean"])
                    last = float(hist_pv[bi, si, -1])
                    meta = meta_by_sn[sn]
                    for h in range(pred.shape[0]):
                        e = float(pred[h] - y[bi, si, h])
                        delta = float(y[bi, si, h] - last)
                        row = {
                            "model": name,
                            "station_sn": sn,
                            "eval_split": split,
                            "issue_time": str(issue_times[bi]),
                            "h": int(h + 1),
                            "horizon": f"h{h + 1}",
                            "pred_pv_norm": float(pred[h]),
                            "true_pv_norm": float(y[bi, si, h]),
                            "capacity_raw": float(meta.get("capacity_raw", np.nan)),
                            "capacity_scale": float(meta.get("capacity_scale", np.nan)),
                            "abs_err_pv_norm": abs(e),
                            "sq_err_pv_norm": e * e,
                            "true_delta_pv_norm": delta,
                            "large_change": abs(delta) >= float(args.large_delta_threshold),
                        }
                        for k, value in enumerate(hist_pv[bi, si], start=1):
                            row[f"hist_pv_norm_{k:02d}"] = float(value)
                        details.append(row)
    detail_df = pd.DataFrame(details)
    metric_rows = []
    if not detail_df.empty:
        grouped = list(detail_df.groupby(["station_sn", "horizon"]))
        grouped += [((sn, "all"), detail_df[detail_df["station_sn"].astype(str) == sn]) for sn in sns]
        for (sn, horizon), g in grouped:
            if g.empty:
                continue
            row = {
                "station_sn": sn,
                "model": name,
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
        raise RuntimeError("PyTorch is required for Stage 4 graph/NWP baseline training.") from exc

    f03.ensure_stage4_dirs()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    f03._configure_torch_runtime(torch, args)

    name = model_name(args)
    targets = pd.read_csv(Path(args.selection_dir) / "target_stations.csv", dtype={"station_sn": str}).reset_index(drop=True)
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    train_time = f03._split_time_df(time_df, "train")

    station_dfs, stats_by_sn, meta_by_sn = {}, {}, {}
    for _, row in targets.iterrows():
        sn = str(row["station_sn"])
        station_dfs[sn], meta_by_sn[sn] = f03.load_station_table(sn, args)
        stats_by_sn[sn] = f03.make_station_stats(station_dfs[sn], train_time)

    adj = build_adjacency(targets, args)
    write_json(
        out_dir / "graph_meta.json",
        {
            "station_order": targets["station_sn"].astype(str).tolist(),
            "distance_adjacency": adj.astype(float).tolist(),
            "graph_k": int(args.graph_k),
            "graph_sigma_km": float(args.graph_sigma_km),
        },
    )

    datasets, loaders = {}, {}
    for split in ["train", "val", "test"]:
        datasets[split] = GraphWeatherDataset(targets, station_dfs, stats_by_sn, f03._split_time_df(time_df, split), args)
        print(f"[{name}] split={split} samples={len(datasets[split])}", flush=True)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.graph_batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
            collate_fn=collate_graph,
        )
    if len(datasets["train"]) < int(args.min_train_samples):
        raise RuntimeError(f"{name} has too few train issue samples: {len(datasets['train'])}")

    hist_dim = 1 + len(f03.FUTURE_NWP_NAMES)
    future_dim = len(f03.FUTURE_NWP_NAMES) + 3
    model = make_model(args, adj, hist_dim, future_dim)
    log_df = train_model(model, loaders, args, device, out_dir / "best.pt")
    log_df.insert(0, "model", name)

    all_metrics, all_details = [], []
    for split in ["val", "test"]:
        metrics, detail = evaluate_graph(model, datasets[split], loaders[split], stats_by_sn, meta_by_sn, split, args, device)
        all_metrics.append(metrics)
        all_details.append(detail)

    meta_df = pd.DataFrame(list(meta_by_sn.values()))
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    agg = f03.aggregate(metrics_df) if not metrics_df.empty else pd.DataFrame()

    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df.to_csv(out_dir / f"{name}_metrics.csv", index=False)
    detail_df.to_csv(out_dir / f"{name}_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / f"{name}_train_log.csv", index=False)
    agg.to_csv(out_dir / f"{name}_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Stage 4 stronger graph baseline with forecast-weather exogenous inputs.",
            "model": name,
            "architecture": str(args.model),
            "inputs": {
                "graph_nodes": "held-out target stations",
                "past_stream": ["PV_norm", *f03.FUTURE_NWP_NAMES],
                "future_stream": [*f03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
                "field_patch": "none",
            },
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"==== Stage 4 {name} aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"detail: {out_dir / f'{name}_eval_detail.csv'}")


if __name__ == "__main__":
    main()
