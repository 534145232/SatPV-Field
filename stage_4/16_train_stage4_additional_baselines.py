#!/usr/bin/env python3
"""Train additional Stage 4 station-level baselines for paper comparison.

This script reuses the corrected Stage 4 station data pipeline from
03_train_per_station_temporal_baseline.py and swaps only the model family.

Station-only models use historical PV plus deterministic future time/solar
features. They do not use future NWP or field patches.

Graph models use historical PV from all held-out target stations with a
distance-based station graph, plus deterministic future time/solar features.
They do not use NWP or field patches.
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
    spec = importlib.util.spec_from_file_location("stage4_temporal_baseline_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_temporal_baseline_base"] = module
    spec.loader.exec_module(module)
    return module


f03 = _load_stage4_temporal_module()

MODEL_CHOICES = ["lstm", "bilstm", "tcn", "transformer", "patchtst", "gcn_lstm", "stgcn"]
GRAPH_MODELS = {"gcn_lstm", "stgcn"}

DEFAULTS = dict(f03.PYCHARM_DEFAULTS)
DEFAULTS.update(
    {
        "model": "lstm",
        "out_dir": "",
        "epochs": 60,
        "patience": 10,
        "hidden_dim": 96,
        "future_hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "batch_size": 128,
        "graph_batch_size": 16,
        "graph_k": 6,
        "graph_sigma_km": 35.0,
        "tcn_kernel_size": 3,
        "transformer_heads": 4,
        "patch_len": 3,
        "patch_stride": 3,
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
    return RUNS_DIR / f"16_{str(args.model)}_baseline_seed{int(args.seed)}"


class LSTMStationOnlyModel:
    def __init__(self, args: argparse.Namespace, future_frames: int, bidirectional: bool):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(args.hidden_dim)
                self.future_frames = int(future_frames)
                self.hist_encoder = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=int(args.num_layers),
                    batch_first=True,
                    dropout=float(args.dropout) if int(args.num_layers) > 1 else 0.0,
                    bidirectional=bool(bidirectional),
                )
                enc_dim = hidden * (2 if bidirectional else 1)
                self.hist_proj = nn.Sequential(nn.Linear(enc_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
                self.future_mlp = nn.Sequential(
                    nn.Linear(3, int(args.future_hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(args.future_hidden_dim)),
                    nn.Linear(int(args.future_hidden_dim), hidden),
                    nn.GELU(),
                )
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                dec_in = hidden + hidden + max(8, hidden // 2)
                self.decoder = nn.Sequential(
                    nn.Linear(dec_in, hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, hist, future):
                import torch

                hist_pv = hist[:, :, :1]
                future_time = future[:, :, -3:]
                enc, _ = self.hist_encoder(hist_pv)
                hist_ctx = self.hist_proj(enc[:, -1])
                fut_ctx = self.future_mlp(future_time)
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                hist_rep = hist_ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([hist_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


class TCNStationOnlyModel:
    def __init__(self, args: argparse.Namespace, future_frames: int):
        import torch.nn as nn

        class TemporalBlock(nn.Module):
            def __init__(self, channels: int, kernel_size: int, dilation: int):
                super().__init__()
                pad = (kernel_size - 1) * dilation
                self.net = nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
                    nn.GELU(),
                )
                self.pad = pad

            def forward(self, x):
                y = self.net(x)
                if y.shape[-1] != x.shape[-1]:
                    y = y[:, :, : x.shape[-1]]
                return x + y

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(args.hidden_dim)
                kernel = int(args.tcn_kernel_size)
                layers = max(1, int(args.num_layers))
                self.future_frames = int(future_frames)
                self.input_proj = nn.Conv1d(1, hidden, kernel_size=1)
                self.blocks = nn.Sequential(*[TemporalBlock(hidden, kernel, 2**i) for i in range(layers)])
                self.norm = nn.LayerNorm(hidden)
                self.future_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.LayerNorm(hidden))
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                self.decoder = nn.Sequential(
                    nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, 1),
                )

            def forward(self, hist, future):
                import torch

                x = hist[:, :, :1].transpose(1, 2)
                ctx = self.blocks(self.input_proj(x))[:, :, -1]
                ctx = self.norm(ctx)
                fut_ctx = self.future_mlp(future[:, :, -3:])
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                ctx_rep = ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([ctx_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


class TransformerStationOnlyModel:
    def __init__(self, args: argparse.Namespace, future_frames: int, patch_mode: bool = False):
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(args.hidden_dim)
                heads = max(1, int(args.transformer_heads))
                if hidden % heads != 0:
                    heads = 1
                self.future_frames = int(future_frames)
                self.patch_mode = bool(patch_mode)
                self.patch_len = int(args.patch_len)
                self.patch_stride = int(args.patch_stride)
                in_dim = self.patch_len if self.patch_mode else 1
                self.input_proj = nn.Linear(in_dim, hidden)
                max_tokens = max(int(args.past_frames), 1 + (int(args.past_frames) - self.patch_len) // max(1, self.patch_stride))
                self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 4,
                    dropout=float(args.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(args.num_layers)))
                self.norm = nn.LayerNorm(hidden)
                self.future_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.LayerNorm(hidden))
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                self.decoder = nn.Sequential(
                    nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, 1),
                )

            def _tokens(self, hist):
                import torch

                x = hist[:, :, 0]
                if not self.patch_mode:
                    return x[:, :, None]
                patches = []
                for start in range(0, x.shape[1] - self.patch_len + 1, max(1, self.patch_stride)):
                    patches.append(x[:, start:start + self.patch_len])
                if not patches:
                    patches = [x[:, -self.patch_len:]]
                return torch.stack(patches, dim=1)

            def forward(self, hist, future):
                import torch

                tok = self.input_proj(self._tokens(hist))
                tok = tok + self.pos_embed[:, : tok.shape[1], :]
                ctx = self.norm(self.encoder(tok).mean(dim=1))
                fut_ctx = self.future_mlp(future[:, :, -3:])
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                ctx_rep = ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([ctx_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


def make_station_model(args: argparse.Namespace):
    model = str(args.model).lower()
    if model == "lstm":
        return LSTMStationOnlyModel(args, int(args.future_frames), bidirectional=False).model
    if model == "bilstm":
        return LSTMStationOnlyModel(args, int(args.future_frames), bidirectional=True).model
    if model == "tcn":
        return TCNStationOnlyModel(args, int(args.future_frames)).model
    if model == "transformer":
        return TransformerStationOnlyModel(args, int(args.future_frames), patch_mode=False).model
    if model == "patchtst":
        return TransformerStationOnlyModel(args, int(args.future_frames), patch_mode=True).model
    raise ValueError(f"{model} is not a station-only model")


def rename_model(metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    out = metrics.copy()
    if "model" in out.columns:
        out["model"] = str(model)
    return out


def run_station_only(args: argparse.Namespace, out_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    f03.ensure_stage4_dirs()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    f03._configure_torch_runtime(torch, args)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])

    all_metrics, all_details, all_logs, all_meta = [], [], [], []
    for idx, row in targets.reset_index(drop=True).iterrows():
        sn = str(row["station_sn"])
        print(f"==== Stage4 {args.model} baseline {idx + 1}/{len(targets)}: {sn} ====", flush=True)
        station_dir = ensure_dir(out_dir / "per_station" / sn)
        station_df, meta = f03.load_station_table(sn, args)
        all_meta.append(meta)
        train_time = f03._split_time_df(time_df, "train")
        stats = f03.make_station_stats(station_df, train_time)
        datasets, loaders = {}, {}
        for split in ["train", "val", "test"]:
            datasets[split] = f03.StationTemporalDataset(sn, station_df, row, f03._split_time_df(time_df, split), stats, args)
            print(f"[{args.model}] {sn} split={split} samples={len(datasets[split])}", flush=True)
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
                collate_fn=f03.collate,
            )
        if len(datasets["train"]) < int(args.min_train_samples):
            print(f"[{args.model}] skip {sn}: train samples={len(datasets['train'])}", flush=True)
            continue
        model = make_station_model(args)
        log_df = f03.train_one(model, loaders, args, device, station_dir / "best.pt")
        log_df.insert(0, "station_sn", sn)
        log_df.insert(1, "model", str(args.model))
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
                "station_only_features": ["historical PV_norm", "future time encoding", "future solar elevation"],
            },
        )
        station_details = []
        for split in ["val", "test"]:
            metrics, detail = f03.evaluate(
                model,
                datasets[split],
                loaders[split],
                stats,
                split,
                args,
                device,
                capacity_raw=float(meta.get("capacity_raw", np.nan)),
                capacity_scale=float(meta.get("capacity_scale", np.nan)),
            )
            metrics = rename_model(metrics, str(args.model))
            detail.insert(0, "model", str(args.model))
            all_metrics.append(metrics)
            station_details.append(detail)
        station_detail = pd.concat(station_details, ignore_index=True) if station_details else pd.DataFrame()
        station_detail.to_csv(station_dir / "eval_detail.csv", index=False)
        all_details.append(station_detail)

    save_outputs(args, out_dir, all_meta, all_metrics, all_details, all_logs)


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
    a = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    k = min(max(1, int(args.graph_k)), max(1, n - 1))
    mask = np.zeros_like(a, dtype=bool)
    for i in range(n):
        nn = np.argsort(dist[i])[1:k + 1]
        mask[i, nn] = True
    mask = mask | mask.T | np.eye(n, dtype=bool)
    a = np.where(mask, a, 0.0).astype(np.float32)
    a = np.maximum(a, a.T)
    deg = np.sum(a, axis=1)
    deg = np.where(deg > 1e-6, deg, 1.0)
    d_inv = 1.0 / np.sqrt(deg)
    return (d_inv[:, None] * a * d_inv[None, :]).astype(np.float32)


class GraphTemporalDataset:
    def __init__(self, targets, station_dfs, stats_by_sn, time_df, args):
        self.targets = targets.reset_index(drop=True)
        self.station_sns = self.targets["station_sn"].astype(str).tolist()
        self.station_dfs = station_dfs
        self.stats_by_sn = stats_by_sn
        self.time_df = time_df.reset_index(drop=True)
        self.args = args
        self.samples = []
        self._build_samples()

    def _has_all(self, sn: str, times: list[pd.Timestamp]) -> bool:
        df = self.station_dfs[sn]
        return all(pd.Timestamp(t) in df.index for t in times)

    def _build_samples(self):
        for i, row in self.time_df.iterrows():
            past = f03._parse_time_list(row["past_times"])
            future = f03._parse_time_list(row["future_times"])
            ok = all(self._has_all(sn, past + future) for sn in self.station_sns)
            if ok:
                self.samples.append(int(i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        past_times = f03._parse_time_list(row["past_times"])
        future_times = f03._parse_time_list(row["future_times"])
        hist_z, hist_raw, y_z, y_raw, future_time = [], [], [], [], []
        for sidx, sn in enumerate(self.station_sns):
            df = self.station_dfs[sn]
            stats = self.stats_by_sn[sn]
            pv_hist = df.loc[past_times, "pv_norm"].to_numpy(np.float32)
            pv_future = df.loc[future_times, "pv_norm"].to_numpy(np.float32)
            hist_z.append(((pv_hist - float(stats["pv_mean"])) / float(stats["pv_std"]))[:, None])
            hist_raw.append(pv_hist)
            y_z.append((pv_future - float(stats["pv_mean"])) / float(stats["pv_std"]))
            y_raw.append(pv_future)
            st = self.targets.iloc[sidx]
            future_time.append(
                np.stack([f03._time_solar_point(ts, float(st["lon"]), float(st["lat"])) for ts in future_times], axis=0)
            )
        return {
            "hist": np.stack(hist_z, axis=0).astype(np.float32),
            "hist_pv": np.stack(hist_raw, axis=0).astype(np.float32),
            "future": np.stack(future_time, axis=0).astype(np.float32),
            "y_z": np.stack(y_z, axis=0).astype(np.float32),
            "y": np.stack(y_raw, axis=0).astype(np.float32),
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def collate_graph(batch):
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


class DenseGraphConv:
    def __init__(self, in_dim: int, out_dim: int):
        import torch.nn as nn

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)

            def forward(self, x, adj):
                import torch

                x = torch.einsum("ij,bjf->bif", adj, x)
                return self.linear(x)

        self.layer = Layer()


def make_graph_model(args: argparse.Namespace, n_nodes: int, adj: np.ndarray):
    import torch
    import torch.nn as nn

    class GraphConv(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x, adj_t):
            return self.linear(torch.einsum("ij,bjf->bif", adj_t, x))

    class GCNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(args.hidden_dim)
            self.future_frames = int(args.future_frames)
            self.register_buffer("adj", torch.as_tensor(adj, dtype=torch.float32))
            self.gcn = GraphConv(1, hidden)
            self.lstm = nn.LSTM(
                hidden,
                hidden,
                num_layers=int(args.num_layers),
                batch_first=True,
                dropout=float(args.dropout) if int(args.num_layers) > 1 else 0.0,
            )
            self.future_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.LayerNorm(hidden))
            self.horizon_embed = nn.Embedding(int(args.future_frames), max(8, hidden // 2))
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, hist, future):
            b, n, t, _ = hist.shape
            xs = []
            for ti in range(t):
                xs.append(torch.relu(self.gcn(hist[:, :, ti, :], self.adj)))
            x = torch.stack(xs, dim=2).reshape(b * n, t, -1)
            enc, _ = self.lstm(x)
            ctx = enc[:, -1].reshape(b, n, -1)
            fut = self.future_mlp(future)
            h_ids = torch.arange(self.future_frames, device=hist.device)
            h_emb = self.horizon_embed(h_ids)[None, None].expand(b, n, -1, -1)
            ctx_rep = ctx[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([ctx_rep, fut, h_emb], dim=-1)).squeeze(-1)

    class STGCN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(args.hidden_dim)
            self.future_frames = int(args.future_frames)
            self.register_buffer("adj", torch.as_tensor(adj, dtype=torch.float32))
            kernel = int(args.tcn_kernel_size)
            pad = kernel // 2
            self.temporal = nn.Sequential(
                nn.Conv1d(1, hidden, kernel, padding=pad),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel, padding=pad),
                nn.GELU(),
            )
            self.gcn = GraphConv(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.future_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.LayerNorm(hidden))
            self.horizon_embed = nn.Embedding(int(args.future_frames), max(8, hidden // 2))
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, hist, future):
            b, n, t, _ = hist.shape
            x = hist.reshape(b * n, t, 1).transpose(1, 2)
            ctx = self.temporal(x)[:, :, -1].reshape(b, n, -1)
            ctx = self.norm(torch.relu(self.gcn(ctx, self.adj)))
            fut = self.future_mlp(future)
            h_ids = torch.arange(self.future_frames, device=hist.device)
            h_emb = self.horizon_embed(h_ids)[None, None].expand(b, n, -1, -1)
            ctx_rep = ctx[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([ctx_rep, fut, h_emb], dim=-1)).squeeze(-1)

    return GCNLSTM() if str(args.model) == "gcn_lstm" else STGCN()


def graph_weighted_loss(pred_z, target_z, target_raw, hist_pv_raw, args):
    import torch
    import torch.nn.functional as F

    last = hist_pv_raw[:, :, -1][:, :, None]
    true_delta = torch.abs(target_raw - last)
    weights = 1.0 + float(args.big_change_loss_weight) * (true_delta >= float(args.large_delta_threshold)).float()
    return (F.smooth_l1_loss(pred_z, target_z, reduction="none") * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def graph_eval_loss(model, loader, args, device: str) -> float:
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


def train_graph(model, loaders, args, device: str, ckpt_path: Path) -> pd.DataFrame:
    import torch

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
        val_loss = graph_eval_loss(model, loaders["val"], args, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{args.model}] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[{args.model}] early stop at epoch={epoch}", flush=True)
                break
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(logs)


def evaluate_graph(model, dataset: GraphTemporalDataset, loader, stats_by_sn, meta_by_sn, split: str, args, device: str):
    import torch

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
                            "model": str(args.model),
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
        for (sn, horizon), g in list(detail_df.groupby(["station_sn", "horizon"])) + [((sn, "all"), detail_df[detail_df["station_sn"].astype(str) == sn]) for sn in sns]:
            if g.empty:
                continue
            row = {
                "station_sn": sn,
                "model": str(args.model),
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


def run_graph(args: argparse.Namespace, out_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    f03.ensure_stage4_dirs()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    f03._configure_torch_runtime(torch, args)

    targets = pd.read_csv(Path(args.selection_dir) / "target_stations.csv", dtype={"station_sn": str}).reset_index(drop=True)
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    station_dfs, stats_by_sn, meta_by_sn = {}, {}, {}
    train_time = f03._split_time_df(time_df, "train")
    for _, row in targets.iterrows():
        sn = str(row["station_sn"])
        station_dfs[sn], meta_by_sn[sn] = f03.load_station_table(sn, args)
        stats_by_sn[sn] = f03.make_station_stats(station_dfs[sn], train_time)
    adj = build_adjacency(targets, args)
    write_json(out_dir / "graph_meta.json", {"station_order": targets["station_sn"].astype(str).tolist(), "adjacency": adj.astype(float).tolist()})

    datasets, loaders = {}, {}
    for split in ["train", "val", "test"]:
        datasets[split] = GraphTemporalDataset(targets, station_dfs, stats_by_sn, f03._split_time_df(time_df, split), args)
        print(f"[{args.model}] graph split={split} samples={len(datasets[split])}", flush=True)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.graph_batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
            collate_fn=collate_graph,
        )
    if len(datasets["train"]) < int(args.min_train_samples):
        raise RuntimeError(f"Graph baseline has too few train issue samples: {len(datasets['train'])}")

    model = make_graph_model(args, len(targets), adj)
    log_df = train_graph(model, loaders, args, device, out_dir / "best.pt")
    log_df.insert(0, "model", str(args.model))
    all_metrics, all_details = [], []
    for split in ["val", "test"]:
        metrics, detail = evaluate_graph(model, datasets[split], loaders[split], stats_by_sn, meta_by_sn, split, args, device)
        all_metrics.append(metrics)
        all_details.append(detail)
    meta_df = pd.DataFrame(list(meta_by_sn.values()))
    save_outputs(args, out_dir, meta_df.to_dict("records"), all_metrics, all_details, [log_df])


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return f03.aggregate(metrics) if not metrics.empty else pd.DataFrame()


def save_outputs(args, out_dir, all_meta, all_metrics, all_details, all_logs) -> None:
    meta_df = pd.DataFrame(all_meta)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    model = str(args.model)
    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df.to_csv(out_dir / f"{model}_metrics.csv", index=False)
    detail_df.to_csv(out_dir / f"{model}_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / f"{model}_train_log.csv", index=False)
    agg = aggregate(metrics_df)
    agg.to_csv(out_dir / f"{model}_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Additional Stage 4 paper baseline.",
            "model": model,
            "spatial_type": "Implicit multi-site" if model in GRAPH_MODELS else "Station-only",
            "inputs": (
                "historical PV from all held-out target stations + future time/solar"
                if model in GRAPH_MODELS
                else "own-station historical PV + future time/solar"
            ),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"==== Stage 4 {model} baseline aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"detail: {out_dir / f'{model}_eval_detail.csv'}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(default_out_dir(args))
    if str(args.model) in GRAPH_MODELS:
        run_graph(args, out_dir)
    else:
        run_station_only(args, out_dir)


if __name__ == "__main__":
    main()
