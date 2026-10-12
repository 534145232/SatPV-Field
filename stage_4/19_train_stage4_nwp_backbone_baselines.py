#!/usr/bin/env python3
"""Train Stage 4 backbone baselines with station-level weather forecast inputs.

These baselines use the same station-level input set as the temporal baseline:

Past stream:
    historical PV_norm + historical ERA5/weather columns

Future stream:
    future weather forecast columns + deterministic time/solar features

No SWR/PV field patch is used. The goal is to compare common temporal
backbones under the same station-level exogenous inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_paths import RUNS_DIR  # noqa: E402


def _load_stage4_temporal_module():
    path = SCRIPT_DIR / "03_train_per_station_temporal_baseline.py"
    spec = importlib.util.spec_from_file_location("stage4_temporal_baseline_for_nwp_backbones", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_temporal_baseline_for_nwp_backbones"] = module
    spec.loader.exec_module(module)
    return module


f03 = _load_stage4_temporal_module()

MODEL_CHOICES = ["lstm", "bilstm", "tcn", "transformer", "patchtst"]

DEFAULTS = dict(f03.PYCHARM_DEFAULTS)
DEFAULTS.update(
    {
        "model": "tcn",
        "out_dir": "",
        "epochs": 60,
        "patience": 10,
        "hidden_dim": 96,
        "future_hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "batch_size": 128,
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
    return RUNS_DIR / f"19_{str(args.model)}_nwp_baseline_seed{int(args.seed)}"


class LSTMWeatherModel:
    def __init__(self, hist_dim: int, future_dim: int, future_frames: int, args: argparse.Namespace, bidirectional: bool):
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
                    bidirectional=bool(bidirectional),
                )
                enc_dim = hidden * (2 if bidirectional else 1)
                self.hist_proj = nn.Sequential(nn.Linear(enc_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
                self.future_mlp = nn.Sequential(
                    nn.Linear(int(future_dim), int(args.future_hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(args.future_hidden_dim)),
                    nn.Linear(int(args.future_hidden_dim), hidden),
                    nn.GELU(),
                )
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                self.decoder = nn.Sequential(
                    nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, hist, future):
                import torch

                enc, _ = self.hist_encoder(hist)
                hist_ctx = self.hist_proj(enc[:, -1])
                fut_ctx = self.future_mlp(future)
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                hist_rep = hist_ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([hist_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


class TCNWeatherModel:
    def __init__(self, hist_dim: int, future_dim: int, future_frames: int, args: argparse.Namespace):
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
                self.input_proj = nn.Conv1d(int(hist_dim), hidden, kernel_size=1)
                self.blocks = nn.Sequential(*[TemporalBlock(hidden, kernel, 2**i) for i in range(layers)])
                self.norm = nn.LayerNorm(hidden)
                self.future_mlp = nn.Sequential(
                    nn.Linear(int(future_dim), hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                )
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                self.decoder = nn.Sequential(
                    nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, 1),
                )

            def forward(self, hist, future):
                import torch

                x = hist.transpose(1, 2)
                ctx = self.blocks(self.input_proj(x))[:, :, -1]
                ctx = self.norm(ctx)
                fut_ctx = self.future_mlp(future)
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                ctx_rep = ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([ctx_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


class TransformerWeatherModel:
    def __init__(self, hist_dim: int, future_dim: int, future_frames: int, args: argparse.Namespace, patch_mode: bool):
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
                self.patch_stride = max(1, int(args.patch_stride))
                in_dim = int(hist_dim) * self.patch_len if self.patch_mode else int(hist_dim)
                self.input_proj = nn.Linear(in_dim, hidden)
                if self.patch_mode:
                    max_tokens = max(1, 1 + (int(args.past_frames) - self.patch_len) // self.patch_stride)
                else:
                    max_tokens = int(args.past_frames)
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
                self.future_mlp = nn.Sequential(
                    nn.Linear(int(future_dim), hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                )
                self.horizon_embed = nn.Embedding(int(future_frames), max(8, hidden // 2))
                self.decoder = nn.Sequential(
                    nn.Linear(hidden + hidden + max(8, hidden // 2), hidden),
                    nn.GELU(),
                    nn.Dropout(float(args.dropout)),
                    nn.Linear(hidden, 1),
                )

            def _tokens(self, hist):
                import torch

                if not self.patch_mode:
                    return hist
                patches = []
                for start in range(0, hist.shape[1] - self.patch_len + 1, self.patch_stride):
                    patches.append(hist[:, start:start + self.patch_len, :].reshape(hist.shape[0], -1))
                if not patches:
                    patches = [hist[:, -self.patch_len:, :].reshape(hist.shape[0], -1)]
                return torch.stack(patches, dim=1)

            def forward(self, hist, future):
                import torch

                tok = self.input_proj(self._tokens(hist))
                tok = tok + self.pos_embed[:, : tok.shape[1], :]
                ctx = self.norm(self.encoder(tok).mean(dim=1))
                fut_ctx = self.future_mlp(future)
                h_ids = torch.arange(self.future_frames, device=future.device)
                h_emb = self.horizon_embed(h_ids)[None].expand(future.shape[0], -1, -1)
                ctx_rep = ctx[:, None].expand(-1, self.future_frames, -1)
                return self.decoder(torch.cat([ctx_rep, fut_ctx, h_emb], dim=-1)).squeeze(-1)

        self.model = Model()


def make_model(hist_dim: int, future_dim: int, args: argparse.Namespace):
    model = str(args.model).lower()
    if model == "lstm":
        return LSTMWeatherModel(hist_dim, future_dim, int(args.future_frames), args, bidirectional=False).model
    if model == "bilstm":
        return LSTMWeatherModel(hist_dim, future_dim, int(args.future_frames), args, bidirectional=True).model
    if model == "tcn":
        return TCNWeatherModel(hist_dim, future_dim, int(args.future_frames), args).model
    if model == "transformer":
        return TransformerWeatherModel(hist_dim, future_dim, int(args.future_frames), args, patch_mode=False).model
    if model == "patchtst":
        return TransformerWeatherModel(hist_dim, future_dim, int(args.future_frames), args, patch_mode=True).model
    raise ValueError(f"Unknown model: {model}")


def rename_model(metrics: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = metrics.copy()
    if "model" in out.columns:
        out["model"] = model_name
    return out


def save_outputs(args, out_dir: Path, all_meta, all_metrics, all_details, all_logs) -> None:
    model = str(args.model)
    model_name = f"{model}_nwp"
    meta_df = pd.DataFrame(all_meta)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df.to_csv(out_dir / f"{model_name}_metrics.csv", index=False)
    detail_df.to_csv(out_dir / f"{model_name}_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / f"{model_name}_train_log.csv", index=False)
    agg = f03.aggregate(metrics_df)
    agg.to_csv(out_dir / f"{model_name}_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Stage 4 NWP/forecast-aware temporal backbone baseline.",
            "model": model_name,
            "inputs": {
                "past_stream": ["PV_norm", *f03.FUTURE_NWP_NAMES],
                "future_stream": [*f03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
                "field_patch": "none",
            },
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"==== Stage 4 {model_name} aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"detail: {out_dir / f'{model_name}_eval_detail.csv'}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(default_out_dir(args))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Stage 4 NWP backbone baseline training.") from exc

    f03.ensure_stage4_dirs()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    f03._configure_torch_runtime(torch, args)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"target_stations.csv not found: {target_path}")
    targets = pd.read_csv(target_path, dtype={"station_sn": str})
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])

    hist_dim = 1 + len(f03.FUTURE_NWP_NAMES)
    future_dim = len(f03.FUTURE_NWP_NAMES) + 3
    model_name = f"{str(args.model)}_nwp"
    all_metrics, all_details, all_logs, all_meta = [], [], [], []

    for idx, row in targets.reset_index(drop=True).iterrows():
        sn = str(row["station_sn"])
        print(f"==== Stage4 {model_name} baseline {idx + 1}/{len(targets)}: {sn} ====", flush=True)
        station_dir = ensure_dir(out_dir / "per_station" / sn)
        station_df, meta = f03.load_station_table(sn, args)
        all_meta.append(meta)
        train_time = f03._split_time_df(time_df, "train")
        stats = f03.make_station_stats(station_df, train_time)
        datasets, loaders = {}, {}
        for split in ["train", "val", "test"]:
            datasets[split] = f03.StationTemporalDataset(sn, station_df, row, f03._split_time_df(time_df, split), stats, args)
            print(f"[{model_name}] {sn} split={split} samples={len(datasets[split])}", flush=True)
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=int(args.batch_size),
                shuffle=(split == "train"),
                num_workers=int(args.num_workers),
                collate_fn=f03.collate,
            )
        if len(datasets["train"]) < int(args.min_train_samples):
            print(f"[{model_name}] skip {sn}: train samples={len(datasets['train'])}", flush=True)
            continue

        model = make_model(hist_dim, future_dim, args)
        log_df = f03.train_one(model, loaders, args, device, station_dir / "best.pt")
        log_df.insert(0, "station_sn", sn)
        log_df.insert(1, "model", model_name)
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
            metrics = rename_model(metrics, model_name)
            detail.insert(0, "model", model_name)
            all_metrics.append(metrics)
            station_details.append(detail)
        station_detail = pd.concat(station_details, ignore_index=True) if station_details else pd.DataFrame()
        station_detail.to_csv(station_dir / "eval_detail.csv", index=False)
        all_details.append(station_detail)

    save_outputs(args, out_dir, all_meta, all_metrics, all_details, all_logs)


if __name__ == "__main__":
    main()
