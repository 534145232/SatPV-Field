#!/usr/bin/env python3
"""Train a multi-site AGCRN extension with predicted FG-MSF field patches.

Each held-out target-station node receives:
  - historical PV_norm and historical weather/NWP columns,
  - future NWP + deterministic time/solar features,
  - station-centered predicted future PV-field patches from Stage 3 / FG-MSF.

This model is intended as an extension/upper-bound variant to test whether the
future PV-field guidance is complementary to implicit station-graph modeling.
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


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m05 = _load_local_module("stage4_05l_field_patch_for_agcrn", SCRIPT_DIR / "05l_train_per_station_temporal_with_stage3_field_patch.py")
m23 = _load_local_module("stage4_23_graph_nwp_for_agcrn_field", SCRIPT_DIR / "23_train_stage4_graph_nwp_baselines.py")

from stage4_metrics import regression_metric_extras  # noqa: E402
from stage4_paths import RUNS_DIR  # noqa: E402


MODEL_NAME = "agcrn_fgmsf_patch"

DEFAULTS = dict(m05.PYCHARM_DEFAULTS)
DEFAULTS.update(
    {
        "out_dir": "",
        "epochs": int(m05.PYCHARM_DEFAULTS["epochs"]),
        "patience": int(m05.PYCHARM_DEFAULTS["patience"]),
        "hidden_dim": 96,
        "future_hidden_dim": 64,
        "field_hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "graph_batch_size": 24,
        "graph_k": 6,
        "graph_sigma_km": 18.0,
        "node_emb_dim": 16,
        "sample_count": 0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=bool(value))
            parser.add_argument(f"--disable_{key}", action="store_false", dest=key)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        elif key == "field_checkpoint":
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir"}:
            parser.add_argument(f"--{key}", type=Path, default=Path(value) if str(value) else Path(""))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    return parser.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    out_arg = str(args.out_dir)
    if out_arg and out_arg not in {".", "./"}:
        return Path(args.out_dir)
    return RUNS_DIR / f"27_{MODEL_NAME}_seed{int(args.seed)}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class GraphFieldPatchDataset:
    def __init__(self, targets, station_dfs, stats_by_sn, time_df, patch_provider, args):
        self.targets = targets.reset_index(drop=True)
        self.station_sns = self.targets["station_sn"].astype(str).tolist()
        self.station_dfs = station_dfs
        self.stats_by_sn = stats_by_sn
        self.time_df = time_df.reset_index(drop=True)
        self.patch_provider = patch_provider
        self.args = args
        self.samples: list[int] = []
        self._build_samples()

    def _has_all(self, sn: str, times: list[pd.Timestamp]) -> bool:
        return all(pd.Timestamp(t) in self.station_dfs[sn].index for t in times)

    def _build_samples(self) -> None:
        for i, row in self.time_df.iterrows():
            past = m05.b03._parse_time_list(row["past_times"])
            future = m05.b03._parse_time_list(row["future_times"])
            if all(self._has_all(sn, past + future) for sn in self.station_sns):
                self.samples.append(int(i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        i = self.samples[idx]
        row = self.time_df.iloc[i]
        issue = pd.Timestamp(row["issue_time"])
        past_times = m05.b03._parse_time_list(row["past_times"])
        future_times = m05.b03._parse_time_list(row["future_times"])
        hist_z, hist_raw, fut_feat, field_patch, y_z, y_raw = [], [], [], [], [], []
        for sidx, sn in enumerate(self.station_sns):
            df = self.station_dfs[sn]
            stats = self.stats_by_sn[sn]
            hist_raw_arr = df.loc[past_times, ["pv_norm", *m05.b03.FUTURE_NWP_NAMES]].to_numpy(np.float32)
            fut_raw_arr = df.loc[future_times, m05.b03.FUTURE_NWP_NAMES].to_numpy(np.float32)
            pv_future = df.loc[future_times, "pv_norm"].to_numpy(np.float32)
            hist_z.append((hist_raw_arr - stats["hist_mean"]) / stats["hist_std"])
            fut_z = (fut_raw_arr - stats["future_mean"]) / stats["future_std"]
            st = self.targets.iloc[sidx]
            time_solar = np.stack(
                [m05.b03._time_solar_point(ts, float(st["lon"]), float(st["lat"])) for ts in future_times],
                axis=0,
            )
            fut_feat.append(np.concatenate([fut_z, time_solar], axis=1))
            field_z, field_mask = self.patch_provider.get(sn, issue)
            field_patch.append(np.stack([field_z, field_mask], axis=1))
            hist_raw.append(hist_raw_arr[:, 0].astype(np.float32))
            y_z.append((pv_future - float(stats["pv_mean"])) / float(stats["pv_std"]))
            y_raw.append(pv_future)
        return {
            "hist": np.stack(hist_z, axis=0).astype(np.float32),
            "future": np.stack(fut_feat, axis=0).astype(np.float32),
            "field_patch": np.stack(field_patch, axis=0).astype(np.float32),
            "hist_pv": np.stack(hist_raw, axis=0).astype(np.float32),
            "y_z": np.stack(y_z, axis=0).astype(np.float32),
            "y": np.stack(y_raw, axis=0).astype(np.float32),
            "issue_time": str(issue),
        }


def collate_graph_field(batch: list[dict]) -> dict:
    return {
        "hist": np.stack([b["hist"] for b in batch]).astype(np.float32),
        "future": np.stack([b["future"] for b in batch]).astype(np.float32),
        "field_patch": np.stack([b["field_patch"] for b in batch]).astype(np.float32),
        "hist_pv": np.stack([b["hist_pv"] for b in batch]).astype(np.float32),
        "y_z": np.stack([b["y_z"] for b in batch]).astype(np.float32),
        "y": np.stack([b["y"] for b in batch]).astype(np.float32),
        "issue_time": [b["issue_time"] for b in batch],
    }


def _tensor(batch: dict, key: str, device: str, torch):
    value = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def make_agcrn_field_model(args: argparse.Namespace, adj: np.ndarray, hist_dim: int, future_dim: int, field_dim: int):
    import torch
    import torch.nn as nn

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
            self.gate = AdaptiveGraphConv(input_dim + hidden_dim, 2 * hidden_dim)
            self.update = AdaptiveGraphConv(input_dim + hidden_dim, hidden_dim)

        def forward(self, x, state, adaptive_adj):
            gates = torch.sigmoid(self.gate(torch.cat([x, state], dim=-1), adaptive_adj))
            z, r = torch.chunk(gates, 2, dim=-1)
            candidate = torch.tanh(self.update(torch.cat([x, r * state], dim=-1), adaptive_adj))
            return z * state + (1.0 - z) * candidate

    class AGCRNField(nn.Module):
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
            self.field_mlp = nn.Sequential(
                nn.Linear(int(field_dim), int(args.field_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(args.field_hidden_dim)),
                nn.Linear(int(args.field_hidden_dim), hidden),
                nn.GELU(),
            )
            self.horizon_embed = nn.Embedding(int(args.future_frames), max(8, hidden // 2))
            self.decoder = nn.Sequential(
                nn.Linear(hidden + hidden + hidden + max(8, hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, hist, future, field_patch):
            b, n, t, _ = hist.shape
            adaptive_adj = m23.make_adaptive_adj(self.nodevec1, self.nodevec2)
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
            field = self.field_mlp(field_patch.flatten(start_dim=3))
            h_ids = torch.arange(self.future_frames, device=hist.device)
            h_emb = self.horizon_embed(h_ids)[None, None].expand(b, n, -1, -1)
            ctx_rep = ctx[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([ctx_rep, fut, field, h_emb], dim=-1)).squeeze(-1)

    return AGCRNField()


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
            patch = _tensor(batch, "field_patch", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            losses.append(float(graph_weighted_loss(model(hist, fut, patch), y, y_raw, hist_pv, args).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(model, loaders, args, device: str, ckpt_path: Path) -> pd.DataFrame:
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
            patch = _tensor(batch, "field_patch", device, torch)
            y = _tensor(batch, "y_z", device, torch)
            y_raw = _tensor(batch, "y", device, torch)
            hist_pv = _tensor(batch, "hist_pv", device, torch)
            opt.zero_grad(set_to_none=True)
            loss = graph_weighted_loss(model(hist, fut, patch), y, y_raw, hist_pv, args)
            loss.backward()
            if float(args.grad_clip) > 0:
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


def evaluate_graph_field(model, dataset, loader, stats_by_sn, meta_by_sn, patch_provider, split: str, args, device: str):
    import torch

    model.eval()
    details = []
    sns = dataset.station_sns
    with torch.no_grad():
        for batch in loader:
            hist = _tensor(batch, "hist", device, torch)
            fut = _tensor(batch, "future", device, torch)
            patch = _tensor(batch, "field_patch", device, torch)
            pred_z = model(hist, fut, patch).detach().cpu().numpy()
            y = np.asarray(batch["y"], dtype=np.float32)
            hist_pv = np.asarray(batch["hist_pv"], dtype=np.float32)
            issue_times = batch["issue_time"]
            for bi in range(pred_z.shape[0]):
                for si, sn in enumerate(sns):
                    stats = stats_by_sn[sn]
                    pred = pred_z[bi, si] * float(stats["pv_std"]) + float(stats["pv_mean"])
                    last = float(hist_pv[bi, si, -1])
                    meta = meta_by_sn[sn]
                    field_center, field_mean, field_support = patch_provider.raw_center_series(sn, issue_times[bi])
                    for h in range(pred.shape[0]):
                        e = float(pred[h] - y[bi, si, h])
                        delta = float(y[bi, si, h] - last)
                        row = {
                            "model": MODEL_NAME,
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
                            "field_patch_center_pv_norm": float(field_center[h]),
                            "field_patch_mean_pv_norm": float(field_mean[h]),
                            "field_patch_support_mean": float(field_support[h]),
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
                "model": MODEL_NAME,
                "eval_split": split,
                "horizon": horizon,
                "n_samples": int(len(g)),
                "mae_pv_norm": float(g["abs_err_pv_norm"].mean()),
                "rmse_pv_norm": float(math.sqrt(g["sq_err_pv_norm"].mean())),
                "high_change_mae_pv_norm": float(g.loc[g["large_change"], "abs_err_pv_norm"].mean()) if bool(g["large_change"].any()) else float("nan"),
                "high_change_n": int(g["large_change"].sum()),
                "field_patch_support_mean": float(g["field_patch_support_mean"].mean()),
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
        raise RuntimeError("PyTorch is required for AGCRN + FG-MSF patch training.") from exc

    m05.ensure_stage4_dirs()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    m05._configure_torch_runtime(torch, args)

    target_path = Path(args.selection_dir) / "target_stations.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"target_stations.csv not found: {target_path}")
    targets = pd.read_csv(target_path, dtype={"station_sn": str}).reset_index(drop=True)
    time_df = pd.read_csv(args.time_index_csv)
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(m05._timestamp_key)
    m05.resolve_stage3_swr_anchor_cache_arg(args)
    time_df = m05.assert_time_index_matches_field_run(time_df, args)
    time_df = m05.filter_time_index_to_swr_anchor_cache(time_df, args)
    time_df.to_csv(out_dir / "station_issue_time_index_aligned_to_swr_anchor04l_field_run.csv", index=False)

    patch_provider = m05.PredictedFieldPatchProvider(targets, time_df, args, device)
    write_json(out_dir / "field_patch_stats.json", patch_provider.stats())

    train_time = m05.b03._split_time_df(time_df, "train")
    station_dfs, stats_by_sn, meta_by_sn = {}, {}, {}
    for _, row in targets.iterrows():
        sn = str(row["station_sn"])
        station_dfs[sn], meta_by_sn[sn] = m05.b03.load_station_table(sn, args)
        stats_by_sn[sn] = m05.b03.make_station_stats(station_dfs[sn], train_time)

    adj = m23.build_adjacency(targets, args)
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
        datasets[split] = GraphFieldPatchDataset(
            targets,
            station_dfs,
            stats_by_sn,
            m05.b03._split_time_df(time_df, split),
            patch_provider,
            args,
        )
        print(f"[{MODEL_NAME}] split={split} samples={len(datasets[split])}", flush=True)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(args.graph_batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
            collate_fn=collate_graph_field,
        )
    if len(datasets["train"]) < int(args.min_train_samples):
        raise RuntimeError(f"{MODEL_NAME} has too few train issue samples: {len(datasets['train'])}")

    hist_dim = 1 + len(m05.b03.FUTURE_NWP_NAMES)
    future_dim = len(m05.b03.FUTURE_NWP_NAMES) + 3
    field_dim = 2 * int(args.field_patch_size) * int(args.field_patch_size)
    model = make_agcrn_field_model(args, adj, hist_dim, future_dim, field_dim)
    log_df = train_model(model, loaders, args, device, out_dir / "best.pt")
    log_df.insert(0, "model", MODEL_NAME)

    all_metrics, all_details = [], []
    for split in ["val", "test"]:
        metrics, detail = evaluate_graph_field(
            model,
            datasets[split],
            loaders[split],
            stats_by_sn,
            meta_by_sn,
            patch_provider,
            split,
            args,
            device,
        )
        all_metrics.append(metrics)
        all_details.append(detail)

    meta_df = pd.DataFrame(list(meta_by_sn.values()))
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    agg = m05.b03.aggregate(metrics_df) if not metrics_df.empty else pd.DataFrame()

    meta_df.to_csv(out_dir / "station_load_meta.csv", index=False)
    metrics_df.to_csv(out_dir / f"{MODEL_NAME}_metrics.csv", index=False)
    detail_df.to_csv(out_dir / f"{MODEL_NAME}_eval_detail.csv", index=False)
    log_df.to_csv(out_dir / f"{MODEL_NAME}_train_log.csv", index=False)
    agg.to_csv(out_dir / f"{MODEL_NAME}_aggregate.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "purpose": "Stage 4 multi-site AGCRN extension with predicted FG-MSF field patches.",
            "model": MODEL_NAME,
            "inputs": {
                "graph_nodes": "held-out target stations",
                "past_stream": ["PV_norm", *m05.b03.FUTURE_NWP_NAMES],
                "future_stream": [*m05.b03.FUTURE_NWP_NAMES, "tod_sin", "tod_cos", "solar_elevation"],
                "field_patch": {
                    "source": "Stage 3 predicted future PV field",
                    "patch_size": int(args.field_patch_size),
                    "channels": ["field_z", "support_mask"],
                },
            },
            "field_patch_stats": patch_provider.stats(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"==== Stage 4 {MODEL_NAME} aggregate ====")
    print(agg[agg["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False) if not agg.empty else "EMPTY")
    print(f"detail: {out_dir / f'{MODEL_NAME}_eval_detail.csv'}")


if __name__ == "__main__":
    main()
