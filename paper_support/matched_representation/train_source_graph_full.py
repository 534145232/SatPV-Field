#!/usr/bin/env python3
"""Train the full 446-source fixed-graph forecaster and cache its guidance.

The cache contains both mandatory graph representations on exactly the same
canonical issue index: the latest historical source-node state repeated over
the eight horizons, and the independently predicted future source-node state.
Future source PV is read only as a masked training/evaluation label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matched_representation_core as core


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parents[1])
    parser.add_argument("--old-data-root", type=Path, default=None)
    parser.add_argument("--stage3-data-dir", type=Path, default=None)
    parser.add_argument("--stage3-model-dir", type=Path, default=None)
    parser.add_argument("--pv-root", type=Path, default=None)
    parser.add_argument("--train-pv-dir", type=Path, default=None)
    parser.add_argument("--val-pv-dir", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=SCRIPT_DIR / "outputs" / "audit")
    parser.add_argument(
        "--phase2-report", type=Path,
        default=SCRIPT_DIR / "outputs" / "phase2" / "phase2_sanity_report.json",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "full" / "source_graph")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--node-emb-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--disable-cudnn-rnn", action="store_true", default=True)
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _tensor(batch: dict, name: str, device: str, torch):
    value = torch.as_tensor(batch[name], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _source_loss_parts(model, batch, device: str, torch):
    import torch.nn.functional as F

    prediction = model(
        _tensor(batch, "source_hist", device, torch),
        _tensor(batch, "future_met", device, torch),
        _tensor(batch, "future_irr", device, torch),
    )
    target = _tensor(batch, "source_target", device, torch)
    valid = _tensor(batch, "source_valid", device, torch)
    numerator = (F.smooth_l1_loss(prediction, target, reduction="none") * valid).sum()
    denominator = torch.clamp(valid.sum(), min=1.0)
    return prediction, numerator, denominator


def evaluate_loss(model, loader, device: str) -> float:
    import torch

    model.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for batch in loader:
            _, loss_sum, weight_sum = _source_loss_parts(model, batch, device, torch)
            numerator += float(loss_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
    return numerator / max(denominator, 1.0)


def train_model(model, loaders, args, device: str, checkpoint: Path) -> pd.DataFrame:
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    best = float("inf")
    bad = 0
    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        numerator = 0.0
        denominator = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            _, loss_sum, weight_sum = _source_loss_parts(model, batch, device, torch)
            loss = loss_sum / weight_sum
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            numerator += float(loss_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
        train_loss = numerator / max(denominator, 1.0)
        val_loss = evaluate_loss(model, loaders["val"], device)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[source-full] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best - 1.0e-6:
            best = val_loss
            bad = 0
            torch.save({"model_state": model.state_dict(), "best_val_loss": best}, checkpoint)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[source-full] early stop at epoch={epoch}", flush=True)
                break
    if not checkpoint.is_file():
        raise RuntimeError("Source training did not produce a valid checkpoint")
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    return pd.DataFrame(rows)


class MetricAccumulator:
    def __init__(self, horizons: int):
        self.n = np.zeros(horizons, dtype=np.int64)
        self.abs_sum = np.zeros(horizons, dtype=np.float64)
        self.sq_sum = np.zeros(horizons, dtype=np.float64)
        self.y_sum = np.zeros(horizons, dtype=np.float64)
        self.y_sq_sum = np.zeros(horizons, dtype=np.float64)

    def add(self, prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        for horizon in range(prediction.shape[-1]):
            mask = np.asarray(valid[..., horizon]) > 0.5
            pred = np.asarray(prediction[..., horizon], dtype=np.float64)[mask]
            true = np.asarray(target[..., horizon], dtype=np.float64)[mask]
            if not true.size:
                continue
            error = pred - true
            self.n[horizon] += true.size
            self.abs_sum[horizon] += np.abs(error).sum()
            self.sq_sum[horizon] += np.square(error).sum()
            self.y_sum[horizon] += true.sum()
            self.y_sq_sum[horizon] += np.square(true).sum()

    @staticmethod
    def _row(n, abs_sum, sq_sum, y_sum, y_sq_sum, horizon):
        n = int(n)
        denominator = float(y_sq_sum - y_sum * y_sum / max(n, 1))
        return {
            "horizon": horizon,
            "n_samples": n,
            "mae": float(abs_sum / n) if n else float("nan"),
            "rmse": float(math.sqrt(sq_sum / n)) if n else float("nan"),
            "r2": 1.0 - float(sq_sum) / denominator if denominator > 1.0e-12 else float("nan"),
        }

    def frame(self, split: str) -> pd.DataFrame:
        rows = []
        for horizon in range(len(self.n)):
            row = self._row(
                self.n[horizon], self.abs_sum[horizon], self.sq_sum[horizon],
                self.y_sum[horizon], self.y_sq_sum[horizon], f"h{horizon + 1}",
            )
            row["eval_split"] = split
            rows.append(row)
        row = self._row(
            self.n.sum(), self.abs_sum.sum(), self.sq_sum.sum(),
            self.y_sum.sum(), self.y_sq_sum.sum(), "all",
        )
        row["eval_split"] = split
        rows.append(row)
        return pd.DataFrame(rows)


def evaluate_metrics(model, loader, pv_mean: float, pv_std: float, device: str, split: str) -> pd.DataFrame:
    import torch

    model.eval()
    accumulator = MetricAccumulator(8)
    with torch.no_grad():
        for batch in loader:
            prediction, _, _ = _source_loss_parts(model, batch, device, torch)
            pred_raw = prediction.detach().cpu().numpy() * pv_std + pv_mean
            target_raw = np.asarray(batch["source_target"], dtype=np.float32) * pv_std + pv_mean
            valid = np.asarray(batch["source_valid"], dtype=np.float32)
            accumulator.add(pred_raw, target_raw, valid)
    return accumulator.frame(split)


def write_guidance_cache(model, loader, time_df: pd.DataFrame, output_dir: Path, device: str) -> None:
    import torch

    n_issue = len(time_df)
    n_nodes = int(model.n_nodes)
    horizons = int(model.future_frames)
    cache_dir = output_dir / "guidance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    predicted = np.lib.format.open_memmap(
        cache_dir / "predicted_z.npy", mode="w+", dtype=np.float32,
        shape=(n_issue, n_nodes, horizons),
    )
    historical = np.lib.format.open_memmap(
        cache_dir / "historical_z.npy", mode="w+", dtype=np.float32,
        shape=(n_issue, n_nodes, horizons),
    )
    predicted_valid = np.lib.format.open_memmap(
        cache_dir / "predicted_valid.npy", mode="w+", dtype=np.uint8,
        shape=(n_issue, n_nodes, horizons),
    )
    historical_valid = np.lib.format.open_memmap(
        cache_dir / "historical_valid.npy", mode="w+", dtype=np.uint8,
        shape=(n_issue, n_nodes, horizons),
    )
    expected_secs = time_df["issue_sec"].astype(np.int64).to_numpy()
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            prediction, _, _ = _source_loss_parts(model, batch, device, torch)
            pred = prediction.detach().cpu().numpy().astype(np.float32)
            latest = np.asarray(batch["latest_source_z"], dtype=np.float32)
            latest_valid = np.asarray(batch["latest_source_valid"], dtype=np.float32)
            issue_secs = np.asarray(batch["issue_sec"], dtype=np.int64)
            end = cursor + len(issue_secs)
            if not np.array_equal(issue_secs, expected_secs[cursor:end]):
                raise AssertionError("Guidance cache DataLoader order differs from canonical issue index")
            predicted[cursor:end] = pred
            historical[cursor:end] = np.repeat(latest[..., None], horizons, axis=2)
            predicted_valid[cursor:end] = 1
            historical_valid[cursor:end] = np.repeat(latest_valid[..., None], horizons, axis=2).astype(np.uint8)
            cursor = end
            if batch_index % 50 == 0 or cursor == n_issue:
                print(f"[source-cache] cached {cursor}/{n_issue} issue times", flush=True)
    if cursor != n_issue:
        raise AssertionError(f"Guidance cache incomplete: {cursor}/{n_issue}")
    for array in (predicted, historical, predicted_valid, historical_valid):
        array.flush()
    np.save(cache_dir / "issue_sec.npy", expected_secs)
    time_df.to_csv(cache_dir / "issue_time_index.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase2 = core.read_json(Path(args.phase2_report))
    if phase2.get("decision") != "READY FOR FULL EXPERIMENT IMPLEMENTATION":
        raise RuntimeError(
            f"Phase-2 gate is missing or not ready: {args.phase2_report}; "
            f"decision={phase2.get('decision')!r}"
        )
    core.set_deterministic_seed(int(args.seed))

    import torch
    from torch.utils.data import DataLoader

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if bool(args.disable_cudnn_rnn) and torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print("[source-full] disabled cuDNN RNN path for Stage-4 runtime parity", flush=True)
    device = str(args.device)
    paths = core.resolve_runtime_paths(args)
    data_dir = paths["stage3_data_dir"]
    model_dir = paths["stage3_model_dir"]
    selection_dir = data_dir / "02_target_selection_lhs50"
    run_dir = data_dir / "03_target_excluded_run"

    all_stations = core.load_station_table(run_dir / "station_split.csv", "full station split")
    sources = core.load_station_table(run_dir / "target_excluded_station_split.csv", "source split")
    targets = core.load_station_table(selection_dir / "target_stations.csv", "target split")
    overlap = set(sources["station_sn"]) & set(targets["station_sn"])
    if len(sources) != 446 or len(targets) != 50 or overlap:
        raise AssertionError(
            f"Invalid source/target partition: source={len(sources)} target={len(targets)} overlap={len(overlap)}"
        )

    time_df = core.canonical_issue_table(selection_dir / "issue_time_index.csv")
    time_df, swr_array, _, swr_cache_by_sec = core.align_time_to_swr(
        time_df, paths["swr_cache_dir"]
    )
    split_counts = time_df.groupby("time_split").size().astype(int).to_dict()
    required_splits = {"train", "val", "test"}
    if set(split_counts) != required_splits:
        raise AssertionError(f"Expected train/val/test after SWR intersection, got {split_counts}")
    time_df.to_csv(output_dir / "common_issue_time_index.csv", index=False)

    pv_stats = core.read_json(model_dir / "pv_field_norm_stats.json")
    ms_stats = core.read_json(model_dir / "multisource_norm_stats.json")
    swr_stats = core.read_json(model_dir / "swr_anchor_norm_stats.json")
    multisource_cache = core._load_multisource_cache(data_dir / "01_multisource_nwp_cache")
    value_store = core.SourceFleetValueStore(
        sources=sources,
        q95_csv=selection_dir / "target_excluded_station_pv_norm_q95.csv",
        required_times=core.all_row_times(time_df),
        train_pv_dir=paths["train_pv_dir"],
        val_pv_dir=paths["val_pv_dir"],
        output_dir=output_dir,
        status_filename="source_station_load_status.csv",
    )

    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        split_time = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
        datasets[split] = core.SourceGraphDataset(
            split_time, sources, all_stations, value_store, multisource_cache,
            swr_array, swr_cache_by_sec, pv_stats, ms_stats, swr_stats,
        )
        loaders[split] = DataLoader(
            datasets[split], batch_size=int(args.batch_size), shuffle=(split == "train"),
            num_workers=int(args.num_workers), pin_memory=str(device).startswith("cuda"),
        )
        print(f"[source-full] split={split} samples={len(datasets[split])}", flush=True)

    fixed_adjacency = core.load_fixed_adjacency(
        sources, Path(args.audit_dir) / "source_graph_knn_edges.csv"
    )
    model = core.make_fixed_graph_source_model(
        fixed_adjacency=fixed_adjacency,
        hist_dim=7,
        future_met_dim=8,
        future_irr_dim=10,
        future_frames=8,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        node_emb_dim=int(args.node_emb_dim),
        dropout=float(args.dropout),
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    checkpoint = output_dir / "source_graph_best.pt"

    first = next(iter(DataLoader(datasets["train"], batch_size=2, shuffle=False, num_workers=0)))
    with torch.no_grad():
        preflight = model(
            _tensor(first, "source_hist", device, torch),
            _tensor(first, "future_met", device, torch),
            _tensor(first, "future_irr", device, torch),
        )
    if tuple(preflight.shape) != (2, 446, 8):
        raise AssertionError(f"Unexpected full source model output shape: {tuple(preflight.shape)}")

    if args.cache_only:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--cache-only requires {checkpoint}")
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])
        train_log = pd.DataFrame()
    else:
        train_log = train_model(model, loaders, args, device, checkpoint)
        train_log.to_csv(output_dir / "source_graph_train_log.csv", index=False)

    metric_frames = []
    for split in ("val", "test"):
        frame = evaluate_metrics(
            model, loaders[split], float(pv_stats["pv_field_mean"]),
            float(pv_stats["pv_field_std"]), device, split,
        )
        metric_frames.append(frame)
        print(frame[frame["horizon"].isin(["all", "h1", "h4", "h8"])].to_string(index=False))
    source_metrics = pd.concat(metric_frames, ignore_index=True)
    source_metrics.insert(0, "model", "fixed_knn_graph_gru_source_forecaster")
    source_metrics.to_csv(output_dir / "source_graph_metrics.csv", index=False)

    full_dataset = core.SourceGraphDataset(
        time_df, sources, all_stations, value_store, multisource_cache,
        swr_array, swr_cache_by_sec, pv_stats, ms_stats, swr_stats,
    )
    full_loader = DataLoader(
        full_dataset, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=str(device).startswith("cuda"),
    )
    write_guidance_cache(model, full_loader, time_df, output_dir, device)
    cache_meta = {
        "schema_version": 1,
        "issue_count": len(time_df),
        "split_counts": split_counts,
        "source_count": len(sources),
        "target_count": len(targets),
        "source_station_sns": sources["station_sn"].astype(str).tolist(),
        "tensor_layout": "[issue, source_node, horizon]",
        "value_dtype": "float32",
        "historical_definition": "latest observed source PV state repeated over all 8 horizons",
        "predicted_definition": "fixed-kNN graph-GRU forecast; no future source PV input",
        "predicted_valid_definition": "model emits every node/horizon; set to one",
        "normalization": pv_stats,
        "future_anchor_weights": core.STAGE3_FUTURE_ANCHOR_WEIGHTS,
        "checkpoint": checkpoint,
    }
    _write_json(output_dir / "guidance_cache" / "metadata.json", cache_meta)
    run_meta = {
        "decision": "SOURCE GRAPH COMPLETE",
        "architecture": "fixed-kNN input/state graph propagation + shared node GRU + future decoder",
        "parameter_count": parameter_count,
        "paths": paths,
        "args": vars(args),
        "split_counts": split_counts,
        "fairness": {
            "same_446_source_stations": len(sources) == 446,
            "same_50_target_stations": len(targets) == 50,
            "source_target_disjoint": not overlap,
            "same_history_window": True,
            "same_8_forecast_horizons": True,
            "same_future_nwp_cache": str(data_dir / "01_multisource_nwp_cache"),
            "same_future_irradiance_cache": str(paths["swr_cache_dir"]),
            "target_not_graph_node": True,
            "future_source_pv_is_label_only": True,
        },
    }
    _write_json(output_dir / "run_meta.json", run_meta)
    print("==== Source graph full experiment complete ====")
    print("checkpoint:", checkpoint)
    print("guidance cache:", output_dir / "guidance_cache")
    print("metrics:", output_dir / "source_graph_metrics.csv")


if __name__ == "__main__":
    main()
