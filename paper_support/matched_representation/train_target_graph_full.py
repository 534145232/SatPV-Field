#!/usr/bin/env python3
"""Train 50 held-out target models with historical/predicted graph guidance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matched_representation_core as core


SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = {
    "historical": "HistoricalGraph",
    "predicted": "PredictedGraph",
}


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
    parser.add_argument("--source-dir", type=Path, default=SCRIPT_DIR / "outputs" / "full" / "source_graph")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "full" / "target_graph")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--future-hidden-dim", type=int, default=64)
    parser.add_argument("--context-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--large-delta-threshold", type=float, default=0.10)
    parser.add_argument("--big-change-loss-weight", type=float, default=0.50)
    parser.add_argument("--min-train-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-incomplete-targets", action="store_true")
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


class GuidanceCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.meta = core.read_json(self.cache_dir / "metadata.json")
        self.issue_sec = np.load(self.cache_dir / "issue_sec.npy", mmap_mode="r")
        self.predicted_z = np.load(self.cache_dir / "predicted_z.npy", mmap_mode="r")
        self.historical_z = np.load(self.cache_dir / "historical_z.npy", mmap_mode="r")
        self.predicted_valid = np.load(self.cache_dir / "predicted_valid.npy", mmap_mode="r")
        self.historical_valid = np.load(self.cache_dir / "historical_valid.npy", mmap_mode="r")
        self.by_sec = {int(value): index for index, value in enumerate(self.issue_sec)}
        shape = tuple(self.predicted_z.shape)
        expected = (len(self.issue_sec), int(self.meta.get("source_count", -1)), 8)
        if shape != expected:
            raise ValueError(f"Predicted guidance shape {shape} != metadata contract {expected}")
        for name in ("historical_z", "predicted_valid", "historical_valid"):
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"Guidance cache shape mismatch for {name}")

    def one(self, issue_sec: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
        index = self.by_sec.get(int(issue_sec))
        if index is None:
            raise KeyError(f"Issue {issue_sec} is absent from source guidance cache")
        values = np.asarray(getattr(self, f"{mode}_z")[index], dtype=np.float32)
        valid = np.asarray(getattr(self, f"{mode}_valid")[index], dtype=np.float32)
        return values, valid


def build_target_arrays(dataset, cache: GuidanceCache, contract: dict, mode: str) -> dict[str, np.ndarray]:
    examples = []
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        values, valid = cache.one(int(item["issue_sec"]), mode)
        context, context_mask = core.make_node_context(values[None], valid[None], contract)
        examples.append(
            {
                **item,
                "node_context": context[0],
                "node_mask": context_mask[0],
            }
        )
    if not examples:
        raise ValueError(f"No target samples for {dataset.station_sn}, mode={mode}")
    float_keys = (
        "target_hist", "target_future", "target_y_z", "target_y", "target_hist_pv",
        "node_context", "node_mask",
    )
    arrays = {
        key: np.stack([item[key] for item in examples]).astype(np.float32)
        for key in float_keys
    }
    arrays["issue_sec"] = np.asarray([item["issue_sec"] for item in examples], dtype=np.int64)
    arrays["issue_time"] = np.asarray([item["issue_time"] for item in examples], dtype=str)
    return arrays


def iterate_batches(arrays: dict, batch_size: int, shuffle: bool, rng: np.random.Generator):
    indices = np.arange(len(arrays["target_hist"]), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        selected = indices[start : start + int(batch_size)]
        yield {key: value[selected] for key, value in arrays.items()}


def _tensor(batch: dict, name: str, device: str, torch):
    value = torch.as_tensor(batch[name], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _loss_parts(prediction, batch, args, device: str, torch):
    import torch.nn.functional as F

    target_z = _tensor(batch, "target_y_z", device, torch)
    target_raw = _tensor(batch, "target_y", device, torch)
    history = _tensor(batch, "target_hist_pv", device, torch)
    weights = 1.0 + float(args.big_change_loss_weight) * (
        torch.abs(target_raw - history[:, -1][:, None]) >= float(args.large_delta_threshold)
    ).float()
    numerator = (F.smooth_l1_loss(prediction, target_z, reduction="none") * weights).sum()
    return numerator, torch.clamp(weights.sum(), min=1.0)


def evaluate_loss(model, arrays, distance_weights, args, device: str) -> float:
    import torch

    model.eval()
    numerator = 0.0
    denominator = 0.0
    weights = torch.as_tensor(distance_weights, dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in iterate_batches(arrays, int(args.batch_size), False, np.random.default_rng(0)):
            prediction = model(
                _tensor(batch, "target_hist", device, torch),
                _tensor(batch, "target_future", device, torch),
                _tensor(batch, "node_context", device, torch),
                _tensor(batch, "node_mask", device, torch),
                weights,
            )
            loss_sum, weight_sum = _loss_parts(prediction, batch, args, device, torch)
            numerator += float(loss_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
    return numerator / max(denominator, 1.0)


def train_one(model, arrays_by_split, contract, args, device: str, checkpoint: Path, seed: int) -> pd.DataFrame:
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    distance_weights = np.asarray(contract["distance_weights"], dtype=np.float32)
    distance_tensor = torch.as_tensor(distance_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(seed))
    best = float("inf")
    bad = 0
    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        numerator = 0.0
        denominator = 0.0
        for batch in iterate_batches(arrays_by_split["train"], int(args.batch_size), True, rng):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                _tensor(batch, "target_hist", device, torch),
                _tensor(batch, "target_future", device, torch),
                _tensor(batch, "node_context", device, torch),
                _tensor(batch, "node_mask", device, torch),
                distance_tensor,
            )
            loss_sum, weight_sum = _loss_parts(prediction, batch, args, device, torch)
            loss = loss_sum / weight_sum
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            numerator += float(loss_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
        train_loss = numerator / max(denominator, 1.0)
        val_loss = evaluate_loss(model, arrays_by_split["val"], distance_weights, args, device)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[target-full] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best - 1.0e-6:
            best = val_loss
            bad = 0
            torch.save({"model_state": model.state_dict(), "best_val_loss": best}, checkpoint)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[target-full] early stop at epoch={epoch}", flush=True)
                break
    if not checkpoint.is_file():
        raise RuntimeError(f"No target checkpoint produced: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    return pd.DataFrame(rows)


def evaluate(model, arrays, distance_weights, stats: dict, station_sn: str, method: str, split: str, args, device: str):
    import torch

    model.eval()
    weights = torch.as_tensor(distance_weights, dtype=torch.float32, device=device)
    details = []
    pv_mean = float(stats["pv_mean"])
    pv_std = float(stats["pv_std"])
    with torch.no_grad():
        for batch in iterate_batches(arrays, int(args.batch_size), False, np.random.default_rng(0)):
            prediction_z = model(
                _tensor(batch, "target_hist", device, torch),
                _tensor(batch, "target_future", device, torch),
                _tensor(batch, "node_context", device, torch),
                _tensor(batch, "node_mask", device, torch),
                weights,
            ).detach().cpu().numpy()
            prediction = prediction_z * pv_std + pv_mean
            target = np.asarray(batch["target_y"], dtype=np.float32)
            history = np.asarray(batch["target_hist_pv"], dtype=np.float32)
            for item in range(prediction.shape[0]):
                last = float(history[item, -1])
                for horizon in range(prediction.shape[1]):
                    error = float(prediction[item, horizon] - target[item, horizon])
                    delta = float(target[item, horizon] - last)
                    details.append(
                        {
                            "station_sn": station_sn,
                            "method": method,
                            "model": method,
                            "eval_split": split,
                            "issue_sec": int(batch["issue_sec"][item]),
                            "issue_time": str(batch["issue_time"][item]),
                            "h": horizon + 1,
                            "horizon": f"h{horizon + 1}",
                            "pred_pv_norm": float(prediction[item, horizon]),
                            "true_pv_norm": float(target[item, horizon]),
                            "abs_err_pv_norm": abs(error),
                            "sq_err_pv_norm": error * error,
                            "true_delta_pv_norm": delta,
                            "large_change": abs(delta) >= float(args.large_delta_threshold),
                        }
                    )
    detail = pd.DataFrame(details)
    rows = []
    if not detail.empty:
        for horizon, group in list(detail.groupby("horizon")) + [("all", detail)]:
            metric = core.regression_metrics(
                group["pred_pv_norm"].to_numpy(), group["true_pv_norm"].to_numpy()
            )
            rows.append(
                {
                    "station_sn": station_sn,
                    "method": method,
                    "model": method,
                    "eval_split": split,
                    "horizon": horizon,
                    "n_samples": int(metric["n"]),
                    "mae_pv_norm": float(metric["mae"]),
                    "rmse_pv_norm": float(metric["rmse"]),
                    "r2_pv_norm": float(metric["r2"]),
                    "high_change_mae_pv_norm": float(
                        group.loc[group["large_change"], "abs_err_pv_norm"].mean()
                    ) if bool(group["large_change"].any()) else float("nan"),
                    "high_change_n": int(group["large_change"].sum()),
                }
            )
    return pd.DataFrame(rows), detail


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    core.set_deterministic_seed(int(args.seed))

    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print("[target-full] disabled cuDNN RNN path to match stable Stage-4 execution", flush=True)
    device = str(args.device)
    paths = core.resolve_runtime_paths(args)
    data_dir = paths["stage3_data_dir"]
    selection_dir = data_dir / "02_target_selection_lhs50"
    run_dir = data_dir / "03_target_excluded_run"
    sources = core.load_station_table(run_dir / "target_excluded_station_split.csv", "source split")
    targets = core.load_station_table(selection_dir / "target_stations.csv", "target split")
    if int(args.target_limit) > 0:
        targets = targets.head(int(args.target_limit)).copy()

    source_run_meta = core.read_json(Path(args.source_dir) / "run_meta.json")
    if source_run_meta.get("decision") != "SOURCE GRAPH COMPLETE":
        raise RuntimeError(
            f"Source graph run is not marked complete: {Path(args.source_dir) / 'run_meta.json'}"
        )
    cache = GuidanceCache(Path(args.source_dir) / "guidance_cache")
    cached_sns = [str(value) for value in cache.meta.get("source_station_sns", [])]
    if cached_sns != sources["station_sn"].astype(str).tolist():
        raise AssertionError("Source station order differs from guidance cache metadata")
    time_df = pd.read_csv(Path(args.source_dir) / "common_issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if not np.array_equal(time_df["issue_sec"].astype(np.int64).to_numpy(), np.asarray(cache.issue_sec)):
        raise AssertionError("Target common issue index differs from source guidance cache")

    b03 = core.stage4_temporal_module()
    station_args = SimpleNamespace(
        train_pv_dir=paths["train_pv_dir"],
        val_pv_dir=paths["val_pv_dir"],
        capacity_scale=1000.0,
        pv_health_max=1.35,
        health_filter=True,
        pv_health_active_threshold=0.05,
        pv_health_flatline_window=8,
        pv_health_flatline_range=1.0e-5,
    )
    train_time = time_df[time_df["time_split"].astype(str) == "train"].reset_index(drop=True)
    all_metrics, all_details, all_logs, station_meta = [], [], [], []
    completed = []
    skipped = []

    for target_index, target_row in targets.reset_index(drop=True).iterrows():
        station_sn = str(target_row["station_sn"])
        print(f"==== target graph {target_index + 1}/{len(targets)}: {station_sn} ====", flush=True)
        station_df, load_meta = b03.load_station_table(station_sn, station_args)
        stats = b03.make_station_stats(station_df, train_time)
        contract = core.target_neighbor_contract(
            Path(args.audit_dir) / "target_source_matched_neighbors.csv", station_sn, sources
        )
        datasets = {}
        for split in ("train", "val", "test"):
            split_time = time_df[time_df["time_split"].astype(str) == split].reset_index(drop=True)
            datasets[split] = core.TargetStationDataset(
                target_row, station_df, split_time, stats, b03
            )
            print(
                f"[target-full] {station_sn} split={split} samples={len(datasets[split])}",
                flush=True,
            )
        if len(datasets["train"]) < int(args.min_train_samples) or not len(datasets["val"]) or not len(datasets["test"]):
            skipped.append(
                {
                    "station_sn": station_sn,
                    "reason": "insufficient_samples",
                    **{f"{split}_samples": len(datasets[split]) for split in datasets},
                }
            )
            continue

        station_dir = output_dir / "per_station" / station_sn
        station_dir.mkdir(parents=True, exist_ok=True)
        station_meta.append(
            {
                **load_meta,
                "neighbor_count": int(contract["neighbor_count"]),
                "distance_tau_km": float(contract["tau_km"]),
                **{f"{split}_samples": len(datasets[split]) for split in datasets},
            }
        )
        write_json(
            station_dir / "normalization_and_neighbors.json",
            {
                "target_stats": stats,
                "source_sns": contract["source_sns"],
                "distance_weights": contract["distance_weights"],
                "tau_km": contract["tau_km"],
                "selection_rule": "source Gaussian support intersects target-centered 5x5 field patch",
            },
        )

        station_mode_details = {}
        for mode, method in METHODS.items():
            mode_dir = station_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            detail_path = mode_dir / "eval_detail.csv"
            metric_path = mode_dir / "metrics.csv"
            log_path = mode_dir / "train_log.csv"
            if args.resume and detail_path.is_file() and metric_path.is_file() and log_path.is_file():
                detail = pd.read_csv(detail_path, dtype={"station_sn": str})
                metrics = pd.read_csv(metric_path, dtype={"station_sn": str})
                log = pd.read_csv(log_path)
                print(f"[target-full] resume existing {station_sn}/{mode}", flush=True)
            else:
                arrays_by_split = {
                    split: build_target_arrays(datasets[split], cache, contract, mode)
                    for split in ("train", "val", "test")
                }
                # Identical seed per station for both modes gives matched model
                # initialization and shuffle order.
                station_seed = int(args.seed) + int(target_index)
                core.set_deterministic_seed(station_seed)
                model = core.make_target_graph_model(
                    target_hist_dim=10,
                    target_future_dim=12,
                    context_dim=4,
                    future_frames=8,
                    hidden_dim=int(args.hidden_dim),
                    future_hidden_dim=int(args.future_hidden_dim),
                    context_hidden_dim=int(args.context_hidden_dim),
                    num_layers=int(args.num_layers),
                    dropout=float(args.dropout),
                ).to(device)
                checkpoint = mode_dir / "best.pt"
                log = train_one(
                    model, arrays_by_split, contract, args, device, checkpoint, station_seed
                )
                log.insert(0, "method", method)
                log.insert(0, "station_sn", station_sn)
                metric_parts, detail_parts = [], []
                for split in ("val", "test"):
                    metric_part, detail_part = evaluate(
                        model, arrays_by_split[split], contract["distance_weights"], stats,
                        station_sn, method, split, args, device,
                    )
                    metric_parts.append(metric_part)
                    detail_parts.append(detail_part)
                metrics = pd.concat(metric_parts, ignore_index=True)
                detail = pd.concat(detail_parts, ignore_index=True)
                log.to_csv(log_path, index=False)
                metrics.to_csv(metric_path, index=False)
                detail.to_csv(detail_path, index=False)
                write_json(
                    mode_dir / "model_meta.json",
                    {
                        "method": method,
                        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                        "seed": station_seed,
                        "args": vars(args),
                    },
                )
            all_logs.append(log)
            all_metrics.append(metrics)
            all_details.append(detail)
            station_mode_details[mode] = detail

        key_cols = ["station_sn", "eval_split", "issue_time", "h"]
        historical_keys = station_mode_details["historical"][key_cols].astype(str)
        predicted_keys = station_mode_details["predicted"][key_cols].astype(str)
        if not historical_keys.equals(predicted_keys):
            raise AssertionError(f"Historical/predicted evaluation rows differ for {station_sn}")
        completed.append(station_sn)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    logs_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    metrics_df.to_csv(output_dir / "target_graph_metrics.csv", index=False)
    details_df.to_csv(output_dir / "target_graph_eval_detail.csv", index=False)
    logs_df.to_csv(output_dir / "target_graph_train_log.csv", index=False)
    pd.DataFrame(station_meta).to_csv(output_dir / "target_station_meta.csv", index=False)
    pd.DataFrame(skipped).to_csv(output_dir / "skipped_targets.csv", index=False)

    expected_count = len(targets)
    all_complete = len(completed) == expected_count and not skipped
    test_details = details_df[details_df.get("eval_split", pd.Series(dtype=str)).astype(str) == "test"].copy()
    sample_intersection = True
    if not test_details.empty:
        key_sets = []
        for method in METHODS.values():
            keys = set(map(tuple, test_details[test_details["method"] == method][["station_sn", "issue_time", "h"]].astype(str).to_numpy()))
            key_sets.append(keys)
        sample_intersection = len(key_sets) == 2 and key_sets[0] == key_sets[1]
    run_meta = {
        "decision": "TARGET GRAPH COMPLETE" if all_complete and sample_intersection else "INCOMPLETE",
        "completed_targets": completed,
        "skipped_targets": skipped,
        "expected_target_count": expected_count,
        "common_issue_counts": time_df.groupby("time_split").size().astype(int).to_dict(),
        "paths": paths,
        "args": vars(args),
        "fairness": {
            "same_446_source_stations": len(sources) == 446,
            "same_50_target_stations": len(targets) == 50,
            "same_target_local_backbone": True,
            "same_station_objective": "weighted SmoothL1 threshold=0.10 total high-change weight=1.5",
            "target_not_graph_node": True,
            "future_source_pv_not_in_target_input": True,
            "historical_predicted_same_evaluation_rows": sample_intersection,
            "all_requested_targets_completed": all_complete,
        },
    }
    write_json(output_dir / "run_meta.json", run_meta)
    print("==== Target graph full experiment ====")
    print("decision:", run_meta["decision"])
    print("completed targets:", len(completed), "/", expected_count)
    print("same graph evaluation rows:", sample_intersection)
    print("details:", output_dir / "target_graph_eval_detail.csv")
    if (not all_complete or not sample_intersection) and not args.allow_incomplete_targets:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
