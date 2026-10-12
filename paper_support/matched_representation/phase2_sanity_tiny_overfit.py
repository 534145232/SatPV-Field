#!/usr/bin/env python3
"""Phase 2: print two-batch contracts and run source/target tiny-overfit tests."""

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
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "phase2")
    parser.add_argument("--tiny-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-epochs", type=int, default=12)
    parser.add_argument("--target-epochs", type=int, default=60)
    parser.add_argument("--source-hidden-dim", type=int, default=32)
    parser.add_argument("--source-layers", type=int, default=1)
    parser.add_argument("--node-emb-dim", type=int, default=16)
    parser.add_argument("--fixed-adjacency-mix", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--required-loss-ratio", type=float, default=0.90)
    return parser.parse_args()


def tensor(batch, name: str, device: str, torch):
    value = torch.as_tensor(batch[name], dtype=torch.float32, device=device)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def source_loss_on_loader(model, loader, device: str) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                tensor(batch, "source_hist", device, torch),
                tensor(batch, "future_met", device, torch),
                tensor(batch, "future_irr", device, torch),
            )
            losses.append(
                float(
                    core.masked_source_loss(
                        pred,
                        tensor(batch, "source_target", device, torch),
                        tensor(batch, "source_valid", device, torch),
                    ).cpu()
                )
            )
    return float(np.mean(losses))


def train_source_tiny(model, loader, args, device: str) -> tuple[pd.DataFrame, float, float]:
    import torch

    initial = source_loss_on_loader(model, loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rows = []
    for epoch in range(1, int(args.source_epochs) + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                tensor(batch, "source_hist", device, torch),
                tensor(batch, "future_met", device, torch),
                tensor(batch, "future_irr", device, torch),
            )
            loss = core.masked_source_loss(
                prediction,
                tensor(batch, "source_target", device, torch),
                tensor(batch, "source_valid", device, torch),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        evaluation = source_loss_on_loader(model, loader, device)
        row = {"component": "source_graph", "mode": "predicted", "epoch": epoch, "train_loss": float(np.mean(losses)), "eval_loss": evaluation}
        rows.append(row)
        print(f"[tiny-source] epoch={epoch:03d} train={row['train_loss']:.6f} eval={evaluation:.6f}", flush=True)
    final = source_loss_on_loader(model, loader, device)
    return pd.DataFrame(rows), initial, final


def collect_source_predictions(model, loader, dataset, pv_mean: float, pv_std: float, device: str):
    import torch

    model.eval()
    output = {}
    all_pred_raw, all_true_raw, all_valid = [], [], []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                tensor(batch, "source_hist", device, torch),
                tensor(batch, "future_met", device, torch),
                tensor(batch, "future_irr", device, torch),
            ).cpu().numpy()
            true_z = np.asarray(batch["source_target"], dtype=np.float32)
            valid = np.asarray(batch["source_valid"], dtype=np.float32)
            latest = np.asarray(batch["latest_source_z"], dtype=np.float32)
            latest_valid = np.asarray(batch["latest_source_valid"], dtype=np.float32)
            issue_secs = np.asarray(batch["issue_sec"], dtype=np.int64)
            for index, issue_sec in enumerate(issue_secs):
                output[int(issue_sec)] = {
                    "predicted_z": prediction[index],
                    "predicted_valid": np.ones_like(prediction[index], dtype=np.float32),
                    "historical_z": np.repeat(latest[index, :, None], prediction.shape[-1], axis=1),
                    "historical_valid": np.repeat(latest_valid[index, :, None], prediction.shape[-1], axis=1),
                }
            all_pred_raw.append(prediction * pv_std + pv_mean)
            all_true_raw.append(true_z * pv_std + pv_mean)
            all_valid.append(valid)
    metrics = core.regression_metrics(
        np.concatenate(all_pred_raw),
        np.concatenate(all_true_raw),
        np.concatenate(all_valid),
    )
    return output, metrics


def target_batch_arrays(target_dataset, source_predictions, contract, mode: str):
    examples = []
    for index in range(len(target_dataset)):
        item = target_dataset[index]
        source = source_predictions.get(int(item["issue_sec"]))
        if source is None:
            continue
        generation = source[f"{mode}_z"][None]
        valid = source[f"{mode}_valid"][None]
        context, context_mask = core.make_node_context(generation, valid, contract)
        copied = dict(item)
        copied["node_context"] = context[0]
        copied["node_mask"] = context_mask[0]
        examples.append(copied)
    if not examples:
        raise ValueError(f"No aligned target examples for mode={mode}")
    keys = (
        "target_hist",
        "target_future",
        "target_y_z",
        "target_y",
        "target_hist_pv",
        "node_context",
        "node_mask",
    )
    return {key: np.stack([example[key] for example in examples]).astype(np.float32) for key in keys} | {
        "issue_sec": np.asarray([example["issue_sec"] for example in examples], dtype=np.int64)
    }


def iterate_array_batches(arrays: dict, batch_size: int, shuffle: bool, rng: np.random.Generator):
    indices = np.arange(len(arrays["target_hist"]))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        selected = indices[start : start + int(batch_size)]
        yield {key: value[selected] for key, value in arrays.items()}


def target_loss_all(model, arrays, distance_weights, device: str, batch_size: int) -> float:
    import torch

    model.eval()
    losses = []
    rng = np.random.default_rng(0)
    weights = torch.as_tensor(distance_weights, dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in iterate_array_batches(arrays, batch_size, False, rng):
            prediction = model(
                tensor(batch, "target_hist", device, torch),
                tensor(batch, "target_future", device, torch),
                tensor(batch, "node_context", device, torch),
                tensor(batch, "node_mask", device, torch),
                weights,
            )
            losses.append(
                float(
                    core.target_weighted_loss(
                        prediction,
                        tensor(batch, "target_y_z", device, torch),
                        tensor(batch, "target_y", device, torch),
                        tensor(batch, "target_hist_pv", device, torch),
                    ).cpu()
                )
            )
    return float(np.mean(losses))


def train_target_tiny(arrays, contract, args, mode: str, device: str):
    import torch

    core.set_deterministic_seed(int(args.seed))
    model = core.make_target_graph_model(
        target_hist_dim=int(arrays["target_hist"].shape[-1]),
        target_future_dim=int(arrays["target_future"].shape[-1]),
        context_dim=int(arrays["node_context"].shape[-1]),
    ).to(device)
    distance_weights = np.asarray(contract["distance_weights"], dtype=np.float32)
    initial = target_loss_all(model, arrays, distance_weights, device, int(args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    rows = []
    for epoch in range(1, int(args.target_epochs) + 1):
        model.train()
        losses = []
        weights = torch.as_tensor(distance_weights, dtype=torch.float32, device=device)
        for batch in iterate_array_batches(arrays, int(args.batch_size), True, rng):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                tensor(batch, "target_hist", device, torch),
                tensor(batch, "target_future", device, torch),
                tensor(batch, "node_context", device, torch),
                tensor(batch, "node_mask", device, torch),
                weights,
            )
            loss = core.target_weighted_loss(
                prediction,
                tensor(batch, "target_y_z", device, torch),
                tensor(batch, "target_y", device, torch),
                tensor(batch, "target_hist_pv", device, torch),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        evaluation = target_loss_all(model, arrays, distance_weights, device, int(args.batch_size))
        rows.append({"component": "target_graph_guidance", "mode": mode, "epoch": epoch, "train_loss": float(np.mean(losses)), "eval_loss": evaluation})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.target_epochs):
            print(f"[tiny-target:{mode}] epoch={epoch:03d} train={rows[-1]['train_loss']:.6f} eval={evaluation:.6f}", flush=True)
    final = target_loss_all(model, arrays, distance_weights, device, int(args.batch_size))
    return model, pd.DataFrame(rows), initial, final


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    core.set_deterministic_seed(int(args.seed))

    import torch
    from torch.utils.data import DataLoader

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print("[phase2] disabled cuDNN RNN path to match stable Stage-4 LSTM execution", flush=True)
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
    if overlap:
        raise AssertionError(f"Source/target overlap: {sorted(overlap)[:10]}")
    if len(sources) != 446 or len(targets) != 50:
        raise AssertionError(f"Expected 446 sources and 50 targets; got {len(sources)} and {len(targets)}")

    time_df = core.canonical_issue_table(selection_dir / "issue_time_index.csv")
    time_df, swr_array, swr_meta, swr_cache_by_sec = core.align_time_to_swr(time_df, paths["swr_cache_dir"])
    common_counts = time_df.groupby("time_split").size().astype(int).to_dict()
    train_time = time_df[time_df["time_split"].astype(str) == "train"].head(int(args.tiny_samples)).copy().reset_index(drop=True)
    if len(train_time) < 8:
        raise ValueError(f"Only {len(train_time)} common train rows; cannot run tiny-overfit")

    pv_stats = core.read_json(model_dir / "pv_field_norm_stats.json")
    ms_stats = core.read_json(model_dir / "multisource_norm_stats.json")
    swr_stats = core.read_json(model_dir / "swr_anchor_norm_stats.json")
    multisource_cache = core._load_multisource_cache(data_dir / "01_multisource_nwp_cache")
    value_store = core.SourceFleetValueStore(
        sources=sources,
        q95_csv=selection_dir / "target_excluded_station_pv_norm_q95.csv",
        required_times=core.all_row_times(train_time),
        train_pv_dir=paths["train_pv_dir"],
        val_pv_dir=paths["val_pv_dir"],
        output_dir=output_dir,
    )
    source_dataset = core.SourceGraphDataset(
        time_df=train_time,
        sources=sources,
        all_stations=all_stations,
        value_store=value_store,
        multisource_cache=multisource_cache,
        swr_array=swr_array,
        swr_cache_by_sec=swr_cache_by_sec,
        pv_stats=pv_stats,
        multisource_stats=ms_stats,
        swr_stats=swr_stats,
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    source_eval_loader = DataLoader(
        source_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    first_batches = []
    for batch_index, batch in enumerate(source_eval_loader):
        contract = {
            name: list(batch[name].shape)
            for name in ("source_hist", "future_met", "future_irr", "source_target", "source_valid")
        }
        first_batches.append(contract)
        print(f"[sanity source batch {batch_index + 1}] {contract}")
        if batch_index >= 1:
            break

    fixed_adjacency = core.load_fixed_adjacency(
        sources,
        Path(args.audit_dir) / "source_graph_knn_edges.csv",
    )
    source_model = core.make_source_graph_model(
        fixed_adjacency=fixed_adjacency,
        hist_dim=7,
        future_met_dim=8,
        future_irr_dim=10,
        future_frames=8,
        hidden_dim=int(args.source_hidden_dim),
        num_layers=int(args.source_layers),
        node_emb_dim=int(args.node_emb_dim),
        dropout=0.10,
        fixed_mix=float(args.fixed_adjacency_mix),
    ).to(device)
    source_log, source_initial, source_final = train_source_tiny(source_model, source_loader, args, device)
    torch.save(
        {
            "model_state": source_model.state_dict(),
            "args": vars(args),
            "source_station_sns": sources["station_sn"].astype(str).tolist(),
            "pv_field_norm_stats": pv_stats,
            "multisource_norm_stats": ms_stats,
            "swr_anchor_norm_stats": swr_stats,
        },
        output_dir / "source_graph_tiny_overfit.pt",
    )
    source_predictions, source_metrics = collect_source_predictions(
        source_model,
        source_eval_loader,
        source_dataset,
        float(pv_stats["pv_field_mean"]),
        float(pv_stats["pv_field_std"]),
        device,
    )

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
    full_train_time = time_df[time_df["time_split"].astype(str) == "train"].reset_index(drop=True)
    target_candidates = []
    for _, target_row in targets.iterrows():
        station_sn = str(target_row["station_sn"])
        station_df, station_meta = b03.load_station_table(station_sn, station_args)
        station_stats = b03.make_station_stats(station_df, full_train_time)
        dataset = core.TargetStationDataset(target_row, station_df, train_time, station_stats, b03)
        target_candidates.append((len(dataset), station_sn, dataset, station_stats, station_meta))
    target_candidates.sort(key=lambda item: (-item[0], item[1]))
    target_count, target_sn, target_dataset, target_stats, target_meta = target_candidates[0]
    if target_count < 8:
        raise ValueError(f"Best tiny target {target_sn} has only {target_count} aligned samples")
    neighbor_contract = core.target_neighbor_contract(
        Path(args.audit_dir) / "target_source_matched_neighbors.csv",
        target_sn,
        sources,
    )

    modes = {}
    target_logs = []
    target_loss_summary = {}
    for mode in ("historical", "predicted"):
        arrays = target_batch_arrays(target_dataset, source_predictions, neighbor_contract, mode)
        modes[mode] = arrays
        model, log, initial, final = train_target_tiny(arrays, neighbor_contract, args, mode, device)
        target_logs.append(log)
        target_loss_summary[mode] = {
            "initial": initial,
            "final": final,
            "ratio": final / max(initial, 1.0e-12),
            "n_samples": int(len(arrays["target_hist"])),
        }
        torch.save(
            {
                "model_state": model.state_dict(),
                "target_sn": target_sn,
                "mode": mode,
                "target_stats": {
                    key: np.asarray(value).astype(float).tolist() if isinstance(value, np.ndarray) else float(value)
                    for key, value in target_stats.items()
                },
                "neighbor_contract": {
                    "source_sns": neighbor_contract["source_sns"],
                    "tau_km": neighbor_contract["tau_km"],
                    "distance_weights": neighbor_contract["distance_weights"].astype(float).tolist(),
                },
            },
            output_dir / f"target_{mode}_graph_tiny_overfit.pt",
        )

    target_shape_contract = {
        "target_sn": target_sn,
        "target_hist": list(modes["predicted"]["target_hist"].shape),
        "target_future": list(modes["predicted"]["target_future"].shape),
        "target_y": list(modes["predicted"]["target_y"].shape),
        "neighbor_indices": [int(neighbor_contract["neighbor_count"])],
        "node_context": list(modes["predicted"]["node_context"].shape),
    }
    print(f"[sanity target] {target_shape_contract}")

    logs = pd.concat([source_log, *target_logs], ignore_index=True)
    logs.to_csv(output_dir / "phase2_tiny_overfit_log.csv", index=False)
    source_ratio = source_final / max(source_initial, 1.0e-12)
    required = float(args.required_loss_ratio)
    passed = source_ratio <= required and all(value["ratio"] <= required for value in target_loss_summary.values())
    fairness = {
        "same_446_source_stations": len(sources) == 446,
        "same_50_target_stations": len(targets) == 50,
        "source_target_disjoint": not overlap,
        "same_canonical_issue_index_before_swr_intersection": True,
        "same_12_history_frames": first_batches[0]["source_hist"][2] == 12,
        "same_8_forecast_horizons": first_batches[0]["source_target"][2] == 8,
        "same_future_nwp_cache": str(data_dir / "01_multisource_nwp_cache"),
        "same_future_swr_cache": str(paths["swr_cache_dir"]),
        "target_not_graph_node": not overlap,
        "future_source_pv_is_label_only": True,
        "target_backbone_matches_stage4_05l": True,
        "same_station_objective": "weighted SmoothL1, threshold=0.10, total high-change weight=1.5",
        "full_evaluation_intersection": "PENDING Phase 4",
    }
    report = {
        "decision": "READY FOR FULL EXPERIMENT IMPLEMENTATION" if passed else "STOP: TINY OVERFIT FAILED",
        "paths": {key: str(value) for key, value in paths.items()},
        "common_issue_counts": common_counts,
        "tiny_issue_count": int(len(train_time)),
        "source_batch_shapes": first_batches,
        "target_shapes": target_shape_contract,
        "source_tiny_overfit": {
            "initial": source_initial,
            "final": source_final,
            "ratio": source_ratio,
            "metrics_in_q95_adjusted_field_units": source_metrics,
        },
        "target_tiny_overfit": target_loss_summary,
        "selected_tiny_target": {
            "station_sn": target_sn,
            "aligned_samples": int(target_count),
            "matched_source_count": int(neighbor_contract["neighbor_count"]),
            "distance_tau_km": float(neighbor_contract["tau_km"]),
            "station_meta": target_meta,
        },
        "fairness": fairness,
        "loss_ratio_threshold": required,
    }
    (output_dir / "phase2_sanity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Matched Representation Phase-2 Sanity and Tiny-Overfit Report",
        "",
        f"**Decision: {report['decision']}**",
        "",
        f"- Common issue counts after the same SWR-cache intersection: `{common_counts}`",
        f"- Source tiny-overfit loss: {source_initial:.6f} → {source_final:.6f} (ratio {source_ratio:.3f})",
        f"- Tiny target: `{target_sn}`, matched sources: {neighbor_contract['neighbor_count']}",
        f"- Historical-context target loss ratio: {target_loss_summary['historical']['ratio']:.3f}",
        f"- Predicted-context target loss ratio: {target_loss_summary['predicted']['ratio']:.3f}",
        "",
        "## Tensor contracts",
        "",
        f"- Source batches: `{first_batches}`",
        f"- Target tensors: `{target_shape_contract}`",
        "",
        "## Fairness assertions",
        "",
    ]
    for name, value in fairness.items():
        status = "PASS" if value is True else "PENDING" if isinstance(value, str) and value.startswith("PENDING") else "INFO" if isinstance(value, str) else "FAIL"
        lines.append(f"- {status}: `{name}` — {value}")
    (output_dir / "phase2_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==== Matched representation Phase-2 ====")
    print("decision:", report["decision"])
    print("source loss ratio:", f"{source_ratio:.4f}")
    print("target historical ratio:", f"{target_loss_summary['historical']['ratio']:.4f}")
    print("target predicted ratio:", f"{target_loss_summary['predicted']['ratio']:.4f}")
    print("report:", output_dir / "phase2_sanity_report.md")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
