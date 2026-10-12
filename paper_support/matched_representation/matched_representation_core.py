#!/usr/bin/env python3
"""Core data/model utilities for the matched field-versus-graph experiment.

The module deliberately lives under paper_support and does not modify the
production Stage-3/Stage-4 code.  It imports their preprocessing helpers so the
graph route uses the same values, masks, time features, NWP cache, SWR anchors,
and target-local station backbone as the field route.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_SUPPORT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STAGE1_DIR = PROJECT_ROOT / "stage_1"
STAGE3_DIR = PROJECT_ROOT / "stage_3"
STAGE4_DIR = PROJECT_ROOT / "stage_4"
for path in (PAPER_SUPPORT_DIR, STAGE1_DIR, STAGE3_DIR, STAGE4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_matched_representation import (  # noqa: E402
    bounds_from_all_stations,
    canonical_issue_table,
    load_station_table,
    resolve_swr_cache_dir,
    xy_to_ij,
)
from paper_support_paths import (  # noqa: E402
    resolve_old_data_root,
    resolve_pv_directories,
    resolve_stage3_data_dir,
)
from build_clear_sky_cache import haurwitz_clear_sky, solar_cos_zenith_bjt  # noqa: E402
from train_fullfield_dense_temporal_forecast import _parse_time_list  # noqa: E402
from train_fullfield_dense_temporal_residual_unet import _grid_lonlat, _time_solar_features  # noqa: E402
from train_fullfield_multisource_residual_unet import (  # noqa: E402
    _health_filter_station_series,
    _load_multisource_cache,
    _normalize_nwp,
)
from visualize_regional_split_frames import load_station_series  # noqa: E402


STAGE3_FUTURE_ANCHOR_WEIGHTS = np.asarray(
    [
        [0.50, 0.50, 0.00, 0.00],
        [0.25, 0.75, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.75, 0.25, 0.00],
        [0.00, 0.50, 0.50, 0.00],
        [0.00, 0.25, 0.75, 0.00],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.75, 0.25],
    ],
    dtype=np.float32,
)


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage4_temporal_module():
    return load_local_module(
        "matched_stage4_temporal",
        STAGE4_DIR / "03_train_per_station_temporal_baseline.py",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def resolve_stage3_model_dir(
    project_root: Path,
    old_data_root: Path,
    explicit: Path | None,
) -> Path:
    run_name = "04l_swr_anchor_motion_fusion_pvnorm_field_model_seed2026"
    candidates = [
        Path(explicit).expanduser() if explicit is not None else None,
        project_root / "stage_3" / "runs" / run_name,
        project_root / "stage3" / "runs" / run_name,
        old_data_root / "stage3" / "runs" / run_name,
        old_data_root / "stage_3" / "runs" / run_name,
    ]
    required = (
        "pv_field_norm_stats.json",
        "multisource_norm_stats.json",
        "swr_anchor_norm_stats.json",
    )
    tried = []
    for candidate in candidates:
        if candidate is None:
            continue
        missing = [name for name in required if not (candidate / name).is_file()]
        tried.append((candidate, missing))
        if not missing:
            return candidate.resolve()
    message = [
        "Could not resolve a Stage-3 model directory with all normalization files.",
        "Required: " + ", ".join(required),
    ]
    for candidate, missing in tried:
        message.append(f"  - {candidate}: missing {missing}")
    message.append("Pass --stage3-model-dir explicitly; do not recompute mismatched node statistics.")
    raise FileNotFoundError("\n".join(message))


def resolve_runtime_paths(args) -> dict[str, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    old_data_root = resolve_old_data_root(
        project_root,
        Path(args.old_data_root).expanduser() if args.old_data_root is not None else None,
        require_exists=True,
    )
    data_dir = resolve_stage3_data_dir(
        project_root,
        old_data_root=old_data_root,
        override=Path(args.stage3_data_dir).expanduser() if args.stage3_data_dir is not None else None,
    )
    model_dir = resolve_stage3_model_dir(
        project_root,
        old_data_root,
        Path(args.stage3_model_dir).expanduser() if args.stage3_model_dir is not None else None,
    )
    train_pv_dir, val_pv_dir = resolve_pv_directories(
        pv_root=Path(args.pv_root).expanduser() if args.pv_root is not None else None,
        train_override=Path(args.train_pv_dir).expanduser() if args.train_pv_dir is not None else None,
        val_override=Path(args.val_pv_dir).expanduser() if args.val_pv_dir is not None else None,
        project_root=project_root,
        old_data_root=old_data_root,
    )
    swr_dir, _ = resolve_swr_cache_dir(data_dir / "05_center_h8_swr_anchor_cache")
    return {
        "project_root": project_root,
        "old_data_root": old_data_root,
        "stage3_data_dir": data_dir,
        "stage3_model_dir": model_dir,
        "train_pv_dir": train_pv_dir,
        "val_pv_dir": val_pv_dir,
        "swr_cache_dir": swr_dir,
    }


def source_loader_args(train_pv_dir: Path, val_pv_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        train_pv_dir=Path(train_pv_dir),
        val_pv_dir=Path(val_pv_dir),
        capacity_scale=1000.0,
        health_filter_frame_cache=True,
        pv_health_max=1.35,
        pv_health_active_threshold=0.05,
        pv_health_flatline_window=8,
        pv_health_flatline_range=1.0e-5,
    )


class SourceFleetValueStore:
    """Time-indexed q95-adjusted source values matching field construction."""

    def __init__(
        self,
        sources: pd.DataFrame,
        q95_csv: Path,
        required_times: Iterable[pd.Timestamp],
        train_pv_dir: Path,
        val_pv_dir: Path,
        output_dir: Path,
        pv_clip: float = 1.5,
        status_filename: str = "phase2_source_station_load_status.csv",
    ):
        self.sources = sources.reset_index(drop=True).copy()
        self.station_sns = self.sources["station_sn"].astype(str).tolist()
        self.pv_clip = float(pv_clip)
        self.times = sorted({pd.Timestamp(value) for value in required_times})
        self.sec_to_index = {int(value.timestamp()): index for index, value in enumerate(self.times)}

        q95 = pd.read_csv(q95_csv, dtype={"station_sn": str})
        q95["station_sn"] = q95["station_sn"].astype(str)
        q95["pv_norm_q95"] = pd.to_numeric(q95["pv_norm_q95"], errors="coerce")
        q95_map = dict(zip(q95["station_sn"], q95["pv_norm_q95"]))
        missing_q95 = [sn for sn in self.station_sns if sn not in q95_map]
        if missing_q95:
            raise ValueError(f"Missing q95 denominators for {len(missing_q95)} sources: {missing_q95[:10]}")

        output_dir.mkdir(parents=True, exist_ok=True)
        loader_args = source_loader_args(train_pv_dir, val_pv_dir)
        series, status = load_station_series(self.sources, loader_args)
        status.to_csv(output_dir / str(status_filename), index=False)
        series = _health_filter_station_series(series, loader_args, output_dir)
        missing_series = [sn for sn in self.station_sns if sn not in series]
        if missing_series:
            raise FileNotFoundError(
                f"PV series missing for {len(missing_series)} source stations; first={missing_series[:10]}"
            )

        n_time = len(self.times)
        n_node = len(self.station_sns)
        values = np.zeros((n_time, n_node), dtype=np.float32)
        valid = np.zeros((n_time, n_node), dtype=np.float32)
        index = pd.DatetimeIndex(self.times)
        for node, sn in enumerate(self.station_sns):
            pv = pd.to_numeric(series[sn]["pv_norm"].reindex(index), errors="coerce").to_numpy(np.float32)
            denominator = float(q95_map[sn])
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"Invalid q95 for source {sn}: {denominator}")
            adjusted = pv / denominator
            mask = np.isfinite(adjusted) & (adjusted >= 0.0)
            values[:, node] = np.where(mask, np.clip(adjusted, 0.0, self.pv_clip), 0.0)
            valid[:, node] = mask.astype(np.float32)
        self.values = values
        self.valid = valid

    def get(self, times: Iterable[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
        indices = [self.sec_to_index[int(pd.Timestamp(value).timestamp())] for value in times]
        return self.values[indices].copy(), self.valid[indices].copy()


def all_row_times(time_df: pd.DataFrame) -> list[pd.Timestamp]:
    values: set[pd.Timestamp] = set()
    for _, row in time_df.iterrows():
        values.update(_parse_time_list(row["past_times"]))
        values.update(_parse_time_list(row["future_times"]))
    return sorted(values)


def align_time_to_swr(time_df: pd.DataFrame, swr_cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[int, int]]:
    meta_path = Path(swr_cache_dir) / "center_h8_swr_anchor_input_cache_meta.csv"
    array_path = Path(swr_cache_dir) / "swr_anchor_64.npy"
    if not meta_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"Incomplete active SWR cache: {swr_cache_dir}")
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    if "issue_sec" not in meta.columns:
        meta["issue_sec"] = meta["issue_time"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    if "cache_idx" not in meta.columns:
        meta["cache_idx"] = np.arange(len(meta), dtype=np.int64)
    cache_by_sec = {
        int(row["issue_sec"]): int(row["cache_idx"])
        for _, row in meta.iterrows()
    }
    out = time_df[time_df["issue_sec"].astype(np.int64).isin(cache_by_sec)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No canonical issue times overlap the active SWR cache")
    return out, np.load(array_path, mmap_mode="r"), meta, cache_by_sec


class SourceGraphDataset:
    """Source-only graph samples with Stage-3-parity inputs and labels."""

    def __init__(
        self,
        time_df: pd.DataFrame,
        sources: pd.DataFrame,
        all_stations: pd.DataFrame,
        value_store: SourceFleetValueStore,
        multisource_cache: dict,
        swr_array: np.ndarray,
        swr_cache_by_sec: dict[int, int],
        pv_stats: dict,
        multisource_stats: dict,
        swr_stats: dict,
        grid_size: int = 64,
    ):
        self.time_df = time_df.reset_index(drop=True).copy()
        self.sources = sources.reset_index(drop=True).copy()
        self.station_sns = self.sources["station_sn"].astype(str).tolist()
        self.value_store = value_store
        self.ms_cache = multisource_cache
        self.swr_array = swr_array
        self.swr_cache_by_sec = swr_cache_by_sec
        self.pv_mean = float(pv_stats["pv_field_mean"])
        self.pv_std = max(float(pv_stats["pv_field_std"]), 1.0e-6)
        self.ms_stats = multisource_stats
        self.swr_mean = np.asarray(swr_stats["mean"], dtype=np.float32).reshape(1, 2, 1, 1)
        self.swr_std = np.maximum(np.asarray(swr_stats["std"], dtype=np.float32), 1.0e-6).reshape(1, 2, 1, 1)
        self.grid_size = int(grid_size)
        self.bounds = bounds_from_all_stations(all_stations)
        self.cells = np.asarray(
            [xy_to_ij(float(row["lon"]), float(row["lat"]), self.bounds, self.grid_size) for _, row in self.sources.iterrows()],
            dtype=np.int64,
        )
        self.lon_grid, self.lat_grid = _grid_lonlat(self.bounds, self.grid_size)
        self.ms_issue_index = {
            int(value): index
            for index, value in enumerate(self.ms_cache["time_df"]["issue_sec"].astype(np.int64).tolist())
        }
        self.feature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.time_df)

    def _sample_grid(self, array: np.ndarray) -> np.ndarray:
        """Sample trailing H,W dimensions at source grid cells."""

        arr = np.asarray(array)
        y = self.cells[:, 0]
        x = self.cells[:, 1]
        return arr[..., y, x]

    def _time_clear_at_nodes(self, timestamp: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        sec = int(pd.Timestamp(timestamp).timestamp())
        cached = self.feature_cache.get(sec)
        if cached is None:
            time_solar_grid = _time_solar_features(timestamp, self.lon_grid, self.lat_grid)
            cosz = solar_cos_zenith_bjt(timestamp, self.lat_grid, self.lon_grid)
            clear_grid = haurwitz_clear_sky(cosz).astype(np.float32) / 1000.0
            time_solar = self._sample_grid(time_solar_grid).T.astype(np.float32)  # [N,3]
            clear = self._sample_grid(clear_grid).astype(np.float32)[:, None]  # [N,1]
            cached = (time_solar, clear)
            self.feature_cache[sec] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.time_df.iloc[int(index)]
        issue_sec = int(row["issue_sec"])
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])

        hist_raw_tn, hist_mask_tn = self.value_store.get(past_times)
        future_raw_tn, future_mask_tn = self.value_store.get(future_times)
        hist_z_tn = (hist_raw_tn - self.pv_mean) / self.pv_std
        hist_z_tn = np.where(hist_mask_tn > 0.5, hist_z_tn, 0.0).astype(np.float32)
        future_z_tn = (future_raw_tn - self.pv_mean) / self.pv_std
        future_z_tn = np.where(future_mask_tn > 0.5, future_z_tn, 0.0).astype(np.float32)

        delta_tn = np.zeros_like(hist_z_tn, dtype=np.float32)
        delta_valid = (hist_mask_tn[1:] > 0.5) & (hist_mask_tn[:-1] > 0.5)
        delta_tn[1:] = np.where(delta_valid, hist_z_tn[1:] - hist_z_tn[:-1], 0.0)
        historical = []
        for time_index, timestamp in enumerate(past_times):
            time_solar, clear = self._time_clear_at_nodes(timestamp)
            historical.append(
                np.concatenate(
                    [
                        hist_z_tn[time_index, :, None],
                        hist_mask_tn[time_index, :, None],
                        time_solar,
                        delta_tn[time_index, :, None],
                        clear,
                    ],
                    axis=1,
                )
            )
        source_hist = np.stack(historical, axis=1).astype(np.float32)  # [N,12,7]

        ms_index = self.ms_issue_index[issue_sec]
        nwp = _normalize_nwp(
            np.asarray(self.ms_cache["nwp"][ms_index], dtype=np.float32),
            np.asarray(self.ms_cache["nwp_valid"][ms_index], dtype=np.uint8),
            self.ms_stats,
            self.grid_size,
        )
        future_covariates = []
        for horizon, timestamp in enumerate(future_times):
            time_solar, clear = self._time_clear_at_nodes(timestamp)
            nwp_nodes = self._sample_grid(nwp[horizon]).T.astype(np.float32)  # [N,4]
            future_covariates.append(np.concatenate([time_solar, nwp_nodes, clear], axis=1))
        future_met = np.stack(future_covariates, axis=1).astype(np.float32)  # [N,8,8]

        swr = np.asarray(self.swr_array[self.swr_cache_by_sec[issue_sec]], dtype=np.float32)
        swr_z = (swr - self.swr_mean) / self.swr_std
        temporal_anchors = swr_z[[3, 4, 5, 6]]  # [T-1h,T,T+1h,T+2h], [4,2,H,W]
        aligned_swr = np.einsum("ha,acxy->hcxy", STAGE3_FUTURE_ANCHOR_WEIGHTS, temporal_anchors)
        aligned_nodes = self._sample_grid(aligned_swr).transpose(2, 0, 1).astype(np.float32)  # [N,8,2]
        # The field forecaster also encodes all four observed hourly SWR
        # anchors. Repeat their node-sampled representation at each decoder
        # horizon so the graph route receives the same seven-anchor evidence,
        # not only the already aligned two-channel future map.
        past_anchor_nodes = self._sample_grid(swr_z[:4]).transpose(2, 0, 1).reshape(len(self.sources), -1)
        past_anchor_nodes = np.repeat(past_anchor_nodes[:, None, :], len(future_times), axis=1)
        future_irr = np.concatenate([aligned_nodes, past_anchor_nodes], axis=2).astype(np.float32)  # [N,8,10]

        return {
            "source_hist": source_hist,
            "future_met": future_met,
            "future_irr": future_irr,
            "source_target": future_z_tn.T.astype(np.float32),
            "source_valid": future_mask_tn.T.astype(np.float32),
            "latest_source_z": hist_z_tn[-1].astype(np.float32),
            "latest_source_valid": hist_mask_tn[-1].astype(np.float32),
            "issue_sec": np.int64(issue_sec),
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def load_fixed_adjacency(sources: pd.DataFrame, edge_csv: Path) -> np.ndarray:
    edges = pd.read_csv(edge_csv, dtype={"source_u_sn": str, "source_v_sn": str})
    station_sns = sources["station_sn"].astype(str).tolist()
    index = {sn: idx for idx, sn in enumerate(station_sns)}
    n = len(station_sns)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for _, row in edges.iterrows():
        u = index.get(str(row["source_u_sn"]))
        v = index.get(str(row["source_v_sn"]))
        if u is None or v is None:
            raise ValueError(f"Graph edge contains a station outside the source split: {row.to_dict()}")
        weight = float(row["weight"])
        adjacency[u, v] = max(adjacency[u, v], weight)
        adjacency[v, u] = max(adjacency[v, u], weight)
    np.fill_diagonal(adjacency, 1.0)
    degree = np.maximum(adjacency.sum(axis=1), 1.0e-6)
    inv = 1.0 / np.sqrt(degree)
    normalized = inv[:, None] * adjacency * inv[None, :]
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Non-finite fixed adjacency")
    return normalized.astype(np.float32)


def make_source_graph_model(
    fixed_adjacency: np.ndarray,
    hist_dim: int,
    future_met_dim: int,
    future_irr_dim: int,
    future_frames: int,
    hidden_dim: int,
    num_layers: int,
    node_emb_dim: int,
    dropout: float,
    fixed_mix: float,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GeoAdaptiveGraphConv(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim * 3, output_dim)

        def forward(self, values, fixed_adj, adaptive_adj):
            fixed_values = torch.einsum("ij,bjf->bif", fixed_adj, values)
            adaptive_values = torch.einsum("ij,bjf->bif", adaptive_adj, values)
            return self.linear(torch.cat([values, fixed_values, adaptive_values], dim=-1))

    class Cell(nn.Module):
        def __init__(self, input_dim: int, hidden: int):
            super().__init__()
            self.gate = GeoAdaptiveGraphConv(input_dim + hidden, hidden * 2)
            self.update = GeoAdaptiveGraphConv(input_dim + hidden, hidden)

        def forward(self, values, state, fixed_adj, adaptive_adj):
            joined = torch.cat([values, state], dim=-1)
            update, reset = torch.chunk(torch.sigmoid(self.gate(joined, fixed_adj, adaptive_adj)), 2, dim=-1)
            candidate = torch.tanh(
                self.update(torch.cat([values, reset * state], dim=-1), fixed_adj, adaptive_adj)
            )
            return update * state + (1.0 - update) * candidate

    class SourceGraphForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_nodes = int(fixed_adjacency.shape[0])
            self.hidden_dim = int(hidden_dim)
            self.future_frames = int(future_frames)
            self.fixed_mix = float(fixed_mix)
            self.register_buffer("fixed_adjacency", torch.as_tensor(fixed_adjacency, dtype=torch.float32))
            self.nodevec1 = nn.Parameter(torch.randn(self.n_nodes, int(node_emb_dim)) * 0.05)
            self.nodevec2 = nn.Parameter(torch.randn(int(node_emb_dim), self.n_nodes) * 0.05)
            self.cells = nn.ModuleList(
                [
                    Cell(int(hist_dim) if layer == 0 else int(hidden_dim), int(hidden_dim))
                    for layer in range(max(1, int(num_layers)))
                ]
            )
            self.future_encoder = nn.Sequential(
                nn.Linear(int(future_met_dim + future_irr_dim), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            )
            self.horizon_embedding = nn.Embedding(int(future_frames), max(8, int(hidden_dim) // 2))
            self.decoder = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2 + max(8, int(hidden_dim) // 2), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )

        def adaptive_adjacency(self):
            return F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)

        def forward(self, source_hist, future_met, future_irr):
            batch, nodes, history, _ = source_hist.shape
            if nodes != self.n_nodes:
                raise ValueError(f"Expected {self.n_nodes} graph nodes, received {nodes}")
            learned = self.adaptive_adjacency()
            # Both message paths are explicit. fixed_mix=0.5 is frozen for the
            # controlled baseline rather than tuned on test performance.
            fixed_messages = self.fixed_mix * self.fixed_adjacency
            learned_messages = (1.0 - self.fixed_mix) * learned
            sequence = source_hist
            final_state = None
            for cell in self.cells:
                state = source_hist.new_zeros(batch, nodes, self.hidden_dim)
                outputs = []
                for step in range(history):
                    state = cell(sequence[:, :, step], state, fixed_messages, learned_messages)
                    outputs.append(state)
                sequence = torch.stack(outputs, dim=2)
                final_state = state
            future = self.future_encoder(torch.cat([future_met, future_irr], dim=-1))
            horizon_ids = torch.arange(self.future_frames, device=source_hist.device)
            horizon = self.horizon_embedding(horizon_ids)[None, None].expand(batch, nodes, -1, -1)
            history_context = final_state[:, :, None].expand(-1, -1, self.future_frames, -1)
            return self.decoder(torch.cat([history_context, future, horizon], dim=-1)).squeeze(-1)

    return SourceGraphForecaster()


def make_fixed_graph_source_model(
    fixed_adjacency: np.ndarray,
    hist_dim: int,
    future_met_dim: int,
    future_irr_dim: int,
    future_frames: int,
    hidden_dim: int = 96,
    num_layers: int = 2,
    node_emb_dim: int = 16,
    dropout: float = 0.10,
):
    """Compute-efficient full-run source forecaster on the fixed geographic graph.

    The Phase-2 recurrent graph cell intentionally exercises dense graph
    propagation at every recurrent gate.  That is useful for a tiny-overfit
    contract test, but unnecessarily expensive for 446 nodes and all issue
    times.  The full-run model below keeps the same information boundary and
    fixed kNN graph while applying graph propagation (i) to every historical
    input frame and (ii) once to the final recurrent state.  The temporal GRU
    is shared across nodes.  No target node or future source-PV label is an
    input.
    """

    import torch
    import torch.nn as nn

    adjacency = np.asarray(fixed_adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"fixed_adjacency must be square, received {adjacency.shape}")
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("Non-finite fixed adjacency")

    class FixedGraphSourceForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_nodes = int(adjacency.shape[0])
            self.hidden_dim = int(hidden_dim)
            self.future_frames = int(future_frames)
            indices = np.stack(np.nonzero(adjacency), axis=0)
            values = adjacency[indices[0], indices[1]]
            sparse = torch.sparse_coo_tensor(
                torch.as_tensor(indices, dtype=torch.long),
                torch.as_tensor(values, dtype=torch.float32),
                size=adjacency.shape,
            ).coalesce()
            self.register_buffer("fixed_adjacency", sparse)
            self.node_embedding = nn.Parameter(
                torch.randn(self.n_nodes, int(node_emb_dim), dtype=torch.float32) * 0.05
            )
            self.history_encoder = nn.GRU(
                input_size=int(hist_dim) * 2 + int(node_emb_dim),
                hidden_size=int(hidden_dim),
                num_layers=max(1, int(num_layers)),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
            self.history_projection = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
            )
            self.future_encoder = nn.Sequential(
                nn.Linear(int(future_met_dim + future_irr_dim), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            )
            self.horizon_embedding = nn.Embedding(
                int(future_frames), max(8, int(hidden_dim) // 2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(
                    int(hidden_dim) * 2 + int(node_emb_dim) + max(8, int(hidden_dim) // 2),
                    int(hidden_dim),
                ),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
                nn.GELU(),
                nn.Linear(int(hidden_dim) // 2, 1),
            )

        def graph_mix(self, values):
            # values: [B,N,...].  Sparse mm is performed over the node axis.
            batch, nodes = values.shape[:2]
            if nodes != self.n_nodes:
                raise ValueError(f"Expected {self.n_nodes} graph nodes, received {nodes}")
            tail = values.shape[2:]
            flat = values.permute(1, 0, *range(2, values.ndim)).reshape(nodes, -1)
            mixed = torch.sparse.mm(self.fixed_adjacency, flat)
            return mixed.reshape(nodes, batch, *tail).permute(1, 0, *range(2, 2 + len(tail)))

        def forward(self, source_hist, future_met, future_irr):
            batch, nodes, history, _ = source_hist.shape
            graph_history = self.graph_mix(source_hist)
            node_emb = self.node_embedding[None, :, None, :].expand(batch, -1, history, -1)
            history_input = torch.cat([source_hist, graph_history, node_emb], dim=-1)
            sequence = history_input.reshape(batch * nodes, history, -1)
            _, state = self.history_encoder(sequence)
            local_state = state[-1].reshape(batch, nodes, self.hidden_dim)
            graph_state = self.graph_mix(local_state)
            history_context = self.history_projection(torch.cat([local_state, graph_state], dim=-1))

            future = self.future_encoder(torch.cat([future_met, future_irr], dim=-1))
            horizon_ids = torch.arange(self.future_frames, device=source_hist.device)
            horizon = self.horizon_embedding(horizon_ids)[None, None].expand(batch, nodes, -1, -1)
            history_rep = history_context[:, :, None].expand(-1, -1, self.future_frames, -1)
            node_rep = self.node_embedding[None, :, None].expand(batch, -1, self.future_frames, -1)
            return self.decoder(
                torch.cat([history_rep, future, node_rep, horizon], dim=-1)
            ).squeeze(-1)

    return FixedGraphSourceForecaster()


def masked_source_loss(prediction, target, valid):
    import torch
    import torch.nn.functional as F

    mask = valid.float()
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


class TargetStationDataset:
    """Exact Stage-4 target-local inputs on a supplied common issue index."""

    def __init__(self, target_row: pd.Series, station_df: pd.DataFrame, time_df: pd.DataFrame, stats: dict, b03):
        self.target_row = target_row
        self.station_sn = str(target_row["station_sn"])
        self.df = station_df
        self.time_df = time_df.reset_index(drop=True)
        self.stats = stats
        self.b03 = b03
        self.samples: list[int] = []
        for index, row in self.time_df.iterrows():
            times = _parse_time_list(row["past_times"]) + _parse_time_list(row["future_times"])
            if all(pd.Timestamp(value) in self.df.index for value in times):
                self.samples.append(int(index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.time_df.iloc[self.samples[int(item)]]
        past_times = _parse_time_list(row["past_times"])
        future_times = _parse_time_list(row["future_times"])
        hist_raw = self.df.loc[past_times, ["pv_norm", *self.b03.FUTURE_NWP_NAMES]].to_numpy(np.float32)
        future_raw = self.df.loc[future_times, self.b03.FUTURE_NWP_NAMES].to_numpy(np.float32)
        target_raw = self.df.loc[future_times, "pv_norm"].to_numpy(np.float32)
        hist_z = (hist_raw - self.stats["hist_mean"]) / self.stats["hist_std"]
        future_z = (future_raw - self.stats["future_mean"]) / self.stats["future_std"]
        time_solar = np.stack(
            [
                self.b03._time_solar_point(
                    timestamp,
                    float(self.target_row["lon"]),
                    float(self.target_row["lat"]),
                )
                for timestamp in future_times
            ],
            axis=0,
        )
        return {
            "target_hist": hist_z.astype(np.float32),
            "target_future": np.concatenate([future_z, time_solar], axis=1).astype(np.float32),
            "target_y_z": ((target_raw - float(self.stats["pv_mean"])) / float(self.stats["pv_std"])).astype(np.float32),
            "target_y": target_raw.astype(np.float32),
            "target_hist_pv": hist_raw[:, 0].astype(np.float32),
            "issue_sec": np.int64(row["issue_sec"]),
            "issue_time": str(pd.Timestamp(row["issue_time"])),
        }


def make_target_graph_model(
    target_hist_dim: int,
    target_future_dim: int,
    context_dim: int,
    future_frames: int = 8,
    hidden_dim: int = 96,
    future_hidden_dim: int = 64,
    context_hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.10,
):
    import torch
    import torch.nn as nn

    class TargetGraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = int(hidden_dim)
            self.future_frames = int(future_frames)
            self.history_encoder = nn.LSTM(
                input_size=int(target_hist_dim),
                hidden_size=hidden,
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
                bidirectional=True,
            )
            self.history_projection = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
            )
            self.future_encoder = nn.Sequential(
                nn.Linear(int(target_future_dim), int(future_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(future_hidden_dim)),
                nn.Linear(int(future_hidden_dim), hidden),
                nn.GELU(),
            )
            self.node_encoder = nn.Sequential(
                nn.Linear(int(context_dim), int(context_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(context_hidden_dim)),
                nn.Linear(int(context_hidden_dim), hidden),
                nn.GELU(),
            )
            self.context_projection = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU()
            )
            self.horizon_embedding = nn.Embedding(int(future_frames), hidden // 2)
            self.decoder = nn.Sequential(
                nn.Linear(hidden * 3 + hidden // 2, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, target_hist, target_future, node_context, node_mask, distance_weights):
            encoded_history, _ = self.history_encoder(target_hist)
            history = self.history_projection(encoded_history[:, -1])
            future = self.future_encoder(target_future)
            nodes = self.node_encoder(node_context)
            weights = distance_weights[None, None, :] * node_mask.float()
            weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1.0e-6)
            context = self.context_projection((nodes * weights[..., None]).sum(dim=2))
            horizon_ids = torch.arange(self.future_frames, device=target_hist.device)
            horizon = self.horizon_embedding(horizon_ids)[None].expand(target_hist.shape[0], -1, -1)
            history = history[:, None].expand(-1, self.future_frames, -1)
            return self.decoder(torch.cat([history, future, context, horizon], dim=-1)).squeeze(-1)

    return TargetGraphModel()


def target_weighted_loss(prediction, target_z, target_raw, history_pv_raw, threshold: float = 0.10, extra_weight: float = 0.5):
    import torch
    import torch.nn.functional as F

    last = history_pv_raw[:, -1][:, None]
    weights = 1.0 + float(extra_weight) * (torch.abs(target_raw - last) >= float(threshold)).float()
    loss = F.smooth_l1_loss(prediction, target_z, reduction="none") * weights
    return loss.sum() / torch.clamp(weights.sum(), min=1.0)


def target_neighbor_contract(
    neighbors_csv: Path,
    target_sn: str,
    sources: pd.DataFrame,
) -> dict[str, Any]:
    frame = pd.read_csv(neighbors_csv, dtype={"target_sn": str, "source_sn": str})
    sub = frame[frame["target_sn"].astype(str) == str(target_sn)].copy()
    if sub.empty:
        raise ValueError(f"No matched source neighbors for target {target_sn}")
    sub = sub.sort_values("rank_by_distance").reset_index(drop=True)
    source_index = {sn: idx for idx, sn in enumerate(sources["station_sn"].astype(str).tolist())}
    indices = np.asarray([source_index[str(sn)] for sn in sub["source_sn"]], dtype=np.int64)
    dx = sub["delta_x_km"].to_numpy(np.float32)
    dy = sub["delta_y_km"].to_numpy(np.float32)
    distance = sub["distance_km"].to_numpy(np.float32)
    tau = max(float(np.median(frame["distance_km"].to_numpy(np.float64))), 1.0e-6)
    geometry = np.stack([dx / tau, dy / tau, distance / tau], axis=1).astype(np.float32)
    logits = -distance / tau
    logits = logits - float(np.max(logits))
    weights = np.exp(logits).astype(np.float32)
    weights /= max(float(weights.sum()), 1.0e-6)
    return {
        "source_indices": indices,
        "source_sns": sub["source_sn"].astype(str).tolist(),
        "geometry": geometry,
        "distance_weights": weights,
        "tau_km": tau,
        "neighbor_count": int(len(sub)),
    }


def make_node_context(
    generation_z: np.ndarray,
    valid: np.ndarray,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Create [B,H,K,4] generation/geometry contexts."""

    values = np.asarray(generation_z, dtype=np.float32)
    masks = np.asarray(valid, dtype=np.float32)
    indices = np.asarray(contract["source_indices"], dtype=np.int64)
    selected = values[:, indices, :].transpose(0, 2, 1)  # [B,H,K]
    selected_mask = masks[:, indices, :].transpose(0, 2, 1)
    geometry = np.asarray(contract["geometry"], dtype=np.float32)
    tiled_geometry = np.broadcast_to(geometry[None, None], (*selected.shape, 3))
    context = np.concatenate([selected[..., None], tiled_geometry], axis=-1).astype(np.float32)
    return context, selected_mask.astype(np.float32)


def regression_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    if valid is not None:
        mask &= np.asarray(valid) > 0.5
    pred = pred[mask]
    true = true[mask]
    if not true.size:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    error = pred - true
    denominator = float(np.sum((true - float(np.mean(true))) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1.0e-12 else float("nan")
    return {
        "n": int(true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
    }
