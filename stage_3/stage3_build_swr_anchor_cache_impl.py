#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", SCRIPT_DIR.parent)).expanduser().resolve()
WORKSPACE = PROJECT_ROOT
STAGE1_DIR = PROJECT_ROOT / "stage_1"
STAGE2_DIR = PROJECT_ROOT / "stage_2"
for p in [STAGE1_DIR, STAGE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_clear_sky_cache import haurwitz_clear_sky, solar_cos_zenith_bjt
from train_3dcnn_ratio_swr import Ratio3DCNN, RatioFrameReader, make_features
from visualize_regional_split_frames import _bounds, ensure_dir, write_json
import stage3_base_pvnorm_field_model as f04
import stage3_paths as p3

try:
    from h8_dataset_paths import CENTER_FRAME_CACHE_DIR, CENTER_INDEX_4H3F_ALL, CENTER_PROCESSED_ROOT
except ModuleNotFoundError:
    CENTER_PROCESSED_ROOT = p3.STAGE1_CENTER_MONTHLY_DIR
    CENTER_INDEX_4H3F_ALL = p3.STAGE1_CENTER_INDEX_ALL
    CENTER_FRAME_CACHE_DIR = p3.STAGE1_CENTER_FRAME_CACHE_DIR


USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

PYCHARM_DEFAULTS = {
    "selection_dir": str(p3.TARGET_SELECTION_DIR),
    "target_run_dir": str(p3.TARGET_EXCLUDED_RUN_DIR),
    "out_dir": str(p3.CENTER_H8_SWR_ANCHOR_CACHE_ROOT),
    "h8_center_index_csv": str(CENTER_INDEX_4H3F_ALL),
    "h8_center_frame_cache_dir": str(CENTER_FRAME_CACHE_DIR),
    "h8_center_processed_root": str(CENTER_PROCESSED_ROOT),
    "h8_netcdf_glob": "H8_region_multivar_*_CENTER_BJT_*.nc",
    "swr_checkpoint": str(p3.SWR_BASE_CHECKPOINT),
    "grid_size": 64,
    "swr_history_frames": 4,
    "swr_future_anchors": 3,
    "cache_dtype": "float16",
    "missing_history_policy": "skip",
    "hour_round_tolerance_minutes": 10.0,
    "max_samples": 0,
    "seed": 2026,
    "device": "cuda",
    "log_interval": 50,
}


def _inject_pycharm_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in PYCHARM_DEFAULTS.items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PV-issue-time aligned BigScale-center H8/SWR-anchor input cache. "
            "This cache is diagnostic-first: it stores SWR model inputs, predicted future SWR anchors, "
            "clear-sky anchors, metadata, and missing-H8 coverage reports."
        )
    )
    for key, value in PYCHARM_DEFAULTS.items():
        if key.endswith("_dir") or key.endswith("_csv") or key.endswith("_root") or key == "swr_checkpoint":
            parser.add_argument(f"--{key}", type=Path, default=Path(value))
        elif key in {"missing_history_policy", "cache_dtype", "device", "h8_netcdf_glob"}:
            parser.add_argument(f"--{key}", type=str, default=str(value))
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=int(value))
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=float(value))
        else:
            parser.add_argument(f"--{key}", type=str, default=str(value))
    parser.add_argument(
        "--force",
        type=int,
        default=0,
        help="Overwrite existing cache arrays.",
    )
    return parser.parse_args()


def _device(name: str) -> str:
    import torch

    if str(name).lower().startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(name)


def _timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def _ts_text(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _hour_key(ts: pd.Timestamp, tolerance_minutes: float) -> int:
    ts = pd.Timestamp(ts)
    rounded = ts.round("h")
    offset_min = abs((ts - rounded).total_seconds()) / 60.0
    if offset_min > float(tolerance_minutes):
        raise ValueError(f"H8 label time {ts} is too far from hourly label: offset={offset_min:.2f} min")
    return _timestamp_key(rounded)


def _grid_lonlat(bounds: dict, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    return f04._grid_lonlat(bounds, int(grid_size))


def _resample_to_grid(arr: np.ndarray, src_lon: np.ndarray, src_lat: np.ndarray, dst_lon: np.ndarray, dst_lat: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from a regular source grid to a destination mesh."""
    arr = np.asarray(arr, dtype=np.float32)
    lon = np.asarray(src_lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(src_lat, dtype=np.float64).reshape(-1)
    if arr.shape[-2] != len(lat) or arr.shape[-1] != len(lon):
        raise ValueError(f"shape mismatch for resample: arr={arr.shape}, lat={len(lat)}, lon={len(lon)}")
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[..., ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr = arr[..., ::-1, :]
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    if dst_lon.ndim == 1 and dst_lat.ndim == 1:
        dst_lon, dst_lat = np.meshgrid(dst_lon, dst_lat)
    else:
        dst_lon, dst_lat = np.broadcast_arrays(dst_lon, dst_lat)
    x = np.interp(dst_lon, lon, np.arange(len(lon), dtype=np.float64))
    y = np.interp(dst_lat, lat, np.arange(len(lat), dtype=np.float64))
    x0 = np.clip(np.floor(x).astype(np.int64), 0, len(lon) - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, len(lat) - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = arr[..., y0, x0]
    v01 = arr[..., y0, x1]
    v10 = arr[..., y1, x0]
    v11 = arr[..., y1, x1]
    return ((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)).astype(np.float32)


class CenterH8Reader:
    def __init__(self, args: argparse.Namespace, channels: list[str]):
        import xarray as xr

        self.args = args
        self.channels = channels
        self.reader = RatioFrameReader(channels, Path(args.h8_center_frame_cache_dir))
        self.time_lookup = self._build_time_lookup()
        self.sorted_keys = np.asarray(sorted(self.time_lookup.keys()), dtype=np.int64)
        nc0 = next(iter(self.time_lookup.values()))["nc_path"]
        with xr.open_dataset(str(nc0)) as ds:
            self.lat = np.asarray(ds["latitude"].values, dtype=np.float32)
            self.lon = np.asarray(ds["longitude"].values, dtype=np.float32)

    def _build_time_lookup(self) -> dict[int, dict[str, Any]]:
        import xarray as xr

        root = Path(self.args.h8_center_processed_root)
        paths = sorted(root.glob(str(self.args.h8_netcdf_glob)))
        if not paths:
            df = pd.read_csv(Path(self.args.h8_center_index_csv))
            paths = sorted({Path(p) for p in df["nc_path"].astype(str).tolist()})
        lookup: dict[int, dict[str, Any]] = {}
        mins, maxs = [], []
        for path in paths:
            with xr.open_dataset(str(path)) as ds:
                if "time_start_bjt" in ds:
                    times = pd.to_datetime(np.asarray(ds["time_start_bjt"].values).astype("datetime64[s]"))
                else:
                    times = pd.to_datetime(np.asarray(ds["time"].values).astype("datetime64[s]")) - pd.Timedelta(minutes=30)
                if len(times) == 0:
                    continue
                mins.append(pd.Timestamp(times.min()))
                maxs.append(pd.Timestamp(times.max()))
                for i, ts in enumerate(times):
                    key = _hour_key(pd.Timestamp(ts), float(self.args.hour_round_tolerance_minutes))
                    lookup[key] = {
                        "nc_path": str(path),
                        "time_idx": int(i),
                        "raw_label_time": _ts_text(pd.Timestamp(ts)),
                        "rounded_label_time": _ts_text(pd.Timestamp(ts).round("h")),
                    }
        if not lookup:
            raise RuntimeError(f"No center H8 timestamps found under {root}")
        print(
            "[center_h8_cache] center H8 lookup: "
            f"files={len(paths)} frames={len(lookup)} raw_label={_ts_text(min(mins))} -> {_ts_text(max(maxs))}",
            flush=True,
        )
        return lookup

    def close(self) -> None:
        self.reader.close()

    def read_frame(self, hit: dict[str, Any]) -> np.ndarray:
        nc_path = str(hit["nc_path"])
        time_idx = int(hit["time_idx"])
        fc = self.reader._open_frame_cache(nc_path)
        if fc is not None:
            lookup = fc["lookup"]
            idx = [lookup[c] for c in self.channels]
            return np.asarray(fc["data"][time_idx, :, :, :][idx], dtype=np.float32)
        ds = self.reader._open_ds(nc_path)
        names = [str(x) for x in ds["channel"].values.tolist()]
        lookup = {name: i for i, name in enumerate(names)}
        idx = [lookup[c] for c in self.channels]
        return np.asarray(ds["data"].isel(time=time_idx, channel=idx).values, dtype=np.float32)

    def history_hits(self, issue: pd.Timestamp, history_frames: int) -> tuple[list[pd.Timestamp], list[dict[str, Any]] | None, str]:
        issue = pd.Timestamp(issue).floor("h")
        times = [issue - pd.Timedelta(hours=history_frames - 1 - j) for j in range(history_frames)]
        hits: list[dict[str, Any]] = []
        missing = []
        for ts in times:
            key = _hour_key(ts, float(self.args.hour_round_tolerance_minutes))
            hit = self.time_lookup.get(key)
            if hit is None:
                missing.append(_ts_text(ts))
            else:
                hits.append(hit)
        if missing:
            return times, None, "|".join(missing)
        return times, hits, ""

    def clear_sky(self, times: list[pd.Timestamp]) -> np.ndarray:
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        out = []
        for ts in times:
            cosz = solar_cos_zenith_bjt(pd.Timestamp(ts), lat_grid, lon_grid)
            out.append(haurwitz_clear_sky(cosz))
        return np.stack(out, axis=0).astype(np.float32)


def load_swr_model(args: argparse.Namespace):
    import torch

    ckpt = torch.load(Path(args.swr_checkpoint), map_location="cpu")
    if "stats" not in ckpt:
        raise ValueError(
            "This private core loader only accepts a bare Ratio3DCNN checkpoint. "
            "For the Stage 3 default residual ConvLSTM correction checkpoint, run "
            "03_build_center_h8_swr_anchor_cache.py so its residual adapter is installed."
        )
    stats = ckpt["stats"]
    channels = [str(x) for x in stats["channels"]]
    width = int(ckpt["args"].get("width", 48)) if isinstance(ckpt.get("args"), dict) else 48
    depth = int(ckpt["args"].get("depth", 4)) if isinstance(ckpt.get("args"), dict) else 4
    dropout = float(ckpt["args"].get("dropout", 0.05)) if isinstance(ckpt.get("args"), dict) else 0.05
    model = Ratio3DCNN(
        in_channels=len(channels),
        future_frames=int(stats.get("future_frames", args.swr_future_anchors)),
        width=width,
        depth=depth,
        dropout=dropout,
        ratio_clip=float(stats["ratio_clip"]),
    )
    state = ckpt.get("model") or ckpt.get("model_state")
    model.load_state_dict(state)
    device = _device(args.device)
    model.to(device).eval()
    return model, stats, channels, device


def main() -> None:
    _inject_pycharm_defaults()
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    import torch

    out_dir = ensure_dir(Path(args.out_dir))
    if any((out_dir / name).exists() for name in ["swr_anchor_64.npy", "h8_history_features_64.npy"]) and not int(args.force):
        raise FileExistsError(f"Cache already exists under {out_dir}. Pass --force 1 to overwrite.")

    station_split = f04._load_station_split(args)
    bounds = _bounds(station_split)
    pv_lon, pv_lat = _grid_lonlat(bounds, int(args.grid_size))

    time_df = pd.read_csv(Path(args.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if int(args.max_samples) > 0 and len(time_df) > int(args.max_samples):
        time_df = time_df.sample(n=int(args.max_samples), random_state=int(args.seed)).sort_values("issue_time").reset_index(drop=True)

    model, stats, channels, device = load_swr_model(args)
    reader = CenterH8Reader(args, channels)
    swr_pos = channels.index("SWR")
    channel_idx = list(range(len(channels)))
    x_low = np.asarray(stats["x_clip_low"], dtype=np.float32).reshape(1, -1, 1, 1)
    x_high = np.asarray(stats["x_clip_high"], dtype=np.float32).reshape(1, -1, 1, 1)
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    x_std = np.asarray(stats["x_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    min_clear_sky = float(stats["min_clear_sky"])
    ratio_clip = float(stats["ratio_clip"])
    history_frames = int(stats.get("history_frames", args.swr_history_frames))
    future_frames = int(stats.get("future_frames", args.swr_future_anchors))
    anchor_count = int(args.swr_history_frames) + int(args.swr_future_anchors)
    dtype = np.float16 if str(args.cache_dtype).lower() == "float16" else np.float32

    valid_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    print(f"[center_h8_cache] scanning PV issue_time rows: {len(time_df)}", flush=True)
    for i, row in time_df.iterrows():
        issue = pd.Timestamp(row["issue_time"])
        x_times, hits, missing = reader.history_hits(issue, history_frames)
        if hits is None:
            if str(args.missing_history_policy).lower() == "fail":
                raise KeyError(f"Missing center H8 history for PV issue_time={_ts_text(issue)}: {missing}")
            missing_rows.append(
                {
                    "original_dataset_idx": int(i),
                    "issue_time": _ts_text(issue),
                    "time_split": str(row.get("time_split", "")),
                    "missing_history_labels": missing,
                }
            )
            continue
        item = row.to_dict()
        item["original_dataset_idx"] = int(i)
        item["_x_times"] = x_times
        item["_hits"] = hits
        valid_rows.append(item)
    if not valid_rows:
        raise RuntimeError("No valid PV issue_time rows with complete center H8 history.")

    n = len(valid_rows)
    h8_feat_path = out_dir / "h8_history_features_64.npy"
    anchor_path = out_dir / "swr_anchor_64.npy"
    clear_path = out_dir / "clear_sky_64.npy"
    pred_path = out_dir / "future_swr_ratio_pred_64.npy"
    h8_feat = np.lib.format.open_memmap(h8_feat_path, mode="w+", dtype=dtype, shape=(n, history_frames, len(channels), int(args.grid_size), int(args.grid_size)))
    anchor = np.lib.format.open_memmap(anchor_path, mode="w+", dtype=dtype, shape=(n, anchor_count, 2, int(args.grid_size), int(args.grid_size)))
    clear64 = np.lib.format.open_memmap(clear_path, mode="w+", dtype=dtype, shape=(n, anchor_count, int(args.grid_size), int(args.grid_size)))
    pred64 = np.lib.format.open_memmap(pred_path, mode="w+", dtype=dtype, shape=(n, int(args.swr_future_anchors), int(args.grid_size), int(args.grid_size)))

    meta_rows = []
    for out_i, row in enumerate(valid_rows):
        issue = pd.Timestamp(row["issue_time"])
        x_times = [pd.Timestamp(t) for t in row["_x_times"]]
        y_times = [issue.floor("h") + pd.Timedelta(hours=j + 1) for j in range(future_frames)]
        hits = row["_hits"]
        x_raw = np.stack([reader.read_frame(hit) for hit in hits], axis=0).astype(np.float32)
        clear = reader.clear_sky(x_times + y_times)
        feat = make_features(x_raw, clear[:history_frames], channel_idx, swr_pos, min_clear_sky, ratio_clip)
        x = np.clip(feat, x_low, x_high)
        x = (x - x_mean) / np.maximum(x_std, 1e-6)
        if hasattr(model, "set_future_clear"):
            model.set_future_clear(clear[history_frames:history_frames + future_frames])
        with torch.no_grad():
            pred_ratio = model(torch.from_numpy(x[None]).to(device)).detach().cpu().numpy()[0].astype(np.float32)
        pred_ratio = np.clip(pred_ratio[: int(args.swr_future_anchors)], 0.0, ratio_clip)

        feat64 = _resample_to_grid(feat, reader.lon, reader.lat, pv_lon, pv_lat)
        past_ratio64 = feat64[:, swr_pos]
        pred_ratio64 = _resample_to_grid(pred_ratio, reader.lon, reader.lat, pv_lon, pv_lat)
        clear64_item = _resample_to_grid(clear[:anchor_count], reader.lon, reader.lat, pv_lon, pv_lat) / 1000.0

        h8_feat[out_i] = feat64.astype(dtype)
        pred64[out_i] = pred_ratio64.astype(dtype)
        clear64[out_i] = clear64_item.astype(dtype)
        anchor[out_i] = np.stack(
            [
                np.concatenate([past_ratio64, pred_ratio64], axis=0),
                clear64_item,
            ],
            axis=1,
        ).astype(dtype)

        meta = {
            "cache_idx": int(out_i),
            "original_dataset_idx": int(row["original_dataset_idx"]),
            "issue_time": _ts_text(issue),
            "time_split": str(row.get("time_split", "")),
            "history_label_times": "|".join(_ts_text(t) for t in x_times),
            "history_raw_h8_label_times": "|".join(str(hit.get("raw_label_time", "")) for hit in hits),
            "future_anchor_label_times": "|".join(_ts_text(t) for t in y_times[: int(args.swr_future_anchors)]),
        }
        meta_rows.append(meta)
        if (out_i + 1) % int(args.log_interval) == 0 or out_i + 1 == n:
            print(f"[center_h8_cache] cached valid {out_i + 1}/{n} | total scanned={len(time_df)}", flush=True)

    h8_feat.flush()
    anchor.flush()
    clear64.flush()
    pred64.flush()
    meta_df = pd.DataFrame(meta_rows)
    missing_df = pd.DataFrame(missing_rows)
    meta_df.to_csv(out_dir / "center_h8_swr_anchor_input_cache_meta.csv", index=False)
    missing_df.to_csv(out_dir / "center_h8_swr_anchor_input_cache_missing.csv", index=False)
    time_df.to_csv(out_dir / "source_pv_issue_time_index.csv", index=False)
    write_json(
        out_dir / "center_h8_swr_anchor_input_cache_meta.json",
        {
            "purpose": "PV-issue-time aligned BigScale center H8/SWR-anchor input cache.",
            "important_scope_note": "Uses the Stage 1 center dataset lat 26-34 lon 116-126.5, not the big-patch crop index as runtime source.",
            "h8_center_processed_root": str(args.h8_center_processed_root),
            "h8_center_frame_cache_dir": str(args.h8_center_frame_cache_dir),
            "h8_center_index_csv": str(args.h8_center_index_csv),
            "swr_checkpoint": str(args.swr_checkpoint),
            "channels": channels,
            "swr_channel_index": int(swr_pos),
            "history_frames": int(history_frames),
            "future_swr_anchors": int(args.swr_future_anchors),
            "grid_size": int(args.grid_size),
            "dtype": str(dtype),
            "num_source_issue_rows": int(len(time_df)),
            "num_valid_rows": int(len(meta_df)),
            "num_missing_rows": int(len(missing_df)),
            "arrays": {
                "h8_history_features_64": "shape [N, 4, C, 64, 64]. SWR channel is SWR/clear-sky ratio after make_features; other channels follow SWR checkpoint input features before z-score normalization.",
                "swr_anchor_64": "shape [N, 7, 2, 64, 64]. channel 0 is SWR ratio, channel 1 is clear-sky SWR in kW/m2. First 4 anchors are historical observed SWR ratio; last 3 are checkpoint-predicted future SWR ratio.",
                "clear_sky_64": "shape [N, 7, 64, 64]. clear-sky SWR in kW/m2.",
                "future_swr_ratio_pred_64": "shape [N, 3, 64, 64]. checkpoint-predicted future SWR ratio.",
            },
        },
    )
    print("==== Center H8/SWR anchor input cache summary ====")
    print(f"source issue rows: {len(time_df)}")
    print(f"valid cached rows: {len(meta_df)}")
    print(f"missing rows: {len(missing_df)}")
    if len(missing_df):
        print("missing by split:")
        print(missing_df["time_split"].value_counts().to_string())
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
