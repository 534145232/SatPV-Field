#!/usr/bin/env python3
"""Generate Stage 4 framework assets for target-station locations and 5x5 field patches.

Outputs:
1. A real PV-field background with the 50 selected target stations overlaid.
2. A few Stage-4-style 5x5 patch demos centered on real target stations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
STAGE4_DIR = PIPELINE_DIR / "stage_4"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from generate_stage3_multisource_input_assets import (  # noqa: E402
    PALETTES,
    field_to_image,
    load_multisource_cache,
    make_pv_cache,
    make_stack,
    robust_limits,
)
from stage4_paths import (  # noqa: E402
    BEST_STAGE3_FIELD_CHECKPOINT,
    BEST_STAGE3_FIELD_MODEL_NAME,
    BEST_STAGE3_FIELD_RUN_DIR,
    CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
    MULTISOURCE_CACHE_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
)
from train_fullfield_dense_temporal_forecast import _parse_time_list  # noqa: E402
from visualize_regional_split_frames import _bounds, _xy_to_ij  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage4_station_patch_framework_assets"
DEFAULT_SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE4_FIELD_PATCH_MODULE = _load_local_module(
    "framework_stage4_field_patch_module",
    STAGE4_DIR / "05l_train_per_station_temporal_with_stage3_field_patch.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--multisource_cache_dir", type=Path, default=MULTISOURCE_CACHE_DIR)
    parser.add_argument("--pv_frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--sampled_issue_times_csv", type=Path, default=DEFAULT_SAMPLED_ISSUE_TIMES_CSV)
    parser.add_argument("--issue_time_column", type=str, default="issue_time")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--field_model_dir", type=Path, default=BEST_STAGE3_FIELD_RUN_DIR)
    parser.add_argument("--field_model_name", type=str, default=BEST_STAGE3_FIELD_MODEL_NAME)
    parser.add_argument("--field_checkpoint", type=Path, default=BEST_STAGE3_FIELD_CHECKPOINT)
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=CENTER_H8_SWR_ANCHOR_CACHE_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--gaussian_sigma_cells", type=float, default=1.0)
    parser.add_argument("--gaussian_radius_cells", type=int, default=2)
    parser.add_argument("--min_grid_weight", type=float, default=1.0e-6)
    parser.add_argument("--image_size", type=int, default=720)
    parser.add_argument("--patch_crop_size", type=int, default=320)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--future_stack_tile", type=int, default=320)
    parser.add_argument("--future_stack_offset", type=int, default=18)
    parser.add_argument("--overview_sample_rank", type=int, default=1)
    parser.add_argument("--patch_demo_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--station_marker_size", type=float, default=36.0)
    parser.add_argument("--highlight_marker_size", type=float, default=44.0)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def timestamp_key(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def resolve_target_csv(selection_dir: Path) -> Path:
    path = Path(selection_dir) / "target_stations.csv"
    if not path.exists():
        raise FileNotFoundError(f"target_stations.csv not found: {path}")
    return path


def resolve_source_split_csv(target_run_dir: Path, selection_dir: Path) -> Path:
    candidates = [
        Path(target_run_dir) / "target_excluded_station_split.csv",
        Path(selection_dir) / "target_excluded_station_split.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find target_excluded_station_split.csv in: {candidates}")


def read_station_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_sn": str})
    missing = {"station_sn", "lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["station_sn", "lon", "lat"]).reset_index(drop=True)
    return df


def load_requested_rows(sampled_issue_times_csv: Path, issue_time_column: str, full_time_df: pd.DataFrame) -> pd.DataFrame:
    if not Path(sampled_issue_times_csv).exists():
        raise FileNotFoundError(f"sampled issue CSV not found: {sampled_issue_times_csv}")
    req = pd.read_csv(sampled_issue_times_csv)
    if str(issue_time_column) not in req.columns:
        raise KeyError(
            f"Column {issue_time_column!r} not found in {sampled_issue_times_csv}. "
            f"Available columns: {list(req.columns)}"
        )
    if "issue_sec" not in full_time_df.columns:
        full_time_df = full_time_df.copy()
        full_time_df["issue_sec"] = pd.to_datetime(full_time_df["issue_time"]).map(timestamp_key)
    by_sec = {int(sec): i for i, sec in enumerate(full_time_df["issue_sec"].astype(np.int64).tolist())}
    rows = []
    missing = []
    for _, row in req.iterrows():
        ts = pd.Timestamp(row[str(issue_time_column)])
        sec = int(ts.timestamp())
        idx = by_sec.get(sec)
        if idx is None:
            missing.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
            continue
        rows.append(full_time_df.iloc[int(idx)].copy())
    if missing:
        raise ValueError(
            f"Could not match requested issue timestamps to multisource issue table. Examples: {', '.join(missing[:5])}"
        )
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def choose_representative_stations(targets: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or targets.empty:
        return targets.iloc[0:0].copy()
    coords = targets[["lon", "lat"]].to_numpy(np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    scale = np.where((hi - lo) > 1.0e-9, hi - lo, 1.0)
    norm = (coords - lo) / scale
    centroid = norm.mean(axis=0)
    chosen = [int(np.argmin(np.sum((norm - centroid[None]) ** 2, axis=1)))]
    while len(chosen) < min(int(count), len(targets)):
        remaining = [i for i in range(len(targets)) if i not in chosen]
        d2 = []
        for idx in remaining:
            dist = min(float(np.sum((norm[idx] - norm[c]) ** 2)) for c in chosen)
            d2.append((dist, idx))
        d2.sort(reverse=True)
        chosen.append(int(d2[0][1]))
    return targets.iloc[chosen].reset_index(drop=True)


def extract_patch(arr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    radius = int(size) // 2
    arr = np.asarray(arr, dtype=np.float32)
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="constant", constant_values=0.0)
    py = int(cy) + radius
    px = int(cx) + radius
    return padded[py - radius : py + radius + 1, px - radius : px + radius + 1].astype(np.float32)


def render_overlay_image(
    field: np.ndarray,
    mask: np.ndarray | None,
    palette: np.ndarray,
    vmin: float,
    vmax: float,
    size: int,
    station_cells: list[tuple[int, int]] | None = None,
    patch_center: tuple[int, int] | None = None,
    patch_size: int | None = None,
    point_size: float = 16.0,
    highlight_size: float = 44.0,
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    arr = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=np.float32) > 0.5
    show = np.ma.masked_where(~valid, arr)

    dpi = 100
    fig = plt.figure(figsize=(float(size) / dpi, float(size) / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    cmap = ListedColormap(np.asarray(palette, dtype=np.float32) / 255.0).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.imshow(show, origin="lower", cmap=cmap, vmin=float(vmin), vmax=float(vmax), interpolation="nearest", aspect="auto")

    if station_cells:
        xs = [int(cx) for _, cx in station_cells]
        ys = [int(cy) for cy, _ in station_cells]
        ax.scatter(
            xs,
            ys,
            s=float(point_size) * 1.22,
            c="#7f1d1d",
            alpha=0.92,
            edgecolors="#f8fafc",
            linewidths=1.25,
            zorder=3,
        )
        ax.scatter(
            xs,
            ys,
            s=float(point_size),
            c="#7f1d1d",
            alpha=0.96,
            edgecolors="#111827",
            linewidths=0.9,
            zorder=4,
        )

    if patch_center is not None:
        cy, cx = int(patch_center[0]), int(patch_center[1])
        ax.scatter([cx], [cy], s=float(highlight_size), c="#f97316", edgecolors="#fff7ed", linewidths=0.9, zorder=6)
        if patch_size is not None:
            radius = int(patch_size) // 2
            x0 = cx - radius - 0.5
            y0 = cy - radius - 0.5
            ax.add_patch(Rectangle((x0, y0), int(patch_size), int(patch_size), fill=False, edgecolor="#ffffff", linewidth=2.2, zorder=5))
            ax.add_patch(Rectangle((x0, y0), int(patch_size), int(patch_size), fill=False, edgecolor="#0f172a", linewidth=0.9, zorder=6))

    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def add_border(image: Image.Image, color: tuple[int, int, int, int] = (15, 23, 42, 220), width: int = 3) -> Image.Image:
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=color, width=int(width))
    return out


def make_demo_panel(full_image: Image.Image, patch_image: Image.Image, gap: int = 28) -> Image.Image:
    height = max(full_image.height, patch_image.height)
    width = full_image.width + int(gap) + patch_image.width
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    canvas.alpha_composite(full_image, (0, (height - full_image.height) // 2))
    canvas.alpha_composite(patch_image, (full_image.width + int(gap), (height - patch_image.height) // 2))
    return canvas


def save_image(path: Path, image: Image.Image) -> str:
    ensure_dir(path.parent)
    image.save(path, format="PNG", optimize=True)
    return str(path)


def build_stage4_field_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = dict(STAGE4_FIELD_PATCH_MODULE.PYCHARM_DEFAULTS)
    overrides = {
        "selection_dir": str(args.selection_dir),
        "target_run_dir": str(args.target_run_dir),
        "frame_cache_dir": str(args.pv_frame_cache_dir),
        "multisource_cache_dir": str(args.multisource_cache_dir),
        "field_model_dir": str(args.field_model_dir),
        "field_model_name": str(args.field_model_name),
        "field_checkpoint": str(args.field_checkpoint),
        "swr_anchor_cache_dir": str(args.swr_anchor_cache_dir),
        "grid_size": int(args.grid_size),
        "gaussian_sigma_cells": float(args.gaussian_sigma_cells),
        "gaussian_radius_cells": int(args.gaussian_radius_cells),
        "min_grid_weight": float(args.min_grid_weight),
        "field_patch_size": int(args.patch_size),
        "device": str(args.device),
        "out_dir": str(args.out_dir),
    }
    defaults.update(overrides)
    cooked: dict[str, object] = {}
    for key, value in defaults.items():
        if key.endswith("_dir") or key.endswith("_csv") or key in {"train_pv_dir", "val_pv_dir", "field_checkpoint"}:
            cooked[key] = Path(value) if str(value).strip() else Path(".")
        else:
            cooked[key] = value
    runtime = argparse.Namespace(**cooked)
    STAGE4_FIELD_PATCH_MODULE.resolve_stage3_swr_anchor_cache_arg(runtime)
    return runtime


def predict_future_fields_and_station_patches(
    overview_row: pd.Series,
    station_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch

    runtime = build_stage4_field_runtime_args(args)
    one_row_df = pd.DataFrame([overview_row]).copy().reset_index(drop=True)
    if "issue_sec" not in one_row_df.columns:
        one_row_df["issue_sec"] = pd.to_datetime(one_row_df["issue_time"]).map(timestamp_key)

    device = "cuda" if str(runtime.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    patch_provider = STAGE4_FIELD_PATCH_MODULE.PredictedFieldPatchProvider(station_rows, one_row_df, runtime, device)

    ckpt_arg = str(getattr(runtime, "field_checkpoint", "")).strip()
    ckpt_path = Path(ckpt_arg) if ckpt_arg else Path(runtime.field_model_dir) / f"{runtime.field_model_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    field_args = patch_provider._field_args()
    split_df = STAGE4_FIELD_PATCH_MODULE.f04._load_station_split(field_args)
    bounds = STAGE4_FIELD_PATCH_MODULE._bounds(split_df)
    ms_cache = STAGE4_FIELD_PATCH_MODULE.f04._load_multisource_cache(Path(runtime.multisource_cache_dir))
    pv_cache = STAGE4_FIELD_PATCH_MODULE.f04.ReadOnlyDiskFrameCache(Path(runtime.frame_cache_dir), one_row_df, field_args)
    pv_stats = ckpt.get("pv_field_norm_stats") or STAGE4_FIELD_PATCH_MODULE._safe_read_json(Path(runtime.field_model_dir) / "pv_field_norm_stats.json")
    ms_stats = ckpt.get("multisource_norm_stats") or STAGE4_FIELD_PATCH_MODULE._safe_read_json(Path(runtime.field_model_dir) / "multisource_norm_stats.json")
    swr04l = STAGE4_FIELD_PATCH_MODULE._load_swr04l()
    base_dataset = STAGE4_FIELD_PATCH_MODULE.f04.TargetExcludedPVNormFieldDataset(
        one_row_df,
        pv_cache,
        ms_cache,
        bounds,
        pv_stats,
        ms_stats,
        str(runtime.base_model_name),
        field_args,
    )
    swr_arr_all, swr_meta, _swr_info = swr04l._load_prebuilt_swr_anchor_cache(Path(runtime.swr_anchor_cache_dir))
    swr_arr = patch_provider._swr_anchor_array_for_time_df(swr_arr_all, swr_meta)
    swr_stats = ckpt.get("swr_anchor_norm_stats")
    if swr_stats is None:
        stats_path = Path(runtime.field_model_dir) / "swr_anchor_norm_stats.json"
        if stats_path.exists():
            swr_stats = STAGE4_FIELD_PATCH_MODULE._safe_read_json(stats_path)
        else:
            swr_stats = swr04l._compute_swr_anchor_stats(swr_arr, Path(runtime.field_model_dir), force=False)
    dataset = swr04l.SWRAnchorPVFieldDataset(base_dataset, swr_arr, field_args, swr_stats)
    model = patch_provider._load_field_model(ckpt)

    x, future_feat, swr_anchor, _y, _m = dataset[0]
    with torch.no_grad():
        pred_z = model(
            torch.from_numpy(x[None]).to(device),
            torch.from_numpy(future_feat[None]).to(device),
            torch.from_numpy(swr_anchor[None]).to(device),
        ).detach().cpu().numpy()
    future_fields = STAGE4_FIELD_PATCH_MODULE.f04._denorm_field(pred_z[0], pv_stats).astype(np.float32)
    latest_mask = np.asarray(x[1, -1], dtype=np.float32)
    future_masks = np.repeat(latest_mask[None], future_fields.shape[0], axis=0).astype(np.float32)

    patch_raw = {
        str(sn): np.asarray(patch_provider.patch_raw[str(sn)][0], dtype=np.float32)
        for sn in station_rows["station_sn"].astype(str)
    }
    patch_mask = {
        str(sn): np.asarray(patch_provider.patch_mask[str(sn)][0], dtype=np.float32)
        for sn in station_rows["station_sn"].astype(str)
    }
    return future_fields, future_masks, patch_raw, patch_mask


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    ms_cache = load_multisource_cache(Path(args.multisource_cache_dir))
    full_time_df = ms_cache["time_df"].copy()
    if "issue_sec" not in full_time_df.columns:
        full_time_df["issue_sec"] = pd.to_datetime(full_time_df["issue_time"]).map(timestamp_key)
    requested_rows = load_requested_rows(Path(args.sampled_issue_times_csv), str(args.issue_time_column), full_time_df)
    if requested_rows.empty:
        raise ValueError("No requested issue times were matched.")

    target_csv = resolve_target_csv(Path(args.selection_dir))
    source_split_csv = resolve_source_split_csv(Path(args.target_run_dir), Path(args.selection_dir))
    targets = read_station_table(target_csv)
    source_split = read_station_table(source_split_csv)
    bounds = _bounds(source_split)
    target_cells = [
        _xy_to_ij(float(row["lon"]), float(row["lat"]), bounds, int(args.grid_size))
        for _, row in targets.iterrows()
    ]

    pv_cache = make_pv_cache(Path(args.pv_frame_cache_dir), full_time_df, args)
    issue_to_idx = {int(sec): i for i, sec in enumerate(full_time_df["issue_sec"].astype(np.int64).tolist())}

    overview_rank = max(1, min(int(args.overview_sample_rank), len(requested_rows)))
    overview_row = requested_rows.iloc[overview_rank - 1]
    overview_issue = pd.Timestamp(overview_row["issue_time"])
    overview_past_times = _parse_time_list(overview_row["past_times"])
    latest_field, latest_mask = pv_cache.get(overview_past_times[-1])
    pv_lo, pv_hi = 0.0, max(0.3, min(1.5, robust_limits([latest_field], 0.01, 0.99)[1]))

    overview_image = render_overlay_image(
        field=latest_field,
        mask=latest_mask,
        palette=PALETTES["pv"],
        vmin=pv_lo,
        vmax=pv_hi,
        size=int(args.image_size),
        station_cells=target_cells,
        point_size=float(args.station_marker_size),
    )
    overview_dir = ensure_dir(out_dir / "target_station_overview")
    overview_path = save_image(
        overview_dir / f"sample_{overview_rank:02d}_{overview_issue:%Y%m%d_%H%M%S}_pv_with_50_target_stations.png",
        overview_image,
    )
    overview_meta = {
        "issue_time": str(overview_issue),
        "sample_rank": int(overview_rank),
        "target_station_count": int(len(targets)),
        "target_csv": str(target_csv),
        "source_split_csv": str(source_split_csv),
        "image_path": overview_path,
        "field_source": "latest historical target-excluded PV_norm field frame (past_times[-1])",
        "grid_mapping": "Stage 4 _xy_to_ij over Stage 3 _bounds(source station split)",
    }
    write_json(overview_dir / "overview_meta.json", overview_meta)

    demo_station_rows = choose_representative_stations(targets, int(args.patch_demo_count))
    future_fields, future_masks, future_patch_raw, future_patch_mask = predict_future_fields_and_station_patches(
        overview_row,
        demo_station_rows,
        args,
    )
    future_lo, future_hi = 0.0, max(0.3, min(1.5, robust_limits(list(future_fields), 0.01, 0.99)[1]))
    future_dir = ensure_dir(out_dir / "future_pv_field")
    future_images = []
    future_frame_paths = []
    for h in range(future_fields.shape[0]):
        image = field_to_image(
            future_fields[h],
            PALETTES["pv"],
            future_lo,
            future_hi,
            int(args.image_size),
            mask=future_masks[h],
            interpolation="nearest",
        )
        future_images.append(image)
        future_frame_paths.append(save_image(future_dir / f"future_pv_h{h + 1:02d}.png", image))
    future_stack = make_stack(future_images, int(args.future_stack_tile), int(args.future_stack_offset))
    future_stack_path = save_image(future_dir / "future_pv_8frame_stack.png", future_stack)
    write_json(
        future_dir / "future_pv_meta.json",
        {
            "issue_time": str(overview_issue),
            "sample_rank": int(overview_rank),
            "frame_paths": future_frame_paths,
            "stack_path": future_stack_path,
            "field_source": "best Stage 3 future PV_norm field predicted by the Stage 4 05l reference pipeline",
        },
    )

    patch_demo_dir = ensure_dir(out_dir / "patch_demos")
    demo_records = []
    demo_count = min(int(args.patch_demo_count), len(demo_station_rows))
    for demo_idx in range(demo_count):
        issue = overview_issue
        field = latest_field
        mask = latest_mask
        station_row = demo_station_rows.iloc[demo_idx]
        sn = str(station_row["station_sn"])
        cy, cx = _xy_to_ij(float(station_row["lon"]), float(station_row["lat"]), bounds, int(args.grid_size))
        patch = extract_patch(field, cy, cx, int(args.patch_size))
        mask_patch = extract_patch(mask, cy, cx, int(args.patch_size))

        field_image = render_overlay_image(
            field=field,
            mask=mask,
            palette=PALETTES["pv"],
            vmin=pv_lo,
            vmax=pv_hi,
            size=int(args.image_size),
            patch_center=(cy, cx),
            patch_size=int(args.patch_size),
            highlight_size=float(args.highlight_marker_size),
        )
        patch_image = field_to_image(
            patch,
            PALETTES["pv"],
            pv_lo,
            pv_hi,
            int(args.patch_crop_size),
            mask=mask_patch,
            interpolation="nearest",
        )
        patch_image = add_border(patch_image, width=3)
        panel = make_demo_panel(field_image, patch_image)

        demo_name = f"demo_{demo_idx + 1:02d}_{issue:%Y%m%d_%H%M%S}_{sn}"
        demo_subdir = ensure_dir(patch_demo_dir / demo_name)
        full_path = save_image(demo_subdir / "full_field_with_patch_box.png", field_image)
        patch_path = save_image(demo_subdir / "patch_5x5_crop.png", patch_image)
        panel_path = save_image(demo_subdir / "patch_demo_panel.png", panel)
        future_station_dir = ensure_dir(demo_subdir / "future_patch_8frames")
        future_station_images = []
        future_station_paths = []
        station_future_raw = future_patch_raw[sn]
        station_future_mask = future_patch_mask[sn]
        for h in range(station_future_raw.shape[0]):
            img = field_to_image(
                station_future_raw[h],
                PALETTES["pv"],
                future_lo,
                future_hi,
                int(args.patch_crop_size),
                mask=station_future_mask[h],
                interpolation="nearest",
            )
            img = add_border(img, width=3)
            future_station_images.append(img)
            future_station_paths.append(save_image(future_station_dir / f"patch_future_h{h + 1:02d}.png", img))
        future_station_stack = make_stack(
            future_station_images,
            int(max(96, args.patch_crop_size // 2)),
            int(max(10, args.future_stack_offset)),
        )
        future_station_stack_path = save_image(future_station_dir / "patch_future_8frame_stack.png", future_station_stack)
        record = {
            "demo_index": int(demo_idx + 1),
            "issue_time": str(issue),
            "station_sn": sn,
            "station_lon": float(station_row["lon"]),
            "station_lat": float(station_row["lat"]),
            "grid_cell_ij": [int(cy), int(cx)],
            "patch_size": int(args.patch_size),
            "shares_overview_frame": True,
            "overview_sample_rank": int(overview_rank),
            "full_field_path": full_path,
            "patch_crop_path": patch_path,
            "panel_path": panel_path,
            "future_patch_frame_paths": future_station_paths,
            "future_patch_stack_path": future_station_stack_path,
            "field_source": "latest historical target-excluded PV_norm field frame (past_times[-1])",
            "patch_extraction": "Stage 4 style zero-padded 5x5 extract centered by _xy_to_ij(lon, lat, bounds, grid_size)",
            "future_patch_source": "best Stage 3 future PV_norm field predicted by the Stage 4 05l reference pipeline",
        }
        write_json(demo_subdir / "demo_meta.json", record)
        demo_records.append(record)

    write_json(
        out_dir / "asset_index.json",
        {
            "purpose": "Framework assets for Stage 4 target-station location overlay and 5x5 patch demonstrations.",
            "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
            "overview": overview_meta,
            "future_pv_field_stack": future_stack_path,
            "patch_demo_sampling_note": "All patch demos are extracted from the same PV field frame used by the overview image; only station centers differ.",
            "patch_demos": demo_records,
        },
    )

    print("==== Stage 4 station/patch framework assets complete ====")
    print(f"overview: {overview_path}")
    print(f"future field stack: {future_stack_path}")
    print(f"patch demos: {len(demo_records)}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
