#!/usr/bin/env python3
"""Export Direct/Motion/Fusion branch assets from the trained Stage 3 motion-fusion model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import MOTION_FUSION_RUN_DIR  # noqa: E402
import stage3_base_pvnorm_field_model as f04  # noqa: E402
import stage3_motion_fusion_model_impl as mf  # noqa: E402
from stage3_motion_fusion import patch_stage3_temporal_anchors  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_motion_branch_assets"
DEFAULT_SAMPLED_ISSUE_TIMES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"
COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]
GATE_COLOR_STOPS = [
    (0.00, (33, 113, 181)),
    (0.35, (222, 235, 247)),
    (0.55, (255, 255, 217)),
    (0.80, (254, 204, 92)),
    (1.00, (253, 141, 60)),
]
FLOW_COLOR_STOPS = [
    (0.00, (222, 235, 247)),
    (0.25, (107, 174, 214)),
    (0.52, (255, 255, 217)),
    (0.78, (254, 204, 92)),
    (1.00, (189, 0, 38)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=MOTION_FUSION_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sampled_issue_times_csv",
        type=Path,
        default=DEFAULT_SAMPLED_ISSUE_TIMES_CSV,
        help="Optional sampled_issue_times.csv from generate_stage3_multisource_input_assets.py. When present, motion-branch samples follow the same issue timestamps and order.",
    )
    parser.add_argument(
        "--issue_time_column",
        type=str,
        default="issue_time",
        help="Timestamp column in --sampled_issue_times_csv.",
    )
    parser.add_argument("--visual_horizons", type=str, default="1,4,8")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--strip_tile", type=int, default=220)
    parser.add_argument("--stack_tile", type=int, default=220)
    parser.add_argument("--stack_offset", type=int, default=16)
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--preview_labels", type=int, default=1)
    parser.add_argument("--pv_cmap", type=str, default="shared_swr")
    parser.add_argument("--gate_cmap", type=str, default="shared_gate")
    parser.add_argument("--flow_cmap", type=str, default="shared_flow")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_loaded_args(raw_args: dict) -> argparse.Namespace:
    cooked: dict[str, object] = {}
    for key, value in raw_args.items():
        if key.endswith("_dir") or key.endswith("_csv") or key.endswith("_root") or key == "swr_checkpoint":
            cooked[key] = Path(value) if str(value).strip() else Path(".")
        elif isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                cooked[key] = 1 if lowered == "true" else 0
            else:
                try:
                    cooked[key] = int(value)
                except ValueError:
                    try:
                        cooked[key] = float(value)
                    except ValueError:
                        cooked[key] = value
        else:
            cooked[key] = value
    return argparse.Namespace(**cooked)


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    run_meta = Path(args.run_dir) / "run_meta.json"
    if run_meta.exists():
        payload = json.loads(run_meta.read_text(encoding="utf-8"))
        ckpt = payload.get("checkpoint")
        if ckpt:
            return Path(ckpt)
    return Path(args.run_dir) / "swr_anchor_motion_fusion_pv_nwp_best.pt"


def _masked_image(
    field: np.ndarray,
    vmin: float,
    vmax: float,
    size: int,
    cmap_name: str,
    mask: np.ndarray | None = None,
    interpolation: str = "nearest",
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    arr = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=np.float32) > 0.5
    show = np.ma.masked_where(~valid, arr)
    cmap_name = str(cmap_name)
    if cmap_name == "shared_swr":
        cmap = ListedColormap(_palette_from_stops(COLOR_STOPS), name="shared_swr")
    elif cmap_name == "shared_gate":
        cmap = ListedColormap(_palette_from_stops(GATE_COLOR_STOPS), name="shared_gate")
    elif cmap_name == "shared_flow":
        cmap = ListedColormap(_palette_from_stops(FLOW_COLOR_STOPS), name="shared_flow")
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    dpi = 100
    fig = plt.figure(figsize=(float(size) / dpi, float(size) / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    ax.imshow(
        show,
        origin="lower",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation=str(interpolation),
        aspect="auto",
    )
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _flow_to_magnitude(flow_xy: np.ndarray) -> np.ndarray:
    flow_xy = np.asarray(flow_xy, dtype=np.float32)
    return np.sqrt(np.square(flow_xy[0]) + np.square(flow_xy[1])).astype(np.float32)


def _palette_from_stops(color_stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    stops = np.asarray([rgb for _, rgb in color_stops], dtype=np.float32)
    x = np.asarray([pos for pos, _ in color_stops], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.float32) / 255.0


def _pick_frame(arrs: np.ndarray, horizon_idx: int, local_idx: int) -> np.ndarray:
    arrs = np.asarray(arrs)
    if arrs.ndim == 0:
        raise ValueError("Branch array must have at least one dimension.")
    if 0 <= int(horizon_idx) < int(arrs.shape[0]):
        return np.asarray(arrs[int(horizon_idx)], dtype=np.float32)
    if 0 <= int(local_idx) < int(arrs.shape[0]):
        return np.asarray(arrs[int(local_idx)], dtype=np.float32)
    raise IndexError(
        f"Cannot pick frame: horizon_idx={horizon_idx}, local_idx={local_idx}, "
        f"available={arrs.shape[0]}"
    )


def _robust_limits(arrays: list[np.ndarray], masks: list[np.ndarray] | None = None, qlo: float = 0.02, qhi: float = 0.98) -> tuple[float, float]:
    chunks = []
    if masks is None:
        masks = [None] * len(arrays)
    for arr, mask in zip(arrays, masks):
        a = np.asarray(arr, dtype=np.float32)
        valid = np.isfinite(a)
        if mask is not None:
            valid &= np.asarray(mask, dtype=np.float32) > 0.5
        vals = a[valid]
        if vals.size:
            chunks.append(vals)
    if not chunks:
        return 0.0, 1.0
    finite = np.concatenate(chunks)
    lo = float(np.nanquantile(finite, qlo))
    hi = float(np.nanquantile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite) + 1.0e-6)
    return lo, hi


def _make_strip(images: list[Image.Image], tile: int, gap: int) -> Image.Image:
    if not images:
        return Image.new("RGBA", (tile, tile), (255, 255, 255, 0))
    w = len(images) * int(tile) + (len(images) - 1) * int(gap)
    canvas = Image.new("RGBA", (w, int(tile)), (255, 255, 255, 0))
    for i, image in enumerate(images):
        tile_image = image.resize((int(tile), int(tile)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile_image, (i * (int(tile) + int(gap)), 0))
    return canvas


def _make_stack(images: list[Image.Image], tile: int, offset: int) -> Image.Image:
    if not images:
        return Image.new("RGBA", (tile, tile), (255, 255, 255, 0))
    w = int(tile) + int(offset) * (len(images) - 1)
    h = int(tile) + int(offset) * (len(images) - 1)
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x = i * int(offset)
        y = i * int(offset)
        tile_image = image.resize((int(tile), int(tile)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile_image, (x, y))
        draw.rectangle(
            [x, y, x + int(tile) - 1, y + int(tile) - 1],
            outline=(215, 221, 230, 230),
            width=2,
        )
    return canvas


def _draw_preview(sample_dir: Path, rows: list[tuple[str, Image.Image]], labels: bool) -> str:
    gap_y = 18
    label_w = 132 if labels else 0
    width = label_w + max(image.width for _, image in rows)
    height = sum(image.height for _, image in rows) + gap_y * (len(rows) - 1)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, image in rows:
        if labels:
            draw.text((4, y + max(0, image.height // 2 - 8)), title, fill=(20, 25, 35, 255))
        canvas.alpha_composite(image, (label_w, y))
        y += image.height + gap_y
    path = sample_dir / "motion_branch_overview.png"
    canvas.save(path, format="PNG", optimize=True)
    return str(path)


def _load_runtime_from_checkpoint(ckpt_path: Path, out_dir: Path, device_override: str) -> tuple[dict, argparse.Namespace]:
    import torch

    state = torch.load(ckpt_path, map_location="cpu")
    runtime = _normalize_loaded_args(state["args"])
    runtime.out_dir = Path(out_dir)
    runtime.device = str(device_override)
    runtime.swr_device = str(device_override)
    return state, runtime


def _build_dataset(runtime: argparse.Namespace, split: str, out_dir: Path):
    station_split = f04._load_station_split(runtime)
    bounds = f04._bounds(station_split)
    ms_cache = f04._load_multisource_cache(Path(runtime.multisource_cache_dir))
    f04._assert_issue_time_alignment(runtime, ms_cache["time_df"])
    time_df = pd.read_csv(Path(runtime.selection_dir) / "issue_time_index.csv")
    time_df["issue_time"] = pd.to_datetime(time_df["issue_time"])
    if "issue_sec" not in time_df.columns:
        time_df["issue_sec"] = time_df["issue_time"].map(f04._timestamp_key)

    pv_cache = f04.ReadOnlyDiskFrameCache(Path(runtime.frame_cache_dir), time_df, runtime)
    pv_stats = f04._compute_or_load_pv_field_stats(pv_cache, time_df, out_dir)
    ms_stats = f04._compute_or_load_multisource_stats(ms_cache, out_dir, time_df)

    prebuilt_swr_arr, prebuilt_swr_meta, _prebuilt_info = mf._load_prebuilt_swr_anchor_cache(Path(runtime.swr_anchor_cache_dir))
    sub, cache_indices, _meta_df = mf._subset_time_df_by_swr_meta(time_df, prebuilt_swr_meta, split)
    swr_arr = np.asarray(prebuilt_swr_arr[cache_indices], dtype=np.float32)

    base = f04.TargetExcludedPVNormFieldDataset(sub, pv_cache, ms_cache, bounds, pv_stats, ms_stats, str(runtime.base_model_name), runtime)
    swr_anchor_stats = None
    if int(getattr(runtime, "normalize_swr_anchor", 1)):
        stats_path = Path(runtime.out_dir) / "swr_anchor_norm_stats.json"
        swr_anchor_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else mf._compute_swr_anchor_stats(swr_arr, out_dir, force=False)
    dataset = mf.SWRAnchorPVFieldDataset(base, swr_arr, runtime, swr_anchor_stats)
    return dataset, pv_stats, bounds


def _build_model(state: dict, runtime: argparse.Namespace, dataset, device: str):
    import torch

    base_in, future_channels = dataset.base.source_dims()
    in_channels = int(base_in) + 2
    future_channels = int(future_channels) + 1
    model = mf.make_swr_anchor_attention_model(
        future_frames=int(runtime.future_frames),
        in_channels=in_channels,
        future_channels=future_channels,
        anchor_channels=2,
        pv_hidden=int(runtime.pv_hidden_channels),
        anchor_hidden=int(runtime.anchor_channels),
        horizon_embed_channels=int(runtime.horizon_embed_channels),
        dropout=float(runtime.dropout),
        swr_history_frames=int(runtime.swr_history_frames),
        swr_future_anchors=int(runtime.swr_future_anchors),
        max_flow_pixels=float(runtime.max_flow_pixels),
    )
    model = patch_stage3_temporal_anchors(model)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model


def _choose_dataset_indices(
    dataset_time_df: pd.DataFrame,
    sample_split: str,
    sample_count: int,
    seed: int,
    sampled_issue_times_csv: Path | None,
    issue_time_column: str,
) -> tuple[np.ndarray, dict]:
    meta: dict[str, object] = {
        "selection_mode": "random",
        "sample_split": str(sample_split),
        "sample_count": int(sample_count),
        "seed": int(seed),
    }
    csv_path = Path(sampled_issue_times_csv) if sampled_issue_times_csv is not None else None
    if csv_path is not None and csv_path.exists():
        req = pd.read_csv(csv_path)
        if str(issue_time_column) not in req.columns:
            raise KeyError(
                f"Column {issue_time_column!r} not found in {csv_path}. Available columns: {list(req.columns)}"
            )
        requested_times = pd.to_datetime(req[str(issue_time_column)]).tolist()
        issue_to_idx: dict[int, int] = {}
        for idx, value in enumerate(pd.to_datetime(dataset_time_df["issue_time"])):
            issue_to_idx[int(pd.Timestamp(value).timestamp())] = int(idx)
        picked: list[int] = []
        missing: list[str] = []
        for value in requested_times:
            key = int(pd.Timestamp(value).timestamp())
            if key in issue_to_idx:
                picked.append(issue_to_idx[key])
            else:
                missing.append(pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S"))
        if missing:
            examples = ", ".join(missing[:5])
            raise ValueError(
                "Could not match requested issue timestamps from "
                f"{csv_path} within motion-branch dataset split={sample_split}. Examples: {examples}"
            )
        meta.update(
            {
                "selection_mode": "sampled_issue_times_csv",
                "sampled_issue_times_csv": str(csv_path),
                "issue_time_column": str(issue_time_column),
                "sample_count": int(len(picked)),
            }
        )
        return np.asarray(picked, dtype=np.int64), meta

    rng = np.random.default_rng(int(seed))
    n = min(int(sample_count), len(dataset_time_df))
    if n <= 0:
        raise ValueError("No samples available for the requested split.")
    picked = np.sort(rng.choice(len(dataset_time_df), size=n, replace=False))
    return np.asarray(picked, dtype=np.int64), meta


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    ckpt_path = _resolve_checkpoint(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Motion-fusion checkpoint not found: {ckpt_path}")

    state, runtime = _load_runtime_from_checkpoint(ckpt_path, out_dir, args.device)
    dataset, pv_stats, bounds = _build_dataset(runtime, str(args.sample_split), out_dir)

    import torch

    device = "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = _build_model(state, runtime, dataset, device)
    future_frame_count = int(runtime.future_frames)
    horizons = f04._parse_visual_horizons(str(args.visual_horizons), future_frame_count)
    all_horizons = list(range(future_frame_count))
    picked, selection_meta = _choose_dataset_indices(
        dataset_time_df=dataset.time_df,
        sample_split=str(args.sample_split),
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        sampled_issue_times_csv=args.sampled_issue_times_csv,
        issue_time_column=str(args.issue_time_column),
    )
    n = int(len(picked))

    manifest = {"samples": []}
    for sample_rank, ds_idx in enumerate(picked, start=1):
        x, future_feat, swr, y, m = dataset[int(ds_idx)]
        issue = pd.Timestamp(dataset.time_df.iloc[int(ds_idx)]["issue_time"])
        sample_dir = ensure_dir(out_dir / f"sample_{sample_rank:02d}_{issue:%Y%m%d_%H%M%S}")

        with torch.no_grad():
            pred_z, _attn, _aligned_swr, warped_z, direct_z, gate_np, flow_np = model(
                torch.from_numpy(x[None]).to(device),
                torch.from_numpy(future_feat[None]).to(device),
                torch.from_numpy(swr[None]).to(device),
                return_attention=True,
            )

        fused = f04._denorm_field(pred_z.detach().cpu().numpy()[0], pv_stats)
        warped = f04._denorm_field(warped_z.detach().cpu().numpy()[0], pv_stats)
        direct = f04._denorm_field(direct_z.detach().cpu().numpy()[0], pv_stats)
        truth = f04._denorm_field(y, pv_stats)
        gate = gate_np.detach().cpu().numpy()[0]
        flow = flow_np.detach().cpu().numpy()[0]

        masks = [np.asarray(m[h], dtype=np.float32) for h in horizons]
        all_masks = [np.asarray(m[h], dtype=np.float32) for h in all_horizons]
        pv_lo, pv_hi = _robust_limits(
            [truth[h] for h in all_horizons] + [fused[h] for h in all_horizons] + [warped[h] for h in all_horizons] + [direct[h] for h in all_horizons],
            all_masks * 4,
        )
        flow_mag_all = [_flow_to_magnitude(flow[h]) for h in all_horizons]
        flow_lo, flow_hi = _robust_limits(flow_mag_all, all_masks)

        branch_specs = [
            ("direct", direct, args.pv_cmap, pv_lo, pv_hi, all_masks),
            ("motion_warped", warped, args.pv_cmap, pv_lo, pv_hi, all_masks),
            ("fusion_gate", gate, args.gate_cmap, 0.0, 1.0, all_masks),
            ("fused_output", fused, args.pv_cmap, pv_lo, pv_hi, all_masks),
            ("truth", truth, args.pv_cmap, pv_lo, pv_hi, all_masks),
            ("flow_magnitude", np.stack(flow_mag_all, axis=0), args.flow_cmap, flow_lo, flow_hi, all_masks),
        ]

        row_images: list[tuple[str, Image.Image]] = []
        sample_record = {
            "sample_rank": int(sample_rank),
            "dataset_idx": int(ds_idx),
            "issue_time": str(issue),
            "visual_horizons_1based": [int(h + 1) for h in horizons],
        }
        for name, arrs, cmap_name, lo, hi, branch_masks in branch_specs:
            all_images = []
            all_frame_paths = []
            selected_images = []
            selected_frame_paths = []
            for local_idx, h in enumerate(all_horizons):
                image = _masked_image(
                    field=_pick_frame(arrs, int(h), int(local_idx)),
                    vmin=float(lo),
                    vmax=float(hi),
                    size=int(args.image_size),
                    cmap_name=str(cmap_name),
                    mask=np.asarray(branch_masks[local_idx], dtype=np.float32),
                )
                all_images.append(image)
                frame_path = sample_dir / name / f"{name}_h{h + 1:02d}.png"
                ensure_dir(frame_path.parent)
                image.save(frame_path, format="PNG", optimize=True)
                all_frame_paths.append(str(frame_path))
                if h in horizons:
                    selected_images.append(image)
                    selected_frame_paths.append(str(frame_path))
            strip = _make_strip(selected_images, int(args.strip_tile), int(args.gap))
            strip_path = sample_dir / f"{name}_strip.png"
            strip.save(strip_path, format="PNG", optimize=True)
            stack = _make_stack(selected_images, int(args.stack_tile), int(args.stack_offset))
            stack_path = sample_dir / f"{name}_stack.png"
            stack.save(stack_path, format="PNG", optimize=True)
            row_images.append((name, strip))
            sample_record[f"{name}_frames_all"] = all_frame_paths
            sample_record[f"{name}_frames"] = selected_frame_paths
            sample_record[f"{name}_strip"] = str(strip_path)
            sample_record[f"{name}_stack"] = str(stack_path)

        sample_record["overview"] = _draw_preview(sample_dir, row_images, labels=bool(int(args.preview_labels)))
        write_json(sample_dir / "sample_meta.json", sample_record)
        manifest["samples"].append(sample_record)
        print(f"[motion_branch_assets] sample {sample_rank}/{n} issue={issue} -> {sample_dir}", flush=True)

    write_json(
        out_dir / "asset_index.json",
        {
            "purpose": "Framework assets from the trained Stage 3 motion-fusion PV-field model: direct branch, motion-warp branch, gate, fused output, truth, and flow magnitude.",
            "run_dir": str(args.run_dir),
            "checkpoint": str(ckpt_path),
            "sample_split": str(args.sample_split),
            "sample_count": int(n),
            "selection": selection_meta,
            "visual_horizons_1based": [int(h + 1) for h in horizons],
            "samples": manifest["samples"],
        },
    )
    print("==== Stage 3 motion-branch assets complete ====")
    print(f"checkpoint: {ckpt_path}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
