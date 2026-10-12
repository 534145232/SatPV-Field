#!/usr/bin/env python3
"""Generate Stage 3 15-minute future SWR-anchor visual assets.

This reads the prebuilt Stage 3 SWR-anchor cache and applies the same fixed
8x4 temporal-anchor weights used by the motion-fusion PV-field model.
"""

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

from stage3_paths import CENTER_H8_SWR_ANCHOR_CACHE_ROOT  # noqa: E402
from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS  # noqa: E402
from stage3_swr_anchor_cache import resolve_active_cache_dir  # noqa: E402


DEFAULT_SAMPLES_CSV = SCRIPT_DIR / "stage3_multisource_input_assets" / "sampled_issue_times.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_15min_swr_anchor_assets"
COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled_issue_times_csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument(
        "--swr_anchor_cache_dir",
        type=Path,
        default=None,
        help="Stage 3 SWR-anchor cache dir. Defaults to active_cache.json under data/05_center_h8_swr_anchor_cache.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--strip_size", type=int, default=260)
    parser.add_argument("--strip_gap", type=int, default=14)
    parser.add_argument("--stack_size", type=int, default=240)
    parser.add_argument("--stack_offset", type=int, default=18)
    parser.add_argument("--preview_labels", type=int, default=0)
    parser.add_argument("--swr_history_frames", type=int, default=4)
    parser.add_argument("--swr_future_anchors", type=int, default=3)
    parser.add_argument("--future_frames_15min", type=int, default=8)
    parser.add_argument(
        "--value_mode",
        choices=["wm2", "ratio", "clear_sky_wm2"],
        default="wm2",
        help="Physical field to render after applying Stage 3 temporal weights.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def timestamp_key(value: object) -> int:
    return int(pd.Timestamp(value).timestamp())


def split_times(text: object) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def fixed_stage3_temporal_weights(n_future: int = 8, n_anchor: int = 4) -> np.ndarray:
    weights = np.zeros((int(n_future), int(n_anchor)), dtype=np.float32)
    if int(n_future) == int(STAGE3_FUTURE_ANCHOR_WEIGHTS.shape[0]) and int(n_anchor) >= int(STAGE3_FUTURE_ANCHOR_WEIGHTS.shape[1]):
        weights[:, : STAGE3_FUTURE_ANCHOR_WEIGHTS.shape[1]] = np.asarray(STAGE3_FUTURE_ANCHOR_WEIGHTS, dtype=np.float32)
        return weights

    anchor_pos = np.arange(int(n_anchor), dtype=np.float32)
    del anchor_pos
    for h in range(int(n_future)):
        pos = h / max(int(n_future) - 1, 1) * max(int(n_anchor) - 1, 0)
        lo = int(np.floor(pos))
        hi = min(lo + 1, int(n_anchor) - 1)
        if lo == hi:
            weights[h, lo] = 1.0
        else:
            weights[h, lo] = float(hi - pos)
            weights[h, hi] = float(pos - lo)
    return weights


def shared_swr_palette() -> np.ndarray:
    stops = np.asarray([rgb for _, rgb in COLOR_STOPS], dtype=np.float32)
    x = np.asarray([pos for pos, _ in COLOR_STOPS], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.uint8)


SHARED_SWR_PALETTE = shared_swr_palette()


def robust_limits(arrays: list[np.ndarray], qlo: float = 0.01, qhi: float = 0.99) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float32)
        vals.append(a[np.isfinite(a)])
    finite = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.asarray([])
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanquantile(finite, qlo))
    hi = float(np.nanquantile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite) + 1e-6)
    return lo, hi


def field_to_image(field: np.ndarray, vmin: float, vmax: float, size: int) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    arr = np.asarray(field, dtype=np.float32)
    show = np.ma.masked_where(~np.isfinite(arr), arr)
    cmap = ListedColormap(np.asarray(SHARED_SWR_PALETTE, dtype=np.float32) / 255.0).copy()
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
        interpolation="nearest",
        aspect="auto",
    )
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def save_image(path: Path, image: Image.Image) -> str:
    ensure_dir(path.parent)
    image.save(path, format="PNG", optimize=True)
    return str(path)


def make_strip(images: list[Image.Image], size: int, gap: int, labels: list[str] | None = None) -> Image.Image:
    if not images:
        return Image.new("RGBA", (int(size), int(size)), (255, 255, 255, 0))
    label_h = 24 if labels else 0
    width = len(images) * int(size) + (len(images) - 1) * int(gap)
    canvas = Image.new("RGBA", (width, int(size) + label_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x = i * (int(size) + int(gap))
        if labels:
            draw.text((x + 4, 4), labels[i], fill=(20, 25, 35, 255))
        tile = image.convert("RGBA").resize((int(size), int(size)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile, (x, label_h))
    return canvas


def make_vertical_stack(images: list[Image.Image], size: int, offset: int) -> Image.Image:
    if not images:
        return Image.new("RGBA", (int(size), int(size)), (255, 255, 255, 0))
    width = int(size) + int(offset) * (len(images) - 1)
    height = int(size) + int(offset) * (len(images) - 1)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x = i * int(offset)
        y = i * int(offset)
        tile = image.convert("RGBA").resize((int(size), int(size)), Image.Resampling.NEAREST)
        canvas.alpha_composite(tile, (x, y))
        draw.rectangle(
            [x, y, x + int(size) - 1, y + int(size) - 1],
            outline=(215, 221, 230, 230),
            width=2,
        )
    return canvas


def resolve_cache_arg(cache_dir: Path | None) -> Path:
    if cache_dir is None:
        return resolve_active_cache_dir(required=True)
    cache_dir = Path(cache_dir)
    if (cache_dir / "swr_anchor_64.npy").exists():
        return cache_dir
    manifest = cache_dir / "active_cache.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        candidate = Path(payload["active_cache_dir"])
        if candidate.exists():
            return candidate
    return cache_dir


def load_anchor_cache(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    cache_dir = Path(cache_dir)
    anchor_path = cache_dir / "swr_anchor_64.npy"
    meta_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.csv"
    info_path = cache_dir / "center_h8_swr_anchor_input_cache_meta.json"
    if not anchor_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 SWR-anchor cache files under {cache_dir}")
    meta = pd.read_csv(meta_path)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"])
    meta["issue_sec"] = meta["issue_time"].map(timestamp_key)
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    return np.load(anchor_path, mmap_mode="r"), meta, info


def weighted_15min_maps(anchor_item: np.ndarray, weights: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[int]]:
    anchor_item = np.asarray(anchor_item, dtype=np.float32)
    # Match Stage 3 motion-fusion forward exactly:
    # A0 = past_swr[-1], A1 = future_swr[0], A2 = future_swr[1], A3 = future_swr[2]
    anchor_start = max(int(args.swr_history_frames) - 1, 0)
    anchor_end = anchor_start + weights.shape[1]
    temporal = anchor_item[anchor_start:anchor_end]
    if temporal.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected {weights.shape[1]} temporal anchors, got {temporal.shape[0]}")
    ratio = temporal[:, 0]
    clear_wm2 = temporal[:, 1] * 1000.0
    if args.value_mode == "ratio":
        maps = np.einsum("ta,ayx->tyx", weights, ratio)
    elif args.value_mode == "clear_sky_wm2":
        maps = np.einsum("ta,ayx->tyx", weights, clear_wm2)
    else:
        # For physical visualization, combine the restored W/m2 maps with the
        # same fixed temporal weights used by Stage 3's anchor mixer.
        restored = ratio * clear_wm2
        maps = np.einsum("ta,ayx->tyx", weights, restored)
    return maps.astype(np.float32), list(range(anchor_start, anchor_end))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    samples = pd.read_csv(args.sampled_issue_times_csv)
    if "issue_time" not in samples.columns:
        raise KeyError(f"sampled issue CSV must contain issue_time: {args.sampled_issue_times_csv}")
    samples["issue_time"] = pd.to_datetime(samples["issue_time"])
    samples["issue_sec"] = samples["issue_time"].map(timestamp_key)

    resolved_anchor_cache_dir = resolve_cache_arg(args.swr_anchor_cache_dir)
    anchor, meta, info = load_anchor_cache(resolved_anchor_cache_dir)
    by_issue_sec = {int(row["issue_sec"]): row for _, row in meta.iterrows()}
    weights = fixed_stage3_temporal_weights(int(args.future_frames_15min), 4)

    manifest_rows = []
    for sample_id, row in samples.reset_index(drop=True).iterrows():
        issue = pd.Timestamp(row["issue_time"])
        meta_row = by_issue_sec.get(int(row["issue_sec"]))
        if meta_row is None:
            raise KeyError(f"No SWR-anchor cache row for sampled issue_time={issue}")
        cache_idx = int(meta_row["cache_idx"])
        maps, temporal_anchor_indices = weighted_15min_maps(anchor[cache_idx], weights, args)
        vmin, vmax = (0.0, max(300.0, robust_limits([maps], 0.01, 0.99)[1])) if args.value_mode != "ratio" else robust_limits([maps], 0.01, 0.99)

        sample_dir = ensure_dir(out_dir / f"sample_{sample_id + 1:02d}_{issue:%Y%m%d_%H%M%S}")
        frame_paths = []
        images = []
        labels = []
        future_times = [issue + pd.Timedelta(minutes=15 * h) for h in range(int(args.future_frames_15min))]
        for h, field in enumerate(maps):
            image = field_to_image(field, vmin, vmax, int(args.image_size))
            images.append(image)
            labels.append(f"t+{15 * h}m")
            frame_paths.append(save_image(sample_dir / "future_swr_15min_anchors" / f"future_swr_15min_h{h + 1:02d}.png", image))
        strip = make_strip(images, int(args.strip_size), int(args.strip_gap), labels if int(args.preview_labels) else None)
        strip_path = save_image(sample_dir / "future_swr_15min_8frame_strip.png", strip)
        stack = make_vertical_stack(images, int(args.stack_size), int(args.stack_offset))
        stack_path = save_image(sample_dir / "future_swr_15min_8frame_stack_vertical.png", stack)
        meta_payload = {
            "sample_id": int(sample_id + 1),
            "issue_time": str(issue),
            "cache_idx": int(cache_idx),
            "value_mode": str(args.value_mode),
            "future_15min_times": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_times],
            "source_history_label_times": str(meta_row.get("history_label_times", "")),
            "source_future_anchor_label_times": str(meta_row.get("future_anchor_label_times", "")),
            "temporal_anchor_indices": temporal_anchor_indices,
            "temporal_anchor_labels": [
                "past_swr[-1]",
                "future_swr[0]",
                "future_swr[1]",
                "future_swr[2]",
            ],
            "fixed_stage3_weights_8x4": weights.tolist(),
            "frame_paths": frame_paths,
            "strip_path": strip_path,
            "vertical_stack_path": stack_path,
        }
        write_json(sample_dir / "future_swr_15min_anchor_meta.json", meta_payload)
        manifest_rows.append(meta_payload)
        print(f"[15min_swr] sample {sample_id + 1}/{len(samples)} issue={issue} -> {sample_dir}", flush=True)

    write_json(
        out_dir / "manifest.json",
        {
            "purpose": "Stage 3 15-minute future SWR-anchor visual assets aligned to sampled issue times.",
            "sampled_issue_times_csv": str(args.sampled_issue_times_csv),
            "swr_anchor_cache_dir": str(resolved_anchor_cache_dir),
            "cache_info": info,
            "value_mode": str(args.value_mode),
            "color_stops": [{"position": float(pos), "rgb": list(map(int, rgb))} for pos, rgb in COLOR_STOPS],
            "fixed_stage3_weights_8x4": weights.tolist(),
            "samples": manifest_rows,
        },
    )
    print("==== Stage 3 15-min SWR-anchor assets complete ====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
