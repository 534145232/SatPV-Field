#!/usr/bin/env python3
"""Generate a compact Stage 3 anchor-weight bar chart asset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_motion_fusion import STAGE3_FUTURE_ANCHOR_WEIGHTS  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "stage3_anchor_weight_bar_assets"
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
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--format",
        choices=["svg", "png", "both"],
        default="svg",
        help="Output format. Default uses vector SVG.",
    )
    parser.add_argument(
        "--mode",
        choices=["all_horizons", "horizon", "sum", "mean"],
        default="all_horizons",
        help="all_horizons: render one chart per 15-min future step; horizon: one selected future step; mean/sum kept for optional diagnostics.",
    )
    parser.add_argument(
        "--horizon_index",
        type=int,
        default=0,
        help="Future horizon index used only when mode=horizon. Range: 0..7.",
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ymax", type=float, default=1.0)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pick_bar_colors() -> list[tuple[float, float, float]]:
    xs = np.asarray([p for p, _ in COLOR_STOPS], dtype=np.float32)
    rgb = np.asarray([c for _, c in COLOR_STOPS], dtype=np.float32) / 255.0
    sample_points = np.asarray([0.10, 0.32, 0.70, 0.92], dtype=np.float32)
    colors = []
    for point in sample_points:
        colors.append(tuple(float(np.interp(point, xs, rgb[:, c])) for c in range(3)))
    return colors


def compute_bar_values(mode: str, horizon_index: int) -> np.ndarray:
    weights = np.asarray(STAGE3_FUTURE_ANCHOR_WEIGHTS, dtype=np.float32)
    if mode == "mean":
        return weights.mean(axis=0)
    if mode == "sum":
        return weights.sum(axis=0)
    if not (0 <= int(horizon_index) < weights.shape[0]):
        raise ValueError(f"horizon_index must be in [0, {weights.shape[0] - 1}], got {horizon_index}")
    return weights[int(horizon_index)]


def mode_suffix(mode: str, horizon_index: int) -> str:
    if mode == "horizon":
        return f"h{int(horizon_index):02d}"
    return mode


def horizon_label(horizon_index: int) -> str:
    minutes = int(horizon_index) * 15
    if minutes == 0:
        return "T"
    return f"T+{minutes}min"


def render_bar_chart(values: np.ndarray, width: int, height: int, dpi: int, ymax: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["A0", "A1", "A2", "A3"]
    colors = pick_bar_colors()

    fig_w = float(width) / float(dpi)
    fig_h = float(height) / float(dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    x = np.arange(len(labels), dtype=np.float32)
    ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor=(0.10, 0.16, 0.25),
        linewidth=1.1,
        zorder=3,
    )

    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, float(ymax))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color((0.20, 0.24, 0.31))
    ax.spines["bottom"].set_color((0.20, 0.24, 0.31))
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", colors=(0.15, 0.18, 0.23), labelsize=18)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.20, top=0.98)
    return fig


def save_single_chart(
    out_dir: Path,
    mode: str,
    values: np.ndarray,
    width: int,
    height: int,
    dpi: int,
    ymax: float,
    out_format: str,
    horizon_index: int | None = None,
) -> dict:
    fig = render_bar_chart(values, width, height, dpi, ymax)
    if mode == "horizon" and horizon_index is not None:
        stem = f"stage3_anchor_weights_bar_h{int(horizon_index):02d}"
    else:
        stem = f"stage3_anchor_weights_bar_{mode}"
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    if out_format in {"png", "both"}:
        fig.savefig(png_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    if out_format in {"svg", "both"}:
        fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    return {
        "mode": mode,
        "horizon_index": None if horizon_index is None else int(horizon_index),
        "horizon_label": None if horizon_index is None else horizon_label(horizon_index),
        "anchor_labels": ["A0", "A1", "A2", "A3"],
        "values": [float(v) for v in values],
        "format": str(out_format),
        "png_path": str(png_path) if out_format in {"png", "both"} else None,
        "svg_path": str(svg_path) if out_format in {"svg", "both"} else None,
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    manifests = []
    if args.mode == "all_horizons":
        weights = np.asarray(STAGE3_FUTURE_ANCHOR_WEIGHTS, dtype=np.float32)
        ymax = max(float(args.ymax), float(np.max(weights)) * 1.10, 0.10)
        for horizon_index in range(weights.shape[0]):
            payload = save_single_chart(
                out_dir=out_dir,
                mode="horizon",
                values=weights[horizon_index],
                width=args.width,
                height=args.height,
                dpi=args.dpi,
                ymax=ymax,
                out_format=str(args.format),
                horizon_index=horizon_index,
            )
            manifests.append(payload)
            write_json(out_dir / f"stage3_anchor_weights_bar_h{horizon_index:02d}_meta.json", payload)
        meta = {
            "mode": args.mode,
            "chart_count": len(manifests),
            "weights_matrix": np.asarray(STAGE3_FUTURE_ANCHOR_WEIGHTS, dtype=np.float32).tolist(),
            "charts": manifests,
        }
        write_json(out_dir / "stage3_anchor_weights_bar_all_horizons_meta.json", meta)
    else:
        values = compute_bar_values(args.mode, args.horizon_index)
        ymax = max(float(args.ymax), float(np.max(values)) * 1.10, 0.10)
        payload = save_single_chart(
            out_dir=out_dir,
            mode=args.mode,
            values=values,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
            ymax=ymax,
            out_format=str(args.format),
            horizon_index=args.horizon_index if args.mode == "horizon" else None,
        )
        meta = {
            **payload,
            "weights_matrix": np.asarray(STAGE3_FUTURE_ANCHOR_WEIGHTS, dtype=np.float32).tolist(),
        }
        write_json(out_dir / f"stage3_anchor_weights_bar_{mode_suffix(args.mode, args.horizon_index)}_meta.json", meta)

    print("==== Stage 3 anchor-weight bar chart complete ====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
