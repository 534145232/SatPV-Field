#!/usr/bin/env python3
"""Plot compact source-density trend panels for the paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
STAGE3_DENSITY_DIR = STAGE4_DIR.parent / "stage_3" / "source_fleet_density"
if str(STAGE3_DENSITY_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DENSITY_DIR))

from density_common import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DENSITY_STAGE3_RUN_ROOT,
    DENSITY_STAGE4_RUN_ROOT,
    parse_ratios,
    ratio_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot source-density trend panels.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default=",".join(str(r) for r in DEFAULT_RATIOS))
    parser.add_argument("--stage3_summary", type=Path, default=None)
    parser.add_argument("--stage4_summary", type=Path, default=None)
    parser.add_argument("--out_pdf", type=Path, default=None)
    parser.add_argument("--only_separate", action="store_true", help="Save only the four separate panels.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def _pick(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
    if preferred in df.columns:
        return pd.to_numeric(df[preferred], errors="coerce")
    if fallback and fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    available = ", ".join(df.columns)
    raise KeyError(f"Missing required column {preferred!r}; available columns: {available}")


def _prepare(args: argparse.Namespace) -> pd.DataFrame:
    stage3_path = args.stage3_summary or (
        DENSITY_STAGE3_RUN_ROOT / f"seed{int(args.seed)}" / "stage3_density_summary.csv"
    )
    stage4_path = args.stage4_summary or (
        DENSITY_STAGE4_RUN_ROOT / f"seed{int(args.seed)}" / "stage4_density_station_summary.csv"
    )
    stage3 = _read(Path(stage3_path))
    stage4 = _read(Path(stage4_path))
    ratios = parse_ratios(args.ratios)
    labels = [ratio_label(r) for r in ratios]

    keep3 = [
        "source_density_ratio",
        "source_density_label",
        "total_test_mae_vs_full",
    ]
    missing3 = [c for c in keep3 if c not in stage3.columns]
    if missing3:
        raise KeyError(f"Stage3 summary misses columns: {missing3}")

    merged = stage4.merge(
        stage3[keep3],
        on=["source_density_ratio", "source_density_label"],
        how="left",
        validate="one_to_one",
    )
    merged = merged[merged["source_density_label"].astype(str).isin(labels)].copy()
    merged["_order"] = merged["source_density_label"].map({lab: i for i, lab in enumerate(labels)})
    merged = merged.sort_values("_order").reset_index(drop=True)
    if len(merged) != len(labels):
        got = sorted(merged["source_density_label"].astype(str).unique().tolist())
        raise RuntimeError(f"Expected density labels {labels}, but found {got}")

    out = pd.DataFrame(
        {
            "density_percent": [int(round(r * 100)) for r in ratios],
            "source_density_label": merged["source_density_label"].astype(str),
            "field_patch_support_mean": _pick(
                merged,
                "field_patch_support_mean",
                "mean_supported_pixel_ratio_target_patch",
            ),
            "nearest_source_distance_km_median": _pick(merged, "nearest_source_distance_km_median"),
            "total_test_mae_vs_full": _pick(merged, "total_test_mae_vs_full"),
            "pooled_station_mae": _pick(merged, "pooled_station_mae", "station_mae"),
        }
    )
    return out


def _plot_one(ax, x: np.ndarray, y: np.ndarray, title: str, ylabel: str, color: str) -> None:
    ax.plot(x, y, color=color, marker="o", linewidth=1.7, markersize=4.8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in x], fontsize=8)
    ax.tick_params(axis="y", labelsize=8, length=2.5)
    ax.tick_params(axis="x", length=2.5)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.55, alpha=0.8)
    ax.grid(True, axis="x", color="#ededed", linewidth=0.45, alpha=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")


def _set_panel_ylim(ax, y: np.ndarray) -> None:
    pad = max((float(np.nanmax(y)) - float(np.nanmin(y))) * 0.18, 1.0e-4)
    ax.set_ylim(float(np.nanmin(y)) - pad, float(np.nanmax(y)) + pad)


def main() -> None:
    args = parse_args()
    data = _prepare(args)
    if args.out_pdf is None:
        args.out_pdf = DENSITY_STAGE4_RUN_ROOT / f"seed{int(args.seed)}" / "source_density_trends.pdf"
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data["density_percent"].to_numpy(np.float64)
    panels = [
        (
            "a_support",
            data["field_patch_support_mean"].to_numpy(np.float64),
            "(a) Field-patch support",
            "Supported-pixel ratio",
            "#2878b5",
        ),
        (
            "b_distance",
            data["nearest_source_distance_km_median"].to_numpy(np.float64),
            "(b) Nearest source distance",
            "Median distance (km)",
            "#6f4aa8",
        ),
        (
            "c_field_mae",
            data["total_test_mae_vs_full"].to_numpy(np.float64),
            "(c) Field MAE vs. full reference",
            "PV-field MAE",
            "#c44e52",
        ),
        (
            "d_station_mae",
            data["pooled_station_mae"].to_numpy(np.float64),
            "(d) Station-level MAE",
            "Pooled station MAE",
            "#26896d",
        ),
    ]
    saved_paths = []
    if not args.only_separate:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.55), sharex=True)
        panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        for ax, (_stem, y, title, ylabel, color) in zip(panel_axes, panels):
            _plot_one(ax, x, y, title, ylabel, color)
            _set_panel_ylim(ax, y)
        for ax in axes[1, :]:
            ax.set_xlabel("Source-fleet density", fontsize=8.5)
        fig.tight_layout(pad=0.55, w_pad=0.9, h_pad=0.75)
        fig.savefig(out_pdf, transparent=False)
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=int(args.dpi), transparent=False)
        plt.close(fig)
        saved_paths.extend([out_pdf, out_png])

    separate_dir = out_pdf.parent / out_pdf.stem
    separate_dir.mkdir(parents=True, exist_ok=True)
    for stem, y, title, ylabel, color in panels:
        fig, ax = plt.subplots(1, 1, figsize=(3.15, 2.25))
        _plot_one(ax, x, y, title, ylabel, color)
        _set_panel_ylim(ax, y)
        ax.set_xlabel("Source-fleet density", fontsize=8.5)
        fig.tight_layout(pad=0.45)
        panel_pdf = separate_dir / f"source_density_trends_{stem}.pdf"
        panel_png = separate_dir / f"source_density_trends_{stem}.png"
        fig.savefig(panel_pdf, transparent=False)
        fig.savefig(panel_png, dpi=int(args.dpi), transparent=False)
        plt.close(fig)
        saved_paths.extend([panel_pdf, panel_png])

    print("==== Source-density trend data ====")
    print(data.to_string(index=False))
    for path in saved_paths:
        print("saved:", path)


if __name__ == "__main__":
    main()
