#!/usr/bin/env python3
"""Generate a two-panel horizon-wise MAE figure for the paper.

Panel (a): station-level MAE across forecast leads.
Panel (b): PV-field MAE across forecast leads.

In addition to the combined two-panel figure, the script also writes two
separate no-title subfigures for compact IEEE placement.

The script reads existing evaluation summaries and does not retrain or
re-evaluate any model. It also exports appendix-ready PV-field MAE/RMSE
tables with forecast leads labeled in minutes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_LEAD_START_MIN = 15
DEFAULT_LEAD_STEP_MIN = 15
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_DATA_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", SCRIPT_ROOT / "data"))
)
SATPV_FIELD_LABEL = "SatPV-Field"
PROPOSED_FIELD_FORECASTER_LABEL = "Proposed field forecaster"

STATION_METHODS = {
    "temporal_baseline": "BiLSTM+NWP",
    "agcrn_nwp": "AGCRN+NWP",
    "fg_msf": SATPV_FIELD_LABEL,
}

STATION_METHOD_ALIASES = {
    "temporal_baseline": ["BiLSTM + NWP", "BiLSTM+NWP"],
    "agcrn_nwp": ["AGCRN + NWP", "AGCRN+NWP"],
    "fg_msf": ["Proposed FG-MSF", "FG-MSF", "SatPV-Field"],
}

FIELD_METHODS = {
    "persistence_last": "Persistence",
    "pv_nwp": "PV+NWP U-Net",
    "sag_pvff": PROPOSED_FIELD_FORECASTER_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT,
        help="Code project root. Default: inferred from this script location.",
    )
    parser.add_argument(
        "--old_data_root", type=Path, default=DEFAULT_OLD_DATA_ROOT,
        help="Root of Stage-3/Stage-4 artifacts. Default: SATPV_DATA_ROOT, OLD_DATA_ROOT, or <repo>/data.",
    )
    parser.add_argument(
        "--stage4_horizon_csv",
        type=Path,
        default=None,
        help=(
            "Stage-4 horizon table. Default: "
            "<root>/stage_4/runs/22_collect_stage4_all_results/"
            "stage4_all_results_by_horizon.csv"
        ),
    )
    parser.add_argument(
        "--stage3_base_csv",
        type=Path,
        default=None,
        help=(
            "Stage-3 baseline field summary. Default: "
            "<root>/stage_3/runs/base_pvnorm_field_model/"
            "pvnorm_field_eval_summary.csv"
        ),
    )
    parser.add_argument(
        "--stage3_sag_csv",
        type=Path,
        default=None,
        help="Stage-3 proposed field forecaster summary. If omitted, common run paths are tried.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--lead_start_min",
        type=int,
        default=DEFAULT_LEAD_START_MIN,
        help="Forecast lead assigned to h1. Default: 15 min from the latest observed PV frame.",
    )
    parser.add_argument(
        "--lead_step_min",
        type=int,
        default=DEFAULT_LEAD_STEP_MIN,
        help="Lead-time spacing between adjacent horizons. Default: 15 min.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/Framework_draw/horizon_wise_mae_assets",
    )
    parser.add_argument("--basename", type=str, default="horizon_wise_mae_two_panel")
    parser.add_argument(
        "--show_panel_labels",
        type=int,
        default=0,
        help="Set to 1 to draw '(a) Station-level MAE' and '(b) PV-field MAE' inside the combined figure.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def horizon_to_step(value: object) -> int | None:
    text = str(value).strip().lower()
    if text.startswith("h"):
        text = text[1:]
    try:
        step = int(text)
    except ValueError:
        return None
    return step if 1 <= step <= 8 else None


def step_to_lead_min(step: int, lead_start_min: int, lead_step_min: int) -> int:
    return int(lead_start_min) + (int(step) - 1) * int(lead_step_min)


def require_file(path: Path, name: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        token = str(path.absolute())
        if token not in seen:
            seen.add(token)
            result.append(path)
    return result


def stage3_roots(root: Path, old_data_root: Path) -> list[Path]:
    return unique_paths(
        [
            root / "stage_3",
            root / "stage3",
            old_data_root / "stage3",
            old_data_root / "stage_3",
            old_data_root,
        ]
    )


def stage4_roots(root: Path, old_data_root: Path) -> list[Path]:
    return unique_paths(
        [
            root / "stage_4",
            root / "stage4",
            old_data_root / "stage4",
            old_data_root / "stage_4",
            old_data_root,
        ]
    )


def choose_existing(explicit: Path | None, candidates: list[Path], name: str) -> Path:
    if explicit is not None:
        return require_file(Path(explicit), name)
    tried = unique_paths(candidates)
    for path in tried:
        if path.is_file():
            print(f"input {name}: {path}", flush=True)
            return path
    options = "\n".join(str(path) for path in tried)
    raise FileNotFoundError(f"Cannot find {name}. Tried:\n{options}")


def pick_col(df: pd.DataFrame, candidates: list[str], path: Path) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot find any of columns {candidates} in {path}. "
        f"Existing columns: {list(df.columns)}"
    )


def load_station_curve(path: Path, split: str, lead_start_min: int, lead_step_min: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_sn": str})
    method_col = pick_col(df, ["method_key", "model", "Method"], path)
    horizon_col = pick_col(df, ["horizon"], path)
    mae_col = pick_col(df, ["MAE", "mae_pv_norm", "mae"], path)
    rmse_col = pick_col(df, ["RMSE", "rmse_pv_norm", "rmse"], path)

    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str) == str(split)].copy()

    rows: list[dict[str, object]] = []
    for key, label in STATION_METHODS.items():
        group = df[df[method_col].astype(str) == key].copy()
        if group.empty:
            aliases = STATION_METHOD_ALIASES.get(key, [])
            group = df[df[method_col].astype(str).isin(aliases)].copy()
        if group.empty:
            raise ValueError(f"Missing station method {key!r} in {path}")

        group["h_step"] = group[horizon_col].map(horizon_to_step)
        group = group[group["h_step"].notna()].copy()
        for _, row in group.iterrows():
            step = int(row["h_step"])
            rows.append(
                {
                    "panel": "station",
                    "method": label,
                    "h_step": step,
                    "lead_min": step_to_lead_min(step, lead_start_min, lead_step_min),
                    "mae": float(row[mae_col]),
                    "rmse": float(row[rmse_col]),
                }
            )

    return pd.DataFrame(rows).sort_values(["method", "h_step"]).reset_index(drop=True)


def candidate_sag_paths(root: Path, old_data_root: Path) -> list[Path]:
    relatives = [
        Path("runs/ablation_swr_dependency/eval_full_swr_future/motion_branch_eval_summary.csv"),
        Path("runs/ablation_motion_branch_last_only/warp_direct_gate/swr_anchor_guided_pvnorm_field_eval_summary.csv"),
        Path("runs/swr_anchor_motion_fusion_pvnorm_field/swr_anchor_guided_pvnorm_field_eval_summary.csv"),
        Path("runs/ablation_motion_branch/warp_direct_gate/swr_anchor_guided_pvnorm_field_eval_summary.csv"),
    ]
    return unique_paths([base / relative for base in stage3_roots(root, old_data_root) for relative in relatives])


def choose_sag_csv(root: Path, old_data_root: Path, explicit: Path | None) -> Path:
    return choose_existing(
        explicit,
        candidate_sag_paths(root, old_data_root),
        "proposed field forecaster CSV",
    )


def load_stage3_summary(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str) == str(split)].copy()

    horizon_col = pick_col(df, ["horizon"], path)
    mae_col = pick_col(df, ["mae_pv_norm_field", "mae_field", "mae"], path)
    rmse_col = pick_col(df, ["rmse_pv_norm_field", "rmse_field", "rmse"], path)
    df["h_step"] = df[horizon_col].map(horizon_to_step)
    df = df[df["h_step"].notna()].copy()
    df["mae_value"] = df[mae_col].astype(float)
    df["rmse_value"] = df[rmse_col].astype(float)
    return df


def load_field_curve(
    base_csv: Path,
    sag_csv: Path,
    split: str,
    lead_start_min: int,
    lead_step_min: int,
) -> pd.DataFrame:
    base = load_stage3_summary(base_csv, split)
    model_col = pick_col(base, ["model", "variant", "method"], base_csv)

    rows: list[dict[str, object]] = []
    for key, label in [
        ("persistence_last", FIELD_METHODS["persistence_last"]),
        ("pv_nwp", FIELD_METHODS["pv_nwp"]),
    ]:
        group = base[base[model_col].astype(str) == key].copy()
        if group.empty:
            raise ValueError(f"Missing field method {key!r} in {base_csv}")
        for _, row in group.iterrows():
            step = int(row["h_step"])
            rows.append(
                {
                    "panel": "field",
                    "method": label,
                    "h_step": step,
                    "lead_min": step_to_lead_min(step, lead_start_min, lead_step_min),
                    "mae": float(row["mae_value"]),
                    "rmse": float(row["rmse_value"]),
                }
            )

    sag = load_stage3_summary(sag_csv, split)
    variant_col = None
    for col in [
        "motion_branch",
        "swr_dependency_variant",
        "anchor_mixer_variant",
        "temporal_collapse_variant",
        "variant",
        "model",
    ]:
        if col in sag.columns:
            variant_col = col
            break
    if variant_col is not None and sag[variant_col].nunique() > 1:
        best_variant = sag.groupby(variant_col)["mae_value"].mean().sort_values().index[0]
        sag = sag[sag[variant_col].astype(str) == str(best_variant)].copy()

    for _, row in sag.iterrows():
        step = int(row["h_step"])
        rows.append(
            {
            "panel": "field",
            "method": FIELD_METHODS["sag_pvff"],
            "h_step": step,
            "lead_min": step_to_lead_min(step, lead_start_min, lead_step_min),
            "mae": float(row["mae_value"]),
            "rmse": float(row["rmse_value"]),
        }
    )

    return pd.DataFrame(rows).sort_values(["method", "h_step"]).reset_index(drop=True)


def _style_axis(ax, leads_min: list[int]) -> None:
    ax.set_xlabel("Forecast lead (min)")
    ax.set_ylabel("MAE")
    ax.set_xticks(leads_min)
    margin = max(3, int(round((leads_min[1] - leads_min[0]) * 0.2))) if len(leads_min) > 1 else 3
    ax.set_xlim(min(leads_min) - margin, max(leads_min) + margin)
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.8)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _draw_panel(ax, curves: pd.DataFrame, panel_key: str, method_order: list[str], leads_min: list[int]) -> None:
    colors = {
        "BiLSTM+NWP": "#4C78A8",
        "AGCRN+NWP": "#F58518",
        SATPV_FIELD_LABEL: "#B00000",
        "Persistence": "#6B7280",
        "PV+NWP U-Net": "#4C78A8",
        PROPOSED_FIELD_FORECASTER_LABEL: "#B00000",
    }
    markers = {
        "BiLSTM+NWP": "o",
        "AGCRN+NWP": "s",
        SATPV_FIELD_LABEL: "^",
        "Persistence": "o",
        "PV+NWP U-Net": "s",
        PROPOSED_FIELD_FORECASTER_LABEL: "^",
    }
    sub = curves[curves["panel"] == panel_key].copy()
    for method in method_order:
        group = sub[sub["method"] == method].sort_values("lead_min")
        if group.empty:
            continue
        ax.plot(
            group["lead_min"],
            group["mae"],
            label=method,
            color=colors.get(method, "black"),
            marker=markers.get(method, "o"),
            linewidth=1.5,
            markersize=3.4,
        )
    _style_axis(ax, leads_min)


def _save_figure(fig, out_dir: Path, basename: str, dpi: int) -> None:
    pdf_path = out_dir / f"{basename}.pdf"
    svg_path = out_dir / f"{basename}.svg"
    png_path = out_dir / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    for path in [pdf_path, svg_path, png_path]:
        print(f"saved: {path}")


def save_appendix_field_table(curves: pd.DataFrame, out_dir: Path) -> None:
    field = curves[curves["panel"] == "field"].copy()
    if field.empty:
        return
    compact = field.sort_values(["lead_min", "method"])[["lead_min", "method", "mae", "rmse"]]
    long_path = out_dir / "horizon_wise_pv_field_mae_rmse_long.csv"
    wide_path = out_dir / "horizon_wise_pv_field_mae_rmse_wide.csv"
    compact.to_csv(long_path, index=False)

    pivot = compact.pivot(index="lead_min", columns="method", values=["mae", "rmse"])
    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    preferred = []
    for method in ["Persistence", "PV+NWP U-Net", PROPOSED_FIELD_FORECASTER_LABEL]:
        preferred.extend([f"{method}_mae", f"{method}_rmse"])
    pivot = pivot[[col for col in preferred if col in pivot.columns]].reset_index()
    pivot.to_csv(wide_path, index=False)
    print(f"saved: {long_path}")
    print(f"saved: {wide_path}")

    print("\n==== Horizon-wise regional PV-field MAE/RMSE ====")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n==== LaTeX-ready rows ====")
    method_names = ["Persistence", "PV+NWP U-Net", PROPOSED_FIELD_FORECASTER_LABEL]
    for _, row in pivot.iterrows():
        values = [f"{int(row['lead_min'])}"]
        for method in method_names:
            mae_col = f"{method}_mae"
            rmse_col = f"{method}_rmse"
            values.append(f"{row[mae_col]:.4f}" if mae_col in row else "--")
            values.append(f"{row[rmse_col]:.4f}" if rmse_col in row else "--")
        print(" & ".join(values) + r" \\")


def save_appendix_station_table(curves: pd.DataFrame, out_dir: Path) -> None:
    station = curves[curves["panel"] == "station"].copy()
    if station.empty:
        return
    compact = station.sort_values(["lead_min", "method"])[["lead_min", "method", "mae", "rmse"]]
    long_path = out_dir / "horizon_wise_station_mae_rmse_long.csv"
    wide_path = out_dir / "horizon_wise_station_mae_rmse_wide.csv"
    compact.to_csv(long_path, index=False)

    pivot = compact.pivot(index="lead_min", columns="method", values=["mae", "rmse"])
    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    preferred = []
    for method in ["BiLSTM+NWP", "AGCRN+NWP", SATPV_FIELD_LABEL]:
        preferred.extend([f"{method}_mae", f"{method}_rmse"])
    pivot = pivot[[col for col in preferred if col in pivot.columns]].reset_index()
    pivot.to_csv(wide_path, index=False)
    print(f"saved: {long_path}")
    print(f"saved: {wide_path}")

    print("\n==== Horizon-wise target-station MAE/RMSE ====")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n==== Station LaTeX-ready rows ====")
    method_names = ["BiLSTM+NWP", "AGCRN+NWP", SATPV_FIELD_LABEL]
    for _, row in pivot.iterrows():
        values = [f"{int(row['lead_min'])}"]
        for method in method_names:
            mae_col = f"{method}_mae"
            rmse_col = f"{method}_rmse"
            values.append(f"{row[mae_col]:.4f}" if mae_col in row else "--")
            values.append(f"{row[rmse_col]:.4f}" if rmse_col in row else "--")
        print(" & ".join(values) + r" \\")


def plot(curves: pd.DataFrame, out_dir: Path, basename: str, dpi: int, show_panel_labels: bool) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    leads_min = sorted(int(v) for v in curves["lead_min"].dropna().unique())

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    panels = [
        ("station", "(a) Station-level MAE", axes[0], ["BiLSTM+NWP", "AGCRN+NWP", SATPV_FIELD_LABEL]),
        ("field", "(b) PV-field MAE", axes[1], ["Persistence", "PV+NWP U-Net", PROPOSED_FIELD_FORECASTER_LABEL]),
    ]

    for panel_key, panel_label, ax, method_order in panels:
        _draw_panel(ax, curves, panel_key, method_order, leads_min)
        if show_panel_labels:
            ax.text(
                0.0,
                1.02,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )

    csv_path = out_dir / f"{basename}_data.csv"
    curves.to_csv(csv_path, index=False)
    save_appendix_station_table(curves, out_dir)
    save_appendix_field_table(curves, out_dir)

    _save_figure(fig, out_dir, basename, dpi)
    plt.close(fig)

    separate_specs = [
        ("station", "horizon_wise_station_mae", ["BiLSTM+NWP", "AGCRN+NWP", SATPV_FIELD_LABEL]),
        ("field", "horizon_wise_pv_field_mae", ["Persistence", "PV+NWP U-Net", PROPOSED_FIELD_FORECASTER_LABEL]),
    ]
    for panel_key, separate_name, method_order in separate_specs:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(3.45, 2.35), constrained_layout=True)
        _draw_panel(ax_single, curves, panel_key, method_order, leads_min)
        _save_figure(fig_single, out_dir, separate_name, dpi)
        plt.close(fig_single)

    print(f"saved: {csv_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    old_data_root = Path(args.old_data_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir is not None else root / "Framework_draw/horizon_wise_mae_assets"

    stage4_horizon_csv = choose_existing(
        args.stage4_horizon_csv,
        [
            base / "runs/22_collect_stage4_all_results/stage4_all_results_by_horizon.csv"
            for base in stage4_roots(root, old_data_root)
        ],
        "stage4_horizon_csv",
    )
    stage3_base_csv = choose_existing(
        args.stage3_base_csv,
        [
            base / "runs/base_pvnorm_field_model/pvnorm_field_eval_summary.csv"
            for base in stage3_roots(root, old_data_root)
        ],
        "stage3_base_csv",
    )
    stage3_sag_csv = choose_sag_csv(root, old_data_root, args.stage3_sag_csv)

    print(f"output directory: {out_dir}", flush=True)

    station_curve = load_station_curve(
        stage4_horizon_csv,
        str(args.split),
        int(args.lead_start_min),
        int(args.lead_step_min),
    )
    field_curve = load_field_curve(
        stage3_base_csv,
        stage3_sag_csv,
        str(args.split),
        int(args.lead_start_min),
        int(args.lead_step_min),
    )
    curves = pd.concat([station_curve, field_curve], ignore_index=True)

    plot(curves, out_dir, str(args.basename), int(args.dpi), bool(int(args.show_panel_labels)))


if __name__ == "__main__":
    main()
