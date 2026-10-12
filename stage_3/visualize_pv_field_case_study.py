#!/usr/bin/env python3
"""Generate PV-field forecasting case-study figures for the paper.

Each selected case is rendered as:
Ground truth | U-Net baseline | Baseline absolute error | Proposed field forecaster | Proposed field-forecast error
for h=15 min, h=60 min, and h=105 min by default.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


STAGE3_DIR = Path(__file__).resolve().parent
ABLATION_MOTION_DIR = STAGE3_DIR / "ablation_motion_branch"
for path in [STAGE3_DIR, ABLATION_MOTION_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage3_base_pvnorm_field_model as f04
import stage3_motion_fusion_model_impl as motion_module
import eval_all_motion_branches as motion_eval
from stage3_paths import RUNS_DIR, STAGE3_DIR as STAGE3_PATH
from stage3_swr_anchor_cache import assert_cache_matches_checkpoint, resolve_active_cache_dir
from swr_dependency import install_swr_dependency_dataset


COLOR_STOPS = [
    (0.00, (8, 48, 107)),
    (0.20, (33, 113, 181)),
    (0.40, (222, 235, 247)),
    (0.54, (255, 255, 217)),
    (0.68, (254, 204, 92)),
    (0.83, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]
FLOW_COLOR_STOPS = [
    (0.00, (222, 235, 247)),
    (0.25, (107, 174, 214)),
    (0.52, (255, 255, 217)),
    (0.78, (254, 204, 92)),
    (1.00, (189, 0, 38)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 3 PV-field case-study figures.")
    parser.add_argument("--out_dir", type=Path, default=STAGE3_PATH / "paper_figures" / "pv_field_case_study")
    parser.add_argument("--baseline_run_dir", type=Path, default=RUNS_DIR / "base_pvnorm_field_model")
    parser.add_argument("--baseline_model", type=str, default="pv_nwp")
    parser.add_argument("--baseline_delta_scale", type=float, default=float(f04.PYCHARM_DEFAULTS.get("delta_scale", 1.0)))
    parser.add_argument("--proposed_run_root", type=Path, default=RUNS_DIR / "ablation_motion_branch_last_only")
    parser.add_argument("--proposed_motion_branch", type=str, default="warp_direct_gate")
    parser.add_argument("--temporal_collapse", type=str, default="last_only")
    parser.add_argument("--swr_dependency", type=str, default="full_swr_future")
    parser.add_argument("--swr_anchor_cache_dir", type=Path, default=None)
    parser.add_argument("--swr_checkpoint", type=Path, default=motion_eval.SWR_BASE_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scan_count", type=int, default=300, help="Number of test samples to score. <=0 scans all.")
    parser.add_argument("--scan_mode", choices=("even", "first", "random"), default="even")
    parser.add_argument(
        "--case_select_mode",
        choices=("random", "score"),
        default="random",
        help="random draws a fresh random batch; score keeps the old improvement-ranked case selection.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_cases", type=int, default=8)
    parser.add_argument(
        "--issue_times",
        type=str,
        default="",
        help="Optional comma-separated issue times to render, e.g. 20230608_120000 or '2023-06-08 12:00:00'. Overrides random/score selection.",
    )
    parser.add_argument("--min_issue_gap_hours", type=float, default=24.0)
    parser.add_argument("--horizons", type=str, default="1,4,7", help="1-based PV horizons. Default 1,4,7 = 15,60,105 min labels.")
    parser.add_argument("--pv_cmap", type=str, default="shared_swr")
    parser.add_argument("--err_cmap", type=str, default="shared_flow")
    parser.add_argument("--pv_vmin", type=float, default=0.0)
    parser.add_argument("--pv_vmax", type=float, default=1.2)
    parser.add_argument("--err_vmin", type=float, default=0.0)
    parser.add_argument("--err_vmax", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save_pdf", type=int, default=1)
    return parser.parse_args()


def parse_horizons(text: str, future_frames: int) -> list[int]:
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw) - 1
        if value < 0 or value >= future_frames:
            raise ValueError(f"Invalid 1-based horizon {raw}; valid range is 1..{future_frames}")
        out.append(value)
    return out or [0, min(6, future_frames - 1)]


def masked_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - truth[valid])))


def masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if not np.any(valid):
        return float("nan")
    return float(np.std(values[valid]))


def denorm(field_z: np.ndarray, pv_stats: dict) -> np.ndarray:
    return f04._denorm_field(np.asarray(field_z, dtype=np.float32), pv_stats)


def issue_gap_ok(issue: pd.Timestamp, chosen: list[dict], min_gap_hours: float) -> bool:
    if min_gap_hours <= 0:
        return True
    for item in chosen:
        gap = abs((pd.Timestamp(issue) - pd.Timestamp(item["issue_time"])).total_seconds()) / 3600.0
        if gap < min_gap_hours:
            return False
    return True


def choose_candidate_indices(n: int, scan_count: int, scan_mode: str, seed: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    if scan_count <= 0 or scan_count >= n:
        return np.arange(n, dtype=int)
    if scan_mode == "first":
        return np.arange(scan_count, dtype=int)
    if scan_mode == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n, size=scan_count, replace=False)).astype(int)
    return np.unique(np.linspace(0, n - 1, scan_count, dtype=int))


def random_case_candidates(args: argparse.Namespace, dataset) -> list[dict]:
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(len(dataset)).astype(int)
    chosen: list[dict] = []
    for idx in order:
        issue = pd.Timestamp(dataset.time_df.iloc[int(idx)]["issue_time"])
        item = {
            "idx": int(idx),
            "issue_time": issue,
            "score": float("nan"),
            "improvement_h105": float("nan"),
            "baseline_mae_h105": float("nan"),
            "proposed_mae_h105": float("nan"),
            "gt_std_h105": float("nan"),
            "case_select_mode": "random",
        }
        if issue_gap_ok(issue, chosen, float(args.min_issue_gap_hours)):
            chosen.append(item)
        if len(chosen) >= int(args.num_cases):
            break
    if len(chosen) < int(args.num_cases):
        seen = {item["idx"] for item in chosen}
        for idx in order:
            if int(idx) in seen:
                continue
            issue = pd.Timestamp(dataset.time_df.iloc[int(idx)]["issue_time"])
            chosen.append(
                {
                    "idx": int(idx),
                    "issue_time": issue,
                    "score": float("nan"),
                    "improvement_h105": float("nan"),
                    "baseline_mae_h105": float("nan"),
                    "proposed_mae_h105": float("nan"),
                    "gt_std_h105": float("nan"),
                    "case_select_mode": "random",
                }
            )
            seen.add(int(idx))
            if len(chosen) >= int(args.num_cases):
                break
    return chosen


def parse_requested_issue_times(text: str) -> list[pd.Timestamp]:
    out = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        if re.fullmatch(r"\d{8}_\d{6}", token):
            out.append(pd.to_datetime(token, format="%Y%m%d_%H%M%S"))
        else:
            out.append(pd.Timestamp(token))
    return out


def requested_issue_time_candidates(args: argparse.Namespace, dataset) -> list[dict]:
    requested = parse_requested_issue_times(str(args.issue_times))
    if not requested:
        return []
    issue_series = pd.to_datetime(dataset.time_df["issue_time"])
    chosen = []
    for issue in requested:
        matches = np.flatnonzero(issue_series.to_numpy() == np.datetime64(issue.to_datetime64()))
        if len(matches) == 0:
            examples = ", ".join(pd.to_datetime(issue_series.head(5)).dt.strftime("%Y-%m-%d %H:%M:%S").tolist())
            raise ValueError(f"Requested issue_time {issue:%Y-%m-%d %H:%M:%S} not found in test dataset. First examples: {examples}")
        idx = int(matches[0])
        chosen.append(
            {
                "idx": idx,
                "issue_time": pd.Timestamp(issue_series.iloc[idx]),
                "score": float("nan"),
                "improvement_h105": float("nan"),
                "baseline_mae_h105": float("nan"),
                "proposed_mae_h105": float("nan"),
                "gt_std_h105": float("nan"),
                "case_select_mode": "requested_issue_time",
            }
        )
    return chosen


def build_everything(args: argparse.Namespace):
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_cache_dir = args.swr_anchor_cache_dir or resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(anchor_cache_dir, args.swr_checkpoint)
    install_swr_dependency_dataset(motion_module, str(args.swr_dependency))

    data_args = SimpleNamespace(
        run_root=Path(args.proposed_run_root),
        temporal_collapse=str(args.temporal_collapse),
        swr_dependency=str(args.swr_dependency),
        motion_branch=str(args.proposed_motion_branch),
        swr_anchor_cache_dir=anchor_cache_dir,
        swr_checkpoint=Path(args.swr_checkpoint),
        out_dir=out_dir / "_eval_cache",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        sample_count=int(args.num_cases),
        visual_horizons=str(args.horizons),
        dpi=int(args.dpi),
        seed=int(args.seed),
    )
    dataset_args, datasets, _loaders, pv_stats, proposed_kwargs = motion_eval.build_data(data_args, anchor_cache_dir, data_args.out_dir)
    device = motion_module._device(str(args.device))
    proposed_models, _skipped = motion_eval.load_models(
        Path(args.proposed_run_root),
        proposed_kwargs,
        device,
        str(args.temporal_collapse),
        str(args.proposed_motion_branch),
    )
    proposed = proposed_models[str(args.proposed_motion_branch)]["model"]

    base_dataset = datasets["test"].base
    base_in, base_future = base_dataset.source_dims()
    baseline_ckpt = Path(args.baseline_run_dir) / f"{args.baseline_model}_best.pt"
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_ckpt}")
    baseline = f04.make_residual_3d_unet_z(
        int(dataset_args.future_frames),
        in_channels=int(base_in),
        future_channels=int(base_future),
        delta_scale=float(args.baseline_delta_scale),
    ).to(device)
    state = torch.load(baseline_ckpt, map_location=device)
    baseline.load_state_dict(state["model_state"])
    baseline.eval()
    proposed.eval()
    return dataset_args, datasets["test"], baseline, proposed, pv_stats, device


def infer_case(dataset, idx: int, baseline, proposed, pv_stats: dict, device: str) -> dict:
    import torch

    x_aug, future_aug, swr, y_z, mask = dataset[int(idx)]
    x_base, future_base, y_base_z, mask_base = dataset.base[int(idx)]
    with torch.no_grad():
        base_pred_z = baseline(
            torch.from_numpy(x_base[None]).to(device),
            torch.from_numpy(future_base[None]).to(device),
        ).detach().cpu().numpy()[0]
        prop_pred_z = proposed(
            torch.from_numpy(x_aug[None]).to(device),
            torch.from_numpy(future_aug[None]).to(device),
            torch.from_numpy(swr[None]).to(device),
        ).detach().cpu().numpy()[0]
    return {
        "idx": int(idx),
        "row": dataset.time_df.iloc[int(idx)].copy(),
        "truth": denorm(y_z, pv_stats),
        "baseline": denorm(base_pred_z, pv_stats),
        "proposed": denorm(prop_pred_z, pv_stats),
        "mask": np.asarray(mask, dtype=np.float32),
        "base_mask": np.asarray(mask_base, dtype=np.float32),
        "base_y": denorm(y_base_z, pv_stats),
    }


def score_cases(args: argparse.Namespace, dataset, baseline, proposed, pv_stats: dict, device: str, horizons: list[int]) -> list[dict]:
    target_h = horizons[-1]
    indices = choose_candidate_indices(len(dataset), int(args.scan_count), str(args.scan_mode), int(args.seed))
    scores = []
    for rank, idx in enumerate(indices, start=1):
        case = infer_case(dataset, int(idx), baseline, proposed, pv_stats, device)
        truth = case["truth"][target_h]
        base_pred = case["baseline"][target_h]
        prop_pred = case["proposed"][target_h]
        mask = case["mask"][target_h]
        base_mae = masked_mae(base_pred, truth, mask)
        prop_mae = masked_mae(prop_pred, truth, mask)
        gt_std = masked_std(truth, mask)
        issue = pd.Timestamp(case["row"]["issue_time"])
        improvement = base_mae - prop_mae
        score = improvement + 0.25 * gt_std
        scores.append(
            {
                "idx": int(idx),
                "issue_time": issue,
                "score": float(score),
                "improvement_h105": float(improvement),
                "baseline_mae_h105": float(base_mae),
                "proposed_mae_h105": float(prop_mae),
                "gt_std_h105": float(gt_std),
                "case_select_mode": "score",
            }
        )
        if rank % 25 == 0 or rank == len(indices):
            print(f"[scan] {rank}/{len(indices)} scored", flush=True)
    scores.sort(key=lambda item: item["score"], reverse=True)

    chosen: list[dict] = []
    for item in scores:
        if issue_gap_ok(item["issue_time"], chosen, float(args.min_issue_gap_hours)):
            chosen.append(item)
        if len(chosen) >= int(args.num_cases):
            break
    if len(chosen) < int(args.num_cases):
        seen = {item["idx"] for item in chosen}
        for item in scores:
            if item["idx"] not in seen:
                chosen.append(item)
                seen.add(item["idx"])
            if len(chosen) >= int(args.num_cases):
                break
    return chosen


def attach_case_score(item: dict, case: dict, horizons: list[int]) -> dict:
    target_h = horizons[-1]
    truth = case["truth"][target_h]
    base_pred = case["baseline"][target_h]
    prop_pred = case["proposed"][target_h]
    mask = case["mask"][target_h]
    base_mae = masked_mae(base_pred, truth, mask)
    prop_mae = masked_mae(prop_pred, truth, mask)
    gt_std = masked_std(truth, mask)
    improvement = base_mae - prop_mae
    out = dict(item)
    out["score"] = float(improvement + 0.25 * gt_std)
    out["improvement_h105"] = float(improvement)
    out["baseline_mae_h105"] = float(base_mae)
    out["proposed_mae_h105"] = float(prop_mae)
    out["gt_std_h105"] = float(gt_std)
    return out


def masked_image(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~(mask > 0.5), arr)


def palette_from_stops(color_stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    stops = np.asarray([rgb for _, rgb in color_stops], dtype=np.float32)
    x = np.asarray([pos for pos, _ in color_stops], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=1).astype(np.float32) / 255.0


def make_shared_cmap(name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap_name = str(name)
    if cmap_name == "shared_swr":
        cmap = ListedColormap(palette_from_stops(COLOR_STOPS), name="shared_swr")
    elif cmap_name == "shared_flow":
        cmap = ListedColormap(palette_from_stops(FLOW_COLOR_STOPS), name="shared_flow")
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def render_case(args: argparse.Namespace, out_dir: Path, case: dict, selected: dict, horizons: list[int]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    import matplotlib.pyplot as plt

    pv_cmap = make_shared_cmap(str(args.pv_cmap))
    err_cmap = make_shared_cmap(str(args.err_cmap))

    fig_h = 1.82 * len(horizons) + 0.42
    fig, axes = plt.subplots(len(horizons), 5, figsize=(9.7, fig_h), squeeze=False)
    col_titles = [
        "Ground Truth",
        "PV+NWP U-Net\n(Baseline)",
        "Baseline\nAbsolute Error",
        "Proposed field\nforecaster",
        "Proposed field\nforecaster\nAbsolute Error",
    ]
    first_pv_im = None
    first_err_im = None
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, pad=6)
    row_labels = []
    for r, h in enumerate(horizons):
        minutes = 15 * (h + 1)
        row_labels.append(f"h = {minutes} min")
        truth = case["truth"][h]
        base_pred = case["baseline"][h]
        prop_pred = case["proposed"][h]
        base_err = np.abs(base_pred - truth)
        prop_err = np.abs(prop_pred - truth)
        mask = case["mask"][h]
        panels = [truth, base_pred, base_err, prop_pred, prop_err]
        for c, values in enumerate(panels):
            ax = axes[r, c]
            if c in (0, 1, 3):
                im = ax.imshow(
                    masked_image(values, mask),
                    origin="lower",
                    cmap=pv_cmap,
                    vmin=float(args.pv_vmin),
                    vmax=float(args.pv_vmax),
                    interpolation="nearest",
                )
                if first_pv_im is None:
                    first_pv_im = im
            else:
                im = ax.imshow(
                    masked_image(values, mask),
                    origin="lower",
                    cmap=err_cmap,
                    vmin=float(args.err_vmin),
                    vmax=float(args.err_vmax),
                    interpolation="nearest",
                )
                if first_err_im is None:
                    first_err_im = im
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(row_labels[-1], fontsize=10, rotation=90, labelpad=12)

    fig.subplots_adjust(left=0.065, right=0.865, top=0.885, bottom=0.07, wspace=0.035, hspace=0.045)
    cax1 = fig.add_axes([0.890, 0.56, 0.018, 0.27])
    cbar1 = fig.colorbar(first_pv_im, cax=cax1)
    cbar1.ax.tick_params(labelsize=8, length=2)
    cbar1.set_label("PV field value", fontsize=9, labelpad=8)
    cax2 = fig.add_axes([0.890, 0.20, 0.018, 0.27])
    cbar2 = fig.colorbar(first_err_im, cax=cax2)
    cbar2.ax.tick_params(labelsize=8, length=2)
    cbar2.set_label("Absolute error", fontsize=9, labelpad=8)

    issue = pd.Timestamp(selected["issue_time"])
    stem = f"case_{selected['rank']:02d}_idx{selected['idx']:05d}_{issue:%Y%m%d_%H%M%S}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=int(args.dpi), transparent=False)
    pdf_path = ""
    svg_path = ""
    if int(args.save_pdf):
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, transparent=False)
        pdf_path = str(pdf)
        svg = out_dir / f"{stem}.svg"
        fig.savefig(svg, transparent=False)
        svg_path = str(svg)
    plt.close(fig)
    return {"png": str(png_path), "pdf": pdf_path, "svg": svg_path}


def write_selected_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_args, dataset, baseline, proposed, pv_stats, device = build_everything(args)
    horizons = parse_horizons(args.horizons, int(dataset_args.future_frames))

    print(f"[data] test common-subset samples: {len(dataset)}", flush=True)
    select_mode = "requested_issue_time" if str(args.issue_times).strip() else str(args.case_select_mode)
    print(
        f"[select] mode={select_mode} scan_count={args.scan_count} "
        f"num_cases={args.num_cases} horizons={args.horizons} seed={args.seed}",
        flush=True,
    )
    if str(args.issue_times).strip():
        selected = requested_issue_time_candidates(args, dataset)
    elif str(args.case_select_mode) == "score":
        selected = score_cases(args, dataset, baseline, proposed, pv_stats, device, horizons)
    else:
        selected = random_case_candidates(args, dataset)

    rows = []
    for rank, item in enumerate(selected, start=1):
        item = dict(item)
        item["rank"] = rank
        case = infer_case(dataset, int(item["idx"]), baseline, proposed, pv_stats, device)
        item = attach_case_score(item, case, horizons)
        rendered = render_case(args, out_dir, case, item, horizons)
        item.update(rendered)
        rows.append(item)
        print(
            f"[case {rank:02d}] idx={item['idx']} issue={pd.Timestamp(item['issue_time']):%Y-%m-%d %H:%M:%S} "
            f"score={item['score']:.4f} improve_h105={item['improvement_h105']:.4f} png={item['png']}",
            flush=True,
        )
    csv_path = out_dir / "selected_cases.csv"
    write_selected_csv(csv_path, rows)
    print(f"saved index: {csv_path}", flush=True)
    print(f"saved figures: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
