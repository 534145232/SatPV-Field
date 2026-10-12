#!/usr/bin/env python3
"""Evaluate full-SWR, history-SWR-only and no-SWR Stage 3 ablations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
BASE_EVAL = STAGE3_DIR / "ablation_motion_branch" / "eval_all_motion_branches.py"
RUN_ROOT = STAGE3_DIR / "runs" / "ablation_swr_dependency"
BASELINE_RUN_ROOT = STAGE3_DIR / "runs" / "ablation_motion_branch_last_only"


EVAL_SPECS = {
    "full_swr_future": {
        "run_root": BASELINE_RUN_ROOT,
        "out_dir": RUN_ROOT / "eval_full_swr_future",
        "note": "Reused best baseline: last_only + warp_direct_gate + full future SWR.",
    },
    "history_swr_only": {
        "run_root": RUN_ROOT / "history_swr_only",
        "out_dir": RUN_ROOT / "eval_history_swr_only",
        "note": "Future SWR slots are replaced by latest observed SWR after normalization.",
    },
    "no_swr": {
        "run_root": RUN_ROOT / "no_swr",
        "out_dir": RUN_ROOT / "eval_no_swr",
        "note": "All normalized SWR-anchor channels are set to zero.",
    },
}


def run_eval(variant: str, extra_args: list[str]) -> Path:
    spec = EVAL_SPECS[variant]
    out_dir = Path(spec["out_dir"])
    cmd = [
        sys.executable,
        str(BASE_EVAL),
        "--motion_branch",
        "warp_direct_gate",
        "--temporal_collapse",
        "last_only",
        "--swr_dependency",
        variant,
        "--run_root",
        str(spec["run_root"]),
        "--out_dir",
        str(out_dir),
        *extra_args,
    ]
    print(f"==== Evaluate SWR-dependency ablation: {variant} ====", flush=True)
    subprocess.check_call(cmd)
    return out_dir / "motion_branch_eval_summary.csv"


def main() -> None:
    extra_args = sys.argv[1:]
    frames = []
    sources = {}
    for variant in EVAL_SPECS:
        csv_path = run_eval(variant, extra_args)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path)
        frame.insert(0, "swr_dependency_variant", variant)
        frame.insert(1, "variant_note", EVAL_SPECS[variant]["note"])
        frames.append(frame)
        sources[variant] = str(csv_path)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    out_dir = RUN_ROOT / "eval_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "swr_dependency_eval_summary.csv"
    combined.to_csv(out_csv, index=False)
    (out_dir / "swr_dependency_eval_meta.json").write_text(
        json.dumps(
            {
                "purpose": "Stage 3 SWR dependency ablation with current best architecture.",
                "baseline": "full_swr_future reuses ablation_motion_branch_last_only/warp_direct_gate.",
                "sources": sources,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    show = combined[
        (combined["eval_split"] == "test")
        & combined["horizon"].astype(str).isin(["all", "h1", "h4", "h8"])
    ]
    columns = [
        col
        for col in [
            "swr_dependency_variant",
            "eval_split",
            "horizon",
            "mae_pv_norm_field",
            "rmse_pv_norm_field",
            "delta_sign_accuracy",
            "large_delta_mae",
            "delta_corr_median",
            "horizon_diversity_ratio",
        ]
        if col in show.columns
    ]
    print("==== Stage 3 SWR-dependency ablation summary ====")
    print(show[columns].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
