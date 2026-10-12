#!/usr/bin/env python3
"""Train field-guided station forecasters for one source-fleet density level."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
PIPELINE_DIR = STAGE4_DIR.parent
STAGE3_DIR = PIPELINE_DIR / "stage_3"
STAGE3_DENSITY_DIR = STAGE3_DIR / "source_fleet_density"
for extra in (STAGE4_DIR, STAGE3_DIR, STAGE3_DENSITY_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from density_common import (
    DEFAULT_SEED,
    density_frame_cache_dir,
    density_selection_dir,
    density_stage3_run_dir,
    density_stage4_run_dir,
    density_target_run_dir,
    ratio_label,
    write_json,
)
from stage4_paths import CENTER_H8_SWR_ANCHOR_CACHE_ROOT, MULTISOURCE_CACHE_DIR, TRAIN_PV_DIR, VAL_PV_DIR


MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate"


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--reuse_full_density", action="store_true")
    wrapper, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return wrapper


def load_stage4_05l():
    path = STAGE4_DIR / "05l_train_per_station_temporal_with_stage3_field_patch.py"
    spec = importlib.util.spec_from_file_location("stage4_density_05l", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Stage 4 05l script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage4_density_05l"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    wrapper = parse_wrapper_args()
    ratio = float(wrapper.ratio)
    if ratio >= 1.0 and wrapper.reuse_full_density:
        print("r100 is configured as the existing full-density FG-MSF reference; no training launched.")
        return

    stage3_run = density_stage3_run_dir(wrapper.seed, ratio)
    checkpoint = stage3_run / f"{MODEL_NAME}_best.pt"
    for path in (
        checkpoint,
        density_selection_dir(wrapper.seed, ratio) / "target_stations.csv",
        density_selection_dir(wrapper.seed, ratio) / "issue_time_index.csv",
        density_frame_cache_dir(wrapper.seed, ratio) / "frame_cache_meta.json",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing density artifact before Stage 4 training: {path}")

    out_dir = density_stage4_run_dir(wrapper.seed, ratio)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "density_fgmsf_meta.json",
        {
            "purpose": "FG-MSF retrained using the predicted PV-field patches from one source-fleet density SAG-PVFF model.",
            "seed": int(wrapper.seed),
            "source_density_ratio": float(ratio),
            "source_density_label": ratio_label(ratio),
            "stage3_field_model_dir": str(stage3_run),
            "stage3_field_checkpoint": str(checkpoint),
            "field_model_name": MODEL_NAME,
        },
    )

    module = load_stage4_05l()
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(density_selection_dir(wrapper.seed, ratio)),
            "time_index_csv": str(density_selection_dir(wrapper.seed, ratio) / "issue_time_index.csv"),
            "target_run_dir": str(density_target_run_dir(wrapper.seed, ratio)),
            "frame_cache_dir": str(density_frame_cache_dir(wrapper.seed, ratio)),
            "multisource_cache_dir": str(MULTISOURCE_CACHE_DIR),
            "field_model_dir": str(stage3_run),
            "field_model_name": MODEL_NAME,
            "field_checkpoint": str(checkpoint),
            "swr_anchor_cache_dir": str(CENTER_H8_SWR_ANCHOR_CACHE_ROOT),
            "base_model_name": "pv_nwp",
            "out_dir": str(out_dir),
            "train_pv_dir": str(TRAIN_PV_DIR),
            "val_pv_dir": str(VAL_PV_DIR),
        }
    )
    print("==== density FG-MSF training ====")
    print("density:", ratio_label(ratio), "seed:", wrapper.seed)
    print("stage3 checkpoint:", checkpoint)
    print("output:", out_dir)
    module.main()


if __name__ == "__main__":
    main()
