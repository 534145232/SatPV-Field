#!/usr/bin/env python3
"""Select the unchanged LHS-50 target set and cache the remaining PV field."""

from stage3_paths import (
    SOURCE_REGIONAL_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    TIME_END,
    TIME_START,
    TRAIN_END,
    TRAIN_PV_DIR,
    VAL_END,
    VAL_PV_DIR,
    ensure_stage3_dirs,
)
from stage3_cli import inject_defaults
import stage3_select_targets_pvnorm_impl as module
from stage3_temporal import build_pv_issue_time_table


def main() -> None:
    ensure_stage3_dirs()
    module.PYCHARM_DEFAULTS.update(
        {
            "run_dir": str(SOURCE_REGIONAL_RUN_DIR),
            "out_dir": str(TARGET_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "frame_cache_dir": str(TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
            "train_pv_dir": str(TRAIN_PV_DIR),
            "val_pv_dir": str(VAL_PV_DIR),
            "time_start": TIME_START,
            "time_end": TIME_END,
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "max_targets": 50,
            "selection_method": "lhs",
            "force_rebuild": 0,
        }
    )
    module.build_multisource_issue_time_table = build_pv_issue_time_table
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()


if __name__ == "__main__":
    main()
