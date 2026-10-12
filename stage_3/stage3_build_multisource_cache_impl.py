#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPT_DIR))

import build_fullfield_multisource_cache as base
import stage3_paths as p3

USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS = True

NEW_DEFAULTS = {
    "run_dir": str(p3.SOURCE_REGIONAL_RUN_DIR),
    "out_dir": str(p3.MULTISOURCE_CACHE_DIR),
    "pv_frame_cache_dir": str(p3.TARGET_EXCLUDED_PV_FRAME_CACHE_DIR),
    "train_pv_dir": str(p3.TRAIN_PV_DIR),
    "val_pv_dir": str(p3.VAL_PV_DIR),
    "freq_minutes": 15,
    "issue_stride_minutes": 60,
    "past_frames": 12,
    "future_frames": 8,
    "grid_size": 64,
    "force_rebuild": 1,
}


def _inject_defaults() -> None:
    if not USE_PYCHARM_DEFAULTS_WHEN_NO_ARGS or len(sys.argv) > 1:
        return
    args = [sys.argv[0]]
    for key, value in NEW_DEFAULTS.items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args


def main() -> None:
    _inject_defaults()
    # The H8/NWP cache itself does not depend on the PV frame cache.  The
    # upstream builder only opens PV frames for optional sample plots, so this
    # wrapper disables that plotting to allow rebuilding multisource data before
    # the target-excluded PV field is constructed.
    class _NoPVFrameCache:
        def __init__(self, *args, **kwargs):
            pass

    base.ReadOnlyDiskFrameCache = _NoPVFrameCache
    base._render_samples = lambda *args, **kwargs: None
    base.main()


if __name__ == "__main__":
    main()
