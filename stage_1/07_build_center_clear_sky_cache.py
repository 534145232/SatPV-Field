"""Build paired center clear-sky SWR fields for fast ratio-model training."""

import build_clear_sky_cache as base

from stage1_paths import (  # noqa: E402
    CENTER_CLEAR_SKY_CACHE_DIR,
    CENTER_INDEX_ALL,
    CENTER_SAMPLE_CACHE_DIR,
    ensure_stage1_dirs,
)


base.DEFAULT_INDEX_CSV = CENTER_INDEX_ALL
base.DEFAULT_SAMPLE_CACHE_DIR = CENTER_SAMPLE_CACHE_DIR
base.DEFAULT_CLEAR_SKY_DIR = CENTER_CLEAR_SKY_CACHE_DIR


if __name__ == "__main__":
    ensure_stage1_dirs()
    base.main()
