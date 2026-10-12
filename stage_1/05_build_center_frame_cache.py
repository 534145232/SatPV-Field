"""Build uncompressed monthly frame cache for the Stage 1 center crop."""

import build_frame_cache as base

from stage1_paths import (  # noqa: E402
    CENTER_FRAME_CACHE_DIR,
    CENTER_INDEX_ALL,
    ensure_stage1_dirs,
)


base.DEFAULT_INDEX_CSV = CENTER_INDEX_ALL
base.DEFAULT_FRAME_CACHE_DIR = CENTER_FRAME_CACHE_DIR


if __name__ == "__main__":
    ensure_stage1_dirs()
    base.main()
