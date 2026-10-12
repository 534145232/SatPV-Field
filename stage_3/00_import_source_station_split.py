#!/usr/bin/env python3
"""Import the one upstream regional station table needed by Stage 3."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from stage3_paths import ARTIFACT_ROOT, SOURCE_REGIONAL_RUN_DIR, ensure_stage3_dirs


DEFAULT_SOURCE = Path(
    os.environ.get("STATION_SPLIT_CSV", ARTIFACT_ROOT / "station_split.csv")
).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    target = SOURCE_REGIONAL_RUN_DIR / "station_split.csv"
    if target.exists() and not args.overwrite:
        print("reuse:", target)
        return
    if not args.source.exists():
        raise FileNotFoundError(
            f"Upstream regional station split not found: {args.source}. "
            "Pass --source explicitly if the upstream metadata was stored elsewhere."
        )
    shutil.copy2(args.source, target)
    print("imported:", args.source)
    print("saved:", target)


if __name__ == "__main__":
    main()
