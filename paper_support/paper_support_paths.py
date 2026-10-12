#!/usr/bin/env python3
"""Portable data-path resolution for paper-support experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


TRAIN_PV_DIRNAMES = (
    "train_directory_cleaned_csv_same_day06_nwp",
    "train_directory_cleaned_csv",
    "train",
)
VAL_PV_DIRNAMES = (
    "val_directory_cleaned_csv_same_day06_nwp",
    "val_directory_cleaned_csv",
    "val",
    "validation",
)

STAGE3_REPRESENTATION_REQUIRED = (
    "02_target_selection_lhs50/target_stations.csv",
    "02_target_selection_lhs50/target_excluded_station_pv_norm_q95.csv",
    "02_target_selection_lhs50/01_target_selection_meta.json",
    "03_target_excluded_run/station_split.csv",
    "03_target_excluded_run/target_excluded_station_split.csv",
    "04_target_excluded_pvnorm_frame_cache/fields_float16.npy",
    "04_target_excluded_pvnorm_frame_cache/masks_uint8.npy",
    "04_target_excluded_pvnorm_frame_cache/timestamp_sec.npy",
    "04_target_excluded_pvnorm_frame_cache/frame_cache_meta.json",
)


def _unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        if value is None:
            continue
        path = Path(value).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def candidate_old_data_roots(
    project_root: Path | None = None,
    explicit: Path | None = None,
) -> list[Path]:
    """Return candidates, preferring explicit and project-sibling storage."""

    env_root = os.environ.get("SATPV_DATA_ROOT", os.environ.get("BIGPV_OLD_DATA_ROOT"))
    project_root = Path(project_root).expanduser() if project_root is not None else None
    sibling_root = (
        project_root.parent.parent / "old_data" if project_root is not None else None
    )
    return _unique_paths(
        [
            explicit,
            Path(env_root) if env_root else None,
            sibling_root,
            project_root / "data" if project_root is not None else None,
        ]
    )


def resolve_old_data_root(
    project_root: Path,
    explicit: Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    candidates = candidate_old_data_roots(project_root, explicit)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    if require_exists:
        raise FileNotFoundError(
            "Could not resolve the external old_data root. Tried:\n"
            + "\n".join(f"  - {path}" for path in candidates)
        )
    return candidates[0]


def _has_csv(path: Path) -> bool:
    return path.is_dir() and next(path.glob("*.csv"), None) is not None


def resolve_pv_directories(
    pv_root: Path | None = None,
    train_override: Path | None = None,
    val_override: Path | None = None,
    *,
    project_root: Path | None = None,
    old_data_root: Path | None = None,
    require_exists: bool = True,
) -> tuple[Path, Path]:
    """Resolve station CSVs from explicit, sibling-cache, or legacy layouts."""

    env_pv_root = os.environ.get("PV_ROOT", os.environ.get("BIGPV_PV_ROOT"))
    roots = _unique_paths(
        [
            pv_root,
            Path(env_pv_root) if env_pv_root else None,
            Path(old_data_root) / "PV" if old_data_root is not None else None,
            *[root / "PV" for root in candidate_old_data_roots(project_root)],
        ]
    )
    train_candidates = (
        [Path(train_override).expanduser()]
        if train_override is not None
        else [root / name for root in roots for name in TRAIN_PV_DIRNAMES]
    )
    val_candidates = (
        [Path(val_override).expanduser()]
        if val_override is not None
        else [root / name for root in roots for name in VAL_PV_DIRNAMES]
    )
    if train_override is None:
        train_candidates.extend(root for root in roots if _has_csv(root))
    if val_override is None:
        val_candidates.extend(root for root in roots if _has_csv(root))

    train_dir = next((path.resolve() for path in train_candidates if _has_csv(path)), None)
    val_dir = next((path.resolve() for path in val_candidates if _has_csv(path)), None)
    if require_exists and (train_dir is None or val_dir is None):
        tried = [
            "Could not resolve PV station CSV directories containing *.csv files.",
            "Train candidates:",
            *[f"  - {path}" for path in train_candidates],
            "Validation/test candidates:",
            *[f"  - {path}" for path in val_candidates],
            "Use --train-pv-dir and --val-pv-dir for a non-standard layout.",
        ]
        raise FileNotFoundError("\n".join(tried))
    return (
        train_dir if train_dir is not None else Path(train_candidates[0]),
        val_dir if val_dir is not None else Path(val_candidates[0]),
    )


def _missing_relative_files(root: Path, required: Iterable[str]) -> list[str]:
    return [name for name in required if not (root / name).is_file()]


def resolve_stage3_data_dir(
    project_root: Path,
    old_data_root: Path | None = None,
    override: Path | None = None,
    *,
    required: Iterable[str] = STAGE3_REPRESENTATION_REQUIRED,
) -> Path:
    """Select a complete Stage-3 data directory from local or old_data."""

    project_root = Path(project_root).expanduser()
    external_roots = _unique_paths(
        [old_data_root, *candidate_old_data_roots(project_root)]
    )
    candidates = _unique_paths(
        [
            override,
            project_root / "stage_3" / "data",
            project_root / "stage3" / "data",
            *[root / "stage3" / "data" for root in external_roots],
            *[root / "stage_3" / "data" for root in external_roots],
        ]
    )
    checks: list[tuple[Path, list[str]]] = []
    for candidate in candidates:
        missing = _missing_relative_files(candidate, required)
        checks.append((candidate, missing))
        if not missing:
            return candidate.resolve()
    lines = [
        "Could not find a complete Stage-3 representation cache.",
        "Checked candidates:",
    ]
    for candidate, missing in checks:
        preview = ", ".join(missing[:3])
        suffix = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        lines.append(f"  - {candidate}: missing {preview}{suffix}")
    raise FileNotFoundError("\n".join(lines))


def resolve_stage1_data_dir(
    stage1_root: Path | None = None,
    *,
    project_root: Path | None = None,
    old_data_root: Path | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve Stage-1 data from project-local or sibling old_data layouts."""

    external_roots = _unique_paths(
        [old_data_root, *candidate_old_data_roots(project_root)]
    )
    candidates = _unique_paths(
        [
            Path(stage1_root) / "data" if stage1_root is not None else None,
            stage1_root,
            Path(project_root) / "stage_1" / "data" if project_root is not None else None,
            Path(project_root) / "stage1" / "data" if project_root is not None else None,
            *[root / "stage1" / "data" for root in external_roots],
            *[root / "stage1" for root in external_roots],
            *[root / "stage_1" / "data" for root in external_roots],
        ]
    )
    for candidate in candidates:
        if (candidate / "02_center_monthly_netcdf").is_dir() or (
            candidate / "03_indices"
        ).is_dir():
            return candidate.resolve()
    if require_exists:
        raise FileNotFoundError(
            "Could not resolve Stage-1 data. Tried:\n"
            + "\n".join(f"  - {path}" for path in candidates)
        )
    return candidates[0]
