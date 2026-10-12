"""Build leakage-safe 4-history -> 3-future indices for Stage 1.

Every sample is kept inside one BJT calendar day and uses only 06:00-17:00.
Train, validation and test remain date-based. Spatial augmentation is applied
to train rows only in the main big-patch index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from stage1_paths import (
    BIG_INDEX_ALL_PATCH,
    BIG_INDEX_DIR,
    BIG_INDEX_MAIN,
    BIG_MONTHLY_DIR,
    CENTER_INDEX_ALL,
    CENTER_INDEX_DIR,
    CENTER_INDEX_TEST,
    CENTER_INDEX_TRAIN,
    CENTER_INDEX_VAL,
    CENTER_MONTHLY_DIR,
    PATCH_BANK_CSV,
    ensure_stage1_dirs,
)


HISTORY_FRAMES = 4
FUTURE_FRAMES = 3
TOTAL_FRAMES = HISTORY_FRAMES + FUTURE_FRAMES
VALID_HOUR_MIN = 6
VALID_HOUR_MAX = 17
MIN_GAP_MINUTES = 45
MAX_GAP_MINUTES = 75

DATA_TIME_START = pd.Timestamp("2022-01-01 07:00:00")
DATA_TIME_END = pd.Timestamp("2023-06-24 16:45:00")
TRAIN_END = pd.Timestamp("2023-01-13 23:59:59")
VAL_END = pd.Timestamp("2023-03-08 23:59:59")

SPLIT_RANGES = {
    "train": (DATA_TIME_START, TRAIN_END),
    "val": (TRAIN_END + pd.Timedelta(seconds=1), VAL_END),
    "test": (VAL_END + pd.Timedelta(seconds=1), DATA_TIME_END),
}
SPLIT_DAY_RANGES = {
    "train": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-13")),
    "val": (pd.Timestamp("2023-01-14"), pd.Timestamp("2023-03-08")),
    "test": (pd.Timestamp("2023-03-09"), pd.Timestamp("2023-06-24")),
}

BIG_LAT_MIN = 18.0
BIG_LAT_MAX = 42.0
BIG_LON_MIN = 105.5
BIG_LON_MAX = 137.0
CENTER_LAT_MIN = 26.0
CENTER_LAT_MAX = 34.0
CENTER_LON_MIN = 116.0
CENTER_LON_MAX = 126.5
PATCH_LAT_SIZE = 8.0
PATCH_LON_SIZE = 10.5
PATCH_LAT_STRIDE = 4.0
PATCH_LON_STRIDE = 5.25


def split_for_sample_day(timestamp: pd.Timestamp) -> str | None:
    day = timestamp.normalize()
    for split, (start_day, end_day) in SPLIT_DAY_RANGES.items():
        if start_day <= day <= end_day:
            return split
    return None


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in values
    ]


def collect_temporal_rows(monthly_dir: Path, source_variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nc_files = sorted(monthly_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No monthly NetCDF files: {monthly_dir}")

    for nc_path in nc_files:
        with xr.open_dataset(nc_path) as ds:
            times = pd.to_datetime(ds["time"].values)
            bjt_times = pd.to_datetime(ds["time_start_bjt"].values)

        count_before = len(rows)
        for start_idx in range(len(times) - TOTAL_FRAMES + 1):
            indices = np.arange(start_idx, start_idx + TOTAL_FRAMES)
            sample_bjt = bjt_times[indices]
            sample_center = times[indices]
            if len({stamp.date() for stamp in sample_bjt}) != 1:
                continue
            if len({stamp.date() for stamp in sample_center}) != 1:
                continue
            if not all(VALID_HOUR_MIN <= stamp.hour <= VALID_HOUR_MAX for stamp in sample_bjt):
                continue
            gaps = np.diff(sample_bjt.values).astype("timedelta64[m]").astype(int)
            if not np.all((gaps >= MIN_GAP_MINUTES) & (gaps <= MAX_GAP_MINUTES)):
                continue

            day = pd.Timestamp(sample_bjt[0].date())
            split = split_for_sample_day(pd.Timestamp(sample_center[0]))
            if split is None:
                continue
            if split_for_sample_day(pd.Timestamp(sample_center[-1])) != split:
                continue

            x_indices = indices[:HISTORY_FRAMES]
            y_indices = indices[HISTORY_FRAMES:]
            window_key = (
                f"{day:%Y%m%d}_"
                f"{sample_bjt[0]:%H%M}_"
                f"{sample_bjt[-1]:%H%M}"
            )
            row: dict[str, object] = {
                "source_variant": source_variant,
                "split": split,
                "sample_day_bjt": day.strftime("%Y-%m-%d"),
                "window_key": window_key,
                "history_frames": HISTORY_FRAMES,
                "future_frames": FUTURE_FRAMES,
                "nc_path": str(nc_path.resolve()),
                "x_start_idx": int(x_indices[0]),
                "x_end_idx": int(x_indices[-1]),
                "y_start_idx": int(y_indices[0]),
                "y_end_idx": int(y_indices[-1]),
                "issue_bjt": sample_bjt[HISTORY_FRAMES - 1].isoformat(),
                "issue_time_bjt": sample_bjt[HISTORY_FRAMES - 1].isoformat(),
            }
            for offset, index in enumerate(x_indices):
                row[f"x{offset}_idx"] = int(index)
                row[f"x{offset}_bjt"] = sample_bjt[index - start_idx].isoformat()
                row[f"x{offset}_center_bjt"] = sample_center[index - start_idx].isoformat()
                row[f"x_time_{offset}_bjt"] = sample_bjt[index - start_idx].isoformat()
            for offset, index in enumerate(y_indices):
                row[f"y{offset}_idx"] = int(index)
                row[f"y{offset}_bjt"] = sample_bjt[index - start_idx].isoformat()
                row[f"y{offset}_center_bjt"] = sample_center[index - start_idx].isoformat()
                row[f"y_time_{offset}_bjt"] = sample_bjt[index - start_idx].isoformat()
            rows.append(row)
        print(f"{nc_path.name}: temporal samples={len(rows) - count_before}")

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No valid temporal rows built from {monthly_dir}")
    assert_leakage_safe(result)
    return result


def assert_leakage_safe(df: pd.DataFrame) -> None:
    if df.duplicated(subset=["window_key"]).any():
        duplicates = df.loc[df.duplicated(subset=["window_key"], keep=False), "window_key"]
        raise AssertionError(f"Duplicate temporal windows found: {duplicates.head().tolist()}")
    for _, row in df.iterrows():
        stamps = [
            pd.Timestamp(row[f"x{index}_bjt"]) for index in range(HISTORY_FRAMES)
        ] + [pd.Timestamp(row[f"y{index}_bjt"]) for index in range(FUTURE_FRAMES)]
        if len({stamp.date() for stamp in stamps}) != 1:
            raise AssertionError(f"Cross-day sample found: {row['window_key']}")
        if not all(VALID_HOUR_MIN <= stamp.hour <= VALID_HOUR_MAX for stamp in stamps):
            raise AssertionError(f"Hour outside 06:00-17:00: {row['window_key']}")
        center_stamps = [
            pd.Timestamp(row[f"x{index}_center_bjt"]) for index in range(HISTORY_FRAMES)
        ] + [
            pd.Timestamp(row[f"y{index}_center_bjt"]) for index in range(FUTURE_FRAMES)
        ]
        expected_split = split_for_sample_day(center_stamps[0])
        if row["split"] != expected_split:
            raise AssertionError(f"Split mismatch: {row['window_key']}")
        if split_for_sample_day(center_stamps[-1]) != expected_split:
            raise AssertionError(f"Sample crosses split boundary: {row['window_key']}")

    split_days = {
        split: set(df.loc[df["split"] == split, "sample_day_bjt"]) for split in SPLIT_RANGES
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_days[left] & split_days[right]
        if overlap:
            raise AssertionError(f"Date leakage between {left} and {right}: {overlap}")


def save_center_indices(center_rows: pd.DataFrame) -> None:
    center_rows.to_csv(CENTER_INDEX_ALL, index=False)
    for split, path in (
        ("train", CENTER_INDEX_TRAIN),
        ("val", CENTER_INDEX_VAL),
        ("test", CENTER_INDEX_TEST),
    ):
        center_rows.loc[center_rows["split"] == split].to_csv(path, index=False)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(values - target).argmin())


def make_patch_bank(big_sample_nc: Path) -> pd.DataFrame:
    with xr.open_dataset(big_sample_nc) as ds:
        latitudes = np.asarray(ds["latitude"].values)
        longitudes = np.asarray(ds["longitude"].values)

    lat_starts = np.arange(BIG_LAT_MIN, BIG_LAT_MAX - PATCH_LAT_SIZE + 0.01, PATCH_LAT_STRIDE)
    lon_starts = np.arange(BIG_LON_MIN, BIG_LON_MAX - PATCH_LON_SIZE + 0.01, PATCH_LON_STRIDE)
    rows: list[dict[str, object]] = []
    patch_id = 0
    for row_index, lat_min in enumerate(lat_starts):
        for col_index, lon_min in enumerate(lon_starts):
            lat_max = lat_min + PATCH_LAT_SIZE
            lon_max = lon_min + PATCH_LON_SIZE
            y0 = min(nearest_index(latitudes, lat_min), nearest_index(latitudes, lat_max))
            y1 = max(nearest_index(latitudes, lat_min), nearest_index(latitudes, lat_max)) + 1
            x0 = min(nearest_index(longitudes, lon_min), nearest_index(longitudes, lon_max))
            x1 = max(nearest_index(longitudes, lon_min), nearest_index(longitudes, lon_max)) + 1
            is_center = (
                np.isclose(lat_min, CENTER_LAT_MIN)
                and np.isclose(lon_min, CENTER_LON_MIN)
            )
            rows.append(
                {
                    "patch_id": patch_id,
                    "patch_row": row_index,
                    "patch_col": col_index,
                    "lat_min": round(float(lat_min), 4),
                    "lat_max": round(float(lat_max), 4),
                    "lon_min": round(float(lon_min), 4),
                    "lon_max": round(float(lon_max), 4),
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "lat_start_idx": y0,
                    "lat_end_idx": y1 - 1,
                    "lon_start_idx": x0,
                    "lon_end_idx": x1 - 1,
                    "is_center": bool(is_center),
                }
            )
            patch_id += 1

    patch_bank = pd.DataFrame(rows)
    if len(patch_bank) != 25 or int(patch_bank["is_center"].sum()) != 1:
        raise AssertionError("Expected exactly 25 patches and one center patch")
    return patch_bank


def expand_spatial(
    temporal_rows: pd.DataFrame, patch_bank: pd.DataFrame, train_all_patch: bool
) -> pd.DataFrame:
    center_patch = patch_bank.loc[patch_bank["is_center"]].iloc[0]
    expanded: list[dict[str, object]] = []
    for temporal_row in temporal_rows.to_dict("records"):
        patches = (
            patch_bank.to_dict("records")
            if train_all_patch or temporal_row["split"] == "train"
            else [center_patch.to_dict()]
        )
        for patch in patches:
            expanded.append({**temporal_row, **patch})
    return pd.DataFrame(expanded)


def save_metadata(center_rows: pd.DataFrame, big_rows: pd.DataFrame) -> None:
    metadata = {
        "history_frames": HISTORY_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "valid_bjt_hours_inclusive": [VALID_HOUR_MIN, VALID_HOUR_MAX],
        "same_bjt_day_required": True,
        "temporal_windows_per_complete_day": 6,
        "dataset_time_start": str(DATA_TIME_START),
        "dataset_time_end": str(DATA_TIME_END),
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
        "split_ranges": {
            split: [str(start), str(end)] for split, (start, end) in SPLIT_RANGES.items()
        },
        "split_day_ranges_for_h8_windows": {
            split: [str(start.date()), str(end.date())]
            for split, (start, end) in SPLIT_DAY_RANGES.items()
        },
        "center_temporal_counts": center_rows["split"].value_counts().to_dict(),
        "big_temporal_counts": big_rows["split"].value_counts().to_dict(),
        "big_main_policy": "train=25 overlap patches; val/test=center patch only",
        "patch_overlap": "50 percent in both latitude and longitude directions",
    }
    with (BIG_INDEX_DIR / "stage1_index_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def assert_center_big_temporal_match(
    center_rows: pd.DataFrame, big_rows: pd.DataFrame
) -> None:
    center_keys = set(center_rows["window_key"])
    big_keys = set(big_rows["window_key"])
    if center_keys != big_keys:
        only_center = sorted(center_keys - big_keys)[:5]
        only_big = sorted(big_keys - center_keys)[:5]
        raise AssertionError(
            "Center and big temporal windows differ. "
            f"only_center={only_center}, only_big={only_big}"
        )


def main() -> None:
    ensure_stage1_dirs()
    print("Building center temporal index: 4 history -> 3 future, BJT 06:00-17:00")
    center_rows = collect_temporal_rows(CENTER_MONTHLY_DIR, "stage1_center_crop_from_big")
    save_center_indices(center_rows)

    print("Building big temporal index: 4 history -> 3 future, BJT 06:00-17:00")
    big_rows = collect_temporal_rows(BIG_MONTHLY_DIR, "stage1_big_region_single_source")
    assert_leakage_safe(big_rows)
    assert_center_big_temporal_match(center_rows, big_rows)

    patch_bank = make_patch_bank(next(iter(sorted(BIG_MONTHLY_DIR.glob("*.nc")))))
    patch_bank.to_csv(PATCH_BANK_CSV, index=False)
    expand_spatial(big_rows, patch_bank, train_all_patch=False).to_csv(
        BIG_INDEX_MAIN, index=False
    )
    expand_spatial(big_rows, patch_bank, train_all_patch=True).to_csv(
        BIG_INDEX_ALL_PATCH, index=False
    )
    save_metadata(center_rows, big_rows)

    print("\nStage 1 indices complete")
    print(f"center index: {CENTER_INDEX_ALL}")
    print(f"patch bank: {PATCH_BANK_CSV}")
    print(f"big main index: {BIG_INDEX_MAIN}")
    print("center counts:", center_rows["split"].value_counts().to_dict())
    print("big temporal counts:", big_rows["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
