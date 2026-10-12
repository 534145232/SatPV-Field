"""Low-memory monthly NetCDF writer for the Stage 1 big H8 source."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


EPOCH = np.datetime64("1970-01-01T00:00:00", "s")


def seconds_since_epoch(values: list[np.datetime64] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values).astype("datetime64[s]")
    return (arr - EPOCH).astype("timedelta64[s]").astype(np.int64)


def collect_valid_frames(base, files: list[Path]) -> tuple[list[dict[str, object]], int]:
    records_by_center: dict[int, dict[str, object]] = {}
    skipped = 0
    for index, path in enumerate(files, start=1):
        ds = None
        try:
            file_utc = base.parse_utc_time_from_filename(path)
            if file_utc is None:
                raise ValueError(f"Cannot parse UTC time from filename: {path.name}")
            ds = base.open_h8_dataset(path)
            (
                start_utc,
                end_utc,
                center_utc,
                center_bjt,
                start_bjt,
                file_bjt,
            ) = base.read_time_from_dataset(ds, file_utc)
            start_hour_bjt = base.get_hour_from_datetime64(start_bjt)
            if not (base.BJT_START_HOUR_KEEP <= start_hour_bjt <= base.BJT_END_HOUR_KEEP):
                raise ValueError(f"Outside BJT start-hour range: {start_bjt}")
            center_key = int(seconds_since_epoch([center_bjt])[0])
            records_by_center.setdefault(
                center_key,
                {
                    "path": path,
                    "time": center_bjt,
                    "time_file_utc": file_utc,
                    "time_start_utc": start_utc,
                    "time_end_utc": end_utc,
                    "time_center_utc": center_utc,
                    "time_file_bjt": file_bjt,
                    "time_start_bjt": start_bjt,
                },
            )
        except Exception as exc:
            skipped += 1
            print(f"[SKIP time scan] {path}: {exc!r}", flush=True)
        finally:
            if ds is not None:
                ds.close()
        if index % 100 == 0 or index == len(files):
            print(f"time scan {index}/{len(files)}", flush=True)
    records = [records_by_center[key] for key in sorted(records_by_center)]
    return records, skipped


def create_time_variable(ds_out, name: str, values: list[np.datetime64]) -> None:
    variable = ds_out.createVariable(name, "i8", ("time",))
    variable.units = "seconds since 1970-01-01 00:00:00"
    variable.calendar = "proleptic_gregorian"
    variable[:] = seconds_since_epoch(values)


def is_complete_month(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        from netCDF4 import Dataset

        with Dataset(path, mode="r") as ds:
            return (
                str(getattr(ds, "stage1_write_complete", "")).lower() == "true"
                and "data" in ds.variables
                and "valid_mask" in ds.variables
                and len(ds.dimensions["time"]) > 0
            )
    except Exception:
        return False


def save_month_low_memory(base, month_bjt: str, files: list[Path]) -> Path | None:
    from netCDF4 import Dataset

    print("\n" + "=" * 80)
    print(f"Low-memory processing Beijing month {month_bjt}")
    print("=" * 80)
    records, skipped = collect_valid_frames(base, files)
    if not records:
        print("No valid frames")
        return None

    first = base.read_one_frame(records[0]["path"])
    _, _, _, _, _, _, _, first_data, first_mask, latitudes, longitudes = first
    num_frames = len(records)
    num_channels = len(base.SELECTED_VARS)
    print(f"frames: {num_frames}")
    print(f"crop shape: {len(latitudes)} x {len(longitudes)}")
    print("channels:", base.SELECTED_VARS)

    out_path = base.month_output_path(month_bjt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with Dataset(tmp_path, mode="w", format="NETCDF4") as ds_out:
            ds_out.createDimension("time", num_frames)
            ds_out.createDimension("channel", num_channels)
            ds_out.createDimension("latitude", len(latitudes))
            ds_out.createDimension("longitude", len(longitudes))

            channel_var = ds_out.createVariable("channel", str, ("channel",))
            channel_var[:] = np.asarray(base.SELECTED_VARS, dtype=object)
            ds_out.createVariable("latitude", "f4", ("latitude",))[:] = latitudes
            ds_out.createVariable("longitude", "f4", ("longitude",))[:] = longitudes

            for name in (
                "time",
                "time_file_utc",
                "time_start_utc",
                "time_end_utc",
                "time_center_utc",
                "time_file_bjt",
                "time_start_bjt",
            ):
                create_time_variable(ds_out, name, [record[name] for record in records])

            data_var = ds_out.createVariable(
                "data",
                "f4",
                ("time", "channel", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                chunksizes=(1, num_channels, len(latitudes), len(longitudes)),
            )
            mask_var = ds_out.createVariable(
                "valid_mask",
                "u1",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                chunksizes=(1, len(latitudes), len(longitudes)),
            )

            center_times = np.asarray([record["time"] for record in records]).astype("datetime64[s]")
            date_days = center_times.astype("datetime64[D]")
            hour = (center_times.astype("datetime64[h]") - date_days).astype("timedelta64[h]").astype(int)
            year_start = center_times.astype("datetime64[Y]")
            doy = (date_days - year_start.astype("datetime64[D]")).astype(int) + 1
            hour_angle = 2.0 * np.pi * hour / 24.0
            doy_angle = 2.0 * np.pi * (doy - 1) / 365.25
            for name, values, dtype in (
                ("hour_sin_bjt", np.sin(hour_angle), "f4"),
                ("hour_cos_bjt", np.cos(hour_angle), "f4"),
                ("doy_sin_bjt", np.sin(doy_angle), "f4"),
                ("doy_cos_bjt", np.cos(doy_angle), "f4"),
                ("hour_bjt", hour, "i2"),
                ("doy_bjt", doy, "i2"),
            ):
                ds_out.createVariable(name, dtype, ("time",))[:] = np.asarray(values)

            for row_index, record in enumerate(records):
                if row_index == 0:
                    data = first_data
                    valid_mask = first_mask
                else:
                    result = base.read_one_frame(record["path"])
                    data = result[7]
                    valid_mask = result[8]
                data_var[row_index] = np.asarray(data, dtype=np.float32)
                mask_var[row_index] = np.asarray(valid_mask, dtype=np.uint8)
                if (row_index + 1) % 20 == 0 or row_index + 1 == num_frames:
                    print(f"written {row_index + 1}/{num_frames}", flush=True)

            ds_out.description = "Stage 1 regional H8/H9 multivariable daytime dataset"
            ds_out.dataset_variant = "stage1_big_region_single_source_low_memory"
            ds_out.selected_vars = ",".join(base.SELECTED_VARS)
            ds_out.target_var = base.TARGET_VAR
            ds_out.crop_lat = f"{base.LAT_MIN} to {base.LAT_MAX}"
            ds_out.crop_lon = f"{base.LON_MIN} to {base.LON_MAX}"
            ds_out.daytime_filter = (
                f"Keep files with Beijing start hour "
                f"{base.BJT_START_HOUR_KEEP}:00-{base.BJT_END_HOUR_KEEP}:00"
            )
            ds_out.skipped_files = skipped
            ds_out.low_memory_writer = "two-pass scan followed by frame-wise NetCDF write"
            ds_out.stage1_write_complete = "true"

        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    print("saved:", out_path)
    return out_path


def save_stage1_metadata(base, saved_files: list[Path]) -> None:
    metadata = {
        "dataset_variant": "stage1_big_region_single_source_low_memory",
        "writer": "two-pass scan followed by frame-wise NetCDF write",
        "selected_vars": base.SELECTED_VARS,
        "target_var": base.TARGET_VAR,
        "crop": {
            "lat_min": base.LAT_MIN,
            "lat_max": base.LAT_MAX,
            "lon_min": base.LON_MIN,
            "lon_max": base.LON_MAX,
        },
        "saved_bjt_start_hours_inclusive": [
            base.BJT_START_HOUR_KEEP,
            base.BJT_END_HOUR_KEEP,
        ],
        "modeling_index_note": "03_build_indices.py selects same-day 4+3 windows using BJT start hours 06:00-17:00.",
        "saved_files": [str(path) for path in sorted(saved_files, key=str)],
    }
    out_path = base.OUT_DIR / "H8_region_multivar_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print("saved metadata:", out_path)
