#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

from h8_dataset_paths import (
    CENTER_LAT_MAX,
    CENTER_LAT_MIN,
    CENTER_LON_MAX,
    CENTER_LON_MIN,
    CENTER_PROCESSED_ROOT,
    RAW_H8_ROOT,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_VARIANT = "center"
RAW_ROOT = RAW_H8_ROOT
OUT_DIR = CENTER_PROCESSED_ROOT

FILE_GLOB = "**/*.nc"

LAT_MIN = CENTER_LAT_MIN
LAT_MAX = CENTER_LAT_MAX
LON_MIN = CENTER_LON_MIN
LON_MAX = CENTER_LON_MAX

NUM_WORKERS = 4

SELECTED_VARS = [
    "SWR",
    "PAR",
    "UVA",
    "UVB",
    "TAOT_02",
    "TAAE",
    "Sample_number",
]

TARGET_VAR = "SWR"

ALLOW_MISSING_VARS = True
SAVE_VALID_MASK = True
OUTPUT_DTYPE = np.float32

BEIJING_OFFSET_HOURS = 8

# 只保留北京时间文件起始小时在 [6, 18] 的数据
# 例如 filename UTC 23:00 -> BJT 07:00
BJT_START_HOUR_KEEP = 6
BJT_END_HOUR_KEEP = 18

FILENAME_TIME_RE = re.compile(r"H0[89]_(\d{8})_(\d{4})_")


def parse_utc_time_from_filename(path):
    m = FILENAME_TIME_RE.search(path.name)
    if not m:
        return None

    ymd = m.group(1)
    hm = m.group(2)

    yyyy = ymd[0:4]
    mm = ymd[4:6]
    dd = ymd[6:8]
    hh = hm[0:2]
    minute = hm[2:4]

    return np.datetime64(
        f"{yyyy}-{mm}-{dd}T{hh}:{minute}:00"
    ).astype("datetime64[s]")


def to_datetime64_seconds(x):
    return np.datetime64(x).astype("datetime64[s]")


def utc_to_bjt(t_utc):
    return (
        t_utc + np.timedelta64(BEIJING_OFFSET_HOURS, "h")
    ).astype("datetime64[s]")


def get_hour_from_datetime64(t):
    t_h = t.astype("datetime64[h]")
    t_d = t_h.astype("datetime64[D]")
    hour = (
        (t_h - t_d)
        .astype("timedelta64[h]")
        .astype(int)
    )
    return int(hour)


def read_time_from_dataset(ds, fallback_file_time_utc):
    if "start_time" in ds:
        time_start_utc = to_datetime64_seconds(
            ds["start_time"].values[0]
        )
    else:
        time_start_utc = fallback_file_time_utc.astype(
            "datetime64[s]"
        )

    if "end_time" in ds:
        time_end_utc = to_datetime64_seconds(
            ds["end_time"].values[0]
        )
    else:
        time_end_utc = (
            time_start_utc + np.timedelta64(1, "h")
        ).astype("datetime64[s]")

    time_center_utc = (
        time_start_utc + (time_end_utc - time_start_utc) / 2
    ).astype("datetime64[s]")

    time_center_bjt = utc_to_bjt(time_center_utc)
    time_start_bjt = utc_to_bjt(time_start_utc)
    time_file_bjt = utc_to_bjt(fallback_file_time_utc)

    return (
        time_start_utc,
        time_end_utc,
        time_center_utc,
        time_center_bjt,
        time_start_bjt,
        time_file_bjt,
    )


def month_key_from_bjt_time(t_bjt):
    s = np.datetime_as_string(
        t_bjt.astype("datetime64[h]"),
        unit="h"
    )
    return s[0:4] + s[5:7]


def open_h8_dataset(path):
    errors = []

    for engine in ["netcdf4", "h5netcdf", None]:
        try:
            if engine is None:
                return xr.open_dataset(
                    path,
                    mask_and_scale=True
                )

            return xr.open_dataset(
                path,
                engine=engine,
                mask_and_scale=True
            )

        except Exception as e:
            errors.append(f"{engine}: {repr(e)}")

    raise RuntimeError(
        "Failed to open file:\n"
        f"{path}\n\n"
        + "\n".join(errors)
    )


def collect_files_by_bjt_month(raw_root):
    """
    Fast grouping:
    - Do NOT open each NetCDF file during grouping.
    - Use filename UTC time converted to BJT for rough month grouping.
    - Exact center time is read later when processing each file.
    - Filter daytime using filename-BJT hour to reduce I/O.
    """

    files = sorted(raw_root.glob(FILE_GLOB))
    groups = {}

    total = 0
    kept = 0

    for f in files:
        total += 1

        time_file_utc = parse_utc_time_from_filename(f)

        if time_file_utc is None:
            continue

        time_file_bjt = utc_to_bjt(time_file_utc)
        hour_bjt = get_hour_from_datetime64(time_file_bjt)

        if not (
            BJT_START_HOUR_KEEP <= hour_bjt <= BJT_END_HOUR_KEEP
        ):
            continue

        kept += 1

        key = month_key_from_bjt_time(time_file_bjt)

        if key not in groups:
            groups[key] = []

        groups[key].append((time_file_bjt, f))

    sorted_groups = {}

    for key, items in groups.items():
        items = sorted(items, key=lambda x: x[0])
        sorted_groups[key] = [p for _, p in items]

    print(f"Total files scanned: {total}")
    print(
        f"Files kept by BJT filename-hour "
        f"{BJT_START_HOUR_KEEP}:00–{BJT_END_HOUR_KEEP}:00: {kept}"
    )

    return sorted_groups


def crop_dataset(ds):
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    if lat[0] > lat[-1]:
        lat_slice = slice(LAT_MAX, LAT_MIN)
    else:
        lat_slice = slice(LAT_MIN, LAT_MAX)

    if lon[0] > lon[-1]:
        lon_slice = slice(LON_MAX, LON_MIN)
    else:
        lon_slice = slice(LON_MIN, LON_MAX)

    return ds.sel(
        latitude=lat_slice,
        longitude=lon_slice
    )


def read_one_frame(path):
    time_file_utc = parse_utc_time_from_filename(path)

    if time_file_utc is None:
        raise ValueError(
            f"Cannot parse UTC time from {path.name}"
        )

    ds = open_h8_dataset(path)

    (
        time_start_utc,
        time_end_utc,
        time_center_utc,
        time_center_bjt,
        time_start_bjt,
        time_file_bjt,
    ) = read_time_from_dataset(ds, time_file_utc)

    # 二次过滤：用真实 start_time 的北京时间过滤，更严格
    start_hour_bjt = get_hour_from_datetime64(time_start_bjt)

    if not (
        BJT_START_HOUR_KEEP <= start_hour_bjt <= BJT_END_HOUR_KEEP
    ):
        ds.close()
        raise ValueError(
            f"Outside BJT start-hour range: "
            f"{time_start_bjt}, hour={start_hour_bjt}"
        )

    ds_crop = crop_dataset(ds)

    lat = ds_crop["latitude"].values.astype(np.float32)
    lon = ds_crop["longitude"].values.astype(np.float32)

    channel_arrays = []

    for var in SELECTED_VARS:
        if var not in ds_crop.data_vars:
            if ALLOW_MISSING_VARS:
                h = len(lat)
                w = len(lon)
                arr = np.full(
                    (h, w),
                    np.nan,
                    dtype=OUTPUT_DTYPE
                )
            else:
                raise KeyError(f"{var} not found")

        else:
            arr = np.asarray(
                ds_crop[var].values,
                dtype=OUTPUT_DTYPE
            )

            arr = np.squeeze(arr)

            while arr.ndim > 2:
                arr = arr[0]

            if arr.ndim != 2:
                raise ValueError(
                    f"{var} is not 2D after squeeze, shape={arr.shape}"
                )

        channel_arrays.append(arr)

    data = np.stack(
        channel_arrays,
        axis=0
    ).astype(OUTPUT_DTYPE)

    valid_mask = None

    if SAVE_VALID_MASK:
        swr_idx = SELECTED_VARS.index("SWR")
        sample_idx = SELECTED_VARS.index("Sample_number")

        valid = (
            np.isfinite(data[swr_idx]) &
            np.isfinite(data[sample_idx]) &
            (data[sample_idx] > 0)
        )

        valid_mask = valid.astype(np.uint8)

    ds.close()

    return (
        time_center_bjt,
        time_file_utc,
        time_start_utc,
        time_end_utc,
        time_center_utc,
        time_start_bjt,
        time_file_bjt,
        data,
        valid_mask,
        lat,
        lon,
    )


def make_time_features(times_bjt):
    times_h = times_bjt.astype("datetime64[h]")
    dt64_d = times_h.astype("datetime64[D]")

    hour = (
        (times_h - dt64_d)
        .astype("timedelta64[h]")
        .astype(np.int32)
    )

    year = times_h.astype("datetime64[Y]")

    doy = (
        (dt64_d - year.astype("datetime64[D]"))
        .astype("timedelta64[D]")
        .astype(np.int32)
        + 1
    )

    hour_angle = 2 * np.pi * hour / 24
    doy_angle = 2 * np.pi * doy / 366

    return xr.Dataset(
        data_vars={
            "hour_sin_bjt": (
                "time",
                np.sin(hour_angle).astype(np.float32)
            ),
            "hour_cos_bjt": (
                "time",
                np.cos(hour_angle).astype(np.float32)
            ),
            "doy_sin_bjt": (
                "time",
                np.sin(doy_angle).astype(np.float32)
            ),
            "doy_cos_bjt": (
                "time",
                np.cos(doy_angle).astype(np.float32)
            ),
            "hour_bjt": (
                "time",
                hour.astype(np.int16)
            ),
            "doy_bjt": (
                "time",
                doy.astype(np.int16)
            ),
        }
    )


def save_month(month_bjt, files):
    print("\n" + "=" * 80)
    print(f"Processing Beijing month {month_bjt}")
    print("=" * 80)

    times_bjt = []
    times_file_utc = []
    times_start_utc = []
    times_end_utc = []
    times_center_utc = []
    times_start_bjt = []
    times_file_bjt = []

    frames = []
    masks = []

    ref_lat = None
    ref_lon = None

    skipped = 0

    for i, path in enumerate(files):
        try:
            (
                t_bjt,
                t_file_utc,
                t_start_utc,
                t_end_utc,
                t_center_utc,
                t_start_bjt,
                t_file_bjt,
                data,
                valid_mask,
                lat,
                lon,
            ) = read_one_frame(path)

            if ref_lat is None:
                ref_lat = lat
                ref_lon = lon

                print(
                    f"Crop shape: {len(ref_lat)} x {len(ref_lon)}"
                )
                print("Channels:", SELECTED_VARS)

            times_bjt.append(t_bjt)
            times_file_utc.append(t_file_utc)
            times_start_utc.append(t_start_utc)
            times_end_utc.append(t_end_utc)
            times_center_utc.append(t_center_utc)
            times_start_bjt.append(t_start_bjt)
            times_file_bjt.append(t_file_bjt)

            frames.append(data)

            if SAVE_VALID_MASK:
                masks.append(valid_mask)

        except Exception as e:
            skipped += 1
            print(f"[SKIP] {path}")
            print(repr(e))

        if (i + 1) % 50 == 0:
            print(f"read {i + 1}/{len(files)}")

    if len(frames) == 0:
        print("No valid frames")
        return None

    times_bjt = np.asarray(times_bjt).astype("datetime64[s]")
    times_file_utc = np.asarray(times_file_utc).astype("datetime64[s]")
    times_start_utc = np.asarray(times_start_utc).astype("datetime64[s]")
    times_end_utc = np.asarray(times_end_utc).astype("datetime64[s]")
    times_center_utc = np.asarray(times_center_utc).astype("datetime64[s]")
    times_start_bjt = np.asarray(times_start_bjt).astype("datetime64[s]")
    times_file_bjt = np.asarray(times_file_bjt).astype("datetime64[s]")

    order = np.argsort(times_bjt)

    times_bjt = times_bjt[order]
    times_file_utc = times_file_utc[order]
    times_start_utc = times_start_utc[order]
    times_end_utc = times_end_utc[order]
    times_center_utc = times_center_utc[order]
    times_start_bjt = times_start_bjt[order]
    times_file_bjt = times_file_bjt[order]

    data_all = np.stack(frames, axis=0)[order]

    _, unique_idx = np.unique(
        times_bjt,
        return_index=True
    )
    unique_idx = np.sort(unique_idx)

    if len(unique_idx) != len(times_bjt):
        print(
            f"[WARN] duplicate center times removed: "
            f"{len(times_bjt) - len(unique_idx)}"
        )

        times_bjt = times_bjt[unique_idx]
        times_file_utc = times_file_utc[unique_idx]
        times_start_utc = times_start_utc[unique_idx]
        times_end_utc = times_end_utc[unique_idx]
        times_center_utc = times_center_utc[unique_idx]
        times_start_bjt = times_start_bjt[unique_idx]
        times_file_bjt = times_file_bjt[unique_idx]
        data_all = data_all[unique_idx]

    ds_out = xr.Dataset(
        data_vars={
            "data": (
                (
                    "time",
                    "channel",
                    "latitude",
                    "longitude"
                ),
                data_all.astype(OUTPUT_DTYPE)
            ),

            "time_file_utc": (
                "time",
                times_file_utc
            ),

            "time_start_utc": (
                "time",
                times_start_utc
            ),

            "time_end_utc": (
                "time",
                times_end_utc
            ),

            "time_center_utc": (
                "time",
                times_center_utc
            ),

            "time_file_bjt": (
                "time",
                times_file_bjt
            ),

            "time_start_bjt": (
                "time",
                times_start_bjt
            ),
        },

        coords={
            # 主时间轴：平均窗口中心时刻，北京时间
            "time": times_bjt,

            "channel": np.asarray(
                SELECTED_VARS,
                dtype=object
            ),

            "latitude": ref_lat,
            "longitude": ref_lon,
        },

        attrs={
            "description":
                "Regional H8/H9 multivariable daytime dataset",

            "time_coordinate":
                "Beijing time UTC+8, using center time of averaging window",

            "daytime_filter":
                f"Keep files with Beijing start hour "
                f"{BJT_START_HOUR_KEEP}:00–{BJT_END_HOUR_KEEP}:00",

            "time_file_utc":
                "UTC time parsed from filename",

            "time_start_utc":
                "observation start time from NetCDF",

            "time_end_utc":
                "observation end time from NetCDF",

            "time_center_utc":
                "(time_start_utc + time_end_utc) / 2",

            "time_start_bjt":
                "observation start time converted to Beijing time",

            "time_file_bjt":
                "filename time converted to Beijing time",

            "selected_vars":
                ",".join(SELECTED_VARS),

            "target_var":
                TARGET_VAR,

            "history_hours":
                3,

            "future_hours":
                2,

            "crop_lat":
                f"{LAT_MIN} to {LAT_MAX}",

            "crop_lon":
                f"{LON_MIN} to {LON_MAX}",

            "skipped_files":
                skipped,
        }
    )

    if SAVE_VALID_MASK:
        mask_all = np.stack(masks, axis=0)[order]

        if len(unique_idx) != len(order):
            mask_all = mask_all[unique_idx]

        ds_out["valid_mask"] = (
            (
                "time",
                "latitude",
                "longitude"
            ),
            mask_all.astype(np.uint8)
        )

    tf = make_time_features(times_bjt)

    for name in tf.data_vars:
        ds_out[name] = tf[name]

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = month_output_path(month_bjt)

    encoding = {
        "data": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (
                24,
                len(SELECTED_VARS),
                len(ref_lat),
                len(ref_lon)
            ),
        },

        "valid_mask": {
            "zlib": True,
            "complevel": 4,
            "dtype": "uint8",
            "chunksizes": (
                24,
                len(ref_lat),
                len(ref_lon)
            ),
        },
    }

    print("Saving:", out_path)
    print("Final shape:", data_all.shape)
    print("BJT center time:", times_bjt[0], "->", times_bjt[-1])
    print("BJT start time:", times_start_bjt[0], "->", times_start_bjt[-1])
    print("UTC file time:", times_file_utc[0], "->", times_file_utc[-1])
    print("UTC start time:", times_start_utc[0], "->", times_start_utc[-1])
    print("UTC end time:", times_end_utc[0], "->", times_end_utc[-1])
    print("Skipped:", skipped)

    try:
        ds_out.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding=encoding
        )

    except Exception as e:
        print("[WARN] netcdf4 save failed:", repr(e))
        print("[INFO] trying h5netcdf")

        ds_out.to_netcdf(
            out_path,
            engine="h5netcdf",
            encoding=encoding
        )

    ds_out.close()

    return out_path


def month_output_path(month_bjt):
    return OUT_DIR / (
        f"H8_region_multivar_"
        f"{month_bjt}_CENTER_BJT_"
        f"daytime_{BJT_START_HOUR_KEEP:02d}_{BJT_END_HOUR_KEEP:02d}_"
        f"lat{LAT_MIN:g}_{LAT_MAX:g}_"
        f"lon{LON_MIN:g}_{LON_MAX:g}.nc"
    )


def runtime_config():
    return {
        "DATASET_VARIANT": DATASET_VARIANT,
        "RAW_ROOT": RAW_ROOT,
        "OUT_DIR": OUT_DIR,
        "LAT_MIN": LAT_MIN,
        "LAT_MAX": LAT_MAX,
        "LON_MIN": LON_MIN,
        "LON_MAX": LON_MAX,
    }


def apply_runtime_config(config):
    globals()["DATASET_VARIANT"] = config["DATASET_VARIANT"]
    globals()["RAW_ROOT"] = config["RAW_ROOT"]
    globals()["OUT_DIR"] = config["OUT_DIR"]
    globals()["LAT_MIN"] = config["LAT_MIN"]
    globals()["LAT_MAX"] = config["LAT_MAX"]
    globals()["LON_MIN"] = config["LON_MIN"]
    globals()["LON_MAX"] = config["LON_MAX"]


def save_month_worker(task):
    config, month_bjt, files = task
    apply_runtime_config(config)
    return save_month(month_bjt, files)


def save_metadata(saved_files):
    meta = {
        "selected_vars": SELECTED_VARS,
        "target_var": TARGET_VAR,

        "crop": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
        },

        "daytime_filter": {
            "timezone": "Beijing time UTC+8",
            "start_hour_keep": BJT_START_HOUR_KEEP,
            "end_hour_keep": BJT_END_HOUR_KEEP,
            "note": "Filtering uses filename-BJT hour first, then NetCDF start_time-BJT hour during reading.",
        },

        "time_coordinate":
            "Beijing time UTC+8, using center time of averaging window",

        "time_variables": {
            "time": "center time in Beijing time UTC+8",
            "time_file_utc": "time parsed from filename",
            "time_start_utc": "start_time inside NetCDF",
            "time_end_utc": "end_time inside NetCDF",
            "time_center_utc": "(time_start_utc + time_end_utc) / 2",
            "time_file_bjt": "filename time converted to Beijing time",
            "time_start_bjt": "start_time converted to Beijing time",
        },

        "task": {
            "history_hours": 3,
            "future_hours": 2,
            "X": "[t-2h, t-1h, t]",
            "Y": "[t+1h, t+2h]",
            "actual_modeling_hours": "usually 07:00–17:00 BJT",
            "saved_hours": "06:00–18:00 BJT for margin",
        },

        "saved_files": [
            str(p) for p in saved_files
        ],
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = OUT_DIR / "H8_region_multivar_metadata.json"

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved metadata:", out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build monthly regional H8 processed NetCDF files.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional BJT months to process, e.g. --months 202303 202304 202305 202306.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("DATASET_VARIANT:", DATASET_VARIANT)
    print("RAW_ROOT:", RAW_ROOT)
    print("OUT_DIR:", OUT_DIR)

    print("Crop:")
    print(f"  lat: {LAT_MIN} to {LAT_MAX}")
    print(f"  lon: {LON_MIN} to {LON_MAX}")

    print("Channels:", SELECTED_VARS)

    print("Time handling:")
    print("  filename time: UTC")
    print("  NetCDF start_time/end_time: UTC")
    print("  main coordinate time: center time in Beijing time UTC+8")

    print("Daytime filtering:")
    print(
        f"  keep Beijing start-hour "
        f"{BJT_START_HOUR_KEEP}:00–{BJT_END_HOUR_KEEP}:00"
    )

    groups = collect_files_by_bjt_month(RAW_ROOT)
    if args.months:
        keep_months = {str(m) for m in args.months}
        groups = {k: v for k, v in groups.items() if str(k) in keep_months}

    if len(groups) == 0:
        print("No files found")
        return

    print("\nFound months:")

    for k in sorted(groups.keys()):
        print(f"{k}: {len(groups[k])} files")

    saved_files = []
    tasks = []

    for month_bjt in sorted(groups.keys()):
        expected_path = month_output_path(month_bjt)
        if bool(args.skip_existing) and expected_path.exists():
            print(f"[SKIP existing] {expected_path}")
            saved_files.append(expected_path)
            continue
        tasks.append((runtime_config(), month_bjt, groups[month_bjt]))

    num_workers = max(1, int(args.num_workers))
    print("Month workers:", num_workers)
    if num_workers == 1 or len(tasks) <= 1:
        for task in tasks:
            out_path = save_month_worker(task)
            if out_path is not None:
                saved_files.append(out_path)
    else:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as executor:
            futures = {executor.submit(save_month_worker, task): task[1] for task in tasks}
            for future in as_completed(futures):
                month_bjt = futures[future]
                try:
                    out_path = future.result()
                    if out_path is not None:
                        saved_files.append(out_path)
                except Exception as exc:
                    print(f"[ERROR] month {month_bjt} failed: {repr(exc)}")
                    raise

    saved_files = sorted(saved_files, key=lambda p: str(p))

    save_metadata(saved_files)

    print("\nDone.")


if __name__ == "__main__":
    main()
