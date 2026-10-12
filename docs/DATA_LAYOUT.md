# Data layout

The repository does not distribute operational PV data, satellite products, NWP files, cached tensors, checkpoints, or result tables. This document specifies the interfaces expected by the code.

## Recommended external layout

With `SATPV_DATA_ROOT=/path/to/satpv_artifacts`, the generated and reusable artifacts follow:

```text
/path/to/satpv_artifacts/
├── station_split.csv
├── PV/
│   ├── train_directory_cleaned_csv_same_day06_nwp/
│   │   └── <station_sn>.csv
│   └── val_directory_cleaned_csv_same_day06_nwp/
│       └── <station_sn>.csv
├── NWP/
│   └── old_NWP/
│       ├── farm_basic_infomation.xlsx
│       └── basic_info_500_with_station.csv
├── stage1/
│   ├── data/       # generated H8 NetCDF, indices, and caches
│   └── runs/
├── stage2/
│   └── runs/       # optional relocated Stage 2 checkpoints
├── stage3/
│   ├── data/       # station split, target list, PV fields, multisource caches
│   └── runs/       # field-model checkpoints and evaluations
└── stage4/
    ├── data/
    └── runs/       # target-station models and evaluation details
```

Raw H8 files can be stored anywhere and are located through `H8_RAW_ROOT`.

## Station metadata

`STATION_SPLIT_CSV` must contain one row per station and at least:

| Column | Meaning |
|---|---|
| `station_sn` | station identifier; treated as a string |
| `lon` | longitude in decimal degrees |
| `lat` | latitude in decimal degrees |

The paper experiment starts from 496 usable stations. Stage 3 selects 50 held-out targets and constructs the regional field from the remaining 446 sources.

## Station PV/NWP CSV files

Each station file must be named `<station_sn>.csv`. The loader accepts aliases, but a portable canonical schema is:

```text
date,elec,Capacity,radiation,pressure,humidity,temperature,
direction10,speed10,tcc,solar_zenith,clearsky_ghi
```

Required fields:

- timestamp: `date`, `datetime`, `timestamp`, `time`, or a documented alias;
- PV power: `elec`, `pv`, `power`, `active_power`, or a documented alias;
- installed capacity: `Capacity`, `capacity`, or `cap`;
- future/station conditions: radiation, pressure, humidity, temperature, 10-m wind direction and speed, total cloud cover, solar zenith, and clear-sky GHI.

The code computes station-level per-unit PV as:

```text
raw PV / (Capacity * 1000)
```

The default `capacity_scale=1000` assumes capacity is stored in MW while power is stored in kW. If your units differ, pass the correct `--capacity_scale` explicitly.

Timestamps must support the 15-min history/future windows. Duplicate timestamps are removed after parsing.

## Himawari-8 NetCDF input

Stage 1 searches `H8_RAW_ROOT` recursively using `**/*.nc`. The processed representation expects:

- dimensions/coordinates compatible with `time`, `channel`, `latitude`, and `longitude`;
- a `data` variable;
- channels `SWR`, `PAR`, `UVA`, `UVB`, `TAOT_02`, `TAAE`, and `Sample_number`;
- a `valid_mask` variable where available.

The Stage 2 ratio model uses the first six physical channels and excludes `Sample_number` from the learned input.

## Regional NWP metadata

`farm_basic_infomation.xlsx` supplies the NWP-point identifier and coordinates. Accepted column aliases include:

- point id: `NO`, `farm_id`, `station`, or `station_id`;
- longitude: `lon` or `longitude`;
- latitude: `lat` or `latitude`.

`basic_info_500_with_station.csv` maps PV stations to their nearest NWP point. It must contain a station id (`SN`/`station_sn`) and a nearest-point id (`nearest_station_NO`, `nearest_station`, `nwp_station`, or `NWP`).

The regional cache uses `radiation`, `direction10`, `speed10`, and `tcc`, including their supported `fore_*`/`nwp_*` aliases.

## Generated Stage 3 artifacts

The full cache build creates:

```text
stage3/data/
├── 00_source_regional_split/station_split.csv
├── 01_multisource_nwp_cache/
├── 02_target_selection_lhs50/
├── 03_target_excluded_run/
├── 04_target_excluded_pvnorm_frame_cache/
└── 05_center_h8_swr_anchor_cache/
```

Do not mix caches created from different station lists, split boundaries, or SWR checkpoints. The code records source-station hashes and checkpoint provenance and checks them before training.

## Privacy and release hygiene

Station ids, coordinates, generation histories, and installed capacities may be sensitive. Before publishing a derivative dataset, confirm that your data agreement permits redistribution and consider anonymization and coordinate coarsening. None of these files are required to be committed to GitHub.

