"""Central paths for Stage 4 target-station PV forecasting."""

import os
from pathlib import Path


STAGE4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", STAGE4_DIR.parent)).expanduser().resolve()
WORKSPACE = PROJECT_ROOT
ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
).expanduser().resolve()
STAGE4_ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_STAGE4_ROOT", ARTIFACT_ROOT / "stage4")
).expanduser().resolve()
DATA_DIR = STAGE4_ARTIFACT_ROOT / "data"
RUNS_DIR = STAGE4_ARTIFACT_ROOT / "runs"

STAGE3_DIR = PROJECT_ROOT / "stage_3"
STAGE3_ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_STAGE3_ROOT", ARTIFACT_ROOT / "stage3")
).expanduser().resolve()
STAGE3_DATA_DIR = STAGE3_ARTIFACT_ROOT / "data"
STAGE3_RUNS_DIR = STAGE3_ARTIFACT_ROOT / "runs"

TARGET_SELECTION_DIR = STAGE3_DATA_DIR / "02_target_selection_lhs50"
TARGET_ISSUE_TIME_INDEX = TARGET_SELECTION_DIR / "issue_time_index.csv"
TARGET_EXCLUDED_RUN_DIR = STAGE3_DATA_DIR / "03_target_excluded_run"
TARGET_EXCLUDED_PV_FRAME_CACHE_DIR = STAGE3_DATA_DIR / "04_target_excluded_pvnorm_frame_cache"
MULTISOURCE_CACHE_DIR = STAGE3_DATA_DIR / "01_multisource_nwp_cache"
CENTER_H8_SWR_ANCHOR_CACHE_ROOT = STAGE3_DATA_DIR / "05_center_h8_swr_anchor_cache"

BEST_STAGE3_FIELD_RUN_DIR = (
    STAGE3_RUNS_DIR / "ablation_motion_branch_last_only" / "warp_direct_gate"
)
BEST_STAGE3_FIELD_MODEL_NAME = "swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate"
BEST_STAGE3_FIELD_CHECKPOINT = BEST_STAGE3_FIELD_RUN_DIR / f"{BEST_STAGE3_FIELD_MODEL_NAME}_best.pt"

PV_ROOT = Path(os.environ.get("PV_ROOT", ARTIFACT_ROOT / "PV")).expanduser().resolve()
TRAIN_PV_DIR = Path(
    os.environ.get("TRAIN_PV_DIR", PV_ROOT / "train_directory_cleaned_csv_same_day06_nwp")
).expanduser().resolve()
VAL_PV_DIR = Path(
    os.environ.get("VAL_PV_DIR", PV_ROOT / "val_directory_cleaned_csv_same_day06_nwp")
).expanduser().resolve()

PER_STATION_TEMPORAL_BASELINE_RUN_DIR = RUNS_DIR / "03_per_station_temporal_baseline_seed2026"
PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR = RUNS_DIR / "05l_per_station_temporal_stage3_field_patch_seed2026"
PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR = RUNS_DIR / "05h_per_station_temporal_historical_field_patch_seed2026"
PER_STATION_STAGE3_SWR_PATCH_RUN_DIR = RUNS_DIR / "06l_per_station_temporal_stage3_swr_patch_seed2026"
PER_STATION_03_VS_05_EVAL_DIR = RUNS_DIR / "07_eval_compare_03_baseline_vs_05_field_patch"


def ensure_stage4_dirs() -> None:
    for path in (
        DATA_DIR,
        RUNS_DIR,
        PER_STATION_TEMPORAL_BASELINE_RUN_DIR,
        PER_STATION_STAGE3_FIELD_PATCH_RUN_DIR,
        PER_STATION_HISTORICAL_FIELD_PATCH_RUN_DIR,
        PER_STATION_STAGE3_SWR_PATCH_RUN_DIR,
        PER_STATION_03_VS_05_EVAL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
