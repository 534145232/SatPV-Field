"""Central paths for the Stage 3 PV-field forecasting pipeline."""

import os
from pathlib import Path


STAGE3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", STAGE3_DIR.parent)).expanduser().resolve()
WORKSPACE = PROJECT_ROOT
ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
).expanduser().resolve()
STAGE3_ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_STAGE3_ROOT", ARTIFACT_ROOT / "stage3")
).expanduser().resolve()
DATA_DIR = STAGE3_ARTIFACT_ROOT / "data"
RUNS_DIR = STAGE3_ARTIFACT_ROOT / "runs"


def _old_data_roots() -> tuple[Path, ...]:
    """Return configured and conventional roots used by relocated servers."""
    values = []
    configured = str(os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", ""))).strip()
    if configured:
        values.append(Path(configured).expanduser())
    values.append(ARTIFACT_ROOT)
    unique = []
    seen = set()
    for value in values:
        key = str(value)
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return tuple(unique)


def _has_payload(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _prefer_existing(local_path: Path, stage_name: str, *relative_parts: str) -> Path:
    """Prefer the in-project artifact, then an external old-data copy."""
    candidates = [local_path]
    candidates.extend(root / stage_name / Path(*relative_parts) for root in _old_data_roots())
    for candidate in candidates:
        if _has_payload(candidate):
            return candidate
    return local_path

STAGE1_DIR = PROJECT_ROOT / "stage_1"
STAGE2_DIR = PROJECT_ROOT / "stage_2"
SOURCE_REGIONAL_RUN_DIR = DATA_DIR / "00_source_regional_split"

STAGE1_CENTER_MONTHLY_DIR = _prefer_existing(
    STAGE1_DIR / "data" / "02_center_monthly_netcdf", "stage1", "data", "02_center_monthly_netcdf"
)
STAGE1_CENTER_INDEX_ALL = _prefer_existing(
    STAGE1_DIR / "data" / "03_indices" / "center" / "h8_center_index_4hist_3future_06_17_all.csv",
    "stage1",
    "data",
    "03_indices",
    "center",
    "h8_center_index_4hist_3future_06_17_all.csv",
)
STAGE1_CENTER_FRAME_CACHE_DIR = _prefer_existing(
    STAGE1_DIR / "data" / "04_cache" / "center_frame_cache_npy",
    "stage1",
    "data",
    "04_cache",
    "center_frame_cache_npy",
)

LEGACY_CENTER_SWR_CACHE_DIR = _prefer_existing(
    DATA_DIR / "00_stage1_center_swr_legacy_adapter", "stage3", "data", "00_stage1_center_swr_legacy_adapter"
)
MULTISOURCE_CACHE_DIR = _prefer_existing(
    DATA_DIR / "01_multisource_nwp_cache", "stage3", "data", "01_multisource_nwp_cache"
)
TARGET_SELECTION_DIR = _prefer_existing(
    DATA_DIR / "02_target_selection_lhs50", "stage3", "data", "02_target_selection_lhs50"
)
TARGET_EXCLUDED_RUN_DIR = _prefer_existing(
    DATA_DIR / "03_target_excluded_run", "stage3", "data", "03_target_excluded_run"
)
TARGET_EXCLUDED_PV_FRAME_CACHE_DIR = _prefer_existing(
    DATA_DIR / "04_target_excluded_pvnorm_frame_cache",
    "stage3",
    "data",
    "04_target_excluded_pvnorm_frame_cache",
)
CENTER_H8_SWR_ANCHOR_CACHE_ROOT = _prefer_existing(
    DATA_DIR / "05_center_h8_swr_anchor_cache", "stage3", "data", "05_center_h8_swr_anchor_cache"
)
CENTER_H8_SWR_ANCHOR_ACTIVE_MANIFEST = CENTER_H8_SWR_ANCHOR_CACHE_ROOT / "active_cache.json"
ANCHOR_ALIGNMENT_SELECTION_DIR = DATA_DIR / "_anchor_alignment_shifted_selection"
MULTISOURCE_VIS_DIR = DATA_DIR / "06_multisource_sample_visualization"
TARGET_EXCLUSION_AUDIT_DIR = DATA_DIR / "07_target_exclusion_audit"
SWR_PREDICTION_VIS_DIR = DATA_DIR / "08_swr_prediction_visualization"
TRAINING_INPUT_AUDIT_DIR = DATA_DIR / "09_training_input_audit"
H8_RESAMPLING_AUDIT_DIR = DATA_DIR / "10_h8_resampling_validation"

MOTION_FUSION_RUN_DIR = RUNS_DIR / "04l_swr_anchor_motion_fusion_pvnorm_field_model_seed2026"

SWR_RATIO_3DCNN_CHECKPOINT = _prefer_existing(
    STAGE2_DIR
    / "runs"
    / "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f"
    / "head_conv3d_decoder"
    / "best.pt",
    "stage2",
    "runs",
    "h8_swr_3dcnn_ratio_stage1_big_patch_5x5_4h3f",
    "head_conv3d_decoder",
    "best.pt",
)
SWR_RESIDUAL_V2_CHECKPOINT = _prefer_existing(
    STAGE2_DIR
    / "runs"
    / "h8_swr_residual_convlstm_on_bigpatch_conv3d_decoder_4h3f"
    / "v2_no_center_weight"
    / "best.pt",
    "stage2",
    "runs",
    "h8_swr_residual_convlstm_on_bigpatch_conv3d_decoder_4h3f",
    "v2_no_center_weight",
    "best.pt",
)
# Downstream Stage 3 defaults use the best complete SWR predictor:
# frozen big-patch Ratio3DCNN conv3d_decoder + residual ConvLSTM correction.
SWR_BASE_CHECKPOINT = SWR_RESIDUAL_V2_CHECKPOINT

PV_ROOT = Path(os.environ.get("PV_ROOT", ARTIFACT_ROOT / "PV")).expanduser().resolve()
NWP_ROOT = Path(os.environ.get("NWP_ROOT", ARTIFACT_ROOT / "NWP")).expanduser().resolve()
NWP_FARM_INFO_XLSX = Path(
    os.environ.get("NWP_FARM_INFO_XLSX", NWP_ROOT / "old_NWP" / "farm_basic_infomation.xlsx")
).expanduser().resolve()
NWP_STATION_MAP_CSV = Path(
    os.environ.get("NWP_STATION_MAP_CSV", NWP_ROOT / "old_NWP" / "basic_info_500_with_station.csv")
).expanduser().resolve()
TRAIN_PV_DIR = Path(
    os.environ.get("TRAIN_PV_DIR", PV_ROOT / "train_directory_cleaned_csv_same_day06_nwp")
).expanduser().resolve()
VAL_PV_DIR = Path(
    os.environ.get("VAL_PV_DIR", PV_ROOT / "val_directory_cleaned_csv_same_day06_nwp")
).expanduser().resolve()

TIME_START = "2022-01-01 07:00:00"
TIME_END = "2023-06-24 16:45:00"
TRAIN_END = "2023-01-13 23:59:59"
VAL_END = "2023-03-08 23:59:59"


def ensure_stage3_dirs() -> None:
    for path in (
        DATA_DIR,
        RUNS_DIR,
        SOURCE_REGIONAL_RUN_DIR,
        LEGACY_CENTER_SWR_CACHE_DIR,
        MULTISOURCE_CACHE_DIR,
        TARGET_SELECTION_DIR,
        TARGET_EXCLUDED_RUN_DIR,
        TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
        CENTER_H8_SWR_ANCHOR_CACHE_ROOT,
        ANCHOR_ALIGNMENT_SELECTION_DIR,
        MULTISOURCE_VIS_DIR,
        TARGET_EXCLUSION_AUDIT_DIR,
        SWR_PREDICTION_VIS_DIR,
        TRAINING_INPUT_AUDIT_DIR,
        H8_RESAMPLING_AUDIT_DIR,
        MOTION_FUSION_RUN_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
