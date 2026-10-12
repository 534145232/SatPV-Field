import os
from pathlib import Path


STAGE1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SATPV_PROJECT_ROOT", STAGE1_DIR.parent)).expanduser().resolve()
ARTIFACT_ROOT = Path(
    os.environ.get("SATPV_DATA_ROOT", os.environ.get("OLD_DATA_ROOT", PROJECT_ROOT / "data"))
).expanduser().resolve()
DATA_DIR = Path(
    os.environ.get("SATPV_STAGE1_ROOT", ARTIFACT_ROOT / "stage1")
).expanduser().resolve() / "data"

BIG_MONTHLY_DIR = DATA_DIR / "01_big_monthly_netcdf"
CENTER_MONTHLY_DIR = DATA_DIR / "02_center_monthly_netcdf"

INDEX_DIR = DATA_DIR / "03_indices"
CENTER_INDEX_DIR = INDEX_DIR / "center"
BIG_INDEX_DIR = INDEX_DIR / "big_patch"

CACHE_DIR = DATA_DIR / "04_cache"
BIG_FRAME_CACHE_DIR = CACHE_DIR / "big_frame_cache_npy"
CENTER_FRAME_CACHE_DIR = CACHE_DIR / "center_frame_cache_npy"
CENTER_SAMPLE_CACHE_DIR = CACHE_DIR / "center_sample_cache_4hist_3future_float32"
CENTER_CLEAR_SKY_CACHE_DIR = CACHE_DIR / "center_clear_sky_cache_4hist_3future"

CENTER_INDEX_ALL = CENTER_INDEX_DIR / "h8_center_index_4hist_3future_06_17_all.csv"
CENTER_INDEX_TRAIN = CENTER_INDEX_DIR / "h8_center_index_4hist_3future_06_17_train.csv"
CENTER_INDEX_VAL = CENTER_INDEX_DIR / "h8_center_index_4hist_3future_06_17_val.csv"
CENTER_INDEX_TEST = CENTER_INDEX_DIR / "h8_center_index_4hist_3future_06_17_test.csv"

PATCH_BANK_CSV = BIG_INDEX_DIR / "patch_bank_25_overlap50.csv"
BIG_INDEX_MAIN = (
    BIG_INDEX_DIR
    / "h8_big_patch_index_4hist_3future_06_17_train25_valtest_center.csv"
)
BIG_INDEX_ALL_PATCH = (
    BIG_INDEX_DIR
    / "h8_big_patch_index_4hist_3future_06_17_allpatch_all_split.csv"
)


def ensure_stage1_dirs() -> None:
    for path in (
        BIG_MONTHLY_DIR,
        CENTER_MONTHLY_DIR,
        CENTER_INDEX_DIR,
        BIG_INDEX_DIR,
        BIG_FRAME_CACHE_DIR,
        CENTER_FRAME_CACHE_DIR,
        CENTER_SAMPLE_CACHE_DIR,
        CENTER_CLEAR_SKY_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
