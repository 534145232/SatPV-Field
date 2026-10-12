from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_paths import SWR_BASE_CHECKPOINT  # noqa: E402
from stage3_swr_anchor_cache import (  # noqa: E402
    assert_cache_matches_checkpoint,
    resolve_active_cache_dir,
)


DEFAULT_MOTION_EVAL_DIR = STAGE3_DIR / "runs" / "ablation_motion_branch" / "eval_all"
DEFAULT_RUN_ROOT = STAGE3_DIR / "runs" / "ablation_temporal_collapse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine temporal-collapse metrics with the reused last-mean baseline."
    )
    parser.add_argument("--motion_eval_dir", type=Path, default=DEFAULT_MOTION_EVAL_DIR)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_RUN_ROOT / "eval_all")
    return parser.parse_args()


def _read_csv(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}\n{hint}")
    return pd.read_csv(path)


def _resolved_text(path_like: str | Path) -> str:
    return str(Path(path_like).expanduser().resolve())


def _validate_motion_baseline(motion_eval_dir: Path, active_cache_dir: Path) -> dict:
    meta_path = motion_eval_dir / "motion_branch_eval_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing motion baseline provenance: {meta_path}\n"
            "Run stage_3/ablation_motion_branch/run_eval_all_motion_branches.sh first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_checkpoint = _resolved_text(SWR_BASE_CHECKPOINT)
    recorded_checkpoint = _resolved_text(meta["swr_checkpoint"])
    if recorded_checkpoint != expected_checkpoint:
        raise RuntimeError(
            "The reusable motion baseline was evaluated with a different SWR checkpoint.\n"
            f"recorded: {recorded_checkpoint}\nexpected: {expected_checkpoint}\n"
            "Rerun stage_3/ablation_motion_branch/run_eval_all_motion_branches.sh."
        )
    recorded_cache = _resolved_text(meta["swr_anchor_cache_dir"])
    expected_cache = _resolved_text(active_cache_dir)
    if recorded_cache != expected_cache:
        raise RuntimeError(
            "The reusable motion baseline was evaluated with a different SWR cache.\n"
            f"recorded: {recorded_cache}\nexpected: {expected_cache}\n"
            "Rerun stage_3/ablation_motion_branch/run_eval_all_motion_branches.sh."
        )
    return meta


def _attach_variant(df: pd.DataFrame, variant: str, source: str) -> pd.DataFrame:
    result = df.copy()
    result.insert(0, "temporal_collapse_variant", variant)
    result.insert(1, "result_source", source)
    return result


def main() -> None:
    args = parse_args()
    active_cache_dir = resolve_active_cache_dir(required=True)
    assert_cache_matches_checkpoint(active_cache_dir, SWR_BASE_CHECKPOINT)
    motion_meta = _validate_motion_baseline(args.motion_eval_dir, active_cache_dir)

    motion_csv = args.motion_eval_dir / "motion_branch_eval_summary.csv"
    motion_df = _read_csv(
        motion_csv,
        "Run stage_3/ablation_motion_branch/run_eval_all_motion_branches.sh first.",
    )
    if "motion_branch" not in motion_df.columns:
        raise RuntimeError(f"Missing motion_branch column in {motion_csv}")
    baseline_df = motion_df.loc[motion_df["motion_branch"] == "warp_direct_gate"].copy()
    if baseline_df.empty:
        raise RuntimeError(f"No warp_direct_gate baseline rows found in {motion_csv}")

    last_only_csv = (
        args.run_root
        / "last_only"
        / "05_validation_visuals"
        / "swr_anchor_guided_validation_eval_summary.csv"
    )
    last_only_df = _read_csv(
        last_only_csv,
        "Run stage_3/ablation_temporal_collapse/run_eval_all_temporal_collapse_ablation.sh.",
    )

    combined = pd.concat(
        [
            _attach_variant(
                baseline_df,
                "last_mean_0p7_0p3_reused_motion_warp_direct_gate",
                str(motion_csv),
            ),
            _attach_variant(last_only_df, "last_only", str(last_only_csv)),
        ],
        ignore_index=True,
        sort=False,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "temporal_collapse_ablation_eval_summary.csv"
    combined.to_csv(out_csv, index=False)
    meta = {
        "purpose": "Temporal-collapse comparison with the 0.7-last + 0.3-mean baseline reused from motion warp_direct_gate.",
        "swr_checkpoint": str(Path(SWR_BASE_CHECKPOINT).resolve()),
        "swr_anchor_cache_dir": str(active_cache_dir.resolve()),
        "baseline_motion_meta": motion_meta,
        "sources": {
            "last_mean_0p7_0p3_reused_motion_warp_direct_gate": str(motion_csv),
            "last_only": str(last_only_csv),
        },
    }
    (args.out_dir / "temporal_collapse_ablation_eval_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    key_rows = combined.loc[
        (combined["eval_split"] == "test")
        & combined["horizon"].astype(str).isin(["all", "h1", "h4", "h8"])
    ]
    columns = [
        col
        for col in [
            "temporal_collapse_variant",
            "eval_split",
            "horizon",
            "mae_pv_norm_field",
            "rmse_pv_norm_field",
            "delta_sign_accuracy",
            "large_delta_mae",
            "delta_corr_median",
        ]
        if col in key_rows.columns
    ]
    print("==== Stage 3 temporal-collapse ablation summary ====")
    print(key_rows[columns].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
