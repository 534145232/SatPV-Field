#!/usr/bin/env python3
"""Collect Stage 3 density observability and field-forecasting results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from density_common import (
    BEST_FULL_STAGE3_RUN_DIR,
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DENSITY_DATA_ROOT,
    DENSITY_STAGE3_RUN_ROOT,
    density_stage3_run_dir,
    parse_ratios,
    ratio_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Stage 3 source-fleet density results.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratios", type=str, default=",".join(str(r) for r in DEFAULT_RATIOS))
    parser.add_argument("--out_dir", type=Path, default=DENSITY_STAGE3_RUN_ROOT)
    parser.add_argument("--full_stage3_run_dir", type=Path, default=BEST_FULL_STAGE3_RUN_DIR)
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    obs_path = DENSITY_DATA_ROOT / f"seed{int(args.seed)}" / "density_observability_summary.csv"
    obs = _read_optional(obs_path)
    rows = []
    for ratio in parse_ratios(args.ratios):
        base = {
            "seed": int(args.seed),
            "source_density_ratio": float(ratio),
            "source_density_label": ratio_label(ratio),
        }
        if not obs.empty:
            hit = obs[obs["source_density_label"].astype(str) == ratio_label(ratio)]
            if not hit.empty:
                base.update(hit.iloc[0].to_dict())

        run_dir = Path(args.full_stage3_run_dir) if float(ratio) >= 1.0 else density_stage3_run_dir(args.seed, ratio)
        native = _read_optional(run_dir / "swr_anchor_guided_pvnorm_field_eval_summary.csv")
        if not native.empty:
            native_all = native[(native["eval_split"].astype(str) == "test") & (native["horizon"].astype(str) == "all")]
            if not native_all.empty:
                row = native_all.iloc[0]
                base["native_test_mae_pv_norm_field"] = row.get("mae_pv_norm_field")
                base["native_test_rmse_pv_norm_field"] = row.get("rmse_pv_norm_field")

        fidelity = _read_optional(density_stage3_run_dir(args.seed, ratio) / "density_fidelity_eval" / "density_field_fidelity_summary.csv")
        if not fidelity.empty:
            for scope, prefix in (
                ("representation_vs_full", "repr"),
                ("end_to_end_vs_full", "total"),
            ):
                hit = fidelity[
                    (fidelity["eval_split"].astype(str) == "test")
                    & (fidelity["horizon"].astype(str) == "all")
                    & (fidelity["metric_scope"].astype(str) == scope)
                ]
                if not hit.empty:
                    base[f"{prefix}_test_mae_vs_full"] = hit.iloc[0].get("mae_pv_norm_field")
                    base[f"{prefix}_test_rmse_vs_full"] = hit.iloc[0].get("rmse_pv_norm_field")
        elif float(ratio) >= 1.0 and "native_test_mae_pv_norm_field" in base:
            base["repr_test_mae_vs_full"] = 0.0
            base["repr_test_rmse_vs_full"] = 0.0
            base["total_test_mae_vs_full"] = base.get("native_test_mae_pv_norm_field")
            base["total_test_rmse_vs_full"] = base.get("native_test_rmse_pv_norm_field")
        rows.append(base)

    out = pd.DataFrame(rows).sort_values("source_density_ratio").reset_index(drop=True)
    out_dir = Path(args.out_dir) / f"seed{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "stage3_density_summary.csv", index=False)
    print(out.to_string(index=False))
    print("saved:", out_dir / "stage3_density_summary.csv")


if __name__ == "__main__":
    main()
