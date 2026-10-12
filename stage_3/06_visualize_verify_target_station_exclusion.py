#!/usr/bin/env python3
"""Plot selected PV targets and audit their exclusion from the PV-field cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stage3_paths import (
    SOURCE_REGIONAL_RUN_DIR,
    TARGET_EXCLUDED_PV_FRAME_CACHE_DIR,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_EXCLUSION_AUDIT_DIR,
    TARGET_SELECTION_DIR,
    ensure_stage3_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize selected PV target stations and verify target-excluded PV-field cache provenance."
    )
    parser.add_argument("--selection_dir", type=Path, default=TARGET_SELECTION_DIR)
    parser.add_argument("--target_run_dir", type=Path, default=TARGET_EXCLUDED_RUN_DIR)
    parser.add_argument("--frame_cache_dir", type=Path, default=TARGET_EXCLUDED_PV_FRAME_CACHE_DIR)
    parser.add_argument("--source_station_split", type=Path, default=SOURCE_REGIONAL_RUN_DIR / "station_split.csv")
    parser.add_argument("--out_dir", type=Path, default=TARGET_EXCLUSION_AUDIT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_station_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required station table not found: {path}")
    table = pd.read_csv(path, dtype={"station_sn": str})
    required = {"station_sn", "lon", "lat"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Station table {path} is missing columns: {missing}")
    table = table.dropna(subset=["station_sn", "lon", "lat"]).copy()
    table["station_sn"] = table["station_sn"].astype(str)
    return table.reset_index(drop=True)


def station_set(table: pd.DataFrame) -> set[str]:
    return set(table["station_sn"].astype(str))


def station_sns_sha1(table: pd.DataFrame) -> str:
    sns = sorted(str(value) for value in table["station_sn"].dropna().astype(str).tolist())
    return hashlib.sha1("\n".join(sns).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_audit(
    out_path: Path,
    all_stations: pd.DataFrame,
    remaining: pd.DataFrame,
    targets: pd.DataFrame,
    checks: dict[str, bool],
    frame_meta: dict,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, summary_ax) = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    ax.scatter(
        all_stations["lon"],
        all_stations["lat"],
        s=20,
        c="#d6dde5",
        edgecolors="#8996a3",
        linewidths=0.3,
        alpha=0.55,
        label=f"all original stations ({len(all_stations)})",
    )
    ax.scatter(
        remaining["lon"],
        remaining["lat"],
        s=27,
        c="#2b8cbe",
        edgecolors="#075985",
        linewidths=0.45,
        alpha=0.78,
        label=f"PV-field source stations ({len(remaining)})",
    )
    ax.scatter(
        targets["lon"],
        targets["lat"],
        s=150,
        marker="*",
        c="#ef4444",
        edgecolors="#7f1d1d",
        linewidths=0.9,
        label=f"selected targets excluded from field ({len(targets)})",
        zorder=5,
    )
    rank_col = "target_rank" if "target_rank" in targets.columns else None
    for index, row in targets.iterrows():
        rank = int(row[rank_col]) if rank_col else index + 1
        ax.annotate(
            str(rank),
            (float(row["lon"]), float(row["lat"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#991b1b",
            weight="bold",
            zorder=6,
        )
    ax.set_title("Stage 3 PV stations: selected targets are excluded before field construction")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)

    summary_ax.axis("off")
    all_pass = all(checks.values())
    lines = [
        "Target-exclusion audit",
        "",
        f"status: {'PASS' if all_pass else 'FAIL'}",
        "",
        f"original stations: {len(all_stations)}",
        f"selected target stations: {len(targets)}",
        f"PV-field source stations: {len(remaining)}",
        f"frame-cache source count: {frame_meta.get('source_station_count', 'missing')}",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}")
    lines.extend(
        [
            "",
            "Required invariant:",
            "selected targets intersect PV-field sources = empty set",
            "",
            "Frame cache provenance:",
            str(frame_meta.get("source_station_policy", "missing")),
        ]
    )
    summary_ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="#166534" if all_pass else "#b91c1c",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_stage3_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_path = args.selection_dir / "target_stations.csv"
    selection_excluded_path = args.selection_dir / "target_excluded_station_split.csv"
    run_excluded_path = args.target_run_dir / "target_excluded_station_split.csv"
    frame_meta_path = args.frame_cache_dir / "frame_cache_meta.json"

    all_stations = read_station_csv(args.source_station_split)
    targets = read_station_csv(target_path)
    selection_excluded = read_station_csv(selection_excluded_path)
    run_excluded = read_station_csv(run_excluded_path)
    frame_meta = load_json(frame_meta_path)

    all_sns = station_set(all_stations)
    target_sns = station_set(targets)
    selection_excluded_sns = station_set(selection_excluded)
    run_excluded_sns = station_set(run_excluded)
    overlap = sorted(target_sns & run_excluded_sns)
    expected_hash = station_sns_sha1(run_excluded)
    frame_hash = str(frame_meta.get("source_station_sns_sha1", ""))
    expected_count = int(len(run_excluded))
    cache_count = int(frame_meta.get("source_station_count", -1))

    checks = {
        "target count is 50": len(targets) == 50,
        "target station IDs are unique": len(targets) == len(target_sns),
        "target-excluded station IDs are unique": len(run_excluded) == len(run_excluded_sns),
        "selection and target-run exclusion lists match": selection_excluded_sns == run_excluded_sns,
        "selected targets are absent from PV-field sources": not overlap,
        "original stations equal targets union PV-field sources": all_sns == target_sns | run_excluded_sns,
        "frame-cache source station count matches exclusion list": cache_count == expected_count,
        "frame-cache source station SHA1 matches exclusion list": frame_hash == expected_hash,
        "frame-cache metadata records target exclusion policy": (
            str(frame_meta.get("source_station_policy", ""))
            == "target stations excluded before splatting PV_norm field frames"
        ),
    }
    report = {
        "purpose": "Audit that selected PV target stations were excluded before Stage 3 PV-field construction.",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "original_station_count": int(len(all_stations)),
            "target_station_count": int(len(targets)),
            "pv_field_source_station_count": int(len(run_excluded)),
            "frame_cache_source_station_count": cache_count,
        },
        "selected_target_overlap_with_pv_field_sources": overlap,
        "expected_frame_cache_source_station_sns_sha1": expected_hash,
        "actual_frame_cache_source_station_sns_sha1": frame_hash,
        "paths": {
            "source_station_split": str(args.source_station_split),
            "target_stations": str(target_path),
            "selection_target_excluded_station_split": str(selection_excluded_path),
            "target_run_target_excluded_station_split": str(run_excluded_path),
            "frame_cache_meta": str(frame_meta_path),
        },
    }
    report_path = args.out_dir / "target_exclusion_audit.json"
    figure_path = args.out_dir / "target_station_distribution_and_exclusion_audit.png"
    write_json(report_path, report)
    targets.to_csv(args.out_dir / "selected_target_stations_for_audit.csv", index=False)
    plot_audit(figure_path, all_stations, run_excluded, targets, checks, frame_meta, args.dpi)

    print("==== Stage 3 target-station exclusion audit ====")
    print(f"status: {report['status']}")
    print(f"original stations: {len(all_stations)}")
    print(f"selected targets: {len(targets)}")
    print(f"PV-field source stations after exclusion: {len(run_excluded)}")
    print(f"target/PV-field overlap: {len(overlap)}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print("figure:", figure_path)
    print("report:", report_path)
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 3 target-station exclusion audit failed. Inspect the saved report and rebuild "
            "02_select_targets_build_pvnorm_cache.py before PV-field training."
        )


if __name__ == "__main__":
    main()
