#!/usr/bin/env python3
"""Print paper-ready Table-IV rows from graph/field evaluation details.

The script is intentionally read-only with respect to experiment outputs.  It
computes the same station-macro MAE, RMSE, and R2 convention used by the
Stage-4 aggregate tables, prints four-decimal values and LaTeX rows, and also
writes CSV/Markdown/TEX artifacts for direct manuscript use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DETAIL_NAME = "per_station_temporal_swr_anchor04l_field_patch_eval_detail.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR.parents[1])
    parser.add_argument(
        "--old-data-root",
        type=Path,
        default=Path(os.environ["OLD_DATA_ROOT"]) if os.environ.get("OLD_DATA_ROOT") else None,
    )
    parser.add_argument(
        "--graph-detail",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "target_graph" / "target_graph_eval_detail.csv",
    )
    parser.add_argument("--historical-field-detail", type=Path, default=None)
    parser.add_argument("--predicted-field-detail", type=Path, default=None)
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Only export HistoricalGraph and PredictedGraph; do not discover field runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "table_iv",
    )
    parser.add_argument("--expected-targets", type=int, default=50)
    return parser.parse_args()


def candidate_roots(project_root: Path, old_data_root: Path | None) -> list[Path]:
    roots = [project_root / "stage_4" / "runs", project_root / "stage4" / "runs"]
    if old_data_root is not None:
        roots.extend(
            [
                old_data_root / "stage4" / "runs",
                old_data_root / "stage4",
                old_data_root / "stage_4" / "runs",
                old_data_root / "stage_4",
            ]
        )
    output = []
    seen = set()
    for root in roots:
        key = str(root.expanduser())
        if key not in seen:
            output.append(root.expanduser())
            seen.add(key)
    return output


def resolve_field_detail(
    explicit: Path | None,
    roots: list[Path],
    run_name: str,
    label: str,
) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{label} detail does not exist: {path}")
        return path.resolve()

    direct = []
    for root in roots:
        direct.extend(
            [
                root / run_name / DETAIL_NAME,
                root / "runs" / run_name / DETAIL_NAME,
            ]
        )
    for path in direct:
        if path.is_file():
            return path.resolve()

    matches = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(DETAIL_NAME):
            if run_name.lower() in str(path).lower():
                matches.append(path.resolve())
    unique = sorted({str(path): path for path in matches}.values(), key=str)
    if len(unique) > 1:
        options = "\n".join(f"  - {path}" for path in unique)
        raise RuntimeError(
            f"Multiple {label} detail files found; pass its path explicitly:\n{options}"
        )
    return unique[0] if unique else None


def canonical_detail(path: Path, forced_method: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station_sn": str})
    if "eval_split" in frame.columns:
        frame = frame[frame["eval_split"].astype(str) == "test"].copy()
    required = {"station_sn", "issue_time", "h", "pred_pv_norm", "true_pv_norm"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")
    if forced_method is not None:
        frame["method"] = forced_method
    elif "method" not in frame.columns:
        if "model" not in frame.columns:
            raise ValueError(f"{path} has neither method nor model")
        frame["method"] = frame["model"].astype(str)
    issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
    if bool(issue_time.isna().any()):
        raise ValueError(f"{path} contains invalid issue_time values")
    frame["issue_sec"] = issue_time.astype("int64") // 1_000_000_000
    frame["h"] = pd.to_numeric(frame["h"], errors="raise").astype(int)
    for column in ("pred_pv_norm", "true_pv_norm"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    columns = [
        "station_sn", "issue_sec", "issue_time", "h", "method",
        "pred_pv_norm", "true_pv_norm",
    ]
    frame = frame[columns].copy()
    key = ["station_sn", "issue_sec", "h", "method"]
    if bool(frame.duplicated(key).any()):
        raise ValueError(f"Duplicate evaluation rows in {path}")
    return frame


def regression_metrics(group: pd.DataFrame) -> dict[str, float]:
    prediction = group["pred_pv_norm"].to_numpy(np.float64)
    target = group["true_pv_norm"].to_numpy(np.float64)
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": 1.0 - float(np.square(error).sum()) / denominator
        if denominator > 1.0e-12
        else float("nan"),
    }


def station_macro_row(frame: pd.DataFrame, method: str, type_name: str) -> dict:
    station_rows = []
    for station_sn, group in frame.groupby("station_sn", sort=True):
        station_rows.append({"station_sn": str(station_sn), **regression_metrics(group)})
    stations = pd.DataFrame(station_rows)
    pooled = regression_metrics(frame)
    return {
        "type": type_name,
        "method": method,
        "mae": float(stations["mae"].mean()),
        "rmse": float(stations["rmse"].mean()),
        "r2": float(stations["r2"].mean()),
        "n_targets": int(stations["station_sn"].nunique()),
        "n_rows": int(len(frame)),
        "pooled_mae": pooled["mae"],
        "pooled_rmse": pooled["rmse"],
        "pooled_r2": pooled["r2"],
    }


def strict_common_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int]:
    key_columns = ["station_sn", "issue_sec", "h"]
    key_sets = [set(map(tuple, frame[key_columns].to_numpy())) for frame in frames.values()]
    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        return {}, 0
    key_frame = pd.DataFrame(sorted(common), columns=key_columns)
    aligned = {}
    reference = None
    for method, frame in frames.items():
        sub = key_frame.merge(frame, on=key_columns, how="left", validate="one_to_one")
        truth = sub["true_pv_norm"].to_numpy(np.float64)
        if reference is None:
            reference = truth
        elif not np.allclose(reference, truth, atol=1.0e-6, rtol=0):
            raise AssertionError(f"Truth differs on the strict intersection for {method}")
        aligned[method] = sub
    return aligned, len(common)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["Type", "Method", "MAE", "RMSE", "R2", "Targets", "Rows"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["type"]),
                    str(row["method"]),
                    f"{float(row['mae']):.4f}",
                    f"{float(row['rmse']):.4f}",
                    f"{float(row['r2']):.4f}",
                    str(int(row["n_targets"])),
                    str(int(row["n_rows"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def latex_rows(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        type_name = str(row["type"]).replace("_", r"\_")
        method = str(row["method"]).replace("_", r"\_")
        rows.append(
            f"{type_name} & {method} & {float(row['mae']):.4f} & "
            f"{float(row['rmse']):.4f} & {float(row['r2']):.4f} \\\\"
        )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    roots = candidate_roots(project_root, args.old_data_root)
    graph_path = Path(args.graph_detail).expanduser()
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph evaluation detail not found: {graph_path}. Run the target stage first."
        )

    historical_path = None
    predicted_path = None
    if not args.graph_only:
        historical_path = resolve_field_detail(
            args.historical_field_detail,
            roots,
            "05h_per_station_temporal_historical_field_patch_seed2026",
            "HistoricalField",
        )
        predicted_path = resolve_field_detail(
            args.predicted_field_detail,
            roots,
            "05l_per_station_temporal_stage3_field_patch_seed2026",
            "PredictedField",
        )

    graph = canonical_detail(graph_path)
    frames = {
        method: group.copy()
        for method, group in graph.groupby("method", sort=False)
        if method in {"HistoricalGraph", "PredictedGraph"}
    }
    missing_graph = {"HistoricalGraph", "PredictedGraph"} - set(frames)
    if missing_graph:
        raise ValueError(f"Graph detail misses methods: {sorted(missing_graph)}")
    if historical_path is not None:
        frames["HistoricalField"] = canonical_detail(historical_path, "HistoricalField")
    if predicted_path is not None:
        frames["PredictedField"] = canonical_detail(predicted_path, "PredictedField")

    presentation = {
        "HistoricalGraph": ("Spatio-temporal", "Historical source-graph guidance"),
        "PredictedGraph": ("Spatio-temporal", "Predicted source-graph guidance"),
        "HistoricalField": ("Spatio-temporal", "Historical PV-field guidance"),
        "PredictedField": ("Proposed", "SatPV-Field"),
    }
    order = ["HistoricalGraph", "PredictedGraph", "HistoricalField", "PredictedField"]
    rows = []
    for method in order:
        if method not in frames:
            continue
        type_name, label = presentation[method]
        rows.append(station_macro_row(frames[method], label, type_name))
    table = pd.DataFrame(rows)

    aligned, common_rows = strict_common_frames(frames)
    strict_rows = []
    for method in order:
        if method not in aligned:
            continue
        type_name, label = presentation[method]
        strict_rows.append(station_macro_row(aligned[method], label, type_name))
    strict_table = pd.DataFrame(strict_rows)

    expected_targets = int(args.expected_targets)
    table["paper_ready"] = table["n_targets"].astype(int) == expected_targets
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "table_iv_fill_values.csv", index=False)
    strict_table.to_csv(output_dir / "table_iv_strict_common_values.csv", index=False)
    tex = latex_rows(table)
    (output_dir / "table_iv_fill_rows.tex").write_text(tex + "\n", encoding="utf-8")

    paths = {
        "graph_detail": str(graph_path.resolve()),
        "historical_field_detail": str(historical_path) if historical_path else None,
        "predicted_field_detail": str(predicted_path) if predicted_path else None,
    }
    (output_dir / "table_iv_input_paths.json").write_text(
        json.dumps(paths, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    decision = (
        "READY TO FILL TABLE IV"
        if bool(table["paper_ready"].all())
        else "NOT PAPER-READY: ONE OR MORE METHODS HAVE FEWER THAN 50 TARGETS"
    )
    report = [
        "# Table IV fill values",
        "",
        f"**Decision: {decision}**",
        "",
        "The main values below use the existing Stage-4 convention: compute metrics per target and then macro-average over targets.",
        "",
        markdown_table(table),
        "",
        "## Direct LaTeX rows",
        "",
        "```latex",
        tex,
        "```",
        "",
        "## Strict representation-comparison intersection",
        "",
        f"Common `(target, issue time, horizon)` rows across all available representation methods: {common_rows}.",
        "",
        markdown_table(strict_table) if not strict_table.empty else "No common rows.",
        "",
        "## Input files",
        "",
        *[f"- `{name}`: `{value}`" for name, value in paths.items()],
    ]
    (output_dir / "table_iv_fill_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("==== TABLE IV VALUES (station-macro, test) ====")
    print(table[["type", "method", "mae", "rmse", "r2", "n_targets", "n_rows"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n==== COPY-READY LATEX ROWS ====")
    print(tex)
    print("\n==== STRICT COMMON-INTERSECTION CHECK ====")
    print(f"methods={list(aligned)} common_rows={common_rows}")
    print("decision:", decision)
    print("report:", output_dir / "table_iv_fill_report.md")


if __name__ == "__main__":
    main()
