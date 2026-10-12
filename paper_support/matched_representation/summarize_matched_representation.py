#!/usr/bin/env python3
"""Summarize the strict same-sample graph-versus-field comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-detail", type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "target_graph" / "target_graph_eval_detail.csv",
    )
    parser.add_argument("--historical-field-detail", type=Path, default=None)
    parser.add_argument("--predicted-field-detail", type=Path, default=None)
    parser.add_argument("--external-results", type=Path, default=None)
    parser.add_argument(
        "--source-metrics", type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "source_graph" / "source_graph_metrics.csv",
    )
    parser.add_argument(
        "--target-run-meta", type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "target_graph" / "run_meta.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=SCRIPT_DIR / "outputs" / "full" / "summary",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_detail(path: Path, forced_method: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"station_sn": str})
    if "eval_split" in frame.columns:
        frame = frame[frame["eval_split"].astype(str) == "test"].copy()
    required = {"station_sn", "issue_time", "h", "pred_pv_norm", "true_pv_norm"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} misses detail columns: {missing}")
    if forced_method is not None:
        frame["method"] = forced_method
    elif "method" not in frame.columns:
        if "model" not in frame.columns:
            raise ValueError(f"{path} has neither method nor model column")
        frame["method"] = frame["model"].astype(str)
    issue_time = pd.to_datetime(frame["issue_time"], errors="coerce")
    if bool(issue_time.isna().any()):
        raise ValueError(f"{path} contains invalid issue_time values")
    frame["issue_sec"] = issue_time.map(lambda value: int(pd.Timestamp(value).timestamp())).astype(np.int64)
    frame["h"] = pd.to_numeric(frame["h"], errors="raise").astype(int)
    frame["pred_pv_norm"] = pd.to_numeric(frame["pred_pv_norm"], errors="raise").astype(float)
    frame["true_pv_norm"] = pd.to_numeric(frame["true_pv_norm"], errors="raise").astype(float)
    key = ["station_sn", "issue_sec", "h", "method"]
    if bool(frame.duplicated(key).any()):
        examples = frame.loc[frame.duplicated(key, keep=False), key].head().to_dict("records")
        raise ValueError(f"Duplicate evaluation keys in {path}: {examples}")
    return frame[["station_sn", "issue_sec", "issue_time", "h", "method", "pred_pv_norm", "true_pv_norm"]]


def strict_intersection(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], set[tuple]]:
    key_columns = ["station_sn", "issue_sec", "h"]
    key_sets = {
        method: set(map(tuple, frame[key_columns].to_numpy()))
        for method, frame in frames.items()
    }
    common = set.intersection(*key_sets.values()) if key_sets else set()
    if not common:
        raise ValueError("No common station/issue/horizon samples across supplied methods")
    common_frame = pd.DataFrame(sorted(common), columns=key_columns)
    aligned = {}
    reference_true = None
    for method, frame in frames.items():
        sub = common_frame.merge(frame, on=key_columns, how="left", validate="one_to_one")
        if bool(sub[["pred_pv_norm", "true_pv_norm"]].isna().any().any()):
            raise AssertionError(f"Intersection merge produced missing values for {method}")
        if reference_true is None:
            reference_true = sub["true_pv_norm"].to_numpy(np.float64)
        elif not np.allclose(reference_true, sub["true_pv_norm"].to_numpy(np.float64), rtol=0, atol=1.0e-6):
            raise AssertionError(f"Target truth differs across methods on the common intersection: {method}")
        aligned[method] = sub
    return aligned, common


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "n_samples": int(target.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": 1.0 - float(np.square(error).sum()) / denominator if denominator > 1.0e-12 else float("nan"),
    }


def method_tables(aligned: dict[str, pd.DataFrame]):
    station_rows = []
    horizon_rows = []
    result_rows = []
    for method, frame in aligned.items():
        per_station = []
        for station_sn, group in frame.groupby("station_sn", sort=True):
            row = {"station_sn": str(station_sn), "method": method}
            row.update(metrics(group["pred_pv_norm"], group["true_pv_norm"]))
            station_rows.append(row)
            per_station.append(row)
        per_station_frame = pd.DataFrame(per_station)
        pooled = metrics(frame["pred_pv_norm"], frame["true_pv_norm"])
        result_rows.append(
            {
                "method": method,
                "mae": float(per_station_frame["mae"].mean()),
                "rmse": float(per_station_frame["rmse"].mean()),
                "r2": float(per_station_frame["r2"].mean()),
                "pooled_mae": pooled["mae"],
                "pooled_rmse": pooled["rmse"],
                "pooled_r2": pooled["r2"],
                "n_stations": int(per_station_frame["station_sn"].nunique()),
                "n_samples": int(len(frame)),
                "sample_scope": "strict_common_detail_intersection",
            }
        )
        for horizon, group in frame.groupby("h", sort=True):
            per_station_horizon = []
            for _, station_group in group.groupby("station_sn", sort=True):
                per_station_horizon.append(
                    metrics(station_group["pred_pv_norm"], station_group["true_pv_norm"])
                )
            macro = pd.DataFrame(per_station_horizon)
            pooled_h = metrics(group["pred_pv_norm"], group["true_pv_norm"])
            horizon_rows.append(
                {
                    "method": method,
                    "h": int(horizon),
                    "horizon_minutes": int(horizon) * 15,
                    "mae": float(macro["mae"].mean()),
                    "rmse": float(macro["rmse"].mean()),
                    "r2": float(macro["r2"].mean()),
                    "pooled_mae": pooled_h["mae"],
                    "pooled_rmse": pooled_h["rmse"],
                    "pooled_r2": pooled_h["r2"],
                    "n_stations": int(group["station_sn"].nunique()),
                    "n_samples": int(len(group)),
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(station_rows), pd.DataFrame(horizon_rows)


def wide_station_table(long_station: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"target_id": sorted(long_station["station_sn"].astype(str).unique())})
    for method in sorted(long_station["method"].unique()):
        sub = long_station[long_station["method"] == method][["station_sn", "mae", "rmse", "r2"]].copy()
        slug = {
            "HistoricalGraph": "historical_graph",
            "PredictedGraph": "predicted_graph",
            "HistoricalField": "historical_field",
            "PredictedField": "predicted_field",
        }.get(method, method.lower().replace(" ", "_"))
        sub = sub.rename(
            columns={
                "station_sn": "target_id",
                "mae": f"{slug}_mae",
                "rmse": f"{slug}_rmse",
                "r2": f"{slug}_r2",
            }
        )
        output = output.merge(sub, on="target_id", how="left", validate="one_to_one")
    if {"predicted_graph_mae", "predicted_field_mae"}.issubset(output.columns):
        output["predicted_graph_minus_field_mae"] = (
            output["predicted_graph_mae"] - output["predicted_field_mae"]
        )
        output["predicted_field_wins"] = output["predicted_graph_minus_field_mae"] > 0
    if {"historical_graph_mae", "historical_field_mae"}.issubset(output.columns):
        output["historical_graph_minus_field_mae"] = (
            output["historical_graph_mae"] - output["historical_field_mae"]
        )
        output["historical_field_wins"] = output["historical_graph_minus_field_mae"] > 0
    return output


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for start in range(0, int(samples), 1000):
        count = min(1000, int(samples) - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def normalize_external_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "mae_pv_norm": "mae",
        "rmse_pv_norm": "rmse",
        "r2_pv_norm": "r2",
        "station_mean_mae_pv_norm": "mae",
        "station_mean_rmse_pv_norm": "rmse",
        "station_mean_r2_pv_norm": "r2",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    required = {"method", "mae", "rmse", "r2"}
    if not required.issubset(frame.columns):
        raise ValueError(f"External results must contain {sorted(required)} after aliasing")
    frame = frame[list(required)].copy()
    frame["sample_scope"] = "external_reported_not_reintersected"
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for the run report."""

    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)

    def render(value) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = read_detail(args.graph_detail)
    frames = {
        method: group.copy()
        for method, group in graph.groupby("method", sort=False)
        if method in {"HistoricalGraph", "PredictedGraph"}
    }
    missing_graph = {"HistoricalGraph", "PredictedGraph"} - set(frames)
    if missing_graph:
        raise ValueError(f"Graph detail misses methods: {sorted(missing_graph)}")
    if args.historical_field_detail is not None:
        frames["HistoricalField"] = read_detail(args.historical_field_detail, "HistoricalField")
    if args.predicted_field_detail is not None:
        frames["PredictedField"] = read_detail(args.predicted_field_detail, "PredictedField")

    aligned, common_keys = strict_intersection(frames)
    pd.DataFrame(
        sorted(common_keys), columns=["station_sn", "issue_sec", "h"]
    ).to_csv(output_dir / "matched_representation_common_keys.csv", index=False)
    results, station_long, horizon = method_tables(aligned)
    if args.external_results is not None:
        external = normalize_external_results(args.external_results)
        results = pd.concat([external, results], ignore_index=True, sort=False)
    station_wide = wide_station_table(station_long)

    results.to_csv(output_dir / "matched_representation_results.csv", index=False)
    station_wide.to_csv(output_dir / "matched_representation_stationwise.csv", index=False)
    horizon.to_csv(output_dir / "matched_representation_horizon.csv", index=False)

    paired = None
    if "predicted_graph_minus_field_mae" in station_wide.columns:
        difference = station_wide["predicted_graph_minus_field_mae"].to_numpy(np.float64)
        low, high = bootstrap_mean_ci(difference, int(args.bootstrap_samples), int(args.seed))
        valid = difference[np.isfinite(difference)]
        paired = {
            "definition": "station MAE(PredictedGraph) - station MAE(PredictedField)",
            "n_stations": int(valid.size),
            "mean_difference": float(valid.mean()),
            "median_difference": float(np.median(valid)),
            "field_wins": int((valid > 0).sum()),
            "graph_wins": int((valid < 0).sum()),
            "ties": int((valid == 0).sum()),
            "bootstrap_95_ci_mean": [low, high],
            "bootstrap_samples": int(args.bootstrap_samples),
        }
        (output_dir / "paired_predicted_field_vs_graph.json").write_text(
            json.dumps(paired, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    source_summary = "not found"
    if Path(args.source_metrics).is_file():
        source = pd.read_csv(args.source_metrics)
        selected = source[(source["eval_split"].astype(str) == "test") & (source["horizon"].astype(str) == "all")]
        if len(selected):
            row = selected.iloc[0]
            source_summary = (
                f"MAE={float(row['mae']):.6f}, RMSE={float(row['rmse']):.6f}, "
                f"R2={float(row['r2']):.6f}, n={int(row['n_samples'])}"
            )

    fairness = {}
    if Path(args.target_run_meta).is_file():
        fairness = json.loads(Path(args.target_run_meta).read_text(encoding="utf-8")).get("fairness", {})
    field_complete = {"HistoricalField", "PredictedField"}.issubset(frames)
    common_target_count = len({key[0] for key in common_keys})
    graph_ready = (
        common_target_count == 50
        and fairness.get("same_446_source_stations") is True
        and fairness.get("same_50_target_stations") is True
        and fairness.get("historical_predicted_same_evaluation_rows") is True
        and fairness.get("all_requested_targets_completed") is True
    )
    if field_complete and graph_ready:
        decision = "COMPLETE GRAPH–FIELD COMPARISON"
    elif not graph_ready:
        decision = "PRELIMINARY/INCOMPLETE GRAPH RUN"
    else:
        decision = "GRAPH RESULTS COMPLETE; FIELD DETAILS STILL REQUIRED"
    lines = [
        "# Matched source-fleet representation comparison",
        "",
        f"**Decision: {decision}**",
        "",
        "## Evaluation contract",
        "",
        f"- Included detail methods: {', '.join(frames)}",
        f"- Strict common `(target, issue time, horizon)` rows: {len(common_keys)}",
        f"- Targets on the common intersection: {common_target_count}",
        "- Main MAE/RMSE/R2 columns are macro-averaged over target-station metrics, matching the Stage-4 table convention.",
        "- Pooled metrics are retained in the CSV as a secondary check.",
        f"- Source-node test sanity: {source_summary}",
        "",
        "## Results",
        "",
        markdown_table(results),
        "",
        "## Fairness checks",
        "",
    ]
    for name, value in fairness.items():
        lines.append(f"- {'PASS' if value is True else 'FAIL' if value is False else 'INFO'}: `{name}` — {value}")
    if paired is not None:
        lines.extend(
            [
                "",
                "## Paired PredictedField versus PredictedGraph analysis",
                "",
                f"- Mean stationwise MAE difference (Graph − Field): {paired['mean_difference']:.6f}",
                f"- Median difference: {paired['median_difference']:.6f}",
                f"- Field wins: {paired['field_wins']}/{paired['n_stations']}",
                f"- Graph wins: {paired['graph_wins']}/{paired['n_stations']}",
                f"- Paired station bootstrap 95% CI: [{paired['bootstrap_95_ci_mean'][0]:.6f}, {paired['bootstrap_95_ci_mean'][1]:.6f}]",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Pending field comparison",
                "",
                "No field-detail CSV was supplied, so field-versus-graph wins and paired confidence intervals were intentionally not calculated.",
                "Re-run this summarizer with both `--historical-field-detail` and `--predicted-field-detail`.",
            ]
        )
    (output_dir / "matched_representation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("==== Matched representation summary ====")
    print("methods:", list(frames))
    print("strict common rows:", len(common_keys))
    print("summary:", output_dir / "matched_representation_summary.md")


if __name__ == "__main__":
    main()
