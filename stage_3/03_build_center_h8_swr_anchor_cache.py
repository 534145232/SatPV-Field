#!/usr/bin/env python3
"""Cache center H8 history and three Stage 2 SWR predictions aligned to PV T."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from stage3_paths import (
    ANCHOR_ALIGNMENT_SELECTION_DIR,
    STAGE1_CENTER_FRAME_CACHE_DIR,
    STAGE1_CENTER_INDEX_ALL,
    STAGE1_CENTER_MONTHLY_DIR,
    STAGE2_DIR,
    SWR_BASE_CHECKPOINT,
    TARGET_EXCLUDED_RUN_DIR,
    TARGET_SELECTION_DIR,
    ensure_stage3_dirs,
)
from stage3_cli import inject_defaults
import stage3_build_swr_anchor_cache_impl as module
from stage3_swr_anchor_cache import (
    activate_cache,
    cache_dir_for_checkpoint,
    cache_matches_checkpoint,
    read_cache_info,
)
from stage3_temporal import assert_swr_alignment


sys.path.insert(0, str(STAGE2_DIR))
import train_3dcnn_ratio_swr as stage2_ratio  # noqa: E402
import train_residual_convlstm_correction_swr as stage2_residual  # noqa: E402


SPATIAL_RESAMPLE_SCHEMA = "bilinear_mesh_v3"


def anchor_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / "center_h8_swr_anchor_input_cache_meta.csv",
        cache_dir / "center_h8_swr_anchor_input_cache_missing.csv",
        cache_dir / "center_h8_swr_anchor_input_cache_meta.json",
    )


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--swr_checkpoint", type=Path, default=SWR_BASE_CHECKPOINT)
    parser.add_argument("--cache_name", type=str, default="")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument(
        "--activate",
        type=int,
        default=1,
        help="Write active_cache.json after a successful build so Stage 3 training uses this cache.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def split_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in str(text).split("|") if str(value).strip()]


def validate_anchor_alignment(cache_dir: Path) -> None:
    meta_csv, _, _ = anchor_paths(cache_dir)
    meta = pd.read_csv(meta_csv)
    if meta.empty:
        raise RuntimeError(f"Empty Stage 3 SWR anchor metadata: {meta_csv}")
    for _, row in meta.iterrows():
        assert_swr_alignment(
            pd.Timestamp(row["issue_time"]),
            split_times(row["history_label_times"]),
            split_times(row["future_anchor_label_times"]),
        )
    print(
        "Validated Stage 3 SWR alignment: "
        "history=T-4h..T-1h, predicted future=T..T+2h | rows=",
        len(meta),
    )


def force_requested() -> bool:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--force" and index + 2 <= len(sys.argv[1:]):
            return bool(int(sys.argv[1:][index + 1]))
        if value.startswith("--force="):
            return bool(int(value.split("=", 1)[1]))
    return False


def prepare_shifted_selection() -> pd.DataFrame:
    """Feed T-1h to the legacy anchor builder so its windows become Stage 3 windows."""
    source_path = TARGET_SELECTION_DIR / "issue_time_index.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Stage 3 PV issue-time index not found: {source_path}")
    source = pd.read_csv(source_path)
    source["issue_time"] = pd.to_datetime(source["issue_time"])
    shifted = source.copy()
    shifted["issue_time"] = shifted["issue_time"] - pd.Timedelta(hours=1)
    shifted["issue_sec"] = shifted["issue_time"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    ANCHOR_ALIGNMENT_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    shifted.to_csv(ANCHOR_ALIGNMENT_SELECTION_DIR / "issue_time_index.csv", index=False)
    return source


def finalize_stage3_alignment(
    source_issue_table: pd.DataFrame,
    cache_dir: Path,
    checkpoint_identity: dict,
) -> None:
    anchor_meta_csv, anchor_missing_csv, anchor_meta_json = anchor_paths(cache_dir)
    meta = pd.read_csv(anchor_meta_csv)
    meta["issue_time"] = pd.to_datetime(meta["issue_time"]) + pd.Timedelta(hours=1)
    meta["issue_sec"] = meta["issue_time"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    meta.to_csv(anchor_meta_csv, index=False)
    if anchor_missing_csv.exists():
        missing = pd.read_csv(anchor_missing_csv)
        if not missing.empty:
            missing["issue_time"] = pd.to_datetime(missing["issue_time"]) + pd.Timedelta(hours=1)
        missing.to_csv(anchor_missing_csv, index=False)
    source_issue_table.to_csv(cache_dir / "source_pv_issue_time_index.csv", index=False)
    with anchor_meta_json.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    info.update(
        {
            "stage3_temporal_protocol": {
                "pv_issue_time": "T is the first 15-minute PV target timestamp.",
                "pv_history": "T-3h through T-15min, 12 frames.",
                "pv_target": "T through T+1h45min, 8 frames.",
                "swr_history": "T-4h through T-1h, 4 observed hourly frames.",
                "swr_future_predictions": "T through T+2h, 3 predicted hourly frames.",
            },
            "stage3_alignment_adapter": (
                "The unchanged legacy SWR-anchor builder is invoked with T-1h internally, "
                "then metadata is restored to PV issue time T and validated."
            ),
            "stage3_swr_cache_version": {
                **checkpoint_identity,
                "cache_dir": str(cache_dir.resolve()),
                "spatial_resample_schema": SPATIAL_RESAMPLE_SCHEMA,
                "switching_note": (
                    "Each Stage 2 SWR checkpoint is cached in its own directory. "
                    "Downstream Stage 3 scripts read the active_cache.json manifest."
                ),
            },
        }
    )
    with anchor_meta_json.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    validate_anchor_alignment(cache_dir)


def load_ratio_base_model(checkpoint: dict, device: str):
    stats = checkpoint["stats"]
    checkpoint_args = checkpoint.get("args", {})
    channels = [str(value) for value in stats["channels"]]
    temporal_head = str(
        checkpoint.get("temporal_head", checkpoint_args.get("temporal_head", "last_mean"))
    )
    history_frames = int(checkpoint_args.get("history_frames", stats.get("history_frames", 4)))
    future_frames = int(checkpoint_args.get("future_frames", stats.get("future_frames", 3)))
    if (history_frames, future_frames) != (4, 3):
        raise ValueError(
            "Stage 3 PV alignment requires a 4-history -> 3-future SWR checkpoint, "
            f"found {history_frames} -> {future_frames}."
        )
    model = stage2_ratio.Ratio3DCNN(
        in_channels=len(channels),
        future_frames=future_frames,
        history_frames=history_frames,
        width=int(checkpoint_args.get("width", 48)),
        depth=int(checkpoint_args.get("depth", 4)),
        dropout=float(checkpoint_args.get("dropout", 0.05)),
        ratio_clip=float(stats["ratio_clip"]),
        temporal_head=temporal_head,
    )
    state = checkpoint.get("model") or checkpoint.get("model_state")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, stats, channels, temporal_head


class ResidualRatioInference:
    """Expose frozen base + residual physical correction as a ratio predictor."""

    def __init__(self, base_model, residual_model, base_stats, residual_stats, residual_args, device):
        import torch

        self.base_model = base_model
        self.residual_model = residual_model
        self.base_stats = base_stats
        self.residual_stats = residual_stats
        self.residual_args = residual_args
        self.device = device
        self.future_clear = None
        self.torch = torch

    def set_future_clear(self, clear_sky_wm2) -> None:
        clear = self.torch.as_tensor(clear_sky_wm2, dtype=self.torch.float32, device=self.device)
        self.future_clear = clear.unsqueeze(0)

    def __call__(self, x_norm):
        if self.future_clear is None:
            raise RuntimeError("Residual SWR inference requires set_future_clear() before forward().")
        clear = self.future_clear
        if clear.shape[0] != x_norm.shape[0]:
            clear = clear.expand(x_norm.shape[0], -1, -1, -1)
        base_ratio = self.base_model(x_norm)
        pred_clip = float(self.residual_args.get("pred_clip", 1200.0))
        residual_clip = float(self.residual_args.get("residual_clip", 200.0))
        residual_scale = float(self.residual_args.get("residual_scale", 100.0))
        base_phy = self.torch.clamp(base_ratio.float() * clear, 0.0, pred_clip)
        residual_input = stage2_residual.build_residual_input(
            x_norm,
            base_phy,
            float(self.residual_stats["swr_mean"]),
            float(self.residual_stats["swr_std"]),
        )
        residual_phy = self.torch.clamp(
            self.residual_model(residual_input).float() * residual_scale,
            -residual_clip,
            residual_clip,
        )
        final_phy = self.torch.clamp(base_phy + residual_phy, 0.0, pred_clip)
        min_clear = float(self.base_stats["min_clear_sky"])
        ratio_clip = float(self.base_stats["ratio_clip"])
        final_ratio = final_phy / clear.clamp_min(min_clear)
        final_ratio = self.torch.where(clear >= min_clear, final_ratio, self.torch.zeros_like(final_ratio))
        return self.torch.clamp(final_ratio, 0.0, ratio_clip)


def load_stage2_swr_model(args):
    import torch

    checkpoint = torch.load(Path(args.swr_checkpoint), map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model_name = str(checkpoint.get("model_name", checkpoint_args.get("model_name", "ratio_3dcnn")))
    device = args.device if not str(args.device).lower().startswith("cuda") or torch.cuda.is_available() else "cpu"
    if model_name == "ratio_3dcnn":
        model, stats, channels, temporal_head = load_ratio_base_model(checkpoint, device)
    elif model_name in {
        "stage2_residual_convlstm_correction",
        "residual_convlstm_correction_on_big3dcnn_ratio",
    }:
        base_checkpoint_path = Path(checkpoint["base_checkpoint"])
        if not base_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Residual SWR checkpoint requires frozen base checkpoint: {base_checkpoint_path}"
            )
        base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
        base_model, stats, channels, temporal_head = load_ratio_base_model(base_checkpoint, device)
        residual_future_frames = int(checkpoint_args.get("future_frames", 3))
        if residual_future_frames != 3:
            raise ValueError(
                "Stage 3 PV alignment requires three residual future frames, "
                f"found {residual_future_frames}."
            )
        residual_model = stage2_residual.ResidualConvLSTMCorrection(
            in_channels=int(checkpoint_args.get("input_channels", len(channels) + 2)),
            future_frames=residual_future_frames,
            width=int(checkpoint_args.get("width", 48)),
            depth=int(checkpoint_args.get("depth", 2)),
            dropout=float(checkpoint_args.get("dropout", 0.05)),
        )
        residual_model.load_state_dict(checkpoint["model"])
        residual_model.to(device).eval()
        model = ResidualRatioInference(
            base_model,
            residual_model,
            stats,
            checkpoint["residual_stats"],
            checkpoint_args,
            device,
        )
        print("Stage 3 residual frozen base:", base_checkpoint_path)
        print("Stage 3 residual base temporal head:", temporal_head)
        print("Stage 3 residual loss variant:", checkpoint.get("loss_variant", "unknown"))
    else:
        raise ValueError(
            f"Unsupported Stage 3 SWR checkpoint model_name={model_name!r}. "
            "Supported adapters: Ratio3DCNN and residual ConvLSTM correction on Ratio3DCNN."
        )
    print("Stage 3 SWR checkpoint:", args.swr_checkpoint)
    print("Stage 3 SWR model:", model_name)
    print("Stage 3 SWR temporal head:", temporal_head)
    return model, stats, channels, device


def main() -> None:
    ensure_stage3_dirs()
    wrapper_args = parse_wrapper_args()
    cache_dir, checkpoint_identity = cache_dir_for_checkpoint(
        wrapper_args.swr_checkpoint,
        cache_name=wrapper_args.cache_name,
        cache_dir=wrapper_args.cache_dir,
    )
    anchor_meta_csv, _, _ = anchor_paths(cache_dir)
    anchor_path = cache_dir / "swr_anchor_64.npy"
    print("Stage 3 future-SWR cache dir:", cache_dir, flush=True)
    print("Stage 3 checkpoint signature:", checkpoint_identity["checkpoint_signature"], flush=True)
    if anchor_path.exists() and anchor_meta_csv.exists() and not force_requested():
        if not cache_matches_checkpoint(cache_dir, checkpoint_identity):
            raise RuntimeError(
                f"Existing cache metadata does not match checkpoint {wrapper_args.swr_checkpoint}. "
                "Use a different --cache_name or rebuild this cache with --force 1."
            )
        cache_info = read_cache_info(cache_dir)
        spatial_schema = str(
            cache_info.get("stage3_swr_cache_version", {}).get("spatial_resample_schema", "")
        )
        if spatial_schema != SPATIAL_RESAMPLE_SCHEMA:
            raise RuntimeError(
                "Existing Stage 3 future-SWR cache predates the corrected 2D bilinear resampling. "
                "Rebuild it once with --force 1."
            )
        try:
            validate_anchor_alignment(cache_dir)
        except AssertionError as exc:
            raise RuntimeError(
                "Existing Stage 3 SWR-anchor cache uses the old temporal offset. "
                "Rebuild with --force 1."
            ) from exc
        if int(wrapper_args.activate):
            activate_cache(cache_dir, checkpoint_identity)
        print("reuse:", cache_dir)
        return
    source_issue_table = prepare_shifted_selection()
    module.PYCHARM_DEFAULTS.update(
        {
            "selection_dir": str(ANCHOR_ALIGNMENT_SELECTION_DIR),
            "target_run_dir": str(TARGET_EXCLUDED_RUN_DIR),
            "out_dir": str(cache_dir),
            "h8_center_index_csv": str(STAGE1_CENTER_INDEX_ALL),
            "h8_center_frame_cache_dir": str(STAGE1_CENTER_FRAME_CACHE_DIR),
            "h8_center_processed_root": str(STAGE1_CENTER_MONTHLY_DIR),
            "h8_netcdf_glob": "*.nc",
            "swr_checkpoint": str(wrapper_args.swr_checkpoint),
            "swr_history_frames": 4,
            "swr_future_anchors": 3,
        }
    )
    module.Ratio3DCNN = stage2_ratio.Ratio3DCNN
    module.RatioFrameReader = stage2_ratio.RatioFrameReader
    module.load_swr_model = load_stage2_swr_model
    inject_defaults(module.PYCHARM_DEFAULTS)
    module.main()
    finalize_stage3_alignment(source_issue_table, cache_dir, checkpoint_identity)
    if int(wrapper_args.activate):
        activate_cache(cache_dir, checkpoint_identity)


if __name__ == "__main__":
    main()
