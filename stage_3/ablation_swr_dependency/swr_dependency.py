"""SWR-dependency ablation helpers for the Stage 3 PV-field model."""

from __future__ import annotations

import numpy as np


SWR_DEPENDENCY_MODES = ("full_swr_future", "history_swr_only", "no_swr")


def transform_swr_anchor(
    swr_anchor: np.ndarray,
    mode: str,
    history_frames: int = 4,
    future_anchors: int = 3,
) -> np.ndarray:
    """Apply the SWR dependency ablation to an already normalized anchor tensor.

    Expected shape is [anchor_time, channel, height, width]. The default Stage 3
    setting is 4 observed history anchors followed by 3 future prediction anchors.
    """
    mode = str(mode)
    if mode not in SWR_DEPENDENCY_MODES:
        raise ValueError(f"Unknown SWR dependency mode: {mode}")
    out = np.asarray(swr_anchor, dtype=np.float32).copy()
    if mode == "full_swr_future":
        return out
    if mode == "history_swr_only":
        latest_idx = int(history_frames) - 1
        future_start = int(history_frames)
        future_end = future_start + int(future_anchors)
        if out.shape[0] < future_end:
            raise ValueError(
                f"Expected at least {future_end} SWR anchors for history_swr_only, got {out.shape[0]}"
            )
        out[future_start:future_end] = out[latest_idx : latest_idx + 1]
        return out
    if mode == "no_swr":
        out[...] = 0.0
        return out
    raise AssertionError(mode)


def install_swr_dependency_dataset(module, mode: str) -> None:
    """Patch module.SWRAnchorPVFieldDataset so __getitem__ returns ablated SWR anchors."""
    mode = str(mode)
    if mode not in SWR_DEPENDENCY_MODES:
        raise ValueError(f"Unknown SWR dependency mode: {mode}")
    if mode == "full_swr_future":
        return
    original_cls = getattr(module, "_original_swr_dependency_dataset_cls", None)
    if original_cls is None:
        original_cls = module.SWRAnchorPVFieldDataset
        module._original_swr_dependency_dataset_cls = original_cls

    class SWRDependencyDataset(original_cls):
        swr_dependency_mode = mode

        def __getitem__(self, idx: int):
            x_aug, future_aug, swr, y, mask = super().__getitem__(idx)
            swr = transform_swr_anchor(
                swr,
                mode=mode,
                history_frames=int(getattr(self.args, "swr_history_frames", 4)),
                future_anchors=int(getattr(self.args, "swr_future_anchors", 3)),
            )
            return x_aug, future_aug, swr, y, mask

    module.SWRAnchorPVFieldDataset = SWRDependencyDataset
    print(f"Installed SWR-dependency ablation dataset: {mode}")
