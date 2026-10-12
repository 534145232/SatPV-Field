"""Last-only temporal-collapse ablation for the Stage 3 motion-fusion model."""

from __future__ import annotations

import types

from stage3_motion_fusion import patch_stage3_temporal_anchors


def patch_last_only_collapse(model):
    """Use only the latest latent step in both temporal encoders."""

    def collapse(self, x):
        return x[:, :, -1]

    model.collapse = types.MethodType(collapse, model)
    model.temporal_collapse_mode = "last_only"
    print("Installed temporal-collapse ablation: last_only")
    print("PV encoder collapse: x[:, :, -1]")
    print("Past-SWR temporal encoder collapse: x[:, :, -1]")
    return model


def install_last_only_collapse_factory(module) -> None:
    """Install fixed Stage 3 anchors plus the last-only collapse ablation."""
    original_factory = module.make_swr_anchor_attention_model

    def make_last_only_model(*args, **kwargs):
        model = patch_stage3_temporal_anchors(original_factory(*args, **kwargs))
        return patch_last_only_collapse(model)

    module.make_swr_anchor_attention_model = make_last_only_model
