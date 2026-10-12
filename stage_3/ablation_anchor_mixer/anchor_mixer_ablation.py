"""Anchor-mixer ablations for the selected Stage 3 model."""

from __future__ import annotations


def patch_no_anchor_mixer(model, anchor_hidden: int):
    """Remove learned all-anchor context while preserving interpolated SWR maps."""
    import torch.nn as nn

    class ZeroAnchorContext(nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.out_channels = int(out_channels)

        def forward(self, x):
            return x.new_zeros((x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]))

    model.anchor_mixer = ZeroAnchorContext(int(anchor_hidden))
    model.anchor_mixer_mode = "none_zero_context"
    return model


def install_no_anchor_mixer_factory(module) -> None:
    """Patch a model factory so the 48-channel anchor context is always zero."""
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        anchor_hidden = kwargs.get("anchor_hidden")
        if anchor_hidden is None:
            raise ValueError("anchor_hidden must be passed by keyword for the anchor-mixer ablation")
        return patch_no_anchor_mixer(model, int(anchor_hidden))

    module.make_swr_anchor_attention_model = make_model

