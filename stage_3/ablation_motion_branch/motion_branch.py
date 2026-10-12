"""Motion-branch ablations for the fixed-anchor Stage 3 PV-field model."""

from __future__ import annotations

import types

from stage3_motion_fusion import patch_stage3_temporal_anchors


MOTION_BRANCHES = ("direct_only", "warp_only", "warp_direct_gate")


def patch_motion_branch(model, mode: str):
    """Select the active output branch while preserving the established inputs."""
    import torch

    mode = str(mode)
    if mode not in MOTION_BRANCHES:
        raise ValueError(f"Unknown motion branch: {mode}. Expected one of {MOTION_BRANCHES}")
    model = patch_stage3_temporal_anchors(model)
    model.motion_branch_mode = mode

    def forward(self, x, future_feat, swr_anchor, return_attention: bool = False):
        pv_feat = self.encode_pv(x)
        anchor_feat = self.encode_anchors(swr_anchor)
        anchor_start = self.swr_history_frames - 1
        anchor_end = anchor_start + self.temporal_anchor_count
        temporal_anchor_feat = anchor_feat[:, anchor_start:anchor_end]
        temporal_anchor_map = swr_anchor[:, anchor_start:anchor_end]
        if temporal_anchor_feat.shape[1] != self.temporal_anchor_count:
            raise ValueError(
                f"Expected {self.temporal_anchor_count} temporal SWR anchors, got {temporal_anchor_feat.shape[1]}"
            )
        past_swr_temporal_feat = self.encode_past_swr_temporal(swr_anchor)
        alignment_ctx = self.alignment_proj(self.encode_alignment(pv_feat, past_swr_temporal_feat))
        preds, fixed_weight_maps, aligned_swr_maps = [], [], []
        warped_maps, direct_maps, gate_maps, flow_maps = [], [], [], []
        b, _, h, w = pv_feat.shape
        last_pv = x[:, 0, -1][:, None]
        anchor_feat_flat = temporal_anchor_feat.reshape(
            b, self.temporal_anchor_count * temporal_anchor_feat.shape[2], h, w
        )
        anchor_map_flat = temporal_anchor_map.reshape(
            b, self.temporal_anchor_count * self.anchor_channels, h, w
        )
        for horizon in range(self.future_frames):
            fut = self.future_proj(future_feat[:, horizon])
            he = self.horizon_embed(
                torch.full((b,), horizon, dtype=torch.long, device=x.device)
            )[:, :, None, None].expand(-1, -1, h, w)
            weights = self.future_anchor_weights[horizon].to(
                x.device, dtype=temporal_anchor_feat.dtype
            )
            weight_map = weights[None, :, None, None].expand(b, -1, h, w)
            anchor_ctx = self.anchor_mixer(
                torch.cat([anchor_feat_flat, anchor_map_flat, weight_map], dim=1)
            )
            future_swr_map = (
                temporal_anchor_map * weights[None, :, None, None, None]
            ).sum(dim=1)
            swr_gate = self.swr_alignment_gate(torch.cat([alignment_ctx, future_swr_map], dim=1))
            aligned_swr_map = future_swr_map * (0.5 + swr_gate)
            motion_in = torch.cat(
                [
                    pv_feat,
                    fut,
                    past_swr_temporal_feat,
                    alignment_ctx,
                    anchor_ctx,
                    future_swr_map,
                    aligned_swr_map,
                    he,
                ],
                dim=1,
            )
            if mode == "direct_only":
                direct = self.direct_decoder(motion_in)
                flow = torch.zeros((b, 2, h, w), dtype=direct.dtype, device=direct.device)
                warped = last_pv
                gate = torch.zeros_like(direct)
                pred = direct
            elif mode == "warp_only":
                flow = self.max_flow_pixels * torch.tanh(self.flow_head(motion_in))
                warped = self.warp_with_flow(last_pv, flow)
                direct = torch.zeros_like(warped)
                gate = torch.ones_like(warped)
                pred = warped
            else:
                flow = self.max_flow_pixels * torch.tanh(self.flow_head(motion_in))
                warped = self.warp_with_flow(last_pv, flow)
                direct = self.direct_decoder(motion_in)
                gate = self.fusion_gate(torch.cat([motion_in, warped, direct], dim=1))
                pred = gate * warped + (1.0 - gate) * direct
            preds.append(pred.squeeze(1))
            if return_attention:
                full = torch.zeros(
                    b,
                    self.swr_history_frames + self.swr_future_anchors,
                    h,
                    w,
                    dtype=temporal_anchor_feat.dtype,
                    device=x.device,
                )
                full[:, anchor_start:anchor_end] = weights[None, :, None, None].expand(
                    b, -1, h, w
                )
                fixed_weight_maps.append(full)
                aligned_swr_maps.append(aligned_swr_map)
                warped_maps.append(warped.squeeze(1))
                direct_maps.append(direct.squeeze(1))
                gate_maps.append(gate.squeeze(1))
                flow_maps.append(flow)
        pred_stack = torch.stack(preds, dim=1)
        if return_attention:
            return (
                pred_stack,
                torch.stack(fixed_weight_maps, dim=1),
                torch.stack(aligned_swr_maps, dim=1),
                torch.stack(warped_maps, dim=1),
                torch.stack(direct_maps, dim=1),
                torch.stack(gate_maps, dim=1),
                torch.stack(flow_maps, dim=1),
            )
        return pred_stack

    model.forward = types.MethodType(forward, model)
    print("Installed Stage 3 motion-branch ablation:", mode)
    return model


def install_motion_branch_factory(module, mode: str) -> None:
    """Patch only the imported training module."""
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        return patch_motion_branch(original_factory(*args, **kwargs), mode)

    module.make_swr_anchor_attention_model = make_model
