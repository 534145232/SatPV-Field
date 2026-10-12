"""Stage 3 temporal-anchor adaptation for the existing 04l motion-fusion model."""

from __future__ import annotations

import types

import numpy as np


STAGE3_FUTURE_ANCHOR_WEIGHTS = np.asarray(
    [
        [0.50, 0.50, 0.00, 0.00],
        [0.25, 0.75, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.75, 0.25, 0.00],
        [0.00, 0.50, 0.50, 0.00],
        [0.00, 0.25, 0.75, 0.00],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.75, 0.25],
    ],
    dtype=np.float32,
)


def patch_stage3_temporal_anchors(model):
    """Shift the four interpolation anchors to [last observed, predicted 0:3]."""
    import torch

    if int(model.future_frames) != 8:
        raise AssertionError(f"Stage 3 expects 8 PV horizons, found {model.future_frames}")
    if int(model.swr_history_frames) != 4 or int(model.swr_future_anchors) != 3:
        raise AssertionError(
            "Stage 3 expects four observed SWR frames and three predicted SWR frames, "
            f"found {model.swr_history_frames}+{model.swr_future_anchors}"
        )
    if int(model.temporal_anchor_count) != 4:
        raise AssertionError(f"Stage 3 expects four interpolation anchors, found {model.temporal_anchor_count}")
    model._buffers["future_anchor_weights"] = torch.from_numpy(STAGE3_FUTURE_ANCHOR_WEIGHTS.copy())

    def forward(self, x, future_feat, swr_anchor, return_attention: bool = False):
        pv_feat = self.encode_pv(x)
        anchor_feat = self.encode_anchors(swr_anchor)
        # Stage 3 anchors are latest observed SWR plus all three predictions:
        # A0=T-1h, A1=T, A2=T+1h, A3=T+2h. Their label centers are
        # T-30min, T+30min, T+90min and T+150min.
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
            flow = self.max_flow_pixels * torch.tanh(self.flow_head(motion_in))
            warped = self.warp_with_flow(last_pv, flow)
            direct = self.direct_decoder(motion_in)
            gate = self.fusion_gate(torch.cat([motion_in, warped, direct], dim=1))
            preds.append((gate * warped + (1.0 - gate) * direct).squeeze(1))
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
    print("Installed Stage 3 SWR interpolation anchors: [T-1h, T, T+1h, T+2h]")
    print("Stage 3 PV-horizon SWR weights:")
    print(STAGE3_FUTURE_ANCHOR_WEIGHTS)
    return model


def install_stage3_motion_fusion_factory(module) -> None:
    """Patch only the imported Stage 3 training module; leave the old project untouched."""
    original_factory = module.make_swr_anchor_attention_model

    def make_stage3_model(*args, **kwargs):
        return patch_stage3_temporal_anchors(original_factory(*args, **kwargs))

    module.make_swr_anchor_attention_model = make_stage3_model
