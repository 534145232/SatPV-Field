"""Learnable temporal-anchor weights for the Stage 3 motion-fusion ablation."""

from __future__ import annotations

import types

import numpy as np


def initial_logits(init_mode: str, fixed_weights: np.ndarray, seed: int, random_std: float) -> np.ndarray:
    mode = str(init_mode).lower()
    if mode == "physical":
        return np.log(np.clip(np.asarray(fixed_weights, dtype=np.float32), 1.0e-6, None))
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return rng.normal(0.0, float(random_std), size=fixed_weights.shape).astype(np.float32)
    raise ValueError(f"Unknown learnable anchor-weight initialization: {init_mode}")


def current_weights(model):
    import torch

    return torch.softmax(model.future_anchor_logits, dim=1)


def patch_learnable_stage3_temporal_anchors(
    model,
    fixed_weights: np.ndarray,
    init_mode: str,
    seed: int = 2026,
    random_std: float = 0.10,
):
    """Replace the fixed 8x4 interpolation table with row-wise softmax logits."""
    import torch
    import torch.nn as nn

    fixed = np.asarray(fixed_weights, dtype=np.float32)
    if fixed.shape != (int(model.future_frames), int(model.temporal_anchor_count)):
        raise ValueError(
            "Anchor-weight shape mismatch: "
            f"expected {(int(model.future_frames), int(model.temporal_anchor_count))}, got {fixed.shape}"
        )
    if int(model.future_frames) != 8:
        raise AssertionError(f"Stage 3 expects 8 PV horizons, found {model.future_frames}")
    if int(model.swr_history_frames) != 4 or int(model.swr_future_anchors) != 3:
        raise AssertionError(
            "Stage 3 expects four observed SWR frames and three predicted SWR frames, "
            f"found {model.swr_history_frames}+{model.swr_future_anchors}"
        )
    if "future_anchor_weights" in model._buffers:
        del model._buffers["future_anchor_weights"]
    if hasattr(model, "future_anchor_weights"):
        delattr(model, "future_anchor_weights")
    logits = initial_logits(init_mode, fixed, seed, random_std)
    model.register_parameter("future_anchor_logits", nn.Parameter(torch.from_numpy(logits)))
    model.anchor_weight_init_mode = str(init_mode)
    model.anchor_weight_random_std = float(random_std)
    model.current_future_anchor_weights = types.MethodType(lambda self: current_weights(self), model)

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
        preds, weight_maps, aligned_swr_maps = [], [], []
        warped_maps, direct_maps, gate_maps, flow_maps = [], [], [], []
        b, _, h, w = pv_feat.shape
        last_pv = x[:, 0, -1][:, None]
        anchor_feat_flat = temporal_anchor_feat.reshape(
            b, self.temporal_anchor_count * temporal_anchor_feat.shape[2], h, w
        )
        anchor_map_flat = temporal_anchor_map.reshape(
            b, self.temporal_anchor_count * self.anchor_channels, h, w
        )
        all_weights = self.current_future_anchor_weights()
        for horizon in range(self.future_frames):
            fut = self.future_proj(future_feat[:, horizon])
            he = self.horizon_embed(
                torch.full((b,), horizon, dtype=torch.long, device=x.device)
            )[:, :, None, None].expand(-1, -1, h, w)
            weights = all_weights[horizon].to(x.device, dtype=temporal_anchor_feat.dtype)
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
                weight_maps.append(full)
                aligned_swr_maps.append(aligned_swr_map)
                warped_maps.append(warped.squeeze(1))
                direct_maps.append(direct.squeeze(1))
                gate_maps.append(gate.squeeze(1))
                flow_maps.append(flow)
        pred_stack = torch.stack(preds, dim=1)
        if return_attention:
            return (
                pred_stack,
                torch.stack(weight_maps, dim=1),
                torch.stack(aligned_swr_maps, dim=1),
                torch.stack(warped_maps, dim=1),
                torch.stack(direct_maps, dim=1),
                torch.stack(gate_maps, dim=1),
                torch.stack(flow_maps, dim=1),
            )
        return pred_stack

    model.forward = types.MethodType(forward, model)
    print(f"Installed learnable Stage 3 SWR anchor weights: init={init_mode}")
    print(model.current_future_anchor_weights().detach().cpu().numpy())
    return model


def install_learnable_anchor_weight_factory(
    module,
    fixed_weights: np.ndarray,
    init_mode: str,
    seed: int = 2026,
    random_std: float = 0.10,
) -> None:
    """Patch only the supplied local model module."""
    original_factory = module.make_swr_anchor_attention_model

    def make_model(*args, **kwargs):
        return patch_learnable_stage3_temporal_anchors(
            original_factory(*args, **kwargs),
            fixed_weights=fixed_weights,
            init_mode=init_mode,
            seed=seed,
            random_std=random_std,
        )

    module.make_swr_anchor_attention_model = make_model
