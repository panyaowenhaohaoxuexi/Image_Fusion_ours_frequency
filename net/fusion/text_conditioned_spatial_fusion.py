# -*- coding: utf-8 -*-
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ConvBNAct, ResidualBlock
from net.restormer_light import TransformerBlock

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class SpatialAlphaFusionBlock(nn.Module):
    """Position-adaptive spatial fusion: alpha * visible + (1-alpha) * infrared."""

    def __init__(self, channels: int = 64, intent_dim: int = 64, use_feedback: bool = True):
        super().__init__()
        self.use_feedback = use_feedback
        in_channels = channels * 3 + intent_dim + (1 if use_feedback else 0)
        self.intent_proj = nn.Linear(intent_dim, intent_dim)
        self.alpha_net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(ConvBNAct(channels, channels, 3, 1, 1, 'gelu'), ResidualBlock(channels))

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor,
                spatial_intent: torch.Tensor, feedback_gate: Optional[torch.Tensor] = None):
        b, _, h, w = vis_feat.shape
        intent_map = self.intent_proj(spatial_intent).view(b, -1, 1, 1).expand(-1, -1, h, w)
        context = [vis_feat, ir_feat, torch.abs(vis_feat - ir_feat), intent_map]
        if self.use_feedback:
            if feedback_gate is None:
                feedback_gate = torch.zeros(b, 1, h, w, device=vis_feat.device, dtype=vis_feat.dtype)
            elif feedback_gate.shape[-2:] != (h, w):
                feedback_gate = F.interpolate(feedback_gate, size=(h, w), mode='bilinear', align_corners=False)
            context.append(feedback_gate)
        alpha = self.alpha_net(torch.cat(context, dim=1))
        fused = alpha * vis_feat + (1.0 - alpha) * ir_feat
        fused = self.refine(fused) + fused
        return fused, alpha


class FeedbackTopDownBlock(nn.Module):
    """Top-down aggregation conditioned by BFSC feedback gate."""

    def __init__(self, channels: int = 64, num_heads: int = 1, ffn_expansion_factor: float = 2.0):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=False)
        self.refine = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, False, 'WithBias'),
            ConvBNAct(channels, channels, 3, 1, 1, 'gelu'),
            ResidualBlock(channels),
        )

    def forward(self, high_feat: torch.Tensor, skip_feat: torch.Tensor,
                feedback_gate: Optional[torch.Tensor] = None):
        high_up = F.interpolate(high_feat, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        if feedback_gate is None:
            feedback_gate = torch.zeros(skip_feat.shape[0], 1, skip_feat.shape[-2], skip_feat.shape[-1],
                                        device=skip_feat.device, dtype=skip_feat.dtype)
        elif feedback_gate.shape[-2:] != skip_feat.shape[-2:]:
            feedback_gate = F.interpolate(feedback_gate, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([high_up, skip_feat, feedback_gate], dim=1))
        return self.refine(x) + x


class TGCSF(nn.Module):
    """Text-guided cross-modal spatial fusion with position-adaptive alpha maps."""

    def __init__(self, channels: int = 64, intent_dim: int = 64, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, init_res_scale: float = 0.20,
                 use_freq_context: bool = True, max_attn_size: int = 32,
                 norm_groups: int = 16):
        super().__init__()
        self.level1_alpha = SpatialAlphaFusionBlock(channels, intent_dim, use_feedback=True)
        self.level2_alpha = SpatialAlphaFusionBlock(channels, intent_dim, use_feedback=True)
        self.level3_alpha = SpatialAlphaFusionBlock(channels, intent_dim, use_feedback=True)
        self.coarse_refine = nn.Sequential(ConvBNAct(channels, channels, 3, 1, 1, 'gelu'), ResidualBlock(channels))
        self.topdown_l2 = FeedbackTopDownBlock(channels, num_heads, ffn_expansion_factor)
        self.topdown_l1 = FeedbackTopDownBlock(channels, num_heads, ffn_expansion_factor)
        self.final_refine = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, False, 'WithBias'),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
        self.out_norm = nn.BatchNorm2d(channels)
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

    @staticmethod
    def _as_three_levels(x: TensorOrPyramid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)):
            if len(x) < 3:
                raise ValueError('spatial feature pyramid must contain [L1, L2, L3].')
            return x[0], x[1], x[2]
        l1 = x
        l2 = F.avg_pool2d(l1, kernel_size=2, stride=2)
        l3 = F.avg_pool2d(l2, kernel_size=2, stride=2)
        return l1, l2, l3

    def forward(self, vis_spa: TensorOrPyramid, ir_spa: TensorOrPyramid, spatial_intent: torch.Tensor,
                feedback_gate: Optional[torch.Tensor] = None, coarse_only: bool = False,
                return_aux: bool = False):
        vis_l1, vis_l2, vis_l3 = self._as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = self._as_three_levels(ir_spa)

        if coarse_only:
            fused_l3, alpha_l3 = self.level3_alpha(vis_l3, ir_l3, spatial_intent, feedback_gate=None)
            coarse = self.coarse_refine(fused_l3) + fused_l3
            coarse = F.interpolate(coarse, size=vis_l1.shape[-2:], mode='bilinear', align_corners=False)
            if not return_aux:
                return coarse
            return coarse, {'coarse_l3': fused_l3, 'alpha_l3': alpha_l3}

        fused_l1, alpha_l1 = self.level1_alpha(vis_l1, ir_l1, spatial_intent, feedback_gate)
        fused_l2, alpha_l2 = self.level2_alpha(vis_l2, ir_l2, spatial_intent, feedback_gate)
        fused_l3, alpha_l3 = self.level3_alpha(vis_l3, ir_l3, spatial_intent, feedback_gate)
        td_l2 = self.topdown_l2(fused_l3, fused_l2, feedback_gate)
        td_l1 = self.topdown_l1(td_l2, fused_l1, feedback_gate)
        refined = self.final_refine(td_l1)
        out = self.out_norm(td_l1 + self.res_scale * refined)

        if not return_aux:
            return out
        aux: Dict[str, torch.Tensor] = {
            'l1_fused': fused_l1,
            'l2_fused': fused_l2,
            'l3_fused': fused_l3,
            'td_l2': td_l2,
            'td_l1': td_l1,
            'alpha_l1': alpha_l1,
            'alpha_l2': alpha_l2,
            'alpha_l3': alpha_l3,
            'feedback_gate': feedback_gate if feedback_gate is not None else torch.zeros_like(alpha_l1),
            'spatial_res_scale': self.res_scale.detach(),
        }
        return out, aux


class TextConditionedSpatialAdaptiveFusion(TGCSF):
    """Backward-compatible name for TGCSF."""
    pass
