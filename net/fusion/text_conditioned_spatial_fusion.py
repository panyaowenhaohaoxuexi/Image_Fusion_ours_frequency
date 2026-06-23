# -*- coding: utf-8 -*-
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ResidualBlock
from net.restormer_light import TransformerBlock

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class SemanticAffineModulation(nn.Module):
    """Modality-specific IN + affine modulation generated from z_fus."""

    def __init__(self, channels: int = 64, intent_dim: int = 64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
        self.mlp = nn.Sequential(
            nn.Linear(intent_dim, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.gamma_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.beta_proj = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, feat: torch.Tensor, z_fus: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        gamma, beta = self.mlp(z_fus).chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1).expand(-1, -1, h, w)
        beta = beta.view(b, c, 1, 1).expand(-1, -1, h, w)
        gamma = self.gamma_proj(gamma)
        beta = self.beta_proj(beta)
        return gamma * self.norm(feat) + beta


class PositionAdaptiveWeightGate(nn.Module):
    """Single-channel IR gate conditioned on local VIS/IR features and I_fus."""

    def __init__(self, channels: int = 64, intent_dim: int = 64):
        super().__init__()
        self.intent_proj = nn.Sequential(
            nn.Linear(intent_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, max(channels // 2, 1), 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 2, 1), 1, 1, 1, 0),
        )

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, spatial_intent: torch.Tensor) -> torch.Tensor:
        b, c, h, w = vis_feat.shape
        intent_map = self.intent_proj(spatial_intent).view(b, c, 1, 1).expand(-1, -1, h, w)
        return torch.sigmoid(self.gate(torch.cat([vis_feat, ir_feat, intent_map], dim=1)))


class ShallowSemanticFusionBlock(nn.Module):
    """L1/L2: semantic modulation followed by lightweight feature fusion."""

    def __init__(self, channels: int = 64, intent_dim: int = 64):
        super().__init__()
        self.vis_mod = SemanticAffineModulation(channels, intent_dim)
        self.ir_mod = SemanticAffineModulation(channels, intent_dim)
        self.weight_gate = PositionAdaptiveWeightGate(channels, intent_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(channels),
        )

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, z_fus: torch.Tensor):
        vis_tilde = self.vis_mod(vis_feat, z_fus)
        ir_tilde = self.ir_mod(ir_feat, z_fus)
        weight_ir = self.weight_gate(vis_feat, ir_feat, z_fus)
        gated = weight_ir * ir_tilde + (1.0 - weight_ir) * vis_tilde
        fused = self.fuse(gated)
        return fused, {'vis_tilde': vis_tilde, 'ir_tilde': ir_tilde, 'weight': weight_ir}


class DeepSemanticCrossModalFusionBlock(nn.Module):
    """L3: semantic modulation, cross-modal attention, and context refinement."""

    def __init__(self, channels: int = 64, intent_dim: int = 64, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0):
        super().__init__()
        heads = _valid_heads(channels, num_heads)
        self.vis_mod = SemanticAffineModulation(channels, intent_dim)
        self.ir_mod = SemanticAffineModulation(channels, intent_dim)
        self.weight_gate = PositionAdaptiveWeightGate(channels, intent_dim)
        self.vis_to_ir = nn.MultiheadAttention(channels, heads)
        self.ir_to_vis = nn.MultiheadAttention(channels, heads)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
        )
        self.context = TransformerBlock(channels, heads, ffn_expansion_factor, False, 'WithBias')
        self.residual = ResidualBlock(channels)

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

    @staticmethod
    def _to_map(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, z_fus: torch.Tensor):
        _, _, h, w = vis_feat.shape
        vis_tilde = self.vis_mod(vis_feat, z_fus)
        ir_tilde = self.ir_mod(ir_feat, z_fus)
        vis_tokens = self._to_tokens(vis_tilde)
        ir_tokens = self._to_tokens(ir_tilde)
        vis_update_t, _ = self.vis_to_ir(
            vis_tokens.transpose(0, 1), ir_tokens.transpose(0, 1), ir_tokens.transpose(0, 1)
        )
        ir_update_t, _ = self.ir_to_vis(
            ir_tokens.transpose(0, 1), vis_tokens.transpose(0, 1), vis_tokens.transpose(0, 1)
        )
        vis_update = vis_update_t.transpose(0, 1)
        ir_update = ir_update_t.transpose(0, 1)
        vis_hat = self._to_map(vis_tokens + vis_update, h, w)
        ir_hat = self._to_map(ir_tokens + ir_update, h, w)
        weight_ir = self.weight_gate(vis_feat, ir_feat, z_fus)
        gated = weight_ir * ir_hat + (1.0 - weight_ir) * vis_hat
        fused = self.fuse(gated)
        fused = self.context(fused)
        fused = self.residual(fused) + fused
        return fused, {
            'vis_tilde': vis_tilde,
            'ir_tilde': ir_tilde,
            'vis_hat': vis_hat,
            'ir_hat': ir_hat,
            'weight': weight_ir,
        }


class TopDownSemanticBlock(nn.Module):
    def __init__(self, channels: int = 64, num_heads: int = 1, ffn_expansion_factor: float = 2.0):
        super().__init__()
        heads = _valid_heads(channels, num_heads)
        self.fuse = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = TransformerBlock(channels, heads, ffn_expansion_factor, False, 'WithBias')

    def forward(self, high_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high_feat, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(self.fuse(torch.cat([high_up, skip_feat], dim=1)))


class TGCSF(nn.Module):
    """Three-level semantically parameterized cross-modal spatial fusion."""

    def __init__(self, channels: int = 64, intent_dim: int = 64, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, init_res_scale: float = 0.20,
                 use_freq_context: bool = True, max_attn_size: int = 32,
                 norm_groups: int = 16):
        super().__init__()
        self.level1 = ShallowSemanticFusionBlock(channels, intent_dim)
        self.level2 = ShallowSemanticFusionBlock(channels, intent_dim)
        self.level3 = DeepSemanticCrossModalFusionBlock(channels, intent_dim, num_heads, ffn_expansion_factor)
        self.topdown_l2 = TopDownSemanticBlock(channels, num_heads, ffn_expansion_factor)
        self.topdown_l1 = TopDownSemanticBlock(channels, num_heads, ffn_expansion_factor)
        self.final_refine = TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, False, 'WithBias')
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
                return_aux: bool = False):
        vis_l1, vis_l2, vis_l3 = self._as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = self._as_three_levels(ir_spa)

        fused_l3, aux_l3 = self.level3(vis_l3, ir_l3, spatial_intent)
        fused_l1, aux_l1 = self.level1(vis_l1, ir_l1, spatial_intent)
        fused_l2, aux_l2 = self.level2(vis_l2, ir_l2, spatial_intent)
        td_l2 = self.topdown_l2(fused_l3, fused_l2)
        td_l1 = self.topdown_l1(td_l2, fused_l1)
        refined = self.final_refine(td_l1)
        out = self.out_norm(td_l1 + self.res_scale * refined)

        if not return_aux:
            return out
        aux: Dict[str, torch.Tensor] = {
            'fused_l1': fused_l1,
            'fused_l2': fused_l2,
            'fused_l3': fused_l3,
            'td_l2': td_l2,
            'td_l1': td_l1,
            'l1_aux': aux_l1,
            'l2_aux': aux_l2,
            'l3_aux': aux_l3,
            'weight_l1': aux_l1['weight'],
            'weight_l2': aux_l2['weight'],
            'weight_l3': aux_l3['weight'],
            'spatial_res_scale': self.res_scale.detach(),
        }
        return out, aux


class TextConditionedSpatialAdaptiveFusion(TGCSF):
    """Backward-compatible name for TGCSF."""
    pass
