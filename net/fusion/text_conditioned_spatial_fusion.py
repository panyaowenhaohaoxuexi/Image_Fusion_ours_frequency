# -*- coding: utf-8 -*-
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ResidualBlock
from net.normalization import get_valid_group_count
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
        self.mod_scale = nn.Parameter(torch.tensor(0.1))

        final_linear = None
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                final_linear = module
                break
        if final_linear is None:
            raise RuntimeError("SemanticAffineModulation MLP has no Linear layer")
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)
        if self.gamma_proj.bias is not None:
            nn.init.zeros_(self.gamma_proj.bias)
        if self.beta_proj.bias is not None:
            nn.init.zeros_(self.beta_proj.bias)

    def forward(self, feat: torch.Tensor, z_fus: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        gamma, beta = self.mlp(z_fus).chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1).expand(-1, -1, h, w)
        beta = beta.view(b, c, 1, 1).expand(-1, -1, h, w)
        gamma = self.gamma_proj(gamma)
        beta = self.beta_proj(beta)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        norm_feat = self.norm(feat)
        semantic_delta = gamma * norm_feat + beta
        scale = torch.tanh(self.mod_scale)
        return feat + scale * semantic_delta


class PositionAdaptiveWeightGate(nn.Module):
    """Channel-spatial IR gate plus a learned image-level aggregation map.

    The returned channel gate has shape ``[B, C, H, W]`` and is used for
    latent feature fusion.  The auxiliary image gate has shape ``[B, 1, H, W]``
    and is reserved for visualization and image reconstruction.
    """

    def __init__(self, channels: int = 64, intent_dim: int = 64):
        super().__init__()
        hidden = max(channels // 4, 8)
        groups = get_valid_group_count(channels)
        self.vis_gate_norm = nn.GroupNorm(groups, channels)
        self.ir_gate_norm = nn.GroupNorm(groups, channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.intent_proj = nn.Sequential(
            nn.Linear(intent_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.image_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        for output_layer in (
            self.channel_mlp[-1], self.spatial_gate[-1],
            self.intent_proj[-1], self.image_gate[-1],
        ):
            nn.init.normal_(output_layer.weight, mean=0.0, std=1e-3)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, spatial_intent: torch.Tensor):
        b, c, h, w = vis_feat.shape
        vis_gate_feat = self.vis_gate_norm(vis_feat)
        ir_gate_feat = self.ir_gate_norm(ir_feat)
        modality_feature = torch.cat([
            vis_gate_feat,
            ir_gate_feat,
            torch.abs(vis_gate_feat - ir_gate_feat),
            vis_gate_feat * ir_gate_feat,
        ], dim=1)
        intent_vector = self.intent_proj(spatial_intent)
        intent_logits = intent_vector.view(b, c, 1, 1)
        intent_map = intent_logits.expand(-1, -1, h, w)
        avg_descriptor = F.adaptive_avg_pool2d(modality_feature, 1)
        max_descriptor = F.adaptive_max_pool2d(modality_feature, 1)
        channel_logits = self.channel_mlp(avg_descriptor) + self.channel_mlp(max_descriptor)
        spatial_logits = self.spatial_gate(torch.cat([modality_feature, intent_map], dim=1))
        gate_logits = channel_logits + spatial_logits + intent_logits
        weight_ir_channel = torch.sigmoid(gate_logits)
        weight_ir_image = torch.sigmoid(self.image_gate(gate_logits))
        return weight_ir_channel, {
            'weight': weight_ir_image,
            'weight_channel': weight_ir_channel,
            'weight_channel_mean': weight_ir_channel.mean(dim=1, keepdim=True),
        }


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
        weight_ir_channel, gate_aux = self.weight_gate(vis_tilde, ir_tilde, z_fus)
        gated = weight_ir_channel * ir_tilde + (1.0 - weight_ir_channel) * vis_tilde
        fused = self.fuse(gated)
        return fused, {'vis_tilde': vis_tilde, 'ir_tilde': ir_tilde, **gate_aux}


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
        weight_ir_channel, gate_aux = self.weight_gate(vis_hat, ir_hat, z_fus)
        gated = weight_ir_channel * ir_hat + (1.0 - weight_ir_channel) * vis_hat
        fused = self.fuse(gated)
        fused = self.context(fused)
        fused = self.residual(fused) + fused
        return fused, {
            'vis_tilde': vis_tilde,
            'ir_tilde': ir_tilde,
            'vis_hat': vis_hat,
            'ir_hat': ir_hat,
            **gate_aux,
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
        self.image_gate_scale_logits = nn.Parameter(torch.tensor([3.0, 0.0, 0.0], dtype=torch.float32))

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
                return_aux: bool = False, return_pyramid: bool = False):
        vis_l1, vis_l2, vis_l3 = self._as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = self._as_three_levels(ir_spa)

        fused_l3, aux_l3 = self.level3(vis_l3, ir_l3, spatial_intent)
        fused_l1, aux_l1 = self.level1(vis_l1, ir_l1, spatial_intent)
        fused_l2, aux_l2 = self.level2(vis_l2, ir_l2, spatial_intent)
        td_l2 = self.topdown_l2(fused_l3, fused_l2)
        td_l1 = self.topdown_l1(td_l2, fused_l1)
        refined = self.final_refine(td_l1)
        out = self.out_norm(td_l1 + self.res_scale * refined)

        spatial_pyramid = {
            'l1': out,
            'l2': td_l2,
            'l3': fused_l3,
        }

        if not return_aux and not return_pyramid:
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
            'weight_raw_l1': aux_l1['weight'],
            'weight_raw_l2': aux_l2['weight'],
            'weight_raw_l3': aux_l3['weight'],
            'weight_channel_l1': aux_l1['weight_channel'],
            'weight_channel_l2': aux_l2['weight_channel'],
            'weight_channel_l3': aux_l3['weight_channel'],
            'weight_channel_mean_l1': aux_l1['weight_channel_mean'],
            'weight_channel_mean_l2': aux_l2['weight_channel_mean'],
            'weight_channel_mean_l3': aux_l3['weight_channel_mean'],
            'spatial_res_scale': self.res_scale.detach(),
        }
        raw_weight_l1 = aux_l1['weight']
        weight_l2_up = F.interpolate(aux_l2['weight'], size=raw_weight_l1.shape[-2:], mode='bilinear', align_corners=False)
        weight_l3_up = F.interpolate(aux_l3['weight'], size=raw_weight_l1.shape[-2:], mode='bilinear', align_corners=False)
        scale_weights = torch.softmax(self.image_gate_scale_logits, dim=0)
        aux['image_gate_scale_weights'] = scale_weights
        aux['weight_multiscale'] = (
            scale_weights[0] * raw_weight_l1
            + scale_weights[1] * weight_l2_up
            + scale_weights[2] * weight_l3_up
        )
        if return_pyramid:
            if return_aux:
                return out, spatial_pyramid, aux
            return out, spatial_pyramid
        return out, aux


class TextConditionedSpatialAdaptiveFusion(TGCSF):
    """Backward-compatible name for TGCSF."""
    pass
