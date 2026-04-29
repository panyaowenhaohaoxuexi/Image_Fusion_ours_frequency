# -*- coding: utf-8 -*-
"""Three-level Text-conditioned spatial adaptive fusion branch.

L1/L2 use SPGFusion-style CSAF; L3 uses SPGFusion-style PSAF. Text-IF-style
FeatureWiseAffine is inserted in every fusion/top-down stage. The condition is
this project's Text Intent Embedding, not CLIP-image/DINO visual semantics.
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import math
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


class TextIFFeatureWiseAffine(nn.Module):
    """Text-IF style FeatureWiseAffine: x = (1 + gamma) * x + beta."""

    def __init__(self, in_channels: int, out_channels: int, use_affine_level: bool = True):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels * 2, out_channels * (1 + int(self.use_affine_level))),
        )

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        if not self.use_affine_level or text_embed is None:
            return x
        batch = x.shape[0]
        gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


class SPGFusionEmbed(nn.Module):
    """SPGFusion-style Fusion_Embed: concat two modality features and 1x1 project."""

    def __init__(self, embed_dim: int, bias: bool = False):
        super().__init__()
        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor) -> torch.Tensor:
        return self.fusion_proj(torch.cat([x_A, x_B], dim=1))


def _safe_group_count(channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SPGCrossAttention(nn.Module):
    """SPGFusion/Text-IF style spatial cross-attention with safe pooling."""

    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 16,
                 max_attn_size: int = 32, residual_scale: float = 1.0):
        super().__init__()
        n_head = _valid_heads(in_channel, n_head)
        self.n_head = n_head
        self.max_attn_size = max_attn_size
        self.residual_scale = residual_scale
        groups = _safe_group_count(in_channel, norm_groups)
        self.norm_A = nn.GroupNorm(groups, in_channel)
        self.norm_B = nn.GroupNorm(groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def _full_attention(self, x_A: torch.Tensor, x_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channel, height, width = x_A.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm_A = self.norm_A(x_A)
        query_A, key_A, value_A = self.qkv_A(norm_A).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)
        norm_B = self.norm_B(x_B)
        query_B, key_B, value_B = self.qkv_B(norm_B).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        attn_A = torch.einsum('bnchw,bncyx->bnhwyx', query_B, key_A).contiguous() / math.sqrt(channel)
        attn_A = torch.softmax(attn_A.view(batch, n_head, height, width, -1), dim=-1).view(batch, n_head, height, width, height, width)
        out_A = torch.einsum('bnhwyx,bncyx->bnchw', attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width)) + norm_A

        attn_B = torch.einsum('bnchw,bncyx->bnhwyx', query_A, key_B).contiguous() / math.sqrt(channel)
        attn_B = torch.softmax(attn_B.view(batch, n_head, height, width, -1), dim=-1).view(batch, n_head, height, width, height, width)
        out_B = torch.einsum('bnhwyx,bncyx->bnchw', attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width)) + norm_B
        return out_A, out_B

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x_A.shape
        if max(h, w) <= self.max_attn_size:
            return self._full_attention(x_A, x_B)
        target_h = min(h, self.max_attn_size)
        target_w = min(w, self.max_attn_size)
        x_A_small = F.adaptive_avg_pool2d(x_A, (target_h, target_w))
        x_B_small = F.adaptive_avg_pool2d(x_B, (target_h, target_w))
        out_A_small, out_B_small = self._full_attention(x_A_small, x_B_small)
        delta_A = F.interpolate(out_A_small - self.norm_A(x_A_small), size=(h, w), mode='bilinear', align_corners=False)
        delta_B = F.interpolate(out_B_small - self.norm_B(x_B_small), size=(h, w), mode='bilinear', align_corners=False)
        return x_A + self.residual_scale * delta_A, x_B + self.residual_scale * delta_B


class SPGAttentionSpatial(nn.Module):
    """SPGFusion/Text-IF style spatial self-attention with safe pooling."""

    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 16,
                 max_attn_size: int = 32, residual_scale: float = 1.0):
        super().__init__()
        n_head = _valid_heads(in_channel, n_head)
        self.n_head = n_head
        self.max_attn_size = max_attn_size
        self.residual_scale = residual_scale
        groups = _safe_group_count(in_channel, norm_groups)
        self.norm = nn.GroupNorm(groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def _full_attention(self, input_feat: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = input_feat.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input_feat)
        query, key, value = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)
        attn = torch.einsum('bnchw,bncyx->bnhwyx', query, key).contiguous() / math.sqrt(channel)
        attn = torch.softmax(attn.view(batch, n_head, height, width, -1), dim=-1).view(batch, n_head, height, width, height, width)
        out = torch.einsum('bnhwyx,bncyx->bnchw', attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input_feat

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        _, _, h, w = input_feat.shape
        if max(h, w) <= self.max_attn_size:
            return self._full_attention(input_feat)
        target_h = min(h, self.max_attn_size)
        target_w = min(w, self.max_attn_size)
        small = F.adaptive_avg_pool2d(input_feat, (target_h, target_w))
        out_small = self._full_attention(small)
        delta = F.interpolate(out_small - small, size=(h, w), mode='bilinear', align_corners=False)
        return input_feat + self.residual_scale * delta


class SPGTextCSAF(nn.Module):
    """SPGFusion CSAF adapted to Text Intent."""

    def __init__(self, norm_nc: int, intent_dim: int = 64, nhidden: int = 64,
                 use_freq_context: bool = True):
        super().__init__()
        self.use_freq_context = use_freq_context
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        self.intent_to_map = nn.Linear(intent_dim, intent_dim)
        self.freq_to_map = nn.Conv2d(norm_nc, intent_dim, 1, 1, 0) if use_freq_context else None
        self.mlp_shared = nn.Sequential(nn.Conv2d(intent_dim, nhidden, 3, 1, 1), nn.ReLU(inplace=True))
        self.mlp_gamma = nn.Sequential(nn.Conv2d(nhidden, norm_nc, 3, 1, 1), nn.Sigmoid())
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def _intent_map(self, x: torch.Tensor, text_intent: torch.Tensor,
                    freq_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = x.shape
        segmap = self.intent_to_map(text_intent).view(b, -1, 1, 1).expand(-1, -1, h, w)
        if self.freq_to_map is not None and freq_feat is not None:
            freq_map = self.freq_to_map(F.interpolate(freq_feat, size=(h, w), mode='bilinear', align_corners=False))
            segmap = segmap + freq_map
        return segmap

    def forward(self, x: torch.Tensor, text_intent: torch.Tensor,
                freq_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(self._intent_map(x, text_intent, freq_feat))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return self.bn(normalized * (1.0 + gamma)) + beta


class SPGTextCSAFPair(nn.Module):
    """L1/L2: CSAF(vis) + CSAF(ir) + Fusion_Embed + optional spatial attention."""

    def __init__(self, channels: int, intent_dim: int = 64, n_head: int = 1,
                 norm_groups: int = 16, max_attn_size: int = 32,
                 use_freq_context: bool = True, use_spatial_attention: bool = True):
        super().__init__()
        self.csaf_vis = SPGTextCSAF(channels, intent_dim, max(64, channels), use_freq_context)
        self.csaf_ir = SPGTextCSAF(channels, intent_dim, max(64, channels), use_freq_context)
        self.feature_fusion = SPGFusionEmbed(channels)
        self.attention_spatial = SPGAttentionSpatial(channels, n_head, norm_groups, max_attn_size) if use_spatial_attention else nn.Identity()
        self.text_affine = TextIFFeatureWiseAffine(intent_dim, channels)
        self.refine = nn.Sequential(ConvBNAct(channels, channels, 3, 1, 1, 'gelu'), ResidualBlock(channels))

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, text_intent: torch.Tensor,
                freq_feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vis_csaf = self.csaf_vis(vis_feat, text_intent, freq_feat)
        ir_csaf = self.csaf_ir(ir_feat, text_intent, freq_feat)
        fused = self.feature_fusion(vis_csaf, ir_csaf)
        fused = self.attention_spatial(fused)
        fused = self.text_affine(fused, text_intent)
        fused = self.refine(fused) + fused
        return fused, {'vis_csaf': vis_csaf, 'ir_csaf': ir_csaf, 'fused': fused}


class SPGTextPSAF(nn.Module):
    """L3: PSAF-style block: cross-attn + CSAF + Fusion_Embed + spatial-attn."""

    def __init__(self, channels: int, intent_dim: int = 64, n_head: int = 1,
                 norm_groups: int = 16, max_attn_size: int = 32,
                 use_freq_context: bool = True):
        super().__init__()
        self.cross_attention = SPGCrossAttention(channels, n_head, norm_groups, max_attn_size)
        self.csaf_vis = SPGTextCSAF(channels, intent_dim, max(64, channels), use_freq_context)
        self.csaf_ir = SPGTextCSAF(channels, intent_dim, max(64, channels), use_freq_context)
        self.feature_fusion = SPGFusionEmbed(channels)
        self.attention_spatial = SPGAttentionSpatial(channels, n_head, norm_groups, max_attn_size)
        self.text_affine = TextIFFeatureWiseAffine(intent_dim, channels)
        self.refine = nn.Sequential(ConvBNAct(channels, channels, 3, 1, 1, 'gelu'), ResidualBlock(channels))

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, text_intent: torch.Tensor,
                freq_feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vis_cross, ir_cross = self.cross_attention(vis_feat, ir_feat)
        vis_csaf = self.csaf_vis(vis_cross, text_intent, freq_feat)
        ir_csaf = self.csaf_ir(ir_cross, text_intent, freq_feat)
        fused = self.feature_fusion(vis_csaf, ir_csaf)
        fused = self.attention_spatial(fused)
        fused = self.text_affine(fused, text_intent)
        fused = self.refine(fused) + fused
        return fused, {'vis_cross': vis_cross, 'ir_cross': ir_cross, 'vis_csaf': vis_csaf, 'ir_csaf': ir_csaf, 'fused': fused}


class TextGuidedTopDownBlock(nn.Module):
    """Top-down aggregation block with Text-IF affine modulation."""

    def __init__(self, channels: int, intent_dim: int, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=False)
        self.text_affine = TextIFFeatureWiseAffine(intent_dim, channels)
        self.refine = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, False, 'WithBias'),
            ConvBNAct(channels, channels, 3, 1, 1, 'gelu'),
            ResidualBlock(channels),
        )

    def forward(self, high_feat: torch.Tensor, skip_feat: torch.Tensor, text_intent: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high_feat, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([high_up, skip_feat], dim=1))
        x = self.text_affine(x, text_intent)
        return self.refine(x) + x


class TextConditionedSpatialAdaptiveFusion(nn.Module):
    """Three-level SPGFusion/Text-IF style spatial branch.

    vis_spa / ir_spa should be [L1, L2, L3]. Passing a single tensor remains
    supported by internally creating pooled L2/L3 features.
    """

    def __init__(self, channels: int = 64, intent_dim: int = 64, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, init_res_scale: float = 0.20,
                 use_freq_context: bool = True, max_attn_size: int = 32,
                 norm_groups: int = 16):
        super().__init__()
        self.use_freq_context = use_freq_context
        self.level1_csaf = SPGTextCSAFPair(channels, intent_dim, num_heads, norm_groups, max_attn_size, use_freq_context, use_spatial_attention=False)
        self.level2_csaf = SPGTextCSAFPair(channels, intent_dim, num_heads, norm_groups, max_attn_size, use_freq_context, use_spatial_attention=True)
        self.level3_psaf = SPGTextPSAF(channels, intent_dim, num_heads, norm_groups, max_attn_size, use_freq_context)
        self.topdown_l2 = TextGuidedTopDownBlock(channels, intent_dim, num_heads, ffn_expansion_factor)
        self.topdown_l1 = TextGuidedTopDownBlock(channels, intent_dim, num_heads, ffn_expansion_factor)
        self.final_text_affine = TextIFFeatureWiseAffine(intent_dim, channels)

        gate_in = channels * 4 + (channels if use_freq_context else 0)
        self.intent_gate = nn.Sequential(nn.Linear(intent_dim, channels), nn.LeakyReLU(inplace=True), nn.Linear(channels, channels))
        self.feature_gate = nn.Sequential(
            nn.Conv2d(gate_in, channels, 1, 1, 0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        mixer_in = channels * 3 + (channels if use_freq_context else 0)
        self.mixer = nn.Sequential(
            ConvBNAct(mixer_in, channels, 3, 1, 1, 'gelu'),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        self.refine = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, False, 'WithBias'),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))
        self.out_norm = nn.BatchNorm2d(channels)

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

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, vis_spa: TensorOrPyramid, ir_spa: TensorOrPyramid, text_intent: torch.Tensor,
                freq_feat: Optional[torch.Tensor] = None, return_aux: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        vis_l1, vis_l2, vis_l3 = self._as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = self._as_three_levels(ir_spa)
        l1_freq = freq_feat
        l2_freq = F.interpolate(freq_feat, size=vis_l2.shape[-2:], mode='bilinear', align_corners=False) if freq_feat is not None else None
        l3_freq = F.interpolate(freq_feat, size=vis_l3.shape[-2:], mode='bilinear', align_corners=False) if freq_feat is not None else None

        fused_l1, aux_l1 = self.level1_csaf(vis_l1, ir_l1, text_intent, l1_freq)
        fused_l2, aux_l2 = self.level2_csaf(vis_l2, ir_l2, text_intent, l2_freq)
        fused_l3, aux_l3 = self.level3_psaf(vis_l3, ir_l3, text_intent, l3_freq)

        td_l2 = self.topdown_l2(fused_l3, fused_l2, text_intent)
        td_l1 = self.topdown_l1(td_l2, fused_l1, text_intent)
        text_spa = self.final_text_affine(td_l1, text_intent)

        diff = torch.abs(vis_l1 - ir_l1)
        common = 0.5 * (vis_l1 + ir_l1)
        high_common = self._high_pass(common)
        gate_context = [vis_l1, ir_l1, diff, text_spa]
        mix_context = [text_spa, common, high_common]
        if self.use_freq_context:
            if freq_feat is None:
                freq_context = torch.zeros_like(vis_l1)
            else:
                freq_context = freq_feat
                if freq_context.shape[-2:] != vis_l1.shape[-2:]:
                    freq_context = F.interpolate(freq_context, size=vis_l1.shape[-2:], mode='bilinear', align_corners=False)
            gate_context.append(freq_context)
            mix_context.append(freq_context)

        feature_gate = self.feature_gate(torch.cat(gate_context, dim=1))
        intent_gate = torch.sigmoid(self.intent_gate(text_intent)).view(text_intent.shape[0], -1, 1, 1)
        gate = torch.clamp(0.65 * feature_gate + 0.35 * intent_gate, 0.0, 1.0)
        gated_modal = gate * vis_l1 + (1.0 - gate) * ir_l1
        mixed = self.mixer(torch.cat(mix_context, dim=1))
        mixed = mixed + gated_modal + text_spa
        refined = self.refine(mixed)
        out = self.out_norm(mixed + self.res_scale * refined)

        if not return_aux:
            return out
        aux = {
            'l1_fused': fused_l1,
            'l2_fused': fused_l2,
            'l3_fused': fused_l3,
            'td_l2': td_l2,
            'text_spa': text_spa,
            'spatial_gate': gate,
            'spatial_res_scale': self.res_scale.detach(),
            'l1_aux': aux_l1,
            'l2_aux': aux_l2,
            'l3_aux': aux_l3,
        }
        return out, aux
