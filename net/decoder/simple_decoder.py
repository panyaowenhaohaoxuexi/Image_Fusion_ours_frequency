# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.restormer_light import TransformerBlock
from net.encoder.blocks import ConvBNAct, ResidualBlock


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class DecoderTextFeatureWiseAffine(nn.Module):
    """Text-IF style FeatureWiseAffine used inside the final decoder."""

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


class TextGuidedDecoderStage(nn.Module):
    """Decoder stage with Text-IF affine + Restormer refinement."""

    def __init__(self, channels: int, intent_dim: int, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, bias: bool = False,
                 layer_norm_type: str = 'WithBias'):
        super().__init__()
        self.text_affine = DecoderTextFeatureWiseAffine(intent_dim, channels)
        self.body = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, bias, layer_norm_type),
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor, text_intent: torch.Tensor = None) -> torch.Tensor:
        x = self.text_affine(x, text_intent)
        return self.body(x) + x


class SimpleDecoder(nn.Module):
    """
    三层 Text-IF-style 融合解码器。

    接口保持：
        forward(inp_img, base_feature, freq_feature, fuse=None, text_intent=None)

    输入 F_spa / F_freq 后构建 L1/L2/L3 内部金字塔；L3、L2、L1 每一级
    都使用 Text-IF FeatureWiseAffine，增强 Text Intent 对最终重建的影响。
    """

    def __init__(self, channels=64, out_channels=1, inner_dim=24, num_blocks=1,
                 num_heads=1, ffn_expansion_factor=2.0, bias=False,
                 LayerNorm_type='WithBias', intent_dim=64, use_text_guidance=True):
        super().__init__()
        self.use_text_guidance = use_text_guidance
        self.reduce_l1 = nn.Conv2d(channels * 2, inner_dim, kernel_size=1, bias=bias)
        self.down_l2 = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, 3, 2, 1, bias=bias), nn.GELU())
        self.down_l3 = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, 3, 2, 1, bias=bias), nn.GELU())

        self.stage_l3 = nn.ModuleList([
            TextGuidedDecoderStage(inner_dim, intent_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l2 = nn.Conv2d(inner_dim * 2, inner_dim, 1, 1, 0, bias=bias)
        self.stage_l2 = nn.ModuleList([
            TextGuidedDecoderStage(inner_dim, intent_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l1 = nn.Conv2d(inner_dim * 2, inner_dim, 1, 1, 0, bias=bias)
        self.stage_l1 = nn.ModuleList([
            TextGuidedDecoderStage(inner_dim, intent_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inner_dim, out_channels, 3, 1, 1, bias=bias),
        )

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: torch.Tensor, text_intent: torch.Tensor):
        for block in blocks:
            x = block(x, text_intent)
        return x

    def forward(self, inp_img: torch.Tensor, base_feature: torch.Tensor, freq_feature: torch.Tensor,
                fuse: str = None, text_intent: torch.Tensor = None):
        text = text_intent if self.use_text_guidance else None
        x_l1 = self.reduce_l1(torch.cat([base_feature, freq_feature], dim=1))
        x_l2 = self.down_l2(x_l1)
        x_l3 = self.down_l3(x_l2)

        d_l3 = self._run_stage(self.stage_l3, x_l3, text)
        d_l2 = self.fuse_l2(torch.cat([
            F.interpolate(d_l3, size=x_l2.shape[-2:], mode='bilinear', align_corners=False),
            x_l2,
        ], dim=1))
        d_l2 = self._run_stage(self.stage_l2, d_l2, text)
        d_l1 = self.fuse_l1(torch.cat([
            F.interpolate(d_l2, size=x_l1.shape[-2:], mode='bilinear', align_corners=False),
            x_l1,
        ], dim=1))
        d_l1 = self._run_stage(self.stage_l1, d_l1, text)

        out = self.head(d_l1)
        out = out + inp_img
        return torch.sigmoid(out), d_l1
