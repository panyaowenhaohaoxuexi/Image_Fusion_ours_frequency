# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ConvBNAct, ResidualBlock
from net.restormer_light import TransformerBlock


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class DecoderStage(nn.Module):
    """Decoder stage with Restormer refinement."""

    def __init__(self, channels: int, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, bias: bool = False,
                 layer_norm_type: str = 'WithBias'):
        super().__init__()
        self.body = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, bias, layer_norm_type),
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class SimpleDecoder(nn.Module):
    """Multi-scale decoder fed only by FSRC pyramid features."""

    def __init__(self, channels=64, out_channels=1, inner_dim=24, num_blocks=1,
                 num_heads=1, ffn_expansion_factor=2.0, bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        self.reduce_l1 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)
        self.reduce_l2 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)
        self.reduce_l3 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)

        self.stage_l3 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l2 = nn.Conv2d(inner_dim * 2, inner_dim, 1, 1, 0, bias=bias)
        self.stage_l2 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l1 = nn.Conv2d(inner_dim * 2, inner_dim, 1, 1, 0, bias=bias)
        self.stage_l1 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inner_dim, out_channels, 3, 1, 1, bias=bias),
        )

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: torch.Tensor):
        for block in blocks:
            x = block(x)
        return x

    def forward(self, D_L1: torch.Tensor, D_L2: torch.Tensor, D_L3: torch.Tensor):
        x_l1 = self.reduce_l1(D_L1)
        x_l2 = self.reduce_l2(D_L2)
        x_l3 = self.reduce_l3(D_L3)

        d_l3 = self._run_stage(self.stage_l3, x_l3)
        d_l2 = self.fuse_l2(torch.cat([
            F.interpolate(d_l3, size=x_l2.shape[-2:], mode='bilinear', align_corners=False),
            x_l2,
        ], dim=1))
        d_l2 = self._run_stage(self.stage_l2, d_l2)
        d_l1 = self.fuse_l1(torch.cat([
            F.interpolate(d_l2, size=x_l1.shape[-2:], mode='bilinear', align_corners=False),
            x_l1,
        ], dim=1))
        d_l1 = self._run_stage(self.stage_l1, d_l1)

        out = self.head(d_l1)
        return torch.sigmoid(out), d_l1
