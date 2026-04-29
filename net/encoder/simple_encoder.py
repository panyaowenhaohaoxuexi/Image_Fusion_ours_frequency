# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from net.restormer_light import OverlapPatchEmbed, TransformerBlock
from net.encoder.blocks import ConvBNAct, ResidualBlock


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class SimpleSharedEncoder(nn.Module):
    """
    三层轻量 Restormer 编码器。

    forward 返回：
        spatial_feats, freq_feat, shared_feat
    其中 spatial_feats = [L1, L2, L3]，分辨率分别为 H/W、H/2/W/2、H/4/W/4。
    频率分支仍使用最高分辨率 freq_feat 做 FFT/token routing。
    """

    def __init__(self, inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1,
                 num_heads=1, ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, inner_dim, bias=bias)
        self.encoder_l1 = nn.Sequential(*[
            TransformerBlock(inner_dim, _valid_heads(inner_dim, num_heads), ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.proj_l1 = nn.Conv2d(inner_dim, feature_dim, kernel_size=1, bias=bias)

        heads_l = _valid_heads(feature_dim, num_heads)
        self.down_l2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.GELU(),
        )
        self.encoder_l2 = nn.Sequential(*[
            TransformerBlock(feature_dim, heads_l, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.down_l3 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.GELU(),
        )
        self.encoder_l3 = nn.Sequential(*[
            TransformerBlock(feature_dim, heads_l, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])

        self.low_pass = nn.AvgPool2d(3, 1, 1)
        self.base_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
        )
        self.freq_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
        )
        self.spa_l2_refine = nn.Sequential(ConvBNAct(feature_dim, feature_dim, 3, 1, 1, 'gelu'), ResidualBlock(feature_dim))
        self.spa_l3_refine = nn.Sequential(ConvBNAct(feature_dim, feature_dim, 3, 1, 1, 'gelu'), ResidualBlock(feature_dim))

    def forward(self, x: torch.Tensor):
        l1_inner = self.encoder_l1(self.patch_embed(x))
        shared_feat = self.proj_l1(l1_inner)

        low_feat = self.low_pass(shared_feat)
        high_feat = shared_feat - low_feat
        spa_l1 = self.base_head(low_feat) + low_feat
        freq_feat = self.freq_head(high_feat) + shared_feat

        l2 = self.encoder_l2(self.down_l2(shared_feat))
        spa_l2 = self.spa_l2_refine(l2) + l2
        l3 = self.encoder_l3(self.down_l3(spa_l2))
        spa_l3 = self.spa_l3_refine(l3) + l3

        return [spa_l1, spa_l2, spa_l3], freq_feat, shared_feat
