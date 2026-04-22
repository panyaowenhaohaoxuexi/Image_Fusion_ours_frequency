# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from net.restormer_light import OverlapPatchEmbed, TransformerBlock, Downsample, Upsample
from net.encoder.blocks import ConvBNAct, ResidualBlock, ShallowRefine


class _ConvStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels)
        )

    def forward(self, x):
        return self.body(x)


class SimpleSharedEncoder(nn.Module):
    """
    三尺度共享编码器（混合式）
    - Level1/2：CNN stage，优先保证效率与细节保真
    - Level3：Transformer stage，补充高层全局建模
    - Top-down aggregation：借鉴 Text-IF 的层级式编解码写法，但保持当前工程接口不变
    """
    def __init__(
        self,
        inp_channels=1,
        feature_dim=64,
        inner_dim=24,
        num_blocks=(1, 1, 2),
        num_heads=(1, 2, 4),
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias'
    ):
        super().__init__()

        if isinstance(num_blocks, int):
            num_blocks = (num_blocks, num_blocks, max(2, num_blocks))
        if isinstance(num_heads, int):
            num_heads = (num_heads, max(2, num_heads), max(4, num_heads))

        dim1 = inner_dim
        dim2 = inner_dim * 2
        dim3 = inner_dim * 4

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim1, bias=bias)

        self.encoder_level1 = _ConvStage(dim1)
        self.down1_2 = Downsample(dim1, bias=bias)

        self.encoder_level2 = _ConvStage(dim2)
        self.down2_3 = Downsample(dim2, bias=bias)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim3, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim3, bias=bias)
        self.fuse_level2 = nn.Sequential(
            nn.Conv2d(dim2 * 2, dim2, kernel_size=1, bias=bias),
            _ConvStage(dim2)
        )

        self.up2_1 = Upsample(dim2, bias=bias)
        self.fuse_level1 = nn.Sequential(
            nn.Conv2d(dim1 * 2, dim1, kernel_size=1, bias=bias),
            _ConvStage(dim1)
        )

        self.shared_proj = nn.Conv2d(dim1, feature_dim, kernel_size=1, bias=bias)
        self.shared_refine = ShallowRefine(feature_dim)

        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.base_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        self.freq_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        l1 = self.encoder_level1(self.patch_embed(x))
        l2 = self.encoder_level2(self.down1_2(l1))
        l3 = self.encoder_level3(self.down2_3(l2))

        l2_td = self.fuse_level2(torch.cat([l2, self.up3_2(l3)], dim=1))
        l1_td = self.fuse_level1(torch.cat([l1, self.up2_1(l2_td)], dim=1))

        shared_feat = self.shared_refine(self.shared_proj(l1_td))
        low_feat = self.low_pass(shared_feat)
        high_feat = shared_feat - low_feat

        base_feat = self.base_head(low_feat) + low_feat
        freq_feat = self.freq_head(high_feat) + high_feat
        return base_feat, freq_feat, shared_feat
