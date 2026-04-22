# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from net.restormer_light import TransformerBlock, Downsample, Upsample
from net.encoder.blocks import ConvBNAct, ResidualBlock
from net.decoder.blocks import ChannelAttention


class _DecodeStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return self.body(x)


class SimpleDecoder(nn.Module):
    """
    三尺度轻量解码器（混合式）
    - 低层以 CNN 为主，保证恢复效率和纹理重建
    - 深层 latent 用 Transformer 补充全局结构建模
    """
    def __init__(
        self,
        channels=64,
        out_channels=1,
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

        self.input_proj = nn.Conv2d(channels * 2, dim1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.decoder_level1 = _DecodeStage(dim1)

        self.down1_2 = Downsample(dim1, bias=bias)
        self.decoder_level2 = _DecodeStage(dim2)

        self.down2_3 = Downsample(dim2, bias=bias)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=dim3, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim3, bias=bias)
        self.reduce_level2 = nn.Conv2d(dim2 * 2, dim2, kernel_size=1, bias=bias)
        self.refine_level2 = _DecodeStage(dim2)

        self.up2_1 = Upsample(dim2, bias=bias)
        self.reduce_level1 = nn.Conv2d(dim1 * 2, dim1, kernel_size=1, bias=bias)
        self.refine_level1 = _DecodeStage(dim1)

        self.refinement = nn.Sequential(
            ConvBNAct(dim1, dim1, 3, 1, 1, activation='gelu'),
            ResidualBlock(dim1)
        )
        self.head = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim1, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, inp_img: torch.Tensor, base_feature: torch.Tensor, freq_feature: torch.Tensor, fuse: str = None):
        x = torch.cat([base_feature, freq_feature], dim=1)
        l1 = self.decoder_level1(self.input_proj(x))
        l2 = self.decoder_level2(self.down1_2(l1))
        l3 = self.latent(self.down2_3(l2))

        l2 = self.refine_level2(self.reduce_level2(torch.cat([l2, self.up3_2(l3)], dim=1)))
        l1 = self.refine_level1(self.reduce_level1(torch.cat([l1, self.up2_1(l2)], dim=1)))
        body_feat = self.refinement(l1)

        out = self.head(body_feat)
        if inp_img is not None:
            out = out + inp_img
        return torch.sigmoid(out), body_feat
