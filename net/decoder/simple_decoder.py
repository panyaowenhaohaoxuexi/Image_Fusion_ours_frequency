# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from net.restormer_light import TransformerBlock


class SimpleDecoder(nn.Module):
    """
    轻量单尺度 Restormer 解码器
    对外接口保持不变：
        forward(inp_img, base_feature, freq_feature, fuse=None)
    """
    def __init__(
        self,
        channels=64,
        out_channels=1,
        inner_dim=24,
        num_blocks=1,
        num_heads=1,
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias'
    ):
        super().__init__()

        self.reduce = nn.Conv2d(channels * 2, inner_dim, kernel_size=1, bias=bias)

        self.decoder_body = nn.Sequential(*[
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inner_dim, out_channels, 3, 1, 1, bias=bias)
        )

    def forward(self, inp_img: torch.Tensor, base_feature: torch.Tensor, freq_feature: torch.Tensor, fuse: str = None):
        x = torch.cat([base_feature, freq_feature], dim=1)
        x = self.reduce(x)
        body_feat = self.decoder_body(x)

        out = self.head(body_feat)
        out = out + inp_img

        return torch.sigmoid(out), body_feat