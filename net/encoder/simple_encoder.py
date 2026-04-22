# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from net.restormer_light import OverlapPatchEmbed, TransformerBlock


class SimpleSharedEncoder(nn.Module):
    """
    轻量单尺度 Restormer 编码器
    对外接口保持不变：
        return base_feat, freq_feat, shared_feat
    """
    def __init__(
        self,
        inp_channels=1,
        feature_dim=64,          # 对外仍然输出 64，兼容后续模块
        inner_dim=24,            # 内部轻量维度
        num_blocks=1,
        num_heads=1,
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias'
    ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, inner_dim, bias=bias)

        self.encoder_body = nn.Sequential(*[
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks)
        ])

        # 投影回当前工程统一使用的 64 通道
        self.shared_proj = nn.Conv2d(inner_dim, feature_dim, kernel_size=1, bias=bias)

        self.low_pass = nn.AvgPool2d(3, 1, 1)

        self.base_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias)
        )

        self.freq_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        feat = self.patch_embed(x)
        feat = self.encoder_body(feat)
        shared_feat = self.shared_proj(feat)

        low_feat = self.low_pass(shared_feat)
        high_feat = shared_feat - low_feat

        base_feat = self.base_head(low_feat) + low_feat
        freq_feat = self.freq_head(high_feat) + shared_feat

        return base_feat, freq_feat, shared_feat