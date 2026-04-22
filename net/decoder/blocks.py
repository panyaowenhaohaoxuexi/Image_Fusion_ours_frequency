# -*- coding: utf-8 -*-
"""
Decoder 相关模块
"""
import torch
import torch.nn as nn
from net.encoder.blocks import ConvBNAct, ResidualBlock, RestormerBlock


class ChannelAttention(nn.Module):
    """轻量通道注意力。"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.mlp(self.avg_pool(x))
        return x * w


class DecodeBlock(nn.Module):
    """
    解码块：
    Conv refine + Restormer + residual
    """
    def __init__(self, channels, num_heads=4, ffn_expansion_factor=2.0):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, 3, 1, 1, activation='gelu')
        self.trans = RestormerBlock(
            dim=channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=False
        )
        self.post = nn.Sequential(
            ResidualBlock(channels),
            ChannelAttention(channels)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.trans(x)
        x = self.post(x)
        return x