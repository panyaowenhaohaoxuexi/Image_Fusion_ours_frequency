# -*- coding: utf-8 -*-
"""干净版本的共享编码器。"""
import torch
import torch.nn as nn
from .blocks import ConvBNAct, ResidualBlock

class SimpleSharedEncoder(nn.Module):
    """输出 base_feat、freq_feat、shared_feat 三路特征。"""
    def __init__(self, inp_channels=1, feature_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(inp_channels, 32, 3, 1, 1),
            ResidualBlock(32),
            ConvBNAct(32, feature_dim, 3, 1, 1),
            ResidualBlock(feature_dim),
            ResidualBlock(feature_dim)
        )
        self.low_pass = nn.AvgPool2d(3, 1, 1)
        self.base_head = nn.Sequential(ConvBNAct(feature_dim, feature_dim, 3, 1, 1), ResidualBlock(feature_dim))
        self.freq_head = nn.Sequential(ConvBNAct(feature_dim, feature_dim, 3, 1, 1), ResidualBlock(feature_dim))
    def forward(self, x: torch.Tensor):
        shared_feat = self.stem(x)
        low_feat = self.low_pass(shared_feat)
        base_feat = self.base_head(low_feat)
        high_residual = shared_feat - low_feat
        freq_feat = self.freq_head(high_residual) + shared_feat
        return base_feat, freq_feat, shared_feat
