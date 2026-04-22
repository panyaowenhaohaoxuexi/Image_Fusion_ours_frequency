# -*- coding: utf-8 -*-
"""基础分支融合模块。"""
import torch
import torch.nn as nn
from net.encoder.blocks import ConvBNAct, ResidualBlock

class SimpleBaseFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, 1, 0), nn.Sigmoid())
        self.mixer = nn.Sequential(
            ConvBNAct(channels * 2, channels, 3, 1, 1),
            ResidualBlock(channels),
            ResidualBlock(channels)
        )
    def forward(self, vis_base: torch.Tensor, ir_base: torch.Tensor):
        concat_feat = torch.cat([vis_base, ir_base], dim=1)
        gate = self.gate(concat_feat)
        gated_base = gate * vis_base + (1.0 - gate) * ir_base
        return self.mixer(concat_feat) + gated_base
