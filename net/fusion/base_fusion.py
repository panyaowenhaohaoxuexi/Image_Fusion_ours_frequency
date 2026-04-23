# -*- coding: utf-8 -*-
"""基础分支融合模块。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # 借鉴 edge-aware 思路：补一条轻量高频残差支路，增强轮廓与边缘保留。
        self.edge_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, 1, 0), nn.Sigmoid())
        self.edge_refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels)
        )

    def forward(self, vis_base: torch.Tensor, ir_base: torch.Tensor):
        concat_feat = torch.cat([vis_base, ir_base], dim=1)
        gate = self.gate(concat_feat)
        gated_base = gate * vis_base + (1.0 - gate) * ir_base

        vis_edge = vis_base - F.avg_pool2d(vis_base, 3, 1, 1)
        ir_edge = ir_base - F.avg_pool2d(ir_base, 3, 1, 1)
        edge_concat = torch.cat([vis_edge, ir_edge], dim=1)
        edge_gate = self.edge_gate(edge_concat)
        fused_edge = edge_gate * vis_edge + (1.0 - edge_gate) * ir_edge
        fused_edge = self.edge_refine(fused_edge)

        return self.mixer(concat_feat) + gated_base + 0.5 * fused_edge
