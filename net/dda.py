# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDA(nn.Module):
    """Dual-Domain Aggregation with asymmetric gated aggregation.

    G is the spatial-branch aggregation map:
    D = G * F_spa + (1 - G) * F_freq.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, F_freq: torch.Tensor, F_spa: torch.Tensor):
        if F_freq.shape[-2:] != F_spa.shape[-2:]:
            F_freq = F.interpolate(F_freq, size=F_spa.shape[-2:], mode="bilinear", align_corners=False)
        context = torch.cat([F_spa, F_freq, torch.abs(F_spa - F_freq)], dim=1)
        gate = self.gate(context)
        fused = gate * F_spa + (1.0 - gate) * F_freq
        return fused, gate
