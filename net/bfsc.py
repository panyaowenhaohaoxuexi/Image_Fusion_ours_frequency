# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class BFSC(nn.Module):
    """Bidirectional frequency-spatial coupling.

    G is the spatial-branch confidence map:
    D = G * F_spa + (1 - G) * F_freq.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, freq_feat: torch.Tensor, spa_feat: torch.Tensor):
        if freq_feat.shape[-2:] != spa_feat.shape[-2:]:
            freq_feat = F.interpolate(freq_feat, size=spa_feat.shape[-2:], mode="bilinear", align_corners=False)
        context = torch.cat([spa_feat, freq_feat, torch.abs(spa_feat - freq_feat)], dim=1)
        gate = self.gate(context)
        fused = gate * spa_feat + (1.0 - gate) * freq_feat
        return fused, gate
