# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialResidualCoupling(nn.Module):
    """Frequency-Spatial Residual Coupling.

    Spatial feature is the base representation, and frequency feature provides
    gated residual compensation from delta = |F_spa - F_freq|.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, F_freq: torch.Tensor, F_spa: torch.Tensor):
        if F_freq.shape[-2:] != F_spa.shape[-2:]:
            F_freq = F.interpolate(F_freq, size=F_spa.shape[-2:], mode="bilinear", align_corners=False)
        delta = torch.abs(F_spa - F_freq)
        gate = self.gate(delta)
        residual = self.residual(F_freq - F_spa)
        fused = F_spa + gate * residual
        return fused, gate


FSRC = FrequencySpatialResidualCoupling


class DDA(FrequencySpatialResidualCoupling):
    """Backward-compatible alias. New code should use FSRC."""
    pass
