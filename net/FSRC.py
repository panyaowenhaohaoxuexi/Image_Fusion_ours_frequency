# -*- coding: utf-8 -*-
"""Frequency-spatial residual coupling with channel-spatial gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.normalization import get_valid_group_count


def _small_output_init(layer: nn.Module) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class FrequencySpatialResidualCoupling(nn.Module):
    """Use frequency-domain residual compensation on a spatial feature base."""

    def __init__(self, channels: int = 64):
        super().__init__()
        hidden = max(channels // 4, 8)
        groups = get_valid_group_count(channels)
        self.spa_norm = nn.GroupNorm(groups, channels)
        self.freq_norm = nn.GroupNorm(groups, channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.difference_residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.joint_context_residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.residual_scale_param = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        for layer in (
            self.channel_mlp[-1], self.spatial_gate[-1],
            self.difference_residual[-1], self.joint_context_residual[-1],
        ):
            _small_output_init(layer)

    def forward(self, F_freq: torch.Tensor, F_spa: torch.Tensor, return_channel_gate: bool = False):
        if F_freq.shape[-2:] != F_spa.shape[-2:]:
            F_freq = F.interpolate(F_freq, size=F_spa.shape[-2:], mode='bilinear', align_corners=False)
        spa_norm = self.spa_norm(F_spa)
        freq_norm = self.freq_norm(F_freq)
        difference = torch.abs(spa_norm - freq_norm)
        product = spa_norm * freq_norm
        gate_feature = torch.cat([spa_norm, freq_norm, difference, product], dim=1)
        channel_logits = (
            self.channel_mlp(F.adaptive_avg_pool2d(gate_feature, 1))
            + self.channel_mlp(F.adaptive_max_pool2d(gate_feature, 1))
        )
        spatial_logits = self.spatial_gate(gate_feature)
        gate_channel = torch.sigmoid(channel_logits + spatial_logits)
        difference_residual = self.difference_residual(freq_norm - spa_norm)
        joint_context_residual = self.joint_context_residual(torch.cat([spa_norm, freq_norm], dim=1))
        frequency_compensation = difference_residual + 0.5 * joint_context_residual
        residual_scale = torch.tanh(self.residual_scale_param)
        fused = F_spa + residual_scale * gate_channel * frequency_compensation
        gate_map = gate_channel.mean(dim=1, keepdim=True)
        if not return_channel_gate:
            return fused, gate_map
        return fused, {
            'gate': gate_map,
            'gate_channel': gate_channel,
            'residual_scale': residual_scale,
            'frequency_compensation': frequency_compensation,
        }


FSRC = FrequencySpatialResidualCoupling
