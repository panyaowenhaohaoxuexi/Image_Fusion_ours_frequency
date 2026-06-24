# -*- coding: utf-8 -*-
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPyramidAdapter(nn.Module):
    """Map single-level frequency fusion output to a three-level feature pyramid."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.refine_l1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.down_l2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.down_l3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, fused_freq: torch.Tensor,
                target_pyramid: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = ("l1", "l2", "l3")
        missing = [level for level in required if level not in target_pyramid]
        if missing:
            raise KeyError(f"target_pyramid missing levels: {missing}")

        freq_l1 = self.refine_l1(fused_freq) + fused_freq
        freq_l2 = self.down_l2(freq_l1)
        freq_l3 = self.down_l3(freq_l2)

        return {
            "l1": self._match_size(freq_l1, target_pyramid["l1"]),
            "l2": self._match_size(freq_l2, target_pyramid["l2"]),
            "l3": self._match_size(freq_l3, target_pyramid["l3"]),
        }
