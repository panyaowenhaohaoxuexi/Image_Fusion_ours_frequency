# -*- coding: utf-8 -*-
"""干净版本的解码器。"""
import torch
import torch.nn as nn
from .blocks import DecodeBlock


class SimpleDecoder(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            DecodeBlock(channels),
            DecodeBlock(channels),
            DecodeBlock(channels),
        )
        self.skip_proj = nn.Conv2d(channels, 32, 1, 1, 0)
        self.img_proj = nn.Conv2d(out_channels, 32, 3, 1, 1)
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(32, out_channels, 3, 1, 1)

    def forward(self, inp_img: torch.Tensor, base_feature: torch.Tensor, freq_feature: torch.Tensor, fuse: str = None):
        x = torch.cat([base_feature, freq_feature], dim=1)
        x = self.fuse(x)
        body_feat = self.body(x)
        recon_feat = self.head(body_feat)
        recon_feat = recon_feat + self.skip_proj(base_feature) + self.img_proj(inp_img)
        out = torch.sigmoid(self.out(recon_feat))
        return out, body_feat
