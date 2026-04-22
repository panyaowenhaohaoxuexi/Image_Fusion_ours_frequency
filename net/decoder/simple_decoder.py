# -*- coding: utf-8 -*-
"""干净版本的解码器。"""
import torch
import torch.nn as nn
from .blocks import DecodeBlock

class SimpleDecoder(nn.Module):
    def __init__(self, channels=64, out_channels=1):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, 1, 1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(DecodeBlock(channels), DecodeBlock(channels), DecodeBlock(channels))
        self.out = nn.Sequential(nn.Conv2d(channels, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, out_channels, 3, 1, 1))
    def forward(self, inp_img: torch.Tensor, base_feature: torch.Tensor, freq_feature: torch.Tensor, fuse: str = None):
        x = torch.cat([base_feature, freq_feature], dim=1)
        x = self.fuse(x)
        x = self.body(x)
        out = torch.sigmoid(self.out(x))
        return out, x
