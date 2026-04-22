# -*- coding: utf-8 -*-
import torch.nn as nn

class ConvBNAct(nn.Module):
    """基础卷积块：卷积 + BN + 激活。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.1, inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """最简单的残差块。"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, 1, 1, 'relu')
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)
