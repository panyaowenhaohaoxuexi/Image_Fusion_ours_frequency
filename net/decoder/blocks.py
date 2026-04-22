# -*- coding: utf-8 -*-
import torch.nn as nn
from net.encoder.blocks import ConvBNAct, ResidualBlock

class DecodeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(channels, channels, 3, 1, 1), ResidualBlock(channels))
    def forward(self, x):
        return self.block(x)
