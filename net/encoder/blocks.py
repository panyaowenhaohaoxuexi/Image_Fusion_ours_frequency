# -*- coding: utf-8 -*-
"""
Encoder 相关基础模块：
1. 基础卷积块 / 残差块
2. 轻量 Restormer Block（MDTA + GDFN）
3. 一些用于 base/freq 分支拆分的小模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 基础 CNN 模块
# =========================
class ConvBNAct(nn.Module):
    """基础卷积块：卷积 + BN + 激活。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            act = nn.LeakyReLU(0.1, inplace=True)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """简单残差块。"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, 1, 1, 'relu')
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


# =========================
# LayerNorm for BCHW
# =========================
class LayerNorm2d(nn.Module):
    """
    对 BCHW 特征做 LayerNorm。
    在通道维上归一化，更接近 Restormer 的用法。
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# =========================
# GDFN
# =========================
class FeedForward(nn.Module):
    """
    Restormer 中的 GDFN 风格前馈网络
    """
    def __init__(self, dim, ffn_expansion_factor=2.0, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# =========================
# MDTA
# =========================
class Attention(nn.Module):
    """
    轻量 Restormer 风格注意力
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c % self.num_heads == 0, "channels 必须能被 num_heads 整除"

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads

        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, head, C_head, HW]
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out


class RestormerBlock(nn.Module):
    """
    轻量 Restormer Block
    """
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.0, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =========================
# 轻量分支模块
# =========================
class DepthwiseSeparableConv(nn.Module):
    """DWConv + PWConv，用于轻量增强。"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ShallowRefine(nn.Module):
    """
    用于 encoder 尾部的轻量细化模块
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            DepthwiseSeparableConv(channels),
            ResidualBlock(channels)
        )

    def forward(self, x):
        return self.block(x)