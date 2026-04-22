# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import torch.nn.functional as F


def split_amplitude_phase(feature: torch.Tensor):
    """
    输入:  [B, C, H, W]
    输出:
        amplitude: [B, C, H, W//2+1]
        phase:     [B, C, H, W//2+1]
    """
    freq = torch.fft.rfft2(feature, dim=(-2, -1), norm='ortho')
    amplitude = torch.abs(freq)
    # 为了兼容一些旧环境，不直接强依赖 torch.angle 的行为差异
    phase = torch.atan2(freq.imag, freq.real + 1e-8)
    return amplitude, phase


def phase_wrap(phase: torch.Tensor) -> torch.Tensor:
    """
    把 phase 约束到 [-pi, pi]
    """
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def rebuild_from_amplitude_phase(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    spatial_size: Tuple[int, int]
):
    """
    由 amplitude + phase 重建空间域特征
    spatial_size: 原始空间尺寸 (H, W)
    """
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    complex_map = torch.complex(real, imag)
    return torch.fft.irfft2(
        complex_map,
        s=spatial_size,
        dim=(-2, -1),
        norm='ortho'
    )


def pad_to_multiple(x: torch.Tensor, patch_size: int):
    """
    把特征图 pad 到 patch_size 的整数倍
    """
    _, _, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, (pad_h, pad_w)


def patchify_feature_map(x: torch.Tensor, patch_size: int):
    """
    输入:
        x: [B, C, H, W]
    输出:
        tokens: [B, N, C*P*P]
        meta:   用于 unpatchify
    """
    x, pad_hw = pad_to_multiple(x, patch_size)
    b, c, h, w = x.shape

    gh = h // patch_size
    gw = w // patch_size

    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, gh, gw, P, P] -> [B, gh, gw, C, P, P]
    patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5).contiguous()
    tokens = patches.view(b, gh * gw, c * patch_size * patch_size)

    # 旧版 torch 不支持 indexing='ij'
    y = torch.linspace(0.0, 1.0, steps=gh, device=x.device)
    x_coord = torch.linspace(0.0, 1.0, steps=gw, device=x.device)
    grid_y, grid_x = torch.meshgrid(y, x_coord)

    coords = torch.stack([grid_y, grid_x], dim=-1)  # [gh, gw, 2]
    coords = coords.view(1, gh * gw, 2).repeat(b, 1, 1)

    meta = {
        'pad_hw': pad_hw,
        'grid_hw': (gh, gw),
        'full_hw': (h, w),
        'channels': c,
        'patch_size': patch_size,
        'coords': coords
    }
    return tokens, meta


def unpatchify_feature_map(tokens: torch.Tensor, meta: dict):
    """
    输入:
        tokens: [B, N, C*P*P]
    输出:
        x: [B, C, H, W]
    """
    b, _, _ = tokens.shape
    gh, gw = meta['grid_hw']
    c = meta['channels']
    patch_size = meta['patch_size']
    full_h, full_w = meta['full_hw']
    pad_h, pad_w = meta['pad_hw']

    x = tokens.view(b, gh, gw, c, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(b, c, full_h, full_w)

    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]

    return x