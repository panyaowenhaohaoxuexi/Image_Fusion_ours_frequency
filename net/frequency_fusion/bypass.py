# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LightweightTokenPreserver(nn.Module):
    """
    未被选中的 token 走轻量保留路径。
    在原有门控混合基础上补一个 detail residual，增强信息保真。
    """

    def __init__(self, token_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )
        self.detail_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([vis_tokens, ir_tokens], dim=-1)
        gate = self.gate(concat)
        mixed = gate * vis_tokens + (1.0 - gate) * ir_tokens

        detail = torch.abs(vis_tokens - ir_tokens)
        detail_weight = self.detail_gate(concat)
        mixed = mixed + 0.5 * detail_weight * detail

        return self.refine(mixed) + mixed
