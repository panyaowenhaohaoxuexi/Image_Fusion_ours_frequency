# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LightweightTokenPreserver(nn.Module):
    """Lightweight preservation path for low-score frequency tokens."""

    def __init__(self, token_dim: int, prior_dim: int = 64, init_res_scale: float = 0.05):
        super().__init__()
        hidden_ca = max(token_dim // 4, 1)
        self.mix_mlp = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_ca),
            nn.GELU(),
            nn.Linear(hidden_ca, token_dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor,
                intent: torch.Tensor = None) -> torch.Tensor:
        fused = self.mix_mlp(torch.cat([vis_tokens, ir_tokens], dim=-1))
        attended = fused * self.channel_attn(fused)
        return attended + self.res_scale * self.ffn(attended)
