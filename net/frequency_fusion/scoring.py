# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class TokenScoreNet(nn.Module):
    """Token 重要性评分网络。"""
    def __init__(self, token_dim: int, prior_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.coord_proj = nn.Linear(2, 16)
        self.prior_proj = nn.Linear(prior_dim, 32)
        self.score_mlp = nn.Sequential(
            nn.Linear(token_dim * 3 + 16 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, coords: torch.Tensor, intent: torch.Tensor):
        _, n, _ = vis_tokens.shape
        diff = torch.abs(vis_tokens - ir_tokens)
        coord_feat = self.coord_proj(coords)
        prior_feat = self.prior_proj(intent).unsqueeze(1).repeat(1, n, 1)
        fused = torch.cat([vis_tokens, ir_tokens, diff, coord_feat, prior_feat], dim=-1)
        score = self.score_mlp(fused).squeeze(-1)
        return score
