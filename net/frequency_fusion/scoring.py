# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class TokenScoreNet(nn.Module):
    """Token 重要性评分网络。

    评分依据显式拆成三类：
    1) Token Feature：可见光 / 红外 token 自身表征；
    2) Cross-modal Difference：两模态 token 的互补差异；
    3) Text Intent Embedding：由固定 prompt bank / CLIP text encoder 得到的高层意图。

    输出 score 只用于 Top-K 排序和路由，不直接作为频谱值的全局缩放因子。
    """

    def __init__(self, token_dim: int, prior_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.GELU(),
        )
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor,
                coords: torch.Tensor, intent: torch.Tensor):
        """
        vis_tokens: [B, N, D]
        ir_tokens:  [B, N, D]
        coords:     [B, N, 2]
        intent:     [B, P]
        """
        _, n, _ = vis_tokens.shape
        token_feature = torch.cat([vis_tokens, ir_tokens], dim=-1)
        cross_modal_difference = torch.abs(vis_tokens - ir_tokens)

        token_feat = self.token_proj(token_feature)
        diff_feat = self.diff_proj(cross_modal_difference)
        coord_feat = self.coord_proj(coords)
        text_feat = self.prior_proj(intent).unsqueeze(1).expand(-1, n, -1)

        fused = torch.cat([token_feat, diff_feat, text_feat, coord_feat], dim=-1)
        score = self.score_mlp(fused).squeeze(-1)
        return score
