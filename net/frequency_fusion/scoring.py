# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class TokenScoreNet(nn.Module):
    """Estimate degradation-aware importance scores for frequency tokens.

    The degradation intent I_deg generates FiLM-style modulation parameters to
    recalibrate intermediate scoring representations before importance scoring
    and Top-K routing. I_deg does not directly change amplitude/phase spectral
    values and does not modulate downstream token fusion features.
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
        self.intent_mod = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
        )
        nn.init.zeros_(self.intent_mod[-1].weight)
        nn.init.zeros_(self.intent_mod[-1].bias)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
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
        token_feature = torch.cat([vis_tokens, ir_tokens], dim=-1)
        cross_modal_difference = torch.abs(vis_tokens - ir_tokens)

        token_feat = self.token_proj(token_feature)
        diff_feat = self.diff_proj(cross_modal_difference)
        coord_feat = self.coord_proj(coords)

        gamma_t, beta_t, gamma_d, beta_d = self.intent_mod(intent).chunk(4, dim=-1)
        token_feat = token_feat * (1.0 + gamma_t.unsqueeze(1)) + beta_t.unsqueeze(1)
        diff_feat = diff_feat * (1.0 + gamma_d.unsqueeze(1)) + beta_d.unsqueeze(1)

        fused = torch.cat([token_feat, diff_feat, coord_feat], dim=-1)
        score = self.score_mlp(fused).squeeze(-1)
        return score
