# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LightweightTokenPreserver(nn.Module):
    """Lightweight preservation path for low-score frequency tokens.

    Low-score tokens use only a local gate to mix VIS/IR tokens, followed by a
    small residual refine block. Text intent is accepted only for caller
    compatibility and does not modulate bypass features.
    """

    def __init__(self, token_dim: int, prior_dim: int = 64, init_res_scale: float = 0.05):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

        # Start close to identity preservation, then learn refinement.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor,
                intent: torch.Tensor = None) -> torch.Tensor:
        concat = torch.cat([vis_tokens, ir_tokens], dim=-1)
        gate = self.gate(concat)
        mixed = gate * vis_tokens + (1.0 - gate) * ir_tokens
        return mixed + self.res_scale * self.refine(mixed)
