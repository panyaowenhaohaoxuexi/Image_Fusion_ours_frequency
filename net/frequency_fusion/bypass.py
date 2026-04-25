# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LightweightTokenPreserver(nn.Module):
    """未被选中的 token 的保真旁路。

    设计重点：
    1. 旁路首先做源模态自适应加权，保证 unselected tokens 不被丢弃；
    2. intent 只做轻量仿射调制，让文本先验影响旁路保留偏好；
    3. refine 采用小残差形式，并将最后一层零初始化，避免训练初期破坏 v4 strong baseline。
    """

    def __init__(self, token_dim: int, prior_dim: int = 64, init_res_scale: float = 0.05):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )
        self.intent_affine = nn.Sequential(
            nn.Linear(prior_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim * 2),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

        # 使旁路初始行为接近“可靠保留”，后续再学习细化。
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, intent: torch.Tensor = None) -> torch.Tensor:
        concat = torch.cat([vis_tokens, ir_tokens], dim=-1)
        gate = self.gate(concat)
        mixed = gate * vis_tokens + (1.0 - gate) * ir_tokens

        if intent is not None:
            gamma, beta = self.intent_affine(intent).chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(1)
            beta = torch.tanh(beta).unsqueeze(1)
            mixed = mixed * (1.0 + 0.1 * gamma) + 0.05 * beta

        return mixed + self.res_scale * self.refine(mixed)
