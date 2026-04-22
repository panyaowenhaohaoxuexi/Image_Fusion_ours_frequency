# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class FixedPromptBank(nn.Module):
    """固定高层先验库。当前阶段不接真实 CLIP。"""

    def __init__(self, prior_dim: int = 64):
        super().__init__()
        self.prompt_names = [
            'salient_targets',
            'structural_contours',
            'fine_textures',
            'balanced_fusion',
        ]
        bank = torch.zeros(len(self.prompt_names), prior_dim, dtype=torch.float32)
        for i in range(len(self.prompt_names)):
            bank[i, i::len(self.prompt_names)] = 1.0
        bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer('prompt_bank', bank)

    def forward(self) -> torch.Tensor:
        return self.prompt_bank


class IntentRouter(nn.Module):
    def __init__(self, in_channels: int, prior_dim: int, num_prompts: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_prompts),
        )
        self.proj = nn.Linear(prior_dim, prior_dim)

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, prompt_bank: torch.Tensor):
        b, c, _, _ = vis_feat.shape
        vis_vec = self.pool(vis_feat).view(b, c)
        ir_vec = self.pool(ir_feat).view(b, c)
        fusion_vec = torch.cat([vis_vec, ir_vec], dim=1)
        prompt_logits = self.mlp(fusion_vec)
        prompt_weight = torch.softmax(prompt_logits, dim=1)
        prompt_bank = prompt_bank.unsqueeze(0).repeat(b, 1, 1)
        intent = torch.sum(prompt_weight.unsqueeze(-1) * prompt_bank, dim=1)
        intent = self.proj(intent)
        return intent, prompt_weight
