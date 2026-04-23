# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class SelectedTokenInteraction(nn.Module):
    """只对 Top-K token 做跨模态交互。兼容旧版 PyTorch。"""
    def __init__(self, token_dim: int, embed_dim: int = 128, num_heads: int = 4, intent_dim: int = 64):
        super().__init__()
        self.vis_proj = nn.Linear(token_dim, embed_dim)
        self.ir_proj = nn.Linear(token_dim, embed_dim)

        # 旧版 PyTorch 不支持 batch_first=True
        self.cross_attn_vis = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_ir = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.norm_vis = nn.LayerNorm(embed_dim)
        self.norm_ir = nn.LayerNorm(embed_dim)
        self.norm_fused = nn.LayerNorm(embed_dim)

        # 借鉴 Text-IF 的 affine 调制思想：intent 不只管选谁，也参与怎么融
        self.intent_affine = nn.Sequential(
            nn.Linear(intent_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.out_proj = nn.Linear(embed_dim, token_dim)

    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, intent: torch.Tensor = None):
        """
        vis_tokens: [B, K, C]
        ir_tokens:  [B, K, C]
        intent:     [B, D]
        """
        vis_embed = self.vis_proj(vis_tokens)
        ir_embed = self.ir_proj(ir_tokens)

        vis_embed_t = vis_embed.transpose(0, 1)
        ir_embed_t = ir_embed.transpose(0, 1)

        vis_update_t, _ = self.cross_attn_vis(vis_embed_t, ir_embed_t, ir_embed_t)
        ir_update_t, _ = self.cross_attn_ir(ir_embed_t, vis_embed_t, vis_embed_t)

        vis_update = vis_update_t.transpose(0, 1)
        ir_update = ir_update_t.transpose(0, 1)

        vis_update = self.norm_vis(vis_embed + vis_update)
        ir_update = self.norm_ir(ir_embed + ir_update)

        fused = 0.5 * (vis_update + ir_update)

        if intent is not None:
            gamma, beta = self.intent_affine(intent).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            fused = fused * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        fused = fused + self.ffn(self.norm_fused(fused))
        return self.out_proj(fused)
