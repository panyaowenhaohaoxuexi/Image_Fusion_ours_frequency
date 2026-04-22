# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SelectedTokenInteraction(nn.Module):
    """只对 Top-K token 做跨模态交互。"""
    def __init__(self, token_dim: int, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.vis_proj = nn.Linear(token_dim, embed_dim)
        self.ir_proj = nn.Linear(token_dim, embed_dim)
        self.cross_attn_vis = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_ir = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_vis = nn.LayerNorm(embed_dim)
        self.norm_ir = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, token_dim)
    def forward(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor):
        vis_embed = self.vis_proj(vis_tokens)
        ir_embed = self.ir_proj(ir_tokens)
        vis_update, _ = self.cross_attn_vis(vis_embed, ir_embed, ir_embed)
        vis_update = self.norm_vis(vis_embed + vis_update)
        ir_update, _ = self.cross_attn_ir(ir_embed, vis_embed, vis_embed)
        ir_update = self.norm_ir(ir_embed + ir_update)
        fused = 0.5 * (vis_update + ir_update)
        return self.out_proj(fused)
