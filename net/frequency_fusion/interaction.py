# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class SelectedTokenInteraction(nn.Module):
    """Selected tokens 的强交互路径。

    Top-K selected tokens 先做双向跨模态交叉注意力，再在 selected-token 子集内部
    做自注意力和 FFN 细化。该模块只处理高分 tokens，体现“关键频率强交互”。
    """

    def __init__(self, token_dim: int, embed_dim: int = 128, num_heads: int = 4, prior_dim: int = 64):
        super().__init__()
        self.vis_proj = nn.Linear(token_dim, embed_dim)
        self.ir_proj = nn.Linear(token_dim, embed_dim)

        # 兼容旧版 PyTorch，不使用 batch_first=True。
        self.cross_attn_vis = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_ir = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.norm_vis = nn.LayerNorm(embed_dim)
        self.norm_ir = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.intent_affine = nn.Sequential(
            nn.Linear(prior_dim, embed_dim * 2),
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
        intent:     [B, P] or None
        """
        vis_embed = self.vis_proj(vis_tokens)   # [B, K, E]
        ir_embed = self.ir_proj(ir_tokens)      # [B, K, E]

        vis_embed_t = vis_embed.transpose(0, 1) # [K, B, E]
        ir_embed_t = ir_embed.transpose(0, 1)

        vis_update_t, _ = self.cross_attn_vis(vis_embed_t, ir_embed_t, ir_embed_t)
        ir_update_t, _ = self.cross_attn_ir(ir_embed_t, vis_embed_t, vis_embed_t)

        vis_update = self.norm_vis(vis_embed + vis_update_t.transpose(0, 1))
        ir_update = self.norm_ir(ir_embed + ir_update_t.transpose(0, 1))

        fused = self.norm_cross(0.5 * (vis_update + ir_update))

        if intent is not None:
            gamma, beta = self.intent_affine(intent).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            fused = fused * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        fused_t = fused.transpose(0, 1)
        self_update_t, _ = self.self_attn(fused_t, fused_t, fused_t)
        fused = self.norm_self(fused + self_update_t.transpose(0, 1))
        fused = fused + self.ffn(self.norm_ffn(fused))
        return self.out_proj(fused)
