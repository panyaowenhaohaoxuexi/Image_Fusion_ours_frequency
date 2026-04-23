# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class SelectedTokenInteraction(nn.Module):
    """
    只对 Top-K token 做跨模态强交互。

    当前版本将原先的：
        Cross-ATT -> 直接平均 -> intent affine -> FFN
    改为：
        双向 Cross-ATT -> token级融合投影 -> token-channel SE重标定 ->
        intent affine -> FFN

    设计动机：
    1) Cross-ATT 负责跨模态信息交换；
    2) SE重标定负责在交互后重新强调关键通道；
    3) intent 继续作为条件向量调制 selected token 的交互结果。

    说明：
    - 这里的 SE 不是标准 2D feature-map SE，而是针对 [B, K, C] token 序列
      的 token-channel SE。
    - 为兼容旧版 PyTorch，MultiheadAttention 仍不使用 batch_first=True。
    """

    def __init__(self, token_dim: int, embed_dim: int = 128, num_heads: int = 4, intent_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.intent_dim = intent_dim

        self.vis_proj = nn.Linear(token_dim, embed_dim)
        self.ir_proj = nn.Linear(token_dim, embed_dim)

        # 旧版 PyTorch 不支持 batch_first=True
        self.cross_attn_vis = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_ir = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.norm_vis = nn.LayerNorm(embed_dim)
        self.norm_ir = nn.LayerNorm(embed_dim)
        self.norm_fused = nn.LayerNorm(embed_dim)

        # 交互后先显式聚合 vis / ir / diff，再交给 SE 进一步重标定
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # token-channel SE：对 selected token 序列做通道重标定
        se_hidden = max(embed_dim // 4, 16)
        self.se_reduce = nn.Linear(embed_dim + intent_dim, se_hidden)
        self.se_expand = nn.Linear(se_hidden, embed_dim)
        self.se_act = nn.GELU()
        self.se_gate = nn.Sigmoid()

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

    def _token_se(self, fused_tokens: torch.Tensor, intent: torch.Tensor = None) -> torch.Tensor:
        """
        fused_tokens: [B, K, C]
        intent:       [B, D] or None
        """
        # 针对 selected token 序列做 squeeze，统计关键 token 的通道响应
        token_stat = fused_tokens.mean(dim=1)

        if intent is None:
            if self.intent_dim > 0:
                zeros = token_stat.new_zeros(token_stat.size(0), self.intent_dim)
                se_inp = torch.cat([token_stat, zeros], dim=-1)
            else:
                se_inp = token_stat
        else:
            se_inp = torch.cat([token_stat, intent], dim=-1)

        gate = self.se_reduce(se_inp)
        gate = self.se_act(gate)
        gate = self.se_expand(gate)
        gate = self.se_gate(gate).unsqueeze(1)

        # 残差式 SE，避免纯乘法压得太狠
        return fused_tokens + fused_tokens * gate

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

        # 双向 Cross-ATT：先交换跨模态信息
        vis_update_t, _ = self.cross_attn_vis(vis_embed_t, ir_embed_t, ir_embed_t)
        ir_update_t, _ = self.cross_attn_ir(ir_embed_t, vis_embed_t, vis_embed_t)

        vis_update = vis_update_t.transpose(0, 1)
        ir_update = ir_update_t.transpose(0, 1)

        vis_update = self.norm_vis(vis_embed + vis_update)
        ir_update = self.norm_ir(ir_embed + ir_update)

        # 比“直接平均”更强：先聚合 vis / ir / diff，再做 token-SE 重标定
        diff = torch.abs(vis_update - ir_update)
        fused = self.fuse_proj(torch.cat([vis_update, ir_update, diff], dim=-1))
        fused = self._token_se(fused, intent)

        if intent is not None:
            gamma, beta = self.intent_affine(intent).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            fused = fused * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        fused = fused + self.ffn(self.norm_fused(fused))
        return self.out_proj(fused)
