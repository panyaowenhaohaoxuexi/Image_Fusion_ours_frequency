# -*- coding: utf-8 -*-
"""Text-conditioned spatial adaptive fusion branch.

该模块对应当前论文方案中的空间域分支：
    vis_spa / ir_spa
    -> cross-modal spatial interaction
    -> Text Intent-conditioned CSAF / PSAF
    -> F_spa

实现原则：
1. 条件源只来自 CLIP text encoder + IntentRouter 得到的 Text Intent Embedding；
2. 不使用 CLIP image encoder / DINO / 对象级 mask；
3. 空间分支与频率分支并行输出 F_spa，频率特征只作为可选参考，不把 F_freq 当作唯一指导信号；
4. 最后一层零初始化，初始阶段接近稳定的 BaseFusion 输出，降低训练不稳定风险。
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ConvBNAct, ResidualBlock
from net.restormer_light import TransformerBlock


class TextFeatureWiseAffine(nn.Module):
    """Text-IF 式文本仿射调制：由 Text Intent 生成 gamma / beta。"""

    def __init__(self, intent_dim: int, channels: int, hidden_dim: Optional[int] = None,
                 gamma_scale: float = 0.1, beta_scale: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(channels * 2, intent_dim)
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.mlp = nn.Sequential(
            nn.Linear(intent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(intent).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.gamma_scale * gamma) + self.beta_scale * beta


class CrossModalSpatialInteraction(nn.Module):
    """轻量跨模态空间交互。

    SPGFusion 的 Cross_attention 是全空间注意力，单尺度 128x128 特征上显存开销较大。
    这里采用 Restormer 风格的通道注意力形式完成双向跨模态交互：
        visible query attends to infrared key/value;
        infrared query attends to visible key/value.
    它仍然建模空间展平后的全局响应，但注意力矩阵尺寸为 CxC，适合当前工程。
    """

    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False, norm_groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0, 'channels must be divisible by num_heads.'
        self.channels = channels
        self.num_heads = num_heads
        self.temperature_vis = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_ir = nn.Parameter(torch.ones(num_heads, 1, 1))

        groups = min(norm_groups, channels)
        self.norm_vis = nn.GroupNorm(groups, channels)
        self.norm_ir = nn.GroupNorm(groups, channels)
        self.qkv_vis = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_ir = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.dw_vis = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1,
                                groups=channels * 3, bias=bias)
        self.dw_ir = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1,
                               groups=channels * 3, bias=bias)
        self.out_vis = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.out_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        d = c // self.num_heads
        return x.view(b, self.num_heads, d, h * w)

    def _cross_update(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      temperature: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * temperature
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = vis_feat.shape
        q_vis, k_vis, v_vis = self.dw_vis(self.qkv_vis(self.norm_vis(vis_feat))).chunk(3, dim=1)
        q_ir, k_ir, v_ir = self.dw_ir(self.qkv_ir(self.norm_ir(ir_feat))).chunk(3, dim=1)

        q_vis = self._reshape(q_vis)
        k_vis = self._reshape(k_vis)
        v_vis = self._reshape(v_vis)
        q_ir = self._reshape(q_ir)
        k_ir = self._reshape(k_ir)
        v_ir = self._reshape(v_ir)

        vis_from_ir = self._cross_update(q_vis, k_ir, v_ir, self.temperature_vis)
        ir_from_vis = self._cross_update(q_ir, k_vis, v_vis, self.temperature_ir)

        vis_from_ir = vis_from_ir.view(b, c, h, w)
        ir_from_vis = ir_from_vis.view(b, c, h, w)
        vis_out = vis_feat + self.scale * self.out_vis(vis_from_ir)
        ir_out = ir_feat + self.scale * self.out_ir(ir_from_vis)
        return vis_out, ir_out


class TextConditionedCSAF(nn.Module):
    """Text Intent-conditioned CSAF。

    借鉴 SPGFusion 中条件归一化式自适应融合的结构思想，但条件输入改为 Text Intent。
    """

    def __init__(self, channels: int, intent_dim: int = 64, norm_groups: int = 8):
        super().__init__()
        groups = min(norm_groups, channels)
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.affine = TextFeatureWiseAffine(intent_dim, channels)
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        modulated = self.affine(self.norm(x), intent)
        return x + self.refine(modulated)


class SpatialSelfAttentionRefine(nn.Module):
    """空间融合后的轻量全局细化。"""

    def __init__(self, channels: int, num_heads: int = 4,
                 ffn_expansion_factor: float = 2.0, bias: bool = False,
                 LayerNorm_type: str = 'WithBias'):
        super().__init__()
        self.block = TransformerBlock(
            dim=channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TextConditionedSpatialAdaptiveFusion(nn.Module):
    """当前方案的空间域分支。

    输入:
        vis_spa / ir_spa: 编码器输出的空间域特征；
        text_intent: 频率分支中同一个 IntentRouter 产生并经 Spatial-Intent Head 投影的意图；
        freq_feat: 可选，仅作为聚合上下文，不作为空间分支的唯一指导。

    输出:
        F_spa；可选返回 gate / branch 特征用于诊断。
    """

    def __init__(self, channels: int = 64, intent_dim: int = 64, num_heads: int = 4,
                 ffn_expansion_factor: float = 2.0, init_res_scale: float = 0.05,
                 use_freq_context: bool = True):
        super().__init__()
        self.use_freq_context = use_freq_context
        self.cross_interaction = CrossModalSpatialInteraction(channels=channels, num_heads=num_heads)
        self.vis_csaf = TextConditionedCSAF(channels=channels, intent_dim=intent_dim)
        self.ir_csaf = TextConditionedCSAF(channels=channels, intent_dim=intent_dim)

        # Text-conditioned fusion gate：决定空间融合中可见光/红外的保留比例。
        self.intent_gate = nn.Sequential(
            nn.Linear(intent_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        gate_in = channels * 4 + (channels if use_freq_context else 0)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(gate_in, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        mixer_in = channels * 3 + (channels if use_freq_context else 0)
        self.mixer = nn.Sequential(
            ConvBNAct(mixer_in, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )
        self.psaf_refine = SpatialSelfAttentionRefine(
            channels=channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.out = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.zeros_(self.out.weight)
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, vis_spa: torch.Tensor, ir_spa: torch.Tensor, text_intent: torch.Tensor,
                freq_feat: Optional[torch.Tensor] = None, return_aux: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        vis_inter, ir_inter = self.cross_interaction(vis_spa, ir_spa)
        vis_cond = self.vis_csaf(vis_inter, text_intent)
        ir_cond = self.ir_csaf(ir_inter, text_intent)

        diff = torch.abs(vis_cond - ir_cond)
        common = 0.5 * (vis_cond + ir_cond)
        hp_common = self._high_pass(common)

        gate_context = [vis_cond, ir_cond, diff, common]
        mix_context = [vis_cond, ir_cond, hp_common]
        if self.use_freq_context and freq_feat is not None:
            gate_context.append(freq_feat)
            mix_context.append(freq_feat)

        gate = self.fusion_gate(torch.cat(gate_context, dim=1))
        intent_gate = torch.sigmoid(self.intent_gate(text_intent)).unsqueeze(-1).unsqueeze(-1)
        gate = torch.clamp(0.5 * gate + 0.5 * intent_gate, 0.0, 1.0)

        gated = gate * vis_cond + (1.0 - gate) * ir_cond
        mixed = self.mixer(torch.cat(mix_context, dim=1)) + gated
        refined = self.psaf_refine(mixed)

        # 初始阶段接近 gated/mixed，后续学习文本条件空间残差，避免破坏原 baseline。
        out = mixed + self.res_scale * self.out(refined)

        if not return_aux:
            return out
        aux = {
            'vis_inter': vis_inter,
            'ir_inter': ir_inter,
            'vis_cond': vis_cond,
            'ir_cond': ir_cond,
            'spatial_gate': gate,
            'spatial_res_scale': self.res_scale.detach(),
        }
        return out, aux
