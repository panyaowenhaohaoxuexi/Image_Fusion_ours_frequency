# -*- coding: utf-8 -*-
"""空间残差补偿模块。

该模块用于把频率分支输出 F_freq 作为参照，检查 vis_base / ir_base 中
仍未被频率融合充分保留的信息，并将有效残差补回 F_base_init。
"""
import torch
import torch.nn as nn

from net.encoder.blocks import ConvBNAct, ResidualBlock


class SpatialResidualCompensation(nn.Module):
    """
    Frequency-reference guided spatial residual compensation.

    输入:
        base_init: BaseFusion(vis_base, ir_base) 得到的初始基础融合特征
        freq_feat: 频率分支输出 F_freq
        vis_base: 可见光基础特征
        ir_base: 红外基础特征

    输出:
        base_out: 补偿后的基础特征 F_base

    设计动机:
        1. F_freq 提供频率融合后的参照，表示频率分支已经融合了什么；
        2. vis_base - F_ref 提供可见光遗漏候选；
        3. ir_base - F_ref 提供红外遗漏候选；
        4. 通过门控判断残差是否值得补，避免把噪声/冗余响应直接加回去。
    """

    def __init__(self, channels: int = 64, init_scale: float = 0.1):
        super().__init__()

        # 将频率融合特征投影到 base 特征空间，作为残差比较参照 F_ref。
        self.freq_ref_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ResidualBlock(channels),
        )

        # 残差 refine 只处理残差候选本身，保持轻量。
        self.vis_res_refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )
        self.ir_res_refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

        # 门控输入使用 [残差, |残差|, 源模态 base, F_ref, F_base_init]。
        # 残差指出“差在哪里”，源模态 base 判断“差异来源”，F_ref 判断“频率分支已有信息”，
        # F_base_init 判断“当前空间底座状态”。
        gate_in_channels = channels * 5
        self.vis_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.ir_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 将可见光补偿、红外补偿与频率参照统一整合为 F_comp。
        self.comp_mixer = nn.Sequential(
            ConvBNAct(channels * 3, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # 可学习补偿强度。初值较小，避免训练初期破坏 strong baseline。
        self.comp_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        base_init: torch.Tensor,
        freq_feat: torch.Tensor,
        vis_base: torch.Tensor,
        ir_base: torch.Tensor,
        return_aux: bool = False,
    ):
        f_ref = self.freq_ref_proj(freq_feat)

        r_vis = vis_base - f_ref
        r_ir = ir_base - f_ref

        vis_gate_input = torch.cat([r_vis, torch.abs(r_vis), vis_base, f_ref, base_init], dim=1)
        ir_gate_input = torch.cat([r_ir, torch.abs(r_ir), ir_base, f_ref, base_init], dim=1)

        g_vis = self.vis_gate(vis_gate_input)
        g_ir = self.ir_gate(ir_gate_input)

        c_vis = g_vis * self.vis_res_refine(r_vis)
        c_ir = g_ir * self.ir_res_refine(r_ir)

        f_comp = self.comp_mixer(torch.cat([c_vis, c_ir, f_ref], dim=1))
        base_out = base_init + self.comp_scale * f_comp

        if not return_aux:
            return base_out

        aux = {
            'freq_ref': f_ref,
            'vis_residual': r_vis,
            'ir_residual': r_ir,
            'vis_gate': g_vis,
            'ir_gate': g_ir,
            'compensation': f_comp,
            'comp_scale': self.comp_scale.detach(),
        }
        return base_out, aux
