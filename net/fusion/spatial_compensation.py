# -*- coding: utf-8 -*-
"""频率参照引导的空间残差补偿模块。

该版本是 v5 的保守增强版：保留“F_freq 参照 -> 源模态残差 -> 门控补偿”的闭环，
但避免补偿分支在训练早期过强地改写 v4 strong baseline。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ConvBNAct, ResidualBlock


class SpatialResidualCompensation(nn.Module):
    """Frequency-reference guided spatial residual compensation.

    输入:
        base_init: BaseFusion(vis_base, ir_base) 得到的初始基础融合特征
        freq_feat: 频率分支输出 F_freq
        vis_base: 可见光基础特征
        ir_base: 红外基础特征

    输出:
        base_out: 补偿后的基础特征 F_base

    关键约束:
        1. 补偿只作为 residual correction，不重新生成 base 特征；
        2. 可见光残差偏向亮度、背景、纹理自然性；
        3. 红外残差偏向热目标主体、显著响应和目标-背景对比；
        4. 最后一层零初始化，使初始输出等价于 v4 的 base_init。
    """

    def __init__(self, channels: int = 64, init_scale: float = 0.03):
        super().__init__()

        self.freq_ref_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ResidualBlock(channels),
        )

        # 可见光补偿更关注纹理和局部亮度连续性，因此显式引入轻量高频残差。
        self.vis_res_refine = nn.Sequential(
            ConvBNAct(channels * 2, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

        # 红外补偿保留目标主体和显著对比，使用原残差与高频残差共同判断。
        self.ir_res_refine = nn.Sequential(
            ConvBNAct(channels * 2, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

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

        self.comp_mixer = nn.Sequential(
            ConvBNAct(channels * 3, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )
        self.comp_out = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 初始严格不破坏 baseline；训练中只学习必要补偿。
        nn.init.zeros_(self.comp_out.weight)
        self.comp_scale = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

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
        r_vis_hp = self._high_pass(r_vis)
        r_ir_hp = self._high_pass(r_ir)

        vis_gate_input = torch.cat([r_vis, torch.abs(r_vis), vis_base, f_ref, base_init], dim=1)
        ir_gate_input = torch.cat([r_ir, torch.abs(r_ir), ir_base, f_ref, base_init], dim=1)

        g_vis = self.vis_gate(vis_gate_input)
        g_ir = self.ir_gate(ir_gate_input)

        c_vis = g_vis * self.vis_res_refine(torch.cat([r_vis, r_vis_hp], dim=1))
        c_ir = g_ir * self.ir_res_refine(torch.cat([r_ir, r_ir_hp], dim=1))

        # mixer 使用 base_init 作为上下文，避免把 F_ref 再次强行注入 base 分支。
        f_comp = self.comp_out(self.comp_mixer(torch.cat([c_vis, c_ir, base_init], dim=1)))
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
