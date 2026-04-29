# -*- coding: utf-8 -*-
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt import FixedPromptBank, CLIPTextPromptBank, IntentRouter
from .fft_utils import (
    split_amplitude_phase,
    phase_wrap,
    rebuild_from_amplitude_phase,
    patchify_feature_map,
    unpatchify_feature_map,
)
from .scoring import TokenScoreNet
from .selection import topk_token_selection, straight_through_topk_mask, gather_tokens, scatter_tokens
from .interaction import SelectedTokenInteraction
from .bypass import LightweightTokenPreserver

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


class FrequencyTopDownBlock(nn.Module):
    """三层频率域 top-down 聚合块。"""

    def __init__(self, channels: int, prior_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.intent_affine = nn.Sequential(
            nn.Linear(prior_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, high_feat: torch.Tensor, skip_feat: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high_feat, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([high_up, skip_feat], dim=1))
        gamma, beta = self.intent_affine(intent).chunk(2, dim=-1)
        x = x * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)) + 0.1 * beta.unsqueeze(-1).unsqueeze(-1)
        return x + self.refine(x)


class HighLevelGuidedFrequencyFusion(nn.Module):
    """多尺度高层文本意图引导的频率 token 选择融合模块。

    现在支持三层频率候选特征 [L1, L2, L3]。每一层均执行：
    FFT -> amplitude / phase token scoring -> Top-K routing -> selected 强交互
    -> unselected 轻量保真旁路 -> amplitude / phase 重组 -> iFFT。
    最后通过 top-down frequency aggregation 得到最终 F_freq。
    """

    def __init__(self, in_channels: int = 64, patch_size: int = 4, prior_dim: int = 64,
                 amp_topk_ratio: float = 0.25, phase_topk_ratio: float = 0.25,
                 token_embed_dim: int = 128, num_heads: int = 4, return_aux: bool = False,
                 use_real_clip_prompt_bank: bool = False, clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None, clip_download_root: str = None, clip_device: str = None,
                 routing_temperature: float = 0.25, num_levels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        self.return_aux = return_aux
        self.routing_temperature = routing_temperature
        self.num_levels = num_levels
        token_dim = in_channels * patch_size * patch_size

        if prompt_texts is None:
            prompt_texts = [
                'salient targets',
                'structural contours',
                'fine textures',
                'balanced fusion',
                'low light enhancement',
            ]

        if use_real_clip_prompt_bank:
            self.prompt_bank = CLIPTextPromptBank(
                prior_dim=prior_dim,
                clip_model_name=clip_model_name,
                prompt_texts=prompt_texts,
                download_root=clip_download_root,
                clip_device=clip_device,
            )
            num_prompts = len(prompt_texts)
        else:
            self.prompt_bank = FixedPromptBank(prior_dim=prior_dim)
            num_prompts = len(self.prompt_bank.prompt_names)

        self.intent_router = IntentRouter(in_channels=in_channels, prior_dim=prior_dim, num_prompts=num_prompts)

        self.frequency_intent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prior_dim, prior_dim),
                nn.LayerNorm(prior_dim),
                nn.GELU(),
                nn.Linear(prior_dim, prior_dim),
            ) for _ in range(num_levels)
        ])
        self.spatial_intent_head = nn.Sequential(
            nn.Linear(prior_dim, prior_dim),
            nn.LayerNorm(prior_dim),
            nn.GELU(),
            nn.Linear(prior_dim, prior_dim),
        )

        self.amp_score = nn.ModuleList([
            TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
            for _ in range(num_levels)
        ])
        self.phase_score = nn.ModuleList([
            TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
            for _ in range(num_levels)
        ])
        self.amp_interaction = nn.ModuleList([
            SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
            for _ in range(num_levels)
        ])
        self.phase_interaction = nn.ModuleList([
            SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
            for _ in range(num_levels)
        ])
        self.amp_bypass = nn.ModuleList([
            LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)
            for _ in range(num_levels)
        ])
        self.phase_bypass = nn.ModuleList([
            LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)
            for _ in range(num_levels)
        ])

        self.level_refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            ) for _ in range(num_levels)
        ])
        self.level_affines = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prior_dim, in_channels * 2),
                nn.GELU(),
                nn.Linear(in_channels * 2, in_channels * 2),
            ) for _ in range(num_levels)
        ])
        self.topdown_l2 = FrequencyTopDownBlock(in_channels, prior_dim)
        self.topdown_l1 = FrequencyTopDownBlock(in_channels, prior_dim)
        self.final_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )

        # 兼容旧脚本可能直接访问 frequency_intent_head 的情况。
        self.frequency_intent_head = self.frequency_intent_heads[0]

    @staticmethod
    def _normalize_token_target(x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return ((x - x_min) / (x_max - x_min + 1e-6)).detach()

    @staticmethod
    def _as_three_levels(x: TensorOrPyramid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)):
            if len(x) < 3:
                raise ValueError('frequency feature pyramid must contain [L1, L2, L3].')
            return x[0], x[1], x[2]
        l1 = x
        l2 = F.avg_pool2d(l1, kernel_size=2, stride=2)
        l3 = F.avg_pool2d(l2, kernel_size=2, stride=2)
        return l1, l2, l3

    def _build_score_target(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, branch_type: str) -> torch.Tensor:
        vis_abs = torch.abs(vis_tokens)
        ir_abs = torch.abs(ir_tokens)
        diff = torch.mean(torch.abs(vis_tokens - ir_tokens), dim=-1)
        energy = 0.5 * (torch.mean(torch.log1p(vis_abs), dim=-1) + torch.mean(torch.log1p(ir_abs), dim=-1))
        max_response = torch.max(torch.mean(vis_abs, dim=-1), torch.mean(ir_abs, dim=-1))
        local_variation = 0.5 * (
            torch.std(vis_tokens, dim=-1, unbiased=False) + torch.std(ir_tokens, dim=-1, unbiased=False)
        )

        if branch_type == 'amp':
            raw_target = 0.45 * diff + 0.35 * energy + 0.20 * max_response
        elif branch_type == 'phase':
            raw_target = 0.55 * diff + 0.30 * local_variation + 0.15 * energy
        else:
            raw_target = 0.50 * diff + 0.25 * energy + 0.25 * local_variation
        return self._normalize_token_target(raw_target)

    def _fuse_branch(self, vis_map, ir_map, intent, scorer, interactor, bypass, keep_ratio, branch_type: str):
        vis_tokens, meta = patchify_feature_map(vis_map, self.patch_size)
        ir_tokens, _ = patchify_feature_map(ir_map, self.patch_size)

        score = scorer(vis_tokens, ir_tokens, meta['coords'], intent)
        topk_index, hard_mask, topk_value = topk_token_selection(score, keep_ratio)
        routing_mask = straight_through_topk_mask(
            score, hard_mask, keep_ratio=keep_ratio, temperature=self.routing_temperature
        ).unsqueeze(-1)

        vis_selected = gather_tokens(vis_tokens, topk_index)
        ir_selected = gather_tokens(ir_tokens, topk_index)
        strong_selected = interactor(vis_selected, ir_selected, intent)

        weak_full = bypass(vis_tokens, ir_tokens, intent)
        strong_full = scatter_tokens(weak_full, strong_selected, topk_index)
        fused_full = routing_mask * strong_full + (1.0 - routing_mask) * weak_full
        fused_map = unpatchify_feature_map(fused_full, meta)

        score_target = self._build_score_target(vis_tokens, ir_tokens, branch_type=branch_type)
        aux = {
            'score': score,
            'score_target': score_target,
            'mask': hard_mask.float(),
            'routing_mask': routing_mask.squeeze(-1),
            'topk_index': topk_index,
            'topk_value': topk_value,
        }
        return fused_map, aux

    def _fuse_level(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor,
                    frequency_intent: torch.Tensor, level_idx: int):
        spatial_size = vis_feat.shape[-2:]
        vis_amp, vis_phase = split_amplitude_phase(vis_feat)
        ir_amp, ir_phase = split_amplitude_phase(ir_feat)

        fused_amp, amp_aux = self._fuse_branch(
            vis_amp, ir_amp, frequency_intent, self.amp_score[level_idx], self.amp_interaction[level_idx],
            self.amp_bypass[level_idx], self.amp_topk_ratio, branch_type='amp'
        )
        fused_phase, phase_aux = self._fuse_branch(
            vis_phase, ir_phase, frequency_intent, self.phase_score[level_idx], self.phase_interaction[level_idx],
            self.phase_bypass[level_idx], self.phase_topk_ratio, branch_type='phase'
        )
        fused_phase = phase_wrap(fused_phase)
        fused_spatial = rebuild_from_amplitude_phase(fused_amp, fused_phase, spatial_size)
        fused_feature = self.level_refines[level_idx](fused_spatial)

        gamma, beta = self.level_affines[level_idx](frequency_intent).chunk(2, dim=-1)
        fused_feature = fused_feature * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)) \
            + 0.1 * beta.unsqueeze(-1).unsqueeze(-1)
        fused_feature = fused_feature + 0.5 * (vis_feat + ir_feat)
        return fused_feature, amp_aux, phase_aux

    def forward(self, vis_feat: TensorOrPyramid, ir_feat: TensorOrPyramid):
        vis_levels = self._as_three_levels(vis_feat)
        ir_levels = self._as_three_levels(ir_feat)

        text_intent, prompt_weight = self.intent_router(vis_levels[0], ir_levels[0], self.prompt_bank())
        spatial_intent = self.spatial_intent_head(text_intent)

        fused_levels: List[torch.Tensor] = []
        frequency_intents: List[torch.Tensor] = []
        amp_aux_list: List[Dict[str, torch.Tensor]] = []
        phase_aux_list: List[Dict[str, torch.Tensor]] = []

        for level_idx, (vis_l, ir_l) in enumerate(zip(vis_levels, ir_levels)):
            frequency_intent = self.frequency_intent_heads[level_idx](text_intent)
            fused_level, amp_aux, phase_aux = self._fuse_level(vis_l, ir_l, frequency_intent, level_idx)
            fused_levels.append(fused_level)
            frequency_intents.append(frequency_intent)
            amp_aux_list.append(amp_aux)
            phase_aux_list.append(phase_aux)

        td_l2 = self.topdown_l2(fused_levels[2], fused_levels[1], frequency_intents[1])
        td_l1 = self.topdown_l1(td_l2, fused_levels[0], frequency_intents[0])
        fused_feature = td_l1 + self.final_refine(td_l1)

        if not self.return_aux:
            return fused_feature

        aux = {
            'intent': text_intent,
            'frequency_intent': frequency_intents,
            'frequency_intent_l1': frequency_intents[0],
            'frequency_intent_l2': frequency_intents[1],
            'frequency_intent_l3': frequency_intents[2],
            'spatial_intent': spatial_intent,
            'prompt_weight': prompt_weight,
            'fused_freq_levels': fused_levels,
            'fused_freq_l1': fused_levels[0],
            'fused_freq_l2': fused_levels[1],
            'fused_freq_l3': fused_levels[2],
            'amp_score': [a['score'] for a in amp_aux_list],
            'phase_score': [p['score'] for p in phase_aux_list],
            'amp_score_target': [a['score_target'] for a in amp_aux_list],
            'phase_score_target': [p['score_target'] for p in phase_aux_list],
            'amp_mask': [a['mask'] for a in amp_aux_list],
            'phase_mask': [p['mask'] for p in phase_aux_list],
            'amp_routing_mask': [a['routing_mask'] for a in amp_aux_list],
            'phase_routing_mask': [p['routing_mask'] for p in phase_aux_list],
            'amp_topk_index': [a['topk_index'] for a in amp_aux_list],
            'phase_topk_index': [p['topk_index'] for p in phase_aux_list],
        }
        for idx in range(self.num_levels):
            level = idx + 1
            aux[f'amp_score_l{level}'] = amp_aux_list[idx]['score']
            aux[f'phase_score_l{level}'] = phase_aux_list[idx]['score']
            aux[f'amp_score_target_l{level}'] = amp_aux_list[idx]['score_target']
            aux[f'phase_score_target_l{level}'] = phase_aux_list[idx]['score_target']
            aux[f'amp_mask_l{level}'] = amp_aux_list[idx]['mask']
            aux[f'phase_mask_l{level}'] = phase_aux_list[idx]['mask']
        return fused_feature, aux
