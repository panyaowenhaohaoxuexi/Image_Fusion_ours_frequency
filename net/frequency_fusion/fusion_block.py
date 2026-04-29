# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
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


class HighLevelGuidedFrequencyFusion(nn.Module):
    """高层先验引导的频率 token 选择融合模块。

    核心逻辑：
    Token Feature + Cross-modal Difference + Text Intent Embedding 共同决定 score；
    score 经 Top-K 排序得到 selected / unselected tokens；
    selected tokens 进入强交互路径，unselected tokens 进入弱交互 / 保真路径；
    最后按原 token 位置重组为完整 amplitude / phase 频谱。
    """

    def __init__(self, in_channels: int = 64, patch_size: int = 4, prior_dim: int = 64,
                 amp_topk_ratio: float = 0.25, phase_topk_ratio: float = 0.25,
                 token_embed_dim: int = 128, num_heads: int = 4, return_aux: bool = False,
                 use_real_clip_prompt_bank: bool = False, clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None, clip_download_root: str = None, clip_device: str = None,
                 routing_temperature: float = 0.25):
        super().__init__()
        self.patch_size = patch_size
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        self.return_aux = return_aux
        self.routing_temperature = routing_temperature
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

        # 同一个 Text Intent Embedding 分成两个投影头：Frequency-Intent Head 和 Spatial-Intent Head。
        self.frequency_intent_head = nn.Sequential(
            nn.Linear(prior_dim, prior_dim),
            nn.LayerNorm(prior_dim),
            nn.GELU(),
            nn.Linear(prior_dim, prior_dim),
        )
        self.spatial_intent_head = nn.Sequential(
            nn.Linear(prior_dim, prior_dim),
            nn.LayerNorm(prior_dim),
            nn.GELU(),
            nn.Linear(prior_dim, prior_dim),
        )

        self.amp_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.phase_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.amp_interaction = SelectedTokenInteraction(
            token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim
        )
        self.phase_interaction = SelectedTokenInteraction(
            token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim
        )
        self.amp_bypass = LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)
        self.phase_bypass = LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
        self.refine_affine = nn.Sequential(
            nn.Linear(prior_dim, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels * 2),
        )

    @staticmethod
    def _normalize_token_target(x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return ((x - x_min) / (x_max - x_min + 1e-6)).detach()

    def _build_score_target(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, branch_type: str) -> torch.Tensor:
        """构造无 GT 条件下的 token 排序软目标。

        该目标只用于辅助 scorer 学排序，不改变前向的 hard Top-K 路由语义。
        amplitude 更偏向能量 / 亮度 / 对比差异；phase 更偏向结构差异 / 局部变化。
        """
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

        # 1) 三类信息共同决定 score。score 的主作用是排序和路由。
        score = scorer(vis_tokens, ir_tokens, meta['coords'], intent)

        # 2) Hard Top-K 负责前向分流：高分 selected，低分 unselected。
        topk_index, hard_mask, topk_value = topk_token_selection(score, keep_ratio)
        routing_mask = straight_through_topk_mask(
            score, hard_mask, keep_ratio=keep_ratio, temperature=self.routing_temperature
        ).unsqueeze(-1)

        # 3) Selected tokens 走强交互路径。
        vis_selected = gather_tokens(vis_tokens, topk_index)
        ir_selected = gather_tokens(ir_tokens, topk_index)
        strong_selected = interactor(vis_selected, ir_selected, intent)

        # 4) Unselected tokens 走弱交互 / 保真路径。该路径先对所有 tokens 计算，
        #    随后由 hard routing mask 保留低分 token 的 weak 输出。
        weak_full = bypass(vis_tokens, ir_tokens, intent)
        strong_full = scatter_tokens(weak_full, strong_selected, topk_index)

        # Forward 等价于 hard mask：selected=strong，unselected=weak；
        # Backward 通过 soft mask 让 Top-K 边界附近的 score 获得梯度。
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

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor):
        spatial_size = vis_feat.shape[-2:]
        text_intent, prompt_weight = self.intent_router(vis_feat, ir_feat, self.prompt_bank())
        frequency_intent = self.frequency_intent_head(text_intent)
        spatial_intent = self.spatial_intent_head(text_intent)
        vis_amp, vis_phase = split_amplitude_phase(vis_feat)
        ir_amp, ir_phase = split_amplitude_phase(ir_feat)

        fused_amp, amp_aux = self._fuse_branch(
            vis_amp, ir_amp, frequency_intent, self.amp_score, self.amp_interaction,
            self.amp_bypass, self.amp_topk_ratio, branch_type='amp'
        )
        fused_phase, phase_aux = self._fuse_branch(
            vis_phase, ir_phase, frequency_intent, self.phase_score, self.phase_interaction,
            self.phase_bypass, self.phase_topk_ratio, branch_type='phase'
        )
        fused_phase = phase_wrap(fused_phase)
        fused_spatial = rebuild_from_amplitude_phase(fused_amp, fused_phase, spatial_size)
        fused_feature = self.refine(fused_spatial)

        gamma, beta = self.refine_affine(frequency_intent).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fused_feature = fused_feature * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        fused_feature = fused_feature + 0.5 * (vis_feat + ir_feat)

        if not self.return_aux:
            return fused_feature
        aux = {
            'intent': text_intent,
            'frequency_intent': frequency_intent,
            'spatial_intent': spatial_intent,
            'prompt_weight': prompt_weight,
            'amp_score': amp_aux['score'],
            'phase_score': phase_aux['score'],
            'amp_score_target': amp_aux['score_target'],
            'phase_score_target': phase_aux['score_target'],
            'amp_mask': amp_aux['mask'],
            'phase_mask': phase_aux['mask'],
            'amp_routing_mask': amp_aux['routing_mask'],
            'phase_routing_mask': phase_aux['routing_mask'],
            'amp_topk_index': amp_aux['topk_index'],
            'phase_topk_index': phase_aux['topk_index'],
        }
        return fused_feature, aux
