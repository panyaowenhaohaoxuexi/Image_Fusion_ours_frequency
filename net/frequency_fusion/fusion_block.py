# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .prompt import FixedPromptBank, IntentRouter
from .fft_utils import split_amplitude_phase, phase_wrap, rebuild_from_amplitude_phase, patchify_feature_map, unpatchify_feature_map
from .scoring import TokenScoreNet
from .selection import topk_token_selection, gather_tokens, scatter_tokens
from .interaction import SelectedTokenInteraction

class HighLevelGuidedFrequencyFusion(nn.Module):
    """高层先验引导的频率 token 选择融合模块。"""
    def __init__(self, in_channels: int = 64, patch_size: int = 4, prior_dim: int = 64,
                 amp_topk_ratio: float = 0.25, phase_topk_ratio: float = 0.25,
                 token_embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        token_dim = in_channels * patch_size * patch_size
        self.prompt_bank = FixedPromptBank(prior_dim=prior_dim)
        self.intent_router = IntentRouter(in_channels=in_channels, prior_dim=prior_dim, num_prompts=4)
        self.amp_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.amp_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads)
        self.phase_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.phase_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads)
        self.refine = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, 3, 1, 1))
    def _fuse_branch(self, vis_map, ir_map, intent, scorer, interactor, keep_ratio):
        vis_tokens, meta = patchify_feature_map(vis_map, self.patch_size)
        ir_tokens, _ = patchify_feature_map(ir_map, self.patch_size)
        coords = meta['coords']
        score = scorer(vis_tokens, ir_tokens, coords, intent)
        topk_index, _, _ = topk_token_selection(score, keep_ratio)
        vis_selected = gather_tokens(vis_tokens, topk_index)
        ir_selected = gather_tokens(ir_tokens, topk_index)
        fused_selected = interactor(vis_selected, ir_selected)
        fused_full = 0.5 * (vis_tokens + ir_tokens)
        fused_full = scatter_tokens(fused_full, fused_selected, topk_index)
        return unpatchify_feature_map(fused_full, meta)
    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor):
        spatial_size = vis_feat.shape[-2:]
        prompt_bank = self.prompt_bank()
        intent, _ = self.intent_router(vis_feat, ir_feat, prompt_bank)
        vis_amp, vis_phase = split_amplitude_phase(vis_feat)
        ir_amp, ir_phase = split_amplitude_phase(ir_feat)
        fused_amp = self._fuse_branch(vis_amp, ir_amp, intent, self.amp_score, self.amp_interaction, self.amp_topk_ratio)
        fused_phase = self._fuse_branch(vis_phase, ir_phase, intent, self.phase_score, self.phase_interaction, self.phase_topk_ratio)
        fused_phase = phase_wrap(fused_phase)
        fused_spatial = rebuild_from_amplitude_phase(fused_amp, fused_phase, spatial_size)
        return self.refine(fused_spatial) + 0.5 * (vis_feat + ir_feat)
