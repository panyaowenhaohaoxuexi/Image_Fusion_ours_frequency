# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .prompt import FixedPromptBank, CLIPTextPromptBank, IntentRouter
from .fft_utils import split_amplitude_phase, phase_wrap, rebuild_from_amplitude_phase, patchify_feature_map, unpatchify_feature_map
from .scoring import TokenScoreNet
from .selection import topk_token_selection, gather_tokens, scatter_tokens
from .interaction import SelectedTokenInteraction
from .bypass import LightweightTokenPreserver


class HighLevelGuidedFrequencyFusion(nn.Module):
    """高层先验引导的频率 token 选择融合模块。"""

    def __init__(self, in_channels: int = 64, patch_size: int = 4, prior_dim: int = 64,
                 amp_topk_ratio: float = 0.25, phase_topk_ratio: float = 0.25,
                 token_embed_dim: int = 128, num_heads: int = 4, return_aux: bool = False,
                 use_real_clip_prompt_bank: bool = False, clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None, clip_download_root: str = None, clip_device: str = None):
        super().__init__()
        self.patch_size = patch_size
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        self.return_aux = return_aux
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
            num_prompts = 4

        self.intent_router = IntentRouter(in_channels=in_channels, prior_dim=prior_dim, num_prompts=num_prompts)

        self.amp_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.phase_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.amp_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
        self.phase_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
        self.amp_bypass = LightweightTokenPreserver(token_dim=token_dim)
        self.phase_bypass = LightweightTokenPreserver(token_dim=token_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
        self.refine_affine = nn.Sequential(
            nn.Linear(prior_dim, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels * 2)
        )

    def _fuse_branch(self, vis_map, ir_map, intent, scorer, interactor, bypass, keep_ratio):
        vis_tokens, meta = patchify_feature_map(vis_map, self.patch_size)
        ir_tokens, _ = patchify_feature_map(ir_map, self.patch_size)
        score = scorer(vis_tokens, ir_tokens, meta['coords'], intent)
        topk_index, mask, topk_value = topk_token_selection(score, keep_ratio)
        vis_selected = gather_tokens(vis_tokens, topk_index)
        ir_selected = gather_tokens(ir_tokens, topk_index)
        fused_selected = interactor(vis_selected, ir_selected, intent)
        fused_full = bypass(vis_tokens, ir_tokens)
        fused_full = scatter_tokens(fused_full, fused_selected, topk_index)
        fused_map = unpatchify_feature_map(fused_full, meta)
        aux = {'score': score, 'mask': mask.float(), 'topk_index': topk_index, 'topk_value': topk_value}
        return fused_map, aux

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor):
        spatial_size = vis_feat.shape[-2:]
        intent, prompt_weight = self.intent_router(vis_feat, ir_feat, self.prompt_bank())
        vis_amp, vis_phase = split_amplitude_phase(vis_feat)
        ir_amp, ir_phase = split_amplitude_phase(ir_feat)
        fused_amp, amp_aux = self._fuse_branch(vis_amp, ir_amp, intent, self.amp_score, self.amp_interaction, self.amp_bypass, self.amp_topk_ratio)
        fused_phase, phase_aux = self._fuse_branch(vis_phase, ir_phase, intent, self.phase_score, self.phase_interaction, self.phase_bypass, self.phase_topk_ratio)
        fused_phase = phase_wrap(fused_phase)
        fused_spatial = rebuild_from_amplitude_phase(fused_amp, fused_phase, spatial_size)
        fused_feature = self.refine(fused_spatial)
        gamma, beta = self.refine_affine(intent).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fused_feature = fused_feature * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        fused_feature = fused_feature + 0.5 * (vis_feat + ir_feat)
        if not self.return_aux:
            return fused_feature
        aux = {
            'intent': intent,
            'prompt_weight': prompt_weight,
            'amp_score': amp_aux['score'],
            'phase_score': phase_aux['score'],
            'amp_mask': amp_aux['mask'],
            'phase_mask': phase_aux['mask'],
            'amp_topk_index': amp_aux['topk_index'],
            'phase_topk_index': phase_aux['topk_index'],
        }
        return fused_feature, aux
