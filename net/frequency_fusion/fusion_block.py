# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_utils import split_amplitude_phase, phase_wrap, rebuild_from_amplitude_phase, patchify_feature_map, unpatchify_feature_map
from .scoring import TokenScoreNet
from .interaction import SelectedTokenInteraction
from .bypass import LightweightTokenPreserver
from .selection import topk_token_selection, straight_through_topk_mask, gather_tokens, scatter_tokens


class TGSFF(nn.Module):
    """Text-guided selective frequency fusion with straight-through Top-K routing."""

    def __init__(self, in_channels: int = 64, patch_size: int = 4, prior_dim: int = 64,
                 amp_topk_ratio: float = 0.25, phase_topk_ratio: float = 0.25,
                 token_embed_dim: int = 128, num_heads: int = 4, return_aux: bool = False,
                 use_real_clip_prompt_bank: bool = False, clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None, clip_download_root: str = None, clip_device: str = None,
                 routing_temperature: float = 0.25):
        super().__init__()
        self.patch_size = patch_size
        self.return_aux = return_aux
        self.routing_temperature = routing_temperature
        self.prior_dim = prior_dim
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        token_dim = in_channels * patch_size * patch_size

        self.amp_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.phase_score = TokenScoreNet(token_dim=token_dim, prior_dim=prior_dim, hidden_dim=token_embed_dim)
        self.amp_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
        self.phase_interaction = SelectedTokenInteraction(token_dim=token_dim, embed_dim=token_embed_dim, num_heads=num_heads, prior_dim=prior_dim)
        self.amp_bypass = LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)
        self.phase_bypass = LightweightTokenPreserver(token_dim=token_dim, prior_dim=prior_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )

    @staticmethod
    def _normalize_token_target(x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return ((x - x_min) / (x_max - x_min + 1e-6)).detach()

    def _build_score_target(self, vis_tokens: torch.Tensor, ir_tokens: torch.Tensor, branch_type: str) -> torch.Tensor:
        vis_abs = torch.abs(vis_tokens)
        ir_abs = torch.abs(ir_tokens)
        diff = torch.mean(torch.abs(vis_tokens - ir_tokens), dim=-1)
        energy = 0.5 * (torch.mean(torch.log1p(vis_abs), dim=-1) + torch.mean(torch.log1p(ir_abs), dim=-1))
        max_response = torch.max(torch.mean(vis_abs, dim=-1), torch.mean(ir_abs, dim=-1))
        local_variation = 0.5 * (torch.std(vis_tokens, dim=-1, unbiased=False) + torch.std(ir_tokens, dim=-1, unbiased=False))
        if branch_type == 'amp':
            raw_target = 0.45 * diff + 0.35 * energy + 0.20 * max_response
        elif branch_type == 'phase':
            raw_target = 0.55 * diff + 0.30 * local_variation + 0.15 * energy
        else:
            raw_target = 0.50 * diff + 0.25 * energy + 0.25 * local_variation
        return self._normalize_token_target(raw_target)

    def _fuse_branch(self, vis_map, ir_map, intent, scorer, interactor, bypass,
                     branch_type: str, keep_ratio: float):
        vis_tokens, meta = patchify_feature_map(vis_map, self.patch_size)
        ir_tokens, _ = patchify_feature_map(ir_map, self.patch_size)
        score = scorer(vis_tokens, ir_tokens, meta['coords'], intent)
        topk_index, hard_mask, topk_value = topk_token_selection(score, keep_ratio)
        routing_mask = straight_through_topk_mask(
            score, hard_mask, keep_ratio, temperature=self.routing_temperature
        ).unsqueeze(-1)

        selected_vis = gather_tokens(vis_tokens, topk_index)
        selected_ir = gather_tokens(ir_tokens, topk_index)
        selected_strong = interactor(selected_vis, selected_ir, intent)
        weak_full = bypass(vis_tokens, ir_tokens, intent)
        strong_full = scatter_tokens(weak_full, selected_strong, topk_index)
        fused_full = routing_mask * strong_full + (1.0 - routing_mask) * weak_full
        fused_map = unpatchify_feature_map(fused_full, meta)
        aux = {
            'score': score,
            'score_target': self._build_score_target(vis_tokens, ir_tokens, branch_type=branch_type),
            'mask': hard_mask,
            'routing_mask': routing_mask.squeeze(-1),
            'topk_index': topk_index,
            'topk_value': topk_value,
        }
        return fused_map, aux

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, frequency_intent: torch.Tensor = None):
        spatial_size = vis_feat.shape[-2:]
        if frequency_intent is None:
            frequency_intent = torch.zeros(vis_feat.shape[0], self.prior_dim, device=vis_feat.device, dtype=vis_feat.dtype)
        vis_amp, vis_phase = split_amplitude_phase(vis_feat)
        ir_amp, ir_phase = split_amplitude_phase(ir_feat)
        fused_amp, amp_aux = self._fuse_branch(
            vis_amp, ir_amp, frequency_intent, self.amp_score, self.amp_interaction,
            self.amp_bypass, 'amp', self.amp_topk_ratio
        )
        fused_phase, phase_aux = self._fuse_branch(
            vis_phase, ir_phase, frequency_intent, self.phase_score, self.phase_interaction,
            self.phase_bypass, 'phase', self.phase_topk_ratio
        )
        fused_phase = phase_wrap(fused_phase)
        fused_spatial = rebuild_from_amplitude_phase(fused_amp, fused_phase, spatial_size)
        fused_feature = self.refine(fused_spatial)

        if not self.return_aux:
            return fused_feature
        aux = {
            'frequency_intent': frequency_intent,
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
            'amp_topk_value': amp_aux['topk_value'],
            'phase_topk_value': phase_aux['topk_value'],
        }
        return fused_feature, aux


class HighLevelGuidedFrequencyFusion(TGSFF):
    """Backward-compatible name for TGSFF."""
    pass
