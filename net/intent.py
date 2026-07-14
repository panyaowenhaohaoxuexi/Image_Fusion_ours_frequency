# -*- coding: utf-8 -*-
import math
from typing import Sequence, Tuple, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.frequency_fusion.prompt import (
    CLIPImageQuery,
    CLIPTextPromptBank,
    DEGRADATION_PROMPT_GROUPS,
    FUSION_PROMPT_GROUPS,
)

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


def _as_three_levels(x: TensorOrPyramid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(x, (list, tuple)):
        if len(x) < 3:
            raise ValueError("spatial feature pyramid must contain [L1, L2, L3].")
        return x[0], x[1], x[2]
    level_one = x
    level_two = F.avg_pool2d(level_one, kernel_size=2, stride=2)
    level_three = F.avg_pool2d(level_two, kernel_size=2, stride=2)
    return level_one, level_two, level_three


class DualDomainTextIntentGenerator(nn.Module):
    """Generate image-level dual intents through frozen CLIP image-text similarity."""

    def __init__(
        self,
        intent_dim: int = 64,
        clip_model_name: str = "ViT-B/32",
        clip_download_root: str = None,
        clip_device: str = None,
        use_learnable_prompt_embedding: bool = False,
    ):
        super().__init__()
        if use_learnable_prompt_embedding:
            raise ValueError(
                "DualDomainTextIntentGenerator requires the frozen CLIP text bank; "
                "LearnablePromptBank is not compatible with CLIP image-text similarity. "
                "Use DualStreamIntentMLP for the non-CLIP-query ablation instead."
            )
        if clip_device is None:
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, _ = clip.load(clip_model_name, device=clip_device, download_root=clip_download_root)
        self.clip_image_query = CLIPImageQuery(clip_model)
        self.deg_prompt_bank = CLIPTextPromptBank(clip_model, DEGRADATION_PROMPT_GROUPS)
        self.fus_prompt_bank = CLIPTextPromptBank(clip_model, FUSION_PROMPT_GROUPS)
        self.logit_scale_deg = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.logit_scale_fus = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        clip_dim = self.deg_prompt_bank().shape[-1]
        self.deg_proj = nn.Linear(clip_dim, intent_dim)
        self.fus_proj = nn.Linear(clip_dim, intent_dim)

    def forward(self, vis_rgb, vis_spa=None, ir_spa=None, vis_freq=None, ir_freq=None):
        del vis_spa, ir_spa, vis_freq, ir_freq
        visible_embedding = self.clip_image_query(vis_rgb)
        deg_bank = self.deg_prompt_bank().to(device=visible_embedding.device, dtype=visible_embedding.dtype)
        fus_bank = self.fus_prompt_bank().to(device=visible_embedding.device, dtype=visible_embedding.dtype)

        logit_scale_deg = self.logit_scale_deg.exp().clamp(max=100.0)
        logit_scale_fus = self.logit_scale_fus.exp().clamp(max=100.0)
        deg_weight = torch.softmax(logit_scale_deg * (visible_embedding @ deg_bank.t()), dim=-1)
        fus_weight = torch.softmax(logit_scale_fus * (visible_embedding @ fus_bank.t()), dim=-1)

        int_deg_raw = F.normalize(deg_weight @ deg_bank, dim=-1)
        int_fus_raw = F.normalize(fus_weight @ fus_bank, dim=-1)
        return self.deg_proj(int_deg_raw), self.fus_proj(int_fus_raw), {
            "deg_prompt_weight": deg_weight.detach(),
            "fus_prompt_weight": fus_weight.detach(),
            "logit_scale_deg": logit_scale_deg.detach().reshape(1),
            "logit_scale_fus": logit_scale_fus.detach().reshape(1),
        }


class DualStreamIntentMLP(nn.Module):
    """SharedEncoder-MLP Query ablation with the v11 CLIP text category bank."""

    def __init__(
        self,
        channels: int = 64,
        intent_dim: int = 64,
        hidden_dim: int = 256,
        clip_model_name: str = "ViT-B/32",
        clip_download_root: str = None,
        clip_device: str = None,
    ):
        super().__init__()
        if clip_device is None:
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(clip_model_name, device=clip_device, download_root=clip_download_root)
        self.deg_prompt_bank = CLIPTextPromptBank(clip_model, DEGRADATION_PROMPT_GROUPS)
        self.fus_prompt_bank = CLIPTextPromptBank(clip_model, FUSION_PROMPT_GROUPS)
        clip_dim = self.deg_prompt_bank().shape[-1]
        self.deg_proj = nn.Linear(clip_dim, intent_dim)
        self.fus_proj = nn.Linear(clip_dim, intent_dim)

        query_dim = channels * 8
        self.deg_weighting_head = self._make_weighting_head(query_dim, hidden_dim, len(DEGRADATION_PROMPT_GROUPS))
        self.fus_weighting_head = self._make_weighting_head(query_dim, hidden_dim, len(FUSION_PROMPT_GROUPS))

    @staticmethod
    def _make_weighting_head(in_dim, hidden_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    @staticmethod
    def _pool(x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _build_image_query(self, vis_spa, ir_spa, vis_freq, ir_freq):
        vis_l1, vis_l2, vis_l3 = _as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = _as_three_levels(ir_spa)
        return torch.cat([
            self._pool(vis_l1), self._pool(ir_l1), self._pool(vis_l2), self._pool(ir_l2),
            self._pool(vis_l3), self._pool(ir_l3), self._pool(vis_freq), self._pool(ir_freq),
        ], dim=1)

    def forward(self, vis_rgb, vis_spa, ir_spa, vis_freq, ir_freq):
        del vis_rgb
        image_query = self._build_image_query(vis_spa, ir_spa, vis_freq, ir_freq)
        deg_weight = torch.softmax(self.deg_weighting_head(image_query), dim=-1)
        fus_weight = torch.softmax(self.fus_weighting_head(image_query), dim=-1)
        deg_bank = self.deg_prompt_bank().to(device=image_query.device, dtype=image_query.dtype)
        fus_bank = self.fus_prompt_bank().to(device=image_query.device, dtype=image_query.dtype)
        int_deg_raw = F.normalize(deg_weight @ deg_bank, dim=-1)
        int_fus_raw = F.normalize(fus_weight @ fus_bank, dim=-1)
        return self.deg_proj(int_deg_raw), self.fus_proj(int_fus_raw), {
            "deg_prompt_weight": deg_weight.detach(),
            "fus_prompt_weight": fus_weight.detach(),
        }
