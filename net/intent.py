# -*- coding: utf-8 -*-
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.frequency_fusion.prompt import CLIPTextPromptBank, LearnablePromptBank

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


DEGRADATION_PROMPTS = [
    "low light enhancement",
    "visible blur suppression",
    "exposure anomaly correction",
    "infrared noise suppression",
    "infrared low contrast enhancement",
    "structural contours restoration",
]

FUSION_PROMPTS = [
    "salient infrared targets",
    "visible fine textures",
    "edge structure preservation",
    "natural scene appearance",
    "balanced infrared visible fusion",
    "local contrast preservation",
]


def _as_three_levels(x: TensorOrPyramid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(x, (list, tuple)):
        if len(x) < 3:
            raise ValueError("spatial feature pyramid must contain [L1, L2, L3].")
        return x[0], x[1], x[2]
    l1 = x
    l2 = F.avg_pool2d(l1, kernel_size=2, stride=2)
    l3 = F.avg_pool2d(l2, kernel_size=2, stride=2)
    return l1, l2, l3


class DualDomainTextIntentGenerator(nn.Module):
    """Generate dual intents by image-driven weighting over two text prompt banks.

    Image features only produce prompt logits. The semantic content of I_deg and
    I_fus is exactly the convex combination of their respective prompt banks.
    """

    def __init__(self, channels: int = 64, intent_dim: int = 64, hidden_dim: int = 256,
                 clip_model_name: str = "ViT-B/32", clip_download_root: str = None,
                 clip_device: str = None, use_clip_prompt_bank: bool = True,
                 use_learnable_prompt_embedding: bool = False):
        super().__init__()
        self.intent_dim = intent_dim
        self.use_learnable_prompt_embedding = use_learnable_prompt_embedding
        if use_learnable_prompt_embedding:
            self.deg_prompt_bank = LearnablePromptBank(len(DEGRADATION_PROMPTS), intent_dim)
            self.fus_prompt_bank = LearnablePromptBank(len(FUSION_PROMPTS), intent_dim)
        else:
            self.deg_prompt_bank = CLIPTextPromptBank(
                prior_dim=intent_dim,
                clip_model_name=clip_model_name,
                prompt_texts=DEGRADATION_PROMPTS,
                download_root=clip_download_root,
                clip_device=clip_device,
                allow_deterministic_fallback=not use_clip_prompt_bank,
            )
            self.fus_prompt_bank = CLIPTextPromptBank(
                prior_dim=intent_dim,
                clip_model_name=clip_model_name,
                prompt_texts=FUSION_PROMPTS,
                download_root=clip_download_root,
                clip_device=clip_device,
                allow_deterministic_fallback=not use_clip_prompt_bank,
            )

        query_dim = channels * 8
        self.deg_weighting_head = self._make_weighting_head(query_dim, hidden_dim, len(DEGRADATION_PROMPTS))
        self.fus_weighting_head = self._make_weighting_head(query_dim, hidden_dim, len(FUSION_PROMPTS))

    @staticmethod
    def _make_weighting_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _build_image_query(self, vis_spa: TensorOrPyramid, ir_spa: TensorOrPyramid,
                           vis_freq: torch.Tensor, ir_freq: torch.Tensor) -> torch.Tensor:
        vis_l1, vis_l2, vis_l3 = _as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = _as_three_levels(ir_spa)
        return torch.cat([
            self._pool(vis_l1), self._pool(ir_l1),
            self._pool(vis_l2), self._pool(ir_l2),
            self._pool(vis_l3), self._pool(ir_l3),
            self._pool(vis_freq), self._pool(ir_freq),
        ], dim=1)

    def forward(self, vis_spa: TensorOrPyramid, ir_spa: TensorOrPyramid,
                vis_freq: torch.Tensor, ir_freq: torch.Tensor):
        image_query = self._build_image_query(vis_spa, ir_spa, vis_freq, ir_freq)
        deg_logits = self.deg_weighting_head(image_query)
        fus_logits = self.fus_weighting_head(image_query)
        deg_prompt_weight = torch.softmax(deg_logits, dim=-1)
        fus_prompt_weight = torch.softmax(fus_logits, dim=-1)
        deg_bank = self.deg_prompt_bank().to(device=image_query.device, dtype=image_query.dtype)
        fus_bank = self.fus_prompt_bank().to(device=image_query.device, dtype=image_query.dtype)
        I_deg = deg_prompt_weight.matmul(deg_bank)
        I_fus = fus_prompt_weight.matmul(fus_bank)
        aux = {
            "deg_prompt_weight": deg_prompt_weight,
            "fus_prompt_weight": fus_prompt_weight,
            "deg_prompt_bank": deg_bank,
            "fus_prompt_bank": fus_bank,
            "deg_prompt_texts": DEGRADATION_PROMPTS,
            "fus_prompt_texts": FUSION_PROMPTS,
            "deg_prompt_logits": deg_logits,
            "fus_prompt_logits": fus_logits,
        }
        return I_deg, I_fus, aux


class DualStreamIntentMLP(DualDomainTextIntentGenerator):
    """Historical compatibility alias.

    New training and inference code must instantiate DualDomainTextIntentGenerator
    directly. This alias preserves old imports without keeping the abandoned image
    MLP intent route alive.
    """
    pass
