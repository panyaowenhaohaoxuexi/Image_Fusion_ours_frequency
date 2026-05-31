# -*- coding: utf-8 -*-
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip  # type: ignore
except Exception:
    clip = None

TensorOrPyramid = Union[torch.Tensor, Sequence[torch.Tensor]]


DEGRADATION_PROMPTS = [
    "handle nighttime low light",
    "handle visible blur",
    "handle exposure anomaly",
    "handle infrared noise",
    "handle infrared low contrast",
]

FUSION_PROMPTS = [
    "preserve edge structure",
    "preserve visible texture",
    "natural overall appearance",
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


class DualStreamIntentMLP(nn.Module):
    """Generate free continuous degradation/fusion intents from spatial and frequency features.

    CLIP text prompts are stored only as detached semantic anchors for the alignment loss;
    they are not used to synthesize z_deg or z_fus.
    """

    def __init__(self, channels: int = 64, intent_dim: int = 64, hidden_dim: int = 256,
                 clip_model_name: str = "ViT-B/32", clip_download_root: str = None,
                 use_clip_prompt_buffer: bool = True):
        super().__init__()
        in_dim = channels * 8  # vis/ir spatial L1-L3 plus vis/ir frequency.
        self.deg_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intent_dim),
        )
        self.fus_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intent_dim),
        )
        deg_prompt, fus_prompt = self._build_prompt_buffers(
            intent_dim, clip_model_name, clip_download_root, use_clip_prompt_buffer
        )
        self.register_buffer("degradation_prompt_bank", deg_prompt)
        self.register_buffer("fusion_prompt_bank", fus_prompt)

    @staticmethod
    def _one_hot_bank(num_prompts: int, intent_dim: int) -> torch.Tensor:
        bank = torch.zeros(num_prompts, intent_dim, dtype=torch.float32)
        for i in range(num_prompts):
            bank[i, i::num_prompts] = 1.0
        return F.normalize(bank, dim=-1)

    @classmethod
    def _build_prompt_buffers(cls, intent_dim: int, clip_model_name: str,
                              clip_download_root: str, use_clip_prompt_buffer: bool):
        if not use_clip_prompt_buffer or clip is None:
            return cls._one_hot_bank(len(DEGRADATION_PROMPTS), intent_dim), cls._one_hot_bank(len(FUSION_PROMPTS), intent_dim)

        device = "cpu"
        clip_model, _ = clip.load(clip_model_name, device=device, download_root=clip_download_root)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            prompts = DEGRADATION_PROMPTS + FUSION_PROMPTS
            tokens = clip.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
        clip_dim = text_features.shape[-1]
        # Deterministic fixed random projection avoids training the semantic anchors.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260531)
        proj = torch.randn(clip_dim, intent_dim, generator=generator) / (clip_dim ** 0.5)
        prompt_bank = F.normalize(text_features.cpu().matmul(proj), dim=-1)
        return prompt_bank[:len(DEGRADATION_PROMPTS)], prompt_bank[len(DEGRADATION_PROMPTS):]

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, vis_spa: TensorOrPyramid, ir_spa: TensorOrPyramid,
                vis_freq: torch.Tensor, ir_freq: torch.Tensor):
        vis_l1, vis_l2, vis_l3 = _as_three_levels(vis_spa)
        ir_l1, ir_l2, ir_l3 = _as_three_levels(ir_spa)
        q = torch.cat([
            self._pool(vis_l1), self._pool(ir_l1),
            self._pool(vis_l2), self._pool(ir_l2),
            self._pool(vis_l3), self._pool(ir_l3),
            self._pool(vis_freq), self._pool(ir_freq),
        ], dim=1)
        z_deg = self.deg_mlp(q)
        z_fus = self.fus_mlp(q)
        return z_deg, z_fus
