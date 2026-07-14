# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


DEGRADATION_PROMPT_GROUPS = {
    "low_light": [
        "a low-light visible image",
        "a dim nighttime scene with poor visibility",
        "an underexposed scene with insufficient brightness",
    ],
    "blur": [
        "a blurry visible image",
        "a scene affected by motion blur",
        "an out-of-focus scene with unclear details",
    ],
    "exposure": [
        "an overexposed visible image",
        "a scene with clipped bright highlights",
        "a washed-out scene with excessive brightness",
    ],
    "structure": [
        "an image with degraded structural contours",
        "a scene with damaged or unclear boundaries",
        "an image with corrupted geometric structures",
    ],
}

FUSION_PROMPT_GROUPS = {
    "texture": [
        "an image containing rich visible textures",
        "a scene with detailed fine-scale textures",
        "an image with abundant high-frequency texture details",
    ],
    "edge": [
        "an image containing clear edge structures",
        "a scene with sharp object boundaries",
        "an image with distinct structural contours",
    ],
    "natural": [
        "an image with a natural scene appearance",
        "a scene with natural luminance and tone",
        "an image with a visually natural intensity distribution",
    ],
    "contrast": [
        "an image with clear local contrast",
        "a scene with distinct regional intensity differences",
        "an image with fine-grained contrast details",
    ],
    # TODO: add salient_ir_target through an independent non-CLIP statistics branch.
}


class CLIPTextPromptBank(nn.Module):
    """Frozen category representatives in CLIP's native text embedding space."""

    def __init__(self, clip_model, prompt_groups: dict):
        super().__init__()
        clip_model.eval()
        model_device = next(clip_model.parameters()).device
        self.prompt_group_names = list(prompt_groups.keys())

        group_vectors = []
        with torch.no_grad():
            for group_name in self.prompt_group_names:
                tokens = clip.tokenize(prompt_groups[group_name]).to(model_device)
                features = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
                group_vectors.append(F.normalize(features.mean(dim=0), dim=0))
        self.register_buffer("prompt_bank", torch.stack(group_vectors, dim=0))

    def forward(self) -> torch.Tensor:
        return self.prompt_bank


class CLIPImageQuery(nn.Module):
    """The sole module that retains the frozen CLIP model."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(False)
        self.clip_model.eval()
        return self

    def forward(self, vis_rgb_clip_ready: torch.Tensor) -> torch.Tensor:
        parameter = next(self.clip_model.parameters())
        image = vis_rgb_clip_ready.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            features = self.clip_model.encode_image(image).float()
        return F.normalize(features, dim=-1)
