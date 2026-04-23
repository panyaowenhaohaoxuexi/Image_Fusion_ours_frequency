# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip  # type: ignore
except Exception:
    clip = None


class FixedPromptBank(nn.Module):
    """固定高层先验库。"""

    def __init__(self, prior_dim: int = 64):
        super().__init__()
        self.prompt_names = [
            'salient_targets',
            'structural_contours',
            'fine_textures',
            'balanced_fusion',
        ]
        bank = torch.zeros(len(self.prompt_names), prior_dim, dtype=torch.float32)
        for i in range(len(self.prompt_names)):
            bank[i, i::len(self.prompt_names)] = 1.0
        bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer('prompt_bank', bank)

    def forward(self) -> torch.Tensor:
        return self.prompt_bank


class CLIPTextPromptBank(nn.Module):
    """冻结 CLIP text encoder，将固定 prompt 编码成可训练投影后的语义先验。"""

    def __init__(self,
                 prior_dim: int = 64,
                 clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None,
                 download_root: str = None,
                 clip_device: str = None):
        super().__init__()
        if clip is None:
            raise ImportError(
                '未检测到 clip 库。请先安装 openai-clip，例如: pip install openai-clip==1.0.1'
            )

        if prompt_texts is None:
            prompt_texts = [
                'salient targets',
                'structural contours',
                'fine textures',
                'balanced fusion',
            ]
        self.prompt_texts = list(prompt_texts)
        self.prompt_names = list(prompt_texts)

        if clip_device is None:
            clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        clip_model, _ = clip.load(clip_model_name, device=clip_device, download_root=download_root)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model
        self.clip_device = clip_device

        with torch.no_grad():
            text_tokens = clip.tokenize(self.prompt_texts).to(clip_device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = F.normalize(text_features, dim=-1)

        self.register_buffer('clip_text_features', text_features.detach().cpu())
        clip_dim = text_features.shape[-1]
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, prior_dim),
            nn.LayerNorm(prior_dim),
        )

    def forward(self) -> torch.Tensor:
        clip_text_features = self.clip_text_features.to(self.proj[0].weight.device)
        prompt_bank = self.proj(clip_text_features)
        prompt_bank = F.normalize(prompt_bank, dim=-1)
        return prompt_bank


class IntentRouter(nn.Module):
    def __init__(self, in_channels: int, prior_dim: int, num_prompts: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_prompts),
        )
        self.proj = nn.Linear(prior_dim, prior_dim)

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor, prompt_bank: torch.Tensor):
        b, c, _, _ = vis_feat.shape
        vis_vec = self.pool(vis_feat).view(b, c)
        ir_vec = self.pool(ir_feat).view(b, c)
        fusion_vec = torch.cat([vis_vec, ir_vec], dim=1)
        prompt_logits = self.mlp(fusion_vec)
        prompt_weight = torch.softmax(prompt_logits, dim=1)
        prompt_bank = prompt_bank.unsqueeze(0).repeat(b, 1, 1)
        intent = torch.sum(prompt_weight.unsqueeze(-1) * prompt_bank, dim=1)
        intent = self.proj(intent)
        return intent, prompt_weight
