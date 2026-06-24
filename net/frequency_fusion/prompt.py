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
            'low_light_enhancement',
        ]
        bank = torch.zeros(len(self.prompt_names), prior_dim, dtype=torch.float32)
        for i in range(len(self.prompt_names)):
            bank[i, i::len(self.prompt_names)] = 1.0
        bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer('prompt_bank', bank)

    def forward(self) -> torch.Tensor:
        return self.prompt_bank


class LearnablePromptBank(nn.Module):
    """Ablation-only prompt bank with learnable vectors."""

    def __init__(self, num_prompts: int, prior_dim: int = 64):
        super().__init__()
        prompt_bank = torch.randn(num_prompts, prior_dim, dtype=torch.float32) * (prior_dim ** -0.5)
        self.prompt_bank = nn.Parameter(F.normalize(prompt_bank, dim=-1))

    def forward(self) -> torch.Tensor:
        return F.normalize(self.prompt_bank, dim=-1)


class CLIPTextPromptBank(nn.Module):
    """冻结 CLIP text encoder，将固定 prompt 编码成可训练投影后的语义先验。"""

    def __init__(self,
                 prior_dim: int = 64,
                 clip_model_name: str = 'ViT-B/32',
                 prompt_texts=None,
                 download_root: str = None,
                 clip_device: str = None,
                 allow_deterministic_fallback: bool = False):
        super().__init__()
        if clip is None and not allow_deterministic_fallback:
            raise ImportError(
                '未检测到 clip 库。请先安装 openai-clip，例如: pip install openai-clip==1.0.1'
            )

        if prompt_texts is None:
            prompt_texts = [
                'salient targets',
                'structural contours',
                'fine textures',
                'balanced fusion',
                'low light enhancement',
            ]
        self.prompt_texts = list(prompt_texts)
        self.prompt_names = list(prompt_texts)

        if clip is None or allow_deterministic_fallback:
            bank = self._deterministic_bank(len(self.prompt_texts), prior_dim)
            self.register_buffer('fallback_prompt_bank', bank)
            self.proj = None
            return

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

    @staticmethod
    def _deterministic_bank(num_prompts: int, prior_dim: int) -> torch.Tensor:
        bank = torch.zeros(num_prompts, prior_dim, dtype=torch.float32)
        for i in range(num_prompts):
            bank[i, i::num_prompts] = 1.0
        return F.normalize(bank, dim=-1)

    def forward(self) -> torch.Tensor:
        if self.proj is None:
            return self.fallback_prompt_bank
        clip_text_features = self.clip_text_features.to(self.proj[0].weight.device)
        prompt_bank = self.proj(clip_text_features)
        prompt_bank = F.normalize(prompt_bank, dim=-1)
        return prompt_bank

