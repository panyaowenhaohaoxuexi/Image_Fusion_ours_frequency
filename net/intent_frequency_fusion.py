import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPromptBank(nn.Module):
    """Fixed prompt bank used as a lightweight stand-in for offline CLIP text embeddings.

    Recommended usage:
    1. Offline encode prompts such as
       ['salient targets', 'structural contours', 'fine textures', 'balanced fusion']
       with a frozen CLIP text encoder.
    2. Save them as a tensor of shape [K, D].
    3. Pass the saved path through ``prompt_embedding_path``.

    If no path is provided, a deterministic orthogonal bank is created so that the
    rest of the architecture can be trained and debugged without external VLM dependencies.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        prompt_embedding_path: Optional[str] = None,
        prompt_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        if prompt_names is None:
            prompt_names = (
                "salient targets",
                "structural contours",
                "fine textures",
                "balanced fusion",
            )
        self.prompt_names = prompt_names

        if prompt_embedding_path is not None:
            bank = torch.load(prompt_embedding_path, map_location="cpu")
            if isinstance(bank, dict):
                if "prompt_bank" not in bank:
                    raise KeyError("prompt_embedding_path dict must contain key 'prompt_bank'.")
                bank = bank["prompt_bank"]
            bank = torch.as_tensor(bank, dtype=torch.float32)
        else:
            num_prompts = len(prompt_names)
            eye = torch.eye(num_prompts, dtype=torch.float32)
            if embed_dim >= num_prompts:
                pad = torch.zeros(num_prompts, embed_dim - num_prompts, dtype=torch.float32)
                bank = torch.cat([eye, pad], dim=1)
            else:
                bank = eye[:, :embed_dim]
            bank = F.normalize(bank, dim=-1)

        if bank.dim() != 2:
            raise ValueError("Prompt bank must be a 2D tensor with shape [K, D].")
        self.register_buffer("prompt_bank", F.normalize(bank, dim=-1), persistent=True)
        self.embed_dim = int(bank.shape[1])
        self.num_prompts = int(bank.shape[0])

    def forward(self) -> torch.Tensor:
        return self.prompt_bank


class IntentRouter(nn.Module):
    def __init__(self, in_channels: int, num_prompts: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_prompts, kernel_size=1, bias=True),
        )

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        logits = self.router(torch.cat([vis_feat, ir_feat], dim=1)).flatten(1)
        return torch.softmax(logits, dim=-1)


class ScoreMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SimpleCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self._reshape(self.q_proj(self.norm1(query)))
        k = self._reshape(self.k_proj(context))
        v = self._reshape(self.v_proj(context))
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(query.shape)
        out = query + self.out_proj(out)
        out = out + self.mlp(self.norm2(out))
        return out


class IntentGuidedFrequencyTokenFusion(nn.Module):
    """Intent-guided Top-K frequency token fusion.

    This module keeps TSFI-Fusion's spatial/common branch intact and introduces an
    explicit phase-amplitude token selection pipeline on top of encoder features.
    The module is intentionally self-contained so it can be plugged into the current
    training loop with minimal changes.
    """

    def __init__(
        self,
        in_channels: int = 64,
        intent_dim: int = 64,
        patch_size: int = 4,
        amp_topk_ratio: float = 0.25,
        phase_topk_ratio: float = 0.25,
        num_heads: int = 8,
        prompt_embedding_path: Optional[str] = None,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.amp_topk_ratio = amp_topk_ratio
        self.phase_topk_ratio = phase_topk_ratio
        self.return_aux = return_aux

        self.prompt_bank = FixedPromptBank(intent_dim, prompt_embedding_path=prompt_embedding_path)
        self.intent_router = IntentRouter(in_channels=in_channels, num_prompts=self.prompt_bank.num_prompts)
        self.intent_to_phase = nn.Linear(self.prompt_bank.embed_dim, intent_dim)
        self.intent_to_amp = nn.Linear(self.prompt_bank.embed_dim, intent_dim)

        amp_in_dim = in_channels * 4 + 2 + intent_dim
        phase_in_dim = in_channels * 8 + 2 + intent_dim
        self.amp_scorer = ScoreMLP(amp_in_dim, hidden_dim=max(128, in_channels * 2))
        self.phase_scorer = ScoreMLP(phase_in_dim, hidden_dim=max(128, in_channels * 2))

        self.amp_cross = SimpleCrossAttention(in_channels, num_heads=num_heads)
        self.phase_cross = SimpleCrossAttention(in_channels * 2, num_heads=num_heads)

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
        )

    @staticmethod
    def _pool_tokens(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)

    @staticmethod
    def _tokens(x: torch.Tensor):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), h, w

    @staticmethod
    def _grid_position(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=height, device=device),
            torch.linspace(-1.0, 1.0, steps=width, device=device),
            indexing='ij',
        )
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        pos = torch.stack([rr, torch.atan2(yy, xx) / math.pi], dim=-1)
        pos = pos.view(1, height * width, 2).repeat(batch, 1, 1)
        return pos

    @staticmethod
    def _topk_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
        b, n = scores.shape
        k = max(1, int(round(n * ratio)))
        _, idx = torch.topk(scores, k=k, dim=1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        k = int(mask.sum(dim=1)[0].item())
        gather_idx = mask.nonzero(as_tuple=False)[:, 1].view(b, k)
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, c)
        return torch.gather(tokens, 1, gather_idx)

    @staticmethod
    def _scatter_tokens(fused_sel: torch.Tensor, fallback: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = fallback.clone()
        b, n, c = fallback.shape
        k = fused_sel.shape[1]
        scatter_idx = mask.nonzero(as_tuple=False)[:, 1].view(b, k)
        scatter_idx = scatter_idx.unsqueeze(-1).expand(-1, -1, c)
        out.scatter_(1, scatter_idx, fused_sel)
        return out

    def _score_amplitude(self, vis_amp: torch.Tensor, ir_amp: torch.Tensor, intent_amp: torch.Tensor):
        vis_pool = self._pool_tokens(vis_amp, self.patch_size)
        ir_pool = self._pool_tokens(ir_amp, self.patch_size)
        vis_tok, h_t, w_t = self._tokens(vis_pool)
        ir_tok, _, _ = self._tokens(ir_pool)
        pos = self._grid_position(vis_tok.shape[0], h_t, w_t, vis_tok.device)
        intent = intent_amp.unsqueeze(1).repeat(1, vis_tok.shape[1], 1)
        feat = torch.cat([
            vis_tok,
            ir_tok,
            torch.abs(vis_tok - ir_tok),
            vis_tok * ir_tok,
            pos,
            intent,
        ], dim=-1)
        scores = self.amp_scorer(feat)
        return scores, vis_tok, ir_tok, h_t, w_t

    def _score_phase(self, vis_phase_cs: torch.Tensor, ir_phase_cs: torch.Tensor, intent_phase: torch.Tensor):
        vis_pool = self._pool_tokens(vis_phase_cs, self.patch_size)
        ir_pool = self._pool_tokens(ir_phase_cs, self.patch_size)
        vis_tok, h_t, w_t = self._tokens(vis_pool)
        ir_tok, _, _ = self._tokens(ir_pool)
        pos = self._grid_position(vis_tok.shape[0], h_t, w_t, vis_tok.device)
        intent = intent_phase.unsqueeze(1).repeat(1, vis_tok.shape[1], 1)
        feat = torch.cat([
            vis_tok,
            ir_tok,
            torch.abs(vis_tok - ir_tok),
            vis_tok * ir_tok,
            pos,
            intent,
        ], dim=-1)
        scores = self.phase_scorer(feat)
        return scores, vis_tok, ir_tok, h_t, w_t

    def _fuse_tokens(self, vis_tok: torch.Tensor, ir_tok: torch.Tensor, scores: torch.Tensor, ratio: float, cross_attn: nn.Module):
        mask = self._topk_mask(scores, ratio)
        vis_sel = self._gather_tokens(vis_tok, mask)
        ir_sel = self._gather_tokens(ir_tok, mask)
        vis_upd = cross_attn(vis_sel, ir_sel)
        ir_upd = cross_attn(ir_sel, vis_sel)
        fused_sel = 0.5 * (vis_upd + ir_upd)
        fallback = 0.5 * (vis_tok + ir_tok)
        fused = self._scatter_tokens(fused_sel, fallback, mask)
        return fused, mask.float()

    def forward(self, vis_feat: torch.Tensor, ir_feat: torch.Tensor):
        b, c, h, w = vis_feat.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Feature size {(h, w)} must be divisible by patch_size={self.patch_size}.")

        router_w = self.intent_router(vis_feat, ir_feat)
        prompt_bank = self.prompt_bank().unsqueeze(0).expand(b, -1, -1)
        intent = torch.bmm(router_w.unsqueeze(1), prompt_bank).squeeze(1)
        intent_phase = self.intent_to_phase(intent)
        intent_amp = self.intent_to_amp(intent)

        vis_fft = torch.fft.fft2(vis_feat.float(), norm='ortho')
        ir_fft = torch.fft.fft2(ir_feat.float(), norm='ortho')

        vis_amp = torch.log1p(torch.abs(vis_fft))
        ir_amp = torch.log1p(torch.abs(ir_fft))
        vis_phase = torch.angle(vis_fft)
        ir_phase = torch.angle(ir_fft)

        vis_phase_cs = torch.cat([torch.cos(vis_phase), torch.sin(vis_phase)], dim=1)
        ir_phase_cs = torch.cat([torch.cos(ir_phase), torch.sin(ir_phase)], dim=1)

        amp_scores, vis_amp_tok, ir_amp_tok, h_t, w_t = self._score_amplitude(vis_amp, ir_amp, intent_amp)
        phase_scores, vis_phase_tok, ir_phase_tok, _, _ = self._score_phase(vis_phase_cs, ir_phase_cs, intent_phase)

        fused_amp_tok, amp_mask = self._fuse_tokens(vis_amp_tok, ir_amp_tok, amp_scores, self.amp_topk_ratio, self.amp_cross)
        fused_phase_tok, phase_mask = self._fuse_tokens(vis_phase_tok, ir_phase_tok, phase_scores, self.phase_topk_ratio, self.phase_cross)

        fused_amp_map = fused_amp_tok.transpose(1, 2).reshape(b, c, h_t, w_t)
        fused_amp_map = F.interpolate(fused_amp_map, size=(h, w), mode='nearest')

        fused_phase_map = fused_phase_tok.transpose(1, 2).reshape(b, c * 2, h_t, w_t)
        fused_phase_map = F.interpolate(fused_phase_map, size=(h, w), mode='nearest')
        fused_cos, fused_sin = torch.chunk(fused_phase_map, chunks=2, dim=1)
        norm = torch.sqrt(fused_cos.pow(2) + fused_sin.pow(2) + 1e-6)
        fused_cos = fused_cos / norm
        fused_sin = fused_sin / norm
        fused_phase_angle = torch.atan2(fused_sin, fused_cos)

        fused_complex_mag = torch.exp(fused_amp_map) - 1.0
        fused_complex = fused_complex_mag * torch.exp(1j * fused_phase_angle)
        fused_feat = torch.fft.ifft2(fused_complex, norm='ortho').real
        fused_feat = fused_feat + 0.5 * (vis_feat + ir_feat)
        fused_feat = self.out_proj(fused_feat)

        if not self.return_aux:
            return fused_feat

        aux: Dict[str, torch.Tensor] = {
            'intent_weights': router_w,
            'intent_embedding': intent,
            'amp_scores': amp_scores,
            'phase_scores': phase_scores,
            'amp_mask': amp_mask,
            'phase_mask': phase_mask,
        }
        return fused_feat, aux
