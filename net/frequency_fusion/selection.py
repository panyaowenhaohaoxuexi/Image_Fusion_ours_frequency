# -*- coding: utf-8 -*-
import torch


def topk_token_selection(score: torch.Tensor, keep_ratio: float):
    """Hard Top-K token selection.

    score: [B, N]
    keep_ratio: selected-token ratio.
    return:
      topk_index: [B, K]
      mask:       [B, N] bool, True for selected tokens
      topk_value: [B, K]
    """
    _, n = score.shape
    k = max(1, int(n * keep_ratio))
    k = min(k, n)
    topk_value, topk_index = torch.topk(score, k=k, dim=1)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, topk_index, True)
    return topk_index, mask, topk_value


def straight_through_topk_mask(score: torch.Tensor,
                               hard_mask: torch.Tensor,
                               keep_ratio: float,
                               temperature: float = 0.25) -> torch.Tensor:
    """Straight-through Top-K routing mask.

    Forward uses the hard Top-K mask, so high-score tokens truly enter the
    selected-token path and low-score tokens enter the unselected-token path.
    Backward uses a soft threshold mask, allowing the score function to receive
    gradient around the Top-K decision boundary.
    """
    _, n = score.shape
    k = max(1, int(n * keep_ratio))
    k = min(k, n)

    score_mean = score.mean(dim=1, keepdim=True)
    score_std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    score_norm = (score - score_mean) / score_std

    topk_value, _ = torch.topk(score_norm, k=k, dim=1)
    threshold = topk_value[:, -1:].detach()
    soft_mask = torch.sigmoid((score_norm - threshold) / max(float(temperature), 1e-4))
    hard_mask = hard_mask.float()
    return hard_mask - soft_mask.detach() + soft_mask


def gather_tokens(tokens: torch.Tensor, index: torch.Tensor):
    _, _, d = tokens.shape
    gather_index = index.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(tokens, dim=1, index=gather_index)


def scatter_tokens(full_tokens: torch.Tensor, selected_tokens: torch.Tensor, index: torch.Tensor):
    _, _, d = full_tokens.shape
    scatter_index = index.unsqueeze(-1).expand(-1, -1, d)
    full_tokens = full_tokens.clone()
    full_tokens.scatter_(1, scatter_index, selected_tokens)
    return full_tokens
