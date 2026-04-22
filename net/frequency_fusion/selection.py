# -*- coding: utf-8 -*-
import torch

def topk_token_selection(score: torch.Tensor, keep_ratio: float):
    _, n = score.shape
    k = max(1, int(n * keep_ratio))
    topk_value, topk_index = torch.topk(score, k=k, dim=1)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, topk_index, True)
    return topk_index, mask, topk_value

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
