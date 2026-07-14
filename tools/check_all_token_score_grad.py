# -*- coding: utf-8 -*-
"""Sanity-check gradient flow through the v11 intent query variants."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from config import CLIP_DOWNLOAD_ROOT, CLIP_MODEL_NAME
from net.intent import DualDomainTextIntentGenerator, DualStreamIntentMLP


def _assert_nonempty_grad(name, parameter):
    if parameter.grad is None or parameter.grad.abs().sum() == 0:
        raise AssertionError(f'{name} has no non-zero gradient')


def _run_clip_similarity_check(device):
    intent = DualDomainTextIntentGenerator(
        intent_dim=8,
        clip_model_name=CLIP_MODEL_NAME,
        clip_download_root=CLIP_DOWNLOAD_ROOT,
        clip_device=str(device),
    ).to(device)
    i_deg, i_fus, aux = intent(torch.rand(2, 3, 224, 224, device=device))
    (i_deg.mean() + i_fus.mean()).backward()
    _assert_nonempty_grad('deg_proj', intent.deg_proj.weight)
    _assert_nonempty_grad('fus_proj', intent.fus_proj.weight)
    _assert_nonempty_grad('logit_scale_deg', intent.logit_scale_deg)
    _assert_nonempty_grad('logit_scale_fus', intent.logit_scale_fus)
    if aux['deg_prompt_weight'].shape != (2, 4) or aux['fus_prompt_weight'].shape != (2, 4):
        raise AssertionError('Expected four-way v11 prompt weights')
    print('CLIP similarity query gradient check passed.')


def _run_shared_encoder_mlp_check(device):
    intent = DualStreamIntentMLP(
        channels=4,
        intent_dim=8,
        hidden_dim=16,
        clip_model_name=CLIP_MODEL_NAME,
        clip_download_root=CLIP_DOWNLOAD_ROOT,
        clip_device=str(device),
    ).to(device)
    pyramid = [torch.rand(2, 4, 16, 16, device=device), torch.rand(2, 4, 8, 8, device=device),
               torch.rand(2, 4, 4, 4, device=device)]
    i_deg, i_fus, _ = intent(torch.rand(2, 3, 224, 224, device=device), pyramid, pyramid, pyramid[0], pyramid[0])
    (i_deg.mean() + i_fus.mean()).backward()
    _assert_nonempty_grad('deg_weighting_head', intent.deg_weighting_head[0].weight)
    _assert_nonempty_grad('fus_weighting_head', intent.fus_weighting_head[0].weight)
    if hasattr(intent, 'clip_model') or any('clip_model' in key for key in intent.state_dict()):
        raise AssertionError('SharedEncoder-MLP ablation must not retain CLIP weights')
    print('SharedEncoder-MLP query gradient check passed.')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _run_clip_similarity_check(device)
    _run_shared_encoder_mlp_check(device)


if __name__ == '__main__':
    main()
