# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from net.frequency_fusion.fusion_block import HighLevelGuidedFrequencyFusion
from net.fusion.text_conditioned_spatial_fusion import TextConditionedSpatialAdaptiveFusion
from utils.loss import TokenRoutingRankingLoss


def main():
    torch.manual_seed(0)
    freq_model = HighLevelGuidedFrequencyFusion(
        in_channels=4,
        patch_size=2,
        prior_dim=8,
        amp_topk_ratio=0.25,
        phase_topk_ratio=0.25,
        token_embed_dim=16,
        num_heads=1,
        return_aux=True,
        use_real_clip_prompt_bank=False,
        routing_temperature=0.25,
    )
    spatial_model = TextConditionedSpatialAdaptiveFusion(
        channels=4,
        intent_dim=8,
        num_heads=1,
        use_freq_context=True,
    )
    score_criterion = TokenRoutingRankingLoss()

    vis = torch.randn(1, 4, 8, 8, requires_grad=True)
    ir = torch.randn(1, 4, 8, 8, requires_grad=True)

    fused_freq, aux = freq_model(vis, ir)
    fused_spa = spatial_model(vis, ir, aux['spatial_intent'], fused_freq)
    aux['amp_score'].retain_grad()
    aux['phase_score'].retain_grad()

    score_loss, _ = score_criterion(aux)
    loss = fused_freq.mean() + fused_spa.abs().mean() + 0.03 * score_loss
    loss.backward()

    amp_grad = aux['amp_score'].grad
    phase_grad = aux['phase_score'].grad
    amp_mask = aux['amp_mask'].bool()
    phase_mask = aux['phase_mask'].bool()

    print('amp_score grad mean(all):', amp_grad.abs().mean().item())
    print('phase_score grad mean(all):', phase_grad.abs().mean().item())
    print('amp_score grad mean(selected):', amp_grad[amp_mask].abs().mean().item())
    print('amp_score grad mean(unselected):', amp_grad[~amp_mask].abs().mean().item())
    print('phase_score grad mean(selected):', phase_grad[phase_mask].abs().mean().item())
    print('phase_score grad mean(unselected):', phase_grad[~phase_mask].abs().mean().item())
    print('spatial_intent grad source exists:', aux['spatial_intent'].requires_grad)
    print('score_loss:', score_loss.item())


if __name__ == '__main__':
    main()
