import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# -*- coding: utf-8 -*-
import torch
from net.frequency_fusion.fusion_block import HighLevelGuidedFrequencyFusion


def main():
    torch.manual_seed(0)
    model = HighLevelGuidedFrequencyFusion(
        in_channels=8,
        patch_size=2,
        prior_dim=16,
        amp_topk_ratio=0.25,
        phase_topk_ratio=0.25,
        token_embed_dim=32,
        num_heads=2,
        return_aux=True,
        use_real_clip_prompt_bank=False,
    )
    vis = torch.randn(2, 8, 16, 16, requires_grad=True)
    ir = torch.randn(2, 8, 16, 16, requires_grad=True)

    fused, aux = model(vis, ir)
    aux["amp_score"].retain_grad()
    aux["phase_score"].retain_grad()

    loss = fused.mean() + fused.abs().mean()
    loss.backward()

    amp_grad = aux["amp_score"].grad
    phase_grad = aux["phase_score"].grad
    amp_mask = aux["amp_mask"].bool()
    phase_mask = aux["phase_mask"].bool()

    print('amp_score grad mean(all):', amp_grad.abs().mean().item())
    print('phase_score grad mean(all):', phase_grad.abs().mean().item())
    print('amp_score grad mean(selected):', amp_grad[amp_mask].abs().mean().item())
    print('amp_score grad mean(unselected):', amp_grad[~amp_mask].abs().mean().item())
    print('phase_score grad mean(selected):', phase_grad[phase_mask].abs().mean().item())
    print('phase_score grad mean(unselected):', phase_grad[~phase_mask].abs().mean().item())

    print('amp_score first layer grad mean:', model.amp_score.score_mlp[0].weight.grad.abs().mean().item())
    print('phase_score first layer grad mean:', model.phase_score.score_mlp[0].weight.grad.abs().mean().item())


if __name__ == '__main__':
    main()
