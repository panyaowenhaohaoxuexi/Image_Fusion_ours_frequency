# -*- coding: utf-8 -*-
import os
import sys
import inspect

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from net.dda import DDA
from net.decoder.simple_decoder import SimpleDecoder
from net.frequency_fusion.fusion_block import HighLevelGuidedFrequencyFusion
from net.fusion.text_conditioned_spatial_fusion import TextConditionedSpatialAdaptiveFusion
from net.intent import DualDomainTextIntentGenerator
from utils.loss import TokenRoutingRankingLoss


def _assert_nonzero_grad(name, tensor):
    if tensor.grad is None:
        raise AssertionError(f'{name} has no gradient')
    value = tensor.grad.abs().mean().item()
    print(f'{name} grad mean:', value)
    if value <= 0.0:
        raise AssertionError(f'{name} gradient is zero')


def _first_param_grad_mean(module):
    grads = [p.grad.abs().mean() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.stack(grads).mean().item()


def _assert_module_grad(name, module):
    value = _first_param_grad_mean(module)
    print(f'{name} param grad mean:', value)
    if value <= 0.0:
        raise AssertionError(f'{name} has no non-zero parameter gradient')


def _assert_convex_hull(generator, mode_name):
    generator.eval()
    torch.manual_seed(100)
    vis_spa_a = torch.randn(2, 4, 8, 8)
    ir_spa_a = torch.randn(2, 4, 8, 8)
    vis_freq_a = torch.randn(2, 4, 8, 8)
    ir_freq_a = torch.randn(2, 4, 8, 8)
    vis_spa_b = torch.randn(2, 4, 8, 8)
    ir_spa_b = torch.randn(2, 4, 8, 8)
    vis_freq_b = torch.randn(2, 4, 8, 8)
    ir_freq_b = torch.randn(2, 4, 8, 8)

    with torch.no_grad():
        deg_a, fus_a, aux_a = generator(vis_spa_a, ir_spa_a, vis_freq_a, ir_freq_a)
        deg_b, fus_b, aux_b = generator(vis_spa_b, ir_spa_b, vis_freq_b, ir_freq_b)

    for suffix, deg, fus, aux in [('a', deg_a, fus_a, aux_a), ('b', deg_b, fus_b, aux_b)]:
        deg_w = aux['deg_prompt_weight']
        fus_w = aux['fus_prompt_weight']
        deg_bank = aux['deg_prompt_bank']
        fus_bank = aux['fus_prompt_bank']
        for name, weight in [(f'deg_{suffix}', deg_w), (f'fus_{suffix}', fus_w)]:
            if weight.shape != (2, 6):
                raise AssertionError(f'{mode_name} {name} weight shape is {tuple(weight.shape)}, expected (2, 6)')
            if torch.any(weight < -1e-7):
                raise AssertionError(f'{mode_name} {name} weight has negative values')
            max_sum_err = (weight.sum(dim=1) - 1.0).abs().max().item()
            if max_sum_err > 1e-5:
                raise AssertionError(f'{mode_name} {name} weight sum error {max_sum_err} > 1e-5')

        deg_recon = deg_w.matmul(deg_bank)
        fus_recon = fus_w.matmul(fus_bank)
        deg_err = (deg - deg_recon).abs().max().item()
        fus_err = (fus - fus_recon).abs().max().item()
        print(f'{mode_name} convex {suffix}: deg_err={deg_err:.8f}, fus_err={fus_err:.8f}')
        if deg_err > 1e-5:
            raise AssertionError(f'{mode_name} I_deg is not exactly weight @ bank')
        if fus_err > 1e-5:
            raise AssertionError(f'{mode_name} I_fus is not exactly weight @ bank')


def _assert_signatures():
    decoder_params = list(inspect.signature(SimpleDecoder.forward).parameters)
    dda_params = list(inspect.signature(DDA.forward).parameters)
    if decoder_params != ['self', 'inp_img', 'base_feature', 'freq_feature']:
        raise AssertionError(f'SimpleDecoder.forward signature is {decoder_params}')
    if dda_params != ['self', 'F_freq', 'F_spa']:
        raise AssertionError(f'DDA.forward signature is {dda_params}')


def _run_full_gradient_check(mode_name: str, use_clip_prompt_bank: bool,
                             use_learnable_prompt_embedding: bool):
    torch.manual_seed(0)
    intent = DualDomainTextIntentGenerator(
        channels=4,
        intent_dim=8,
        hidden_dim=16,
        use_learnable_prompt_embedding=use_learnable_prompt_embedding,
        use_clip_prompt_bank=use_clip_prompt_bank,
    )
    freq_model = HighLevelGuidedFrequencyFusion(
        in_channels=4,
        patch_size=2,
        prior_dim=8,
        amp_topk_ratio=0.25,
        phase_topk_ratio=0.25,
        token_embed_dim=16,
        num_heads=1,
        return_aux=True,
        routing_temperature=0.25,
    )
    spatial_model = TextConditionedSpatialAdaptiveFusion(
        channels=4,
        intent_dim=8,
        num_heads=1,
    )
    dda = DDA(channels=4)
    decoder = SimpleDecoder(
        channels=4,
        out_channels=1,
        inner_dim=8,
        num_blocks=1,
        num_heads=1,
    )
    score_criterion = TokenRoutingRankingLoss()

    vis = torch.randn(1, 4, 8, 8, requires_grad=True)
    ir = torch.randn(1, 4, 8, 8, requires_grad=True)
    decoder_skip = torch.randn(1, 1, 8, 8)

    I_deg, I_fus, intent_aux = intent(vis, ir, vis, ir)
    I_deg.retain_grad()
    I_fus.retain_grad()

    fused_freq, freq_aux = freq_model(vis, ir, frequency_intent=I_deg)
    if not torch.equal(freq_aux['frequency_intent'], I_deg):
        raise AssertionError(f'{mode_name} frequency branch did not receive I_deg')
    fused_spa, spa_aux = spatial_model(vis, ir, I_fus, return_aux=True)
    dual_feature, dda_gate = dda(fused_freq, fused_spa)
    fused_img, _ = decoder(decoder_skip, dual_feature, fused_freq)

    freq_aux['amp_score'].retain_grad()
    freq_aux['phase_score'].retain_grad()
    dda_gate.retain_grad()
    spa_aux['weight_l1'].retain_grad()

    score_loss, _ = score_criterion(freq_aux)
    loss = (
        fused_img.mean()
        + dual_feature.abs().mean()
        + fused_spa.abs().mean()
        + fused_freq.abs().mean()
        + 0.03 * score_loss
        + I_deg.mean()
        + I_fus.mean()
    )
    loss.backward()

    if freq_aux['amp_topk_index'].numel() == 0:
        raise AssertionError(f'{mode_name} amp_topk_index is empty')
    if freq_aux['phase_topk_index'].numel() == 0:
        raise AssertionError(f'{mode_name} phase_topk_index is empty')
    for key in ['weight_l1', 'weight_l2', 'weight_l3']:
        if key not in spa_aux:
            raise AssertionError(f'{mode_name} missing spa_aux[{key}]')

    print(f'{mode_name} amp_topk_index shape:', tuple(freq_aux['amp_topk_index'].shape))
    print(f'{mode_name} phase_topk_index shape:', tuple(freq_aux['phase_topk_index'].shape))
    _assert_nonzero_grad(f'{mode_name} I_deg', I_deg)
    _assert_nonzero_grad(f'{mode_name} I_fus', I_fus)
    _assert_nonzero_grad(f'{mode_name} amp_score', freq_aux['amp_score'])
    _assert_nonzero_grad(f'{mode_name} phase_score', freq_aux['phase_score'])
    _assert_nonzero_grad(f'{mode_name} spatial weight_l1', spa_aux['weight_l1'])
    _assert_nonzero_grad(f'{mode_name} dda_gate', dda_gate)
    _assert_module_grad(f'{mode_name} deg_weighting_head', intent.deg_weighting_head)
    _assert_module_grad(f'{mode_name} fus_weighting_head', intent.fus_weighting_head)
    _assert_module_grad(f'{mode_name} amp_score module', freq_model.amp_score)
    _assert_module_grad(f'{mode_name} phase_score module', freq_model.phase_score)
    _assert_module_grad(f'{mode_name} spatial weight generator', spatial_model.level1.weight_gate)
    _assert_module_grad(f'{mode_name} dda gate generator', dda.gate)

    _assert_convex_hull(intent, mode_name)
    print(f'{mode_name} score_loss:', score_loss.item())


def main():
    _assert_signatures()
    _run_full_gradient_check('fallback', use_clip_prompt_bank=False, use_learnable_prompt_embedding=False)
    _run_full_gradient_check('clip', use_clip_prompt_bank=True, use_learnable_prompt_embedding=False)
    _run_full_gradient_check('learnable', use_clip_prompt_bank=False, use_learnable_prompt_embedding=True)


if __name__ == '__main__':
    main()
