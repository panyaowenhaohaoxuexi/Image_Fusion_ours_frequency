"""Run a CLIP-free forward/backward diagnostic for the v13 fusion components."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from net.FSRC import FSRC
from net.decoder.simple_decoder import SimpleDecoder
from net.fusion.text_conditioned_spatial_fusion import TextConditionedSpatialAdaptiveFusion


def _stats(name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().float()
    print(
        f'{name}: shape={tuple(value.shape)} min={value.min().item():.6f} '
        f'max={value.max().item():.6f} mean={value.mean().item():.6f} std={value.std(unbiased=False).item():.6f}'
    )


def _module_grad_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().abs().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(sum(values, torch.zeros((), device=next(module.parameters()).device)).item())


def _axis_variation(gate: torch.Tensor) -> Tuple[float, float]:
    value = gate.detach().float()
    channel_std = value.std(dim=1, unbiased=False).mean().item()
    spatial_std = value.flatten(2).std(dim=2, unbiased=False).mean().item()
    return channel_std, spatial_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(29)

    spatial = TextConditionedSpatialAdaptiveFusion(channels=64, intent_dim=64, num_heads=1).to(device)
    fsrc_modules = [FSRC(channels=64).to(device) for _ in range(3)]
    decoder = SimpleDecoder(channels=64, out_channels=1, inner_dim=32, num_blocks=2, max_residual_scale=0.4).to(device)
    image_vis = (0.2 + 0.6 * torch.rand(1, 1, 32, 32, device=device)).requires_grad_()
    image_ir = (0.2 + 0.6 * torch.rand(1, 1, 32, 32, device=device)).requires_grad_()
    intent = torch.randn(1, 64, device=device, requires_grad=True)
    vis_pyramid = [torch.randn(1, 64, size, size, device=device, requires_grad=True) for size in (32, 16, 8)]
    ir_pyramid = [torch.randn(1, 64, size, size, device=device, requires_grad=True) for size in (32, 16, 8)]
    freq_pyramid = [torch.randn(1, 64, size, size, device=device, requires_grad=True) for size in (32, 16, 8)]

    _, spatial_pyramid, spatial_aux = spatial(vis_pyramid, ir_pyramid, intent, return_aux=True, return_pyramid=True)
    weight_multiscale = spatial_aux['weight_multiscale']
    weight_multiscale.retain_grad()
    fsrc_outputs, fsrc_aux = [], []
    for index, module in enumerate(fsrc_modules, start=1):
        fused, aux = module(freq_pyramid[index - 1], spatial_pyramid[f'l{index}'], return_channel_gate=True)
        fsrc_outputs.append(fused)
        fsrc_aux.append(aux)
    out, _, decoder_aux = decoder(
        fsrc_outputs[0], fsrc_outputs[1], fsrc_outputs[2],
        image_vis=image_vis, image_ir=image_ir, weight_ir=weight_multiscale, return_aux=True,
    )
    out.mean().backward()

    print('image_gate_scale_weights:', spatial_aux['image_gate_scale_weights'].detach().cpu().tolist())
    for level in (1, 2, 3):
        _stats(f'raw_weight_l{level}', spatial_aux[f'weight_l{level}'])
        _stats(f'weight_channel_l{level}', spatial_aux[f'weight_channel_l{level}'])
    _stats('weight_multiscale', weight_multiscale)
    _stats('weight_channel_mean_l1', spatial_aux['weight_channel_mean_l1'])
    print('mean_abs_multiscale_minus_l1:', (weight_multiscale.detach() - spatial_aux['weight_l1'].detach()).abs().mean().item())
    print('mean_abs_image_gate_minus_channel_mean:', (spatial_aux['weight_l1'].detach() - spatial_aux['weight_channel_mean_l1'].detach()).abs().mean().item())
    channel_std, spatial_std = _axis_variation(spatial_aux['weight_channel_l1'])
    print(f'weight_channel_l1_channel_std={channel_std:.6f} spatial_std={spatial_std:.6f}')
    channel_std, spatial_std = _axis_variation(fsrc_aux[0]['gate_channel'])
    print(f'fsrc_gate_channel_channel_std={channel_std:.6f} spatial_std={spatial_std:.6f}')
    print('fsrc_residual_scale_param:', fsrc_modules[0].residual_scale_param.detach().item())
    print('tanh(fsrc_residual_scale_param):', torch.tanh(fsrc_modules[0].residual_scale_param.detach()).item())
    _stats('base_image', decoder_aux['base_image'])
    _stats('residual', decoder_aux['residual'])
    _stats('fused_output', out)
    saturation = ((out.detach() <= 1e-4) | (out.detach() >= 1.0 - 1e-4)).float().mean().item()
    print('fused_output_saturation_ratio:', saturation)
    print('weight_multiscale_grad_norm:', weight_multiscale.grad.abs().sum().item())
    print('image_gate_scale_logits_grad_norm:', spatial.image_gate_scale_logits.grad.abs().sum().item())
    for index, level in enumerate((spatial.level1, spatial.level2, spatial.level3), start=1):
        print(f'level{index}_image_gate_grad_norm:', _module_grad_norm(level.weight_gate.image_gate))
    for index, module in enumerate(fsrc_modules, start=1):
        print(f'fsrc_l{index}_channel_gate_grad_norm:', _module_grad_norm(module.channel_mlp))
        print(f'fsrc_l{index}_spatial_gate_grad_norm:', _module_grad_norm(module.spatial_gate))
        print(f'fsrc_l{index}_difference_residual_grad_norm:', _module_grad_norm(module.difference_residual))
        print(f'fsrc_l{index}_joint_context_grad_norm:', _module_grad_norm(module.joint_context_residual))
    print('decoder_skip_l2_grad_norm:', _module_grad_norm(decoder.fuse_l2))
    print('decoder_skip_l1_grad_norm:', _module_grad_norm(decoder.fuse_l1))
    print('decoder_head_grad_norm:', _module_grad_norm(decoder.head))


if __name__ == '__main__':
    main()
