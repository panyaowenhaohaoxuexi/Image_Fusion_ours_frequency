# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoder.blocks import ConvBNAct, ResidualBlock
from net.restormer_light import TransformerBlock


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class DecoderStage(nn.Module):
    """Decoder stage with Restormer refinement."""

    def __init__(self, channels: int, num_heads: int = 1,
                 ffn_expansion_factor: float = 2.0, bias: bool = False,
                 layer_norm_type: str = 'WithBias'):
        super().__init__()
        self.body = nn.Sequential(
            TransformerBlock(channels, _valid_heads(channels, num_heads), ffn_expansion_factor, bias, layer_norm_type),
            ConvBNAct(channels, channels, 3, 1, 1, activation='gelu'),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class GatedSkipFusion(nn.Module):
    """Fuse an upsampled decoder feature with a same-scale skip feature."""

    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.GELU(),
            ResidualBlock(channels),
        )
        final_gate_conv = self.gate[2]
        nn.init.normal_(final_gate_conv.weight, mean=0.0, std=1e-3)
        if final_gate_conv.bias is not None:
            nn.init.zeros_(final_gate_conv.bias)

    def forward(self, high_up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if high_up.shape[-2:] != skip.shape[-2:]:
            high_up = F.interpolate(high_up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip_gate = self.gate(torch.cat([high_up, skip], dim=1))
        mixed = skip_gate * skip + (1.0 - skip_gate) * high_up
        return self.refine(mixed)


class SimpleDecoder(nn.Module):
    """Multi-scale decoder with optional image-base residual reconstruction."""

    def __init__(self, channels=64, out_channels=1, inner_dim=32, num_blocks=2,
                 num_heads=1, ffn_expansion_factor=2.0, bias=False,
                 LayerNorm_type='WithBias', max_residual_scale: float = 0.4):
        super().__init__()
        self.max_residual_scale = float(max_residual_scale)
        self.reduce_l1 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)
        self.reduce_l2 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)
        self.reduce_l3 = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=bias)

        self.stage_l3 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l2 = GatedSkipFusion(inner_dim, bias=bias)
        self.stage_l2 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.fuse_l1 = GatedSkipFusion(inner_dim, bias=bias)
        self.stage_l1 = nn.ModuleList([
            DecoderStage(inner_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inner_dim, out_channels, 3, 1, 1, bias=bias),
        )
        final_head_conv = self.head[-1]
        nn.init.normal_(final_head_conv.weight, mean=0.0, std=1e-3)
        if final_head_conv.bias is not None:
            nn.init.zeros_(final_head_conv.bias)
        self.residual_scale_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: torch.Tensor):
        for block in blocks:
            x = block(x)
        return x

    @staticmethod
    def _validate_image_inputs(
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        weight_ir: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> None:
        inputs = {'image_vis': image_vis, 'image_ir': image_ir, 'weight_ir': weight_ir}
        for name, tensor in inputs.items():
            if tensor.ndim != 4:
                raise ValueError(f'{name} must be a BCHW tensor, got shape {tuple(tensor.shape)}.')
            if tensor.device != decoder_feature.device:
                raise ValueError(f'{name} must be on decoder device {decoder_feature.device}, got {tensor.device}.')
        if image_vis.shape[1] != 1 or image_ir.shape[1] != 1 or weight_ir.shape[1] != 1:
            raise ValueError('image_vis, image_ir, and weight_ir must each have one channel.')
        if image_vis.shape[0] != image_ir.shape[0] or image_vis.shape[0] != weight_ir.shape[0]:
            raise ValueError('image_vis, image_ir, and weight_ir must have matching batch sizes.')
        if image_vis.shape[-2:] != image_ir.shape[-2:]:
            raise ValueError('image_vis and image_ir must have matching spatial dimensions.')

    def forward(
        self,
        D_L1: torch.Tensor,
        D_L2: torch.Tensor,
        D_L3: torch.Tensor,
        image_vis: torch.Tensor = None,
        image_ir: torch.Tensor = None,
        weight_ir: torch.Tensor = None,
        return_aux: bool = False,
    ):
        x_l1 = self.reduce_l1(D_L1)
        x_l2 = self.reduce_l2(D_L2)
        x_l3 = self.reduce_l3(D_L3)

        d_l3 = self._run_stage(self.stage_l3, x_l3)
        d_l2 = self.fuse_l2(
            F.interpolate(d_l3, size=x_l2.shape[-2:], mode='bilinear', align_corners=False),
            x_l2,
        )
        d_l2 = self._run_stage(self.stage_l2, d_l2)
        d_l1 = self.fuse_l1(
            F.interpolate(d_l2, size=x_l1.shape[-2:], mode='bilinear', align_corners=False),
            x_l1,
        )
        d_l1 = self._run_stage(self.stage_l1, d_l1)

        residual_logits = self.head(d_l1)
        residual = torch.tanh(residual_logits)
        source_inputs = {'image_vis': image_vis, 'image_ir': image_ir, 'weight_ir': weight_ir}
        supplied = {name: value is not None for name, value in source_inputs.items()}
        if any(supplied.values()) and not all(supplied.values()):
            missing = ', '.join(name for name, is_supplied in supplied.items() if not is_supplied)
            raise ValueError(f'Image-base reconstruction requires all source inputs; missing: {missing}.')

        if all(supplied.values()):
            self._validate_image_inputs(image_vis, image_ir, weight_ir, d_l1)
            image_vis = image_vis.to(dtype=d_l1.dtype)
            image_ir = image_ir.to(dtype=d_l1.dtype)
            weight_ir = weight_ir.to(dtype=d_l1.dtype)
            weight_ir = F.interpolate(weight_ir, size=image_vis.shape[-2:], mode='bilinear', align_corners=False)
            weight_ir = weight_ir.clamp(0.0, 1.0)
            base_image = weight_ir * image_ir + (1.0 - weight_ir) * image_vis
            if residual.shape[-2:] != base_image.shape[-2:]:
                residual = F.interpolate(residual, size=base_image.shape[-2:], mode='bilinear', align_corners=False)
            residual_scale = self.max_residual_scale * torch.sigmoid(self.residual_scale_param)
            out = torch.clamp(base_image + residual_scale * residual, 0.0, 1.0)
            if out.shape != image_vis.shape or out.shape != image_ir.shape:
                raise RuntimeError('Decoder output must match both source-image shapes.')
            aux = {
                'base_image': base_image,
                'residual': residual,
                'residual_scale': residual_scale,
            }
        else:
            out = torch.sigmoid(residual_logits)
            aux = {
                'base_image': None,
                'residual': residual,
                'residual_scale': None,
            }

        if return_aux:
            return out, d_l1, aux
        return out, d_l1
