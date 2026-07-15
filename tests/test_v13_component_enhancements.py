import math
import unittest

import torch

from net.FSRC import FSRC
from net.decoder.simple_decoder import SimpleDecoder
from net.fusion.text_conditioned_spatial_fusion import (
    PositionAdaptiveWeightGate,
    TextConditionedSpatialAdaptiveFusion,
)


def module_grad_norm(module: torch.nn.Module) -> torch.Tensor:
    values = [parameter.grad.abs().sum() for parameter in module.parameters() if parameter.grad is not None]
    return sum(values, torch.zeros(()))


class PositionAdaptiveWeightGateTests(unittest.TestCase):
    def test_channel_and_image_gates_have_independent_gradients(self):
        torch.manual_seed(7)
        gate = PositionAdaptiveWeightGate(channels=64, intent_dim=64)
        vis = torch.randn(1, 64, 32, 32, requires_grad=True)
        ir = torch.randn(1, 64, 32, 32, requires_grad=True)
        intent = torch.randn(1, 64, requires_grad=True)

        channel_gate, aux = gate(vis, ir, intent)

        self.assertEqual(channel_gate.shape, (1, 64, 32, 32))
        self.assertEqual(aux["weight"].shape, (1, 1, 32, 32))
        self.assertEqual(aux["weight_channel"].shape, channel_gate.shape)
        self.assertEqual(aux["weight_channel_mean"].shape, (1, 1, 32, 32))
        self.assertIsNot(aux["weight"], aux["weight_channel_mean"])
        for value in (channel_gate, aux["weight"], aux["weight_channel_mean"]):
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((value >= 0.0) & (value <= 1.0)).all())
            self.assertLess(abs(value.mean().item() - 0.5), 0.05)

        (aux["weight"].mean() + channel_gate.mean()).backward()
        for gradient in (vis.grad, ir.grad, intent.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(module_grad_norm(gate.image_gate)))
        self.assertGreater(module_grad_norm(gate.image_gate).item(), 0.0)

        with torch.no_grad():
            final_image_gate = gate.image_gate[-1]
            final_image_gate.weight.zero_()
            final_image_gate.bias.fill_(0.2)
        _, known_aux = gate(vis.detach(), ir.detach(), intent.detach())
        self.assertGreater(
            (known_aux["weight"] - known_aux["weight_channel_mean"]).abs().mean().item(),
            0.01,
        )


class FSRCTests(unittest.TestCase):
    def test_channel_spatial_coupling_has_detailed_aux_and_first_step_gradients(self):
        torch.manual_seed(11)
        module = FSRC(channels=64)
        freq = torch.randn(1, 64, 16, 16, requires_grad=True)
        spa = torch.randn(1, 64, 32, 32, requires_grad=True)

        fused, aux = module(freq, spa, return_channel_gate=True)

        self.assertEqual(fused.shape, spa.shape)
        self.assertEqual(aux["gate"].shape, (1, 1, 32, 32))
        self.assertEqual(aux["gate_channel"].shape, spa.shape)
        self.assertEqual(aux["frequency_compensation"].shape, spa.shape)
        self.assertAlmostEqual(aux["residual_scale"].item(), math.tanh(0.1), places=5)
        self.assertTrue(((aux["gate_channel"] >= 0.0) & (aux["gate_channel"] <= 1.0)).all())

        fused.square().mean().backward()
        for gradient in (freq.grad, spa.grad, module.residual_scale_param.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        for branch in (
            module.channel_mlp,
            module.spatial_gate,
            module.difference_residual,
            module.joint_context_residual,
        ):
            self.assertTrue(torch.isfinite(module_grad_norm(branch)))
            self.assertGreater(module_grad_norm(branch).item(), 0.0)

        default_fused, gate_map = module(freq.detach(), spa.detach())
        self.assertEqual(default_fused.shape, spa.shape)
        self.assertEqual(gate_map.shape, (1, 1, 32, 32))


class DecoderTests(unittest.TestCase):
    def test_residual_and_compatibility_interfaces(self):
        torch.manual_seed(13)
        decoder = SimpleDecoder(channels=64, out_channels=1, inner_dim=32, num_blocks=2, max_residual_scale=0.4)
        l1 = torch.randn(1, 64, 32, 32, requires_grad=True)
        l2 = torch.randn(1, 64, 16, 16, requires_grad=True)
        l3 = torch.randn(1, 64, 8, 8, requires_grad=True)
        vis = torch.rand(1, 1, 32, 32, requires_grad=True)
        ir = torch.rand(1, 1, 32, 32, requires_grad=True)
        weight = torch.rand(1, 1, 16, 16, requires_grad=True)

        out, _, aux = decoder(l1, l2, l3, image_vis=vis, image_ir=ir, weight_ir=weight, return_aux=True)
        self.assertEqual(out.shape, vis.shape)
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())
        self.assertAlmostEqual(aux["residual_scale"].item(), 0.2, places=5)
        out.mean().backward()
        for gradient in (l1.grad, l2.grad, l3.grad, vis.grad, ir.grad, weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

        compat_out, _, compat_aux = decoder(l1.detach(), l2.detach(), l3.detach(), return_aux=True)
        self.assertTrue(((compat_out >= 0.0) & (compat_out <= 1.0)).all())
        self.assertIsNone(compat_aux["base_image"])
        self.assertIsNone(compat_aux["residual_scale"])
        with self.assertRaisesRegex(ValueError, "image_ir, weight_ir"):
            decoder(l1.detach(), l2.detach(), l3.detach(), image_vis=vis.detach())


class MultiScaleGateSmokeTests(unittest.TestCase):
    def test_all_image_gates_receive_decoder_gradients(self):
        torch.manual_seed(17)
        spatial = TextConditionedSpatialAdaptiveFusion(channels=64, intent_dim=64, num_heads=1)
        fsrc = [FSRC(channels=64) for _ in range(3)]
        decoder = SimpleDecoder(channels=64, out_channels=1, inner_dim=32, num_blocks=2, max_residual_scale=0.4)
        vis = torch.rand(1, 1, 32, 32, requires_grad=True)
        ir = torch.rand(1, 1, 32, 32, requires_grad=True)
        intent = torch.randn(1, 64, requires_grad=True)
        vis_pyramid = [torch.randn(1, 64, 32, 32, requires_grad=True), torch.randn(1, 64, 16, 16, requires_grad=True), torch.randn(1, 64, 8, 8, requires_grad=True)]
        ir_pyramid = [torch.randn_like(value, requires_grad=True) for value in vis_pyramid]
        freq_pyramid = [torch.randn_like(value, requires_grad=True) for value in vis_pyramid]

        _, spatial_pyramid, spatial_aux = spatial(vis_pyramid, ir_pyramid, intent, return_aux=True, return_pyramid=True)
        weight_multiscale = spatial_aux["weight_multiscale"]
        weight_multiscale.retain_grad()
        for key in (
            "weight_l1", "weight_l2", "weight_l3",
            "weight_raw_l1", "weight_raw_l2", "weight_raw_l3",
            "weight_channel_l1", "weight_channel_l2", "weight_channel_l3",
            "weight_channel_mean_l1", "weight_channel_mean_l2", "weight_channel_mean_l3",
            "image_gate_scale_weights",
        ):
            self.assertIn(key, spatial_aux)
        self.assertTrue(((weight_multiscale >= -1e-6) & (weight_multiscale <= 1.0 + 1e-6)).all())
        self.assertEqual(tuple(spatial_aux["image_gate_scale_weights"].shape), (3,))
        self.assertAlmostEqual(spatial_aux["image_gate_scale_weights"].sum().item(), 1.0, places=6)
        decoded_features = [
            fsrc[index](freq_pyramid[index], spatial_pyramid[f"l{index + 1}"])[0]
            for index in range(3)
        ]
        out, _ = decoder(
            decoded_features[0], decoded_features[1], decoded_features[2],
            image_vis=vis, image_ir=ir, weight_ir=weight_multiscale,
        )
        out.mean().backward()

        for level in (spatial.level1, spatial.level2, spatial.level3):
            self.assertGreater(module_grad_norm(level.weight_gate.image_gate).item(), 0.0)
        self.assertIsNotNone(spatial.image_gate_scale_logits.grad)
        self.assertGreater(spatial.image_gate_scale_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(weight_multiscale.grad)
        self.assertGreater(weight_multiscale.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
