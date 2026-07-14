import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fusion.text_conditioned_spatial_fusion import SemanticAffineModulation
from utils.loss import FrequencyConsistencyLoss


def _last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise RuntimeError("SemanticAffineModulation MLP has no Linear layer")


class FrequencyConsistencyLossTests(unittest.TestCase):
    def _assert_finite_scalar(self, value: torch.Tensor) -> None:
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value).item())

    def _expected_mask(self, loss: FrequencyConsistencyLoss, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        ir_saliency = torch.sigmoid(loss._standardize(ir))
        diff_saliency = torch.sigmoid(loss._standardize(ir - vis))
        response = 0.5 * (
            ir_saliency * (1.0 - diff_saliency)
            + diff_saliency * (1.0 - ir_saliency)
        )
        threshold = response.mean(dim=(-2, -1), keepdim=True) + response.std(
            dim=(-2, -1), keepdim=True, unbiased=False
        )
        return (response > threshold).to(response.dtype)

    def test_main_shapes_mask_pairings_and_backward(self):
        torch.manual_seed(7)
        loss = FrequencyConsistencyLoss()
        visible = torch.randn(2, 1, 64, 64)
        infrared = torch.randn(2, 1, 64, 64)
        fused = torch.randn(2, 1, 64, 64, requires_grad=True)

        total_loss, pos_loss, neg_loss = loss(visible, infrared, fused)
        self._assert_finite_scalar(total_loss)
        self._assert_finite_scalar(pos_loss)
        self._assert_finite_scalar(neg_loss)
        total_loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

        mask = loss._adaptive_mask(infrared, visible)
        expected_mask = self._expected_mask(loss, infrared, visible)
        torch.testing.assert_close(mask, expected_mask)
        self.assertTrue(torch.logical_or(mask == 0, mask == 1).all().item())

        per_sample_mask = torch.cat(
            [
                loss._adaptive_mask(infrared[index : index + 1], visible[index : index + 1])
                for index in range(infrared.shape[0])
            ],
            dim=0,
        )
        torch.testing.assert_close(mask, per_sample_mask)

        inv_mask = 1.0 - mask
        f_fg = fused.detach() * mask
        f_bg = fused.detach() * inv_mask
        ir_fg = infrared * mask
        ir_bg = infrared * inv_mask
        vis_fg = visible * mask
        vis_bg = visible * inv_mask
        expected_neg_loss = (
            loss._fft_l1(f_fg, ir_bg)
            + loss._fft_l1(f_fg, vis_fg)
            + loss._fft_l1(f_bg, ir_fg)
            + loss._fft_l1(f_bg, vis_bg)
        )
        torch.testing.assert_close(neg_loss, expected_neg_loss)

    def test_edge_cases_remain_finite(self):
        torch.manual_seed(11)
        loss = FrequencyConsistencyLoss()
        cases = [
            (torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)),
            (torch.full((1, 1, 8, 8), 0.25), torch.full((1, 1, 8, 8), 0.25), torch.full((1, 1, 8, 8), 0.25)),
            (torch.randn(1, 1, 8, 8), None, torch.randn(1, 1, 8, 8)),
        ]
        equal_visible = cases[2][0]
        cases[2] = (equal_visible, equal_visible.clone(), cases[2][2])

        for visible, infrared, fused_values in cases:
            fused = fused_values.clone().requires_grad_(True)
            total_loss, pos_loss, neg_loss = loss(visible, infrared, fused)
            for value in (total_loss, pos_loss, neg_loss):
                self._assert_finite_scalar(value)
            total_loss.backward()
            self.assertIsNotNone(fused.grad)
            self.assertTrue(torch.isfinite(fused.grad).all().item())


class SemanticAffineModulationTests(unittest.TestCase):
    def test_identity_initialization_and_optimizer_step(self):
        torch.manual_seed(23)
        module = SemanticAffineModulation(channels=4, intent_dim=3)
        feat = torch.randn(2, 4, 8, 8, requires_grad=True)
        z_fus = torch.randn(2, 3)
        target = torch.randn(2, 4, 8, 8) + 0.75

        output_before = module(feat, z_fus)
        self.assertEqual(output_before.shape, feat.shape)
        self.assertLess(torch.max(torch.abs(output_before - feat)).item(), 1e-6)

        final_linear = _last_linear(module.mlp)
        final_linear_before = [parameter.detach().clone() for parameter in final_linear.parameters()]
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        optimizer.zero_grad()
        F.mse_loss(output_before, target).backward()

        for gradient in (
            feat.grad,
            module.mod_scale.grad,
            final_linear.weight.grad,
            module.gamma_proj.weight.grad,
            module.beta_proj.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())

        optimizer.step()
        final_linear_delta = max(
            (parameter.detach() - before).abs().max().item()
            for parameter, before in zip(final_linear.parameters(), final_linear_before)
        )
        self.assertTrue(torch.isfinite(torch.tensor(final_linear_delta)).item())
        self.assertGreater(final_linear_delta, 0.0)

        with torch.no_grad():
            output_after = module(feat.detach(), z_fus)
        output_delta = output_after - output_before.detach()
        self.assertTrue(torch.isfinite(output_delta).all().item())
        self.assertGreater(output_delta.abs().max().item(), 0.0)

        optimizer.zero_grad()
        next_feat = feat.detach().clone().requires_grad_(True)
        F.mse_loss(module(next_feat, z_fus), target).backward()
        self.assertIsNotNone(next_feat.grad)
        self.assertTrue(torch.isfinite(next_feat.grad).all().item())


if __name__ == "__main__":
    unittest.main()
