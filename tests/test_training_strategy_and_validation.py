# -*- coding: utf-8 -*-
import json
import math
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.loss import CorrelationConsistencyLoss, LocalContrastLoss, cc
from utils.training_utils import (
    build_lr_scheduler,
    compute_validation_score,
    get_frequency_weight,
)
from utils.val_metrics import _quantize_to_uint8, _validate_source_uint8, compute_val_metrics
from utils.validation_dataset import PairedValidationDataset


# ---------------------------------------------------------------------------
# 8.1 CorrelationConsistencyLossTests
# ---------------------------------------------------------------------------

class CorrelationConsistencyLossTests(unittest.TestCase):
    def _assert_finite_scalar(self, value: torch.Tensor) -> None:
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value).item())

    def test_random_input(self):
        torch.manual_seed(42)
        criterion = CorrelationConsistencyLoss()
        vis = torch.randn(2, 1, 64, 64)
        ir = torch.randn(2, 1, 64, 64)
        fused = torch.randn(2, 1, 64, 64, requires_grad=True)

        loss = criterion(vis, ir, fused)
        self._assert_finite_scalar(loss)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_constant_input(self):
        cv = torch.full((1, 1, 8, 8), 0.5)
        fused = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
        criterion = CorrelationConsistencyLoss()

        loss = criterion(cv, cv, fused)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_small_input(self):
        torch.manual_seed(7)
        criterion = CorrelationConsistencyLoss()
        vis = torch.randn(1, 1, 4, 4)
        ir = torch.randn(1, 1, 4, 4)
        fused = torch.randn(1, 1, 4, 4, requires_grad=True)

        loss = criterion(vis, ir, fused)
        self._assert_finite_scalar(loss)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())


# ---------------------------------------------------------------------------
# 8.2 LocalContrastLossTests
# ---------------------------------------------------------------------------

class LocalContrastLossTests(unittest.TestCase):
    def _assert_finite_scalar(self, value: torch.Tensor) -> None:
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value).item())

    def test_random_input(self):
        torch.manual_seed(42)
        criterion = LocalContrastLoss(window_size=7)
        vis = torch.randn(2, 1, 64, 64)
        ir = torch.randn(2, 1, 64, 64)
        fused = torch.randn(2, 1, 64, 64, requires_grad=True)

        loss = criterion(vis, ir, fused)
        self._assert_finite_scalar(loss)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_constant_input(self):
        cv = torch.full((1, 1, 64, 64), 0.5)
        fused = torch.full((1, 1, 64, 64), 0.5, requires_grad=True)
        criterion = LocalContrastLoss(window_size=7)

        loss = criterion(cv, cv, fused)
        self._assert_finite_scalar(loss)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_small_input(self):
        torch.manual_seed(7)
        criterion = LocalContrastLoss(window_size=3)
        vis = torch.randn(1, 1, 8, 8)
        ir = torch.randn(1, 1, 8, 8)
        fused = torch.randn(1, 1, 8, 8, requires_grad=True)

        loss = criterion(vis, ir, fused)
        self._assert_finite_scalar(loss)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_window_size_validation(self):
        with self.assertRaises(ValueError):
            LocalContrastLoss(window_size=0)
        with self.assertRaises(ValueError):
            LocalContrastLoss(window_size=4)
        # Odd positive should work
        criterion = LocalContrastLoss(window_size=5)
        self.assertEqual(criterion.window_size, 5)


# ---------------------------------------------------------------------------
# 8.3 FrequencyWarmupTests
# ---------------------------------------------------------------------------

class FrequencyWarmupTests(unittest.TestCase):
    def test_warmup_values(self):
        """Verify exact warmup schedule from epoch 0 to 49."""
        self.assertAlmostEqual(get_frequency_weight(0, warmup_epochs=10, final_weight=0.5), 0.0)
        self.assertAlmostEqual(get_frequency_weight(1, warmup_epochs=10, final_weight=0.5), 0.5 / 9, places=5)
        self.assertAlmostEqual(get_frequency_weight(8, warmup_epochs=10, final_weight=0.5), 0.5 * 8 / 9, places=5)
        self.assertAlmostEqual(get_frequency_weight(9, warmup_epochs=10, final_weight=0.5), 0.5)
        self.assertAlmostEqual(get_frequency_weight(10, warmup_epochs=10, final_weight=0.5), 0.5)
        self.assertAlmostEqual(get_frequency_weight(49, warmup_epochs=10, final_weight=0.5), 0.5)


# ---------------------------------------------------------------------------
# 8.4 LRSchedulerTests
# ---------------------------------------------------------------------------

class LRSchedulerTests(unittest.TestCase):
    def test_warmup_cosine_schedule(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = build_lr_scheduler(
            optimizer=optimizer, num_epochs=50,
            warmup_epochs=5, base_lr=1e-4, min_lr=1e-6,
        )

        used_lrs = []
        for epoch in range(50):
            used_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()
            dummy_loss = parameter.sum() * 0.0
            dummy_loss.backward()
            optimizer.step()
            scheduler.step()

        # Epoch 1-5 should increase
        for i in range(4):
            self.assertLess(used_lrs[i], used_lrs[i + 1])

        # Epoch 5 should be at base_lr
        self.assertAlmostEqual(used_lrs[4], 1e-4, places=8)

        # Epoch 6 should start decreasing
        self.assertLess(used_lrs[5], used_lrs[4])

        # Epoch 6-49 should generally decrease
        for i in range(5, 49):
            self.assertLessEqual(used_lrs[i + 1], used_lrs[i] + 1e-12)

        # Epoch 50 should be close to min_lr
        self.assertAlmostEqual(used_lrs[49], 1e-6, delta=1e-7)

        # All LRs must be positive and finite
        for lr in used_lrs:
            self.assertGreater(lr, 0)
            self.assertTrue(math.isfinite(lr))


# ---------------------------------------------------------------------------
# 8.5 ValidationDatasetTests
# ---------------------------------------------------------------------------

class ValidationDatasetTests(unittest.TestCase):
    def _create_temp_image_pair(self, tmpdir, basename, vis_size=(32, 32), ir_size=None):
        """Create a pair of VIS (RGB) and IR (grayscale) images."""
        if ir_size is None:
            ir_size = vis_size

        vis_dir = os.path.join(tmpdir, "visible")
        ir_dir = os.path.join(tmpdir, "infrared")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(ir_dir, exist_ok=True)

        vis_img = np.random.randint(0, 256, (*vis_size, 3), dtype=np.uint8)
        ir_img = np.random.randint(0, 256, ir_size, dtype=np.uint8)

        vis_path = os.path.join(vis_dir, basename)
        ir_path = os.path.join(ir_dir, basename)

        Image.fromarray(vis_img).save(vis_path)
        Image.fromarray(ir_img).save(ir_path)

        return vis_dir, ir_dir

    def test_correct_pairs_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                self._create_temp_image_pair(tmpdir, f"{i:04d}.png")

            ds = PairedValidationDataset(
                os.path.join(tmpdir, "visible"),
                os.path.join(tmpdir, "infrared"),
                expected_pairs=20,
            )
            self.assertEqual(len(ds), 20)
            self.assertEqual(ds.filenames, sorted(ds.filenames))

            # Load one item
            data_vis_y, data_ir, data_vis_rgb_raw, metric_vis_u8, metric_ir_u8, fname = ds[0]
            self.assertEqual(metric_vis_u8.dtype, torch.uint8)
            self.assertEqual(metric_ir_u8.dtype, torch.uint8)
            self.assertEqual(metric_vis_u8.ndim, 4)
            self.assertEqual(metric_vis_u8.shape[0], 1)
            self.assertEqual(metric_vis_u8.shape[1], 1)

    def test_wrong_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(19):
                self._create_temp_image_pair(tmpdir, f"{i:04d}.png")
            with self.assertRaises(ValueError):
                PairedValidationDataset(
                    os.path.join(tmpdir, "visible"),
                    os.path.join(tmpdir, "infrared"),
                    expected_pairs=20,
                )

    def test_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                self._create_temp_image_pair(tmpdir, f"{i:04d}.png")
            # Remove one IR file
            os.remove(os.path.join(tmpdir, "infrared", "0000.png"))
            with self.assertRaises(ValueError):
                PairedValidationDataset(
                    os.path.join(tmpdir, "visible"),
                    os.path.join(tmpdir, "infrared"),
                    expected_pairs=20,
                )

    def test_excludes_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                self._create_temp_image_pair(tmpdir, f"{i:04d}.png")
            # Create hidden file that should be ignored
            hidden_path = os.path.join(tmpdir, "visible", ".hidden.png")
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(hidden_path)

            ds = PairedValidationDataset(
                os.path.join(tmpdir, "visible"),
                os.path.join(tmpdir, "infrared"),
                expected_pairs=20,
            )
            self.assertEqual(len(ds), 20)

    def test_metric_source_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_image_pair(tmpdir, "0000.png", vis_size=(32, 32), ir_size=(16, 16))
            for i in range(1, 20):
                self._create_temp_image_pair(tmpdir, f"{i:04d}.png")

            ds = PairedValidationDataset(
                os.path.join(tmpdir, "visible"),
                os.path.join(tmpdir, "infrared"),
                expected_pairs=20,
            )
            with self.assertRaises(ValueError):
                _ = ds[0]


# ---------------------------------------------------------------------------
# 8.6 ValidationScoreTests
# ---------------------------------------------------------------------------

class ValidationScoreTests(unittest.TestCase):
    def setUp(self):
        self.base = {"EN": 7.0, "SD": 40.0, "SCD": 1.6, "VIF": 1.0, "QABF": 0.7, "MI": 3.6}
        self.weights = {"EN": 0.15, "SD": 0.10, "SCD": 0.20, "VIF": 0.20, "QABF": 0.25, "MI": 0.10}

    def test_equal_metrics_yields_score_one(self):
        score, ratios = compute_validation_score(self.base, self.base, self.weights)
        self.assertAlmostEqual(score, 1.0, places=5)
        for r in ratios.values():
            self.assertAlmostEqual(r, 1.0, places=5)

    def test_all_improved_yields_score_above_one(self):
        improved = {k: v * 1.05 for k, v in self.base.items()}
        score, _ = compute_validation_score(improved, self.base, self.weights)
        self.assertGreater(score, 1.0)

    def test_single_drop_penalizes(self):
        dropped = dict(self.base)
        dropped["QABF"] = 0.35  # 50% drop
        score_drop, ratios_drop = compute_validation_score(dropped, self.base, self.weights)

        all_up = {k: v * 1.05 for k, v in self.base.items()}
        score_up, _ = compute_validation_score(all_up, self.base, self.weights)

        self.assertLess(score_drop, score_up)
        self.assertLess(ratios_drop["QABF"], 1.0)

    def test_missing_field_raises(self):
        metrics = dict(self.base)
        del metrics["EN"]
        with self.assertRaises(KeyError):
            compute_validation_score(metrics, self.base, self.weights)

    def test_nan_raises(self):
        metrics = dict(self.base)
        metrics["EN"] = float("nan")
        with self.assertRaises(ValueError):
            compute_validation_score(metrics, self.base, self.weights)

    def test_inf_raises(self):
        metrics = dict(self.base)
        metrics["EN"] = float("inf")
        with self.assertRaises(ValueError):
            compute_validation_score(metrics, self.base, self.weights)


# ---------------------------------------------------------------------------
# 8.7 BestCheckpointTests
# ---------------------------------------------------------------------------

class BestCheckpointTests(unittest.TestCase):
    def test_best_logic_mock(self):
        """Mock test: verify is_best flag logic without actual checkpoint I/O."""
        # The logic: is_best = finite_validation and val_score > best_val_score
        # - NaN val_score: is_best = False
        # - Decreasing score: is_best = False
        # - Increasing score (finite): is_best = True

        best_val_score = -float("inf")

        # First valid score -> best
        val_score = 1.0
        finite = math.isfinite(val_score) and True
        is_best = finite and val_score > best_val_score
        self.assertTrue(is_best)
        best_val_score = val_score

        # Lower score -> not best
        val_score = 0.9
        finite = math.isfinite(val_score) and True
        is_best = finite and val_score > best_val_score
        self.assertFalse(is_best)

        # NaN -> not best
        val_score = float("nan")
        finite = math.isfinite(val_score) and True
        is_best = finite and val_score > best_val_score
        self.assertFalse(is_best)

        # Higher score -> best
        val_score = 1.1
        finite = math.isfinite(val_score) and True
        is_best = finite and val_score > best_val_score
        self.assertTrue(is_best)


# ---------------------------------------------------------------------------
# 8.8 ValMetricsRoundTripTests
# ---------------------------------------------------------------------------

class ValMetricsRoundTripTests(unittest.TestCase):
    def test_round_trip_consistency(self):
        """Verify compute_val_metrics matches disk-based metric evaluation."""
        from metric.Metric_torch import (
            EN_function,
            MI_function,
            Qabf_function,
            SCD_function,
            SD_function,
            VIF_function,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create known source images
            h, w = 64, 64
            vis_arr = np.random.RandomState(42).randint(10, 245, (h, w), dtype=np.uint8)
            ir_arr = np.random.RandomState(43).randint(10, 245, (h, w), dtype=np.uint8)

            vis_path = os.path.join(tmpdir, "vis.png")
            ir_path = os.path.join(tmpdir, "ir.png")
            Image.fromarray(vis_arr).save(vis_path)
            Image.fromarray(ir_arr).save(ir_path)

            # Read back via PIL (same as eval_torch.py)
            vis_pil = np.array(Image.open(vis_path).convert("L"), dtype=np.uint8)
            ir_pil = np.array(Image.open(ir_path).convert("L"), dtype=np.uint8)

            # Create fused output as float [0,1]
            fused_arr = np.random.RandomState(44).rand(h, w).astype(np.float32)
            fused_tensor = torch.from_numpy(fused_arr).unsqueeze(0).unsqueeze(0)

            # Build uint8 source tensors (1, 1, H, W)
            vis_u8_tensor = torch.from_numpy(vis_pil.copy()).unsqueeze(0).unsqueeze(0)
            ir_u8_tensor = torch.from_numpy(ir_pil.copy()).unsqueeze(0).unsqueeze(0)

            # Compute via val_metrics
            result = compute_val_metrics(fused_tensor, vis_u8_tensor, ir_u8_tensor)

            # Compute via disk round-trip (PIL -> float tensor/numpy)
            fused_u8 = _quantize_to_uint8(fused_tensor)
            fused_float_disk = torch.from_numpy(fused_u8.astype(np.float32))
            vis_float_disk = torch.from_numpy(vis_pil.astype(np.float32))
            ir_float_disk = torch.from_numpy(ir_pil.astype(np.float32))

            expected = {
                "EN": EN_function(fused_float_disk).item(),
                "SD": SD_function(fused_float_disk).item(),
                "SCD": SCD_function(ir_float_disk, vis_float_disk, fused_float_disk).item(),
                "VIF": VIF_function(ir_float_disk, vis_float_disk, fused_float_disk).item(),
                "MI": MI_function(
                    ir_pil.astype(np.int32), vis_pil.astype(np.int32), fused_u8.astype(np.int32), gray_level=256
                ),
                "QABF": float(Qabf_function(
                    ir_pil.astype(np.float32), vis_pil.astype(np.float32), fused_u8.astype(np.float32)
                )),
            }

            for k in expected:
                self.assertAlmostEqual(result[k], expected[k], delta=1e-5,
                                       msg=f"Mismatch on {k}")

    def test_vis_float32_raises_type_error(self):
        fused = torch.rand(1, 1, 16, 16)
        vis_bad = torch.rand(1, 1, 16, 16)  # float32 not uint8
        ir_good = torch.randint(0, 256, (1, 1, 16, 16), dtype=torch.uint8)
        with self.assertRaises(TypeError):
            compute_val_metrics(fused, vis_bad, ir_good)

    def test_batch_size_not_one_raises(self):
        fused = torch.rand(2, 1, 16, 16)
        vis = torch.randint(0, 256, (2, 1, 16, 16), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            compute_val_metrics(fused, vis, vis)

    def test_channel_not_one_raises(self):
        fused = torch.rand(1, 3, 16, 16)
        vis = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.uint8)
        with self.assertRaises((ValueError, TypeError)):
            compute_val_metrics(fused, vis, vis)

    def test_size_mismatch_raises(self):
        fused = torch.rand(1, 1, 16, 16)
        vis = torch.randint(0, 256, (1, 1, 32, 32), dtype=torch.uint8)
        ir = torch.randint(0, 256, (1, 1, 16, 16), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            compute_val_metrics(fused, vis, ir)


# ---------------------------------------------------------------------------
# ModeRecoveryTests
# ---------------------------------------------------------------------------

class ModeRecoveryTests(unittest.TestCase):
    def test_training_mode_restore(self):
        """Verify try/finally pattern restores training mode."""
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

        modules = [DummyModule(), DummyModule()]
        previous_modes = [m.training for m in modules]
        self.assertTrue(all(previous_modes))

        # Simulate run_validation pattern
        try:
            for m in modules:
                m.eval()
            self.assertTrue(all(not m.training for m in modules))
            raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        finally:
            for m, mode in zip(modules, previous_modes):
                m.train(mode)

        self.assertTrue(all(m.training for m in modules))


if __name__ == "__main__":
    unittest.main()
