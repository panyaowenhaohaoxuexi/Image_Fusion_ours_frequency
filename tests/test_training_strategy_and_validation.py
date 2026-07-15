# -*- coding: utf-8 -*-
import importlib
import json
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.loss import CorrelationConsistencyLoss, LocalContrastLoss, cc
from utils.training_utils import (
    _MODEL_KEYS,
    build_lr_scheduler,
    compute_validation_score,
    get_frequency_weight,
    load_modules_from_checkpoint,
    save_checkpoint,
    should_update_best,
    validate_checkpoint_metadata,
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

    def test_window_seven_accepts_four_by_four_input(self):
        criterion = LocalContrastLoss(window_size=7)
        image = torch.randn(1, 1, 4, 4)
        self.assertTrue(torch.isfinite(criterion._local_std(image)).all().item())

    def test_window_seven_rejects_reflect_padding_on_three_by_three_input(self):
        criterion = LocalContrastLoss(window_size=7)
        with self.assertRaisesRegex(
            ValueError, "^Input spatial size \(3, 3\) must be larger than reflect padding 3\.$"
        ):
            criterion._local_std(torch.randn(1, 1, 3, 3))

    def test_local_std_rejects_non_four_dimensional_input(self):
        criterion = LocalContrastLoss(window_size=7)
        with self.assertRaisesRegex(
            ValueError, "^LocalContrastLoss expects input with shape \(N, C, H, W\)\.$"
        ):
            criterion._local_std(torch.randn(1, 4, 4))


class CorrelationCoefficientTests(unittest.TestCase):
    def test_float32_and_float64_inputs_are_finite(self):
        for dtype in (torch.float32, torch.float64):
            left = torch.randn(1, 1, 8, 8, dtype=dtype)
            right = torch.randn(1, 1, 8, 8, dtype=dtype)
            self.assertTrue(torch.isfinite(cc(left, right)).item())

    def test_integer_inputs_are_rejected(self):
        right = torch.rand(1, 1, 8, 8)
        for dtype in (torch.int32, torch.uint8):
            with self.assertRaisesRegex(TypeError, "floating-point"):
                cc(torch.ones(1, 1, 8, 8, dtype=dtype), right)
            with self.assertRaisesRegex(TypeError, "floating-point"):
                cc(right, torch.ones(1, 1, 8, 8, dtype=dtype))

    def test_shape_and_dimension_errors_remain_value_errors(self):
        with self.assertRaises(ValueError):
            cc(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 7, 8))
        with self.assertRaisesRegex(ValueError, "shape \(N, C, H, W\)"):
            cc(torch.rand(1, 8, 8), torch.rand(1, 8, 8))


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
            self.assertEqual(metric_vis_u8.ndim, 3)
            self.assertEqual(metric_vis_u8.shape[0], 1)
            self.assertEqual(metric_ir_u8.shape[0], 1)
            self.assertEqual(metric_vis_u8.shape[-2:], (32, 32))
            self.assertEqual(metric_ir_u8.shape[-2:], (32, 32))

    def test_textured_pair_dataloader_metrics_are_four_dimensional_and_finite(self):
        """Metric source images gain the batch dimension exactly once in DataLoader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(123)
            vis_dir = os.path.join(tmpdir, "visible")
            ir_dir = os.path.join(tmpdir, "infrared")
            os.makedirs(vis_dir)
            os.makedirs(ir_dir)
            height, width = 64, 64
            yy, xx = np.mgrid[:height, :width]
            texture = ((xx * 11 + yy * 17) % 256).astype(np.uint8)
            vis_rgb = np.stack((texture, rng.randint(0, 256, (height, width), dtype=np.uint8),
                                np.roll(texture, 7, axis=1)), axis=-1)
            ir = ((texture.astype(np.uint16) + rng.randint(0, 64, (height, width))) % 256).astype(np.uint8)
            Image.fromarray(vis_rgb).save(os.path.join(vis_dir, "0000.png"))
            Image.fromarray(ir).save(os.path.join(ir_dir, "0000.png"))

            dataset = PairedValidationDataset(vis_dir, ir_dir, expected_pairs=1)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            _, _, _, metric_vis_u8, metric_ir_u8, _ = next(iter(loader))
            self.assertEqual(metric_vis_u8.shape, (1, 1, height, width))
            self.assertEqual(metric_ir_u8.shape, (1, 1, height, width))
            self.assertEqual(metric_vis_u8.dtype, torch.uint8)
            self.assertEqual(metric_ir_u8.dtype, torch.uint8)

            fused = torch.from_numpy(rng.uniform(0.05, 0.95, (1, 1, height, width)).astype(np.float32))
            metrics = compute_val_metrics(fused, metric_vis_u8, metric_ir_u8)
            self.assertEqual(set(metrics), {"EN", "SD", "SCD", "VIF", "QABF", "MI"})
            self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))

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
    valid_metrics = {"EN": 7.0, "SD": 40.0, "SCD": 1.6, "VIF": 1.0, "QABF": 0.7, "MI": 3.6}

    def test_should_update_best_rejects_invalid_or_non_improving_scores(self):
        self.assertFalse(should_update_best(1.0, 0.0, {}))
        self.assertTrue(should_update_best(1.0, 0.0, self.valid_metrics))
        self.assertFalse(should_update_best(1.0, 1.0, self.valid_metrics))
        self.assertFalse(should_update_best(0.9, 1.0, self.valid_metrics))
        for value in (float("nan"), float("inf"), -float("inf")):
            self.assertFalse(should_update_best(value, 0.0, self.valid_metrics))
        self.assertFalse(should_update_best(1.0, float("nan"), self.valid_metrics))
        self.assertFalse(should_update_best(1.0, float("inf"), self.valid_metrics))
        self.assertTrue(should_update_best(1.0, -float("inf"), self.valid_metrics))
        for invalid_metric in (float("nan"), float("inf"), -float("inf")):
            metrics = dict(self.valid_metrics, EN=invalid_metric)
            self.assertFalse(should_update_best(1.0, 0.0, metrics))


# ---------------------------------------------------------------------------
# 8.8 ValMetricsRoundTripTests
# ---------------------------------------------------------------------------

class ValMetricsRoundTripTests(unittest.TestCase):
    def _assert_finite_metrics(self, fused_device: torch.device):
        rng = np.random.RandomState(99)
        image = torch.from_numpy(rng.uniform(0.05, 0.95, (1, 1, 64, 64)).astype(np.float32)).to(fused_device)
        vis = torch.from_numpy(rng.randint(0, 256, (1, 1, 64, 64), dtype=np.uint8))
        ir = torch.from_numpy(rng.randint(0, 256, (1, 1, 64, 64), dtype=np.uint8))
        metrics = compute_val_metrics(image, vis, ir)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in metrics.values()))

    def test_cpu_metrics_are_finite(self):
        self._assert_finite_metrics(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_metrics_are_finite(self):
        self._assert_finite_metrics(torch.device("cuda"))

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

class CheckpointUtilitiesTests(unittest.TestCase):
    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

    def _modules(self):
        self.assertEqual(len(_MODEL_KEYS), 9)
        return [self.DummyModule() for _ in _MODEL_KEYS]

    def _optimizer_and_scheduler(self, modules):
        optimizer = torch.optim.SGD(
            [parameter for module in modules for parameter in module.parameters()], lr=0.1
        )
        return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def _checkpoint_metadata(self, intent_generator):
        from config import MODEL_VERSION, USE_CLIP_IMAGE_QUERY
        return {
            "model_version": MODEL_VERSION,
            "use_clip_image_query": USE_CLIP_IMAGE_QUERY,
            "intent_generator_type": type(intent_generator).__name__,
        }

    def test_save_checkpoint_is_atomic_complete_and_overwritable(self):
        modules = self._modules()
        optimizer, scheduler = self._optimizer_and_scheduler(modules)
        metrics = {"EN": 7.0, "SD": 40.0, "SCD": 1.6, "VIF": 1.0, "QABF": 0.7, "MI": 3.6}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "latest.pth")
            save_checkpoint(path, modules, optimizer, scheduler, epoch=1, val_score=1.0,
                            val_metrics=metrics, val_ratios={"EN": 1.0}, is_best=False)
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".tmp"))
            first = torch.load(path, map_location="cpu")
            self.assertTrue(set(_MODEL_KEYS).issubset(first))
            self.assertTrue({"model_version", "use_clip_image_query", "intent_generator_type", "epoch",
                             "val_score", "val_metrics", "val_ratios", "optimizer", "scheduler", "is_best"}.issubset(first))
            self.assertEqual(first["epoch"], 1)
            save_checkpoint(path, modules, optimizer, scheduler, epoch=2, val_score=1.1,
                            val_metrics=metrics, val_ratios={"EN": 1.1}, is_best=False)
            self.assertFalse(os.path.exists(path + ".tmp"))
            self.assertEqual(torch.load(path, map_location="cpu")["epoch"], 2)

    def test_save_checkpoint_rejects_wrong_module_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            for count in (len(_MODEL_KEYS) - 1, len(_MODEL_KEYS) + 1):
                with self.assertRaisesRegex(
                    ValueError, f"Expected exactly {len(_MODEL_KEYS)} modules, got {count}\."
                ):
                    save_checkpoint(path, self._modules()[:count] if count < len(_MODEL_KEYS)
                                    else self._modules() + [self.DummyModule()])

    def test_save_checkpoint_cleans_temp_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            original = b"existing checkpoint"
            with open(path, "wb") as file:
                file.write(original)
            with mock.patch("utils.training_utils.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    save_checkpoint(path, self._modules())
            with open(path, "rb") as file:
                self.assertEqual(file.read(), original)
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_load_modules_requires_complete_strict_matching_states(self):
        source = self._modules()
        checkpoint = {key: module.state_dict() for key, module in zip(_MODEL_KEYS, source)}
        restored = self._modules()
        load_modules_from_checkpoint(restored, checkpoint)
        for expected, actual in zip(source, restored):
            for expected_parameter, actual_parameter in zip(expected.parameters(), actual.parameters()):
                self.assertTrue(torch.equal(expected_parameter, actual_parameter))

        missing = dict(checkpoint)
        del missing[_MODEL_KEYS[0]]
        with self.assertRaises(KeyError):
            load_modules_from_checkpoint(self._modules(), missing)

        wrong_name = dict(checkpoint)
        wrong_name[_MODEL_KEYS[0]] = {"unknown.weight": torch.zeros(2, 2)}
        with self.assertRaises(RuntimeError):
            load_modules_from_checkpoint(self._modules(), wrong_name)

        wrong_shape = dict(checkpoint)
        wrong_shape[_MODEL_KEYS[0]] = {"layer.weight": torch.zeros(3, 2), "layer.bias": torch.zeros(2)}
        with self.assertRaises(RuntimeError):
            load_modules_from_checkpoint(self._modules(), wrong_shape)

    def test_load_modules_rejects_wrong_destination_module_counts(self):
        checkpoint = {
            key: module.state_dict()
            for key, module in zip(_MODEL_KEYS, self._modules())
        }
        for count in (len(_MODEL_KEYS) - 1, len(_MODEL_KEYS) + 1):
            modules = self._modules()[:count] if count < len(_MODEL_KEYS) else self._modules() + [self.DummyModule()]
            with self.assertRaisesRegex(
                ValueError, f"Expected exactly {len(_MODEL_KEYS)} modules, got {count}\."
            ):
                load_modules_from_checkpoint(modules, checkpoint)

    def test_validate_checkpoint_metadata_rejects_wrong_or_missing_fields(self):
        intent_generator = self.DummyModule()
        valid = self._checkpoint_metadata(intent_generator)
        validate_checkpoint_metadata(valid, intent_generator)
        for key, bad_value in (
            ("model_version", "wrong-version"),
            ("use_clip_image_query", not valid["use_clip_image_query"]),
            ("intent_generator_type", "OtherIntent"),
        ):
            bad = dict(valid, **{key: bad_value})
            with self.assertRaises(RuntimeError):
                validate_checkpoint_metadata(bad, intent_generator)
        for key in tuple(valid):
            missing = dict(valid)
            del missing[key]
            with self.assertRaises(RuntimeError):
                validate_checkpoint_metadata(missing, intent_generator)

    def test_only_true_best_decision_replaces_best_file(self):
        modules = self._modules()
        metrics = {"EN": 7.0, "SD": 40.0, "SCD": 1.6, "VIF": 1.0, "QABF": 0.7, "MI": 3.6}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "best.pth")
            save_checkpoint(path, modules, epoch=1, val_score=1.0, val_metrics=metrics, is_best=True)
            original_epoch = torch.load(path, map_location="cpu")["epoch"]
            rejected = [
                (1.0, 1.0, metrics), (0.9, 1.0, metrics), (float("nan"), 1.0, metrics),
                (float("inf"), 1.0, metrics), (1.1, float("inf"), metrics),
                (1.1, 1.0, dict(metrics, EN=float("nan"))),
            ]
            for score, best_score, candidate_metrics in rejected:
                if should_update_best(score, best_score, candidate_metrics):
                    save_checkpoint(path, modules, epoch=2, val_score=score,
                                    val_metrics=candidate_metrics, is_best=True)
            self.assertEqual(torch.load(path, map_location="cpu")["epoch"], original_epoch)
            self.assertTrue(should_update_best(1.1, 1.0, metrics))
            save_checkpoint(path, modules, epoch=2, val_score=1.1, val_metrics=metrics, is_best=True)
            self.assertEqual(torch.load(path, map_location="cpu")["epoch"], 2)


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


class RunValidationTests(unittest.TestCase):
    def test_empty_loader_raises_approved_runtime_error(self):
        missing = object()
        previous_train = sys.modules.get("train", missing)
        previous_clip = sys.modules.get("clip", missing)
        previous_net_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "net" or name.startswith("net.")
        }
        sys.modules.pop("train", None)
        clip_stub = types.ModuleType("clip")
        sys.modules["clip"] = clip_stub
        try:
            train_module = importlib.import_module("train")
            empty_loader = torch.utils.data.DataLoader([], batch_size=1)
            modules = [nn.Identity() for _ in range(9)]
            with self.assertRaisesRegex(
                RuntimeError, r"^Validation loader produced no metric results\.$"
            ):
                train_module.run_validation(modules, empty_loader, torch.device("cpu"))
        finally:
            for name in list(sys.modules):
                if (name == "net" or name.startswith("net.")) and name not in previous_net_modules:
                    sys.modules.pop(name, None)
            sys.modules.update(previous_net_modules)
            if previous_train is missing:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = previous_train
            if previous_clip is missing:
                sys.modules.pop("clip", None)
            else:
                sys.modules["clip"] = previous_clip
        self.assertEqual(
            {
                name: module
                for name, module in sys.modules.items()
                if name == "net" or name.startswith("net.")
            },
            previous_net_modules,
        )
        self.assertIs(sys.modules.get("train", missing), previous_train)
        self.assertIs(sys.modules.get("clip", missing), previous_clip)


if __name__ == "__main__":
    unittest.main()
