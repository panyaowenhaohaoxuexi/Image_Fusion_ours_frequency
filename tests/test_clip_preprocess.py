import unittest

import torch

from utils.clip_preprocess import CLIP_MEAN, CLIP_STD, preprocess_clip_rgb


class ClipPreprocessTests(unittest.TestCase):
    def test_normalizes_then_zero_pads_non_square_input(self):
        rgb = torch.zeros(1, 3, 200, 100, dtype=torch.float32)

        output = preprocess_clip_rgb(rgb)

        self.assertEqual(tuple(output.shape), (1, 3, 224, 224))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.allclose(output[:, :, :, :56], torch.zeros_like(output[:, :, :, :56])))
        expected_pixel = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        expected_value = (torch.zeros_like(expected_pixel) - expected_pixel) / torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(output[:, :, :, 56:57], expected_value))

    def test_accepts_unbatched_rgb(self):
        output = preprocess_clip_rgb(torch.rand(3, 32, 48))

        self.assertEqual(tuple(output.shape), (1, 3, 224, 224))
        self.assertEqual(output.dtype, torch.float32)
