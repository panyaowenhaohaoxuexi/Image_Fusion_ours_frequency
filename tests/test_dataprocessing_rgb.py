import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from skimage.io import imsave

from tools.dataprocessing import build_h5


class BuildH5Tests(unittest.TestCase):
    def test_build_h5_keeps_float32_rgb_patches_aligned_with_y_and_ir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_dir, vi_dir, output_dir = root / "ir", root / "vi", root / "output"
            ir_dir.mkdir()
            vi_dir.mkdir()
            ir_image = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1))
            rgb_image = np.stack([
                np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1)),
                np.tile(np.array([255, 0, 255, 0], dtype=np.uint8), (4, 1)),
                np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1)),
            ], axis=-1)
            imsave(ir_dir / "sample.png", ir_image)
            imsave(vi_dir / "sample.png", rgb_image)

            build_h5(str(ir_dir), str(vi_dir), str(output_dir), "sample", img_size=2, stride=2)

            with h5py.File(output_dir / "sample_imgsize_2_stride_2.h5", "r") as h5_file:
                self.assertEqual(set(h5_file.keys()), {"ir_patchs", "vis_patchs", "vis_rgb_patchs"})
                self.assertEqual(set(h5_file["ir_patchs"].keys()), set(h5_file["vis_patchs"].keys()))
                self.assertEqual(set(h5_file["vis_patchs"].keys()), set(h5_file["vis_rgb_patchs"].keys()))
                rgb_patch = h5_file["vis_rgb_patchs"]["0"]
                self.assertEqual(rgb_patch.shape, (3, 2, 2))
                self.assertEqual(rgb_patch.dtype, np.dtype(np.float32))
                self.assertTrue(np.all((rgb_patch[...] >= 0.0) & (rgb_patch[...] <= 1.0)))

    def test_build_h5_can_write_a_new_rgb_suffix_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_dir, vi_dir, output_dir = root / "ir", root / "vi", root / "output"
            ir_dir.mkdir()
            vi_dir.mkdir()
            checkerboard = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1))
            imsave(ir_dir / "sample.png", checkerboard)
            imsave(vi_dir / "sample.png", np.stack([checkerboard, checkerboard, checkerboard], axis=-1))

            build_h5(str(ir_dir), str(vi_dir), str(output_dir), "sample", img_size=2, stride=2, output_suffix="_rgb")

            self.assertTrue((output_dir / "sample_imgsize_2_stride_2_rgb.h5").is_file())

    def test_build_h5_rejects_unmatched_ir_and_visible_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_dir, vi_dir, output_dir = root / "ir", root / "vi", root / "output"
            ir_dir.mkdir()
            vi_dir.mkdir()
            checkerboard = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1))
            imsave(ir_dir / "ir_only.png", checkerboard)
            imsave(vi_dir / "vis_only.png", np.stack([checkerboard, checkerboard, checkerboard], axis=-1))

            with self.assertRaisesRegex(ValueError, "file names do not match"):
                build_h5(str(ir_dir), str(vi_dir), str(output_dir), "sample", img_size=2, stride=2)
