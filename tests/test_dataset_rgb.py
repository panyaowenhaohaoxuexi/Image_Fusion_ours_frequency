import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.dataset import H5Dataset


class H5DatasetTests(unittest.TestCase):
    def test_returns_aligned_rgb_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patches.h5"
            with h5py.File(path, "w") as h5_file:
                for group_name, array in {
                    "ir_patchs": np.full((1, 2, 2), 0.1, dtype=np.float32),
                    "vis_patchs": np.full((1, 2, 2), 0.2, dtype=np.float32),
                    "vis_rgb_patchs": np.full((3, 2, 2), 0.3, dtype=np.float32),
                }.items():
                    h5_file.create_group(group_name).create_dataset("0", data=array)

            vis, ir, rgb = H5Dataset(str(path))[0]

        self.assertEqual(tuple(vis.shape), (1, 2, 2))
        self.assertEqual(tuple(ir.shape), (1, 2, 2))
        self.assertEqual(tuple(rgb.shape), (3, 2, 2))
        self.assertEqual(rgb.dtype, torch.float32)
