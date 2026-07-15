# -*- coding: utf-8 -*-
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_image_files(directory: str) -> list:
    """Return sorted list of image filenames (excluding non-image files)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = []
    for name in os.listdir(directory):
        if name.startswith("."):
            continue
        if name.lower() == "thumbs.db":
            continue
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in _VALID_EXTENSIONS:
            files.append(name)
    return sorted(files)


class PairedValidationDataset(Dataset):
    """Fixed validation dataset with 20 paired VIS/IR images."""

    def __init__(
        self,
        visible_dir: str,
        infrared_dir: str,
        visible_rgb_dir: Optional[str] = None,
        expected_pairs: int = 20,
    ):
        super().__init__()

        vis_files = _list_image_files(visible_dir)
        ir_files = _list_image_files(infrared_dir)

        if len(vis_files) != expected_pairs or len(ir_files) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} image pairs, got VIS={len(vis_files)} IR={len(ir_files)}"
            )

        if vis_files != ir_files:
            raise ValueError(
                f"VIS and IR filenames do not match. "
                f"VIS extra: {set(vis_files) - set(ir_files)}, "
                f"IR extra: {set(ir_files) - set(vis_files)}"
            )

        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.visible_rgb_dir = visible_rgb_dir
        self.filenames = vis_files

        if visible_rgb_dir is not None:
            rgb_files = _list_image_files(visible_rgb_dir)
            if rgb_files != vis_files:
                raise ValueError(
                    f"VIS_RGB filenames do not match VIS/IR filenames. "
                    f"RGB extra: {set(rgb_files) - set(vis_files)}, "
                    f"VIS extra: {set(vis_files) - set(rgb_files)}"
                )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        vi_path = os.path.join(self.visible_dir, filename)
        ir_path = os.path.join(self.infrared_dir, filename)

        if self.visible_rgb_dir is not None:
            rgb_path = os.path.join(self.visible_rgb_dir, filename)
        else:
            rgb_path = vi_path

        # --- Network inputs (same as test.py: OpenCV BGR) ---
        data_vis_bgr = cv2.imread(vi_path, cv2.IMREAD_COLOR)
        data_ir_np = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if data_vis_bgr is None:
            raise RuntimeError(f"Failed to read VIS image: {vi_path}")
        if data_ir_np is None:
            raise RuntimeError(f"Failed to read IR image: {ir_path}")

        height, width = data_vis_bgr.shape[:2]
        if data_ir_np.shape[:2] != (height, width):
            logging.warning(
                "Validation pair resized: %s, IR %s -> VIS %s",
                filename, data_ir_np.shape[:2], (height, width),
            )
            data_ir_np = cv2.resize(data_ir_np, (width, height), interpolation=cv2.INTER_LINEAR)

        # VIS grayscale from Y channel (same as test.py)
        data_vis_y_np, _, _ = cv2.split(cv2.cvtColor(data_vis_bgr, cv2.COLOR_BGR2YCrCb))
        data_vis_y = torch.from_numpy(data_vis_y_np[None].astype(np.float32) / 255.0)

        # IR grayscale
        data_ir = torch.from_numpy(data_ir_np[None].astype(np.float32) / 255.0)

        # VIS RGB for CLIP: read from rgb_path (may differ from vi_path)
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        data_vis_rgb_raw = torch.from_numpy(
            cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # --- Metric source images (same as eval_torch.py: PIL convert("L")) ---
        metric_vis_np = np.array(Image.open(vi_path).convert("L"), dtype=np.uint8)
        metric_ir_np = np.array(Image.open(ir_path).convert("L"), dtype=np.uint8)

        if metric_vis_np.shape != metric_ir_np.shape:
            raise ValueError(
                f"Metric source size mismatch for {filename}: "
                f"VIS={metric_vis_np.shape}, IR={metric_ir_np.shape}"
            )

        # Already uint8 [0,255], no further quantization needed
        metric_vis_u8 = torch.from_numpy(metric_vis_np.copy()).unsqueeze(0).unsqueeze(0)
        metric_ir_u8 = torch.from_numpy(metric_ir_np.copy()).unsqueeze(0).unsqueeze(0)

        return (data_vis_y, data_ir, data_vis_rgb_raw,
                metric_vis_u8, metric_ir_u8, filename)
