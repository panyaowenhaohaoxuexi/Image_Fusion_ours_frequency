# -*- coding: utf-8 -*-
"""In-memory validation metrics that exactly replicate test.py -> eval_torch.py.

Convention (matching test.py normalize_to_uint8 + eval_torch.py evaluation_one):
  1. Fused: model output [0,1] -> quantize to uint8 matching test.py.
  2. VIS/IR sources: PIL convert("L") -> uint8 (passed directly, no re-quantization).
  3. EN, SD, SCD, VIF: uint8 -> float tensor.
  4. MI: uint8 -> int32 ndarray.
  5. QABF: uint8 -> float32 ndarray (eval_torch.py uses float32, not float64).
"""

import numpy as np
import torch

from metric.Metric_torch import (
    EN_function,
    MI_function,
    Qabf_function,
    SCD_function,
    SD_function,
    VIF_function,
)


def _quantize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Reproduce test.py normalize_to_uint8: round(clamp(x,0,1)*255).astype(uint8).

    Uses NumPy round/astype for exact consistency with test.py.
    Uses explicit axis squeeze rather than generic squeeze() to avoid
    collapsing spatial dims when H=1 or W=1.
    """
    if tensor.ndim != 4:
        raise ValueError("Expected tensor with shape (B, C, H, W).")
    if tensor.shape[0] != 1:
        raise ValueError("Validation metric computation requires batch_size=1.")
    if tensor.shape[1] != 1:
        raise ValueError("Validation metric computation requires one-channel images.")

    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .cpu()
        .numpy()
    )
    # Squeeze B and C dims explicitly (not .squeeze())
    array = np.squeeze(array, axis=(0, 1))
    return np.round(array).astype(np.uint8)


def _validate_source_uint8(
    tensor: torch.Tensor,
    name: str,
) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W).")
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} requires batch_size=1.")
    if tensor.shape[1] != 1:
        raise ValueError(f"{name} requires one channel.")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be torch.uint8, got {tensor.dtype}.")
    array = tensor.detach().cpu().numpy()
    return np.squeeze(array, axis=(0, 1))


def compute_val_metrics(
    fused: torch.Tensor,
    vis_u8: torch.Tensor,
    ir_u8: torch.Tensor,
) -> dict:
    """Compute EN, SD, SCD, VIF, QABF, MI for a single fused image.

    Args:
        fused: (1, 1, H, W) torch float tensor in [0, 1] (model output).
               Quantized to uint8 matching test.py normalize_to_uint8.
        vis_u8: (1, 1, H, W) torch.uint8 tensor in [0, 255] from PIL convert("L").
        ir_u8:  (1, 1, H, W) torch.uint8 tensor in [0, 255] from PIL convert("L").

    Returns:
        dict with keys EN, SD, SCD, VIF, QABF, MI as Python floats.
    """
    # Step 1: Quantize fused to uint8 (exactly matching test.py normalize_to_uint8)
    fused_u8 = _quantize_to_uint8(fused)

    # Step 2: Validate source uint8 images
    vis_u8 = _validate_source_uint8(vis_u8, "vis_u8")
    ir_u8 = _validate_source_uint8(ir_u8, "ir_u8")

    # Step 3: Check size consistency
    if fused_u8.shape != vis_u8.shape:
        raise ValueError(
            f"Fused/VIS metric size mismatch: {fused_u8.shape} vs {vis_u8.shape}"
        )
    if fused_u8.shape != ir_u8.shape:
        raise ValueError(
            f"Fused/IR metric size mismatch: {fused_u8.shape} vs {ir_u8.shape}"
        )

    # Step 4: Convert back to types matching eval_torch.py evaluation_one:
    #   line 40-42: float tensor (from np.array(PIL 'L') -> tensor)
    #   line 44:    int32 ndarray
    #   line 45:    float32 ndarray (NOT float64!)
    fused_float = torch.from_numpy(fused_u8.astype(np.float32))
    vis_float = torch.from_numpy(vis_u8.astype(np.float32))
    ir_float = torch.from_numpy(ir_u8.astype(np.float32))

    fused_int32 = fused_u8.astype(np.int32)
    vis_int32 = vis_u8.astype(np.int32)
    ir_int32 = ir_u8.astype(np.int32)

    fused_f32 = fused_u8.astype(np.float32)
    vis_f32 = vis_u8.astype(np.float32)
    ir_f32 = ir_u8.astype(np.float32)

    # Step 5: Compute metrics
    EN = EN_function(fused_float).item()
    SD = SD_function(fused_float).item()
    SCD = SCD_function(ir_float, vis_float, fused_float).item()
    VIF = VIF_function(ir_float, vis_float, fused_float).item()
    MI = MI_function(ir_int32, vis_int32, fused_int32, gray_level=256)
    QABF = float(Qabf_function(ir_f32, vis_f32, fused_f32))

    return {
        "EN": float(EN),
        "SD": float(SD),
        "SCD": float(SCD),
        "VIF": float(VIF),
        "QABF": QABF,
        "MI": float(MI),
    }
