# v11 综合训练优化实施计划（修订版）

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在不修改网络主体结构的前提下，联合优化训练策略（单 AdamW + warmup/cosine）、损失函数（新增 Correlation + LocalContrast）、频率损失渐进 warmup，以及基于固定 20 对验证集的最佳权重选择机制。

**Architecture:** 保持现有 SharedEncoder + DualDomainTextIntentGenerator + TGSFF + FrequencyPyramidAdapter + TextConditionedSpatialFusion + FSRC×3 + FusionDecoder 不变；新增 PairedValidationDataset 用于每个 epoch 验证；train.py 重构训练循环。

**Tech Stack:** PyTorch, h5py, opencv-python, numpy, PIL, clip (openai)

---

## 修改文件总览

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `metric/Metric_torch.py` | 修改 | 修复包导入兼容性（仅导入，不修改公式） |
| 2 | `utils/loss.py` | 修改 | 修正 `cc()` 的零方差反向数值稳定性；新增 `CorrelationConsistencyLoss` 和 `LocalContrastLoss` |
| 3 | `config.py` | 修改 | 追加验证集路径、训练超参数、METRIC_WEIGHTS |
| 4 | `utils/validation_dataset.py` | 新建 | PairedValidationDataset |
| 5 | `utils/training_utils.py` | 新建 | build_lr_scheduler、get_frequency_weight、compute_validation_score、CSV |
| 6 | `utils/val_metrics.py` | 新建 | compute_val_metrics（严格模拟 test.py uint8 量化） |
| 7 | `train.py` | 重写 | 单 AdamW + warmup/cosine + 全局 clip + 新损失 + 验证 + best/latest |
| 8 | `tests/test_training_strategy_and_validation.py` | 新建 | 8 个测试类 |

## 不修改的文件

- `net/` 下所有文件（编码器/解码器/TGSFF/SAM/FSRC/intent 等）
- `test.py`
- `metric/Qabf.py`、`metric/Nabf.py`、`metric/ssim.py`、`metric/eval_torch.py`
- `utils/dataset.py`（H5Dataset）
- `utils/clip_preprocess.py`
- `utils/img_read_save.py`

---

## Task 1: 修复 `metric/Metric_torch.py` 包导入兼容性

**Objective:** 使 `metric/Metric_torch.py` 能被 `utils/val_metrics.py` 从任意目录 import

**Files:**
- Modify: `metric/Metric_torch.py:1-10`

**变更：**

将：
```python
from Qabf import get_Qabf
from Nabf import get_Nabf
from ssim import ssim, ms_ssim
```

改为兼容导入：
```python
try:
    from .Qabf import get_Qabf
    from .Nabf import get_Nabf
    from .ssim import ssim, ms_ssim
except ImportError:
    from Qabf import get_Qabf
    from Nabf import get_Nabf
    from ssim import ssim, ms_ssim
```

**不修改**任何指标函数的公式。

---

## Task 2: 新增 `CorrelationConsistencyLoss` 和 `LocalContrastLoss`

**Objective:** 在 `utils/loss.py` 末尾追加两个新 loss 类

**Files:**
- Modify: `utils/loss.py`（修改 `cc()`，并在文件末尾追加两个新损失类）

**Step 1: 修改 cc() 函数 — 修正零方差输入下的反向数值稳定性**

将 `eps` 从分母外侧移到每个 `sqrt()` 内部，确保 `sqrt(0)` 的反向梯度稳定：

```python
def cc(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(
            f"cc input shapes must match, "
            f"got {img1.shape} and {img2.shape}."
        )
    if img1.ndim != 4:
        raise ValueError(
            "cc expects tensors with shape (N, C, H, W)."
        )

    eps = torch.finfo(img1.dtype).eps
    n, c, _, _ = img1.shape

    img1 = img1.reshape(n, c, -1)
    img2 = img2.reshape(n, c, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    numerator = torch.sum(img1 * img2, dim=-1)

    norm1 = torch.sqrt(torch.sum(img1 ** 2, dim=-1) + eps)
    norm2 = torch.sqrt(torch.sum(img2 ** 2, dim=-1) + eps)
    denominator = norm1 * norm2

    corr = numerator / denominator
    return torch.clamp(corr, -1.0, 1.0).mean()
```

此修改仅影响零方差和极低方差输入的数值稳定性，不改变正常图像上的相关系数计算结果。

**Step 2: 追加 CorrelationConsistencyLoss**

```python
class CorrelationConsistencyLoss(nn.Module):
    """Preserve correlation between fused and source images."""

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        vis = image_vis[:, :1]
        ir = image_ir[:, :1]
        fused_y = fused[:, :1]

        loss_vis = 1.0 - cc(fused_y, vis)
        loss_ir = 1.0 - cc(fused_y, ir)

        loss = loss_vis + loss_ir

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "CorrelationConsistencyLoss produced NaN or Inf."
            )

        return loss
```

**Step 3: 追加 LocalContrastLoss（使用 reflect padding）**

```python
class LocalContrastLoss(nn.Module):
    """Match fused local contrast to a stable source-derived target."""

    def __init__(
        self,
        window_size: int = 7,
        eps: float = 1e-6,
    ):
        super().__init__()

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                "window_size must be a positive odd integer."
            )

        self.window_size = window_size
        self.padding = window_size // 2
        self.eps = eps

    def _local_std(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        # Use reflect padding to avoid edge artifacts from zero-padding
        image_padded = F.pad(
            image,
            [self.padding] * 4,
            mode="reflect",
        )
        image_sq_padded = image_padded * image_padded

        local_mean = F.avg_pool2d(
            image_padded,
            kernel_size=self.window_size,
            stride=1,
            padding=0,
        )

        local_square_mean = F.avg_pool2d(
            image_sq_padded,
            kernel_size=self.window_size,
            stride=1,
            padding=0,
        )

        local_variance = (
            local_square_mean
            - local_mean * local_mean
        ).clamp_min(0.0)

        return torch.sqrt(
            local_variance + self.eps
        )

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        vis = image_vis[:, :1]
        ir = image_ir[:, :1]
        fused_y = fused[:, :1]

        std_vis = self._local_std(vis)
        std_ir = self._local_std(ir)
        std_fused = self._local_std(fused_y)

        std_max = torch.maximum(std_vis, std_ir)
        std_mean = 0.5 * (std_vis + std_ir)

        target_std = (
            0.7 * std_max + 0.3 * std_mean
        ).detach()

        loss = F.l1_loss(std_fused, target_std)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "LocalContrastLoss produced NaN or Inf."
            )

        return loss
```

---

## Task 3: 新增 `PairedValidationDataset`

**Objective:** 创建验证数据集类，加载固定 20 对图像

**Files:**
- Create: `utils/validation_dataset.py`

**要求：**
1. 只扫描合法图片扩展名（`.png` `.jpg` `.jpeg` `.bmp` `.tif` `.tiff`），排除 `Thumbs.db`、隐藏文件、子目录
2. visible/infrared 按文件名严格一一配对
3. `visible_rgb_dir` 不为 None 时，CLIP RGB 从该目录读取；否则从 visible_dir 读取
4. visible_y 始终从 visible_dir 读取
5. IR 尺寸不一致时 resize（与 test.py 行为一致），但打印 warning 日志
7. 不使用任何数据增强、随机裁剪、随机翻转
8. 按确定性顺序排序
9. 返回 `(visible_y, infrared, visible_rgb_raw, metric_vis_u8, metric_ir_u8, filename)`

```python
# -*- coding: utf-8 -*-
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
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
        from PIL import Image

        metric_vis_np = np.array(Image.open(vi_path).convert("L"), dtype=np.uint8)
        metric_ir_np = np.array(Image.open(ir_path).convert("L"), dtype=np.uint8)

        if metric_vis_np.shape != metric_ir_np.shape:
            raise ValueError(
                f"Metric source size mismatch for {filename}: "
                f"VIS={metric_vis_np.shape}, IR={metric_ir_np.shape}"
            )

        # Already uint8 [0,255], no further quantization needed
        metric_vis_u8 = torch.from_numpy(metric_vis_np.copy()).unsqueeze(0)
        metric_ir_u8 = torch.from_numpy(metric_ir_np.copy()).unsqueeze(0)

        return (data_vis_y, data_ir, data_vis_rgb_raw,
                metric_vis_u8, metric_ir_u8, filename)
```

---

## Task 4: 更新 `config.py`

**Objective:** 追加所有新配置项

**Files:**
- Modify: `config.py`

在文件末尾追加（`VAL_BASELINE_CHECKPOINT` 使用占位符，由用户填写）：

```python
# --- Validation (fixed 20 pairs) ---
VAL_VISIBLE_DIR = r"F:\1_paper_pan\2_Image_Fusion\validation_20\visible"
VAL_INFRARED_DIR = r"F:\1_paper_pan\2_Image_Fusion\validation_20\infrared"
VAL_VISIBLE_RGB_DIR = None  # None = reuse visible/ for CLIP RGB
VAL_EXPECTED_PAIRS = 20

VAL_BASELINE_CHECKPOINT = r"<FILL_ME_PATH_TO_FIXED_V11_CHECKPOINT>"
VAL_BASELINE_JSON = r"F:\1_paper_pan\2_Image_Fusion\validation_baseline_metrics.json"

# --- Training hyperparameters ---
BASE_LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)

WARMUP_EPOCHS = 5
FREQUENCY_WARMUP_EPOCHS = 10
GLOBAL_GRAD_CLIP = 0.1

COEFF_FUSION = 1.0
COEFF_SSIM = 2.0
COEFF_FREQ_FINAL = 0.5
COEFF_CORRELATION = 0.2
COEFF_LOCAL_CONTRAST = 0.2

BATCH_SIZE = 8

# --- Validation score weights ---
METRIC_WEIGHTS = {
    "EN": 0.15,
    "SD": 0.10,
    "SCD": 0.20,
    "VIF": 0.20,
    "QABF": 0.25,
    "MI": 0.10,
}
```

---

## Task 5: 新增 `utils/val_metrics.py`

**Objective:** 严格模拟 test.py 的 uint8 量化过程，在内存中计算六项指标

**Files:**
- Create: `utils/val_metrics.py`

**关键：先量化到 uint8，再从 uint8 回转为各指标所需类型，与 `metric/eval_torch.py` 行为完全一致：**
- EN/SD/SCD/VIF：uint8 → float tensor（`eval_torch.py:40-42`）
- MI：uint8 → int32 ndarray（`eval_torch.py:44`）
- QABF：uint8 → float32 ndarray（`eval_torch.py:45`，注意是 float32 不是 float64）

```python
# -*- coding: utf-8 -*-
"""In-memory validation metrics that exactly replicate test.py -> eval_torch.py.

Convention (matching test.py normalize_to_uint8 + eval_torch.py evaluation_one):
  1. Quantize: fused_u8 = round(clamp(x, 0, 1) * 255).to(uint8)
  2. EN, SD, SCD, VIF: uint8 -> float tensor (same as PIL 'L' -> np.array -> tensor)
  3. MI: uint8 -> int32 ndarray
  4. QABF: uint8 -> float32 ndarray (eval_torch.py uses float32, not float64)
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

    # Step 3: Compute metrics
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
```

---

## Task 6: 新增 `utils/training_utils.py`

**Objective:** 集中训练辅助函数

**Files:**
- Create: `utils/training_utils.py`

```python
# -*- coding: utf-8 -*-
import csv
import math
import os

import torch


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
    warmup_epochs: int = 5,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if warmup_epochs >= num_epochs:
        raise ValueError("warmup_epochs must be smaller than num_epochs.")

    min_ratio = min_lr / base_lr

    def lr_lambda(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)

        cosine_epochs = num_epochs - warmup_epochs
        progress = float(epoch_index - warmup_epochs + 1) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_value

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Frequency warmup
# ---------------------------------------------------------------------------

def get_frequency_weight(
    epoch_index: int,
    warmup_epochs: int = 10,
    final_weight: float = 0.5,
) -> float:
    if warmup_epochs <= 1:
        return float(final_weight)
    if epoch_index >= warmup_epochs - 1:
        return float(final_weight)
    progress = float(epoch_index) / float(warmup_epochs - 1)
    return float(final_weight) * progress


# ---------------------------------------------------------------------------
# Validation score
# ---------------------------------------------------------------------------

def compute_validation_score(
    metrics: dict,
    baseline_metrics: dict,
    metric_weights: dict,
):
    ratios = {}

    for name in metric_weights:
        if name not in metrics:
            raise KeyError(f"Missing validation metric: {name}")
        if name not in baseline_metrics:
            raise KeyError(f"Missing baseline metric: {name}")

        current_value = float(metrics[name])
        baseline_value = float(baseline_metrics[name])

        if not math.isfinite(current_value):
            raise ValueError(f"Validation metric {name} is not finite.")
        if not math.isfinite(baseline_value):
            raise ValueError(f"Baseline metric {name} is not finite.")

        baseline_value = max(baseline_value, 1e-8)
        ratios[name] = max(current_value / baseline_value, 1e-8)

    geometric_score = math.exp(
        sum(
            metric_weights[name] * math.log(ratios[name])
            for name in metric_weights
        )
    )

    worst_ratio = min(ratios.values())
    validation_score = 0.8 * geometric_score + 0.2 * worst_ratio

    return validation_score, ratios


# ---------------------------------------------------------------------------
# Training history CSV
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "epoch",
    "learning_rate",
    "avg_grad_norm",
    "train_total",
    "train_fusion",
    "train_weighted_fusion",
    "train_ssim",
    "train_weighted_ssim",
    "train_frequency",
    "frequency_weight",
    "train_weighted_frequency",
    "train_correlation",
    "train_weighted_correlation",
    "train_local_contrast",
    "train_weighted_local_contrast",
    "val_EN",
    "val_SD",
    "val_SCD",
    "val_VIF",
    "val_QABF",
    "val_MI",
    "val_score",
    "worst_ratio",
    "is_best",
    "best_epoch_so_far",
    "best_score_so_far",
]


def init_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)


def append_csv(csv_path: str, **kwargs) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [kwargs.get(col, "") for col in _CSV_HEADER]
        writer.writerow(row)
```

---

## Task 7: 重构 `train.py`

**Objective:** 完整重写训练入口

**Files:**
- Modify: `train.py`（完全重写）

完整代码见 [附录 A](#appendix-a)。

关键设计要点：

1. **`build_model` 只定义一次**：train.py 复用同一个 `build_model`，baseline 和训练模型都调用它
2. **单 AdamW**：收集所有模块 `requires_grad=True` 参数，id 去重检查
3. **LR 日志在 epoch 开始时读取**：`current_lr = optimizer.param_groups[0]["lr"]` 在 epoch 循环最前面，训练结束后 `scheduler.step()`
4. **平均 grad_norm**：累计 `grad_norm_sum` 和 `num_batches`，epoch 结束时 `avg_grad_norm = sum / count`
5. **每个 batch 直接累计 weighted loss**（不是 epoch 平均后再乘系数）
6. **Baseline 严格校验**：所有 9 个 key 必须存在 + 复用 test.py 的 `validate_checkpoint_metadata`
7. **Baseline JSON 严格校验**：6 个字段存在 + 有限 + >0 + validation filenames 一致性验证 + validation_count + baseline_checkpoint_path 完整路径对比 + checkpoint 文件 size/mtime
8. **Baseline JSON 保存额外元数据**：`validation_filenames`、`validation_count`、`baseline_checkpoint_path`、`baseline_checkpoint_size`、`baseline_checkpoint_mtime`、baseline metrics
9. **`run_validation` 用 try/finally 恢复训练模式**：
   ```python
   previous_modes = [m.training for m in modules]
   try:
       for m in modules: m.eval()
       ...  # validation
   finally:
       for m, mode in zip(modules, previous_modes):
           m.train(mode)
   ```
10. **Baseline 模块完整释放**：
    ```python
    del baseline_modules
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    ```
11. **原子写入 checkpoint**：先写 `.tmp`，成功再 `os.replace()`
12. **best 更新前检查 `math.isfinite(val_score)`**：非有限值时记录错误、不覆盖 best

---

## Task 8: 新增测试

**Objective:** 覆盖所有新功能

**Files:**
- Create: `tests/test_training_strategy_and_validation.py`

**8 个测试类：**

### 8.1 CorrelationConsistencyLossTests
- 随机输入 → 标量 + finite + backward + fused.grad 存在且有限
- 常数输入（`torch.full((1,1,8,8), 0.5)`）→ 前向 finite → **反向传播成功，fused.grad 存在且全有限**（验证 `sqrt(0)` 梯度稳定性）
- 小尺寸输入（1, 1, 4, 4）→ backward 成功，fused.grad 存在且有限

### 8.2 LocalContrastLossTests
- 随机输入 → 标量 + finite + backward + fused.grad 存在且有限
- 常数输入 → finite
- 8×8 小尺寸
- window_size 验证（偶数 → ValueError，0 → ValueError）

### 8.3 FrequencyWarmupTests
- epoch_index=0 → 0.0
- epoch_index=1 → 0.055555...（使用 `math.isclose(rel_tol=1e-5)`）
- epoch_index=8 → 0.444444...
- epoch_index=9 → 0.5
- epoch_index=10 → 0.5
- epoch_index=49 → 0.5

### 8.4 LRSchedulerTests
```python
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
```

检查：
- Epoch 1～5 学习率递增（epoch 5 达到 1e-4）
- Epoch 6 开始下降
- Epoch 50 接近 `1e-6`（`abs(lr - 1e-6) < 1e-7`）
- 所有学习率均为有限正数

不要只连续调用 `scheduler.step()`，避免 PyTorch 跳过首个学习率并产生调用顺序警告。

### 8.5 ValidationDatasetTests
创建临时目录：
- 20 对正确数据 → 加载成功，排序稳定
- 缺一张 infrared → ValueError
- visible/infrared 文件名不一致 → ValueError
- 数量 19 → ValueError
- 数量 21 → ValueError
- 包含非图片文件（Thumbs.db）→ 不计入
- VIS/IR PIL 灰度尺寸一致时正常返回
- `metric_vis_u8.dtype == torch.uint8`
- `metric_ir_u8.dtype == torch.uint8`
- 返回 shape 为 `(1, H, W)`
- VIS/IR 指标源图尺寸不一致时直接 ValueError
- 网络输入 IR resize 不影响指标源图的严格尺寸检查

### 8.6 ValidationScoreTests
- metrics == baseline → score ≈ 1.0（`abs(score - 1.0) < 1e-6`）
- 所有提高 5% → score > 1.0
- 单项下降 50% → worst_ratio < 1.0，score < 全提高情况
- 缺少字段 → KeyError
- NaN → ValueError
- Inf → ValueError

### 8.7 BestCheckpointTests
mock 方式：
- val_score 提高 → 覆盖 best
- val_score 降低 → 不覆盖 best
- val_score NaN → 不覆盖 best
- latest 每轮更新
- best checkpoint 包含 epoch + val_metrics + val_ratios

### 8.8 ValMetricsRoundTripTests
**不导入 `metric/eval_torch.py`**（其内部使用 `from Metric_torch import *`，非 package 导入会失败）。

测试流程：
1. 创建 VIS、IR 临时图像，使用 `PIL.Image.open(...).convert("L")` 读取
2. 转换为 `shape=(1,1,H,W)` 的 `torch.uint8` 作为 `metric_vis_u8`、`metric_ir_u8`
3. fused 使用 `[0,1]` float tensor
4. 调用 `compute_val_metrics(fused, metric_vis_u8, metric_ir_u8)`
5. 与磁盘读取后直接调用 `metric.Metric_torch` 六项函数的结果比较
6. 六项指标误差小于 `1e-5`

错误输入测试：
- `vis_u8` 为 float32 → TypeError
- `ir_u8` 为 float32 → TypeError
- `batch_size != 1` → ValueError
- `channel != 1` → ValueError
- fused/VIS/IR 尺寸不一致 → ValueError

---

## 附录 A: 完整 `train.py`

```python
# -*- coding: utf-8 -*-
import datetime
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BASE_LR,
    BATCH_SIZE,
    BETAS,
    CHECKPOINT_TAG,
    CLIP_DOWNLOAD_ROOT,
    CLIP_MODEL_NAME,
    COEFF_CORRELATION,
    COEFF_FREQ_FINAL,
    COEFF_FUSION,
    COEFF_LOCAL_CONTRAST,
    COEFF_SSIM,
    FREQUENCY_WARMUP_EPOCHS,
    GLOBAL_GRAD_CLIP,
    METRIC_WEIGHTS,
    MIN_LR,
    MODEL_DIRECTORY,
    MODEL_VERSION,
    TRAIN_H5_PATH,
    USE_CLIP_IMAGE_QUERY,
    VAL_BASELINE_CHECKPOINT,
    VAL_BASELINE_JSON,
    VAL_EXPECTED_PAIRS,
    VAL_INFRARED_DIR,
    VAL_VISIBLE_DIR,
    VAL_VISIBLE_RGB_DIR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from net.Network import (
    DualDomainTextIntentGenerator,
    DualStreamIntentMLP,
    FSRC,
    FrequencyPyramidAdapter,
    FusionDecoder,
    SharedEncoder,
    TextConditionedSpatialFusion,
)
from net.frequency_fusion import TGSFF
from utils.clip_preprocess import preprocess_clip_rgb
from utils.dataset import H5Dataset
from utils.loss import (
    CorrelationConsistencyLoss,
    FrequencyConsistencyLoss,
    Fusionloss,
    LocalContrastLoss,
    SimpleSSIMLoss,
)
from utils.training_utils import (
    append_csv,
    build_lr_scheduler,
    compute_validation_score,
    get_frequency_weight,
    init_csv,
)
from utils.val_metrics import compute_val_metrics
from utils.validation_dataset import PairedValidationDataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ---------------------------------------------------------------------------
# Single build_model (shared by training and baseline)
# ---------------------------------------------------------------------------

def build_model(device: torch.device):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    if USE_CLIP_IMAGE_QUERY:
        intent_core = DualDomainTextIntentGenerator(
            intent_dim=64, clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
        )
    else:
        intent_core = DualStreamIntentMLP(
            channels=64, intent_dim=64, hidden_dim=256,
            clip_model_name=CLIP_MODEL_NAME, clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
        )
    intent_generator = nn.DataParallel(intent_core).to(device)
    frequency_fusion = nn.DataParallel(TGSFF(
        in_channels=64, patch_size=4, amp_topk_ratio=0.30, phase_topk_ratio=0.35,
        token_embed_dim=128, num_heads=4, return_aux=True, routing_temperature=0.25,
    )).to(device)
    frequency_pyramid_adapter = nn.DataParallel(FrequencyPyramidAdapter(channels=64)).to(device)
    spatial_fusion = nn.DataParallel(TextConditionedSpatialFusion(
        channels=64, intent_dim=64, num_heads=4, ffn_expansion_factor=2.0,
        init_res_scale=0.05, use_freq_context=False,
    )).to(device)
    fsrc_l1 = nn.DataParallel(FSRC(channels=64)).to(device)
    fsrc_l2 = nn.DataParallel(FSRC(channels=64)).to(device)
    fsrc_l3 = nn.DataParallel(FSRC(channels=64)).to(device)
    fusion_decoder = nn.DataParallel(FusionDecoder(
        channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    return encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_MODEL_KEYS = (
    'shared_encoder', 'intent_generator', 'frequency_fusion', 'frequency_pyramid_adapter',
    'spatial_fusion', 'fsrc_l1', 'fsrc_l2', 'fsrc_l3', 'fusion_decoder',
)


def save_checkpoint(
    path,
    modules,
    optimizer=None, scheduler=None,
    epoch=None, val_score=None, val_metrics=None, val_ratios=None,
    is_best=False,
):
    checkpoint = {
        'model_version': MODEL_VERSION,
        'use_clip_image_query': USE_CLIP_IMAGE_QUERY,
        'intent_generator_type': type(unwrap(modules[1])).__name__,
    }
    for key, module in zip(_MODEL_KEYS, modules):
        checkpoint[key] = module.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = int(epoch)
    if val_score is not None:
        checkpoint['val_score'] = float(val_score)
    if val_metrics is not None:
        checkpoint['val_metrics'] = dict(val_metrics)
    if val_ratios is not None:
        checkpoint['val_ratios'] = dict(val_ratios)
    checkpoint['is_best'] = bool(is_best)

    # Atomic write
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def validate_checkpoint_metadata(checkpoint, intent_generator):
    """Reuse test.py metadata validation."""
    expected_type = type(unwrap(intent_generator)).__name__
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise RuntimeError('Checkpoint model version mismatch.')
    if checkpoint.get('use_clip_image_query') != USE_CLIP_IMAGE_QUERY:
        raise RuntimeError('Checkpoint query variant mismatch.')
    if checkpoint.get('intent_generator_type') != expected_type:
        raise RuntimeError('Checkpoint intent generator type mismatch.')


def load_modules_from_checkpoint(modules, checkpoint):
    """Strict load with all keys required."""
    for key, module in zip(_MODEL_KEYS, modules):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {key}")
        module.load_state_dict(checkpoint[key], strict=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(modules, valloader, device):
    """Run inference on all validation pairs and return averaged metrics."""
    previous_modes = [m.training for m in modules]

    try:
        for m in modules:
            m.eval()

        all_metrics = {k: [] for k in METRIC_WEIGHTS}

        with torch.no_grad():
            for (data_vis, data_ir, data_vis_rgb_raw,
                 metric_vis_u8, metric_ir_u8, _) in valloader:
                data_vis = data_vis.to(device)
                data_ir = data_ir.to(device)
                data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))

                vis_spa, vis_freq, _ = modules[0](data_vis)
                ir_spa, ir_freq, _ = modules[0](data_ir)
                i_deg, i_fus, _ = modules[1](data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
                fused_freq, _ = modules[2](vis_freq, ir_freq, frequency_intent=i_deg)
                _, spatial_pyramid, _ = modules[4](vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
                freq_pyramid = modules[3](fused_freq, target_pyramid=spatial_pyramid)
                d_l1, _ = modules[5](freq_pyramid['l1'], spatial_pyramid['l1'])
                d_l2, _ = modules[6](freq_pyramid['l2'], spatial_pyramid['l2'])
                d_l3, _ = modules[7](freq_pyramid['l3'], spatial_pyramid['l3'])
                fused_image, _ = modules[8](d_l1, d_l2, d_l3)

                metrics = compute_val_metrics(
                    fused=fused_image,
                    vis_u8=metric_vis_u8,
                    ir_u8=metric_ir_u8,
                )
                for k in METRIC_WEIGHTS:
                    all_metrics[k].append(metrics[k])

    finally:
        for m, mode in zip(modules, previous_modes):
            m.train(mode)

    avg_metrics = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
    return avg_metrics


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def validate_baseline_json(json_path, val_filenames):
    """Validate baseline JSON has all required fields and matches current val set."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check required metric fields
    for k in METRIC_WEIGHTS:
        if k not in data:
            raise KeyError(f"Baseline JSON missing metric key: {k}")
        v = float(data[k])
        if not math.isfinite(v):
            raise ValueError(f"Baseline metric {k} is not finite: {v}")
        if v <= 0:
            raise ValueError(f"Baseline metric {k} is not positive: {v}")

    # Check validation filenames consistency
    saved_filenames = data.get("validation_filenames")
    if saved_filenames is None:
        raise ValueError("Baseline JSON missing validation_filenames field.")
    if saved_filenames != val_filenames:
        raise ValueError(
            "Validation filenames changed since baseline was generated. "
            "Delete the baseline JSON and re-run to regenerate."
        )

    # Check validation_count
    saved_count = data.get("validation_count")
    if saved_count is None:
        raise ValueError("Baseline JSON missing validation_count field.")
    if saved_count != len(val_filenames):
        raise ValueError(
            "Baseline validation_count does not match the current validation set."
        )

    # Check baseline checkpoint path consistency
    saved_checkpoint = data.get("baseline_checkpoint_path")
    if saved_checkpoint is None:
        raise ValueError("Baseline JSON missing baseline_checkpoint_path.")
    if os.path.abspath(saved_checkpoint) != os.path.abspath(VAL_BASELINE_CHECKPOINT):
        raise ValueError(
            "Baseline checkpoint path changed. "
            "Delete the old baseline JSON and regenerate it."
        )

    # Check that the referenced checkpoint still exists with same size/mtime
    if not os.path.isfile(saved_checkpoint):
        raise ValueError(
            f"Baseline checkpoint no longer exists: {saved_checkpoint}"
        )
    saved_size = data.get("baseline_checkpoint_size")
    if saved_size is None:
        raise ValueError(
            "Baseline JSON missing baseline_checkpoint_size."
        )
    saved_mtime = data.get("baseline_checkpoint_mtime")
    if saved_mtime is None:
        raise ValueError(
            "Baseline JSON missing baseline_checkpoint_mtime."
        )
    current_stat = os.stat(saved_checkpoint)
    if int(saved_size) != int(current_stat.st_size):
        raise ValueError(
            "Baseline checkpoint size changed. "
            "Delete the old baseline JSON and regenerate it."
        )
    if abs(float(saved_mtime) - float(current_stat.st_mtime)) > 1.0:
        raise ValueError(
            "Baseline checkpoint mtime changed. "
            "Delete the old baseline JSON and regenerate it."
        )

    return data


def generate_baseline(device, val_filenames):
    """Generate baseline metrics JSON from the fixed checkpoint."""
    if not os.path.isfile(VAL_BASELINE_CHECKPOINT):
        raise FileNotFoundError(f"Baseline checkpoint not found: {VAL_BASELINE_CHECKPOINT}")

    # Build separate baseline modules
    baseline_modules = build_model(device)

    try:
        # Load checkpoint to CPU first to reduce peak GPU memory
        checkpoint = torch.load(VAL_BASELINE_CHECKPOINT, map_location="cpu")

        # Strict metadata validation (reuse test.py logic)
        validate_checkpoint_metadata(checkpoint, baseline_modules[1])

        # Strict key validation (all 9 keys must exist) and load to modules on GPU
        load_modules_from_checkpoint(baseline_modules, checkpoint)

        # Release CPU checkpoint immediately after loading
        del checkpoint
        gc.collect()

        val_dataset = PairedValidationDataset(
            visible_dir=VAL_VISIBLE_DIR,
            infrared_dir=VAL_INFRARED_DIR,
            visible_rgb_dir=VAL_VISIBLE_RGB_DIR,
            expected_pairs=VAL_EXPECTED_PAIRS,
        )
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=torch.cuda.is_available())

        metrics = run_validation(baseline_modules, valloader, device)

        checkpoint_stat = os.stat(VAL_BASELINE_CHECKPOINT)
        baseline_data = {
            **metrics,
            "validation_filenames": val_filenames,
            "validation_count": len(val_filenames),
            "baseline_checkpoint_path": VAL_BASELINE_CHECKPOINT,
            "baseline_checkpoint_size": checkpoint_stat.st_size,
            "baseline_checkpoint_mtime": checkpoint_stat.st_mtime,
        }

        baseline_json_dir = os.path.dirname(VAL_BASELINE_JSON) or "."
        os.makedirs(baseline_json_dir, exist_ok=True)
        with open(VAL_BASELINE_JSON, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"Baseline saved to: {VAL_BASELINE_JSON}")
        print(f"Baseline metrics: {metrics}")

    finally:
        del baseline_modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training forward pass
# ---------------------------------------------------------------------------

def training_forward(modules, data_vis, data_ir, data_vis_clip):
    """Shared forward pass for training. Returns fused_image."""
    shared_encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, \
        spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder = modules

    vis_spa, vis_freq, _ = shared_encoder(data_vis)
    ir_spa, ir_freq, _ = shared_encoder(data_ir)
    i_deg, i_fus, _ = intent_generator(data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
    fused_freq, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=i_deg)
    _, spatial_pyramid, _ = spatial_fusion(vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
    freq_pyramid = frequency_pyramid_adapter(fused_freq, target_pyramid=spatial_pyramid)
    d_l1, _ = fsrc_l1(freq_pyramid['l1'], spatial_pyramid['l1'])
    d_l2, _ = fsrc_l2(freq_pyramid['l2'], spatial_pyramid['l2'])
    d_l3, _ = fsrc_l3(freq_pyramid['l3'], spatial_pyramid['l3'])
    fused_image, _ = fusion_decoder(d_l1, d_l2, d_l3)

    return fused_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Validation dataset (used for both baseline and per-epoch validation) ---
    val_dataset = PairedValidationDataset(
        visible_dir=VAL_VISIBLE_DIR,
        infrared_dir=VAL_INFRARED_DIR,
        visible_rgb_dir=VAL_VISIBLE_RGB_DIR,
        expected_pairs=VAL_EXPECTED_PAIRS,
    )
    val_filenames = list(val_dataset.filenames)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                           pin_memory=torch.cuda.is_available())

    # --- Baseline ---
    if not os.path.isfile(VAL_BASELINE_JSON):
        print("Baseline JSON not found, generating from checkpoint...")
        generate_baseline(device, val_filenames)

    baseline_data = validate_baseline_json(VAL_BASELINE_JSON, val_filenames)
    baseline_metrics = {k: baseline_data[k] for k in METRIC_WEIGHTS}
    print(f"Baseline metrics: {baseline_metrics}")

    # --- Loss criteria ---
    criteria_fusion = Fusionloss().to(device)
    criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
    criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)
    criteria_correlation = CorrelationConsistencyLoss().to(device)
    criteria_local_contrast = LocalContrastLoss(window_size=7, eps=1e-6).to(device)

    num_epochs = 50

    # --- Build model ---
    modules = build_model(device)

    # --- Single AdamW optimizer ---
    trainable_params = []
    for module in modules:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])

    param_ids = [id(p) for p in trainable_params]
    if len(param_ids) != len(set(param_ids)):
        raise RuntimeError("Duplicate trainable parameters were found across modules.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=BASE_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR,
        min_lr=MIN_LR,
    )

    # --- Data ---
    trainloader = DataLoader(H5Dataset(TRAIN_H5_PATH), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Checkpoint paths ---
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    best_checkpoint_path = os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_best.pth')
    latest_checkpoint_path = os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_latest.pth')

    # --- CSV ---
    csv_path = os.path.join(MODEL_DIRECTORY, f'training_history_{timestamp}.csv')
    init_csv(csv_path)

    # --- Best tracking ---
    best_val_score = -float("inf")
    best_epoch = -1
    best_metrics = None
    best_ratios = None

    torch.backends.cudnn.benchmark = True
    previous_time = time.time()

    for epoch in range(num_epochs):
        # Read LR at START of epoch (before training)
        current_lr = optimizer.param_groups[0]["lr"]

        # === Training ===
        for module in modules:
            module.train()

        # Accumulators for per-batch weighted losses
        epoch_total = 0.0
        epoch_weighted_fusion = 0.0
        epoch_weighted_ssim = 0.0
        epoch_weighted_freq = 0.0
        epoch_weighted_corr = 0.0
        epoch_weighted_local_ctr = 0.0
        epoch_fusion_raw = 0.0
        epoch_ssim_raw = 0.0
        epoch_freq_raw = 0.0
        epoch_corr_raw = 0.0
        epoch_local_ctr_raw = 0.0
        grad_norm_sum = 0.0
        num_batches = 0

        for index, (data_vis, data_ir, data_vis_rgb_raw) in enumerate(trainloader):
            data_vis, data_ir = data_vis.to(device), data_ir.to(device)
            data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))

            optimizer.zero_grad(set_to_none=True)

            fused_image = training_forward(modules, data_vis, data_ir, data_vis_clip)

            fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
            ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
            freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
            correlation_loss = criteria_correlation(data_vis, data_ir, fused_image)
            local_contrast_loss = criteria_local_contrast(data_vis, data_ir, fused_image)

            current_freq_weight = get_frequency_weight(
                epoch_index=epoch,
                warmup_epochs=FREQUENCY_WARMUP_EPOCHS,
                final_weight=COEFF_FREQ_FINAL,
            )

            weighted_fusion = COEFF_FUSION * fusion_loss
            weighted_ssim = COEFF_SSIM * ssim_loss
            weighted_freq = current_freq_weight * freq_loss
            weighted_corr = COEFF_CORRELATION * correlation_loss
            weighted_local_contrast = COEFF_LOCAL_CONTRAST * local_contrast_loss

            loss = weighted_fusion + weighted_ssim + weighted_freq + weighted_corr + weighted_local_contrast
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GLOBAL_GRAD_CLIP, norm_type=2.0)
            optimizer.step()

            # Accumulate per-batch values
            epoch_total += loss.item()
            epoch_weighted_fusion += weighted_fusion.item()
            epoch_weighted_ssim += weighted_ssim.item()
            epoch_weighted_freq += weighted_freq.item()
            epoch_weighted_corr += weighted_corr.item()
            epoch_weighted_local_ctr += weighted_local_contrast.item()
            epoch_fusion_raw += fusion_loss.item()
            epoch_ssim_raw += ssim_loss.item()
            epoch_freq_raw += freq_loss.item()
            epoch_corr_raw += correlation_loss.item()
            epoch_local_ctr_raw += local_contrast_loss.item()
            grad_norm_sum += float(grad_norm) if not isinstance(grad_norm, torch.Tensor) else grad_norm.item()
            num_batches += 1

            batches_done = epoch * len(trainloader) + index
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - previous_time))
            previous_time = time.time()
            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] [fusion: %.4f] [ssim: %.4f] [freq: %.4f] [corr: %.4f] [ctr: %.4f] ETA: %.10s' % (
                    epoch + 1, num_epochs, index + 1, len(trainloader), loss.item(),
                    fusion_loss.item(), ssim_loss.item(), freq_loss.item(),
                    correlation_loss.item(), local_contrast_loss.item(), time_left,
                )
            )

        # scheduler step AFTER training of this epoch
        scheduler.step()

        # Compute epoch averages
        n = max(num_batches, 1)
        avg_total = epoch_total / n
        avg_fusion = epoch_fusion_raw / n
        avg_weighted_fusion = epoch_weighted_fusion / n
        avg_ssim = epoch_ssim_raw / n
        avg_weighted_ssim = epoch_weighted_ssim / n
        avg_freq = epoch_freq_raw / n
        avg_weighted_freq = epoch_weighted_freq / n
        avg_corr = epoch_corr_raw / n
        avg_weighted_corr = epoch_weighted_corr / n
        avg_local_ctr = epoch_local_ctr_raw / n
        avg_weighted_local_ctr = epoch_weighted_local_ctr / n
        avg_grad_norm = grad_norm_sum / n
        current_freq_w = get_frequency_weight(epoch_index=epoch, warmup_epochs=FREQUENCY_WARMUP_EPOCHS, final_weight=COEFF_FREQ_FINAL)

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] "
              f"lr={current_lr:.6e} "
              f"grad_norm={avg_grad_norm:.4f} "
              f"total={avg_total:.6f} "
              f"fusion={avg_fusion:.6f} w_fus={avg_weighted_fusion:.6f} "
              f"ssim={avg_ssim:.6f} w_ssim={avg_weighted_ssim:.6f} "
              f"freq={avg_freq:.6f} fw={current_freq_w:.6f} w_freq={avg_weighted_freq:.6f} "
              f"corr={avg_corr:.6f} w_corr={avg_weighted_corr:.6f} "
              f"ctr={avg_local_ctr:.6f} w_ctr={avg_weighted_local_ctr:.6f}")

        # === Validation ===
        val_metrics = run_validation(modules, valloader, device)

        # Compute validation score with exception handling for non-finite metrics
        try:
            val_score, val_ratios = compute_validation_score(
                metrics=val_metrics,
                baseline_metrics=baseline_metrics,
                metric_weights=METRIC_WEIGHTS,
            )
        except (ValueError, KeyError) as error:
            print(f"[WARNING] Validation scoring failed: {error}")
            val_score = float("nan")
            val_ratios = {name: float("nan") for name in METRIC_WEIGHTS}

        worst_ratio = min(val_ratios.values())

        print(f"[Validation] EN={val_metrics['EN']:.4f} SD={val_metrics['SD']:.4f} "
              f"SCD={val_metrics['SCD']:.4f} VIF={val_metrics['VIF']:.4f} "
              f"QABF={val_metrics['QABF']:.4f} MI={val_metrics['MI']:.4f} "
              f"score={val_score:.6f} worst_ratio={worst_ratio:.4f}")

        # --- Determine best ---
        finite_validation = (
            math.isfinite(val_score)
            and all(math.isfinite(float(v)) for v in val_metrics.values())
        )

        is_best = False
        if finite_validation and val_score > best_val_score:
            is_best = True
            best_val_score = val_score
            best_epoch = epoch + 1
            best_metrics = dict(val_metrics)
            best_ratios = dict(val_ratios)

            save_checkpoint(
                path=best_checkpoint_path,
                modules=modules,
                optimizer=optimizer, scheduler=scheduler,
                epoch=best_epoch, val_score=best_val_score,
                val_metrics=best_metrics, val_ratios=best_ratios,
                is_best=True,
            )
            print(f"[Best] epoch={best_epoch} score={best_val_score:.6f} "
                  f"metrics={best_metrics} ratios={best_ratios}")
            print(f"[Best] saved to {best_checkpoint_path}")
        elif not finite_validation:
            print(f"[WARNING] Validation metrics non-finite (val_score={val_score}), "
                  f"skipping best checkpoint update.")

        # Always save latest
        save_checkpoint(
            path=latest_checkpoint_path,
            modules=modules,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch + 1, val_score=val_score,
            val_metrics=val_metrics, val_ratios=val_ratios,
            is_best=is_best,
        )

        # --- CSV ---
        append_csv(
            csv_path,
            epoch=epoch + 1,
            learning_rate=current_lr,
            avg_grad_norm=avg_grad_norm,
            train_total=avg_total,
            train_fusion=avg_fusion,
            train_weighted_fusion=avg_weighted_fusion,
            train_ssim=avg_ssim,
            train_weighted_ssim=avg_weighted_ssim,
            train_frequency=avg_freq,
            frequency_weight=current_freq_w,
            train_weighted_frequency=avg_weighted_freq,
            train_correlation=avg_corr,
            train_weighted_correlation=avg_weighted_corr,
            train_local_contrast=avg_local_ctr,
            train_weighted_local_contrast=avg_weighted_local_ctr,
            val_EN=val_metrics['EN'],
            val_SD=val_metrics['SD'],
            val_SCD=val_metrics['SCD'],
            val_VIF=val_metrics['VIF'],
            val_QABF=val_metrics['QABF'],
            val_MI=val_metrics['MI'],
            val_score=val_score,
            worst_ratio=worst_ratio,
            is_best=1 if is_best else 0,
            best_epoch_so_far=best_epoch,
            best_score_so_far=best_val_score,
        )

    # === Training completed ===
    print(f"\n{'='*60}")
    print(f"Training completed")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation score: {best_val_score:.6f}")
    print(f"Best validation metrics: {best_metrics}")
    print(f"Best validation ratios: {best_ratios}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Training history: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
```

---

## 修正对照表

| # | 修正点 | 落实位置 |
|---|--------|---------|
| 1 | cc() 仅修正零方差反向梯度稳定性 | Task 2 — `eps` 移到每个 `sqrt()` 内部；不改变相关系数目标 |
| 2 | 验证指标先量化 uint8 再计算 | `utils/val_metrics.py` — `_quantize_to_uint8` → float/int32/float32 |
| 3 | Metric_torch.py 兼容导入 | Task 1 — try/except 导入 Qabf/Nabf/ssim |
| 4 | visible_rgb_dir 真正被使用 | `utils/validation_dataset.py` — `rgb_path = os.path.join(...)`；合法扩展名过滤 |
| 5 | IR resize 打印 warning | `utils/validation_dataset.py` — `logging.warning(...)` |
| 6 | LR 在 epoch 开始时读取 | `train.py` — `current_lr` 在循环最前；`scheduler.step()` 在最后 |
| 7 | grad_norm 累计平均 | `train.py` — `grad_norm_sum` + `num_batches` |
| 8 | Baseline checkpoint 严格校验 | `load_modules_from_checkpoint` — 9 key 全部必须存在 |
| 9 | Baseline JSON 严格校验 + size/mtime | `validate_baseline_json` — 6 字段 + finite + >0 + filenames + count + checkpoint path + size + mtime |
| 10 | LocalContrastLoss reflect padding | `utils/loss.py` — `F.pad(..., mode="reflect")` + `avg_pool2d(..., padding=0)` |
| 11 | run_validation try/finally 恢复模式 | `train.py` — `previous_modes` + `try/finally` |
| 12 | Baseline modules 完整释放 | `generate_baseline` — `del` + `gc.collect()` + `torch.cuda.empty_cache()` |
| 13 | build_model 不复制 | `train.py` 单一定义，训练和 baseline 共用 |
| 14 | 原子写入 checkpoint | `save_checkpoint` — `.tmp` + `os.replace()` |
| 15 | best 用 finite_validation 判断 | `train.py` — `finite_validation = isfinite(score) and all(isfinite(metrics))` |
| 16 | CSV 记录真实加权项 | `train.py` — 每个 batch 直接累计 weighted loss |
| 17 | 不修改 test.py | 不操作 `test.py` |
| 18 | 总体设计不变 | 50 epoch、单 AdamW、warmup/cosine、clip 0.1、新损失权重 0.2、频率 warmup、best/latest、无 EMA |
| — | **三轮修正** | |
| 6 | Baseline checkpoint 加载到 CPU + 释放 | `generate_baseline` — `map_location="cpu"`，load 后 `del checkpoint; gc.collect()` |
| 7 | Baseline JSON size/mtime 必填字段 | `validate_baseline_json` — `saved_size is None` / `saved_mtime is None` 直接报错；`int()`/`float()` 转换后用 `>`/`!=` 比较 |
| 8 | 验证指标量化使用 NumPy | `utils/val_metrics.py` — `np.squeeze(array, axis=(0,1))` + `np.round().astype(np.uint8)`；不用 `torch.squeeze()` |
| 9 | 测试文案修正 + LR 测试按真实训练循环 | CorrelationConsistencyLossTests "不变 NaN" → "不出现 NaN 或 Inf"；LRSchedulerTests 按 optimizer.step() + scheduler.step() 顺序循环 |
| — | **四轮修正** | |
| 10 | 验证源图用 PIL convert("L") | `PairedValidationDataset.__getitem__` 新增 `metric_vis_u8`/`metric_ir_u8`；`compute_val_metrics` 接口改为 `(fused, vis_u8, ir_u8)`；`run_validation` 解包 6 元素 |
| 11 | LRSchedulerTests 显式 parameter | `parameter = torch.nn.Parameter(torch.zeros(1))` 显式定义 |
| 12 | baseline JSON dirname 空字符串处理 | `os.path.dirname(...) or "."` |
| 13 | 测试更新 | ValidationDatasetTests 增加 uint8 dtype/shape/尺寸校验；ValMetricsRoundTripTests 改为 PIL→uint8 直接传入 + 错误输入测试 |

---

## 验收标准

- [ ] `cc()` 仅修改零方差输入下的反向数值稳定性，不改变相关系数目标
- [ ] 训练集：原训练 H5，完整保留
- [ ] 验证集：固定 20 对
- [ ] 训练轮数：50
- [ ] 优化器：单个 AdamW（参数去重）
- [ ] 学习率：epoch 开始时读取日志，scheduler.step() 在 epoch 训练后
- [ ] 梯度裁剪：全局 `max_norm=0.1`，epoch 平均梯度范数
- [ ] 总损失：`1.0*L_fusion + 2.0*L_ssim + λ(epoch)*L_freq + 0.2*L_corr + 0.2*L_local_contrast`
- [ ] 频率权重：epoch 1→0.0 ... epoch 10→0.5 ... epoch 50→0.5
- [ ] 验证频率：每个 epoch 一次
- [ ] 验证指标严格模拟 test.py uint8 量化
- [ ] 验证后训练模式正确恢复（try/finally）
- [ ] 最佳权重：综合分数提高时覆盖 best.pth（原子写入）
- [ ] latest 权重：每个 epoch 覆盖（原子写入）
- [ ] val_score 非有限时不覆盖 best
- [ ] Baseline 校验：9 key 全部存在 + metadata + filenames 一致
- [ ] EMA：不使用
- [ ] 网络主体：未修改
- [ ] FrequencyConsistencyLoss：未修改
- [ ] SemanticAffineModulation：未修改
- [ ] gamma_proj/beta_proj：保留
- [ ] test.py：未修改
- [ ] Metric_torch.py：仅修改导入，不修改公式
- [ ] 所有单元测试通过
