# -*- coding: utf-8 -*-
import csv
import math
import os

import torch

from config import MODEL_VERSION, USE_CLIP_IMAGE_QUERY


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_MODEL_KEYS = (
    'shared_encoder', 'intent_generator', 'frequency_fusion', 'frequency_pyramid_adapter',
    'spatial_fusion', 'fsrc_l1', 'fsrc_l2', 'fsrc_l3', 'fusion_decoder',
)


def unwrap(module):
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def _require_module_count(modules) -> None:
    module_count = len(modules)
    if module_count != len(_MODEL_KEYS):
        raise ValueError(
            f"Expected exactly {len(_MODEL_KEYS)} modules, got {module_count}."
        )


def save_checkpoint(
    path,
    modules,
    optimizer=None, scheduler=None,
    epoch=None, val_score=None, val_metrics=None, val_ratios=None,
    is_best=False,
):
    _require_module_count(modules)
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

    parent_directory = os.path.dirname(path)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)
    tmp_path = path + '.tmp'
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def validate_checkpoint_metadata(checkpoint, intent_generator):
    """Validate metadata required for a compatible strict checkpoint load."""
    expected_type = type(unwrap(intent_generator)).__name__
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise RuntimeError('Checkpoint model version mismatch.')
    if checkpoint.get('use_clip_image_query') != USE_CLIP_IMAGE_QUERY:
        raise RuntimeError('Checkpoint query variant mismatch.')
    if checkpoint.get('intent_generator_type') != expected_type:
        raise RuntimeError('Checkpoint intent generator type mismatch.')


def load_modules_from_checkpoint(modules, checkpoint):
    """Strictly load every required model state from a checkpoint."""
    _require_module_count(modules)
    for key, module in zip(_MODEL_KEYS, modules):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {key}")
        module.load_state_dict(checkpoint[key], strict=True)


def should_update_best(val_score: float, best_val_score: float, val_metrics: dict) -> bool:
    if not val_metrics:
        return False

    current_score = float(val_score)
    current_best = float(best_val_score)

    if not math.isfinite(current_score):
        return False
    if math.isnan(current_best):
        return False
    if not all(math.isfinite(float(value)) for value in val_metrics.values()):
        return False

    return current_score > current_best


def should_run_validation(
    epoch_number: int,
    validation_start_epoch: int,
) -> bool:
    """Return whether validation should run for a one-based epoch number."""
    if epoch_number < 1:
        raise ValueError("epoch_number must use one-based indexing and be >= 1.")
    if validation_start_epoch < 1:
        raise ValueError("validation_start_epoch must be >= 1.")
    return epoch_number >= validation_start_epoch


def validate_positive_baseline_metrics(metrics: dict, metric_names) -> dict:
    """Validate the fixed validation baseline required for ratio scoring."""
    validated = {}
    for name in metric_names:
        if name not in metrics:
            raise KeyError(f"Validation baseline missing metric: {name}")

        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(
                f"Validation baseline metric {name} is not finite: {value}"
            )
        if value <= 0:
            raise ValueError(
                f"Validation baseline metric {name} must be positive for ratio "
                f"scoring, got {value}"
            )
        validated[name] = value

    return validated


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
