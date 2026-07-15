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
