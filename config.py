# -*- coding: utf-8 -*-
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
USE_CLIP_IMAGE_QUERY = True
MODEL_VERSION = "v11"
QUERY_VARIANT = "clip_image_query" if USE_CLIP_IMAGE_QUERY else "shared_encoder_mlp_query"
CHECKPOINT_TAG = f"{MODEL_VERSION}_{QUERY_VARIANT}"

CLIP_MODEL_NAME = str(PROJECT_ROOT / "weight" / "clip" / "ViT-B-32.pt")
CLIP_DOWNLOAD_ROOT = str(PROJECT_ROOT / "weight" / "clip")
TRAIN_H5_PATH = r"F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS_train_imgsize_128_stride_200_rgb.h5"
MODEL_DIRECTORY = str(PROJECT_ROOT / "models")

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


def validate_runtime_config() -> None:
    """Fail before training when required local inputs are unavailable."""
    errors = []

    if not os.path.isfile(TRAIN_H5_PATH):
        errors.append(f"TRAIN_H5_PATH must be an existing file: {TRAIN_H5_PATH}")
    if not os.path.isdir(VAL_VISIBLE_DIR):
        errors.append(f"VAL_VISIBLE_DIR must be an existing directory: {VAL_VISIBLE_DIR}")
    if not os.path.isdir(VAL_INFRARED_DIR):
        errors.append(f"VAL_INFRARED_DIR must be an existing directory: {VAL_INFRARED_DIR}")
    if VAL_VISIBLE_RGB_DIR is not None and not os.path.isdir(VAL_VISIBLE_RGB_DIR):
        errors.append(
            f"VAL_VISIBLE_RGB_DIR must be an existing directory when configured: {VAL_VISIBLE_RGB_DIR}"
        )
    if '<FILL_ME' in VAL_BASELINE_CHECKPOINT:
        errors.append("VAL_BASELINE_CHECKPOINT still contains a <FILL_ME...> placeholder.")
    elif not os.path.isfile(VAL_BASELINE_CHECKPOINT):
        errors.append(f"VAL_BASELINE_CHECKPOINT must be an existing file: {VAL_BASELINE_CHECKPOINT}")
    if not os.path.isfile(CLIP_MODEL_NAME):
        errors.append(f"CLIP_MODEL_NAME must be an existing file: {CLIP_MODEL_NAME}")

    baseline_parent = os.path.dirname(VAL_BASELINE_JSON)
    if baseline_parent:
        os.makedirs(baseline_parent, exist_ok=True)
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    if errors:
        raise RuntimeError("Invalid runtime configuration:\n- " + "\n- ".join(errors))
