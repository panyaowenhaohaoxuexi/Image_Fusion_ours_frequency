# -*- coding: utf-8 -*-
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
