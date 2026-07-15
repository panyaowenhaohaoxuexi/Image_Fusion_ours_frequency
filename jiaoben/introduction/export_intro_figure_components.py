# -*- coding: utf-8 -*-
"""
Export component images for the Introduction figure.

Goal:
- You handle the final layout yourself.
- This script only exports the image pieces:
    1) visible_marked.png
    2) infrared_marked.png
    3) without_area1_feature.png
    4) without_area2_feature.png
    5) with_area1_feature.png
    6) with_area2_feature.png
- It also saves raw numpy arrays for later custom plotting.

Recommended semantics:
- area1 = degradation region  -> visualize frequency-branch feature map
- area2 = fusion region       -> visualize spatial-branch weight map

Run this script from the repository root.
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from test import build_model, validate_checkpoint_metadata, _load_state
from utils.clip_preprocess import preprocess_clip_rgb


# ---------------------------
# Basic utilities
# ---------------------------
def parse_box(values):
    if len(values) != 4:
        raise ValueError("Box must be x1 y1 x2 y2")
    x1, y1, x2, y2 = map(int, values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Require x2>x1 and y2>y1")
    return [x1, y1, x2, y2]


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def normalize_map(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    x = np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)
    return x


def tensor_to_map(x):
    """
    x: torch.Tensor [B, C, H, W] or [B, 1, H, W]
    return: numpy [H, W] normalized
    """
    x = x.detach().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.abs().mean(dim=0)
    return normalize_map(x.cpu().numpy())


def save_gray_and_npy(gray_map, png_path, npy_path):
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(png_path, gray_map, cmap="RdBu_r", vmin=0.0, vmax=1.0)
    np.save(npy_path, gray_map.astype(np.float32))


def save_green_and_npy(gray_map, png_path, npy_path):
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(png_path, gray_map, cmap="Greens", vmin=0.0, vmax=1.0)
    np.save(npy_path, gray_map.astype(np.float32))


def crop_map(x, box):
    x1, y1, x2, y2 = box
    return x[y1:y2, x1:x2]


def draw_boxes(vis_rgb, ir_gray, area1, area2, out_dir):
    vis_draw = vis_rgb.copy()
    ir_rgb = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)

    red = (255, 80, 60)
    green = (120, 200, 60)

    for img in [vis_draw, ir_rgb]:
        cv2.rectangle(img, (area1[0], area1[1]), (area1[2], area1[3]), red, 2)
        cv2.rectangle(img, (area2[0], area2[1]), (area2[2], area2[3]), green, 2)

        cv2.circle(img, (area1[2] - 10, area1[1] - 8 if area1[1] - 8 > 18 else area1[1] + 18), 14, red, -1)
        cv2.putText(img, "1", (area1[2] - 16, area1[1] - 2 if area1[1] - 2 > 18 else area1[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(img, (area2[2] - 10, area2[1] - 8 if area2[1] - 8 > 18 else area2[1] + 18), 14, green, -1)
        cv2.putText(img, "2", (area2[2] - 16, area2[1] - 2 if area2[1] - 2 > 18 else area2[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(Path(out_dir) / "visible_marked.png"), cv2.cvtColor(vis_draw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "infrared_marked.png"), cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR))


# ---------------------------
# Model loading
# ---------------------------
def load_modules(device, ckpt_path):
    modules = build_model(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    validate_checkpoint_metadata(checkpoint, modules[1])

    keys = (
        "shared_encoder",
        "intent_generator",
        "frequency_fusion",
        "frequency_pyramid_adapter",
        "spatial_fusion",
        "fsrc_l1",
        "fsrc_l2",
        "fsrc_l3",
        "fusion_decoder",
    )
    for module, key in zip(modules, keys):
        _load_state(module, checkpoint, key, strict=True)
        module.eval()
    return modules


# ---------------------------
# Forward logic
# ---------------------------
@torch.no_grad()
def forward_two_settings(device, ckpt_path, vis_path, ir_path):
    vis_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if vis_bgr is None:
        raise FileNotFoundError(f"Cannot read visible image: {vis_path}")
    if ir_gray is None:
        raise FileNotFoundError(f"Cannot read infrared image: {ir_path}")

    h, w = vis_bgr.shape[:2]
    if ir_gray.shape[:2] != (h, w):
        ir_gray = cv2.resize(ir_gray, (w, h), interpolation=cv2.INTER_LINEAR)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_y = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    vis_tensor = torch.from_numpy(vis_y.astype(np.float32) / 255.0)[None, None].to(device)
    ir_tensor = torch.from_numpy(ir_gray.astype(np.float32) / 255.0)[None, None].to(device)

    vis_rgb_tensor = (
        torch.from_numpy(vis_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    vis_clip = preprocess_clip_rgb(vis_rgb_tensor)

    (
        shared_encoder,
        intent_generator,
        frequency_fusion,
        frequency_pyramid_adapter,
        spatial_fusion,
        fsrc_l1,
        fsrc_l2,
        fsrc_l3,
        fusion_decoder,
    ) = load_modules(device, ckpt_path)

    vis_spa, vis_freq, _ = shared_encoder(vis_tensor)
    ir_spa, ir_freq, _ = shared_encoder(ir_tensor)

    i_deg, i_fus, _ = intent_generator(vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
    i_shared = 0.5 * (i_deg + i_fus)

    # shared-intent setting (without disentanglement)
    fused_freq_shared, freq_aux_shared = frequency_fusion(
        vis_freq, ir_freq, frequency_intent=i_shared
    )
    _, spatial_pyramid_shared, spa_aux_shared = spatial_fusion(
        vis_spa, ir_spa, i_shared, return_aux=True, return_pyramid=True
    )

    # disentangled setting
    fused_freq_dis, freq_aux_dis = frequency_fusion(
        vis_freq, ir_freq, frequency_intent=i_deg
    )
    _, spatial_pyramid_dis, spa_aux_dis = spatial_fusion(
        vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True
    )

    # Frequency-domain aligned feature maps (for area1 / degradation region)
    # Using fused_freq feature magnitude gives a spatially aligned activation map.
    freq_map_shared = tensor_to_map(fused_freq_shared)
    freq_map_dis = tensor_to_map(fused_freq_dis)

    # Spatial-domain map (for area2 / fusion region)
    # weight_l1 is the IR preference map at full resolution.
    spatial_weight_shared = normalize_map(
        spa_aux_shared["weight_l1"][0, 0].detach().float().cpu().numpy()
    )
    spatial_weight_dis = normalize_map(
        spa_aux_dis["weight_l1"][0, 0].detach().float().cpu().numpy()
    )

    # Optional extra raw outputs
    extras = {
        "full_frequency_shared": freq_map_shared,
        "full_frequency_dis": freq_map_dis,
        "full_spatial_shared": spatial_weight_shared,
        "full_spatial_dis": spatial_weight_dis,
        "amp_score_shared": freq_aux_shared["amp_score"][0].detach().cpu().numpy(),
        "phase_score_shared": freq_aux_shared["phase_score"][0].detach().cpu().numpy(),
        "amp_score_dis": freq_aux_dis["amp_score"][0].detach().cpu().numpy(),
        "phase_score_dis": freq_aux_dis["phase_score"][0].detach().cpu().numpy(),
    }

    return {
        "vis_rgb": vis_rgb,
        "ir_gray": ir_gray,
        "freq_map_shared": freq_map_shared,
        "freq_map_dis": freq_map_dis,
        "spatial_weight_shared": spatial_weight_shared,
        "spatial_weight_dis": spatial_weight_dis,
        "extras": extras,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", type=str, required=True, help="Path to visible RGB image.")
    parser.add_argument("--ir", type=str, required=True, help="Path to infrared grayscale image.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--area1", nargs=4, required=True, help="Degradation region: x1 y1 x2 y2")
    parser.add_argument("--area2", nargs=4, required=True, help="Fusion region: x1 y1 x2 y2")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    area1 = parse_box(args.area1)
    area2 = parse_box(args.area2)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    results = forward_two_settings(device, args.ckpt, args.vis, args.ir)

    # save top images with rectangles
    draw_boxes(results["vis_rgb"], results["ir_gray"], area1, area2, out_dir)

    # crop the corresponding regions
    # area1 -> frequency feature map
    without_area1 = crop_map(results["freq_map_shared"], area1)
    with_area1 = crop_map(results["freq_map_dis"], area1)

    # area2 -> spatial weighting map
    without_area2 = crop_map(results["spatial_weight_shared"], area2)
    with_area2 = crop_map(results["spatial_weight_dis"], area2)

    # Save PNG + raw NPY
    save_gray_and_npy(
        without_area1,
        out_dir / "without_area1_feature.png",
        out_dir / "without_area1_feature.npy",
    )
    save_gray_and_npy(
        with_area1,
        out_dir / "with_area1_feature.png",
        out_dir / "with_area1_feature.npy",
    )
    save_green_and_npy(
        without_area2,
        out_dir / "without_area2_feature.png",
        out_dir / "without_area2_feature.npy",
    )
    save_green_and_npy(
        with_area2,
        out_dir / "with_area2_feature.png",
        out_dir / "with_area2_feature.npy",
    )

    # Also save full-size maps in case you want to recolor / re-crop yourself
    save_gray_and_npy(
        results["freq_map_shared"],
        out_dir / "full_frequency_shared.png",
        out_dir / "full_frequency_shared.npy",
    )
    save_gray_and_npy(
        results["freq_map_dis"],
        out_dir / "full_frequency_dis.png",
        out_dir / "full_frequency_dis.npy",
    )
    save_green_and_npy(
        results["spatial_weight_shared"],
        out_dir / "full_spatial_shared.png",
        out_dir / "full_spatial_shared.npy",
    )
    save_green_and_npy(
        results["spatial_weight_dis"],
        out_dir / "full_spatial_dis.png",
        out_dir / "full_spatial_dis.npy",
    )

    # Save extra raw token scores too
    np.save(out_dir / "amp_score_shared.npy", results["extras"]["amp_score_shared"])
    np.save(out_dir / "phase_score_shared.npy", results["extras"]["phase_score_shared"])
    np.save(out_dir / "amp_score_dis.npy", results["extras"]["amp_score_dis"])
    np.save(out_dir / "phase_score_dis.npy", results["extras"]["phase_score_dis"])

    print("Saved outputs to:", out_dir)
    print("Main files:")
    print("  visible_marked.png")
    print("  infrared_marked.png")
    print("  without_area1_feature.png")
    print("  without_area2_feature.png")
    print("  with_area1_feature.png")
    print("  with_area2_feature.png")


if __name__ == "__main__":
    main()
