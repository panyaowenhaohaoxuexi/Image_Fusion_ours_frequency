# -*- coding: utf-8 -*-
"""
Screen MSRS test-set image pairs: find the one where intent disentanglement
produces the largest feature-map difference.

Usage (from repo root):
  python jiaoben/introduction/screen_best_pair.py
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch

from test import build_model, validate_checkpoint_metadata, _load_state
from utils.clip_preprocess import preprocess_clip_rgb

# =====================================================================
# CONFIG
# =====================================================================
TEST_DIR = r"F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS-main_autodl\test"
CKPT_PATH = r"F:\1_paper_pan\2_Image_Fusion\3_methods_pth_images\Ours\Complete\v11\pth\v11_clip_image_query_TextIntentDualDomainFusion_latest.pth"
DEVICE = None
TOP_K = 10  # report top-K pairs
SAMPLE_EVERY = 8  # screen every N-th image to go faster (1 = all)
# =====================================================================


def load_modules(device, ckpt_path):
    modules = build_model(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    validate_checkpoint_metadata(checkpoint, modules[1])
    keys = (
        "shared_encoder", "intent_generator", "frequency_fusion",
        "frequency_pyramid_adapter", "spatial_fusion",
        "fsrc_l1", "fsrc_l2", "fsrc_l3", "fusion_decoder",
    )
    for module, key in zip(modules, keys):
        _load_state(module, checkpoint, key, strict=True)
        module.eval()
    return modules


@torch.no_grad()
def score_one_pair(device, modules, vis_path, ir_path):
    (
        shared_encoder, intent_generator, frequency_fusion,
        frequency_pyramid_adapter, spatial_fusion,
        fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder,
    ) = modules

    vis_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if vis_bgr is None or ir_gray is None:
        return None

    h, w = vis_bgr.shape[:2]
    if ir_gray.shape[:2] != (h, w):
        ir_gray = cv2.resize(ir_gray, (w, h), interpolation=cv2.INTER_LINEAR)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_y = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    vis_t = torch.from_numpy(vis_y.astype(np.float32) / 255.0)[None, None].to(device)
    ir_t = torch.from_numpy(ir_gray.astype(np.float32) / 255.0)[None, None].to(device)
    vis_rgb_t = (
        torch.from_numpy(vis_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    vis_clip = preprocess_clip_rgb(vis_rgb_t)

    vis_spa, vis_freq, _ = shared_encoder(vis_t)
    ir_spa, ir_freq, _ = shared_encoder(ir_t)
    int_fre, int_spa, _ = intent_generator(vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
    int_shared = 0.5 * (int_fre + int_spa)

    def to_2d(t):
        x = t.detach().float()
        if x.ndim == 4:
            x = x[0]
        return x.abs().mean(dim=0).cpu().numpy().astype(np.float32)

    # --- shared ---
    fused_freq_shared, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=int_shared)
    _, _, spa_aux_shared = spatial_fusion(vis_spa, ir_spa, int_shared, return_aux=True, return_pyramid=True)

    # --- disentangled ---
    fused_freq_dis, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=int_fre)
    _, _, spa_aux_dis = spatial_fusion(vis_spa, ir_spa, int_spa, return_aux=True, return_pyramid=True)

    # Spatial: pick single most intent-responsive channel of vis_tilde
    vt_shared = spa_aux_shared["l1_aux"]["vis_tilde"].detach().float()[0]  # [C,H,W]
    vt_dis = spa_aux_dis["l1_aux"]["vis_tilde"].detach().float()[0]
    best_c = int((vt_dis - vt_shared).abs().mean(dim=(1, 2)).argmax())
    spa_shared = vt_shared[best_c].cpu().numpy().astype(np.float32)
    spa_dis = vt_dis[best_c].cpu().numpy().astype(np.float32)

    freq_shared = to_2d(fused_freq_shared)
    freq_dis = to_2d(fused_freq_dis)

    # Upsample freq to match spatial resolution
    hs, ws = spa_shared.shape
    if freq_shared.shape != (hs, ws):
        freq_shared = cv2.resize(freq_shared, (ws, hs), interpolation=cv2.INTER_LINEAR)
        freq_dis = cv2.resize(freq_dis, (ws, hs), interpolation=cv2.INTER_LINEAR)

    # Texture richness: mean Laplacian magnitude of visible image
    gray = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture_score = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())

    # Difference metrics (raw, before any normalization)
    diff_freq_mean = float(np.abs(freq_dis - freq_shared).mean())
    diff_spa_mean = float(np.abs(spa_dis - spa_shared).mean())

    # Combined spatial score: diff × texture
    spa_texture_score = diff_spa_mean * texture_score

    return {
        "diff_freq_mean": diff_freq_mean,
        "diff_spa_mean": diff_spa_mean,
        "texture_score": texture_score,
        "spa_texture_score": spa_texture_score,
    }


def main():
    device = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading model...")
    modules = load_modules(device, CKPT_PATH)
    print("Model loaded.\n")

    vi_dir = Path(TEST_DIR) / "vi"
    ir_dir = Path(TEST_DIR) / "ir"
    names = sorted([p.name for p in vi_dir.iterdir() if p.suffix in (".png", ".jpg", ".bmp")])

    # Sample every N-th image
    names = names[::SAMPLE_EVERY]
    total = len(names)
    print(f"Screening {total} image pairs (every {SAMPLE_EVERY}{'th' if SAMPLE_EVERY > 1 else ''} image)...\n")

    results = []
    for i, name in enumerate(names):
        vi_path = str(vi_dir / name)
        ir_path = str(ir_dir / name)
        score = score_one_pair(device, modules, vi_path, ir_path)
        if score is not None:
            results.append((name, score))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total}...")

    # ---- Rank by spatial diff × texture (main target) ----
    spa_sorted = sorted(results, key=lambda t: t[1]["spa_texture_score"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP {TOP_K} — spatial diff × texture  (what you care about)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'File':<20} {'Spatial_diff':>14} {'Texture':>12} {'Spa×Tex':>12}")
    print("-" * 80)
    for rank, (name, s) in enumerate(spa_sorted[:TOP_K], 1):
        print(f"{rank:<5} {name:<20} {s['diff_spa_mean']:>14.6f} {s['texture_score']:>12.6f} {s['spa_texture_score']:>12.6f}")

    print(f"\n  Best for spatial:  {spa_sorted[0][0]}")
    print(f"    VI : {vi_dir / spa_sorted[0][0]}")
    print(f"    IR : {ir_dir / spa_sorted[0][0]}")

    # ---- Also show pure spatial diff ranking (no texture) ----
    pure_sorted = sorted(results, key=lambda t: t[1]["diff_spa_mean"], reverse=True)
    print(f"\n{'='*80}")
    print(f"  TOP 5 — pure spatial diff  (no texture weighting)")
    print(f"{'='*80}")
    for rank, (name, s) in enumerate(pure_sorted[:5], 1):
        print(f"  {rank}. {name}  spatial_diff={s['diff_spa_mean']:.6f}  texture={s['texture_score']:.6f}")


if __name__ == "__main__":
    main()
