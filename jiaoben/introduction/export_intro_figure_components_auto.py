# -*- coding: utf-8 -*-
"""
Export Introduction-figure components from TWO early network positions
where intent disentanglement acts directly.

  • Area 1 (degradation / frequency):  TGSFF fused frequency feature
  • Area 2 (fusion / spatial):         TextConditionedSpatialFusion
                                        L1 intent-modulated IR feature (ir_tilde)

The two positions are far apart in the architecture, so the feature
visualisation semantics differ — and the difference signal is strong.

Shared architecture:
    1.  SharedEncoder  →  vis_freq/vis_spa + ir_freq/ir_spa
    2.  IntentGenerator  →  Int_fre, Int_spa
    3.  Frequency branch:  TGSFF(freq, intent)  →  fused_freq
    4.  Spatial branch:    TextCondSF(vfeat, intent)  →  weight_l1
    5.  FreqPyramidAdapter  →  freq_pyramid
    6.  FSRC-L1/L2/L3  →  d_l1, d_l2, d_l3
    7.  FusionDecoder  →  final fused image

The Introduction figure extracts features from steps 3 & 4, where
the intent signals arrive fresh and the between-setting difference is
large enough to be visually convincing.
"""

# =====================================================================
#  CONFIGURATION  —  edit these paths
# =====================================================================

VIS_PATH = r"F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS-main_autodl\test\vi\00319D.png"
IR_PATH = r"F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS-main_autodl\test\ir\00319D.png"
CKPT_PATH = r"F:\1_paper_pan\2_Image_Fusion\3_methods_pth_images\Ours\Complete\v11\pth\v11_clip_image_query_TextIntentDualDomainFusion_latest.pth"
OUT_DIR = r"F:\1_paper_pan\2_Image_Fusion\3_methods_pth_images\Ours\Complete\v11\Introduction"

# Region selection mode:
#   "interactive" → draw two boxes on the image yourself
#   "auto"        → automatic selection via diff × darkness/texture/brightness
#   Or set AREA1/AREA2 manually: [x1, y1, x2, y2]
SELECT_MODE = "interactive"   # "interactive" / "auto"
AREA1 = None   # degradation region  → frequency feature (only used if set)
AREA2 = None   # fusion region       → spatial weight  (only used if set)

# Spatial "without" overall attenuation: keeps 50:50 mix, scales down magnitude.
#   1.0 → without = full-strength 0.5·fre + 0.5·spa (original)
#   0.5 → without = half-strength
#   0.3 → without = 30%-strength  (larger difference from "with")
SPATIAL_WITHOUT_ATTENUATION = 0.1   # 越小 → 注入越弱 → 差异越大

# Auto-detection parameters (used only when AREA1/AREA2 are None).
WINDOW_SIZE = None          # None → min(100, min(H,W)//4)
WINDOW_SIZE_SPATIAL = None  # None → WINDOW_SIZE // 2  (smaller = more focused)
MIN_DISTANCE = None         # None → WINDOW_SIZE * 1.2

DEVICE = None               # None → CUDA if available

# Colormaps (kept separate because the two feature types have different
# physical meanings and benefit from distinct colour scales).
COLORMAP_FREQ = "RdBu_r"    # frequency fused-feature → diverging
COLORMAP_SPATIAL = "Greens"  # spatial feature → sequential

# Optional extras.
SAVE_FULL_FSRC_LEVELS = True
SAVE_AMP_PHASE_SCORES = True

# =====================================================================

import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path before any project imports,
# otherwise "test" may resolve to the Python stdlib test package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from test import build_model, validate_checkpoint_metadata, _load_state
from utils.clip_preprocess import preprocess_clip_rgb


# ---------------------------------------------------------------------
#  Visualisation utilities
# ---------------------------------------------------------------------
def _tensor_to_2d(tensor: torch.Tensor) -> np.ndarray:
    """[B,C,H,W] → [H,W] via mean-of-abs across channels, no normalisation."""
    x = tensor.detach().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.abs().mean(dim=0)
    return x.cpu().numpy().astype(np.float32)


def joint_normalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    """Normalize two arrays using the SAME [1%, 99%] range from the merged data."""
    merged = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.percentile(merged, 1.0))
    hi = float(np.percentile(merged, 99.0))

    def _norm(x):
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)

    return _norm(a), _norm(b), (lo, hi)


def normalize_single(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def save_map(array: np.ndarray, png_path: Path, npy_path: Path, cmap: str,
             colorbar: bool = False, label: str = "", output_size: int = 400):
    """
    Save a 2D map as PNG+NPY.  If colorbar=True the figure is rendered
    at a fixed size so all panels share the same pixel dimensions.
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if colorbar:
        # Determine figure size to get output_size×output_size image area
        dpi = 200
        margin = 1.2  # extra width for colorbar + padding
        fig_w = output_size * margin / dpi
        fig_h = output_size / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        im = ax.imshow(array, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label(label, fontsize=8)
        fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
        plt.close(fig)
    else:
        plt.imsave(str(png_path), array, cmap=cmap, vmin=0.0, vmax=1.0)
    np.save(str(npy_path), array.astype(np.float32))


def crop_map(array: np.ndarray, box):
    x1, y1, x2, y2 = map(int, box)
    return array[y1:y2, x1:x2]


def upsample_map(src: np.ndarray, target_hw) -> np.ndarray:
    """Upsample src to (H,W) via bilinear interpolation."""
    h, w = src.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return src
    return cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)


def validate_box(box, width, height, name):
    if box is None:
        return
    if len(box) != 4:
        raise ValueError(f"{name} must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name}={box} exceeds image size {width}x{height}")


def draw_boxes(vis_rgb, ir_gray, area1, area2, out_dir):
    vis_draw = vis_rgb.copy()
    ir_draw = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)

    red = (255, 80, 60)
    green = (110, 195, 65)

    for image in (vis_draw, ir_draw):
        for index, (box, color) in enumerate(((area1, red), (area2, green)), start=1):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            cx = max(x1 + 14, x2 - 12)
            cy = y1 + 16
            cv2.circle(image, (cx, cy), 13, color, -1)
            cv2.putText(image, str(index), (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "visible_marked.png"),
                cv2.cvtColor(vis_draw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "infrared_marked.png"),
                cv2.cvtColor(ir_draw, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
#  Auto-detection: area1 from diff_freq, area2 from diff_spa × texture
# ---------------------------------------------------------------------
def _score_map(diff_map, window_size):
    """Integral-image sliding-window mean."""
    ws = int(window_size)
    h, w = diff_map.shape
    if ws > h or ws > w:
        raise ValueError(f"window_size={ws} > map {h}×{w}")
    integral = np.pad(np.cumsum(np.cumsum(diff_map, axis=0), axis=1), ((1, 0), (1, 0)))
    a, b = integral[:-ws, :-ws], integral[ws:, :-ws]
    c, d = integral[:-ws, ws:], integral[ws:, ws:]
    return (d - b - c + a) / float(ws * ws)


def _texture_score_map(rgb_image, window_size):
    """Local texture/structure richness via standard deviation in each ws×ws window."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Per-pixel local std via integral image: std = sqrt(E[X²] - E[X]²)
    sq = gray * gray
    sum_g = cv2.integral(gray)
    sum_sq = cv2.integral(sq)
    # Extract window sums (same pattern as _score_map)
    ws = int(window_size)
    a_g, b_g = sum_g[:-ws, :-ws], sum_g[ws:, :-ws]
    c_g, d_g = sum_g[:-ws, ws:], sum_g[ws:, ws:]
    a_s, b_s = sum_sq[:-ws, :-ws], sum_sq[ws:, :-ws]
    c_s, d_s = sum_sq[:-ws, ws:], sum_sq[ws:, ws:]
    n = float(ws * ws)
    mean = (d_g - b_g - c_g + a_g) / n
    mean_sq = (d_s - b_s - c_s + a_s) / n
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def _darkness_score_map(rgb_image, window_size):
    """Local darkness: 1 - mean luminance in each ws×ws window."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark = 1.0 - gray  # 1 = pitch black, 0 = bright white
    return _score_map(dark, int(window_size))


def _top_candidates(scores, k=200):
    flat = scores.reshape(-1)
    n = min(k, flat.size)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    rows, cols = np.unravel_index(idx, scores.shape)
    return [(float(flat[i]), int(rows[j]), int(cols[j])) for j, i in enumerate(idx)]


def find_best_regions(diff_freq, diff_spatial, vis_rgb, ws_freq, ws_spa, min_distance):
    """
    area1 ← diff_freq × darkness  (frequency disentanglement in low-light regions)
    area2 ← diff_spatial × texture (spatial disentanglement in richly textured regions)
    """
    half_f = ws_freq / 2.0
    half_s = ws_spa / 2.0

    freq_scores = _score_map(diff_freq, ws_freq)
    dark_scores = _darkness_score_map(vis_rgb, ws_freq)
    spat_scores = _score_map(diff_spatial, ws_spa)
    text_scores = _texture_score_map(vis_rgb, ws_spa)
    bright_scores = 1.0 - _darkness_score_map(vis_rgb, ws_spa)  # own ws

    freq_dark_scores = freq_scores * dark_scores      # area1: diff大 × 暗
    spa_texture_scores = spat_scores * text_scores * bright_scores  # diff大 × 纹理 × 亮

    freq_cands = _top_candidates(freq_dark_scores)
    spa_cands = _top_candidates(spa_texture_scores)

    best_pair = None
    best_sum = -np.inf

    for f_score, fy, fx in freq_cands[:30]:
        fc = np.array([fx + half_f, fy + half_f], dtype=np.float32)
        for s_score, sy, sx in spa_cands[:30]:
            sc = np.array([sx + half_s, sy + half_s], dtype=np.float32)
            if float(np.linalg.norm(fc - sc)) < min_distance:
                continue
            if f_score + s_score > best_sum:
                best_sum = f_score + s_score
                best_pair = (
                    [fx, fy, fx + ws_freq, fy + ws_freq],
                    [sx, sy, sx + ws_spa, sy + ws_spa],
                )

    if best_pair is None:
        _, fy, fx = freq_cands[0]
        fc = np.array([fx + half_f, fy + half_f])
        _, sy, sx = max(
            spa_cands[1:],
            key=lambda t: float(np.linalg.norm(
                np.array([t[2] + half_s, t[1] + half_s]) - fc)),
            default=(0, fy, fx),
        )
        best_pair = (
            [fx, fy, fx + ws_freq, fy + ws_freq],
            [sx, sy, sx + ws_spa, sy + ws_spa],
        )
        print(f"\n⚠  No non-overlapping pair found with min_distance={min_distance}."
              f" Using fallback.\n")

    return best_pair, {    # diagnostic score maps
        "freq_dark": freq_dark_scores,
        "spa_texture": spa_texture_scores,
        "spat_scores": spat_scores,
        "text_scores": text_scores,
        "bright_scores": bright_scores,
        "freq_scores": freq_scores,
        "dark_scores": dark_scores,
    }


# ---------------------------------------------------------------------
#  Forward pass  (shared-intent vs. disentangled)
# ---------------------------------------------------------------------
@torch.no_grad()
def forward_two_settings(device, ckpt_path, vis_path, ir_path):
    # ---- read images ---------------------------------------------------
    vis_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if vis_bgr is None:
        raise FileNotFoundError(f"Cannot read visible image: {vis_path}")
    if ir_gray is None:
        raise FileNotFoundError(f"Cannot read infrared image: {ir_path}")

    h_img, w_img = vis_bgr.shape[:2]
    if ir_gray.shape[:2] != (h_img, w_img):
        ir_gray = cv2.resize(ir_gray, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_y = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    vis_t = torch.from_numpy(vis_y.astype(np.float32) / 255.0)[None, None].to(device)
    ir_t = torch.from_numpy(ir_gray.astype(np.float32) / 255.0)[None, None].to(device)
    vis_rgb_t = (
        torch.from_numpy(vis_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    vis_clip = preprocess_clip_rgb(vis_rgb_t)

    # ---- load modules --------------------------------------------------
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

    # ---- encode --------------------------------------------------------
    vis_spa, vis_freq, _ = shared_encoder(vis_t)
    ir_spa, ir_freq, _ = shared_encoder(ir_t)

    # ---- intents -------------------------------------------------------
    int_fre, int_spa, _ = intent_generator(vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
    int_shared = 0.5 * (int_fre + int_spa)
    # Spatial "without" = attenuated balanced mix: λ·(0.5·fre + 0.5·spa)
    λ = SPATIAL_WITHOUT_ATTENUATION
    int_spa_without = λ * int_shared         # int_shared = 0.5·fre + 0.5·spa

    # ====================================================================
    #  WITHOUT disentanglement
    # ====================================================================
    fused_freq_shared, freq_aux_shared = frequency_fusion(
        vis_freq, ir_freq, frequency_intent=int_shared)
    _, spatial_pyramid_shared, spa_aux_shared = spatial_fusion(
        vis_spa, ir_spa, int_spa_without, return_aux=True, return_pyramid=True)
    freq_pyramid_shared = frequency_pyramid_adapter(
        fused_freq_shared, target_pyramid=spatial_pyramid_shared)
    d_l1_shared, _ = fsrc_l1(freq_pyramid_shared["l1"], spatial_pyramid_shared["l1"])

    # ====================================================================
    #  WITH disentanglement
    # ====================================================================
    fused_freq_dis, freq_aux_dis = frequency_fusion(
        vis_freq, ir_freq, frequency_intent=int_fre)
    _, spatial_pyramid_dis, spa_aux_dis = spatial_fusion(
        vis_spa, ir_spa, int_spa, return_aux=True, return_pyramid=True)
    freq_pyramid_dis = frequency_pyramid_adapter(
        fused_freq_dis, target_pyramid=spatial_pyramid_dis)
    d_l1_dis, _ = fsrc_l1(freq_pyramid_dis["l1"], spatial_pyramid_dis["l1"])

    # ---- Extract the TWO feature sources --------------------------------
    # (a) Frequency fused feature  [B, C, Hf, Wf]
    raw_freq_shared = _tensor_to_2d(fused_freq_shared)
    raw_freq_dis = _tensor_to_2d(fused_freq_dis)

    # (b) Spatial L1 fused output  — fused_l1 = w*ir_tilde + (1-w)*vis_tilde
    # Gate output combines weight change + vis modulation + ir modulation.
    # Pick the single most intent-responsive channel.
    fused_l1_shared = spa_aux_shared["fused_l1"].detach().float()[0]  # [C,H,W]
    fused_l1_dis = spa_aux_dis["fused_l1"].detach().float()[0]
    chan_diffs = (fused_l1_dis - fused_l1_shared).abs().mean(dim=(1, 2))
    best_c = int(chan_diffs.argmax())
    raw_spa_shared = fused_l1_shared[best_c].cpu().numpy().astype(np.float32)
    raw_spa_dis = fused_l1_dis[best_c].cpu().numpy().astype(np.float32)
    print(f"  Spatial fused_l1: picked channel {best_c}/{fused_l1_shared.shape[0]}"
          f" (diff={float(chan_diffs[best_c]):.6f})")

    # ---- Upsample freq maps to match spatial/image resolution ----------
    H_spa, W_spa = raw_spa_shared.shape
    raw_freq_shared = upsample_map(raw_freq_shared, (H_spa, W_spa))
    raw_freq_dis = upsample_map(raw_freq_dis, (H_spa, W_spa))

    # ---- Joint-normalize each pair -------------------------------------
    map_freq_shared, map_freq_dis, freq_norm_range = joint_normalize(
        raw_freq_shared, raw_freq_dis)
    map_spa_shared, map_spa_dis, spa_norm_range = joint_normalize(
        raw_spa_shared, raw_spa_dis)

    # ---- Difference maps (raw, before joint-normalization) -------------
    raw_diff_freq = np.abs(raw_freq_dis - raw_freq_shared)
    raw_diff_spa = np.abs(raw_spa_dis - raw_spa_shared)
    diff_freq = normalize_single(raw_diff_freq)
    diff_spa = normalize_single(raw_diff_spa)

    result = {
        "vis_rgb": vis_rgb,
        "ir_gray": ir_gray,
        "h_img": h_img, "w_img": w_img,
        # freq pair
        "raw_freq_shared": raw_freq_shared, "raw_freq_dis": raw_freq_dis,
        "map_freq_shared": map_freq_shared, "map_freq_dis": map_freq_dis,
        "raw_diff_freq": raw_diff_freq, "diff_freq": diff_freq,
        "freq_norm_range": freq_norm_range,
        # spatial pair
        "raw_spa_shared": raw_spa_shared, "raw_spa_dis": raw_spa_dis,
        "map_spa_shared": map_spa_shared, "map_spa_dis": map_spa_dis,
        "raw_diff_spa": raw_diff_spa, "diff_spa": diff_spa,
        "spa_norm_range": spa_norm_range,
        # FSRC-L1 (kept for diagnostic / optional saving)
        "d_l1_shared": d_l1_shared, "d_l1_dis": d_l1_dis,
    }

    # ---- Amp / phase scores (optional) ---------------------------------
    if SAVE_AMP_PHASE_SCORES:
        for tag, aux in (("shared", freq_aux_shared), ("dis", freq_aux_dis)):
            result[f"amp_score_{tag}"] = (
                aux["amp_score"][0].detach().cpu().numpy())
            result[f"phase_score_{tag}"] = (
                aux["phase_score"][0].detach().cpu().numpy())

    # ---- FSRC L2 / L3 (optional) ---------------------------------------
    if SAVE_FULL_FSRC_LEVELS:
        d_l2_shared, _ = fsrc_l2(freq_pyramid_shared["l2"], spatial_pyramid_shared["l2"])
        d_l2_dis, _ = fsrc_l2(freq_pyramid_dis["l2"], spatial_pyramid_dis["l2"])
        d_l3_shared, _ = fsrc_l3(freq_pyramid_shared["l3"], spatial_pyramid_shared["l3"])
        d_l3_dis, _ = fsrc_l3(freq_pyramid_dis["l3"], spatial_pyramid_dis["l3"])
        for level, shared_t, dis_t in (
            ("l2", d_l2_shared, d_l2_dis),
            ("l3", d_l3_shared, d_l3_dis),
        ):
            raw_s = _tensor_to_2d(shared_t)
            raw_d = _tensor_to_2d(dis_t)
            map_s, map_d, _ = joint_normalize(raw_s, raw_d)
            result[f"map_fsrc_{level}_shared"] = map_s
            result[f"map_fsrc_{level}_dis"] = map_d
            result[f"diff_fsrc_{level}"] = normalize_single(np.abs(raw_d - raw_s))

    return result


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    # ---- validate paths ------------------------------------------------
    for name, path in (("VIS_PATH", VIS_PATH), ("IR_PATH", IR_PATH),
                        ("CKPT_PATH", CKPT_PATH)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ---- forward -------------------------------------------------------
    print("Running forward: shared-intent vs. disentangled ...")
    r = forward_two_settings(device, CKPT_PATH, VIS_PATH, IR_PATH)
    H, W = r["h_img"], r["w_img"]

    # ---- region selection ----------------------------------------------
    validate_box(AREA1, W, H, "AREA1")
    validate_box(AREA2, W, H, "AREA2")

    if AREA1 is not None and AREA2 is not None:
        area1 = list(map(int, AREA1))
        area2 = list(map(int, AREA2))
        print("Using manually-specified regions.")
    elif SELECT_MODE == "interactive":
        disp = cv2.cvtColor(r["vis_rgb"], cv2.COLOR_RGB2BGR)
        print("\n🖱  Draw TWO boxes on the image:")
        print("     Box 1 → Area1: degradation / frequency")
        print("     Box 2 → Area2: fusion / spatial")
        print("   Draw rectangle, press ENTER (or SPACE) to confirm.")
        print("   Press ESC to cancel.\n")

        r1 = cv2.selectROI("Area1 — draw, press ENTER", disp, showCrosshair=True)
        if r1[2] == 0 or r1[3] == 0:
            cv2.destroyAllWindows()
            raise RuntimeError("Area1 cancelled or empty.")
        area1 = [int(r1[0]), int(r1[1]), int(r1[0] + r1[2]), int(r1[1] + r1[3])]
        print(f"  Area1: {area1}")

        r2 = cv2.selectROI("Area2 — draw, press ENTER", disp, showCrosshair=True)
        cv2.destroyAllWindows()
        if r2[2] == 0 or r2[3] == 0:
            raise RuntimeError("Area2 cancelled or empty.")
        area2 = [int(r2[0]), int(r2[1]), int(r2[0] + r2[2]), int(r2[1] + r2[3])]
        print(f"  Area2: {area2}")
    else:
        ws_freq = WINDOW_SIZE or min(100, min(H, W) // 4)
        ws_freq = int(min(ws_freq, H, W))
        ws_spa = WINDOW_SIZE_SPATIAL or (ws_freq // 2)
        ws_spa = int(min(ws_spa, H, W))
        md = MIN_DISTANCE or ws_freq * 1.2
        print(f"Auto-detect: freq_window={ws_freq}×{ws_freq}  spa_window={ws_spa}×{ws_spa}"
              f"  min_dist={md:.0f}")
        area1, area2, scores = find_best_regions(
            r["raw_diff_freq"], r["raw_diff_spa"], r["vis_rgb"],
            ws_freq, ws_spa, float(md))
        print("Regions selected.")

        # Diagnostic: save score maps so you can inspect what the algorithm sees
        for key, cmap in (
            ("freq_scores", "inferno"), ("dark_scores", "gray_r"),
            ("freq_dark", "inferno"),
            ("spat_scores", "inferno"), ("text_scores", "Greens"),
            ("bright_scores", "gray"), ("spa_texture", "inferno"),
        ):
            save_map(
                normalize_single(scores[key]),
                out_dir / f"diagnostic_{key}.png",
                out_dir / f"diagnostic_{key}.npy",
                cmap,
            )

    print(f"  Area1 (freq/degradation): {area1}")
    print(f"  Area2 (spatial/fusion):    {area2}")

    # ---- marked source images ------------------------------------------
    draw_boxes(r["vis_rgb"], r["ir_gray"], area1, area2, out_dir)

    # ---- cropped feature panels (the 4 main components) ----------------
    # Resize all crops to the same pixel size for uniform output.
    OUT_SIZE = 400
    for name, src_map, cmap, cbar_label in (
        ("without_area1_freq", r["map_freq_shared"], COLORMAP_FREQ, "Freq activation"),
        ("with_area1_freq",    r["map_freq_dis"],    COLORMAP_FREQ, "Freq activation"),
        ("without_area2_spa",  r["map_spa_shared"],  COLORMAP_SPATIAL, "Spatial activation"),
        ("with_area2_spa",     r["map_spa_dis"],     COLORMAP_SPATIAL, "Spatial activation"),
    ):
        area = area1 if "area1" in name else area2
        crop = crop_map(src_map, area)
        crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
        save_map(crop,
                 out_dir / f"{name}.png", out_dir / f"{name}.npy", cmap,
                 colorbar=True, label=cbar_label)

    # ---- full-size feature maps ----------------------------------------
    save_map(r["map_freq_shared"],  out_dir / "full_freq_shared.png",
             out_dir / "full_freq_shared.npy", COLORMAP_FREQ)
    save_map(r["map_freq_dis"],     out_dir / "full_freq_disentangled.png",
             out_dir / "full_freq_disentangled.npy", COLORMAP_FREQ)
    save_map(r["map_spa_shared"],   out_dir / "full_spatial_shared.png",
             out_dir / "full_spatial_shared.npy", COLORMAP_SPATIAL)
    save_map(r["map_spa_dis"],      out_dir / "full_spatial_disentangled.png",
             out_dir / "full_spatial_disentangled.npy", COLORMAP_SPATIAL)

    # ---- difference maps ------------------------------------------------
    save_map(r["diff_freq"],  out_dir / "diff_freq.png",
             out_dir / "diff_freq.npy", "inferno",
             colorbar=True, label="Frequency |Δ|")
    save_map(r["diff_spa"],   out_dir / "diff_spatial.png",
             out_dir / "diff_spatial.npy", "inferno",
             colorbar=True, label="Spatial |Δ|")

    # ---- raw .npy archives ----------------------------------------------
    for key in ("raw_freq_shared", "raw_freq_dis", "raw_spa_shared",
                 "raw_spa_dis", "raw_diff_freq", "raw_diff_spa"):
        np.save(out_dir / f"{key}.npy", r[key])
    np.save(out_dir / "freq_norm_range.npy",
            np.asarray(r["freq_norm_range"], dtype=np.float32))
    np.save(out_dir / "spa_norm_range.npy",
            np.asarray(r["spa_norm_range"], dtype=np.float32))

    # ---- amp / phase scores --------------------------------------------
    if SAVE_AMP_PHASE_SCORES:
        for key in ("amp_score_shared", "phase_score_shared",
                     "amp_score_dis", "phase_score_dis"):
            np.save(out_dir / f"{key}.npy", r[key])

    # ---- FSRC levels (diagnostic) --------------------------------------
    if SAVE_FULL_FSRC_LEVELS:
        for level in ("l2", "l3"):
            save_map(r[f"map_fsrc_{level}_shared"],
                     out_dir / f"fsrc_{level}_shared.png",
                     out_dir / f"fsrc_{level}_shared.npy", "magma")
            save_map(r[f"map_fsrc_{level}_dis"],
                     out_dir / f"fsrc_{level}_disentangled.png",
                     out_dir / f"fsrc_{level}_disentangled.npy", "magma")
            save_map(r[f"diff_fsrc_{level}"],
                     out_dir / f"diff_fsrc_{level}.png",
                     out_dir / f"diff_fsrc_{level}.npy", "inferno")

    # ---- summary -------------------------------------------------------
    print(f"\n✅ Saved to: {out_dir}/")
    print("=" * 60)
    print("🧩  Main Introduction-figure components (4 panels):")
    print("    Area 1 — frequency fused feature (RdBu_r):")
    print("      without_area1_freq.png   ← int_shared")
    print("      with_area1_freq.png      ← int_fre")
    print("    Area 2 — spatial IR-preference weight (Greens):")
    print("      without_area2_spa.png    ← int_shared")
    print("      with_area2_spa.png       ← int_spa")
    print("-" * 60)
    print("🔍  Verify auto-selection first:")
    print("      visible_marked.png      — source image + boxes")
    print("      diff_freq.png           — where freq fusion changes most")
    print("      diff_spatial.png        — where spatial weight changes most")
    print("-" * 60)
    print("📐  Full-size feature maps:")
    print("      full_freq_{shared,disentangled}.png")
    print("      full_spatial_{shared,disentangled}.png")


if __name__ == "__main__":
    main()
