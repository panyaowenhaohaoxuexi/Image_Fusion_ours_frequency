# -*- coding: utf-8 -*-
"""
Feature-level Introduction figure for Text-FSFuse v11.

Two modes:
1) Demo layout:
   python plot_intro_dual_intent_feature_level.py --demo

2) Real model visualization:
   python plot_intro_dual_intent_feature_level.py \
       --ir path/to/ir.png \
       --vis path/to/vis.png \
       --ckpt path/to/checkpoint.pth \
       --output intro_dual_intent_real.png

Place this script in the repository root so it can import test.py and net/.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Repository-specific imports are loaded lazily in real-model mode,
# so --demo can run anywhere.


def _normalize_array(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], 1.0)
    hi = np.percentile(x[finite], 99.0)
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _tensor_map(x: torch.Tensor, reduction: str = "mean_abs") -> np.ndarray:
    x = x.detach().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        if reduction == "mean":
            x = x.mean(dim=0)
        elif reduction == "mean_abs":
            x = x.abs().mean(dim=0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return _normalize_array(x.cpu().numpy())


def _score_to_grid(score: torch.Tensor, spectral_hw: Tuple[int, int], patch_size: int) -> np.ndarray:
    h, w = spectral_hw
    gh = math.ceil(h / patch_size)
    gw = math.ceil(w / patch_size)
    score = score.detach().float()[0]
    if score.numel() != gh * gw:
        raise RuntimeError(
            f"Score length {score.numel()} does not match expected token grid {gh}x{gw}."
        )
    return _normalize_array(score.reshape(gh, gw).cpu().numpy())


def _mask_to_grid(mask: torch.Tensor, spectral_hw: Tuple[int, int], patch_size: int) -> np.ndarray:
    h, w = spectral_hw
    gh = math.ceil(h / patch_size)
    gw = math.ceil(w / patch_size)
    mask = mask.detach().float()[0]
    if mask.numel() != gh * gw:
        raise RuntimeError(
            f"Mask length {mask.numel()} does not match expected token grid {gh}x{gw}."
        )
    return mask.reshape(gh, gw).cpu().numpy()


def _read_inputs(ir_path: str, vis_path: str, device: torch.device):
    ir_np = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vis_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    if ir_np is None:
        raise FileNotFoundError(f"Cannot read infrared image: {ir_path}")
    if vis_bgr is None:
        raise FileNotFoundError(f"Cannot read visible image: {vis_path}")

    h, w = vis_bgr.shape[:2]
    if ir_np.shape[:2] != (h, w):
        ir_np = cv2.resize(ir_np, (w, h), interpolation=cv2.INTER_LINEAR)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_y = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    ir = torch.from_numpy(ir_np.astype(np.float32) / 255.0)[None, None].to(device)
    vis = torch.from_numpy(vis_y.astype(np.float32) / 255.0)[None, None].to(device)
    vis_rgb_tensor = (
        torch.from_numpy(vis_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    from utils.clip_preprocess import preprocess_clip_rgb
    vis_clip = preprocess_clip_rgb(vis_rgb_tensor)
    return ir_np, vis_rgb, ir, vis, vis_clip


def _load_modules(device: torch.device, ckpt_path: str):
    from test import build_model, validate_checkpoint_metadata, _load_state

    modules = build_model(device)
    (
        encoder,
        intent_generator,
        frequency_fusion,
        frequency_pyramid_adapter,
        spatial_fusion,
        fsrc_l1,
        fsrc_l2,
        fsrc_l3,
        fusion_decoder,
    ) = modules

    checkpoint = torch.load(ckpt_path, map_location=device)
    validate_checkpoint_metadata(checkpoint, intent_generator)
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


@torch.no_grad()
def collect_real_features(
    ir_path: str,
    vis_path: str,
    ckpt_path: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    ir_np, vis_rgb, ir, vis, vis_clip = _read_inputs(ir_path, vis_path, device)
    modules = _load_modules(device, ckpt_path)
    encoder, intent_generator, frequency_fusion, _, spatial_fusion, *_ = modules

    vis_spa, vis_freq, _ = encoder(vis)
    ir_spa, ir_freq, _ = encoder(ir)

    int_fre, int_spa, intent_aux = intent_generator(
        vis_clip, vis_spa, ir_spa, vis_freq, ir_freq
    )

    _, freq_aux = frequency_fusion(
        vis_freq, ir_freq, frequency_intent=int_fre
    )
    _, _, spa_aux = spatial_fusion(
        vis_spa,
        ir_spa,
        int_spa,
        return_aux=True,
        return_pyramid=True,
    )

    from net.frequency_fusion.fft_utils import split_amplitude_phase

    vis_amp, vis_phase = split_amplitude_phase(vis_freq)
    ir_amp, ir_phase = split_amplitude_phase(ir_freq)

    amp_gap = torch.log1p(vis_amp).sub(torch.log1p(ir_amp)).abs().mean(dim=1)
    phase_delta = torch.atan2(
        torch.sin(vis_phase - ir_phase),
        torch.cos(vis_phase - ir_phase),
    ).abs().mean(dim=1)

    spectral_hw = tuple(vis_amp.shape[-2:])
    patch_size = frequency_fusion.module.patch_size

    return {
        "ir_image": ir_np,
        "vis_image": vis_rgb,
        "amp_gap": _normalize_array(amp_gap[0].cpu().numpy()),
        "phase_gap": _normalize_array(phase_delta[0].cpu().numpy()),
        "amp_score": _score_to_grid(freq_aux["amp_score"], spectral_hw, patch_size),
        "phase_score": _score_to_grid(freq_aux["phase_score"], spectral_hw, patch_size),
        "amp_mask": _mask_to_grid(freq_aux["amp_mask"], spectral_hw, patch_size),
        "phase_mask": _mask_to_grid(freq_aux["phase_mask"], spectral_hw, patch_size),
        "ir_spatial": _tensor_map(ir_spa[0]),
        "vis_spatial": _tensor_map(vis_spa[0]),
        "spatial_weight": spa_aux["weight_l1"][0, 0].detach().float().cpu().numpy(),
        "deg_prompt_weight": intent_aux["deg_prompt_weight"][0].cpu().numpy(),
        "fus_prompt_weight": intent_aux["fus_prompt_weight"][0].cpu().numpy(),
    }


def build_demo_features(size: int = 160) -> Dict[str, np.ndarray]:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    target = np.exp(-((x - 0.25) ** 2 + (y + 0.05) ** 2) / 0.035)
    road = np.exp(-((y - 0.45 * x - 0.25) ** 2) / 0.025)
    texture = 0.5 + 0.25 * np.sin(18 * x) * np.cos(14 * y)

    ir = _normalize_array(0.35 * road + 1.4 * target + 0.08 * np.cos(5 * x))
    vis_gray = _normalize_array(0.65 * texture + 0.35 * road - 0.15 * target)
    vis_rgb = np.stack(
        [vis_gray, _normalize_array(vis_gray * 0.9 + 0.1 * road), _normalize_array(vis_gray * 0.8)],
        axis=-1,
    )

    fsize = 44
    fy, fx = np.mgrid[-1:1:complex(fsize), -1:1:complex(fsize)]
    amp_gap = _normalize_array(
        np.exp(-((fx + 0.35) ** 2 + (fy - 0.1) ** 2) / 0.12)
        + 0.7 * np.exp(-((fx - 0.4) ** 2 + (fy + 0.35) ** 2) / 0.06)
    )
    phase_gap = _normalize_array(
        np.abs(np.sin(5 * fx + 2 * fy))
        * np.exp(-(fx**2 + fy**2) / 1.3)
    )

    token_n = 11
    ty, tx = np.mgrid[-1:1:complex(token_n), -1:1:complex(token_n)]
    amp_score = _normalize_array(
        np.exp(-((tx + 0.35) ** 2 + (ty - 0.1) ** 2) / 0.18)
        + 0.7 * np.exp(-((tx - 0.35) ** 2 + (ty + 0.35) ** 2) / 0.12)
    )
    phase_score = _normalize_array(np.abs(np.sin(4 * tx + 1.5 * ty)))
    amp_mask = (amp_score >= np.quantile(amp_score, 0.70)).astype(np.float32)
    phase_mask = (phase_score >= np.quantile(phase_score, 0.65)).astype(np.float32)

    ir_spatial = _normalize_array(0.8 * target + 0.35 * road)
    vis_spatial = _normalize_array(0.85 * texture + 0.45 * road)
    spatial_weight = _normalize_array(0.75 * target + 0.25 * (1.0 - texture))

    return {
        "ir_image": np.round(ir * 255).astype(np.uint8),
        "vis_image": np.round(vis_rgb * 255).astype(np.uint8),
        "amp_gap": amp_gap,
        "phase_gap": phase_gap,
        "amp_score": amp_score,
        "phase_score": phase_score,
        "amp_mask": amp_mask,
        "phase_mask": phase_mask,
        "ir_spatial": ir_spatial,
        "vis_spatial": vis_spatial,
        "spatial_weight": spatial_weight,
        "deg_prompt_weight": np.array([0.55, 0.12, 0.22, 0.11]),
        "fus_prompt_weight": np.array([0.34, 0.28, 0.16, 0.22]),
    }


def _show(ax, image, title: str):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_figure(data: Dict[str, np.ndarray], output: str, title_suffix: str = ""):
    fig = plt.figure(figsize=(14.5, 6.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.05, 1.05, 1.0, 0.72, 1.0, 1.0],
        height_ratios=[1, 1],
    )

    ax_ir = fig.add_subplot(grid[0, 0])
    ax_vis = fig.add_subplot(grid[1, 0])
    _show(ax_ir, data["ir_image"], "Infrared image")
    _show(ax_vis, data["vis_image"], "Visible image")

    ax_amp_gap = fig.add_subplot(grid[0, 1])
    ax_phase_gap = fig.add_subplot(grid[0, 2])
    _show(ax_amp_gap, data["amp_gap"], "Amplitude discrepancy")
    _show(ax_phase_gap, data["phase_gap"], "Phase discrepancy")

    ax_ir_spa = fig.add_subplot(grid[1, 1])
    ax_vis_spa = fig.add_subplot(grid[1, 2])
    _show(ax_ir_spa, data["ir_spatial"], "Infrared spatial feature")
    _show(ax_vis_spa, data["vis_spatial"], "Visible spatial feature")

    ax_int_fre = fig.add_subplot(grid[0, 3])
    ax_int_spa = fig.add_subplot(grid[1, 3])
    for ax, label, subtitle in [
        (ax_int_fre, r"$\mathrm{Int}_{fre}$", "token-wise\nspectral reliability"),
        (ax_int_spa, r"$\mathrm{Int}_{spa}$", "pixel-wise\nmodality preference"),
    ]:
        ax.axis("off")
        ax.text(
            0.5,
            0.60,
            label,
            ha="center",
            va="center",
            fontsize=17,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.55", fill=False),
        )
        ax.text(0.5, 0.25, subtitle, ha="center", va="center", fontsize=9)

    ax_amp_score = fig.add_subplot(grid[0, 4])
    ax_phase_score = fig.add_subplot(grid[0, 5])
    _show(ax_amp_score, data["amp_score"], "Amplitude-token scores")
    _show(ax_phase_score, data["phase_score"], "Phase-token scores")

    ax_weight = fig.add_subplot(grid[1, 4])
    _show(ax_weight, data["spatial_weight"], "Infrared preference map")

    ax_summary = fig.add_subplot(grid[1, 5])
    ax_summary.axis("off")
    summary = (
        "Different feature forms\n"
        "require different intents\n\n"
        r"$N$ spectral tokens"
        "\n"
        r"$H\times W$ spatial locations"
    )
    ax_summary.text(
        0.5,
        0.52,
        summary,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.45,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.88, wspace=0.30, hspace=0.35)

    fig.text(
        0.5,
        0.975,
        "Dual intents for heterogeneous feature decisions" + title_suffix,
        ha="center",
        va="top",
        fontsize=15,
        weight="bold",
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Generate a layout preview without model files.")
    parser.add_argument("--ir", type=str, default=None)
    parser.add_argument("--vis", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default="intro_dual_intent_feature_level.png")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.demo:
        data = build_demo_features()
        plot_figure(data, args.output, title_suffix=" (layout preview)")
        print(f"Saved preview to: {args.output}")
        return

    missing = [name for name, value in (("--ir", args.ir), ("--vis", args.vis), ("--ckpt", args.ckpt)) if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    data = collect_real_features(args.ir, args.vis, args.ckpt, device)
    plot_figure(data, args.output)
    print(f"Saved real feature visualization to: {args.output}")


if __name__ == "__main__":
    main()
