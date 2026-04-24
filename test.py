# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging

from net.Network import SharedEncoder, FusionDecoder, BaseFusion, SpatialResidualCompensation
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.img_read_save import img_save, image_read_cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r'/root/Image_Fusion_ours_frequency_v1/models/HighLevelGuidedFreqFusion_Clean_latest.pth'

# =========================
# CLIP 语义引导配置
# =========================
USE_REAL_CLIP_PROMPT_BANK = True
CLIP_MODEL_NAME = r'E:\yizuo_SCI\Code\Image_Fusion_ours_frequency\weight\clip\ViT-B-32.pt'
CLIP_DOWNLOAD_ROOT = r'E:\yizuo_SCI\weights\clip'
PROMPT_TEXTS = [
    'salient targets',
    'structural contours',
    'fine textures',
    'balanced fusion',
    'low light enhancement',
]


def _load_state(module, checkpoint, new_key, old_key, strict=True):
    if new_key in checkpoint:
        module.load_state_dict(checkpoint[new_key], strict=strict)
        return True
    if old_key in checkpoint:
        module.load_state_dict(checkpoint[old_key], strict=strict)
        return True
    if strict:
        raise KeyError(f'Checkpoint missing both {new_key} and {old_key}.')
    return False


for dataset_name in ["MSRS"]:
    print("\n" * 2 + "=" * 80)
    print("The test result of " + dataset_name + ' :')
    path = './test_img/'
    test_folder = os.path.join(path, dataset_name)
    test_out_folder = os.path.join(r'./test_result', dataset_name)
    os.makedirs(test_out_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = nn.DataParallel(SharedEncoder(inp_channels=1, feature_dim=64)).to(device)
    decoder = nn.DataParallel(FusionDecoder(channels=64, out_channels=1)).to(device)
    base_fusion = nn.DataParallel(BaseFusion(channels=64)).to(device)
    spatial_compensation = nn.DataParallel(SpatialResidualCompensation(channels=64, init_scale=0.1)).to(device)
    frequency_fusion = nn.DataParallel(
        HighLevelGuidedFrequencyFusion(
            in_channels=64,
            patch_size=4,
            amp_topk_ratio=0.25,
            phase_topk_ratio=0.25,
            token_embed_dim=128,
            num_heads=4,
            use_real_clip_prompt_bank=USE_REAL_CLIP_PROMPT_BANK,
            clip_model_name=CLIP_MODEL_NAME,
            prompt_texts=PROMPT_TEXTS,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
        )
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    _load_state(encoder, checkpoint, 'shared_encoder', 'DIDF_Encoder')
    _load_state(decoder, checkpoint, 'fusion_decoder', 'DIDF_Decoder')
    _load_state(base_fusion, checkpoint, 'base_fusion', 'BaseFuseLayer')
    _load_state(frequency_fusion, checkpoint, 'frequency_fusion', 'DetailFuseLayer')
    loaded_compensation = _load_state(spatial_compensation, checkpoint, 'spatial_compensation', 'SpatialCompensation', strict=False)
    if not loaded_compensation:
        print('[Warning] checkpoint does not contain spatial_compensation; compensation is disabled by setting comp_scale=0.0.')
        comp_module = spatial_compensation.module if hasattr(spatial_compensation, 'module') else spatial_compensation
        comp_module.comp_scale.data.zero_()

    encoder.eval(); decoder.eval(); base_fusion.eval(); frequency_fusion.eval(); spatial_compensation.eval()
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, 'ir')):
            data_ir = image_read_cv2(os.path.join(test_folder, 'ir', img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_vis = cv2.split(image_read_cv2(os.path.join(test_folder, 'vi', img_name), mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_vis_bgr = cv2.imread(os.path.join(test_folder, 'vi', img_name))
            _, data_vis_cr, data_vis_cb = cv2.split(cv2.cvtColor(data_vis_bgr, cv2.COLOR_BGR2YCrCb))
            data_ir = torch.FloatTensor(data_ir).to(device)
            data_vis = torch.FloatTensor(data_vis).to(device)

            vis_base, vis_freq, _ = encoder(data_vis)
            ir_base, ir_freq, _ = encoder(data_ir)
            fused_base_init = base_fusion(vis_base, ir_base)
            fused_freq = frequency_fusion(vis_freq, ir_freq)
            fused_base = spatial_compensation(fused_base_init, fused_freq, vis_base, ir_base)
            decoder_skip = 0.5 * (data_vis + data_ir)
            data_fuse, _ = decoder(decoder_skip, fused_base, fused_freq)
            data_fuse = (data_fuse - torch.min(data_fuse)) / (torch.max(data_fuse) - torch.min(data_fuse) + 1e-8)
            fi = np.squeeze((data_fuse * 255).cpu().numpy()).astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_vis_cr, data_vis_cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
