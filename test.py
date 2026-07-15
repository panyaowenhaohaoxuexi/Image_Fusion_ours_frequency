# -*- coding: utf-8 -*-
import logging
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import (
    CHECKPOINT_TAG,
    CLIP_DOWNLOAD_ROOT,
    CLIP_MODEL_NAME,
    DECODER_INNER_DIM,
    DECODER_MAX_RESIDUAL_SCALE,
    DECODER_NUM_BLOCKS,
    MODEL_DIRECTORY,
    MODEL_VERSION,
    USE_CLIP_IMAGE_QUERY,
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
from utils.img_read_save import img_save


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ckpt_path = os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_latest.pth')


def build_model(device):
    encoder = nn.DataParallel(SharedEncoder(
        inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    if USE_CLIP_IMAGE_QUERY:
        intent_core = DualDomainTextIntentGenerator(
            intent_dim=64, clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
        )
    else:
        intent_core = DualStreamIntentMLP(
            channels=64, intent_dim=64, hidden_dim=256, clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
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
        channels=64, out_channels=1, inner_dim=DECODER_INNER_DIM, num_blocks=DECODER_NUM_BLOCKS,
        max_residual_scale=DECODER_MAX_RESIDUAL_SCALE, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    return encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def validate_checkpoint_metadata(checkpoint, intent_generator):
    expected_type = type(unwrap(intent_generator)).__name__
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise RuntimeError('Checkpoint model version mismatch.')
    if checkpoint.get('use_clip_image_query') != USE_CLIP_IMAGE_QUERY:
        raise RuntimeError('Checkpoint query variant mismatch.')
    if checkpoint.get('intent_generator_type') != expected_type:
        raise RuntimeError('Checkpoint intent generator type mismatch.')


def _load_state(module, checkpoint, key, strict=True):
    if key not in checkpoint:
        raise KeyError(f'Checkpoint missing key: {key}')
    module.load_state_dict(checkpoint[key], strict=strict)


def normalize_to_uint8(tensor):
    return np.round(np.squeeze((tensor.detach().clamp(0.0, 1.0) * 255.0).cpu().numpy())).astype(np.uint8)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(
        f'[Model] version={MODEL_VERSION} decoder_inner_dim={DECODER_INNER_DIM} '
        f'decoder_num_blocks={DECODER_NUM_BLOCKS} decoder_max_residual_scale={DECODER_MAX_RESIDUAL_SCALE}'
    )
    if not os.path.isfile(CLIP_MODEL_NAME):
        raise FileNotFoundError(f'CLIP checkpoint not found: {CLIP_MODEL_NAME}')
    modules = build_model(device)
    (encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion,
     fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder) = modules
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    validate_checkpoint_metadata(checkpoint, intent_generator)
    keys = ('shared_encoder', 'intent_generator', 'frequency_fusion', 'frequency_pyramid_adapter',
            'spatial_fusion', 'fsrc_l1', 'fsrc_l2', 'fsrc_l3', 'fusion_decoder')
    for module, key in zip(modules, keys):
        _load_state(module, checkpoint, key, strict=True)
    for module in modules:
        module.eval()

    for dataset_name in ['MSRS_v10']:
        test_folder = os.path.join('./test_img/', dataset_name)
        test_out_folder = os.path.join('./test_result', dataset_name)
        gray_out_folder, color_out_folder = os.path.join(test_out_folder, 'gray'), os.path.join(test_out_folder, 'color')
        os.makedirs(gray_out_folder, exist_ok=True)
        os.makedirs(color_out_folder, exist_ok=True)
        ir_folder, vi_folder = os.path.join(test_folder, 'ir'), os.path.join(test_folder, 'vi')
        if not os.path.isdir(ir_folder) or not os.path.isdir(vi_folder):
            raise FileNotFoundError(f'Test folders not found: {ir_folder}, {vi_folder}')

        with torch.no_grad():
            for img_name in sorted(os.listdir(ir_folder)):
                ir_path, vi_path = os.path.join(ir_folder, img_name), os.path.join(vi_folder, img_name)
                data_ir_np = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
                data_vis_bgr = cv2.imread(vi_path, cv2.IMREAD_COLOR)
                if data_ir_np is None or data_vis_bgr is None:
                    print(f'Skip {img_name}: image read failed.')
                    continue
                height, width = data_vis_bgr.shape[:2]
                if data_ir_np.shape[:2] != (height, width):
                    data_ir_np = cv2.resize(data_ir_np, (width, height), interpolation=cv2.INTER_LINEAR)
                data_vis_rgb_raw = torch.from_numpy(
                    cv2.cvtColor(data_vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw)
                data_vis_y, data_vis_cr, data_vis_cb = cv2.split(cv2.cvtColor(data_vis_bgr, cv2.COLOR_BGR2YCrCb))
                data_ir = torch.from_numpy(data_ir_np[None, None].astype(np.float32) / 255.0).to(device)
                data_vis = torch.from_numpy(data_vis_y[None, None].astype(np.float32) / 255.0).to(device)

                vis_spa, vis_freq, _ = encoder(data_vis)
                ir_spa, ir_freq, _ = encoder(data_ir)
                i_deg, i_fus, _ = intent_generator(data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
                fused_freq, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=i_deg)
                _, spatial_pyramid, spatial_aux = spatial_fusion(vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
                freq_pyramid = frequency_pyramid_adapter(fused_freq, target_pyramid=spatial_pyramid)
                d_l1, _ = fsrc_l1(freq_pyramid['l1'], spatial_pyramid['l1'])
                d_l2, _ = fsrc_l2(freq_pyramid['l2'], spatial_pyramid['l2'])
                d_l3, _ = fsrc_l3(freq_pyramid['l3'], spatial_pyramid['l3'])
                data_fuse, _ = fusion_decoder(
                    d_l1, d_l2, d_l3, image_vis=data_vis, image_ir=data_ir,
                    weight_ir=spatial_aux['weight_multiscale'],
                )
                fused_y = normalize_to_uint8(data_fuse)
                save_name = os.path.splitext(img_name)[0]
                img_save(fused_y, save_name, gray_out_folder)
                rgb_fusion = cv2.cvtColor(cv2.merge((fused_y, data_vis_cr, data_vis_cb)), cv2.COLOR_YCrCb2RGB)
                img_save(rgb_fusion, save_name, color_out_folder)
                print(f'Saved: {img_name}')


if __name__ == '__main__':
    main()
