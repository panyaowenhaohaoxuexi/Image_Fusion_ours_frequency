# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging

from net.Network import SharedEncoder, FusionDecoder, TextConditionedSpatialFusion, DualStreamIntentMLP, BFSC
from net.frequency_fusion import TGSFF
from utils.img_read_save import img_save, image_read_cv2

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ckpt_path = r'./models/TextIntentDualDomainFusion_latest.pth'

CLIP_MODEL_NAME = r'E:\yizuo_SCI\1_Code\Image_Fusion_ours_frequency\weight\clip\ViT-B-32.pt'
CLIP_DOWNLOAD_ROOT = r'E:\yizuo_SCI\weights\clip'


def build_model(device):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    intent_generator = nn.DataParallel(
        DualStreamIntentMLP(
            channels=64,
            intent_dim=64,
            hidden_dim=256,
            clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
            use_clip_prompt_buffer=False,
        )
    ).to(device)
    frequency_fusion = nn.DataParallel(
        TGSFF(
            in_channels=64,
            patch_size=4,
            amp_topk_ratio=0.30,
            phase_topk_ratio=0.35,
            token_embed_dim=128,
            num_heads=4,
            return_aux=True,
            routing_temperature=0.25,
        )
    ).to(device)
    spatial_fusion = nn.DataParallel(
        TextConditionedSpatialFusion(
            channels=64,
            intent_dim=64,
            num_heads=4,
            ffn_expansion_factor=2.0,
            init_res_scale=0.05,
            use_freq_context=False,
        )
    ).to(device)
    bfsc = nn.DataParallel(BFSC(channels=64)).to(device)
    decoder = nn.DataParallel(
        FusionDecoder(channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    return encoder, intent_generator, frequency_fusion, spatial_fusion, bfsc, decoder


def _load_state(module, checkpoint, key, strict=True):
    if key not in checkpoint:
        raise KeyError(f'Checkpoint missing key: {key}')
    module.load_state_dict(checkpoint[key], strict=strict)


def normalize_to_uint8(tensor):
    tensor = tensor.clamp(0.0, 1.0)
    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor) + 1e-8)
    return np.squeeze((tensor * 255.0).cpu().numpy()).astype(np.uint8)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, intent_generator, frequency_fusion, spatial_fusion, bfsc, decoder = build_model(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    _load_state(encoder, checkpoint, 'shared_encoder', strict=True)
    _load_state(intent_generator, checkpoint, 'intent_generator', strict=True)
    _load_state(frequency_fusion, checkpoint, 'frequency_fusion', strict=True)
    _load_state(spatial_fusion, checkpoint, 'spatial_fusion', strict=True)
    _load_state(bfsc, checkpoint, 'bfsc', strict=True)
    _load_state(decoder, checkpoint, 'fusion_decoder', strict=True)

    for module in [encoder, intent_generator, frequency_fusion, spatial_fusion, bfsc, decoder]:
        module.eval()

    for dataset_name in ['MSRS']:
        print('\n' * 2 + '=' * 80)
        print('The test result of ' + dataset_name + ' :')
        test_folder = os.path.join('./test_img/', dataset_name)
        test_out_folder = os.path.join(r'./test_result', dataset_name)
        os.makedirs(test_out_folder, exist_ok=True)

        ir_folder = os.path.join(test_folder, 'ir')
        vi_folder = os.path.join(test_folder, 'vi')
        if not os.path.isdir(ir_folder):
            raise FileNotFoundError(f'IR folder not found: {ir_folder}')
        if not os.path.isdir(vi_folder):
            raise FileNotFoundError(f'VI folder not found: {vi_folder}')

        with torch.no_grad():
            for img_name in sorted(os.listdir(ir_folder)):
                ir_path = os.path.join(ir_folder, img_name)
                vi_path = os.path.join(vi_folder, img_name)
                if not os.path.isfile(vi_path):
                    print(f'Skip {img_name}: visible image not found.')
                    continue

                data_ir_np = image_read_cv2(ir_path, mode='GRAY')
                data_vis_ycrcb = image_read_cv2(vi_path, mode='YCrCb')
                data_vis_bgr = cv2.imread(vi_path)
                if data_ir_np is None or data_vis_ycrcb is None or data_vis_bgr is None:
                    print(f'Skip {img_name}: image read failed.')
                    continue

                data_vis_y, data_vis_cr, data_vis_cb = cv2.split(data_vis_ycrcb)
                data_ir = torch.FloatTensor(data_ir_np[np.newaxis, np.newaxis, ...] / 255.0).to(device)
                data_vis = torch.FloatTensor(data_vis_y[np.newaxis, np.newaxis, ...] / 255.0).to(device)

                vis_spa, vis_freq, _ = encoder(data_vis)
                ir_spa, ir_freq, _ = encoder(data_ir)
                z_deg, z_fus = intent_generator(vis_spa, ir_spa, vis_freq, ir_freq)
                fused_freq, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=z_deg)
                fused_spa_coarse, _ = spatial_fusion(vis_spa, ir_spa, z_fus, coarse_only=True, return_aux=True)
                _, g_coarse = bfsc(fused_freq, fused_spa_coarse)
                fused_spa, _ = spatial_fusion(vis_spa, ir_spa, z_fus, feedback_gate=g_coarse, return_aux=True)
                dual_feature, _ = bfsc(fused_freq, fused_spa)

                decoder_skip = 0.5 * (data_vis + data_ir)
                data_fuse, _ = decoder(decoder_skip, dual_feature, fused_freq, text_intent=z_fus)

                fi = normalize_to_uint8(data_fuse)
                ycrcb_fi = np.dstack((fi, data_vis_cr, data_vis_cb))
                rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
                img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)

        print(f'Finished testing {dataset_name}. Results saved to: {test_out_folder}')


if __name__ == '__main__':
    main()
