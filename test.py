# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging

from net.Network import SharedEncoder, FusionDecoder, TextConditionedSpatialFusion
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.img_read_save import img_save, image_read_cv2

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ckpt_path = r'./models/TextIntentDualDomainFusion_latest.pth'

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


def _load_state(module, checkpoint, key, strict=True):
    """
    当前版本 checkpoint 应包含：
        shared_encoder
        frequency_fusion
        spatial_fusion
        fusion_decoder

    测试当前最新模型时建议 strict=True，
    避免部分参数未加载但程序仍继续运行。
    """
    if key not in checkpoint:
        raise KeyError(f'Checkpoint missing key: {key}')
    module.load_state_dict(checkpoint[key], strict=strict)


def build_model(device):
    encoder = nn.DataParallel(
        SharedEncoder(
            inp_channels=1,
            feature_dim=64,
            inner_dim=24,
            num_blocks=1,
            num_heads=1,
            ffn_expansion_factor=2.0
        )
    ).to(device)

    frequency_fusion = nn.DataParallel(
        HighLevelGuidedFrequencyFusion(
            in_channels=64,
            patch_size=4,
            amp_topk_ratio=0.30,
            phase_topk_ratio=0.35,
            token_embed_dim=128,
            num_heads=4,
            return_aux=True,
            routing_temperature=0.25,
            use_real_clip_prompt_bank=USE_REAL_CLIP_PROMPT_BANK,
            clip_model_name=CLIP_MODEL_NAME,
            prompt_texts=PROMPT_TEXTS,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
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

    decoder = nn.DataParallel(
        FusionDecoder(
            channels=64,
            out_channels=1,
            inner_dim=24,
            num_blocks=1,
            num_heads=1,
            ffn_expansion_factor=2.0
        )
    ).to(device)

    return encoder, frequency_fusion, spatial_fusion, decoder


def load_checkpoint(ckpt_path, encoder, frequency_fusion, spatial_fusion, decoder, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)

    _load_state(encoder, checkpoint, 'shared_encoder', strict=True)
    _load_state(frequency_fusion, checkpoint, 'frequency_fusion', strict=True)
    _load_state(spatial_fusion, checkpoint, 'spatial_fusion', strict=True)
    _load_state(decoder, checkpoint, 'fusion_decoder', strict=True)


def normalize_to_uint8(tensor):
    tensor = tensor.clamp(0.0, 1.0)
    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor) + 1e-8)
    image = np.squeeze((tensor * 255.0).cpu().numpy()).astype(np.uint8)
    return image


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder, frequency_fusion, spatial_fusion, decoder = build_model(device)
    load_checkpoint(ckpt_path, encoder, frequency_fusion, spatial_fusion, decoder, device)

    encoder.eval()
    frequency_fusion.eval()
    spatial_fusion.eval()
    decoder.eval()

    for dataset_name in ['MSRS']:
        print('\n' * 2 + '=' * 80)
        print('The test result of ' + dataset_name + ' :')

        path = './test_img/'
        test_folder = os.path.join(path, dataset_name)
        test_out_folder = os.path.join(r'./test_result', dataset_name)
        os.makedirs(test_out_folder, exist_ok=True)

        ir_folder = os.path.join(test_folder, 'ir')
        vi_folder = os.path.join(test_folder, 'vi')

        if not os.path.isdir(ir_folder):
            raise FileNotFoundError(f'IR folder not found: {ir_folder}')
        if not os.path.isdir(vi_folder):
            raise FileNotFoundError(f'VI folder not found: {vi_folder}')

        img_names = sorted(os.listdir(ir_folder))

        with torch.no_grad():
            for img_name in img_names:
                ir_path = os.path.join(ir_folder, img_name)
                vi_path = os.path.join(vi_folder, img_name)

                if not os.path.isfile(vi_path):
                    print(f'Skip {img_name}: visible image not found.')
                    continue

                data_ir = image_read_cv2(ir_path, mode='GRAY')
                data_vis_ycrcb = image_read_cv2(vi_path, mode='YCrCb')
                data_vis_bgr = cv2.imread(vi_path)

                if data_ir is None or data_vis_ycrcb is None or data_vis_bgr is None:
                    print(f'Skip {img_name}: image read failed.')
                    continue

                data_vis_y, data_vis_cr, data_vis_cb = cv2.split(data_vis_ycrcb)

                data_ir = data_ir[np.newaxis, np.newaxis, ...] / 255.0
                data_vis = data_vis_y[np.newaxis, np.newaxis, ...] / 255.0

                data_ir = torch.FloatTensor(data_ir).to(device)
                data_vis = torch.FloatTensor(data_vis).to(device)

                vis_spa, vis_freq, _ = encoder(data_vis)
                ir_spa, ir_freq, _ = encoder(data_ir)

                fused_freq, freq_aux = frequency_fusion(vis_freq, ir_freq)

                if 'spatial_intent' not in freq_aux:
                    raise KeyError('frequency_fusion aux must contain spatial_intent.')

                spatial_intent = freq_aux['spatial_intent']

                # 注意：这里不传入 fused_freq / freq_context，保持频率域与空间域并行。
                fused_spa = spatial_fusion(vis_spa, ir_spa, spatial_intent)

                decoder_skip = 0.5 * (data_vis + data_ir)

                data_fuse, _ = decoder(
                    decoder_skip,
                    fused_spa,
                    fused_freq,
                    text_intent=spatial_intent
                )

                fi = normalize_to_uint8(data_fuse)

                ycrcb_fi = np.dstack((fi, data_vis_cr, data_vis_cb))
                rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)

                img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)

        print(f'Finished testing {dataset_name}. Results saved to: {test_out_folder}')


if __name__ == '__main__':
    main()