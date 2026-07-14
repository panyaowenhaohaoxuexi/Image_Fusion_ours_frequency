# -*- coding: utf-8 -*-
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_TAG,
    CLIP_DOWNLOAD_ROOT,
    CLIP_MODEL_NAME,
    MODEL_DIRECTORY,
    MODEL_VERSION,
    TRAIN_H5_PATH,
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
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, FrequencyConsistencyLoss, SimpleSSIMLoss


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def build_model(device: torch.device):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    if USE_CLIP_IMAGE_QUERY:
        intent_core = DualDomainTextIntentGenerator(
            intent_dim=64,
            clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
            clip_device=str(device),
        )
    else:
        intent_core = DualStreamIntentMLP(
            channels=64,
            intent_dim=64,
            hidden_dim=256,
            clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
            clip_device=str(device),
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
        channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    return encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def save_checkpoint(path, encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter,
                    spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder):
    checkpoint = {
        'model_version': MODEL_VERSION,
        'use_clip_image_query': USE_CLIP_IMAGE_QUERY,
        'intent_generator_type': type(unwrap(intent_generator)).__name__,
        'shared_encoder': encoder.state_dict(),
        'intent_generator': intent_generator.state_dict(),
        'frequency_fusion': frequency_fusion.state_dict(),
        'frequency_pyramid_adapter': frequency_pyramid_adapter.state_dict(),
        'spatial_fusion': spatial_fusion.state_dict(),
        'fsrc_l1': fsrc_l1.state_dict(),
        'fsrc_l2': fsrc_l2.state_dict(),
        'fsrc_l3': fsrc_l3.state_dict(),
        'fusion_decoder': fusion_decoder.state_dict(),
    }
    torch.save(checkpoint, path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criteria_fusion = Fusionloss().to(device)
    criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
    criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)
    num_epochs, lr, weight_decay, batch_size = 50, 1e-4, 0.0, 8
    coeff_fusion, coeff_ssim, coeff_freq = 1.0, 2.0, 0.5
    clip_grad_norm_value, optim_step, optim_gamma = 0.01, 20, 0.5

    modules = build_model(device)
    (shared_encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion,
     fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder) = modules
    optimizers = [torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, module.parameters()),
                                      lr=lr, weight_decay=weight_decay) for module in modules]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
                  for optimizer in optimizers]
    trainloader = DataLoader(H5Dataset(TRAIN_H5_PATH), batch_size=batch_size, shuffle=True, num_workers=0)
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    torch.backends.cudnn.benchmark = True
    previous_time = time.time()

    for epoch in range(num_epochs):
        for index, (data_vis, data_ir, data_vis_rgb_raw) in enumerate(trainloader):
            data_vis, data_ir = data_vis.to(device), data_ir.to(device)
            data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))
            for module in modules:
                module.train()
            for optimizer in optimizers:
                optimizer.zero_grad()

            vis_spa, vis_freq, _ = shared_encoder(data_vis)
            ir_spa, ir_freq, _ = shared_encoder(data_ir)
            i_deg, i_fus, _ = intent_generator(data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
            fused_freq, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=i_deg)
            _, spatial_pyramid, _ = spatial_fusion(vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
            freq_pyramid = frequency_pyramid_adapter(fused_freq, target_pyramid=spatial_pyramid)
            d_l1, _ = fsrc_l1(freq_pyramid['l1'], spatial_pyramid['l1'])
            d_l2, _ = fsrc_l2(freq_pyramid['l2'], spatial_pyramid['l2'])
            d_l3, _ = fsrc_l3(freq_pyramid['l3'], spatial_pyramid['l3'])
            fused_image, _ = fusion_decoder(d_l1, d_l2, d_l3)

            fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
            ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
            freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
            loss = coeff_fusion * fusion_loss + coeff_ssim * ssim_loss + coeff_freq * freq_loss
            loss.backward()
            for module in modules:
                nn.utils.clip_grad_norm_(module.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            for optimizer in optimizers:
                optimizer.step()

            batches_done = epoch * len(trainloader) + index
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - previous_time))
            previous_time = time.time()
            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] [fusion: %.6f] [ssim: %.6f] [freq: %.6f] ETA: %.10s' % (
                    epoch, num_epochs, index, len(trainloader), loss.item(), fusion_loss.item(),
                    ssim_loss.item(), freq_loss.item(), time_left,
                )
            )
        for scheduler in schedulers:
            scheduler.step()
        for optimizer in optimizers:
            optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'], 1e-6)

    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    save_checkpoint(os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_{timestamp}.pth'), *modules)
    save_checkpoint(os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_latest.pth'), *modules)


if __name__ == '__main__':
    main()
