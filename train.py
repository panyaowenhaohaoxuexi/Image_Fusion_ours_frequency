# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net.Network import SharedEncoder, FusionDecoder, TextConditionedSpatialFusion, DualDomainTextIntentGenerator, DDA
from net.frequency_fusion import TGSFF
from utils.dataset import H5Dataset
from utils.loss import (
    Fusionloss,
    SimpleSSIMLoss,
    FrequencyConsistencyLoss,
    IntentAlignmentLoss,
    CLIPSemanticConsistencyLoss,
    TokenRoutingRankingLoss,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CLIP_MODEL_NAME = r'E:\yizuo_SCI\1_Code\Image_Fusion_ours_frequency\weight\clip\ViT-B-32.pt'
CLIP_DOWNLOAD_ROOT = r'E:\yizuo_SCI\1_Code\Image_Fusion_ours_frequency\weight\clip'
USE_LEARNABLE_PROMPT_EMBEDDING = False


def build_model(device: str, use_learnable_prompt_embedding: bool = USE_LEARNABLE_PROMPT_EMBEDDING):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    intent_generator = nn.DataParallel(
        DualDomainTextIntentGenerator(
            channels=64,
            intent_dim=64,
            hidden_dim=256,
            clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT,
            use_clip_prompt_bank=True,
            use_learnable_prompt_embedding=use_learnable_prompt_embedding,
        )
    ).to(device)
    freq_fusion = nn.DataParallel(
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
    dda = nn.DataParallel(DDA(channels=64)).to(device)
    decoder = nn.DataParallel(
        FusionDecoder(channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    return encoder, intent_generator, freq_fusion, spatial_fusion, dda, decoder


def save_checkpoint(path: str, encoder, intent_generator, freq_fusion, spatial_fusion, dda, decoder):
    checkpoint = {
        'shared_encoder': encoder.state_dict(),
        'intent_generator': intent_generator.state_dict(),
        'frequency_fusion': freq_fusion.state_dict(),
        'spatial_fusion': spatial_fusion.state_dict(),
        'dda': dda.state_dict(),
        'fusion_decoder': decoder.state_dict(),
    }
    torch.save(checkpoint, path)


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criteria_fusion = Fusionloss().to(device)
    criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
    criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)
    criteria_align = IntentAlignmentLoss(temperature=0.07).to(device)
    criteria_sem = CLIPSemanticConsistencyLoss(clip_model_name=CLIP_MODEL_NAME, download_root=CLIP_DOWNLOAD_ROOT).to(device)
    criteria_route = TokenRoutingRankingLoss().to(device)

    num_epochs = 70
    lr = 1e-4
    weight_decay = 0.0
    batch_size = 8
    coeff_fusion = 1.0
    coeff_ssim = 2.0
    coeff_freq = 0.5
    coeff_align = 0.05
    coeff_sem = 0.1
    coeff_route = 0.05
    clip_grad_norm_value = 0.01
    optim_step = 20
    optim_gamma = 0.5

    shared_encoder, intent_generator, frequency_fusion, spatial_fusion, dda, fusion_decoder = build_model(device)

    optimizers = [
        torch.optim.Adam(filter(lambda p: p.requires_grad, shared_encoder.parameters()), lr=lr, weight_decay=weight_decay),
        torch.optim.Adam(filter(lambda p: p.requires_grad, intent_generator.parameters()), lr=lr, weight_decay=weight_decay),
        torch.optim.Adam(filter(lambda p: p.requires_grad, frequency_fusion.parameters()), lr=lr, weight_decay=weight_decay),
        torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_fusion.parameters()), lr=lr, weight_decay=weight_decay),
        torch.optim.Adam(filter(lambda p: p.requires_grad, dda.parameters()), lr=lr, weight_decay=weight_decay),
        torch.optim.Adam(filter(lambda p: p.requires_grad, fusion_decoder.parameters()), lr=lr, weight_decay=weight_decay),
    ]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=optim_step, gamma=optim_gamma) for opt in optimizers]

    trainloader = DataLoader(
        H5Dataset(r'E:\yizuo_SCI\2_Datasets\MSRS\MSRS_train_imgsize_128_stride_200.h5'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    loader = {'train': trainloader}
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')

    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    for epoch in range(num_epochs):
        for i, (data_vis, data_ir) in enumerate(loader['train']):
            data_vis, data_ir = data_vis.to(device), data_ir.to(device)
            for module in [shared_encoder, intent_generator, frequency_fusion, spatial_fusion, dda, fusion_decoder]:
                module.train()
            for optimizer in optimizers:
                optimizer.zero_grad()

            vis_spa, vis_freq, _ = shared_encoder(data_vis)
            ir_spa, ir_freq, _ = shared_encoder(data_ir)
            I_deg, I_fus, intent_aux = intent_generator(vis_spa, ir_spa, vis_freq, ir_freq)

            fused_freq, freq_aux = frequency_fusion(vis_freq, ir_freq, frequency_intent=I_deg)
            fused_spa, spa_aux = spatial_fusion(vis_spa, ir_spa, I_fus, return_aux=True)
            dual_feature, dda_gate = dda(fused_freq, fused_spa)

            decoder_skip = 0.5 * (data_vis + data_ir)
            fused_image, _ = fusion_decoder(decoder_skip, dual_feature, fused_freq)

            fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
            ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
            freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
            align_loss, _ = criteria_align(
                I_deg, I_fus, intent_aux['deg_prompt_bank'], intent_aux['fus_prompt_bank']
            )
            sem_loss, _ = criteria_sem(data_vis, data_ir, fused_image)
            route_loss, _ = criteria_route(freq_aux)

            loss = (
                coeff_fusion * fusion_loss
                + coeff_ssim * ssim_loss
                + coeff_freq * freq_loss
                + coeff_align * align_loss
                + coeff_sem * sem_loss
                + coeff_route * route_loss
            )
            loss.backward()

            for module in [shared_encoder, intent_generator, frequency_fusion, spatial_fusion, dda, fusion_decoder]:
                nn.utils.clip_grad_norm_(module.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            for optimizer in optimizers:
                optimizer.step()

            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] [align: %.6f] [sem: %.6f] [route: %.6f] ETA: %.10s' % (
                    epoch, num_epochs, i, len(loader['train']), loss.item(),
                    align_loss.item(), sem_loss.item(), route_loss.item(), time_left
                )
            )

        for scheduler in schedulers:
            scheduler.step()
        for optimizer in optimizers:
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                optimizer.param_groups[0]['lr'] = 1e-6

    os.makedirs('models', exist_ok=True)
    save_checkpoint(os.path.join('models', 'TextIntentDualDomainFusion_' + timestamp + '.pth'),
                    shared_encoder, intent_generator, frequency_fusion, spatial_fusion, dda, fusion_decoder)
    save_checkpoint(os.path.join('models', 'TextIntentDualDomainFusion_latest.pth'),
                    shared_encoder, intent_generator, frequency_fusion, spatial_fusion, dda, fusion_decoder)


if __name__ == '__main__':
    main()
