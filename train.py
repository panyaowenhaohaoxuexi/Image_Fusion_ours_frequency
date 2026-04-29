# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net.Network import SharedEncoder, FusionDecoder, TextConditionedSpatialFusion
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, SimpleSSIMLoss, FrequencyConsistencyLoss, TokenRoutingRankingLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


def build_model(device: str):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)

    freq_fusion = nn.DataParallel(
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
            use_freq_context=True,
        )
    ).to(device)

    decoder = nn.DataParallel(
        FusionDecoder(channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    return encoder, freq_fusion, spatial_fusion, decoder


def save_checkpoint(path: str, encoder, freq_fusion, spatial_fusion, decoder):
    checkpoint = {
        'shared_encoder': encoder.state_dict(),
        'frequency_fusion': freq_fusion.state_dict(),
        'spatial_fusion': spatial_fusion.state_dict(),
        'fusion_decoder': decoder.state_dict(),
    }
    torch.save(checkpoint, path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criteria_fusion = Fusionloss().to(device)
criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)
criteria_score = TokenRoutingRankingLoss(bce_weight=1.0, rank_weight=0.5, separation_margin=0.35).to(device)

num_epochs = 70
lr = 1e-4
weight_decay = 0.0
batch_size = 8
coeff_fusion = 1.0
coeff_ssim = 2.0
coeff_freq = 0.5
coeff_score = 0.03
score_warmup_epochs = 10
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

shared_encoder, frequency_fusion, spatial_fusion, fusion_decoder = build_model(device)

optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, shared_encoder.parameters()), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, frequency_fusion.parameters()), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_fusion.parameters()), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(filter(lambda p: p.requires_grad, fusion_decoder.parameters()), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

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
        shared_encoder.train(); frequency_fusion.train(); spatial_fusion.train(); fusion_decoder.train()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        vis_spa, vis_freq, _ = shared_encoder(data_vis)
        ir_spa, ir_freq, _ = shared_encoder(data_ir)

        freq_out = frequency_fusion(vis_freq, ir_freq)
        if not isinstance(freq_out, tuple):
            raise RuntimeError('frequency_fusion must return aux because spatial_fusion needs spatial_intent.')
        fused_freq, freq_aux = freq_out

        spatial_intent = freq_aux['spatial_intent']
        fused_spa = spatial_fusion(vis_spa, ir_spa, spatial_intent, fused_freq)

        decoder_skip = 0.5 * (data_vis + data_ir)
        fused_image, _ = fusion_decoder(decoder_skip, fused_spa, fused_freq)

        fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
        ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
        freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
        score_loss, _ = criteria_score(freq_aux)

        score_weight = coeff_score * min(1.0, float(epoch + 1) / float(score_warmup_epochs))
        loss = coeff_fusion * fusion_loss + coeff_ssim * ssim_loss + coeff_freq * freq_loss + score_weight * score_loss
        loss.backward()

        nn.utils.clip_grad_norm_(shared_encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(frequency_fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(spatial_fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(fusion_decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            '\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] [score: %.6f] ETA: %.10s' % (
                epoch, num_epochs, i, len(loader['train']), loss.item(), score_loss.item(), time_left
            )
        )

    scheduler1.step(); scheduler2.step(); scheduler3.step(); scheduler4.step()
    for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

os.makedirs('models', exist_ok=True)
save_checkpoint(os.path.join('models', 'TextIntentDualDomainFusion_' + timestamp + '.pth'), shared_encoder, frequency_fusion, spatial_fusion, fusion_decoder)
save_checkpoint(os.path.join('models', 'TextIntentDualDomainFusion_latest.pth'), shared_encoder, frequency_fusion, spatial_fusion, fusion_decoder)
