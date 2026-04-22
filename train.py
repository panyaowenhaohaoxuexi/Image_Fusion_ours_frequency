# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net.Network import SharedEncoder, FusionDecoder, BaseFusion
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, SimpleSSIMLoss, FrequencyConsistencyLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def build_model(device: str):
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

    base_fusion = nn.DataParallel(BaseFusion(channels=64)).to(device)

    freq_fusion = nn.DataParallel(
        HighLevelGuidedFrequencyFusion(
            in_channels=64,
            patch_size=4,
            amp_topk_ratio=0.25,
            phase_topk_ratio=0.25,
            token_embed_dim=128,
            num_heads=4
        )
    ).to(device)

    return encoder, decoder, base_fusion, freq_fusion


def save_checkpoint(path: str, encoder, decoder, base_fusion, freq_fusion):
    checkpoint = {
        'shared_encoder': encoder.state_dict(),
        'fusion_decoder': decoder.state_dict(),
        'base_fusion': base_fusion.state_dict(),
        'frequency_fusion': freq_fusion.state_dict(),
    }
    torch.save(checkpoint, path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criteria_fusion = Fusionloss().to(device)
criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)

num_epochs = 5
lr = 1e-4
weight_decay = 0.0
batch_size = 8
coeff_fusion = 1.0
coeff_ssim = 2.0
coeff_freq = 0.5
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
shared_encoder, fusion_decoder, base_fusion, frequency_fusion = build_model(device)

optimizer1 = torch.optim.Adam(shared_encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(fusion_decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(base_fusion.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(frequency_fusion.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

trainloader = DataLoader(H5Dataset(r"E:\yizuo_SCI\2_Datasets\MSRS_train_imgsize_128_stride_200.h5"), batch_size=batch_size, shuffle=True, num_workers=0)
loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_vis, data_ir) in enumerate(loader['train']):
        data_vis, data_ir = data_vis.to(device), data_ir.to(device)
        shared_encoder.train(); fusion_decoder.train(); base_fusion.train(); frequency_fusion.train()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        vis_base, vis_freq, _ = shared_encoder(data_vis)
        ir_base, ir_freq, _ = shared_encoder(data_ir)
        fused_base = base_fusion(vis_base, ir_base)
        fused_freq = frequency_fusion(vis_freq, ir_freq)
        fused_image, _ = fusion_decoder(data_vis, fused_base, fused_freq)

        fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
        ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
        freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
        loss = coeff_fusion * fusion_loss + coeff_ssim * ssim_loss + coeff_freq * freq_loss
        loss.backward()

        nn.utils.clip_grad_norm_(shared_encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(fusion_decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(base_fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(frequency_fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s" % (epoch, num_epochs, i, len(loader['train']), loss.item(), time_left))

    scheduler1.step(); scheduler2.step(); scheduler3.step(); scheduler4.step()
    for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

os.makedirs('models', exist_ok=True)
save_checkpoint(os.path.join('models', 'HighLevelGuidedFreqFusion_Clean_' + timestamp + '.pth'), shared_encoder, fusion_decoder, base_fusion, frequency_fusion)
save_checkpoint(os.path.join('models', 'HighLevelGuidedFreqFusion_Clean_latest.pth'), shared_encoder, fusion_decoder, base_fusion, frequency_fusion)
