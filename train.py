# -*- coding: utf-8 -*-
from net.Network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.dataset import H5Dataset
import os, sys, time, datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, SimpleSSIMLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
criteria_ssim = SimpleSSIMLoss(window_size=11)

num_epochs = 120
lr = 1e-4
weight_decay = 0
batch_size = 8
coeff_fusion = 1.0
coeff_ssim = 2.0
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder(inp_channels=1, feature_dim=64)).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder(channels=64, out_channels=1)).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(channels=64)).to(device)
DetailFuseLayer = nn.DataParallel(HighLevelGuidedFrequencyFusion(in_channels=64, patch_size=4, amp_topk_ratio=0.25, phase_topk_ratio=0.25, token_embed_dim=128, num_heads=4)).to(device)

optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"), batch_size=batch_size, shuffle=True, num_workers=0)
loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        DIDF_Encoder.train(); DIDF_Decoder.train(); BaseFuseLayer.train(); DetailFuseLayer.train()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        # 1) 编码
        feature_V_B, feature_V_F, _ = DIDF_Encoder(data_VIS)
        feature_I_B, feature_I_F, _ = DIDF_Encoder(data_IR)
        # 2) 基础融合分支
        feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
        # 3) 高频频率融合分支
        feature_F_D = DetailFuseLayer(feature_V_F, feature_I_F)
        # 4) 解码输出
        data_Fuse, _ = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

        fusion_loss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
        ssim_loss = criteria_ssim(data_Fuse, data_VIS) + criteria_ssim(data_Fuse, data_IR)
        loss = coeff_fusion * fusion_loss + coeff_ssim * ssim_loss
        loss.backward()

        nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
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

checkpoint = {'DIDF_Encoder': DIDF_Encoder.state_dict(), 'DIDF_Decoder': DIDF_Decoder.state_dict(), 'BaseFuseLayer': BaseFuseLayer.state_dict(), 'DetailFuseLayer': DetailFuseLayer.state_dict()}
os.makedirs('models', exist_ok=True)
torch.save(checkpoint, os.path.join('models', 'HighLevelGuidedFreqFusion_Clean_' + timestamp + '.pth'))
torch.save(checkpoint, os.path.join('models', 'HighLevelGuidedFreqFusion_Clean_latest.pth'))
