# -*- coding: utf-8 -*-
"""
单阶段端到端训练脚本

当前版本不再采用 TSFI-Fusion 的“两阶段训练（先自编码重建，再做融合）”策略，
而是从第 1 个 epoch 开始，直接联合训练以下四个模块：
1. 编码器 Restormer_Encoder
2. 基础分支融合层 BaseFeatureExtraction
3. 高频/频率分支融合层 HighLevelGuidedFrequencyFusion
4. 解码器 Restormer_Decoder

训练主线：
输入红外/可见光 -> 编码器 -> 基础分支融合 + 高频分支融合 -> 解码器 -> 融合图像 -> 直接反向传播
"""

from net.Network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 训练配置
# -----------------------------------------------------------------------------
model_str = "HighLevelGuidedFreqFusion_SingleStage"
num_epochs = 120
lr = 1e-4
weight_decay = 0.0
batch_size = 8
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# 损失系数
coeff_decomp = 2.0  # 分解约束权重：基础特征相关、细节特征解相关

# 设备
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# -----------------------------------------------------------------------------
# 模型构建
# -----------------------------------------------------------------------------
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(
    HighLevelGuidedFrequencyFusion(
        in_channels=64,
        patch_size=4,
        amp_topk_ratio=0.25,
        phase_topk_ratio=0.25,
        token_embed_dim=128,
        num_heads=4,
    )
).to(device)

# -----------------------------------------------------------------------------
# 优化器与学习率调度器
# -----------------------------------------------------------------------------
optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

# 融合损失：由现有工程中的亮度损失 + 梯度损失组成
criteria_fusion = Fusionloss()

# -----------------------------------------------------------------------------
# 数据加载
# -----------------------------------------------------------------------------
trainloader = DataLoader(
    H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
loader = {"train": trainloader}

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# 开始训练
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader["train"]):
        # 将数据送入设备
        data_VIS = data_VIS.to(device, non_blocking=True)
        data_IR = data_IR.to(device, non_blocking=True)

        # 设置训练模式
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        # 梯度清零
        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # 1. 编码器提取特征
        feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
        feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

        # 2. 基础分支融合：处理低频/共享信息
        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)

        # 3. 高频/频率分支融合：使用我们的高层先验引导频率 token 融合模块
        feature_F_D = DetailFuseLayer(feature_V, feature_I)

        # 4. 解码器重建最终融合图像
        data_Fuse, _ = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

        # 5. 损失函数
        # 基础特征希望相关，细节特征希望解相关
        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        loss_decomp = (cc_loss_D ** 2) / (1.01 + cc_loss_B)

        # 亮度/显著目标 + 梯度纹理 约束
        fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
        loss = fusionloss + coeff_decomp * loss_decomp

        # 反向传播
        loss.backward()

        # 梯度裁剪，避免训练不稳定
        nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # 进度显示
        batches_done = epoch * len(loader["train"]) + i
        batches_left = num_epochs * len(loader["train"]) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f | fusion: %.6f | decomp: %.6f] ETA: %.10s"
            % (
                epoch + 1,
                num_epochs,
                i + 1,
                len(loader["train"]),
                loss.item(),
                fusionloss.item(),
                loss_decomp.item(),
                time_left,
            )
        )

    # 每个 epoch 后更新学习率
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

    # 设置学习率下限，避免衰减过小
    if optimizer1.param_groups[0]["lr"] <= 1e-6:
        optimizer1.param_groups[0]["lr"] = 1e-6
    if optimizer2.param_groups[0]["lr"] <= 1e-6:
        optimizer2.param_groups[0]["lr"] = 1e-6
    if optimizer3.param_groups[0]["lr"] <= 1e-6:
        optimizer3.param_groups[0]["lr"] = 1e-6
    if optimizer4.param_groups[0]["lr"] <= 1e-6:
        optimizer4.param_groups[0]["lr"] = 1e-6

# 保存权重
checkpoint = {
    "DIDF_Encoder": DIDF_Encoder.state_dict(),
    "DIDF_Decoder": DIDF_Decoder.state_dict(),
    "BaseFuseLayer": BaseFuseLayer.state_dict(),
    "DetailFuseLayer": DetailFuseLayer.state_dict(),
}

torch.save(checkpoint, os.path.join("models", model_str + "_" + timestamp + ".pth"))
torch.save(checkpoint, os.path.join("models", model_str + "_latest.pth"))
print("\n训练结束，权重已保存到 models/ 目录。")
