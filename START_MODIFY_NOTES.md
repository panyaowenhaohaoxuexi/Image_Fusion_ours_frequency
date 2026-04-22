# 本次代码修改说明

本次改动的核心目标是：

1. 删除 TSFI-Fusion 的两阶段训练策略；
2. 改为单阶段端到端训练；
3. 保留当前工程中可复用的编码器 / 基础分支 / 解码器外壳；
4. 将真正的细节融合主路径替换为我们的高层先验引导频率 token 融合模块。

## 当前训练策略

当前 `train.py` 已经改为 **单阶段训练**：

- 第 1 个 epoch 起就联合训练：
  - `Restormer_Encoder`
  - `BaseFeatureExtraction`
  - `HighLevelGuidedFrequencyFusion`
  - `Restormer_Decoder`
- 不再存在 Phase I / Phase II 的拆分
- 不再进行“先自编码重建，再做融合”的训练流程

## 当前损失

当前版本采用：

- `Fusionloss`：亮度 / 显著性 + 梯度约束
- `loss_decomp`：基础特征相关、细节特征解相关

这是一个先求稳、先跑通结构的版本。

## 当前推理

`test.py` 默认加载的新权重文件为：

- `models/HighLevelGuidedFreqFusion_SingleStage_latest.pth`
