# 当前版本修改说明

本版本已经**彻底移除 TSFI-Fusion 的方法性模块**，只保留工程外壳：

- 删除 TSFI 的原始编码器/解码器实现
- 删除 CATM / BSC / DDE / CIIM / DDIM / INN / FDCM 等方法性模块
- 改为干净版本：
  - `net/encoder/simple_encoder.py`
  - `net/fusion/base_fusion.py`
  - `net/decoder/simple_decoder.py`
- 保留你自己的核心创新：
  - `net/frequency_fusion/` 中的高层先验引导频率 token 融合模块

当前主干：
红外/可见光 -> 共享编码器 -> 基础融合分支 + 高频频率融合分支 -> 解码器 -> 融合图像

当前训练方式：
- 单阶段端到端训练
- 不再使用 TSFI-Fusion 的两阶段训练策略
