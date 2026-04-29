# 当前版本修改说明

- 新增 `net/fusion/text_conditioned_spatial_fusion.py`，实现空间域的 Text-conditioned Spatial Adaptive Fusion。
- `HighLevelGuidedFrequencyFusion` 新增两个投影头：
  - `frequency_intent_head`：用于 frequency token scoring / routing；
  - `spatial_intent_head`：用于空间域文本条件调制。
- `train.py` 的主链路已从“BaseFusion + SpatialResidualCompensation”调整为：
  - `F_freq = HighLevelGuidedFrequencyFusion(vis_freq, ir_freq)`
  - `F_spa = TextConditionedSpatialFusion(vis_spa, ir_spa, spatial_intent, F_freq)`
  - `fused image = FusionDecoder(skip, F_spa, F_freq)`
- `test.py` 已同步调整为新结构，并默认读取 `models/TextIntentDualDomainFusion_latest.pth`。
- 保留 `BaseFusion` 和 `SpatialResidualCompensation` 的导出仅用于历史兼容，当前训练主链路不再依赖它们。
- 未引入 CLIP image encoder、DINO、Text-IF 整网权重或对象级 mask。
