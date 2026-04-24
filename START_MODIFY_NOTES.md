# 当前版本修改说明

- 训练/测试脚本统一切换为新命名：`SharedEncoder / BaseFusion / FusionDecoder`
- `net/intent_frequency_fusion.py` 改为历史兼容包装文件
- 新增 `net/frequency_fusion/bypass.py`，显式处理 unselected token 轻量路径
- 重写 `utils/loss.py`，补上 `SimpleSSIMLoss` 并增加 `FrequencyConsistencyLoss`
- `dataprocessing.py` 删除硬编码路径，改为命令行参数输入


- 新增 `net/fusion/spatial_compensation.py`，实现频率参照引导的空间残差补偿模块。
- `train.py/test.py` 已接入 `SpatialResidualCompensation`：
  - `F_base_init = BaseFusion(vis_base, ir_base)`
  - `F_freq = HighLevelGuidedFrequencyFusion(vis_freq, ir_freq)`
  - `F_base = SpatialResidualCompensation(F_base_init, F_freq, vis_base, ir_base)`
- 新 checkpoint 会保存 `spatial_compensation` 权重。
- `test.py` 对旧 checkpoint 兼容：若 checkpoint 中没有 `spatial_compensation`，会将 `comp_scale` 置 0，避免未训练补偿模块影响旧模型测试结果。
