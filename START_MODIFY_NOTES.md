# 当前版本修改说明

- 训练/测试脚本统一切换为新命名：`SharedEncoder / BaseFusion / FusionDecoder`
- `net/intent_frequency_fusion.py` 改为历史兼容包装文件
- 新增 `net/frequency_fusion/bypass.py`，显式处理 unselected token 轻量路径
- 重写 `utils/loss.py`，补上 `SimpleSSIMLoss` 并增加 `FrequencyConsistencyLoss`
- `dataprocessing.py` 删除硬编码路径，改为命令行参数输入
